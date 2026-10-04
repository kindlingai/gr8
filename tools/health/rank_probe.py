#!/usr/bin/env python3
"""Rank health probe and boot admission for a gr8 cluster. Python 3 standard library only.

  rank_probe.py cluster --hosts h0,h1,.. --ips a,b,.. [--ips2 a,b,..] --home DIR --proxy-cpu N [--out FILE]
      Run from the head BEFORE a boot. Probes every rank (over ssh), measures the ring of RDMA write rates, compares
      the ranks with each other and prints one line per refused rank with a named reason. Exit 0 = admitted,
      3 = refused. --out writes the full result (the record `gate.py run --health` attaches to a run record).
  rank_probe.py local --proxy-cpu N [--gpu-tool PATH] [--ips a,b,..] [--ips2 ..]
      One node's measurements as JSON (what `cluster` runs on each rank; it ships this file over ssh).

What a rank must pass (reason codes; thresholds are `Thresholds` below and flags of `cluster`):
  NODE_BUSY           a g6m4 process, a busy GPU or too little free memory: a model is booted here. Nothing is measured.
  PROBE_FAILED        the node did not answer, or a tool the probe needs is missing (gpuprobe, ib_write_bw).
  GPU_WEDGED          `nvidia-smi -L` hangs or fails (an NVIDIA RM lock held, e.g. by a huge coredump: power cycle only).
  GPU_CLOCK_LOW       SM clock under load below the minimum (a USB-PD / power throttle sits at 500-850 MHz).
  GPU_HW_THROTTLE     a hardware slowdown / thermal / power-brake throttle reason is active under load.
  LINK_DOWN           an RDMA port that carries a rail is not ACTIVE.
  LINK_PCIE_DEGRADED  the ConnectX PCIe link runs below its maximum speed or width.
  LINK_SLOW           RDMA write rate out of this rank on a rail is below the floor or well below its peers (the latched
                      ~12-14 Gb/s fault: the link reports 200G Gen5 x4 and only a transfer shows it).
  HBM_SLOW / H2D_SLOW device-memory read rate / host-to-device copy rate well below the other ranks.
  PROXY_CPU_OFFLINE   a CPU the proxy threads are pinned to is offline or not in this process's CPU set.
  PROXY_CPU_BUSY      another thread keeps a proxy CPU busy while idle (the proxy would be starved).
The probe only reads and measures: it changes no clock, link, driver or power setting and refuses to run on a node
where a model is booted."""
import argparse
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

THROTTLE_BITS = {0x1: 'idle', 0x2: 'app_clocks', 0x4: 'sw_power_cap', 0x8: 'hw_slowdown', 0x10: 'sync_boost',
                 0x20: 'sw_thermal', 0x40: 'hw_thermal', 0x80: 'hw_power_brake', 0x100: 'display_clocks'}
HW_THROTTLE = 0x8 | 0x40 | 0x80          # reasons that are faults whenever they are active under load
PCIE_GBPS_PER_LANE = {'2.5': 2.0, '5.0': 4.0, '8.0': 7.88, '16.0': 15.75, '32.0': 31.5, '64.0': 63.0}  # after encoding


@dataclass
class Thresholds:
    min_clock_mhz: float = 1500.0      # SM clock under load; stock ~2400, capped runs ~2200, a power throttle 500-850
    min_link_gbps: float = 80.0        # RDMA write rate per rail; healthy ~110, the latched fault 12-14
    link_peer_tol: float = 0.15        # a rail may be this far below the median of the same rail on all ranks
    hbm_tol: float = 0.08              # read rate may be this far below the median of the ranks
    h2d_tol: float = 0.08
    max_proxy_cpu_busy_pct: float = 25.0
    min_free_gib: float = 16.0         # below this a model is taken to be loaded
    busy_gpu_util_pct: float = 20.0
    wedge_timeout_s: float = 15.0
    load_seconds: float = 4.0


# ---------------------------------------------------------------------------------------------- one node

class Host:
    """Everything the local probe reads from the machine; tests replace it."""
    def __init__(self, root='/'):
        self.root = root

    def path(self, p):
        return os.path.join(self.root, p.lstrip('/'))

    def read(self, p):
        try:
            with open(self.path(p)) as f:
                return f.read().strip()
        except OSError:
            return None

    def listdir(self, p):
        try:
            return sorted(os.listdir(self.path(p)))
        except OSError:
            return []

    def run(self, argv, timeout=10.0):
        """-> (returncode, stdout, stderr); returncode None = timed out (the process is killed)."""
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            return None, (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), ''
        except OSError as e:
            return 127, '', str(e)

    def popen(self, argv):
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def sleep(self, s):
        time.sleep(s)


def parse_cpu_list(s):
    """'0-3,8,10-11' -> [0,1,2,3,8,10,11]"""
    out = []
    for part in (s or '').split(','):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    return out


def parse_throttle(v):
    """clocks_throttle_reasons.active ('0x0000000000000008' or a decimal) -> (mask, [names])"""
    try:
        mask = int(v, 16) if str(v).lower().startswith('0x') else int(v)
    except (TypeError, ValueError):
        return None, []
    return mask, [n for b, n in THROTTLE_BITS.items() if mask & b]


def smi_query(h, fields, timeout=10.0):
    """One nvidia-smi query. -> (dict field -> str or None for [N/A], error text or None)."""
    rc, out, err = h.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'], timeout)
    if rc is None:
        return {}, 'nvidia-smi timed out'
    if rc != 0:
        return {}, (err or out).strip()[:200] or f'nvidia-smi rc {rc}'
    cells = [c.strip() for c in out.strip().splitlines()[0].split(',')] if out.strip() else []
    if len(cells) != len(fields):
        return {}, f'unexpected nvidia-smi output {out.strip()[:100]!r}'
    return {f: (None if c.upper().startswith('[N/A]') or c.upper() == 'N/A' else c) for f, c in zip(fields, cells)}, None


def smi_state(h):
    """clocks, utilization, power, temperature and throttle reasons, trying both spellings of the reasons field."""
    base = ['clocks.sm', 'clocks.max.sm', 'utilization.gpu', 'power.draw', 'temperature.gpu']
    for reasons in ('clocks_throttle_reasons.active', 'clocks_event_reasons.active', None):
        d, err = smi_query(h, base + ([reasons] if reasons else []))
        if err is None:
            def num(k):
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    return None
            mask, names = parse_throttle(d.get(reasons)) if reasons else (None, [])
            return {'sm_mhz': num('clocks.sm'), 'max_sm_mhz': num('clocks.max.sm'), 'util_pct': num('utilization.gpu'),
                    'power_w': num('power.draw'), 'temp_c': num('temperature.gpu'), 'throttle_mask': mask,
                    'throttle': names}
        last = err
        if 'timed out' in err:           # a wedged driver: do not wait three times
            break
    return {'error': last}


def busy_reasons(h, th):
    """Why this node must not be probed: a model is booted on it."""
    why = []
    for pid in h.listdir('/proc'):
        if pid.isdigit() and (h.read(f'/proc/{pid}/comm') or '') == 'g6m4':
            why.append(f'g6m4 process {pid} is running')
    mem = h.read('/proc/meminfo') or ''
    m = re.search(r'MemAvailable:\s+(\d+) kB', mem)
    if m and int(m.group(1)) / 2 ** 20 < th.min_free_gib:
        why.append(f'MemAvailable {int(m.group(1)) / 2 ** 20:.1f} GiB (< {th.min_free_gib:g}): a model looks loaded')
    st = smi_state(h)
    if st.get('util_pct') is not None and st['util_pct'] >= th.busy_gpu_util_pct:
        why.append(f'GPU utilization {st["util_pct"]:g}%')
    return why, st


def wedge_check(h, th):
    """GPU_WEDGED evidence: nvidia-smi -L must answer. D-state processes and coredump writers are listed as context."""
    rc, out, err = h.run(['nvidia-smi', '-L'], th.wedge_timeout_s)
    dstate, core = [], []
    for pid in h.listdir('/proc'):
        if not pid.isdigit():
            continue
        stat = h.read(f'/proc/{pid}/stat') or ''
        m = re.match(r'\d+ \((.*)\) (\S)', stat)
        if not m:
            continue
        comm, state = m.group(1), m.group(2)
        if state == 'D':
            dstate.append(f'{pid}:{comm}')
        if 'coredump' in comm or comm.startswith('apport') or comm == 'core':
            core.append(f'{pid}:{comm}')
    wedged = rc is None or rc != 0 or 'GPU 0' not in out
    detail = None
    if wedged:
        detail = (f'nvidia-smi -L timed out after {th.wedge_timeout_s:g} s' if rc is None
                  else f'nvidia-smi -L failed: {(err or out).strip()[:150]}' if rc != 0 else 'nvidia-smi -L lists no GPU')
    return {'wedged': wedged, 'detail': detail, 'd_state': dstate[:10], 'coredump_procs': core[:10]}


def roce_gid(ip):
    """IPv4 address -> the text of its IPv4-mapped GID as sysfs shows it, or None"""
    try:
        b = socket.inet_aton(ip)
    except OSError:
        return None
    return '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' % tuple(b)


def rails_state(h, ips):
    """For each fabric address: its netdev, RDMA device, RoCE v2 GID index, port state, and the ConnectX PCIe link."""
    out = []
    for rail, ip in enumerate(ips):
        if not ip:
            continue
        r = {'rail': rail, 'ip': ip, 'netdev': None, 'ibdev': None, 'gid_index': None, 'state': None, 'rate': None,
             'pcie': None}
        mapped = roce_gid(ip)
        for ib in h.listdir('/sys/class/infiniband'):
            base = f'/sys/class/infiniband/{ib}/ports/1'
            for gi in h.listdir(f'{base}/gids'):
                gid = h.read(f'{base}/gids/{gi}')
                typ = h.read(f'{base}/gid_attrs/types/{gi}') or ''
                if gid and mapped and gid.lower() == mapped and 'v2' in typ.lower():
                    r.update(ibdev=ib, gid_index=int(gi), netdev=h.read(f'{base}/gid_attrs/ndevs/{gi}'),
                             state=h.read(f'{base}/state'), phys_state=h.read(f'{base}/phys_state'), rate=h.read(f'{base}/rate'))
                    dev = f'/sys/class/infiniband/{ib}/device'
                    pc = {k: h.read(f'{dev}/{k}') for k in ('current_link_speed', 'current_link_width', 'max_link_speed',
                                                            'max_link_width')}
                    if all(pc.values()):
                        def gts(s):
                            return s.split()[0]
                        r['pcie'] = {'cur_speed': gts(pc['current_link_speed']), 'cur_width': int(pc['current_link_width']),
                                     'max_speed': gts(pc['max_link_speed']), 'max_width': int(pc['max_link_width'])}
                        for k in ('cur', 'max'):
                            p = r['pcie']
                            p[f'{k}_gbps'] = round(PCIE_GBPS_PER_LANE.get(p[f'{k}_speed'], 0.0) * p[f'{k}_width'], 1)
                    break
            if r['ibdev']:
                break
        out.append(r)
    return out


def cpu_times(h):
    """{cpu: (busy jiffies, total jiffies)} from /proc/stat"""
    t = {}
    for ln in (h.read('/proc/stat') or '').splitlines():
        m = re.match(r'cpu(\d+) (.*)', ln)
        if m:
            v = [int(x) for x in m.group(2).split()]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            t[int(m.group(1))] = (sum(v[:8]) - idle, sum(v[:8]))
    return t


def proxy_cpu_state(h, proxy_cpus, sample_s=1.5):
    """Online / allowed / busy while the node is idle, for the CPUs the proxy threads are pinned to."""
    online = set(parse_cpu_list(h.read('/sys/devices/system/cpu/online') or ''))
    allowed = set()
    for ln in (h.read('/proc/self/status') or '').splitlines():
        if ln.startswith('Cpus_allowed_list:'):
            allowed = set(parse_cpu_list(ln.split(':', 1)[1]))
    a = cpu_times(h)
    h.sleep(sample_s)
    b = cpu_times(h)
    busy = {}
    for c in proxy_cpus:
        if c in a and c in b and b[c][1] > a[c][1]:
            busy[c] = round(100.0 * (b[c][0] - a[c][0]) / (b[c][1] - a[c][1]), 1)
    return {'cpus': list(proxy_cpus), 'offline': [c for c in proxy_cpus if online and c not in online],
            'not_allowed': [c for c in proxy_cpus if allowed and c not in allowed], 'busy_pct': busy,
            'allowed_other': sorted(allowed - set(proxy_cpus))}


def parse_gpuprobe(text):
    """gpuprobe stdout -> (dict of its JSON line, load started seen)"""
    res, seen = None, False
    for ln in text.splitlines():
        ln = ln.strip()
        if ln == 'LOAD':
            seen = True
        elif ln.startswith('{'):
            try:
                res = json.loads(ln)
            except ValueError:
                pass
    return res, seen


def gpu_load(h, th, tool, taskset_cpus):
    """Run gpuprobe and sample the GPU while its read kernels run. -> dict"""
    if not tool or not os.path.exists(h.path(tool)):
        return {'error': f'gpuprobe not found at {tool} (scripts/build.sh builds it, scripts/stage.sh copies it)'}
    argv = [tool, '--seconds', f'{th.load_seconds:g}']
    if taskset_cpus and shutil.which('taskset'):
        argv = ['taskset', '-c', ','.join(map(str, taskset_cpus))] + argv
    p = h.popen(argv)
    lines, ev = [], threading.Event()

    def reader():
        for ln in p.stdout:
            lines.append(ln)
            if ln.strip() == 'LOAD':
                ev.set()
        ev.set()                                        # the tool ended (an error before the load phase)
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    samples = []
    if ev.wait(30):
        h.sleep(min(1.0, th.load_seconds / 4))          # let the clock settle under load
        for _ in range(3):
            s = smi_state(h)
            if 'error' not in s:
                samples.append(s)
            h.sleep(0.5)
    try:
        p.wait(timeout=th.load_seconds + 30)
    except subprocess.TimeoutExpired:
        p.kill()
        return {'error': 'gpuprobe did not finish'}
    t.join(5)
    err_text = p.stderr.read().strip()[:200] if p.stderr else ''
    for f in (p.stdout, p.stderr):
        if f:
            f.close()
    res, seen = parse_gpuprobe(''.join(lines))
    if p.returncode != 0 or not res or 'error' in res:
        return {'error': (res or {}).get('error') or err_text or f'gpuprobe rc {p.returncode}'}
    clocks = [s['sm_mhz'] for s in samples if s.get('sm_mhz') is not None]
    mask = 0
    for s in samples:
        mask |= s.get('throttle_mask') or 0
    return {'hbm_gbs': res['hbm_gbs'], 'h2d_gbs': res['h2d_gbs'], 'device': res.get('device'),
            'clock_load_mhz': statistics.median(clocks) if clocks else None, 'clock_samples': clocks,
            'max_sm_mhz': samples[0].get('max_sm_mhz') if samples else None,
            'util_load_pct': max([s.get('util_pct') or 0 for s in samples], default=None),
            'power_w': samples[-1].get('power_w') if samples else None,
            'temp_c': samples[-1].get('temp_c') if samples else None,
            'throttle_mask': mask, 'throttle': [n for b, n in THROTTLE_BITS.items() if mask & b]}


def local_probe(h, th, proxy_cpu, gpu_tool, rail_ips, n_proxy=5):
    """One node. Never raises for a bad node: what is wrong is in the result."""
    t0 = time.time()
    res = {'host': h.read('/proc/sys/kernel/hostname') or '?'}
    why, st = busy_reasons(h, th)
    if why:
        res.update(busy=why, elapsed_s=round(time.time() - t0, 1))
        return res
    res['wedge'] = wedge_check(h, th)
    res['rails'] = rails_state(h, rail_ips)
    proxy_cpus = [proxy_cpu - k for k in range(n_proxy) if proxy_cpu - k >= 0]
    res['proxy_cpu'] = proxy_cpu_state(h, proxy_cpus)
    if not res['wedge']['wedged']:
        res['gpu'] = gpu_load(h, th, gpu_tool, res['proxy_cpu']['allowed_other'])
    res['elapsed_s'] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------------------------- link ring

IB_WRITE_BW = 'ib_write_bw -F -s 1048576 --report_gbits'


def parse_ib_bw(text):
    """Average bandwidth (Gb/s) from the result table of ib_write_bw, or None.
    Row: #bytes #iterations BW peak[Gb/sec] BW average[Gb/sec] MsgRate[Mpps]"""
    for ln in reversed(text.splitlines()):
        f = ln.split()
        if len(f) >= 5:
            try:
                int(f[0]); int(f[1]); float(f[2])
                return float(f[3])
            except ValueError:
                continue
    return None


def ring_script(r, n, rails, dur=3, port0=18700, tmo=25):
    """Shell script for rank r: a server for each rail (peer r-1 writes to me) and a client for each rail (I write to
    rank r+1). `rails` = per rank, per rail: {'ibdev','gid_index','ip'}. All ranks run it at once."""
    to = (r + 1) % n
    lines = ['command -v ib_write_bw >/dev/null || { echo "@@MISSING ib_write_bw"; exit 0; }', 'T=$(mktemp -d)']
    for k, mine in enumerate(rails[r]):
        base = f'{IB_WRITE_BW} -d {mine["ibdev"]} -x {mine["gid_index"]} -D {dur} -p {port0 + k}'
        lines.append(f'timeout {tmo} {base} >/dev/null 2>&1 &')
    lines.append('sleep 1.5')
    # The TCP control connection of ib_write_bw goes to the peer's first (fabric) address: a host firewall can drop TCP on the
    # other rail's subnet, and the engine's own handshake uses only the first address too. The RDMA data path is the rail's
    # device and GID, which the two sides exchange over that connection.
    tgt = rails[to][0]['ip']
    for k, mine in enumerate(rails[r]):
        base = f'{IB_WRITE_BW} -d {mine["ibdev"]} -x {mine["gid_index"]} -D {dur} -p {port0 + k}'
        lines.append(f'( for i in 1 2 3 4 5 6; do o=$(timeout {tmo} {base} {tgt} 2>&1); c=$?; [ $c = 0 ] && break; sleep 1; done; '
                     f'{{ echo "@@CLIENT rail={k} to={to} rc=$c"; echo "$o"; }} > $T/{k} ) &')
    lines.append('wait')
    lines.append('cat $T/*; rm -rf $T')    # one block per rail, never interleaved
    return '\n'.join(lines) + '\n'


def parse_ring_output(text):
    """-> ({rail: gbps or None}, error or None)"""
    if '@@MISSING' in text:
        return {}, 'ib_write_bw is not installed (package perftest)'
    out = {}
    for block in text.split('@@CLIENT ')[1:]:
        head, _, body = block.partition('\n')
        m = re.match(r'rail=(\d+) to=(\d+) rc=(\d+)', head)
        if m:
            if int(m.group(1)) in out:
                return {}, f'RDMA write test reported rail {m.group(1)} twice'
            out[int(m.group(1))] = {'to': int(m.group(2)), 'rc': int(m.group(3)), 'gbps': parse_ib_bw(body),
                                    'tail': body.strip()[-160:] if parse_ib_bw(body) is None or int(m.group(3)) else ''}
    if not out:
        return {}, 'RDMA write test produced no client output'
    return out, None


# ---------------------------------------------------------------------------------------------- verdict

def median_of(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def judge(ranks, th):
    """ranks: list of {'rank','host','error'?,'local'?,'link'?: {rail: {'gbps','to','rc','tail'}}, 'link_error'?}.
    -> list of refusals {'rank','host','code','detail'}; empty = every rank admitted."""
    bad = []

    def add(rk, code, detail):
        bad.append({'rank': rk['rank'], 'host': rk.get('host'), 'code': code, 'detail': detail})
    probed = []
    for rk in ranks:
        loc = rk.get('local')
        if rk.get('error') or loc is None:
            add(rk, 'PROBE_FAILED', rk.get('error') or 'no result')
        elif loc.get('busy'):
            add(rk, 'NODE_BUSY', '; '.join(loc['busy']))
        else:
            probed.append(rk)
    # the ring runs only when every rank was probed and none is wedged (probe_cluster); then each rank owes a result per rail
    ring_expected = (len(ranks) > 1 and len(probed) == len(ranks)
                     and not any((rk['local'].get('wedge') or {}).get('wedged') for rk in probed)
                     and all(r.get('ibdev') for rk in probed for r in rk['local'].get('rails', [])))
    hbm_med = median_of([rk['local'].get('gpu', {}).get('hbm_gbs') for rk in probed]) if len(probed) >= 3 else None
    h2d_med = median_of([rk['local'].get('gpu', {}).get('h2d_gbs') for rk in probed]) if len(probed) >= 3 else None
    rails_n = max([len(rk['local'].get('rails', [])) for rk in probed], default=0)
    rail_med = {}
    for k in range(rails_n):
        rail_med[k] = median_of([(rk.get('link') or {}).get(k, {}).get('gbps') for rk in probed
                                 if (rk.get('link') or {}).get(k, {}).get('rc') == 0]) if len(probed) >= 3 else None
    for rk in probed:
        loc = rk['local']
        w = loc.get('wedge') or {}
        if w.get('wedged'):
            ctx = (f' (D-state: {", ".join(w["d_state"])})' if w.get('d_state') else '') + \
                  (f' (coredump: {", ".join(w["coredump_procs"])})' if w.get('coredump_procs') else '')
            add(rk, 'GPU_WEDGED', f'{w["detail"]}{ctx}; only a power cycle recovers a held RM lock')
        g = loc.get('gpu')
        if g is not None and g.get('error'):
            add(rk, 'PROBE_FAILED', f'gpuprobe: {g["error"]}')
        elif g:
            clk = g.get('clock_load_mhz')
            if clk is None:
                add(rk, 'PROBE_FAILED', 'no GPU clock sample under load')
            elif clk < th.min_clock_mhz:
                names = ', '.join(g.get('throttle') or []) or 'none reported'
                add(rk, 'GPU_CLOCK_LOW', f'SM clock {clk:g} MHz under load (minimum {th.min_clock_mhz:g}; throttle reasons: {names})')
            if (g.get('throttle_mask') or 0) & HW_THROTTLE:
                add(rk, 'GPU_HW_THROTTLE', 'hardware throttle active under load: ' + ', '.join(g['throttle']))
            if hbm_med and g['hbm_gbs'] < hbm_med * (1 - th.hbm_tol):
                add(rk, 'HBM_SLOW', f'device-memory read {g["hbm_gbs"]:g} GB/s, median of ranks {hbm_med:g} (tolerance {th.hbm_tol:.0%})')
            if h2d_med and g['h2d_gbs'] < h2d_med * (1 - th.h2d_tol):
                add(rk, 'H2D_SLOW', f'host-to-device copy {g["h2d_gbs"]:g} GB/s, median of ranks {h2d_med:g} (tolerance {th.h2d_tol:.0%})')
        pc = loc.get('proxy_cpu') or {}
        if pc.get('offline') or pc.get('not_allowed'):
            add(rk, 'PROXY_CPU_OFFLINE', f'proxy CPU(s) {pc.get("offline") or pc.get("not_allowed")} offline or outside the CPU set')
        for c, pct in (pc.get('busy_pct') or {}).items():
            if pct > th.max_proxy_cpu_busy_pct:
                add(rk, 'PROXY_CPU_BUSY', f'CPU {c} is {pct:g}% busy while the node is idle (limit {th.max_proxy_cpu_busy_pct:g}%)')
        rails = loc.get('rails') or []
        for r in rails:
            if r.get('ibdev') is None:
                add(rk, 'PROBE_FAILED', f'no RDMA device owns {r["ip"]} (rail {r["rail"]})')
                continue
            if r.get('state') and 'ACTIVE' not in r['state'].upper():
                add(rk, 'LINK_DOWN', f'{r["ibdev"]} (rail {r["rail"]}) port state {r["state"]}')
            p = r.get('pcie')
            if p and (p['cur_gbps'] < p['max_gbps']):
                add(rk, 'LINK_PCIE_DEGRADED', f'{r["ibdev"]} (rail {r["rail"]}) PCIe {p["cur_speed"]} GT/s x{p["cur_width"]} '
                                              f'(max {p["max_speed"]} GT/s x{p["max_width"]})')
        if ring_expected:
            if rk.get('link_error'):
                add(rk, 'PROBE_FAILED', rk['link_error'])
            link = {int(k): v for k, v in (rk.get('link') or {}).items()}
            for r in rails:                       # fail closed: one successful client result per configured rail
                k = r['rail']
                l = link.get(k)
                if l is None:
                    if not rk.get('link_error'):
                        add(rk, 'PROBE_FAILED', f'RDMA write test: no result for rail {k} (the client did not report)')
                elif l.get('rc') != 0 or l.get('gbps') is None:
                    add(rk, 'PROBE_FAILED', f'RDMA write test rail {k} to rank {l.get("to")}: rc {l.get("rc")}, '
                                            f'{"no bandwidth row" if l.get("gbps") is None else "bandwidth " + format(l["gbps"], "g") + " Gb/s"}'
                                            f'{": " + l["tail"] if l.get("tail") else ""}')
                else:
                    floor = th.min_link_gbps
                    if rail_med.get(k):
                        floor = max(floor, rail_med[k] * (1 - th.link_peer_tol))
                    if l['gbps'] < floor:
                        add(rk, 'LINK_SLOW', f'RDMA write rail {k} to rank {l["to"]}: {l["gbps"]:g} Gb/s (floor {floor:g}; '
                                             f'the rank sending is named: its TX, or rank {l["to"]}\'s RX)')
    return bad


# ---------------------------------------------------------------------------------------------- cluster

class Ssh:
    """Runs things on the ranks' nodes; tests replace it."""
    def __init__(self, hosts):
        self.hosts = hosts

    def run(self, rank, argv, stdin=None, timeout=60):
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', self.hosts[rank]] + argv
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return None, '', 'timed out'


def probe_cluster(ssh, n, script_text, args, th, rail_ips, log=lambda m: None):
    """-> list of per-rank dicts for judge()"""
    ranks = [{'rank': r, 'host': ssh.hosts[r]} for r in range(n)]

    def one(r):
        argv = ['python3', '-', 'local', '--proxy-cpu', str(args.proxy_cpu), '--gpu-tool', f'{args.home}/bin/gpuprobe',
                '--rail-ips', ','.join(ips[r] for ips in rail_ips), '--load-seconds', f'{th.load_seconds:g}']
        rc, out, err = ssh.run(r, argv, stdin=script_text, timeout=90)
        if rc is None or rc != 0:
            ranks[r]['error'] = f'probe did not run on {ssh.hosts[r]}: {"timed out" if rc is None else (err.strip()[-200:] or f"rc {rc}")}'
            return
        try:
            ranks[r]['local'] = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            ranks[r]['error'] = f'unreadable probe output from {ssh.hosts[r]}: {out.strip()[-120:]!r}'
    with ThreadPoolExecutor(max_workers=max(n, 1)) as ex:
        list(ex.map(one, range(n)))
    log('local probes done')
    ok = [rk for rk in ranks if rk.get('local') and not rk['local'].get('busy') and not rk['local'].get('wedge', {}).get('wedged')]
    if len(ok) == n and n > 1 and all(r.get('ibdev') for rk in ok for r in rk['local']['rails']):
        rails = [[{'ibdev': r['ibdev'], 'gid_index': r['gid_index'], 'ip': r['ip']} for r in rk['local']['rails']] for rk in ranks]

        def ring(r):
            rc, out, err = ssh.run(r, ['bash', '-s'], stdin=ring_script(r, n, rails), timeout=60)
            if rc is None or rc != 0:
                ranks[r]['link_error'] = f'RDMA write test did not run: {"timed out" if rc is None else err.strip()[-160:]}'
                return
            link, e = parse_ring_output(out)
            if e:
                ranks[r]['link_error'] = e
            ranks[r]['link'] = link
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(ring, range(n)))
        log('link ring done')
    return ranks


def cmd_cluster(a):
    t0 = time.time()
    hosts = a.hosts.split(',')
    ips = a.ips.split(',')
    ips2 = a.ips2.split(',') if a.ips2 else []
    n = len(hosts)
    if len(ips) != n or (ips2 and len(ips2) != n):
        sys.exit(f'--ips / --ips2 need one address per host ({n})')
    th = Thresholds(**{k: getattr(a, k) for k in asdict(Thresholds()) if getattr(a, k, None) is not None})
    rail_ips = [ips] + ([ips2] if ips2 else [])
    script = open(os.path.abspath(__file__)).read()
    ranks = probe_cluster(Ssh(hosts), n, script, a, th, rail_ips)
    bad = judge(ranks, th)
    result = {'format': 1, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'admitted': not bad,
              'elapsed_s': round(time.time() - t0, 1), 'thresholds': asdict(th), 'hosts': hosts, 'ranks': ranks,
              'refusals': bad}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(result, f, indent=1)
    print(render(result))
    sys.exit(0 if not bad else 3)


def render(result):
    rows = []
    for rk in result['ranks']:
        loc = rk.get('local') or {}
        g = loc.get('gpu') or {}
        link = rk.get('link') or {}
        rows.append(f'  rank {rk["rank"]} {rk.get("host", "?")}: clock {g.get("clock_load_mhz", "-")} MHz, '
                    f'HBM read {g.get("hbm_gbs", "-")} GB/s, H2D {g.get("h2d_gbs", "-")} GB/s, '
                    f'RDMA TX {", ".join(f"{l.get("gbps")}" for _, l in sorted(link.items())) or "-"} Gb/s')
    lines = [f'rank health probe, {result["elapsed_s"]} s'] + rows
    if result['admitted']:
        lines.append(f'ADMITTED: all {len(result["ranks"])} ranks passed')
    else:
        lines.append('REFUSED:')
        for b in result['refusals']:
            lines.append(f'  rank {b["rank"]} ({b["host"]}): {b["code"]}: {b["detail"]}')
    return '\n'.join(lines)


def cmd_local(a):
    th = Thresholds(load_seconds=a.load_seconds)
    res = local_probe(Host(), th, a.proxy_cpu, a.gpu_tool, [x for x in a.rail_ips.split(',') if x])
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('local', help="one node's measurements as JSON")
    p.add_argument('--proxy-cpu', type=int, required=True)
    p.add_argument('--gpu-tool', default='gpuprobe')
    p.add_argument('--rail-ips', default='', help="this node's fabric address on each rail, comma separated")
    p.add_argument('--load-seconds', type=float, default=Thresholds.load_seconds)
    p.set_defaults(func=cmd_local)
    p = sub.add_parser('cluster', help='probe every rank and decide admission (exit 3 = refused)')
    p.add_argument('--hosts', required=True, help='ssh names in rank order, comma separated')
    p.add_argument('--ips', required=True, help='rail 1 address of each rank')
    p.add_argument('--ips2', default='', help='rail 2 address of each rank')
    p.add_argument('--home', required=True, help='G6_HOME on the nodes (bin/gpuprobe lives there)')
    p.add_argument('--proxy-cpu', type=int, required=True)
    p.add_argument('--out')
    for k, v in asdict(Thresholds()).items():
        p.add_argument('--' + k.replace('_', '-'), dest=k, type=float, default=None, help=f'default {v:g}')
    p.set_defaults(func=cmd_cluster)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
