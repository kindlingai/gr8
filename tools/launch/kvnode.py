#!/usr/bin/env python3
"""./gr8 kv <command>: a KV node on this box, started like a model. Python 3 standard library only.

    up --gb N [OPTIONS] [@FILE]   lock N GiB of this box's memory and serve it as KV page blocks (kvman-remote,
                                  engine/kvman-rs/REMOTE-TIER.md), announce it in the mentat group, and follow it until
                                  it serves
    down                          stop this box's KV node
    status [all]                  this box's node: its memory, leases and what is free; `all`: every KV node the
                                  mentat group announces
    logs [-f]                     the node's log

OPTIONS are --key value (or KEY=value lines in a config file, or KEY in the environment), as for ./gr8 glm53: KEYS
below lists them. --no-rdma serves over TCP only, --foreground runs the supervisor in this process. The node joins the model's mentat group
(gr8-glm53 by default) as an agent without GPUs, so the cluster's election never counts it, and announces the service
`gr8-kv` with its port, and in meta its size and its RDMA addresses. Exit status: 0 serving (or done), 1 failed,
2 bad options, 3 status: not running.
"""
import fcntl
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mentat  # noqa: E402
import options as opt  # noqa: E402

HERE = os.path.abspath(__file__)
REPO = opt.REPO
SERVICE = mentat.KV_SERVICE

# key: (default, meaning). The flag is --<key in lower case without KV_, _ as ->, except where FLAGS says otherwise.
KEYS = {
    'KV_GB': (None, 'GiB of this box\'s memory to lock and serve'),
    'KV_PORT': ('7350', 'the node\'s TCP port (RDMA connections are set up through it too)'),
    'KV_BIND': ('0.0.0.0', 'the address the node listens on'),
    'KV_MIN_FREE_GB': ('10', 'refuse to start when locking KV_GB would leave less MemAvailable than this'),
    'KV_RDMA': ('1', '0 = serve over TCP only'),
    'KV_BIN': (None, 'the kvman-remote binary (default bin/kvman-remote, else the crate\'s target/release)'),
    'MENTAT': ('localhost', 'the local mentat daemon to announce through; none = do not announce'),
    'GROUP': ('gr8-glm53', 'mentat group the node joins'),
    'NODE_IP': (None, "this box's LAN address, when mentat cannot say"),
    'G6_HOME': (REPO, 'working directory: the node writes out/kv/'),
}
FLAGS = {'G6_HOME': '--g6-home', 'MENTAT': '--mentat', 'GROUP': '--group', 'NODE_IP': '--node-ip'}
SWITCHES = {'--no-config', '--foreground', '--no-rdma'}


def flag_of(key):
    return FLAGS.get(key, '--' + key[3:].lower().replace('_', '-'))


KEY_OF_FLAG = {flag_of(k): k for k in KEYS}


class Options:
    def __init__(self, values, origin, file, args, switches):
        self.values, self.origin, self.file, self.args, self.switches = values, origin, file, args, switches

    def get(self, key, default=None):
        v = self.values.get(key)
        return default if v is None or v == '' else v


def parse(argv, environ=None):
    """Command line > one config file (@FILE, else the one ./gr8 glm53 reads) > environment > default. A shared config
    file holds the model's keys too: this reads only its own."""
    environ = os.environ if environ is None else environ
    cmd, at, args, switches = {}, None, [], set()
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('@'):
            at = a[1:]
        elif a in SWITCHES:
            switches.add(a)
        elif '=' in a and a.split('=', 1)[0] in KEYS:
            k, _, v = a.partition('=')
            cmd[k] = v
        elif a.startswith('--'):
            name, eq, val = a.partition('=')
            key = KEY_OF_FLAG.get(name)
            if key is None:
                raise opt.OptionError(f'unknown option {name}: ./gr8 kv --help lists them')
            if not eq:
                i += 1
                if i >= len(argv):
                    raise opt.OptionError(f'{name} needs a value')
                val = argv[i]
            cmd[key] = val
        else:
            args.append(a)
        i += 1
    if '--no-rdma' in switches:
        cmd['KV_RDMA'] = '0'
    path = at or (None if '--no-config' in switches else opt.default_file(environ))
    filev = opt.parse_file(path) if path and os.path.isfile(path) else {}
    if at and not os.path.isfile(at):
        raise opt.OptionError(f'no config file {at}')
    values, origin = {}, {}
    for k, (dflt, _) in KEYS.items():
        for src, d in (('command line', cmd), ('config file', filev), ('environment', environ)):
            if k in d:
                values[k], origin[k] = d[k], src
                break
        else:
            values[k], origin[k] = dflt, 'default'
    values['G6_HOME'] = os.path.abspath(values['G6_HOME'])
    return Options(values, origin, path, args, switches)


def validate(o):
    try:
        gb = float(o.get('KV_GB') or 'nan')
    except ValueError:
        gb = float('nan')
    if not gb > 0:
        raise opt.OptionError('name the memory to lock: ./gr8 kv up --gb N (GiB)')
    for k in ('KV_PORT', 'KV_MIN_FREE_GB'):
        try:
            float(o.get(k))
        except (TypeError, ValueError):
            raise opt.OptionError(f'{k}={o.get(k)!r}: want a number')


def out_dir(o):
    return os.path.join(o.get('G6_HOME'), 'out', 'kv')


def log(msg):
    for line in str(msg).splitlines() or ['']:
        print(f'{time.strftime("%H:%M:%S")} {line}', flush=True)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def node_bin(o):
    for p in (o.get('KV_BIN'), os.path.join(o.get('G6_HOME'), 'bin', 'kvman-remote'),
              os.path.join(REPO, 'engine', 'kvman-rs', 'target', 'release', 'kvman-remote')):
        if p and os.access(p, os.X_OK):
            return p
    return None


def running(out):
    """The pid of this box's KV supervisor, or None (it holds out/kv/kv.lock while it runs)."""
    try:
        with open(os.path.join(out, 'kv.lock'), 'a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return read_json(os.path.join(out, 'state.json')).get('pid') or -1
            fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        pass
    return None


# ---- the node's STATS request (engine/kvman-rs/src/remote/proto.rs) ---------------------------------------------------
MAGIC = 0x5653_3647 ^ 1
STATS = 7


def stats(host, port, timeout=3):
    """The node's own account of itself: bytes, free, leases."""
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.sendall(struct.pack('<II3Q', MAGIC, STATS, 0, 0, 0))
        h = recv_exact(s, 32)
        magic, status, a, _, c = struct.unpack('<II3Q', h)
        if magic != MAGIC:
            raise OSError(f'{host}:{port} is not a KV node of this version')
        if status:
            raise OSError(f'{host}:{port}: {recv_exact(s, c).decode(errors="replace")}')
        return json.loads(recv_exact(s, a))


def recv_exact(s, n):
    buf = b''
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise OSError('the KV node closed the connection')
        buf += c
    return buf


def gib(n):
    return f'{n / (1 << 30):.2f} GiB'


def describe(st):
    lines = [f'{gib(st["bytes"])} {"locked" if st.get("locked") else "NOT locked"}, {gib(st["free"])} free '
             f'(largest range {gib(st["largest_free"])}), {len(st["leases"])} leases, {st.get("opened", 0)} opened '
             f'since start, up {st.get("uptime_s", 0)} s']
    for le in st['leases']:
        lines.append(f'  lease {le["id"]:<4} {le["name"]:<32} {gib(le["bytes"])}, blocks of {le["block"]} B, '
                     f'{le["conns"]} connections, {"RDMA on " + ",".join(le["rdma"]) if le["rdma"] else "TCP"}, '
                     f'{gib(le["read"])} read and {gib(le["written"])} written over TCP, {le["age_s"]} s old')
    return '\n'.join(lines)


def kv_meta(ready, node_ip):
    """The agent's meta: what a client needs before it connects."""
    return {'gr8.kv.port': ready['port'], 'gr8.kv.bytes': ready['bytes'], 'gr8.kv.lan': node_ip,
            'gr8.kv.rdma': ','.join(r['ip'] for r in ready.get('rdma', [])), 'gr8.state': 'serving'}


# ---- the supervisor: start the node, announce it, stop both ----------------------------------------------------------
class Supervisor:
    def __init__(self, o):
        self.o, self.out = o, out_dir(o)
        self.node = self.agent = None
        self.stop_ev = threading.Event()
        self.locked = False
        self.node_out = None     # the node's output: this process's (the supervisor's log)

    def set_state(self, state, **kw):
        if not self.locked:
            return
        tmp = os.path.join(self.out, 'state.json.tmp')
        with open(tmp, 'w') as f:
            json.dump({'state': state, 'pid': os.getpid(), 'time': time.time(), **kw}, f)
        os.replace(tmp, os.path.join(self.out, 'state.json'))

    def run(self):
        o = self.o
        os.makedirs(self.out, exist_ok=True)
        self.lock = open(os.path.join(self.out, 'kv.lock'), 'w')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('a KV node already runs here: ./gr8 kv down first')
        self.locked = True
        self.set_state('starting')
        b = node_bin(o)
        if not b:
            raise RuntimeError('no kvman-remote: ./gr8 glm53 build, or cargo build --release --manifest-path '
                               'engine/kvman-rs/Cargo.toml')
        ready_path = os.path.join(self.out, 'ready.json')
        try:
            os.unlink(ready_path)
        except FileNotFoundError:
            pass
        argv = [b, '--gb', o.get('KV_GB'), '--port', o.get('KV_PORT'), '--bind', o.get('KV_BIND'),
                '--min-free-gb', o.get('KV_MIN_FREE_GB'), '--ready', ready_path]
        if o.get('KV_RDMA') == '0':
            argv.append('--no-rdma')
        log(f'starting {" ".join(argv)}')
        self.node = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=self.node_out, stderr=self.node_out,
                                     start_new_session=True)
        with open(os.path.join(self.out, 'node.pid'), 'w') as f:
            f.write(f'{self.node.pid}\n')
        ready = {}
        while not ready:
            if self.node.poll() is not None:
                raise RuntimeError(f'kvman-remote exited ({self.node.returncode}) before it served')
            if self.stop_ev.wait(0.1):
                return
            r = read_json(ready_path)
            if r.get('pid') == self.node.pid:
                ready = r
        rd = ', '.join(f'{r["ip"]} ({r["dev"]}, GID {r["gid"]})' for r in ready.get('rdma', [])) or 'none'
        log(f'serving {gib(ready["bytes"])} on port {ready["port"]}, locked {ready["locked"]}; RDMA on {rd}')
        self.announce(ready)
        self.set_state('serving', port=ready['port'], node_pid=self.node.pid)
        while not self.stop_ev.wait(1):
            if self.node.poll() is not None:
                raise RuntimeError(f'kvman-remote exited ({self.node.returncode})')

    def announce(self, ready):
        o = self.o
        if o.get('MENTAT') == 'none':
            log('not announced (--mentat none)')
            return
        d = mentat.Daemon(o.get('MENTAT'))
        node_ip = o.get('NODE_IP')
        if not node_ip:
            try:
                node_ip = mentat.node_view(d.local_status())[1]
            except OSError as e:
                raise RuntimeError(f'no mentat daemon at {d.http} ({e}): start mentatd on this box, or pass '
                                   '--mentat none')
        g = o.get('GROUP')
        self.agent = mentat.Agent(d, g, f'{g}@{SERVICE}@{node_ip}', node_ip, gpus=0, log=log,
                                  services={SERVICE: {'host': '', 'port': int(ready['port']), 'path': '',
                                                      'provider': 'gr8'}})
        self.agent.set_meta(**kv_meta(ready, node_ip))
        self.agent.start()
        log(f'announced to mentat as {g}@{SERVICE}@{node_ip}')

    def stop_node(self):
        p = self.node
        if not p or p.poll() is not None:
            return
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        log(f'kvman-remote (pid {p.pid}) stopped')

    def shutdown(self, code, why):
        if not self.locked:
            return
        if self.agent:
            self.agent.stop()
        self.stop_node()
        self.set_state(f'failed: {why}' if code else 'stopped')
        self.lock.close()


def supervise(o, node_out=None):
    s = Supervisor(o)
    s.node_out = node_out
    signal.signal(signal.SIGTERM, lambda *_: s.stop_ev.set())
    signal.signal(signal.SIGINT, lambda *_: s.stop_ev.set())
    code, why = 0, ''
    try:
        s.run()
    except RuntimeError as e:
        code, why = 1, str(e)
        log(why)
    except Exception as e:
        import traceback
        log(traceback.format_exc())
        code, why = 1, f'{type(e).__name__}: {e}'
    s.shutdown(code, why)
    return code


# ---- commands ---------------------------------------------------------------------------------------------------------
def parsed(argv, check=False):
    try:
        o = parse(argv)
        if check:
            validate(o)
        return o
    except opt.OptionError as e:
        print(f'gr8 kv: {e}', file=sys.stderr)
        sys.exit(2)


def cmd_up(argv):
    o = parsed(argv, check=True)
    out = out_dir(o)
    os.makedirs(out, exist_ok=True)
    pid = running(out)
    if pid:
        print(f'gr8 kv: a KV node already runs on this box (supervisor pid {pid}): ./gr8 kv down first',
              file=sys.stderr)
        return 1
    if '--foreground' in o.switches:
        return supervise(o)
    logp = os.path.join(out, 'kv.log')
    start = os.path.getsize(logp) if os.path.exists(logp) else 0
    with open(logp, 'ab') as lf:
        child = subprocess.Popen([sys.executable, HERE, 'supervise', *argv], stdout=lf, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    print(f'KV supervisor pid {child.pid}, log {logp}', flush=True)
    with open(logp, 'rb') as f:
        f.seek(start)
        while True:
            data = f.read()
            if data:
                sys.stdout.write(data.decode(errors='replace'))
                sys.stdout.flush()
            rc = child.poll()
            st = read_json(os.path.join(out, 'state.json'))
            if rc is not None:
                sys.stdout.write(f.read().decode(errors='replace'))
                return rc or 1
            if st.get('state') == 'serving' and st.get('pid') == child.pid:
                return 0
            time.sleep(0.2)


def cmd_down(argv):
    o = parsed(argv)
    out = out_dir(o)
    pid = running(out)
    if not pid or pid < 0:
        print('this box: no KV node runs')
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(200):
        if not running(out):
            print('this box: KV node stopped')
            return 0
        time.sleep(0.1)
    print(f'the KV supervisor (pid {pid}) did not stop; killing it and its node')
    node = read_json(os.path.join(out, 'state.json')).get('node_pid')
    for p in (pid, node):
        if p:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return 0


def cmd_status(argv):
    o = parsed(argv)
    if 'all' in o.args:
        return status_all(o)
    out = out_dir(o)
    pid, st = running(out), read_json(os.path.join(out, 'state.json'))
    if not pid:
        print(f'this box: no KV node runs (last state: {st.get("state", "none")})')
        return 3
    print(f'this box: KV supervisor pid {pid}, {st.get("state")}')
    if st.get('port'):
        try:
            print(describe(stats('127.0.0.1', st['port'])))
        except OSError as e:
            print(f'  the node does not answer on port {st["port"]}: {e}')
    return 0


def status_all(o):
    if o.get('MENTAT') == 'none':
        print('status all asks mentat; --mentat none', file=sys.stderr)
        return 2
    d = mentat.Daemon(o.get('MENTAT'))
    try:
        snap = d.group_status(o.get('GROUP'))
    except (OSError, RuntimeError) as e:
        print(f'gr8 kv: mentat: {e}', file=sys.stderr)
        return 1
    nodes = mentat.kv_nodes(snap, o.get('GROUP'))
    if not nodes:
        print(f'no KV node in group {o.get("GROUP")}')
        return 3
    for lan, port, meta in nodes:
        print(f'{lan}:{port}  RDMA {meta.get("gr8.kv.rdma") or "none"}')
        try:
            print(describe(stats(lan, port)))
        except OSError as e:
            print(f'  does not answer: {e}')
    return 0


def cmd_logs(argv):
    follow = '-f' in argv
    o = parsed([a for a in argv if a != '-f'])
    p = os.path.join(out_dir(o), 'kv.log')
    try:
        f = open(p, 'rb')
    except OSError as e:
        print(f'gr8 kv: {p}: {e.strerror}', file=sys.stderr)
        return 1
    with f:
        f.seek(max(0, os.path.getsize(p) - 8192))
        while True:
            sys.stdout.write(f.read().decode(errors='replace'))
            sys.stdout.flush()
            if not follow:
                return 0
            time.sleep(1)


def main(argv):
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__)
        print('KEYS:')
        for k, (d, m) in KEYS.items():
            print(f'  {flag_of(k):<16} {k:<16} {m}' + (f' (default {d})' if d and k != 'G6_HOME' else ''))
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'supervise':
        return supervise(parsed(rest, check=True))
    f = {'up': cmd_up, 'down': cmd_down, 'status': cmd_status, 'logs': cmd_logs}.get(cmd)
    if not f:
        print(f'gr8 kv: unknown command {cmd}\n{__doc__}', file=sys.stderr)
        return 2
    return f(rest)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
