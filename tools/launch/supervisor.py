"""The per-box supervisor: `./gr8 glm53 up` runs one on every box, with the same options.

In order: resolve the options and the model set; find the other boxes (the peer list, or the mentat group) and this
box's rank; start the control server on the LAN port; run the local checks and make this rank's model files ready;
handshake with the head; start this box's g6m4. The head also starts the front end, waits for the engine, starts the
shim and announces the API. Then the supervisor stays, answering status, logs and down on the control port, and stops
the cluster when a rank's engine exits.
"""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request

import fabric
import handshake
import hfsource
import mentat
import modelset
import options as opt
import steering_local

REPO = opt.REPO
# What the engine gets from the supervisor's environment, besides the plan's own variables.
ENGINE_ENV_KEEP = ('PATH', 'HOME', 'USER', 'LANG', 'LD_LIBRARY_PATH')
ENGINE_ENV_PREFIXES = ('CUDA_', 'NVIDIA_')


class Stop(Exception):
    """The boot ends here: code is the exit status."""

    def __init__(self, code, why):
        super().__init__(why)
        self.code = code


def log(msg):
    for line in str(msg).splitlines() or ['']:
        print(f'{time.strftime("%H:%M:%S")} {line}', flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError):
        return False
    except PermissionError:
        return True


def read_pid(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def started_at(pid, proc='/proc'):
    """When pid started, in seconds since the epoch, or None where /proc cannot say (macOS, a hidden process)."""
    try:
        with open(os.path.join(proc, str(pid), 'stat')) as f:
            ticks = int(f.read().rsplit(')', 1)[1].split()[19])    # field 22, starttime; fields after comm start at 3
        with open(os.path.join(proc, 'stat')) as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith('btime '))
    except (OSError, IndexError, ValueError, StopIteration):
        return None
    return btime + ticks / os.sysconf('SC_CLK_TCK')


def recorded(path, proc='/proc'):
    """The pid in an out/<name>.pid file while that process runs, else None. The pid files outlive their processes
    (a reboot, a kill -9), and a process that started after its file was written took the number later: it is not
    ours, so nothing refuses a boot for it or signals it."""
    pid = read_pid(path)
    if not (pid and alive(pid)):
        return None
    t = started_at(pid, proc)
    try:
        if t is not None and t > os.path.getmtime(path) + 2:      # 2 s: btime is whole seconds
            return None
    except OSError:
        return None
    return pid


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def mem_available_gib():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return round(int(line.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return None


def version(home):
    """The checkout's commit, else the VERSION file of an unpacked release, else the engine build id."""
    try:
        v = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                           timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        v = ''
    if not v:
        try:
            with open(os.path.join(REPO, 'VERSION')) as f:
                v = f.read().strip()
        except OSError:
            v = f'engine {engine_build(home)[:12]}'
    return v


def resolve_model(o, refresh=False, log=log):
    """(model set, variant, error): the model set and its exlr8 v2 variant, either None with the reason in error. The
    variant's manifest becomes V2_WIDTHS, which the engine arguments take. With an https:// --model the experts, index
    slices and manifests come from the release (ms.release), and V2 and V2_SLICES point into the cache. The release is
    the whole model (V2_DENSE: the engine plans every weight from its files), unless --base-packs names packs for the
    non-expert weights. A local --model that holds the release's dense/ is used the same way. refresh: ask the hub
    what the revision is now (prepare), instead of the commit recorded at the last resolve."""
    ms = variant = release = None
    g6load = os.path.join(os.path.abspath(o.get('G6_HOME')), 'bin', 'g6load')   # plans and checks a Hugging Face fetch
    if os.access(g6load, os.X_OK):
        os.environ.setdefault('G6LOAD', g6load)
    try:
        model = o.get('MODEL')
        hf = hfsource.parse_url(model)
        if hf:
            if not o.get('MODEL_CACHE'):
                raise modelset.ModelError(f'--model {model} needs --model-cache <local dir>: each box fetches its part '
                                          'there')
            release = hfsource.Release(o.get('MODEL_CACHE'), *hf, refresh=refresh, log=log)
            release.fetch_meta()
            o.values['V2'], o.values['V2_SLICES'] = release.v2, release.slices
            if o.get('BASE_PACKS'):
                model = o.get('BASE_PACKS')
            else:
                o.values['V2_DENSE'] = release.dir
                ms = modelset.release_set(release.dir, int(o.get('TP')), o.get('MODEL_DIR') or release.tokenizer,
                                          str(release))
        elif modelset.is_release(model):
            experts = os.path.join(model, 'experts')
            o.values['V2'] = o.get('V2') or (experts if os.path.isdir(experts) else model)
            o.values['V2_SLICES'] = o.get('V2_SLICES') or os.path.join(model, 'index-slices')
            o.values['V2_DENSE'] = model
            ms = modelset.release_set(model, int(o.get('TP')), o.get('MODEL_DIR'))
        if ms is None:
            ms = modelset.resolve(model, o.get('MODEL_CACHE'), int(o.get('TP')),
                                  o.get('MODEL_DIR') or (release.tokenizer if release else None))
            if release:
                ms.source = f'{release}, base packs {ms.source}'
        ms.release = release
        search = [d for d in (model, os.path.join(ms.cache, 'sets', ms.id) if ms.cache else None,
                              release.dir if release else None) if d and os.path.isdir(d)]
        spec = o.get('VARIANT') or o.get('V2_WIDTHS')
        variant = modelset.resolve_variant(spec, o.get('V2'), search, int(o.get('TP')))
        if variant.path:
            o.values['V2_WIDTHS'] = variant.path
        steering_local.prepare(o, ms)
        ms.drafter = drafter_of(o, ms, refresh, log)
        return ms, variant, ''
    except modelset.ModelError as e:
        return ms, variant, f'model: {e}'


def drafter_of(o, ms, refresh, log):
    """The DFlash2 drafter for this boot: hfsource.Drafter for --drafter, else None. At TP=4 with no drafter at all
    (none in the model set either), DFLASH defaults to 0 with a notice, and an explicit --dflash 1 is refused."""
    if o.get('TP') != '4':
        return None
    if o.get('DRAFTER'):
        if not o.get('MODEL_CACHE'):
            raise modelset.ModelError('--drafter needs --model-cache <local dir>: each box converts its rank there')
        if o.get('DFLASH') == '0':
            return None
        d = hfsource.Drafter(o.get('MODEL_CACHE'), *hfsource.parse_url(o.get('DRAFTER')), refresh=refresh, log=log)
        o.values['DFLASH_DIR'] = d.out
        return d
    if ms.set['dflash'] or o.get('DFLASH') == '0' or o.get('DFLASH_DIR'):
        return None
    hint = f'--drafter {hfsource.DRAFTER_URL} (CC BY-NC-ND 4.0: each box fetches and converts it)'
    if o.get('DFLASH') == '1':
        raise modelset.ModelError(f'--dflash 1, but there is no DFlash2 drafter: add {hint}')
    log(f'no DFlash2 drafter: MTP only. For DFlash2, add {hint}')
    o.values['DFLASH'] = '0'
    return None


def running_engines(proc='/proc'):
    """[(pid, cwd)] of every `g6m4 --rank` process on this box, wherever it was started from."""
    out = []
    for pid in os.listdir(proc) if os.path.isdir(proc) else []:
        if not pid.isdigit():
            continue
        try:
            with open(os.path.join(proc, pid, 'cmdline'), 'rb') as f:
                argv = f.read().split(b'\0')
            cwd = os.readlink(os.path.join(proc, pid, 'cwd'))
        except OSError:
            continue
        if b'--rank' in argv and any(os.path.basename(a) in (b'g6m4', b'g6m4-int4') for a in argv[:2]):
            out.append((int(pid), cwd))
    return out


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            msg = json.loads(self.rfile.readline(1 << 24) or b'null')
            reply = self.server.sup.on_rpc(msg, self.client_address[0]) if isinstance(msg, dict) else None
        except (ValueError, OSError) as e:
            reply = {'t': 'error', 'error': str(e)}
        try:
            self.wfile.write((json.dumps(reply or {'t': 'error', 'error': 'bad request'}) + '\n').encode())
        except OSError:
            pass


class Supervisor:
    def __init__(self, o):
        self.o = o
        self.home = os.path.abspath(o.get('G6_HOME'))
        self.out = os.path.join(self.home, 'out')
        self.port = o.int('CONTROL_PORT')
        self.stop_ev = threading.Event()
        self.stop_why = ''
        self.stop_all = False         # on the head: stop every box as well
        self.stop_failed = False      # the stop is a failure (exit 1): a rank failed, here or elsewhere
        self.head = None              # handshake.Head on rank 0
        self.head_ready = threading.Event()
        self.engine = None
        self.helpers = {}             # the head's front end and shim: name -> Popen
        self.agent = self.api_agent = None
        self.rank, self.lan, self.members, self.hs = None, None, [], {}
        self.state = 'starting'
        self.servers = []
        self.locked = self.started = False
        self.ms, self.model_error, self.rails, self.fabric_error = None, '', ('', ''), ''
        self.offers = self.variant = self.model_id = None
        self.lingered = False
        self.check_n, self.check_lock = 0, threading.Lock()   # the engine's DIR/check.<n>, numbered per boot

    # ---- state ----------------------------------------------------------------------------------------------------
    def set_state(self, state, **meta):
        self.state = state
        if not self.locked:
            return
        tmp = os.path.join(self.out, 'gr8.state.tmp')
        with open(tmp, 'w') as f:
            json.dump({'state': state, 'rank': self.rank, 'pid': os.getpid(), 'time': time.time()}, f)
        os.replace(tmp, os.path.join(self.out, 'gr8.state'))
        if self.agent:
            self.agent.set_meta(**{'gr8.state': state[:1024], 'gr8.rank': self.rank}, **meta)

    # ---- the control port -----------------------------------------------------------------------------------------
    def serve(self, addrs):
        for a in dict.fromkeys(addrs):
            try:
                s = Server((a, self.port), Handler)
            except OSError as e:
                raise Stop(1, f'cannot listen on {a}:{self.port} ({e.strerror}): another gr8 supervisor, or the '
                              'address is not on this box')
            s.sup = self
            threading.Thread(target=s.serve_forever, daemon=True).start()
            self.servers.append(s)

    def allowed(self, peer):
        """The members' LAN and fabric addresses, and this box's own. Known once the members are."""
        known = set(self.members) | set(self.rails) | set(self.hs.get('ips', [])) | set(self.hs.get('ips2', []))
        if self.head:
            known |= {a for h in self.head.have.values() for a in h['rails']}
        return peer in known or fabric.is_local(peer)

    def on_rpc(self, m, peer):
        t = m.get('t')
        if t == 'ping':
            return {'t': 'pong', 'rank': self.rank}
        if not self.allowed(peer):
            log(f'ignored {t!r} from {peer}: not a member')
            return {'t': 'error', 'error': f'{peer} is not a member of this cluster'}
        if t in ('hello', 'verdict'):
            if self.rank not in (0, None):
                return {'t': 'refused', 'field': 'rank', 'values': {}, 'exit': 4,
                        'detail': f'{self.lan} is rank {self.rank}, not the head'}
            if not self.head_ready.is_set():
                return {'t': 'retry', 'detail': 'the head has not finished its checks' if self.state == 'preparing'
                        else f'the head is {self.state}'}
            if t == 'hello' and self.stop_ev.is_set() and self.head.verdict and self.head.verdict['t'] == 'ok':
                # a booted cluster going down: a box that restarted already (a container) waits for the next boot
                return {'t': 'retry', 'detail': 'the head is stopping its boot'}
            return self.head.rpc(m, peer)
        if t == 'status':
            return self.status()
        if t == 'down':
            if m.get('epoch') and (self.hs or {}).get('epoch') != m['epoch']:
                return {'t': 'ok', 'detail': 'not this boot'}   # the head ending a boot this supervisor is not in
            self.request_stop(m.get('why') or f'down from {peer}', failed=bool(m.get('failed')))
            return {'t': 'ok'}
        if t == 'abort':
            self.request_stop(f'rank {m.get("rank")} ({peer}) failed: {m.get("why")}', all_boxes=True)
            return {'t': 'ok'}
        if t == 'serving':
            if self.state == 'loading':
                self.set_state('serving')
                log(f'the head serves {m.get("url")}')
            return {'t': 'ok'}
        if t == 'logs':
            return self.read_log(m.get('which', 'engine'), int(m.get('offset', -8192)))
        if t == 'check':
            return {'t': 'check', **self.local_check()}
        if t == 'check_all':
            return self.check_all()
        return {'t': 'error', 'error': f'unknown request {t!r}'}

    def request_stop(self, why, all_boxes=False, failed=False):
        if not self.stop_ev.is_set():
            self.stop_why = why
            self.stop_all = all_boxes
            self.stop_failed = failed or all_boxes
            self.stop_ev.set()

    def status(self):
        pid = read_pid(os.path.join(self.out, 'g6m4.pid'))
        s = {'t': 'status', 'rank': self.rank, 'lan': self.lan, 'state': self.state, 'engine_pid': pid,
             'engine': bool(recorded(os.path.join(self.out, 'g6m4.pid'))), 'mem_gib': mem_available_gib(),
             'home': self.home,
             'variant': self.variant.name if self.variant else None, 'model_id': self.model_id}
        if s['engine'] is False:
            err = os.path.join(self.out, f'g6m4.r{self.rank}.err')
            s['last_line'] = tail(err, 1).strip()
        if self.rank == 0:
            for p in ('shim', 'front'):
                s[p] = bool(recorded(os.path.join(self.out, f'{p}.pid')))
        return s

    def local_check(self, timeout=60):
        """This box's self-checks: the engine's weights against the sums it sealed when it was ready to serve
        (engine/runtime/weight_check.cuh), checked on a stream of their own while it serves."""
        r = {'rank': self.rank, 'lan': self.lan, 'state': self.state, 'mem_gib': mem_available_gib(),
             'compaction_proactiveness': read_int('/proc/sys/vm/compaction_proactiveness')}
        if self.state != 'serving' or not (self.engine and self.engine.poll() is None):
            return {**r, 'ok': False, 'error': f'the engine is not serving ({self.state})'}
        with self.check_lock:   # the engine answers check.<n> in order
            n, self.check_n = self.check_n, self.check_n + 1
            req = os.path.join(self.home, 's2', f'check.{n}')
            open(req, 'w').close()
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                try:
                    with open(req + '.json') as f:
                        w = json.load(f)
                    return {**r, 'ok': bool(w.get('ok')), 'weights': w}
                except FileNotFoundError:
                    time.sleep(0.2)
                except ValueError as e:
                    return {**r, 'ok': False, 'error': f'check.{n}.json: {e}'}
        return {**r, 'ok': False, 'error': f'the engine did not answer check.{n} within {timeout} s'}

    def check_all(self):
        """The head asks every rank for its self-checks at once (the front end's /health?deep)."""
        if self.rank != 0:
            return {'t': 'error', 'error': f'rank {self.rank} is not the head'}
        ranks = [None] * len(self.members)

        def one(r):
            if r == self.rank:
                ranks[r] = self.local_check()
            else:
                a = self.tell(r, {'t': 'check'}, timeout=75)
                ranks[r] = a if a and a.get('t') == 'check' else {'rank': r, 'ok': False, 'error': f'no answer: {a}'}
        ts = [threading.Thread(target=one, args=(r,)) for r in range(len(self.members))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return {'t': 'check', 'ok': all(x.get('ok') for x in ranks), 'ranks': ranks, 'time': time.time()}

    def read_log(self, which, offset):
        name = {'shim': 'shim.log', 'front': 'front.log', 'supervisor': 'supervisor.log'}.get(which, f'g6m4.r{self.rank}.err')
        p = os.path.join(self.out, name)
        try:
            size = os.path.getsize(p)
            start = size + offset if offset < 0 else min(offset, size)
            with open(p, 'rb') as f:
                f.seek(max(start, 0))
                data = f.read(1 << 20)
            return {'t': 'log', 'data': data.decode(errors='replace'), 'offset': max(start, 0) + len(data)}
        except OSError as e:
            return {'t': 'log', 'data': f'{p}: {e.strerror}\n', 'offset': 0}

    # ---- the boot -------------------------------------------------------------------------------------------------
    def run(self):
        o = self.o
        os.makedirs(self.out, exist_ok=True)
        self.lock = open(os.path.join(self.out, 'supervisor.lock'), 'w')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Stop(1, f'a supervisor already runs here (pid {read_pid(os.path.join(self.out, "supervisor.pid"))}): '
                          './gr8 glm53 down first')
        self.locked = True
        with open(os.path.join(self.out, 'supervisor.pid'), 'w') as f:
            f.write(f'{os.getpid()}\n')
        for p in ('handshake.json', 'gr8.state'):
            try:
                os.unlink(os.path.join(self.out, p))
            except FileNotFoundError:
                pass
        self.set_state('starting')
        for w in o.warnings:
            log(f'warning: {w}')
        log(f'gr8 {version(self.home)}, options ({o.file or "no config file"}; command line > file > environment > default):')
        log(opt.summary(o))
        tp = int(o.get('TP'))
        # a box that cannot resolve its model still joins, so every box hears why the boot stops
        self.ms, self.variant, self.model_error = resolve_model(o)
        if self.model_error:
            self.ms = None
            log(self.model_error)
        else:
            self.model_id = modelset.model_id(self.ms, self.variant)
            log(f'model set {self.ms.id[:16]} from {self.ms.source}, variant {self.variant}, id {self.model_id[:16]}')
            if self.variant.path:
                log(f'variant {self.variant.name}: about {modelset.expert_gb(self.variant.path, tp):.1f} GB of experts '
                    f'on the busiest rank at TP={tp}')
        self.tmpl = opt.engine_plan(o)
        self.build = engine_build(self.home, engine_bin(o))
        self.offers = modelset.offers(self.ms, self.tmpl) if self.ms else []
        log(f'this box can serve rank {mentat.format_offers(self.offers) or "none"} of the model set')
        self.rendezvous(tp)
        c, text, digest = opt.canonical(o, self.tmpl, self.members if opt.mode(o) == 'peers' else None, self.variant)
        self.opts, self.digest = c, digest
        log(f'rank {self.rank} of {tp} ({self.lan}, rails {self.rails[0]} {self.rails[1]}), cluster options '
            f'{digest[:16]}, engine {self.build[:16]}')
        if self.agent:
            self.agent.set_meta(**{'gr8.set': self.model_id or 'none', 'gr8.variant': str(self.variant or ''),
                                   'gr8.opts': digest,
                                   'gr8.ips': ','.join(self.rails)})
        self.serve([a for a in (self.lan, *self.rails) if a])
        self.set_state('preparing')
        fails = self.checks()
        for f in fails:
            log(f'check failed: {f}')
        hello = {'t': 'hello', 'proto': handshake.PROTO, 'rank': self.rank, 'lan': self.lan, 'members': self.members,
                 'rails': list(self.rails), 'home': self.home, 'opts': c, 'opts_digest': digest,
                 'set_id': self.model_id or 'none',
                 'set_source': f'{self.ms.source}, variant {self.variant}' if self.ms else self.model_error,
                 'engine_build': self.build, 'fails': fails, 'version': version(self.home),
                 'offers': mentat.format_offers(self.offers)}
        self.set_state('handshake')
        verdict = self.handshake_head(hello) if self.rank == 0 else self.handshake_worker(hello)
        if verdict.get('t') == 'ok' and fails:   # never boot past a failed check, whatever the verdict
            verdict = {'t': 'refused', 'field': 'checks', 'values': {}, 'detail': '; '.join(fails), 'exit': 4}
        if verdict.get('t') != 'ok':
            log(handshake.describe(verdict))
            e = Stop(verdict.get('exit', 4), f'handshake refused: {verdict["field"]}')
            e.logged = True
            raise e
        self.hs = verdict
        with open(os.path.join(self.out, 'handshake.json'), 'w') as f:
            json.dump({**verdict, 'rank': self.rank, 'control_port': self.port}, f, indent=1)
        self.boot()

    def rendezvous(self, tp):
        o = self.o
        if opt.mode(o) == 'peers':
            self.members = opt.resolve_peers(o)
            try:
                self.rank = fabric.own_index(self.members, o.get('NODE_IP'))
            except RuntimeError as e:
                raise Stop(2, str(e))
            self.lan = self.members[self.rank]
            log(f'peers: {",".join(self.members)}; this box is rank {self.rank}')
        else:
            self.daemon = mentat.Daemon(o.get('MENTAT'))
            try:
                local = self.daemon.local_status()
            except OSError as e:
                raise Stop(1, f'no mentat daemon at {self.daemon.http} ({e}). Start mentatd on this box, or pass --peers')
            node_ip, _, _ = mentat.node_view(local)
            group = o.get('GROUP')
            self.agent = mentat.Agent(self.daemon, group, f'{group}@gr8@{node_ip}', node_ip, log=log)
            self.agent.set_meta(**{'gr8.version': version(self.home), 'gr8.tp': tp, 'gr8.shape': o.get('SHAPE'),
                                   'gr8.state': 'joining', 'gr8.offers': mentat.format_offers(self.offers)})
            self.agent.start()
            self.set_state('joining')
            try:
                ranks, _ = mentat.elect(self.daemon, group, tp, o.get('HEAD'), o.int('JOIN_TIMEOUT'), log=log)
            except mentat.NoAssignment as e:
                raise Stop(4, str(e))
            except RuntimeError as e:
                raise Stop(1, str(e))
            self.members = [lan for lan, _, _ in ranks]
            if node_ip not in self.members:
                raise Stop(1, f'this box ({node_ip}) is not among the members {",".join(self.members)}')
            self.rank, self.lan = self.members.index(node_ip), node_ip
            log(f'mentat group {group}: {",".join(self.members)}; this box is rank {self.rank}')
            if not o.get('IPS') and o.get('SHAPE') != 'ring':
                self.rails = ranks[self.rank][2]
                for p in mentat.probe_problems(local, ranks, node_ip, self.rails):
                    log(f'warning: {p}')
        if o.get('SHAPE') == 'ring':    # no switch: the engine bootstraps over the LAN and finds its devices by name
            self.rails = (self.lan, self.lan)
        elif o.get('IPS'):
            self.rails = (opt.split_list(o.get('IPS'))[self.rank], opt.split_list(o.get('IPS2'))[self.rank])
        elif opt.mode(o) == 'peers':
            try:
                self.rails = fabric.rails()
            except RuntimeError as e:
                self.fabric_error = f'fabric: {e}'
                log(self.fabric_error)

    def checks(self, copy=True):
        """What would stop this box's boot, as text. copy: make this rank's model files ready first. The rank's files
        are checked only once the rank is known, and the front end only on rank 0."""
        o, fails, plan = self.o, [e for e in (self.model_error, self.fabric_error) if e], self.tmpl
        held = self.offers is not None and self.rank is not None and self.rank not in self.offers
        if held:
            fails.append(f'this box holds rank {mentat.format_offers(self.offers) or "none"} of the model set, but it is '
                         f'rank {self.rank} ({self.lan}) in the member list'
                         + (f': list it at position {self.offers[0] + 1} of --peers' if self.offers else ''))
        try:
            if not self.ms or held:
                pass
            elif copy:
                modelset.prepare(self.ms, self.rank, plan, log=log)
                rel = self.ms.release
                if rel and rel.check(int(o.get('TP')), self.rank, self.variant):
                    rel.fetch_rank(int(o.get('TP')), self.rank, self.variant)
                if self.ms.drafter and self.ms.drafter.check(self.rank):
                    self.ms.drafter.fetch_rank(self.rank)
            if self.ms and self.rank is not None and not held:
                fails += modelset.validate_rank(self.ms, self.rank, plan)
                if self.ms.release:
                    fails += self.ms.release.check(int(o.get('TP')), self.rank, self.variant)
                if self.ms.drafter:
                    fails += self.ms.drafter.check(self.rank)
            if self.ms:
                fails += modelset.check_tokenizer(self.ms)
        except (modelset.ModelError, OSError, RuntimeError) as e:
            fails.append(f'model: {e}')
        p = recorded(os.path.join(self.out, 'g6m4.pid'))
        if p:
            fails.append(f'an engine is already running (pid {p}): ./gr8 glm53 down first')
        for pid, cwd in running_engines():
            if pid != p:
                fails.append(f'a g6m4 --rank process is running (pid {pid}, cwd {cwd}): stop it first')
        if o.get('CHECK') == '0':
            return fails
        if self.build == 'missing':
            name = engine_bin(o)
            fails.append(f'no {self.home}/bin/{name}: ' + ('KV_FORMAT=int4 ' if name != 'g6m4' else '') + './gr8 glm53 build')
        if not os.path.isdir('/dev/infiniband'):
            fails.append('no /dev/infiniband')
        if self.rank == 0:
            if not os.access(self.front_bin(), os.X_OK):
                fails.append(f'no {self.front_bin()}: ./gr8 glm53 build (or scripts/build_front.sh bin/)')
            if not os.access(self.py(), os.X_OK):
                fails.append(f'no {self.py()} for the shim: python3 -m venv {self.home}/venv && '
                             f'{self.home}/venv/bin/pip install -r frontend/requirements.txt')
        m = mem_available_gib()
        log(f'MemAvailable {m} GiB' if m is not None else 'MemAvailable unknown')
        if m is not None and m < 8:
            need = (f'about {modelset.expert_gb(self.variant.path, int(o.get("TP"))):.0f} GB of experts'
                    if self.variant and self.variant.path else 'most of the box')
            log(f'warning: MemAvailable is {m} GiB before boot: something else holds memory (a rank at '
                f'TP={o.get("TP")} loads {need}, plus its dense weights and KV pool)')
        c = read_int('/proc/sys/vm/compaction_proactiveness')
        if c:
            log(f'warning: vm.compaction_proactiveness is {c}: background compaction stalls the collectives of a full '
                'box; as root, sysctl -w vm.compaction_proactiveness=0 (README, What you need)')
        return fails

    def handshake_head(self, hello):
        self.head = handshake.Head(hello)
        self.head_ready.set()
        tp = len(self.members)
        log(f'head: waiting for {tp - 1} hellos on {self.lan}:{self.port}')
        if self.head.wait_all(opt_timeout(self.o), self.stop_ev):
            bad = self.rail_check()
            if bad:
                self.head.decide(handshake.refusal('fabric', self.head.have, lambda x: ','.join(x['rails']), bad))
            else:
                have = self.head.have
                self.head.decide({'t': 'ok', 'epoch': int(time.time()), 'members': self.members,
                                  'ips': [have[r]['rails'][0] for r in range(tp)],
                                  'ips2': [have[r]['rails'][1] for r in range(tp)],
                                  'homes': {str(r): have[r]['home'] for r in range(tp)}, 'set_id': self.model_id,
                                  'variant': self.variant.name,
                                  'opts_digest': self.digest, 's2port': self.tmpl['knobs'].get('S2PORT'),
                                  'fabric_port': opt.FABRIC_PORT, 'tp': tp, 'port': self.o.get('PORT')})
        elif self.stop_ev.is_set():
            self.head.decide({'t': 'refused', 'field': 'stopped', 'values': {}, 'detail': self.stop_why, 'exit': 1})
        v = self.head.verdict
        if v['t'] != 'ok':
            self.linger()
        return v

    def linger(self):
        """Answer late hellos and polls with the head's refusal for a while before leaving."""
        if not self.lingered:
            self.lingered = True
            self.stop_ev.wait(float(os.environ.get('GR8_REFUSAL_LINGER', '15')))

    def rail_check(self):
        """From each of the head's rails to the same rail of every worker: the control port must answer."""
        bad = []
        for r, h in sorted(self.head.have.items()):
            if r == 0:
                continue
            for k in range(2):
                try:
                    handshake.call(h['rails'][k], self.port, {'t': 'ping'}, timeout=3, bind=self.rails[k])
                except OSError as e:
                    bad.append(f'rank {r} ({h["lan"]}) does not answer on rail {k} at {h["rails"][k]} from '
                               f'{self.rails[k]} ({e})')
        return '; '.join(bad)

    def handshake_worker(self, hello):
        log(f'sending the hello to the head {self.members[0]}:{self.port}')
        return handshake.send_hello(self.members[0], self.port, hello, self.lan, opt_timeout(self.o), log, self.stop_ev)

    def engine_env(self, plan):
        """The plan's variables over an allowlist of this process's environment."""
        env = {k: v for k, v in os.environ.items() if k in ENGINE_ENV_KEEP or k.startswith(ENGINE_ENV_PREFIXES)}
        env.update(kv.split('=', 1) for kv in plan['env'] if '=' in kv)
        lib = os.path.join(self.home, 'bin', 'lib')
        if os.path.isdir(lib):        # cuBLAS from build.sh BUNDLE_LIBS=1, at any TP
            env['LD_LIBRARY_PATH'] = lib
        return env

    # ---- after the handshake --------------------------------------------------------------------------------------
    def front_bin(self):
        return self.o.get('FRONT_BIN') or os.path.join(REPO, 'bin', 'g6front-rs')

    def py(self):
        return self.o.get('PY') or os.path.join(self.home, 'venv', 'bin', 'python')

    def spawn(self, name, argv, cwd, env=None):
        """Start a helper detached from this process, with its pid in out/<name>.pid."""
        with open(os.path.join(self.out, f'{name}.log'), 'ab') as lf:
            p = subprocess.Popen(argv, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=env, start_new_session=True)
        with open(os.path.join(self.out, f'{name}.pid'), 'w') as f:
            f.write(f'{p.pid}\n')
        self.helpers[name] = p
        return p

    def kv_store(self):
        """KV_STORE for this box's engine: each remote:auto becomes the KV node mentat announces, by this box's rails."""
        stores = opt.kv_stores(self.o)
        if not any(t['host'] == 'auto' for t in stores):
            return None
        g = self.o.get('GROUP')
        try:
            nodes = mentat.kv_nodes(self.daemon.group_status(g), g)
        except (OSError, RuntimeError, ValueError) as e:
            raise Stop(1, f'--kv-store remote:auto: mentat: {e}')
        target = mentat.kv_target(nodes, self.rails)
        if not target:
            raise Stop(1, f'--kv-store remote:auto: no KV node in group {g} (./gr8 kv up on a box with spare memory)')
        log(f'KV store: the KV node at {target}')
        return ';'.join(t['text'].replace('remote:auto', f'remote:{target}', 1) if t['host'] == 'auto' else t['text']
                        for t in stores)

    def boot(self):
        o, hs, r = self.o, self.hs, self.rank
        plan = opt.engine_plan(o, rank=r, rank_dir=self.ms.rank_dir(r), aux_dir=self.ms.rank_dir(r), home=self.home,
                               epoch=hs['epoch'], set_id=self.model_id, kv_store=self.kv_store())
        work = os.path.join(self.home, 's2')
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        self.check_n = 0
        env = self.engine_env(plan)
        argv = ['./bin/' + engine_bin(o), '--rank', str(r), '--ips', ','.join(hs['ips']), '--ips2', ','.join(hs['ips2']),
                '--proxy-cpu', o.get('PROXY_CPU'), *plan['args']]
        if o.get('SHAPE') == 'ring':
            argv[1:1] = ['--ring', *(['--ring-devs', o.get('RING_DEVS')] if o.get('RING_DEVS') else [])]
        log(f'engine: {" ".join(plan["env"])} {" ".join(argv)}')
        rotate_logs(self.out, [f'g6m4.r{r}.txt', f'g6m4.r{r}.err'])
        with open(os.path.join(self.out, f'g6m4.r{r}.txt'), 'wb') as so, \
                open(os.path.join(self.out, f'g6m4.r{r}.err'), 'wb') as se:
            self.engine = subprocess.Popen(argv, cwd=self.home, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
        with open(os.path.join(self.out, 'g6m4.pid'), 'w') as f:
            f.write(f'{self.engine.pid}\n')
        self.started = True
        self.set_state('loading')
        if r == 0:
            self.start_front(plan)
        ready = os.path.join(work, 'ready.s2')
        tmo = float(plan['knobs'].get('READY_TMO') or 1800)
        t0, ask_at = time.monotonic(), tmo
        log(f'engine started (epoch {hs["epoch"]}); waiting for it to load (up to {tmo:.0f}s)')
        while True:
            self.watch()
            if r == 0:
                if os.path.exists(ready):
                    break
                if time.monotonic() - t0 > tmo:
                    raise Stop(1, f'the engines did not load within {tmo:.0f}s')
            elif self.state == 'serving':
                break
            elif time.monotonic() - t0 > ask_at and self.head_serves():
                break
            elif time.monotonic() - t0 > ask_at:
                ask_at += 30
            elif time.monotonic() - t0 > tmo + 600:
                raise Stop(1, 'the head did not report serving')
            time.sleep(1)
        if r == 0:
            log(f'engines ready after {time.monotonic() - t0:.0f}s')
            self.start_shim(work)
        self.run_forever()

    def start_front(self, plan):
        o = self.o
        sock = os.path.join(self.home, 'run', 'engine.sock')
        os.makedirs(os.path.dirname(sock), exist_ok=True)
        try:
            os.unlink(sock)
        except FileNotFoundError:
            pass
        front = [self.front_bin()]
        if o.get('DEFAULT_MAX_TOKENS') != '0':
            front += ['--default-max-tokens', o.get('DEFAULT_MAX_TOKENS')]
        else:   # a request within the limit, guard rows included, shares decode steps: keep a page in hand for the guard
            front += ['--share-tokens', str(int(plan['knobs']['SHARE_LIMIT']) - 64)]
        if o.get('DEFAULT_REASONING_EFFORT'):
            front += ['--default-reasoning-effort', o.get('DEFAULT_REASONING_EFFORT')]
        if o.get('DEFAULT_THINKING_BUDGET') != '0':
            front += ['--default-thinking-budget', o.get('DEFAULT_THINKING_BUDGET')]
        self.spawn('front', self.pin() + front + ['--model-dir', self.ms.tokenizer, '--engine-sock', sock,
                                                  '--port', o.get('PORT'), '--supervisor', f'{self.lan}:{self.port}'],
                   os.path.join(REPO, 'frontend'))

    def pin(self):
        """At TP=6 and TP=3 the shim and front end stay off the proxy CPUs (PROXY_CPU-7..PROXY_CPU), as the launchers do."""
        cpu = int(self.o.get('PROXY_CPU'))
        if self.o.get('TP') in ('6', '3') and cpu >= 8 and shutil.which('taskset'):
            return ['taskset', '-c', f'0-{cpu - 8}']
        return []

    def start_shim(self, work):
        """The shim drives the head's engine directory. The head engine passes requests to the other ranks itself, over
        its control links."""
        sock = os.path.join(self.home, 'run', 'engine.sock')
        argv = [self.py(), os.path.join(REPO, 'frontend', 'g6s2shim.py'), '--dir', work, '--sock', sock]
        if self.o.get('TP') == '6':
            argv += ['--stall-timeout', '2400']
        self.spawn('shim', self.pin() + argv, REPO, env={**os.environ, 'PYTHONPATH': os.path.join(REPO, 'frontend')})
        for _ in range(150):
            if os.path.exists(sock):
                break
            self.watch()
            time.sleep(0.2)
        else:
            raise Stop(1, f'the shim did not open {sock}:\n{tail(os.path.join(self.out, "shim.log"), 20)}')
        port = self.o.get('PORT')
        code = 0
        for _ in range(300):
            try:
                code = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).status
            except OSError as e:
                code = getattr(e, 'code', 0)
            if code == 200:
                break
            self.watch()
            time.sleep(0.2)
        url = f'http://{self.lan}:{port}/v1'
        if code != 200:
            raise Stop(1, f'the front end /health answers {code or "nothing"}:\n{tail(os.path.join(self.out, "front.log"), 20)}')
        log(f'serving: {url}')
        self.set_state('serving')
        for r in range(1, len(self.members)):
            for _ in range(10):
                if self.tell(r, {'t': 'serving', 'url': url}):
                    break
                time.sleep(1)
        if self.agent:
            g = self.o.get('GROUP')
            self.api_agent = mentat.Agent(self.daemon, g, f'{g}@gr8-api@{self.lan}', self.lan, gpus=0, log=log,
                                          services={'openai': {'host': '', 'port': int(port), 'path': '/v1',
                                                               'provider': 'g6'}})
            self.api_agent.start()
            log(f'announced the API to mentat as {g}@gr8-api@{self.lan}')

    def tell(self, r, msg, timeout=5):
        try:
            return handshake.call(self.members[r], self.port, msg, timeout=timeout, bind=self.lan)
        except OSError as e:
            log(f'rank {r} ({self.members[r]}): {e}')
            return None

    def tell_head(self, msg):
        return self.tell(0, msg)

    def head_serves(self):
        """A worker that missed the head's serving message asks for it."""
        s = self.tell_head({'t': 'status'})
        if s and s.get('state') == 'serving':
            self.set_state('serving')
            return True
        return False

    def watch(self):
        """Raise Stop when this rank's engine has exited or someone asked to stop."""
        if self.stop_ev.is_set():
            raise Stop(1 if self.stop_failed else 0, self.stop_why)
        if self.engine and self.engine.poll() is not None:
            name = f'g6m4.r{self.rank}'
            reason = exit_reason([os.path.join(self.out, f'{name}.err'), os.path.join(self.out, f'{name}.txt')])
            why = f'engine exited ({exit_status(self.engine.returncode)})' + (f': {reason}' if reason else '') + \
                  f'; last lines of {name}.err:\n' + tail(os.path.join(self.out, f'{name}.err'), 20)
            # the weights on the GPU differ from the bytes read (engine/runtime/weight_check.cuh): exit 3, as the rank
            # health refusal does, so a restart policy can tell it from a crash
            raise Stop(3 if reason.startswith('g6: weight check') else 1, why)
        # the head's front end and shim: without either the API is gone, so the cluster stops as for an engine
        # (polling also reaps them; a zombie still answered kill(pid, 0), and status called it running)
        for name, p in self.helpers.items():
            if p.poll() is not None:
                label = {'front': 'front end'}.get(name, name)
                raise Stop(1, f'{label} exited ({p.returncode}); last lines of {name}.log:\n' +
                           tail(os.path.join(self.out, f'{name}.log'), 20))

    def run_forever(self):
        while True:
            self.watch()
            time.sleep(2)

    # ---- stopping -------------------------------------------------------------------------------------------------
    def shutdown(self, code, why):
        """Stop this box (and, on the head after a failure, every box). A supervisor that did not get the lock touches
        nothing, and one that started no engine leaves the engine processes alone."""
        if not self.locked:
            return
        if code:
            self.set_state(f'failed: {why.splitlines()[0][:200]}' if why else 'failed')
        else:
            self.set_state('stopping')
        first = why.splitlines()[0] if why else ''
        if self.rank == 0 and not self.hs and code:   # workers still waiting for a verdict hear why
            v = {'t': 'refused', 'field': 'head', 'values': {}, 'detail': f'the head stopped: {first}', 'exit': code}
            if self.head is None:
                self.head = handshake.Head.refusing(v)
            else:
                self.head.decide(v)
            self.head_ready.set()
            self.stop_ev.clear()
            self.linger()
        if self.hs and self.rank == 0 and (code or self.stop_all):
            # a failure fails every box (exit 1), so a box that restarts on failure boots again with the others
            for r in range(1, len(self.members)):
                self.tell(r, {'t': 'down', 'why': f'the head stopped the cluster: {first}', 'failed': bool(code),
                              'epoch': self.hs['epoch']})
        elif self.hs and self.rank and code and not self.stop_failed:
            self.tell_head({'t': 'abort', 'rank': self.rank, 'why': first})
        if self.started:
            stop_local(self.home, log)
        if self.agent:
            self.agent.stop()
        if self.api_agent:
            self.api_agent.stop()
        if not code:
            self.set_state('stopped')
        for s in self.servers:
            s.shutdown()


def engine_bin(o):
    """The engine binary under bin/ for KV_FORMAT: g6m4 (the fp8 KV row) or g6m4-int4 (docs/KV4.md)."""
    return 'g6m4-int4' if o.get('KV_FORMAT') == 'int4' else 'g6m4'


def engine_build(home, name='g6m4'):
    """sha256 over bin/NAME (the engine for this KV format) and every file in bin/lib (the bundled cuBLAS), or
    'missing'."""
    b = os.path.join(home, 'bin')
    if not os.path.isfile(os.path.join(b, name)):
        return 'missing'
    lib = os.path.join(b, 'lib')
    names = [name] + sorted(os.path.join('lib', n) for n in os.listdir(lib)) if os.path.isdir(lib) else [name]
    h = hashlib.sha256()
    for n in names:
        if os.path.isfile(os.path.join(b, n)):
            h.update(f'{n} {opt.file_sha256(os.path.join(b, n))}\n'.encode())
    return h.hexdigest()


def opt_timeout(o):
    return float(o.int('JOIN_TIMEOUT'))


def tail(path, n, nbytes=65536):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return b'\n'.join(f.read().splitlines()[-n:]).decode(errors='replace')
    except OSError:
        return ''


# Lines that say why an engine stopped: die() and the error words, CUDA and fabric checks, a Rust panic, a C++ abort.
EXIT_REASON_PREFIXES = ('g6: ', 'g6m4: ', 'CUDA ', '[fabric ', 'terminate called')


def exit_reason(paths):
    """The last line in the first of paths that names why the engine stopped, or ''. Threads that keep writing after a
    fatal line can push it out of the tail."""
    for p in paths:
        lines = tail(p, 100000, nbytes=4 << 20).splitlines()
        for line in reversed(lines):
            s = line.strip()
            if s.startswith(EXIT_REASON_PREFIXES) or ("thread '" in s and 'panicked' in s):
                return s
    return ''


def exit_status(code):
    """A return code as text: a negative one is the signal that killed the process."""
    if code is None or code >= 0:
        return str(code)
    try:
        name = signal.Signals(-code).name
    except ValueError:
        name = f'signal {-code}'
    return f'{code}, {name}' + (': the OOM killer? see dmesg' if name == 'SIGKILL' else '')


KEEP_BOOTS = 3


def rotate_logs(out, names, keep=KEEP_BOOTS):
    """Move the last boot's files among names to out/prev1, shifting older boots to prev2..prev<keep>, so a restart
    keeps the logs of the boots before it."""
    if not any(os.path.exists(os.path.join(out, n)) for n in names):
        return
    shutil.rmtree(os.path.join(out, f'prev{keep}'), ignore_errors=True)
    for i in range(keep - 1, 0, -1):
        d = os.path.join(out, f'prev{i}')
        if os.path.isdir(d):
            os.replace(d, os.path.join(out, f'prev{i + 1}'))
    prev = os.path.join(out, 'prev1')
    os.makedirs(prev)
    for n in names:
        if os.path.exists(os.path.join(out, n)):
            os.replace(os.path.join(out, n), os.path.join(prev, n))


def signal_pid(pid, sig, name, log=log):
    """Signal one of this box's processes; one that already exited, or another account's, is reported, not raised."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        log(f'{name} (pid {pid}) belongs to another account: not stopped')


def stop_local(home, log=log):
    """What scripts/stop.sh does, on this box: shim and front end, then quit files, then the engine."""
    out = os.path.join(home, 'out')
    for p in ('front', 'shim'):
        pid = recorded(os.path.join(out, f'{p}.pid'))
        if pid:
            signal_pid(pid, signal.SIGTERM, p, log)
    pid = recorded(os.path.join(out, 'g6m4.pid'))
    if not pid:
        return False
    if os.path.isdir(os.path.join(home, 's2')):
        open(os.path.join(home, 's2', 'quit'), 'w').close()
    for _ in range(100):
        if not alive(pid):
            break
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        time.sleep(0.1)
    if alive(pid):
        signal_pid(pid, signal.SIGTERM, 'engine', log)
        for _ in range(50):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            if not alive(pid):
                break
            time.sleep(0.1)
    log(f'engine (pid {pid}) stopped')
    return True


def main(o):
    sup = Supervisor(o)
    signal.signal(signal.SIGTERM, lambda *_: sup.request_stop('SIGTERM'))
    # Ctrl-C on `up --foreground`: stop as for SIGTERM. Uncaught, KeyboardInterrupt skipped shutdown() and left
    # the engine (in its own session) running.
    signal.signal(signal.SIGINT, lambda *_: sup.request_stop('SIGINT'))
    try:
        sup.run()
        code = 0
    except Stop as e:
        code, why = e.code, str(e)
        if why and not getattr(e, 'logged', False):
            log(why)
        sup.shutdown(code, why)
    except opt.OptionError as e:
        log(f'options: {e}')
        sup.shutdown(2, str(e))
        code = 2
    except Exception as e:   # anything unexpected still stops the engine and says why
        import traceback
        log(traceback.format_exc())
        sup.shutdown(1, f'{type(e).__name__}: {e}')
        code = 1
    sys.exit(code)
