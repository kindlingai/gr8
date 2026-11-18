"""The per-box supervisor: `./gr8 glm53 up` runs one on every box, with the same options.

In order: resolve the options and the model set; find the other boxes (the peer list, or the mentat group) and this
box's rank; start the control server on the LAN port; run the local checks and make this rank's model files ready;
handshake with the head; start this box's g6m4. The head also starts the front end, waits for the engine, starts the
shim and announces the API. Then the supervisor stays, answering status, logs and down on the control port, and stops
the cluster when a rank's engine exits.

`./gr8 <model> scale` (the head's `scale` request) changes a serving cluster's width by an orderly restart: every box
of the new member list makes its files ready while the old one serves, the front end holds new requests and the running
ones finish, then the old engines quit (their KV store kept for the new ranks, GR8_KV_REWIDTH) and the new member list
boots. A failure after the switch boots the old member list again (a rollback).
"""
import fcntl
import glob
import hashlib
import json
import os
import re
import secrets
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
import kvnode
import mlsource
import mentat
import models
import modelset
import options as opt
import steering_local
import steering_hf

REPO = opt.REPO
# What the engine gets from the supervisor's environment, besides the plan's own variables.
ENGINE_ENV_KEEP = ('PATH', 'HOME', 'USER', 'LANG', 'LD_LIBRARY_PATH', 'DISPRAM_DISABLE', 'DISPRAM_SOCKET',
                   'DISPRAM_KEY_FILE')
ENGINE_ENV_PREFIXES = ('CUDA_', 'NVIDIA_')


class Stop(Exception):
    """The boot ends here: code is the exit status."""

    def __init__(self, code, why):
        super().__init__(why)
        self.code = code


class DispramFallback(Exception):
    """Retry the cluster once without probed display memory."""


class Rescale(Exception):
    """scale_go: stop this box's engine and boot the scale's member list."""


class Rollback(Exception):
    """scale_abort after the switch: boot the old member list again."""


def dispram_key(env):
    """The key a dispram request carries: DISPRAM_KEY_FILE's text, trimmed (engine/dispram_client.h's rule), or
    Kindling OS dispramd's own key when it is unset. None when the file cannot give a valid key."""
    path = env.get('DISPRAM_KEY_FILE') or ''
    if not path:
        return 'kindlingai_1'
    try:
        with open(path, 'rb') as f:
            raw = f.read(4097)
    except OSError:
        return None
    key = raw.strip(b' \t\r\n')
    if len(raw) > 4096 or not 1 <= len(key) <= 127 or any(c < 0x21 or c > 0x7e or c in b'"\\' for c in key):
        return None
    return key.decode('ascii')


def probe_dispram(o, environ=None, timeout=1.0):
    env = os.environ if environ is None else environ
    if o.get('DISPRAM') != '1' or str(env.get('DISPRAM_DISABLE', '')).lower() in ('1', 'true', 'yes', 'on'):
        return 0
    kind = getattr(socket, 'SOCK_SEQPACKET', None)
    if kind is None:
        return 0
    key = dispram_key(env)
    if key is None:
        return 0
    conn = None
    try:
        conn = socket.socket(socket.AF_UNIX, kind)
        conn.settimeout(timeout)
        conn.connect(env.get('DISPRAM_SOCKET') or '/run/dispram/dispram.sock')
        conn.sendall(json.dumps({'key': key, 'op': 'info'}, separators=(',', ':')).encode())
        d = json.loads(conn.recv(4096))
        if not isinstance(d, dict):
            return 0
        free = d.get('free')
        if type(free) is not int or free <= 0 or 'error' in d:
            return 0
        return free // (2 << 20) * (2 << 20)
    except (OSError, ValueError, TypeError):
        return 0
    finally:
        if conn is not None:
            conn.close()


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
    variant's manifest becomes V2_WIDTHS, which the engine arguments take. --model is the release: an https:// URL,
    whose experts, index slices, manifests and dense files each box fetches into --model-cache (ms.release), or a local
    copy used in place. The release is the whole model (V2_DENSE: the engine plans every weight from its files).
    refresh: ask the hub what the revision is now (prepare), instead of the commit recorded at the last resolve."""
    ms = variant = release = None
    g6load = os.path.join(os.path.abspath(o.get('GR8_WORK_DIR')), 'bin', 'g6load')   # plans and checks a Hugging Face fetch
    if os.access(g6load, os.X_OK):
        os.environ.setdefault('G6LOAD', g6load)
    try:
        model = o.get('MODEL')
        hf = hfsource.parse_url(model)
        steering = steering_hf.request(o)
        if steering and not hf:
            raise modelset.ModelError('--steering-profile hf:... requires an HF release (--model https://...)')
        if models.of(o) is models.PSEUDO:
            if not modelset.is_pseudo(model):
                raise modelset.ModelError(f'--model {model}: want a pseudo engine release, a directory with '
                                          'responses.toml and the tokenizer files')
            ms = modelset.pseudo_set(model, int(o.get('TP')), o.get('MODEL_DIR'))
        elif models.of(o) is models.INK2:   # the published safetensors checkpoint, read in place
            m = models.of(o)
            if not modelset.is_ink2(model):
                raise modelset.ModelError(f'--model {model}: want a local copy of the {m.title} checkpoint (config.json, '
                                          'model.safetensors.index.json and its shards, and the tokenizer files)'
                                          if model else f'no model: pass --model <{m.title} checkpoint directory>')
            ms = modelset.ink2_set(model, int(o.get('TP')), o.get('MODEL_DIR'))
        elif models.of(o).native_release:   # an exlr8 native release, read in place: no index slices, no variants
            m = models.of(o)
            if not modelset.is_release(model):
                raise modelset.ModelError(f'--model {model}: want a local copy of the {m.title} exlr8 native '
                                          'release (dense/, experts/, manifests/, tokenizer.json and tokenizer_config.json)'
                                          if model else f'no model: pass --model <{m.title} release directory>')
            ms = modelset.ds_set(model, int(o.get('TP')), o.get('MODEL_DIR'))
        elif hf:
            if not o.get('MODEL_CACHE'):
                raise modelset.ModelError(f'--model {model} needs --model-cache <local dir>: each box fetches its part '
                                          'there')
            if o.get('FETCH') == 'model-loader':
                release = mlsource.of(o, *hf, log=log)
            else:
                release = hfsource.Release(o.get('MODEL_CACHE'), *hf, refresh=refresh, log=log,
                                           layout=o.get('LAYOUT'))
            # GLM-5.3-Flash image input: each rank fetches the vision tower too
            release.vision = bool(models.of(o).vision_knob) and opt.extra_env_value(o, models.of(o).vision_knob) == '1'
            release.fetch_meta(steering[0] if steering else None)
            o.values['V2'], o.values['V2_SLICES'] = release.v2, release.slices
            model = release.dir
            ms = modelset.release_set(model, int(o.get('TP')), o.get('MODEL_DIR') or release.tokenizer, str(release))
        elif modelset.is_release(model):
            experts = os.path.join(model, 'experts')
            o.values['V2'] = o.get('V2') or (experts if os.path.isdir(experts) else model)
            o.values['V2_SLICES'] = o.get('V2_SLICES') or os.path.join(model, 'index-slices')
            ms = modelset.release_set(model, int(o.get('TP')), o.get('MODEL_DIR'))
        else:
            raise modelset.ModelError(f'--model {model}: want an exlr8 release, https://huggingface.co/<org>/<repo>[@rev] '
                                      'or a local copy of one (dense/, experts/, index-slices/, manifests/)'
                                      if model else 'no model: pass --model <exlr8 release URL or directory>')
        # a release for the other model is refused here, by name, before the variant, the drafter or any weights
        models.check_config(models.of(o), ms.tokenizer)
        o.values['V2_DENSE'] = model
        ms.release = release
        search = [model]
        spec = o.get('VARIANT') or o.get('V2_WIDTHS')
        variant = modelset.resolve_variant(spec, o.get('V2'), search, int(o.get('TP')), models.of(o))
        if variant.path:
            o.values['V2_WIDTHS'] = variant.path
        if o.get('GRAMMAR') == '1' and not o.get('GRAMMAR_DIR'):
            o.values['GRAMMAR_DIR'] = ms.tokenizer
        if o.get('GRAMMAR') != '1' or not models.of(o).grammar:   # g6m4_ds41 takes no grammars
            o.values['GRAMMAR_DIR'] = None
        steering_hf.bind(o, release)
        steering_local.prepare(o, ms)
        ms.drafter = drafter_of(o, ms, refresh, log)
        return ms, variant, ''
    except (modelset.ModelError, models.ModelMismatch) as e:
        return ms, variant, f'model: {e}'


def drafter_of(o, ms, refresh, log):
    """The DFlash2 drafter for this boot: hfsource.Drafter for --drafter, else None. At TP=4, TP=2 and TP=1 with no drafter
    (no --drafter, no DFLASH_DIR), DFLASH defaults to 0 with a notice, and an explicit --dflash 1 is refused. TP=6 (the
    drafter's body replicated on every rank) the same, except that DFLASH defaults to 0 there: --dflash 1 turns it on."""
    if o.get('TP') not in ('4', '6', '2', '1') or not models.of(o).drafter:
        return None
    if o.get('TP') == '6' and o.get('DFLASH') != '1':
        if o.get('DRAFTER'):
            log('--drafter at TP=6 is used with --dflash 1 only: MTP only')
        return None
    if o.get('DRAFTER'):
        if not o.get('MODEL_CACHE'):
            raise modelset.ModelError('--drafter needs --model-cache <local dir>: each box converts its rank there')
        if o.get('DFLASH') == '0':
            return None
        if o.get('FETCH') == 'model-loader':
            d = mlsource.drafter_of(o, *hfsource.parse_url(o.get('DRAFTER')), log=log)
        else:
            d = hfsource.Drafter(o.get('MODEL_CACHE'), *hfsource.parse_url(o.get('DRAFTER')), refresh=refresh,
                                 log=log, tp=int(o.get('TP')))
        o.values['DFLASH_DIR'] = d.out
        return d
    if o.get('DFLASH') == '0' or o.get('DFLASH_DIR'):
        return None
    url = models.of(o).drafter_url
    hint = f'--drafter {url} (CC BY-NC-ND 4.0: each box fetches and converts it)'
    if o.get('DFLASH') == '1':
        raise modelset.ModelError(f'--dflash 1, but there is no DFlash2 drafter: add {hint}')
    log(f'no DFlash2 drafter: MTP only. For DFlash2, add {hint}')
    o.values['DFLASH'] = '0'
    return None


ENGINES = {n.encode() for m in models.MODELS.values() for n in (m.engine, m.engine + '-int4')}


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
        if b'--rank' in argv and any(os.path.basename(a) in ENGINES for a in argv[:2]):
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
        self.home = os.path.abspath(o.get('GR8_WORK_DIR'))
        self.out = os.path.join(self.home, 'out')
        self.port = o.int('CONTROL_PORT')
        self.boot_attempt = 0
        self.dispram_off = False
        self.fallback_by_epoch = {}
        self.stop_ev = threading.Event()
        self.stop_why = ''
        self.stop_all = False         # on the head: stop every box as well
        self.stop_failed = False      # the stop is a failure (exit 1): a rank failed, here or elsewhere
        self.head = None              # handshake.Head on rank 0
        self.head_ready = threading.Event()
        self.engine = None
        self.helpers = {}             # the head's front end and shim: name -> Popen
        self.donor = False            # a KV node beside a Flash serve runs (donate)
        self.agent = self.api_agent = None
        self.rank, self.lan, self.members, self.hs = None, None, [], {}
        self.state = 'starting'
        self.state_lock = threading.RLock()
        self.serving_urls = {}        # retain early notifications by epoch until the handshake chooses a boot
        self.servers = []
        self.locked = self.started = False
        self.ms, self.model_error, self.rails, self.fabric_error = None, '', ('', ''), ''
        # with mentat: every rank's rails as this box's election chose them (the head's choice is the one used), and
        # this box's rdma addresses, any of which the head may choose for it
        self.elected, self.rdma = None, []
        # --shape ring with mentat: the plan, this box's devices, and the client link that holds the ring claim
        self.ring, self.ring_devs, self.ring_ports, self.claim_link = '', None, {}, None
        self.offers = self.variant = self.model_id = None
        self.lingered = False
        self.refusal_logged = False   # the head printed its refusal before the extra wait
        self.check_n, self.check_lock = 0, threading.Lock()   # the engine's DIR/check.<n>, numbered per boot
        # ./gr8 <model> scale: the head's record of the running scale (scale_begin), this box's part in it (scale_pending),
        # what the switch left (for a rollback), and the store service the new ranks read this box's old store through
        self.scale = self.pending = self.prev = None
        self.scale_go, self.scale_abort, self.scale_done = threading.Event(), threading.Event(), threading.Event()
        self.scale_n = 0
        # the options every scale this cluster went through added (the head's): a box that joins later boots with them too
        self.scale_words = []
        self.scale_seen = set()       # every member of a scale, old and new: they keep asking status and logs
        self.peer_proc = None         # bin/kvman-peer
        self.hello_poll = handshake.POLL_S   # a worker's verdict poll: shorter while a scale's hold runs
        # POOL: every box that may serve, resolved; the planned layout of the running width (the head's, for adopt); and
        # the TP and POOL this head booted under, which its kept width (out/scale.json) must match at the next up
        self.pool, self.layout_text, self.base = [], None, [o.get('TP'), o.get('POOL')]

    # ---- state ----------------------------------------------------------------------------------------------------
    def set_state(self, state, **meta):
        with self.state_lock:
            url = self.serving_urls.get(self.hs.get('epoch'))
            if state == 'loading' and url is not None:
                state = 'serving'
                log(f'the head serves {url}')
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
        listening = {s.server_address[0] for s in self.servers}
        for a in dict.fromkeys(addrs):
            if a in listening:
                continue
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
        known |= set(self.pool)
        known |= {a for r in self.elected or [] for a in r} | self.scale_seen
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
            att = int(m.get('attempt') or 0)
            if att > self.boot_attempt:
                return {'t': 'retry', 'detail': 'the head is resetting for dispram fallback'}
            if att < self.boot_attempt:
                if t == 'verdict' and att == 0 and self.boot_attempt == 1 and self.dispram_off:
                    return {'t': 'dispram_fallback', 'detail': 'the head fell back to baseline CAP'}
                return {'t': 'refused', 'field': 'attempt', 'values': {}, 'exit': 4, 'detail': 'stale boot attempt'}
            if self.rank not in (0, None):
                return {'t': 'refused', 'field': 'rank', 'values': {}, 'exit': 4,
                        'detail': f'{self.lan} is rank {self.rank}, not the head'}
            s = self.scale
            rolling = s and s['phase'] == 'failed' and not s.get('closed')   # the head boots the old list again
            if t == 'hello' and s and (s['phase'] in ('prepare', 'drain', 'switch') or rolling) and \
                    m.get('members') != self.members:
                # a hello for another member list while a scale runs: a box that joins it, or a late one, or one of the
                # old list during the head's rollback. After the scale (done, or failed and closed) such a box gets the
                # handshake's refusal, so a restart policy does not loop
                if s['phase'] in ('prepare', 'drain') and m.get('members') == s['members'] and \
                        peer not in s['old_members'] and s.get('msg'):
                    if peer not in s['ready']:
                        return {'t': 'scale', **s['msg']}
                return {'t': 'retry', 'detail': 'the cluster is rolling back to its old member list' if rolling
                        else f'the cluster is changing its member list (scale {s["phase"]})'}
            if t == 'hello' and self.pool and m.get('members') != self.members and not self.stop_ev.is_set():
                # a box of the pool that counted another member list (another TP, or the width the head kept from its
                # last scale): it takes this one, or waits as a standby
                return self.adopt_msg()
            if not self.head_ready.is_set():
                return {'t': 'retry', 'detail': 'the head has not finished its checks' if self.state == 'preparing'
                        else f'the head is {self.state}'}
            if t == 'hello' and self.stop_ev.is_set() and self.head.verdict and self.head.verdict['t'] == 'ok':
                # a booted cluster going down: a box that restarted already (a container) waits for the next boot
                return {'t': 'retry', 'detail': 'the head is stopping its boot'}
            return self.head.rpc(m, peer)
        if t == 'pool':
            return self.on_pool(peer)
        if t == 'dispram_fallback':
            epoch = m.get('epoch')
            why = m.get('why') or 'dispram shortfall'
            bcast = False
            with self.state_lock:
                if not isinstance(epoch, int) or self.boot_attempt != 0:
                    return {'t': 'ok', 'detail': 'ignored'}
                if self.hs and epoch != self.hs.get('epoch'):
                    return {'t': 'ok', 'detail': 'not this boot'}
                if epoch not in self.fallback_by_epoch:
                    self.fallback_by_epoch[epoch] = why
                    bcast = self.rank == 0
            if bcast:
                for r in range(1, len(self.members)):
                    self.tell(r, {'t': 'dispram_fallback', 'epoch': epoch, 'why': why}, timeout=2)
            return {'t': 'ok'}
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
            epoch = m.get('epoch')
            if not isinstance(epoch, int):
                return {'t': 'ok', 'detail': 'no boot epoch'}
            with self.state_lock:
                if self.hs and epoch != self.hs.get('epoch'):
                    return {'t': 'ok', 'detail': 'not this boot'}
                self.serving_urls[epoch] = m.get('url', '')
                if self.state == 'loading' and epoch == self.hs.get('epoch'):
                    self.set_state('serving')
                    log(f'the head serves {self.serving_urls[epoch]}')
            return {'t': 'ok'}
        if t == 'logs':
            return self.read_log(m.get('which', 'engine'), int(m.get('offset', -8192)))
        if t == 'check':
            return {'t': 'check', **self.local_check()}
        if t == 'check_all':
            return self.check_all()
        if t == 'scale':
            return self.scale_begin(m)
        if t == 'scale_prepare':
            def prepare():
                ok, why = self.scale_prepare(m)
                self.tell_head({'t': 'scale_ready', 'id': m['id'], 'lan': self.lan, 'ok': ok, 'why': why})
            threading.Thread(target=prepare, daemon=True).start()
            return {'t': 'ok'}
        if t == 'scale_ready':
            s = self.scale
            if s and s['id'] == m.get('id') and s['phase'] == 'prepare':
                s['ready'][m.get('lan')] = (bool(m.get('ok')), m.get('why') or '')
            return {'t': 'ok'}
        if t == 'scale_go':
            if self.pending and self.pending['id'] == m.get('id'):
                self.scale_go.set()
            return {'t': 'ok'}
        if t == 'scale_abort':
            self.on_scale_abort(m.get('id'), m.get('why') or 'aborted')
            return {'t': 'ok'}
        if t == 'scale_failed':
            if self.scale and self.scale['id'] == m.get('id'):
                self.scale_abort_all(self.scale, f'{peer}: {m.get("why")}')
            return {'t': 'ok'}
        if t == 'scale_done':
            if self.pending and self.pending['id'] == m.get('id'):
                self.scale_done.set()
                if not self.pending.get('leaves'):
                    self.finish_scale()
            return {'t': 'ok'}
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
             'variant': self.variant.name if self.variant else None, 'model_id': self.model_id,
             'members': self.members}
        if s['engine'] is False:
            err = os.path.join(self.out, f'g6m4.r{self.rank}.err')
            s['last_line'] = tail(err, 1).strip()
        if self.rank == 0:
            for p in ('shim', 'front'):
                s[p] = bool(recorded(os.path.join(self.out, f'{p}.pid')))
            if self.scale:
                s['scale'] = {k: self.scale.get(k) for k in ('id', 'phase', 'error', 'members', 'old_members', 'pause_s',
                                                             'drain_s', 'timeline', 'config')}
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
        # gr8 down reads this box's quit wait here: a store given only on the up command line is in self.o, not in
        # the environment down parses
        with open(os.path.join(self.out, 'quit_wait'), 'w') as f:
            f.write(f'{quit_wait(self.o)}\n')
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
        for line in opt.kv_describe(o):
            log(line)
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
        self.offers = None if self.ms else []     # a release serves any rank; a box without a model serves none
        log(f'this box can serve rank {mentat.format_offers(self.offers) or "none"} of the model set')
        self.rendezvous(tp)
        if self.rank is None:   # a box of POOL outside the member list
            self.serve([self.lan])
            fails = self.standby()
        else:
            self.cluster_opts()
            log(f'rank {self.rank} of {tp} ({self.lan}, rails {self.rails[0]} {self.rails[1]}), cluster options '
                f'{self.digest[:16]}, engine {self.build[:16]}')
            if self.agent:
                self.agent.set_meta(**{'gr8.set': self.model_id or 'none', 'gr8.variant': str(self.variant or ''),
                                       'gr8.opts': self.digest,
                                       'gr8.ips': ','.join(self.rails)})
            self.serve([a for a in (self.lan, *self.rails, *self.rdma) if a])
            self.set_state('preparing')
            fails = self.checks()
        for f in fails:
            log(f'check failed: {f}')
        while True:
            dbytes = 0 if self.dispram_off else probe_dispram(self.o)
            try:
                hello = {'t': 'hello', 'proto': handshake.PROTO, 'rank': self.rank, 'lan': self.lan, 'members': self.members,
                         'rails': list(self.rails), 'home': self.home, 'opts': self.opts, 'opts_digest': self.digest,
                         'set_id': self.model_id or 'none',
                         'set_source': f'{self.ms.source}, variant {self.variant}' if self.ms else self.model_error,
                         'engine_build': self.build, 'fails': fails, 'version': version(self.home),
                         'offers': mentat.format_offers(self.offers),
                         'dispram_bytes': dbytes, 'attempt': self.boot_attempt}
                self.set_state('handshake')
                verdict = self.handshake(hello)
                if verdict.get('t') == 'scale':   # this box joins a running cluster's scale
                    fails = self.join_scale(verdict)
                    continue
                if verdict.get('t') == 'adopt':   # a pool's head runs another member list: take it, or stand by
                    fails = self.adopt(verdict)
                    continue
                if verdict.get('t') == 'dispram_fallback':
                    raise DispramFallback(verdict.get('detail') or 'dispram shortfall')
                if verdict.get('t') == 'ok' and fails:   # never boot past a failed check, whatever the verdict
                    verdict = {'t': 'refused', 'field': 'checks', 'values': {}, 'detail': '; '.join(fails), 'exit': 4}
                if verdict.get('t') != 'ok':
                    if not self.refusal_logged:
                        log(handshake.describe(verdict))
                    e = Stop(verdict.get('exit', 4), f'handshake refused: {verdict["field"]}')
                    e.logged = True
                    raise e
                with self.state_lock:
                    self.hs = verdict
                    self.mark('handshake')
                    self.fallback_by_epoch = {e: w for e, w in self.fallback_by_epoch.items() if e == verdict['epoch']}
                    self.serving_urls = {e: url for e, url in self.serving_urls.items() if e == verdict['epoch']}
                if verdict.get('ips'):     # the head's rails for this box, which may differ from its own election's
                    self.rails = (verdict['ips'][self.rank], verdict['ips2'][self.rank])
                    if self.agent:
                        self.agent.set_meta(**{'gr8.ips': ','.join(self.rails)})
                with open(os.path.join(self.out, 'handshake.json'), 'w') as f:
                    json.dump({**verdict, 'rank': self.rank, 'control_port': self.port}, f, indent=1)
                self.boot()
                break
            except DispramFallback as fb:
                if self.boot_attempt != 0:
                    raise Stop(1, str(fb))
                epoch = self.hs.get('epoch')
                log(f'dispram shortfall on boot attempt 0 ({fb}); falling back to baseline CAP with GR8_DISPRAM=0')
                self.head_ready.clear()
                notice = {'t': 'dispram_fallback', 'epoch': epoch, 'why': str(fb)}
                if self.rank == 0:
                    for r in range(1, len(self.members)):
                        self.tell(r, notice, timeout=2)
                else:
                    self.tell_head({**notice, 'rank': self.rank})
                stop_local(self.home, log, quit_s=2)
                for child in [self.engine, *self.helpers.values()]:
                    if child is not None:
                        child.wait()
                self.engine = None
                self.helpers.clear()
                self.started = False
                with self.state_lock:
                    self.head = None
                    self.hs = {}
                    self.fallback_by_epoch.clear()
                    self.dispram_off = True
                    self.boot_attempt = 1
                continue
            except Rescale:
                fails = self.rescale()
            except Rollback:
                fails = self.rollback()
            except Stop as e:
                p = self.pending
                pool_joiner = p and p.get('joiner') and self.pool
                if not (p and (p.get('switched') or pool_joiner)) or self.stop_ev.is_set():
                    raise
                # the new member list failed to boot: every box of the old one boots it again (a standby that joined
                # stands by again)
                log(f'scale {p["id"]}: the new member list failed here: {e}')
                if self.rank == 0 and self.scale:
                    self.scale_abort_all(self.scale, f'rank 0: {str(e).splitlines()[0] if str(e) else e}')
                else:
                    self.tell(0, {'t': 'scale_failed', 'id': p['id'], 'why': str(e).splitlines()[0] if str(e) else ''})
                    if not self.scale_abort.wait(opt_timeout(self.o)):
                        raise
                fails = self.rollback()

    def rendezvous(self, tp):
        o = self.o
        if opt.mode(o) == 'peers':
            self.members = self.names(lambda: opt.resolve_peers(o))
            if o.get('POOL'):
                self.pool = self.names(lambda: opt.resolve_peers(o, 'POOL'))
                if not set(self.members) <= set(self.pool) or self.members[0] != self.pool[0]:
                    raise Stop(2, 'PEERS must be boxes of POOL, with the same head first')
            try:
                self.rank = fabric.own_index(self.members, o.get('NODE_IP'))
            except RuntimeError as e:
                try:
                    i = fabric.own_index(self.pool, o.get('NODE_IP')) if self.pool else None
                except RuntimeError:
                    i = None
                if i is None:
                    raise Stop(2, str(e))
                self.rank, self.lan = None, self.pool[i]
                log(f'pool: {",".join(self.pool)}; members {",".join(self.members)}; this box ({self.lan}) is a standby')
                return
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
            publish = self.ring_publisher() if o.get('SHAPE') == 'ring' else None
            if publish:
                self.claim_link = mentat.ClaimLink(self.daemon, group, f'{group}@gr8-ring@{node_ip}')
            self.agent.start()
            self.set_state('joining')
            try:
                ranks, _ = mentat.elect(self.daemon, group, tp, o.get('HEAD'), o.int('JOIN_TIMEOUT'), log=log,
                                        ring=self.claim_link, publish=publish)
            except mentat.NoAssignment as e:
                raise Stop(4, str(e))
            except RuntimeError as e:
                raise Stop(1, str(e))
            self.members = [lan for lan, _, _ in ranks]
            if node_ip not in self.members:
                raise Stop(1, f'this box ({node_ip}) is not among the members {",".join(self.members)}')
            self.rank, self.lan = self.members.index(node_ip), node_ip
            log(f'mentat group {group}: {",".join(self.members)}; this box is rank {self.rank}')
            if publish:
                self.ring_from(ranks)
            if not o.get('IPS') and o.get('SHAPE') != 'ring':
                self.rails = ranks[self.rank][2]
                self.elected = [r[2] for r in ranks]
                self.rdma = mentat.node_view(local)[2]
                for p in mentat.probe_problems(local, ranks, node_ip, self.rails):
                    log(f'warning: {p}')
        self.rails, self.fabric_error = self.rails_for(o, self.rank)
        if self.fabric_error:
            log(self.fabric_error)

    def rails_for(self, o, rank):
        """(this box's rails as rank `rank` under options o, why there are none): the LAN address with no switch or one
        box (the engine bootstraps over the LAN), --ips/--ips2, this box's RoCE v2 addresses with --peers, else the
        ones mentat's election chose."""
        if o.get('SHAPE') == 'ring' or not fabric_needed(o):
            return (self.lan, self.lan), ''
        if o.get('IPS'):
            return (opt.split_list(o.get('IPS'))[rank], opt.split_list(o.get('IPS2'))[rank]), ''
        if opt.mode(o) != 'peers':
            return self.rails, ''
        try:
            return fabric.rails(), ''
        except RuntimeError as e:
            return self.rails, f'fabric: {e}'

    def cluster_opts(self):
        """The engine template, its build and the cluster options the handshake compares, for self.o."""
        self.tmpl = opt.engine_plan(self.o)
        self.build = engine_build(self.home, engine_bin(self.o))
        self.opts, _, self.digest = opt.canonical(self.o, self.tmpl, self.members if opt.mode(self.o) == 'peers' else None,
                                                  self.variant, ring=self.ring)

    def ring_publisher(self):
        """--shape ring in mentat mode: a callable that publishes this box's gr8.roots meta (its RoCE v2 addresses and
        their PCIe root) when it changes. Stop(1) when this box has no RoCE v2 address."""
        def read_ports():
            try:
                return fabric.ring_ports()
            except OSError as e:
                raise Stop(1, f'--shape ring: cannot read /sys/class/infiniband ({e})')
        self.ring_ports = read_ports()
        if not self.ring_ports:
            raise Stop(1, '--shape ring: no ACTIVE RoCE v2 address under /sys/class/infiniband: are the ring cables '
                          'in and addressed?')
        said = [None]

        def publish():
            # read again each time: a port that finishes link training during the wait joins the ring
            self.ring_ports = read_ports()
            text = mentat.format_roots({a: r for a, (_, r) in self.ring_ports.items()})
            if text != said[0]:
                said[0] = text
                self.agent.set_meta(**{mentat.ROOTS_META: text})
        return publish

    def handshake(self, hello):
        """The handshake's verdict, after which the ring claim is let go: every box that passed it has the view, and
        one that failed claims afresh on its next boot."""
        try:
            return self.handshake_head(hello) if self.rank == 0 else self.handshake_worker(hello)
        finally:
            self.release_claim()

    def release_claim(self):
        """Close the ring claim's link: once every box has, mentat drops the claim and the next boot solves afresh."""
        if self.claim_link:
            self.claim_link.close()

    def ring_from(self, ranks):
        """This box's --ring-devs from the ring mentat's probes form: its four addresses, mapped to device names. An
        explicit --ring-devs wins."""
        ports = ranks[self.rank][2]
        self.ring = mentat.ring_text([lan for lan, _, _ in ranks], [p for _, _, p in ranks], self.claim_link.tag)
        log(f'ring (mentat claim {self.claim_link.tag}): {" -> ".join(lan for lan, _, _ in ranks)} -> {ranks[0][0]}; '
            f'this box\'s relay-out, next, relay-in, prev: {",".join(ports)}')
        devs = [self.ring_ports[a][0] for a in ports]
        for a, d in zip(ports, devs):
            others = sorted(x for x, (dd, _) in self.ring_ports.items() if dd == d and x != a)
            if others:   # the engine takes the device's first RoCE v2 IPv4 GID (fabric.cpp ring_gid_index)
                log(f'warning: --shape ring: {d} carries {a} and {", ".join(others)}; the engine uses its first RoCE '
                    'v2 address, which may not be the one on the cable')
        if self.o.get('RING_DEVS'):
            if opt.split_list(self.o.get('RING_DEVS')) != devs:
                log(f'warning: --ring-devs {self.o.get("RING_DEVS")} overrides the cabling mentat found '
                    f'({",".join(devs)})')
        else:
            self.ring_devs = ','.join(devs)

    def ring_args(self):
        """The engine's ring flags: --ring, and --ring-devs from the option or else from mentat's probes."""
        if self.o.get('SHAPE') != 'ring':
            return []
        devs = self.o.get('RING_DEVS') or self.ring_devs
        return ['--ring', *(['--ring-devs', devs] if devs else [])]

    def checks(self, copy=True, p=None):
        """What would stop this box's boot, as text. copy: fetch this rank's model files first. The rank's files are
        checked only once the rank is known, and the front end only on rank 0. p: a scale's new options and rank for this
        box (scale_pending), checked while the engine serves: no running-engine or memory checks."""
        if p:
            o, rank, ms, variant, build = p['o'], p['rank'], p['ms'], p['variant'], p['build']
            fails = [e for e in (p['model_error'], p['fabric_error']) if e]
        else:
            o, rank, ms, variant, build = self.o, self.rank, self.ms, self.variant, self.build
            fails = [e for e in (self.model_error, self.fabric_error) if e]
        try:
            if not ms or rank is None:
                pass
            elif copy:
                rel = ms.release
                if rel and (getattr(rel, 'pins', False) or rel.check(int(o.get('TP')), rank, variant)):
                    rel.fetch_rank(int(o.get('TP')), rank, variant)
                if ms.drafter and (getattr(ms.drafter, 'pins', False) or ms.drafter.check(rank)):
                    ms.drafter.fetch_rank(rank)
            if ms and rank is not None:
                if ms.release:
                    fails += ms.release.check(int(o.get('TP')), rank, variant)
                if ms.drafter:
                    fails += ms.drafter.check(rank)
            if ms:
                fails += modelset.check_tokenizer(ms)
        except (modelset.ModelError, OSError, RuntimeError) as e:
            fails.append(f'model: {e}')
        if not p:
            pid = recorded(os.path.join(self.out, 'g6m4.pid'))
            if pid:
                fails.append(f'an engine is already running (pid {pid}): ./gr8 glm53 down first')
            for e, cwd in running_engines():
                if e != pid:
                    fails.append(f'a g6m4 --rank process is running (pid {e}, cwd {cwd}): stop it first')
        if o.get('CHECK') == '0':
            return fails
        if build == 'missing':
            name = engine_bin(o)
            fails.append(f'no {self.home}/bin/{name}: ' + ('KV_FORMAT=int4 ' if name.endswith('-int4') else '') +
                         f'./gr8 {models.of(o).name} build')
        if fabric_needed(o) and not os.path.isdir('/dev/infiniband'):
            fails.append('no /dev/infiniband')
        if rank == 0:
            if not os.access(self.front_bin(), os.X_OK):
                fails.append(f'no {self.front_bin()}: ./gr8 glm53 build (or scripts/build_front.sh bin/)')
            if not os.access(self.py(), os.X_OK):
                fails.append(f'no {self.py()} for the shim: python3 -m venv {self.home}/venv && '
                             f'{self.home}/venv/bin/pip install -r frontend/requirements.txt')
        if p:
            return fails
        m = mem_available_gib()
        log(f'MemAvailable {m} GiB' if m is not None else 'MemAvailable unknown')
        if m is not None and m < models.of(o).min_free_gib:
            fails.append(f'MemAvailable is {m} GiB: the engine stops a boot under GR8_MEM_MIN (8 GiB), so something else '
                         'holds this box\'s memory; free it first')
        elif m is not None and m < 8:
            need = (f'about {modelset.expert_gb(variant.path, int(o.get("TP"))):.0f} GB of experts'
                    if variant and variant.path else 'most of the box')
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
        halt = Either(self.stop_ev, self.scale_abort)   # a scale's rollback ends a handshake of its new member list
        if self.head.wait_all(opt_timeout(self.o), halt):
            have = self.head.have
            # with mentat the head's election decides every rank's rails: each box chose from its own probe table,
            # and two tables can disagree during a boot
            rails = self.elected or [tuple(have[r]['rails']) for r in range(tp)]
            for r in range(1, tp):
                if tuple(have[r]['rails']) != tuple(rails[r]):
                    log(f'head: rank {r} proposed rails {" ".join(have[r]["rails"])}; using {" ".join(rails[r])}')
            bad = self.rail_check(rails)
            if bad:
                self.head.decide(handshake.refusal('fabric', have, lambda x: ','.join(rails[x['rank']]), bad))
            else:
                dispram_bytes = 0 if self.dispram_off else min(
                    max(0, int(have[r].get('dispram_bytes') or 0)) // (2 << 20) * (2 << 20) for r in range(tp))
                if dispram_bytes > 0:
                    log(f'head: dispram min free across {tp} ranks: {dispram_bytes >> 20} MiB')
                self.last_epoch = max(int(time.time()), (getattr(self, 'last_epoch', 0) or 0) + 1)
                self.head.decide({'t': 'ok', 'epoch': self.last_epoch, 'members': self.members,
                                  'dispram_bytes': dispram_bytes,
                                  'ips': [rails[r][0] for r in range(tp)],
                                  'ips2': [rails[r][1] for r in range(tp)],
                                  'homes': {str(r): have[r]['home'] for r in range(tp)}, 'set_id': self.model_id,
                                  'variant': self.variant.name,
                                  'opts_digest': self.digest, 's2port': self.tmpl['knobs'].get('S2PORT'),
                                  'fabric_port': opt.FABRIC_PORT, 'tp': tp, 'port': self.o.get('PORT')})
        elif halt.is_set():
            self.head.decide({'t': 'refused', 'field': 'stopped', 'values': {},
                              'detail': self.stop_why or 'the scale was aborted', 'exit': 1})
        v = self.head.verdict
        if v['t'] != 'ok' and not self.scale_abort.is_set():
            # say why now, not after the extra wait: with a box down that wait is up to JOIN_TIMEOUT
            log(handshake.describe(v))
            self.refusal_logged = True
            if not self.stop_ev.is_set():   # a stopped head does not linger: no wait to announce
                log(self.linger_plan())
            self.linger()
        return v

    def linger_plan(self):
        """The head's line before linger(): how long it keeps answering, and for which ranks."""
        stay = float(os.environ.get('GR8_REFUSAL_LINGER', '15'))
        more = opt_timeout(self.o)
        waiting = self.head.untold() if self.head is not None else []
        if not waiting or more <= 0 or self.head.gave_up:   # JOIN_TIMEOUT 0 here: no extra wait (wait_told(0))
            return f'head: answering late hellos with the refusal for {stay:.0f} s'
        ranks = ', '.join(str(r) for r in waiting)
        return (f'head: answering late hellos with the refusal for {stay:.0f} s, then up to {more:.0f} s more '
                f'until rank{"s" if len(waiting) > 1 else ""} {ranks} {"have" if len(waiting) > 1 else "has"} it')

    def linger(self):
        """Answer late hellos and polls with the head's refusal for GR8_REFUSAL_LINGER seconds, then on until every
        worker has it (a box still fetching says hello late), at most JOIN_TIMEOUT more, before leaving."""
        if not self.lingered:
            self.lingered = True
            if self.stop_ev.wait(float(os.environ.get('GR8_REFUSAL_LINGER', '15'))):
                return
            if self.head is not None:
                self.head.wait_told(opt_timeout(self.o), self.stop_ev)

    def rail_check(self, rails):
        """From each of the head's rails to the same rail of every worker (rails: every rank's pair): the control port
        must answer."""
        bad = []
        for r, h in sorted(self.head.have.items()):
            if r == 0:
                continue
            for k in range(2):
                try:
                    handshake.call(rails[r][k], self.port, {'t': 'ping'}, timeout=3, bind=rails[0][k])
                except OSError as e:
                    bad.append(f'rank {r} ({h["lan"]}) does not answer on rail {k} at {rails[r][k]} from '
                               f'{rails[0][k]} ({e})')
        return '; '.join(bad)

    def handshake_worker(self, hello):
        log(f'sending the hello to the head {self.members[0]}:{self.port}')
        v = handshake.send_hello(self.members[0], self.port, hello, self.lan, opt_timeout(self.o), log,
                                 Either(self.stop_ev, self.scale_abort), poll_s=self.hello_poll)
        if v.get('t') == 'error':   # the head answers members only: a box its cluster does not list (no scale adds it)
            v = {'t': 'refused', 'field': 'member list', 'values': {}, 'exit': 4,
                 'detail': f'the head {self.members[0]}: {v.get("error")}'}
        return v

    def engine_env(self, plan):
        """The plan's variables over an allowlist of this process's environment."""
        env = {k: v for k, v in os.environ.items() if k in ENGINE_ENV_KEEP or k.startswith(ENGINE_ENV_PREFIXES)}
        env.update(kv.split('=', 1) for kv in plan['env'] if '=' in kv)
        if self.dispram_off:
            env['GR8_DISPRAM'] = '0'
            env.pop('GR8_DISPRAM_MIN_BYTES', None)
        lib = os.path.join(self.home, 'bin', 'lib')
        if os.path.isdir(lib):        # cuBLAS from build.sh BUNDLE_LIBS=1, at any TP
            env['LD_LIBRARY_PATH'] = lib
        if self.borrow_idle(plan.get('knobs') or {}):
            # this box's host ledger (donate() creates it); the engine writes the bytes it may lend next to it (.gpu)
            out = kvnode.out_dir(kvnode.parse(['--work-dir', self.home], dict(os.environ)))
            os.makedirs(out, exist_ok=True)
            env['GR8_GLM53F_KV_LEDGER'] = os.path.join(out, 'ledger')
        p = self.pending
        if p and p.get('rewidth') and (p['switched'] or p['joiner']):
            # a scale: the engine rewrites the KV store the old member list left for this rank (knobs/serving.yaml)
            env['GR8_KV_REWIDTH'] = p['rewidth']
        return env

    def borrow_idle(self, knobs):
        return models.of(self.o).checkpoint_cache and knobs.get('FLASH_CAP_POLICY') == 'borrow-idle'

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

    def kv_node(self, flag):
        """host:port of the KV node mentat announces that this box uses, by its rails and rank (mentat.kv_target)."""
        g = self.o.get('GROUP')
        try:
            nodes = mentat.kv_nodes(self.daemon.group_status(g), g)
        except (OSError, RuntimeError, ValueError) as e:
            raise Stop(1, f'{flag} remote:auto: mentat: {e}')
        target = mentat.kv_target(nodes, self.rails, self.rank, mentat.local_nets())
        if not target:
            raise Stop(1, f'{flag} remote:auto: no KV node in group {g} (./gr8 kv up on a box with spare memory)')
        return target

    def kv_store(self):
        """KV_STORE for this box's engine: each remote:auto becomes the KV node mentat announces, by this box's rails."""
        stores = opt.kv_stores(self.o)
        if not any(t['host'] == 'auto' for t in stores):
            return None
        target = self.kv_node('--kv-store')
        log(f'KV store: the KV node at {target}')
        return opt.kv_store_with(self.o, target)

    def kv_rows(self):
        """KV_ROWS for this box's engine: remote:auto becomes the KV node by this box's rails, as for KV_STORE (odd ranks
        on the node's second root)."""
        t = opt.kv_rows(self.o)
        if not t or t['host'] != 'auto':
            return None
        target = self.kv_node('--kv-rows')
        log(f'KV rows: the KV node at {target}')
        return opt.kv_rows_with(self.o, target)

    def boot(self):
        o, hs, r = self.o, self.hs, self.rank
        plan = opt.engine_plan(o, rank=r, rank_dir=self.ms.dir, aux_dir=self.ms.dir, home=self.home,
                               epoch=hs['epoch'], set_id=self.model_id, kv_store=self.kv_store(),
                               kv_rows=self.kv_rows(),
                               dispram_bytes=0 if self.dispram_off else int(hs.get('dispram_bytes') or 0))
        work = os.path.join(self.home, 's2')
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        self.check_n = 0
        env = self.engine_env(plan)
        # second_rail='prefill' (Flash) takes the second rail for prefill only
        ips2 = '--ips2' if models.of(o).second_rail == 'all' else '--pf-ips2'
        argv = ['./bin/' + engine_bin(o), '--rank', str(r), '--ips', ','.join(hs['ips']), ips2, ','.join(hs['ips2']),
                '--proxy-cpu', o.get('PROXY_CPU'), *plan['args']]
        argv[1:1] = self.ring_args()
        log(f'engine: {" ".join(plan["env"])} {" ".join(argv)}')
        rotate_logs(self.out, [f'g6m4.r{r}.txt', f'g6m4.r{r}.err'])
        with open(os.path.join(self.out, f'g6m4.r{r}.txt'), 'wb') as so, \
                open(os.path.join(self.out, f'g6m4.r{r}.err'), 'wb') as se:
            self.engine = subprocess.Popen(argv, cwd=self.home, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
        with open(os.path.join(self.out, 'g6m4.pid'), 'w') as f:
            f.write(f'{self.engine.pid}\n')
        self.started = True
        self.mark('started')
        self.set_state('loading')
        if r == 0 and 'front' not in self.helpers:   # a scale keeps the front end, which holds requests through it
            self.start_front(plan)
        ready = os.path.join(work, 'ready.s2')
        tmo = float(plan['knobs'].get('READY_TMO') or 1800)
        t0, ask_at = time.monotonic(), tmo
        log(f'engine started (epoch {hs["epoch"]}); waiting for it to load (up to {tmo:.0f}s)')
        err = os.path.join(self.out, f'g6m4.r{r}.err')
        # a scale's boot relays its stage lines for every model (the hold's parts); other boots where the model asks
        relay = StageRelay(err, SCALE_STAGE) if self.pending else StageRelay(err) if models.of(o).stage_relay else None
        while True:
            self.watch()
            for line in relay.new() if relay else ():
                log(f'{time.monotonic() - t0:6.0f}s {line}')
            if r == 0:
                if os.path.exists(ready):
                    self.mark('ready')
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
            time.sleep(0.1 if r == 0 else 1)
        for line in relay.new() if relay else ():
            log(f'{time.monotonic() - t0:6.0f}s {line}')
        if r == 0:
            log(f'engines ready after {time.monotonic() - t0:.0f}s')
            self.start_shim(work)
            self.scale_served()
        if not self.donor:
            self.donate(plan['knobs'])
        self.run_forever()

    def donate(self, knobs):
        """GLM-5.3-Flash with FLASH_KV_DONATE_GIB > 0: lend this box's spare memory, measured now that the serve has
        booted, to a KV node beside it: flash_donor_budget GiB, a host ledger crediting them to the node,
        and ./gr8 kv up --lazy-lock --ledger. The node is its own supervisor: its failure, or no budget, leaves the
        serve as it is."""
        if not models.of(self.o).checkpoint_cache or float(knobs.get('FLASH_KV_DONATE_GIB') or 0) <= 0:
            return
        ko = kvnode.parse(['--work-dir', self.home], dict(os.environ))
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'KV_MIN_FREE_GB': ko.get('KV_MIN_FREE_GB'),
               **{k: knobs.get(k) or '' for k in ('FLASH_KV_DONATE_GIB', 'FLASH_KV_GROWTH_GIB')}}
        script = os.path.join(opt.REPO, 'scripts', 'engine_args.sh')
        r = subprocess.run(['bash', '-c', '. "$0"; flash_donor_budget', script], env=env, capture_output=True,
                           text=True)
        gib = int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else 0
        if gib <= 0:
            why = r.stderr.strip() or 'MemAvailable leaves nothing above FLASH_KV_GROWTH_GIB + KV_MIN_FREE_GB'
            log(f'KV donation: none ({why})')
            return
        out = kvnode.out_dir(ko)
        os.makedirs(out, exist_ok=True)
        ledger = os.path.join(out, 'ledger')
        # FLASH_CAP_POLICY=borrow-idle: the GPU's lendable KV chunks (the engine wrote their bytes at boot) are booked
        # as the GPU's, and the node may grow into them once lent
        gpu = gpu_lendable(ledger) if self.borrow_idle(knobs) else 0
        if self.borrow_idle(knobs) and not gpu:
            log(f'KV donation: no {ledger}.gpu from the engine; it lends no GPU chunks')
        lb = ledger_bin(self.home)
        try:
            if not lb:
                raise RuntimeError('no kvman-ledger: ./gr8 glm53 build')
            # a transfer left pending by a node or rank that died with its memory: its bytes go to free
            subprocess.run([lb, ledger, 'recover', '--stale-s', '0'], capture_output=True, text=True)
            n = str(gib << 30)
            for argv in ([lb, ledger, 'init', '--total', str((gib << 30) + gpu), '--node', n, '--gpu', str(gpu)],
                         [sys.executable, kvnode.HERE, 'up', '--work-dir', self.home, '--gb', str(gib + (gpu >> 30)),
                          '--lazy-lock',
                          '--ledger', ledger, '--group', knobs['FLASH_KV_GROUP'],
                          # this serve's mentat, or none when it finds its boxes by PEERS (kv up announces through
                          # localhost otherwise, and fails on a box without mentatd)
                          '--mentat', self.o.get('MENTAT') or 'none']):
                p = subprocess.run(argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
                if p.returncode:
                    raise RuntimeError(f'{os.path.basename(argv[1] if argv[0] == sys.executable else argv[0])}: '
                                       f'{(p.stderr or p.stdout).strip()[-400:]}')
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
            log(f'KV donation of {gib} GiB failed; the serve goes on without it: {e}')
            return
        self.donor = True
        log(f'KV donation: a KV node beside the serve for group {knobs["FLASH_KV_GROUP"]}, {gib} GiB, locked lease '
            f'by lease (ledger {ledger})' + (f', and {gpu} bytes of GPU KV chunks lent while idle' if gpu else ''))

    def start_front(self, plan):
        o = self.o
        sock = os.path.join(self.home, 'run', 'engine.sock')
        os.makedirs(os.path.dirname(sock), exist_ok=True)
        try:
            os.unlink(sock)
        except FileNotFoundError:
            pass
        model = models.of(o)
        front = [self.front_bin()]
        if model.dialect != 'glm':
            front += ['--dialect', model.dialect]
        if model.served:
            front += ['--served-model-name', *model.served]
        if plan['knobs'].get('REUSE') == '1' or opt.kv_stores(o) or model.dialect == 'glm':
            # glm keeps prefix reuse on; other models report reused rows in the a line when REUSE=1 or --kv-store is set
            front += ['--enable-prompt-tokens-details']
        if o.get('DEFAULT_MAX_TOKENS') != '0':
            front += ['--default-max-tokens', o.get('DEFAULT_MAX_TOKENS')]
        else:   # a request within the limit, guard rows included, shares decode steps: keep a page in hand for the guard
            front += ['--share-tokens', str(int(plan['knobs']['SHARE_LIMIT']) - 64)]
        if o.get('DEFAULT_REASONING_EFFORT'):
            front += ['--default-reasoning-effort', o.get('DEFAULT_REASONING_EFFORT')]
        if o.get('DEFAULT_THINKING_BUDGET') != '0':
            front += ['--default-thinking-budget', o.get('DEFAULT_THINKING_BUDGET')]
        for k in ('DEFAULT_PRESENCE_PENALTY', 'DEFAULT_FREQUENCY_PENALTY'):
            if o.get(k) != '0':
                front += ['--' + k.lower().replace('_', '-'), o.get(k)]
        if o.get('DEFAULT_THINK_END_BOOST'):
            front += ['--default-think-end-boost', o.get('DEFAULT_THINK_END_BOOST')]
        if o.get('DEFAULT_SEQUENCE_PENALTY'):
            front += ['--default-sequence-penalty', o.get('DEFAULT_SEQUENCE_PENALTY')]
        if o.get('FORCED_TOOL_THINKING') == '1':
            front += ['--forced-tool-thinking', '1']
        if o.get('LITERAL_CONTENT') == '0':
            front += ['--literal-content', '0']
        if o.get('TOOL_REPEAT_GUARD') == '1':
            front += ['--tool-repeat-guard', '1']
            for k, dflt in (('TOOL_REPEAT_GUARD_REPEATS', '2'), ('TOOL_REPEAT_GUARD_TEMPERATURE', '0.7'),
                            ('TOOL_REPEAT_GUARD_RETRIES', '2')):
                if o.get(k) != dflt:
                    front += ['--' + k.lower().replace('_', '-'), o.get(k)]
        if o.get('ALLOW_THINKING_OFF') == '1' and model.dialect == 'glm':   # ds41 turns thinking off by itself
            front += ['--allow-thinking-off']
        if o.get('CLAMP_MAX_TOKENS') == '1':
            front += ['--clamp-max-tokens']
        if o.get('UNDECLARED_TOOL_CALLS') != 'emit':
            front += ['--undeclared-tool-calls', o.get('UNDECLARED_TOOL_CALLS')]
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
        # the sessions the engine runs: --sess, else the model's default from scripts/engine_args.sh (DS41 5; GLM-5.3 8;
        # GLM-5.3-Flash 8, 4 or 1 at TP 4, 3 or 2), never a fixed 8 (a DS41 shim that divided the 5 slabs by 8 capped a
        # request at 163,830 tokens)
        sess = opt.engine_plan(self.o)['knobs'].get('SESS') or self.o.get('SESS') or '8'
        argv = [self.py(), os.path.join(REPO, 'frontend', 'g6s2shim.py'), '--dir', work, '--sock', sock,
                '--sessions', sess]
        model = models.of(self.o)
        meta = models.release_metadata(model, self.ms.tokenizer if self.ms else '')
        if model.per_slot_pool:   # pool_rows = SESS * per-session CAP, unlike a shared pool
            argv += ['--capacity-slots', sess]
        if not meta['vocab'] or not meta['eos_ids']:
            raise opt.OptionError(f'{model.name}: set vocab/eos_ids in Model or provide release config.json metadata')
        if not model.shim_legacy:
            argv += ['--model-name', (model.served or (model.name,))[0], '--vocab-size', str(meta['vocab']),
                     '--eos-ids', ','.join(map(str, meta['eos_ids']))]
        if self.o.get('TP') == '6':
            argv += ['--stall-timeout', '2400']
        # the longest request (prompt + max_tokens): MAX_MODEL_LEN, else the model's max positions. The shim takes the
        # smaller of it and the pool's room (pool_rows / capacity_slots - guard), and the front end reads the result from
        # the shim's hello, so the two never disagree
        argv += ['--max-model-len', str(self.o.get('MAX_MODEL_LEN') or meta['max_positions'])]
        # one request's rows: the engine's --cap, which GR8_CAP_TOTAL may make smaller than the shared pool (the engine
        # fails its epoch on a longer request, so the shim must refuse it first)
        cap = self.o.get('CAP')
        if cap and not model.per_slot_pool:
            argv += ['--cap-rows', str(int(cap) // 64 * 64)]
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
                if self.tell(r, {'t': 'serving', 'url': url, 'epoch': self.hs['epoch']}):
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
        return self.call(self.members[r], msg, timeout, f'rank {r} ({self.members[r]})')

    def call(self, addr, msg, timeout=5, who=None):
        try:
            return handshake.call(addr, self.port, msg, timeout=timeout, bind=self.lan)
        except OSError as e:
            log(f'{who or addr}: {e}')
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
        """Raise Stop when this rank's engine has exited or someone asked to stop; Rescale and Rollback for a scale."""
        if self.stop_ev.is_set():
            raise Stop(1 if self.stop_failed else 0, self.stop_why)
        p = self.pending
        if p and p.get('joiner') and self.pool and self.scale_abort.is_set():
            raise Rollback()   # a standby that joined an aborted scale stands by again
        if p and not p.get('joiner'):
            if p.get('switched') and self.scale_abort.is_set():
                raise Rollback()
            if not p.get('switched') and self.scale_go.is_set():
                raise Rescale()
        can_fallback = self.state == 'loading' and self.boot_attempt == 0 and int(self.hs.get('dispram_bytes') or 0) > 0
        with self.state_lock:
            fb_why = self.fallback_by_epoch.get(self.hs.get('epoch'))
        if can_fallback and fb_why:
            raise DispramFallback(fb_why)
        if self.engine and self.engine.poll() is not None:
            name = f'g6m4.r{self.rank}'
            reason = exit_reason([os.path.join(self.out, f'{name}.err'), os.path.join(self.out, f'{name}.txt')])
            if can_fallback and (self.engine.returncode == 75 or reason.startswith('g6: dispram shortfall')):
                raise DispramFallback(reason or 'engine exited 75 (dispram shortfall)')
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
        self.hello_poll = handshake.POLL_S
        while True:
            self.watch()
            time.sleep(0.5 if self.pending else 2)

    # ---- scale: ./gr8 <model> scale --------------------------------------------------------------------------------
    def scale_begin(self, m):
        """The head takes `scale` from the CLI: the new options are this boot's, the CLI's words and the pins (the
        running variant, SHARE_LIMIT and REUSE_GRID, which keep the KV store's R and grid). Checked here, then run on a
        thread (scale_run); the CLI follows its phases in status."""
        model = models.of(self.o)
        with self.state_lock:
            busy = self.scale and self.scale['phase'] in ('prepare', 'drain', 'switch')
            why = (f'{self.lan} is rank {self.rank}, not the head' if self.rank != 0 else
                   f'scale {self.scale["id"]} is running ({self.scale["phase"]})' if busy else
                   f'the cluster is {self.state}, not serving' if self.state != 'serving' else
                   f'./gr8 {model.name} does not scale' if not model.scalable else
                   'scale needs the mesh: SHAPE=ring is refused' if self.o.get('SHAPE') == 'ring' else '')
        if why:
            return {'t': 'error', 'error': why}
        words = [str(w) for w in m.get('argv') or []]
        if not self.pool and not names_peers(words):
            return {'t': 'error', 'error': 'scale needs --peers LIST, the new member list with the head first (or a '
                                           'cluster booted with --pool)'}
        drops = scale_drops(self.pool, words)
        delta = scale_argv(self.scale_words, drops) + words + scale_pins(self.o, self.variant, self.tmpl['knobs'])
        try:
            o = opt.parse(scale_argv(self.o.argv, drops) + delta)
            opt.validate(o)
            members = opt.resolve_peers(o)
        except opt.OptionError as e:
            return {'t': 'error', 'error': f'options: {e}'}
        tp = o.get('TP')
        why = ('scale needs --peers, the new member list' if opt.mode(o) != 'peers' else
               f'TP={tp}: ./gr8 {model.name} serves TP {model.tp_sizes}' if tp not in model.tp else
               f'the head {self.lan} stays the first member: --peers {self.lan},...' if members[0] != self.lan else
               'the new member list must be boxes of POOL' if self.pool and not set(members) <= set(self.pool) else
               scale_fit(model, self.variant, tp))
        if why:
            return {'t': 'error', 'error': why}
        try:   # the hold covers a boot of the new width and, when it fails, one of the old (READY_TMO each)
            ready = max(float(opt.engine_plan(o)['knobs'].get('READY_TMO') or 0),
                        float(self.tmpl['knobs'].get('READY_TMO') or 0))
        except (OSError, RuntimeError, ValueError, opt.OptionError):
            ready = 0.0
        self.scale_n += 1
        s = {'id': f'{self.hs["epoch"]}.{self.scale_n}', 'members': members, 'old_members': list(self.members),
             'argv': delta, 'phase': 'prepare', 'error': '', 'ready': {}, 'msg': None, 'token': secrets.token_hex(16),
             'drain': float(m.get('drain_timeout') or 300), 'boot_s': max(900.0, ready + 120), 'marks': [],
             'extra_env': o.get('EXTRA_ENV'), 'live': bool(m.get('live'))}
        self.scale_seen |= set(members) | set(self.members)
        self.scale = s
        log(f'scale {s["id"]}: TP={self.o.get("TP")} {",".join(self.members)} -> TP={tp} {",".join(members)}')
        threading.Thread(target=self.scale_run, args=(s,), daemon=True).start()
        return {'t': 'ok', 'id': s['id'], 'joiners': [a for a in members if a not in self.members and a not in self.pool]}

    def scale_run(self, s):
        """The head's coordinator: plan the layout, every box of the new member list makes its files ready while the
        old one serves, the front end holds new requests until the running ones finish (at most s['drain'] seconds),
        then every old member switches (scale_go). Any failure before the switch leaves the cluster as it was."""
        try:
            # old_set: the running model set id, which names the width; the new ranks check the old store against it
            layout = self.scale_layout(s)
            s['argv'] = s['argv'] + scale_compact(s['extra_env'], len(s['members']),
                                                  self.variant.name if self.variant else '', s['members'], layout)
            s['msg'] = {'id': s['id'], 'argv': s['argv'], 'members': s['members'], 'old_members': s['old_members'],
                        'layout': layout, 'token': s['token'], 'old_set': self.model_id}
            for a in s['old_members']:
                if a != self.lan:
                    r = self.call(a, {'t': 'scale_prepare', **s['msg']}, timeout=10)
                    if not r or r.get('t') != 'ok':
                        raise RuntimeError(f'{a} did not take the scale: {r}')
            joiners = [a for a in s['members'] if a not in s['old_members'] and a not in self.pool]
            standbys = [a for a in s['members'] if a not in s['old_members'] and a in self.pool]
            if joiners:
                log(f'scale {s["id"]}: waiting for {", ".join(joiners)}: run ./gr8 {models.of(self.o).name} up with '
                    f'--tp {len(s["members"])} --peers {",".join(s["members"])} there')
            if standbys:
                log(f'scale {s["id"]}: waiting for the standbys {", ".join(standbys)} to take the scale')
            s['ready'][self.lan] = self.scale_prepare(s['msg'])
            need, end = set(s['members']) | set(s['old_members']), time.monotonic() + opt_timeout(self.o)
            while not need <= set(s['ready']):
                bad = [f'{a}: {w}' for a, (ok, w) in s['ready'].items() if not ok]
                if bad:
                    raise RuntimeError('; '.join(bad))
                if time.monotonic() > end:
                    raise RuntimeError(f'no answer from {", ".join(sorted(need - set(s["ready"])))} within '
                                       f'{opt_timeout(self.o):.0f}s (JOIN_TIMEOUT)')
                if self.stop_ev.is_set() or s['phase'] != 'prepare':
                    return
                time.sleep(0.2)
            bad = [f'{a}: {w}' for a, (ok, w) in s['ready'].items() if not ok]
            if bad:
                raise RuntimeError('; '.join(bad))
            if s.get('live'):
                # no engine module installs switch_width yet (#156), so an in-process cutover cannot happen: --live says
                # why and runs the drain/restart scale below
                ok, why = live_switch_capable(self.o, models.of(self.o))
                log(f'scale {s["id"]}: --live: {why if not ok else "the engine has no in-process width switch yet"}; '
                    'falling back to the drain/restart scale')
            s['phase'] = 'drain'
            self.front_hold(s['drain'] + s['boot_s'])
            s['held'] = time.time()
            log(f'scale {s["id"]}: every box is ready; new requests wait, the running ones finish (up to {s["drain"]:.0f}s)')
            while (h := self.front_hold())['running']:
                if time.time() - s['held'] > s['drain']:
                    raise RuntimeError(f'{h["running"]} requests still running after {s["drain"]:.0f}s (--drain-timeout)')
                if self.stop_ev.is_set() or s['phase'] != 'drain':
                    return
                time.sleep(0.1)
            self.mark('drained')
            s['drain_s'] = round(time.time() - s['held'], 1)
            s['phase'] = 'switch'
            log(f'scale {s["id"]}: drained in {s["drain_s"]}s; switching')
            for a in s['old_members']:
                if a != self.lan:
                    self.call(a, {'t': 'scale_go', 'id': s['id']})
            self.scale_go.set()
        except Exception as e:   # noqa: BLE001 - a scale that fails before the switch changes nothing
            self.scale_abort_all(s, str(e) or type(e).__name__)

    def scale_layout(self, s):
        """GLM-5.3 / Qwen3.8-Flash-Next: the new members' layout planned from the running one (bin/g6layout plan), so
        each box keeps the expert halves it holds and fetches only what it lacks. None: the new width's formula."""
        if models.of(self.o) not in (models.GLM53, models.QW38) or not (self.variant and self.variant.path):
            return None
        d = os.path.join(self.out, 'scale')
        os.makedirs(d, exist_ok=True)
        try:
            g = os.path.join(os.path.dirname(hfsource.fetch_ranks().g6load()), 'g6layout')
            src = self.o.get('LAYOUT')
            run = [] if src else [['formula', '--members', ','.join(s['old_members']), '--out', os.path.join(d, 'from.json')]]
            src = src or os.path.join(d, 'from.json')
            out = os.path.join(d, 'plan.json')
            for argv in run + [['plan', '--from', src, '--members', ','.join(s['members']), '--out', out]]:
                r = subprocess.run([g, *argv, '--widths', self.variant.path], capture_output=True, text=True, timeout=120)
                if r.returncode:
                    raise RuntimeError(f'g6layout {argv[0]}: {(r.stderr or r.stdout).strip()[-300:]}')
            with open(out) as f:
                return f.read()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
            log(f'scale {s["id"]}: no planned layout ({e}); the new width takes its formula')
            return None

    def scale_pending(self, m, joiner=False):
        """This box's part in scale m: its token and layout files, the options it boots with (this boot's plus the
        scale's), its rank in the new and the old member list, the new rank's model and rails, and the KV store the
        engine rewrites (GR8_KV_REWIDTH). Raises OptionError."""
        d = os.path.join(self.out, 'scale')
        os.makedirs(d, exist_ok=True)
        token = os.path.join(d, 'token')
        fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, (m['token'] + '\n').encode())
        os.close(fd)
        extra = ['--layout', layout_file(d, m['layout'])] if m.get('layout') else []
        o = opt.parse(scale_argv(self.o.argv, scale_drops(self.pool, m['argv'])) + list(m['argv']) + extra)
        opt.validate(o)
        members, old = list(m['members']), list(m['old_members'])
        rank = members.index(self.lan) if self.lan in members else None
        old_rank = old.index(self.lan) if self.lan in old else None
        p = {'id': m['id'], 'o': o, 'members': members, 'old_members': old, 'rank': rank, 'old_rank': old_rank,
             'joiner': joiner, 'leaves': rank is None, 'switched': False, 'ms': None, 'variant': None, 'model_id': None,
             'model_error': '', 'rails': None, 'fabric_error': '', 'build': None, 'store': None, 'rewidth': None}
        if rank is not None:
            p['ms'], p['variant'], p['model_error'] = resolve_model(o)
            if p['model_error']:
                p['ms'] = None
            else:
                p['model_id'] = modelset.model_id(p['ms'], p['variant'])
            p['rails'], p['fabric_error'] = self.rails_for(o, rank)
            p['build'] = engine_build(self.home, engine_bin(o))
        root = scale_store(self.o)
        if root:
            p['store'] = os.path.join(self.home, root)
            p['rewidth'] = f'{len(old)},{"x" if old_rank is None else old_rank},{token},{m["old_set"]},' + \
                ','.join(f'{a}:{self.port + 3}' for a in old)
        self.scale_go.clear()
        self.scale_abort.clear()
        self.scale_done.clear()
        self.pending = p
        return p

    def scale_prepare(self, m):
        """An old member's part before the switch, while its engine serves: its new rank's files made ready (fetched when
        missing) and checked. (ok, why)."""
        try:
            p = self.scale_pending(m)
            if p['leaves']:
                return True, 'leaves'
            fails = self.checks(p=p)
            if not fails:
                self.scale_compact_prepare(p)
            return not fails, '; '.join(fails)
        except Exception as e:   # noqa: BLE001 - the head hears why
            return False, f'{type(e).__name__}: {e}'

    def join_scale(self, v):
        """A box the head's scale adds: it boots with the scale's options, makes its files ready, tells the head, and
        says hello again; the head admits it to the new member list's handshake. Returns its failed checks."""
        log(f'scale {v["id"]}: joining {",".join(v["members"])}')
        try:
            p = self.scale_pending(v, joiner=True)
        except opt.OptionError as e:
            self.tell(0, {'t': 'scale_ready', 'id': v['id'], 'lan': self.lan, 'ok': False, 'why': f'options: {e}'})
            raise Stop(2, f'scale options: {e}')
        if p['rank'] is None:
            raise Stop(1, f'scale {v["id"]}: {self.lan} is not in {",".join(p["members"])}')
        self.hello_poll = 0.1
        self.swap_in(p)
        fails = self.checks()
        if not fails:
            self.scale_compact_prepare(p)
        self.tell(0, {'t': 'scale_ready', 'id': p['id'], 'lan': self.lan, 'ok': not fails, 'why': '; '.join(fails)})
        if fails:
            raise Stop(4, 'scale: this box failed its checks: ' + '; '.join(fails))
        return fails

    # ---- POOL: standbys -------------------------------------------------------------------------------------------
    def names(self, fn):
        """fn(), a resolution of names. With POOL, again every 2 s while it fails (a pod's name resolves once the pod
        exists), until JOIN_TIMEOUT (0: forever) or a stop; then Stop(2). Without POOL a failure raises at once."""
        end, said = time.monotonic() + opt_timeout(self.o), False
        while True:
            try:
                return fn()
            except opt.OptionError as e:
                if not self.o.get('POOL'):
                    raise
                if not said:
                    log(f'waiting for the pool\'s names to resolve: {e}')
                    said = True
                if self.stop_ev.wait(2) or (opt_timeout(self.o) and time.monotonic() > end):
                    raise Stop(2, str(e))

    def standby(self):
        """A box of POOL outside the member list: no engine, no handshake. It asks the head every 2 s for a scale that
        names it (it joins as join_scale does) or for the running member list, which it takes when it holds this box
        (adopt). Returns the failed checks of the boot it joins."""
        self.rank, self.hs = None, {}
        try:
            os.unlink(os.path.join(self.out, 'handshake.json'))
        except FileNotFoundError:
            pass
        self.set_state('standby')
        skip = None
        while True:
            if self.stop_ev.is_set():
                raise Stop(1 if self.stop_failed else 0, self.stop_why)
            try:
                v = handshake.call(self.pool[0], self.port, {'t': 'pool'}, timeout=5, bind=self.lan)
            except (OSError, ValueError):
                v = {}
            if v.get('t') == 'scale' and v.get('id') != skip:
                try:
                    return self.join_scale(v)
                except Stop as e:
                    if self.stop_ev.is_set():
                        raise
                    log(f'scale {v["id"]}: {e}; this box stays a standby')
                    self.pending, self.rank, skip = None, None, v['id']
                    self.set_state('standby')
            elif v.get('t') == 'adopt' and self.lan in (v.get('members') or []):
                return self.adopt(v)
            self.stop_ev.wait(2)

    def adopt(self, v):
        """The head's running member list and options (adopt_msg): this box boots its rank in them, or stands by when it
        has none. Returns the failed checks."""
        p = self.scale_pending({'id': 'adopt', 'argv': v['argv'], 'members': v['members'], 'old_members': [],
                                'layout': v.get('layout'), 'token': '', 'old_set': ''}, joiner=True)
        self.pending = None
        if p['rank'] is None:
            return self.standby()
        log(f'the head runs TP={len(p["members"])} {",".join(p["members"])}: this box takes rank {p["rank"]}')
        self.swap_in(p)
        return self.checks()

    def adopt_msg(self):
        return {'t': 'adopt', 'argv': self.scale_words, 'members': self.members, 'layout': self.layout_text}

    def on_pool(self, peer):
        """A standby asks the pool's head what to do: join the scale that names it, wait, or take the member list."""
        if self.rank != 0 or not self.pool:
            return {'t': 'error', 'error': f'{self.lan} is not the head of a pool'}
        s = self.scale
        if s and s['phase'] in ('prepare', 'drain') and s.get('msg') and peer in s['members'] and \
                peer not in s['old_members'] and peer not in s['ready']:
            return {'t': 'scale', **s['msg']}
        if s and (s['phase'] in ('prepare', 'drain', 'switch') or (s['phase'] == 'failed' and not s.get('closed'))):
            return {'t': 'retry', 'detail': f'scale {s["id"]} is running ({s["phase"]})'}
        return self.adopt_msg()

    def scale_compact_prepare(self, p):
        """GR8_LOAD_COMPACT: this box writes its new rank's compact copies (scale_compact names the directory) while the
        old width serves. Without them the new engine reads the release files, as it would anyway."""
        words = (p['o'].get('EXTRA_ENV') or '').split()
        i = compact_word(words)
        if i is None or not words[i].partition('=')[2]:   # only the scale's own knob: the engine reads no other
            return
        try:
            done = prepare_compact(p['o'], p['rank'], p['variant'])
        except (modelset.ModelError, OSError, RuntimeError, ValueError) as e:
            first = (str(e).splitlines() or [type(e).__name__])[0]
            log(f'scale {p["id"]}: warning: no compact copies for the new width ({first}); it reads the release files')
            return
        if done:
            log(f'scale {p["id"]}: compact copies: {done[1]["checked"]} halves in {done[0]}')

    def swap_in(self, p):
        """This box takes the scale's options, members and rank (peers mode from now on)."""
        self.o, self.members, self.rank, self.rails = p['o'], p['members'], p['rank'], p['rails']
        self.ms, self.variant, self.model_id = p['ms'], p['variant'], p['model_id']
        self.model_error, self.fabric_error, self.elected = p['model_error'], p['fabric_error'], None
        self.reset_boot()
        self.cluster_opts()
        try:   # a standby listened on its LAN address only; the head's rail check pings the rails
            self.serve([a for a in (self.lan, *self.rails, *self.rdma) if a])
        except Stop as e:
            log(f'warning: {e}')

    def reset_boot(self):
        """Forget the last boot's handshake, as a boot that starts again does."""
        with self.state_lock:
            self.head, self.hs = None, {}
            self.fallback_by_epoch.clear()
            self.serving_urls.clear()
            self.boot_attempt, self.dispram_off = 0, False
        self.head_ready.clear()
        self.lingered = self.refusal_logged = False

    def stop_engine(self, served=True):
        """This box's engine and shim stop (an orderly quit: the KV store keeps the pool's runs); the front end stays.
        served=False: the engine never reached its serve loop, the only place it reads s2/quit, so it gets SIGTERM at
        once (it has written nothing an orderly quit would keep)."""
        stop_local(self.home, log, quit_wait(self.o) if served else 0, keep=('front',))
        for child in (self.helpers.pop('shim', None), self.engine):
            if child is not None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        self.engine = None
        if served:   # what the quit stored and how long it took
            for line in tail(os.path.join(self.out, f'g6m4.r{self.rank}.err'), 40).splitlines():
                if 'quit s' in line or 'quit fast' in line:
                    log(line.removeprefix('# '))

    def mark(self, name):
        """The head notes when a phase of a scale's hold ended, for its timeline (scale_served)."""
        s = self.scale
        if self.rank == 0 and s and s.get('held') and not s.get('closed'):
            s['marks'].append((name, time.time()))

    def rescale(self):
        """scale_go: this box's engine quits and its KV store is kept for the new ranks (served by kvman-peer). A box
        that stays boots the new member list; one that leaves serves its store until the scale is done, then exits.
        Returns the failed checks."""
        p = self.pending
        p['switched'] = True
        new = f'TP={p["o"].get("TP")} {",".join(p["members"])}'
        log(f'scale {p["id"]}: stopping the engine for {new}')
        self.set_state('switching')
        self.stop_engine()
        self.mark('stopped')
        if p['store']:
            p['dir'] = store_prep(p['store'], p['old_rank'], p['rank'], models.of(self.o).checkpoint_cache)
            self.start_peer(p['dir'], os.path.join(self.out, 'scale', 'token'))
        if p['leaves']:
            log(f'scale {p["id"]}: this box leaves; its KV store serves the new ranks until the scale is done')
            wait = opt_timeout(self.o) + float(self.tmpl['knobs'].get('READY_TMO') or 1800)
            end = time.monotonic() + wait
            while not self.scale_done.wait(0.5):
                if self.scale_abort.is_set():
                    return self.rollback()
                if self.stop_ev.is_set():
                    raise Stop(1 if self.stop_failed else 0, self.stop_why)
                if time.monotonic() > end:   # the head is gone, or its scale_done never came
                    self.stop_peer()
                    if p.get('dir'):
                        store_undo(p['store'], p['old_rank'], p['rank'], models.of(self.o).checkpoint_cache)
                    self.pending = None
                    raise Stop(1, f'scale {p["id"]}: no word from the head within {wait:.0f}s (JOIN_TIMEOUT + '
                                  'READY_TMO); this box stops')
            self.stop_peer()
            if p.get('dir'):
                store_undo(p['store'], p['old_rank'], p['rank'], models.of(self.o).checkpoint_cache)
            self.pending = None
            if self.pool:
                self.reset_boot()
                self.scale_go.clear()
                self.scale_done.clear()
                log(f'scale {p["id"]}: this box left the member list; it stays in the pool as a standby')
                return self.standby()
            raise Stop(0, f'scale {p["id"]}: this box left the cluster ({new} serves)')
        self.hello_poll = 0.1
        self.prev = {'o': self.o, 'members': self.members, 'rank': self.rank, 'rails': self.rails, 'ms': self.ms,
                     'variant': self.variant, 'model_id': self.model_id, 'model_error': self.model_error,
                     'fabric_error': self.fabric_error, 'elected': self.elected}
        self.swap_in(p)
        return self.checks(copy=False)

    def rollback(self):
        """scale_abort after the switch: whatever the new member list started here stops, the KV store goes back as it
        was, and this box boots the old member list. Returns the failed checks."""
        p = self.pending
        log(f'scale {p["id"]}: rolling back to TP={(self.prev or {"o": self.o})["o"].get("TP")}')
        self.set_state('rolling back')
        self.stop_engine(served=False)
        self.mark('stopped')
        self.hello_poll = 0.1
        self.stop_peer()
        if p.get('dir'):
            store_undo(p['store'], p['old_rank'], p['rank'], models.of(self.o).checkpoint_cache)
        self.reset_boot()   # head_ready clears before self.members is the old list again: its hellos wait (on_rpc)
        if self.prev:
            for k, v in self.prev.items():
                setattr(self, k, v)
        self.pending = self.prev = None
        self.scale_go.clear()
        self.scale_abort.clear()
        if p['joiner']:   # a standby that joined (a joiner from `up` stops instead: on_scale_abort)
            return self.standby()
        self.cluster_opts()
        return self.checks(copy=False)

    def on_scale_abort(self, sid, why):
        """scale_abort: a joiner stops (a standby that joined stands by again: watch raises Rollback); an old member that
        switched rolls back (watch raises Rollback); before the switch nothing changed."""
        p = self.pending
        if not p or p['id'] != sid:
            return
        if p['joiner'] and not self.pool:
            self.request_stop(f'scale {sid} was aborted: {why}', failed=True)
        elif p['joiner'] or p['switched']:
            self.scale_abort.set()
        else:
            log(f'scale {sid}: aborted before the switch ({why}); nothing changed here')
            self.pending = None
            self.scale_go.clear()

    def scale_abort_all(self, s, why):
        """The head ends scale s: every box of both member lists hears it, and the hold ends once a member list serves
        again (now, before the switch)."""
        with self.state_lock:
            if s['phase'] in ('done', 'failed'):
                return
            switched = s['phase'] == 'switch'
            s['phase'], s['error'] = 'failed', why
        log(f'scale {s["id"]}: failed: {why}; ' + ('the old member list boots again' if switched else 'nothing changed'))
        if switched:
            self.mark('failed')
        if switched and s.get('held'):   # the old member list's boot: the hold lasts it out
            try:
                self.front_hold(s.get('boot_s', 900))
            except OSError as e:
                log(f'scale {s["id"]}: the front end did not take the longer hold: {e}')
        if not switched:
            s['closed'] = True
            if s.get('held'):
                self.release_hold(s)
        for a in sorted((set(s['old_members']) | set(s['members'])) - {self.lan}):
            self.call(a, {'t': 'scale_abort', 'id': s['id'], 'why': why})
        self.on_scale_abort(s['id'], why)

    def scale_served(self):
        """The head serves after a scale's switch: the new member list (done: the old one's leftovers go) or, after a
        rollback, the old one again. Either way the front end's hold ends."""
        s = self.scale
        if not s or s.get('closed'):
            return
        self.mark('serving')
        s['closed'] = True
        self.release_hold(s)
        prev, parts = s.get('held'), []
        for name, t in s['marks'] if prev else ():
            parts.append(f'{name} +{t - prev:.1f}s')
            prev = t
        s['timeline'] = ', '.join(parts)
        log(f'scale {s["id"]}: timeline: {s["timeline"]}')
        if s['phase'] == 'switch':
            self.scale_words = s['argv']
            s['phase'] = 'done'
            config = {opt.KEY_OF_FLAG[f]: v for f, v in zip(s['argv'][::2], s['argv'][1::2]) if f in opt.KEY_OF_FLAG}
            s['config'] = [f'{k}={v}' for k, v in config.items()]
            if self.pool:   # the width this head boots at its next up, while TP and POOL stay (pool_resume)
                self.layout_text = s['msg'].get('layout')
                keep = os.path.join(self.out, 'scale.json')
                with open(keep + '.tmp', 'w') as f:
                    json.dump({'argv': s['argv'], 'layout': self.layout_text, 'base': self.base}, f)
                os.replace(keep + '.tmp', keep)
            for a in sorted((set(s['old_members']) | set(s['members'])) - {self.lan}):
                threading.Thread(target=self.send_done, args=(a, s['id']), daemon=True).start()
            self.finish_scale()
            log(f'scale {s["id"]}: done: TP={self.o.get("TP")} {",".join(self.members)} serves; requests waited '
                f'{s["pause_s"]}s (drain {s.get("drain_s")}s)')
        else:
            log(f'scale {s["id"]}: the old member list serves again; requests waited {s["pause_s"]}s')

    def send_done(self, a, sid, tries=5):
        """scale_done to box a, again until it answers: a box that misses it keeps its old store and kvman-peer (and a
        box that leaves would wait for it until its deadline)."""
        for i in range(tries):
            r = self.call(a, {'t': 'scale_done', 'id': sid})
            if r and r.get('t') == 'ok':
                return True
            time.sleep(1 + i)
        log(f'scale {sid}: {a} did not take scale_done after {tries} tries')
        return False

    def release_hold(self, s):
        try:
            self.front_hold(0)
        except OSError as e:
            log(f'scale {s["id"]}: the front end did not take the end of the hold: {e}')
        s['pause_s'] = round(time.time() - s['held'], 1) if s.get('held') else 0.0

    def finish_scale(self):
        """scale_done: the new member list serves. The old width's store files and the store service go, and so do the
        compact copies of earlier widths (scale_compact's names; the operator's own directory stays)."""
        p = self.pending
        if not p:
            return
        self.stop_peer()
        if p.get('dir'):
            store_commit(p['dir'], models.of(self.o).checkpoint_cache)
        self.pending = self.prev = None
        for e in (self.scale_go, self.scale_abort, self.scale_done):
            e.clear()
        words = (self.o.get('EXTRA_ENV') or '').split()
        i = compact_word(words)
        cur = os.path.join(self.home, words[i].partition('=')[2]) if i is not None else ''   # the engine runs in home
        if COMPACT_SUFFIX.search(cur):
            for d in glob.glob(glob.escape(COMPACT_SUFFIX.sub('', cur)) + '.tp*-*'):
                if d != cur and COMPACT_SUFFIX.search(d) and os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                    log(f'scale: removed the compact copies of an earlier width, {d}')

    def front_hold(self, seconds=None):
        """The head's front end's request hold (/internal/hold): set for `seconds` (0 ends it), or read. Its JSON."""
        q = '' if seconds is None else f'?seconds={int(seconds)}'
        req = urllib.request.Request(f'http://127.0.0.1:{self.o.get("PORT")}/internal/hold{q}',
                                     data=None if seconds is None else b'', method='GET' if seconds is None else 'POST')
        with NO_PROXY.open(req, timeout=5) as f:
            return json.load(f)

    def start_peer(self, d, token):
        """bin/kvman-peer serves this box's KV store folder to the new ranks, on CONTROL_PORT + 3."""
        self.stop_peer()   # one from a scale whose scale_done never came holds the port
        b = os.path.join(self.home, 'bin', 'kvman-peer')
        if not os.access(b, os.X_OK):
            log(f'scale: warning: no {b}: the new ranks start their KV stores empty (the stored sessions prefill again)')
            return
        listen = f'{self.lan}:{self.port + 3}'
        path = os.path.join(self.out, 'kvman-peer.log')
        start = os.path.getsize(path) if os.path.exists(path) else 0
        with open(path, 'ab') as lf:
            self.peer_proc = subprocess.Popen([b, '--dir', d, '--listen', listen, '--token-file', token], stdout=lf,
                                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        end = time.monotonic() + 5
        while time.monotonic() < end and self.peer_proc.poll() is None:   # serving, or exited (a port in use)
            with open(path, 'rb') as f:
                f.seek(start)
                if b'serving' in f.read():
                    break
            time.sleep(0.05)
        if self.peer_proc.poll() is not None:
            with open(path, 'rb') as f:
                f.seek(start)
                why = f.read().decode(errors='replace').strip().splitlines()[-1:] or ['no output']
            log(f'scale: warning: kvman-peer exited ({exit_status(self.peer_proc.returncode)}: {why[0]}): the new ranks '
                'start their KV stores empty (the stored sessions prefill again)')
            self.peer_proc = None
            return
        log(f'scale: {d} served to the new ranks on {listen} (kvman-peer pid {self.peer_proc.pid})')

    def stop_peer(self):
        if self.peer_proc:
            self.peer_proc.terminate()
            try:
                self.peer_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.peer_proc.kill()
                self.peer_proc.wait()
            self.peer_proc = None

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
        if self.donor:
            subprocess.run([sys.executable, kvnode.HERE, 'down', '--work-dir', self.home], capture_output=True,
                           timeout=60, stdin=subprocess.DEVNULL)
            self.donor = False
        if self.started:
            stop_local(self.home, log, quit_wait(self.o))
        self.stop_peer()
        self.release_claim()
        if self.agent:
            self.agent.stop()
        if self.api_agent:
            self.api_agent.stop()
        if not code:
            self.set_state('stopped')
        for s in self.servers:
            s.shutdown()


def gpu_lendable(ledger):
    """The bytes of KV chunks the Flash engine may lend (it writes LEDGER.gpu at boot under borrow-idle), or 0."""
    try:
        with open(ledger + '.gpu') as f:
            v = f.read().strip()
    except OSError:
        return 0
    return int(v) if v.isdigit() else 0


def ledger_bin(home):
    for p in (os.path.join(home, 'bin', 'kvman-ledger'),
              os.path.join(opt.REPO, 'engine', 'target', 'release', 'kvman-ledger')):
        if os.access(p, os.X_OK):
            return p
    return None


def fabric_needed(o):
    """Whether the ranks talk over RDMA: a model with a fabric, on more than one box."""
    return models.of(o).fabric and int(o.get('TP') or 0) > 1


def engine_bin(o):
    """The model's engine binary under bin/ for KV_FORMAT: g6m4 (the fp8 KV row) or g6m4-int4 (docs/KV4.md) for
    GLM-5.3, g6m4_flash for GLM-5.3-Flash."""
    name = models.of(o).engine
    return name + '-int4' if o.get('KV_FORMAT') == 'int4' else name


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


# The engine's boot-stage lines on stderr: "# rank R: boot 41.3 s <phase>" at each phase of the boot (connect, weights, graph
# capture), and the loader's "# rank R: pack 72.1 GB loaded + sha256-checked in 190.2 s (0.38 GB/s)" / "... -> device image".
BOOT_STAGE = re.compile(r'^# rank (\d+): (boot \d+(?:\.\d+)? s .*|pack \d.*|.* -> device image .*)$')
# A scale's boot also relays the store rewrite, the weight check, the boot capture and the load caches: the hold's parts.
SCALE_STAGE = re.compile(BOOT_STAGE.pattern + r'|^# rank \d+: .*(?: rewidth |weight check: |entries verified).*$'
                         r'|^# (?:s2 boot capture|flash serve capture|gen-cache|load-compact): .*$')


class StageRelay:
    """Reads a rank's engine stderr as it grows and returns the boot-stage lines, so the supervisor log shows each rank's
    breakdown while the engine loads (startup first: where the minutes go, per rank)."""

    def __init__(self, path, pattern=BOOT_STAGE):
        self.path, self.pos, self.partial, self.pattern = path, 0, b'', pattern

    def new(self):
        try:
            with open(self.path, 'rb') as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return []
        self.pos += len(data)
        *whole, self.partial = (self.partial + data).split(b'\n')
        return [m.group(0)[2:] for m in (self.pattern.match(ln.decode(errors='replace')) for ln in whole) if m]


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


def quit_wait(o):
    """Seconds an engine gets for its orderly quit before SIGTERM: 120 with a persist=on ssd store that keeps sessions
    (KV_STORE keep=sessions or all, or KV_KEEP): the quit stores the pool's runs there for the next boot. Else 10.
    The engine also needs --set-id for that store's index; the launcher always passes one (modelset.py set_id)."""
    try:
        stores = opt.kv_stores(o)
    except Exception:
        return 10
    keep = o.get('KV_KEEP') or 'system'
    whole = any(t['kind'] == 'ssd' and t.get('persist') and (t.get('keep') or keep) != 'system' for t in stores)
    return 120 if whole else 10


def recorded_quit_wait(out, o):
    """The quit wait the running supervisor recorded at its start (out/quit_wait), else quit_wait(o)."""
    try:
        with open(os.path.join(out, 'quit_wait')) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return quit_wait(o)


def stop_local(home, log=log, quit_s=10, keep=()):
    """Stop this box: shim and front end, then quit files, then the engine (quit_s seconds for an orderly quit). keep:
    the helpers to leave running (a scale keeps the front end)."""
    out = os.path.join(home, 'out')
    for p in ('front', 'shim'):
        if p in keep:
            continue
        pid = recorded(os.path.join(out, f'{p}.pid'))
        if pid:
            signal_pid(pid, signal.SIGTERM, p, log)
    pid = recorded(os.path.join(out, 'g6m4.pid'))
    if not pid:
        return False
    if os.path.isdir(os.path.join(home, 's2')):
        open(os.path.join(home, 's2', 'quit'), 'w').close()
    for _ in range(int(quit_s * 10)):
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


# The front end's hold, read and set from this box itself: a proxy the environment names must not take it.
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Either:
    """Set when any of the events is: what a handshake stops on (a stop, or a scale's abort)."""

    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(e.is_set() for e in self.events)


# what an up command line names per member of its list: a scale gives the new list's own (or none)
SCALE_DROPS = ('MENTAT', 'IPS', 'IPS2')


def names_peers(words):
    """Whether a command line's words name PEERS."""
    return any(w.partition('=')[0] in (opt.flag_of('PEERS'), 'PEERS') for w in words)


def scale_drops(pool, words):
    """What a scale drops from the earlier command lines: SCALE_DROPS, and in a pool PEERS too unless the scale's words
    name it (the members are then POOL's first TP)."""
    return SCALE_DROPS + ('PEERS',) if pool and not names_peers(words) else SCALE_DROPS


def scale_argv(argv, drops=SCALE_DROPS):
    """An up command line without --mentat (a scale names its members with --peers, and the cluster runs in peers mode
    from then on) and without --ips/--ips2, which list the old members' rails: the scale command gives the new ones."""
    flags = {opt.flag_of(k) for k in drops}
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in flags:
            i += 2
            continue
        if not (a.partition('=')[0] in flags or a.partition('=')[0] in drops):
            out.append(a)
        i += 1
    return out


def scale_pins(o, variant, knobs):
    """What a scale keeps from the running boot: the variant, the KV format (a TP default such as GLM-5.3's int4 at TP=4
    would otherwise resolve again at the new width, and another format is another engine and KV row, so the store would
    not carry), SHARE_LIMIT and REUSE_GRID (the store's R and grid)."""
    pins = ['--variant', o.get('VARIANT') or variant.name] if variant else []
    pins += ['--kv-format', o.get('KV_FORMAT') or 'fp8']
    if knobs.get('SHARE_LIMIT'):
        pins += ['--share-limit', knobs['SHARE_LIMIT']]
    if knobs.get('REUSE_GRID') not in (None, '', '0'):
        pins += ['--reuse-grid', knobs['REUSE_GRID']]
    return pins


def scale_fit(model, variant, tp):
    """'' when the running variant fits a rank at TP=tp, else why: it must be no larger than the TP's default variants
    (models.py default_variant), by bits for a named variant and, for a manifest path, by the experts on the busiest
    rank against the defaults' manifests beside it (none found: not checked)."""
    defaults = model.default_variant.get(int(tp), ())
    if not variant or not defaults or variant.name == 'home':
        return ''
    if variant.name in model.variants:
        top = max(model.variants[d] for d in defaults)
        if model.variants[variant.name] <= top:
            return ''
        size = f'{model.variants[variant.name]} bits, over the {top} of the TP={tp} default'
    else:
        if not variant.path:
            return ''
        d = os.path.dirname(variant.path)
        found = [p for n in defaults for p in (os.path.join(d, f'exlr8-manifest-{n}.json'),
                                               os.path.join(d, 'manifests', f'exlr8-manifest-{n}.json'))
                 if os.path.isfile(p)]
        if not found:
            return ''
        try:
            have, top = modelset.expert_gb(variant.path, int(tp)), max(modelset.expert_gb(p, int(tp)) for p in found)
        except (OSError, ValueError, KeyError) as e:
            return f'--variant {variant.name}: cannot size it at TP={tp}: {e}'
        if have <= top + 0.05:
            return ''
        size = f'{have:.1f} GB of experts on the busiest rank, over the {top:.1f} GB of the TP={tp} default'
    return (f'--variant {variant.name} does not fit a rank at TP={tp} ({size}): boot the cluster with --variant '
            f'{defaults[0]} (or one no larger) before it scales to TP={tp}')


def scale_store(o):
    """The persisted ssd store a scale carries to the new ranks (the engine's GR8_KV_REWIDTH): GLM-5.3's first
    persist=on ssd store, GLM-5.3-Flash's checkpoint store (its first store) when it is one. Its path, or None."""
    stores = opt.kv_stores(o)
    if models.of(o).checkpoint_cache:
        stores = stores[:1]
    return next((t['path'] for t in stores if t['kind'] == 'ssd' and t['persist']), None)


def _env_kv(extra_env):
    """Parse KEY=VAL words from EXTRA_ENV into a dict."""
    out = {}
    for w in (extra_env or '').split():
        if '=' in w:
            k, _, v = w.partition('=')
            out[k] = v
    return out


def live_switch_capable(o, model):
    """Whether a running boot supports Phase B in-process live width switching (`./gr8 <model> scale --live`, #156).
    Requires all three Phase B dynamic primitives enabled in EXTRA_ENV:
      - GR8_FABRIC_POOL=1 (reconfigurable fabric groups, #154)
      - GR8_PTAB>=1 (chunked weight pool, #151)
      - GR8_KV_CHUNKS=1 (GLM) or GR8_GLM53F_KV_CHUNKS=1 (Flash) (resizable VMM KV pool, #153)
    Returns (ok, reason)."""
    kv = _env_kv(o.get('EXTRA_ENV'))
    missing = []
    if kv.get('GR8_FABRIC_POOL', '0') in ('', '0'):
        missing.append('GR8_FABRIC_POOL=1')
    ptab = kv.get('GR8_PTAB', '0')
    if not ptab.isdigit() or int(ptab) < 1:
        missing.append('GR8_PTAB>=1')
    kv_key = 'GR8_GLM53F_KV_CHUNKS' if model.build == 'flash' else 'GR8_KV_CHUNKS'   # build, not name: Flash is glm53f
    if kv.get(kv_key, '0') in ('', '0'):
        missing.append(f'{kv_key}=1')
    if missing:
        return False, 'missing ' + ', '.join(missing)
    return True, ''


def layout_file(d, text):
    """d/layout-<sha256 of text>.json, written once (tmp + rename): the engine's gen cache keys on the layout file's path,
    size, mtime and inode, so a scale that plans a layout seen before keeps its GR8_GEN_CACHE entries. Its path."""
    p = os.path.join(d, f'layout-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json')
    if not os.path.exists(p):
        with open(p + '.tmp', 'w') as f:
            f.write(text)
        os.replace(p + '.tmp', p)
    return p


# a compact directory scale_compact named for one width: <the operator's directory>.tp<N>-<tag>
COMPACT_SUFFIX = re.compile(r'\.tp\d+-[0-9a-f]{8}$')
COMPACT_KEYS = ('GR8_LOAD_COMPACT', 'G6_LOAD_COMPACT')


def compact_word(words):
    """The index of the last GR8_LOAD_COMPACT= (or G6_LOAD_COMPACT=) word of EXTRA_ENV's words, the one engine_args.sh
    applies, or None."""
    return next((i for i in range(len(words) - 1, -1, -1)
                 if words[i].partition('=')[1] and words[i].partition('=')[0] in COMPACT_KEYS), None)


def scale_compact(extra_env, tp, variant, members, layout):
    """The scale's --extra-env with GR8_LOAD_COMPACT renamed for the new width: <dir>.tp<N>-<tag>, tag from the variant,
    the members and the layout (the halves each rank holds), the same path on every box. [] without the knob (or with
    it empty)."""
    words = (extra_env or '').split()
    i = compact_word(words)
    if i is None or not words[i].partition('=')[2]:
        return []
    key, _, d = words[i].partition('=')
    tag = hashlib.sha256(f'{variant}\n{",".join(members)}\n{layout or ""}'.encode()).hexdigest()[:8]
    words[i] = f'{key}={COMPACT_SUFFIX.sub("", d)}.tp{tp}-{tag}'
    return [opt.flag_of('EXTRA_ENV'), ' '.join(words)]


def prepare_compact(o, rank, variant):
    """An opt-in prepare step, never a boot-time rewrite: this rank's compact copies in GR8_LOAD_COMPACT, the same
    local directory the engine reads. (directory, g6load's result), or None when no directory is named."""
    directory = os.environ.get('GR8_LOAD_COMPACT', os.environ.get('G6_LOAD_COMPACT', ''))
    # engine_args.sh splits EXTRA_ENV on whitespace, normalizes legacy names and applies the last assignment.
    for word in (o.get('EXTRA_ENV') or '').split():
        key, sep, value = word.partition('=')
        if sep and key in COMPACT_KEYS:
            directory = value
    if not directory:
        return None
    if not variant.path or not o.get('V2_SLICES'):
        raise modelset.ModelError('GR8_LOAD_COMPACT prepare needs an exlr8 v2 release with index slices and a width manifest')
    import fetch_ranks
    return directory, fetch_ranks.compact_rank(o.get('V2'), directory, o.get('V2_SLICES'), int(o.get('TP')), rank,
                                               variant.path, o.get('LAYOUT'))


# GLM-5.3-Flash's checkpoint store files (engine/runtime/s2/flash_ckpt_index.h); GLM-5.3's store keeps `blocks` in
# place and its `index` as index.prev
FLASH_FILES = ('flash-ckpt', 'flash-ckpt.index', 'flash-ckpt.tokens')


def rm(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def store_folders(root, old_rank, new_rank):
    """(this box's store folder at its old rank, at its new one): root/r<rank>, as the engine names them. A box that
    leaves keeps its old folder."""
    old = os.path.join(root, f'r{old_rank}')
    return old, old if new_rank is None else os.path.join(root, f'r{new_rank}')


def store_prep(root, old_rank, new_rank, flash):
    """At the switch: this box's store folder follows its rank, and the old width's files become *.prev for the new
    engine to rewrite (GLM-5.3 copies its index; Flash renames its checkpoint files). Returns the folder."""
    old, d = store_folders(root, old_rank, new_rank)
    if d != old:
        rm(d)                        # a stale folder of an earlier boot at this rank
        if os.path.isdir(old):
            os.rename(old, d)
    os.makedirs(d, exist_ok=True)
    if flash:
        for n in FLASH_FILES:
            p = os.path.join(d, n)
            if os.path.exists(p):
                rm(p + '.prev')
                os.rename(p, p + '.prev')
    elif os.path.exists(os.path.join(d, 'index')):
        shutil.copy2(os.path.join(d, 'index'), os.path.join(d, 'index.prev'))
    return d


def store_undo(root, old_rank, new_rank, flash):
    """A rollback (or a box that left): store_prep undone, so the old member list boots its store as it was."""
    old, d = store_folders(root, old_rank, new_rank)
    if flash:
        for n in FLASH_FILES:
            p = os.path.join(d, n)
            if os.path.exists(p + '.prev'):
                rm(p)
                os.rename(p + '.prev', p)
    elif os.path.exists(os.path.join(d, 'index.prev')):
        os.replace(os.path.join(d, 'index.prev'), os.path.join(d, 'index'))
    if d != old and os.path.isdir(d):
        rm(old)
        os.rename(d, old)


def store_commit(d, flash):
    """The scale is done: the old width's files go."""
    for n in (FLASH_FILES if flash else ('index',)):
        rm(os.path.join(d, n + '.prev'))


def pool_resume(o):
    """A pool's head boots the width of its last scale (out/scale.json, scale_served) while the up's TP and POOL are
    the ones that scale ran under: (options, record), or (o, None). A record for other TP or POOL goes."""
    if not o.get('POOL'):
        return o, None
    out = os.path.join(os.path.abspath(o.get('GR8_WORK_DIR')), 'out')
    path = os.path.join(out, 'scale.json')
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return o, None
    if rec.get('base') != [o.get('TP'), o.get('POOL')]:
        os.unlink(path)
        return o, None
    try:
        extra = []
        if rec.get('layout'):
            os.makedirs(os.path.join(out, 'scale'), exist_ok=True)
            extra = ['--layout', layout_file(os.path.join(out, 'scale'), rec['layout'])]
        n = opt.parse(scale_argv(o.argv, scale_drops(True, rec['argv'])) + rec['argv'] + extra)
        opt.validate(n)
    except (OSError, opt.OptionError) as e:
        log(f'warning: {path}: {e}; booting the configured width')
        return o, None
    return n, rec


def main(o):
    o, kept = pool_resume(o)
    sup = Supervisor(o)
    if kept:
        sup.scale_words, sup.layout_text, sup.base = kept['argv'], kept.get('layout'), kept['base']
        log(f'the pool\'s last scale left TP={o.get("TP")}: booting that width ({os.path.join(sup.out, "scale.json")}; '
            f'up with another --tp or --pool forgets it)')
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
