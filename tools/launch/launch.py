#!/usr/bin/env python3
"""./gr8 glm53 <command> for the symmetric launch: the same command on every box. Python 3 standard library only.

    up [OPTIONS] [@FILE]       start this box's supervisor in the background and follow it until the cluster serves
    down                       stop every box of the last boot (or this box alone)
    restart [OPTIONS]          down, then up
    status                     every box: its state, the engine, MemAvailable; the head's shim, front end and /health
    health                     this box alone, in one line: exit 0 when its supervisor reports the cluster serving
                               (and, on the head, the front end's /health answers 200), else 1. A readiness check for
                               a compose healthcheck or a Kubernetes exec probe, on every rank
    logs [RANK|shim|front|supervisor] [-f]   a log, from any box
    check [OPTIONS]            this box's checks
    show-config [OPTIONS]      the effective options, the cluster options the handshake compares, their digest, the
                               model set id
    prepare [--rank R] [OPTIONS]   fill --model-cache with this box's part of the model, without booting: with an
                               https:// --model the release's files for this rank (it asks the hub for the
                               revision's commit again), and the drafter
    knobs [WORD..] [OPTIONS]   every option: its flag, scope and meaning, and with GR8_MODEL set the value each TP
                               runs with these options. WORDs keep the options whose name holds one of them

OPTIONS are --key value (or KEY=value lines in a config file, or KEY in the environment); options.py lists them.
--foreground runs the supervisor in this process. Exit status: 0 serving (or done), 1 failed, 2 bad options, 4 the
handshake refused the boot.
"""
import fcntl
import http.client
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handshake  # noqa: E402
import kvnode  # noqa: E402
import models  # noqa: E402
import modelset  # noqa: E402
import options as opt  # noqa: E402
import supervisor as sv  # noqa: E402

HERE = os.path.abspath(__file__)


def parse(argv, check=True):
    try:
        o = opt.parse(argv)
        if check:
            opt.validate(o)
        return o
    except opt.OptionError as e:
        print(f'gr8: {e}', file=sys.stderr)
        sys.exit(2)


def out_dir(o):
    return os.path.join(os.path.abspath(o.get('GR8_WORK_DIR')), 'out')


def supervisor_running(out):
    """The pid of this box's supervisor, or None (it holds out/supervisor.lock while it runs)."""
    try:
        with open(os.path.join(out, 'supervisor.lock'), 'a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return sv.read_pid(os.path.join(out, 'supervisor.pid')) or -1
            fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        pass
    return None


def read_state(out):
    try:
        with open(os.path.join(out, 'gr8.state')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_handshake(out):
    try:
        with open(os.path.join(out, 'handshake.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cmd_up(argv):
    o = parse(argv)
    out = out_dir(o)
    os.makedirs(out, exist_ok=True)
    pid = supervisor_running(out)
    if pid:
        print(f'gr8: a supervisor already runs on this box (pid {pid}): ./gr8 glm53 down first', file=sys.stderr)
        return 1
    if '--foreground' in o.switches:
        sv.main(o)
    logp = os.path.join(out, 'supervisor.log')
    start = os.path.getsize(logp) if os.path.exists(logp) else 0
    with open(logp, 'ab') as lf:
        child = subprocess.Popen([sys.executable, HERE, 'supervise', *argv], stdout=lf, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    print(f'supervisor pid {child.pid}, log {logp}', flush=True)
    try:
        with open(logp, 'rb') as f:
            f.seek(start)
            while True:
                data = f.read()
                if data:
                    sys.stdout.write(data.decode(errors='replace'))
                    sys.stdout.flush()
                rc = child.poll()
                st = read_state(out).get('state', '')
                if rc is not None:
                    sys.stdout.write(f.read().decode(errors='replace'))
                    return rc
                if st == 'serving' and read_state(out).get('pid') == child.pid:
                    return 0
                time.sleep(0.5)
    except KeyboardInterrupt:
        print('\nthe boot goes on in the background: ./gr8 glm53 status, ./gr8 glm53 logs supervisor, ./gr8 glm53 down')
        return 130


def members_of(o):
    """(members, control port, this box's rank) from the last handshake, or None."""
    hs = read_handshake(out_dir(o))
    if not hs:
        return None
    return hs['members'], hs.get('control_port', o.int('CONTROL_PORT')), hs.get('rank')


def rpc(m, addr, msg, timeout=5):
    """A request to a member's supervisor, from this box's LAN address: the supervisors answer members only."""
    members, port, me = m
    return handshake.call(addr, port, msg, timeout=timeout, bind=members[me] if me is not None else None)


def cmd_down(argv):
    o = parse(argv, check=False)
    out, home = out_dir(o), os.path.abspath(o.get('GR8_WORK_DIR'))
    m = members_of(o)
    if m:
        members, _, me = m
        results = {}

        def one(r, addr):
            try:
                rpc(m, addr, {'t': 'down', 'why': 'gr8 glm53 down'})
                results[r] = 'stopping'
            except OSError as e:
                results[r] = f'no supervisor answers ({e})'
        ts = [threading.Thread(target=one, args=(r, a)) for r, a in enumerate(members) if r != me]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for r in sorted(results):
            print(f'rank {r} ({members[r]}): {results[r]}')
    else:
        print('no handshake on this box: stopping this box only')
    pid = supervisor_running(out)
    if pid and pid > 0:
        os.kill(pid, signal.SIGTERM)
        for _ in range(300):
            if not supervisor_running(out):
                break
            time.sleep(0.1)
        else:
            print(f'the supervisor (pid {pid}) did not stop; killing it')
            os.kill(pid, signal.SIGKILL)
    stopped = sv.stop_local(home, lambda s: print(s))
    print(f'this box: {"stopped" if pid or stopped else "nothing was running"}')
    return 0


def cmd_status(argv):
    o = parse(argv, check=False)
    out = out_dir(o)
    m = members_of(o)
    st, pid = read_state(out), supervisor_running(out)
    if not m:
        print(f'this box: supervisor {"pid " + str(pid) if pid else "not running"}, state {st.get("state", "none")}')
        return 0 if pid else 3
    members = m[0]
    for r, addr in enumerate(members):
        try:
            s = rpc(m, addr, {'t': 'status'}, timeout=3)
        except OSError as e:
            print(f'rank {r}  {addr}: no supervisor answers ({e})')
            continue
        eng = f'engine running (pid {s["engine_pid"]})' if s.get('engine') else 'engine stopped'
        line = f'rank {r}  {addr}: {s.get("state")}, {eng}, MemAvailable {s.get("mem_gib")} GiB, ' \
               f'variant {s.get("variant")}'
        if s.get('last_line'):
            line += f'\n        last log line: {s["last_line"]}'
        if r == 0:
            line += f'\n        shim {"running" if s.get("shim") else "stopped"}, ' \
                    f'front end {"running" if s.get("front") else "stopped"}'
        print(line)
    url = f'http://{members[0]}:{(read_handshake(out) or {}).get("port") or o.get("PORT")}'
    try:
        code = urllib.request.urlopen(url + '/health', timeout=3).status
    except OSError as e:
        code = getattr(e, 'code', 0)
    print(f'API     {url}/v1 ' + ('answers (/health 200)' if code == 200 else
                                  f'/health {code}: not ready' if code else 'answers nothing'))
    return 0


def cmdline(pid, proc='/proc'):
    """pid's command line as its words, or None where neither /proc nor ps can say. /proc/<pid>/cmdline on Linux;
    elsewhere (macOS) `ps -o command=`, whose words split at spaces, so a path with a space in it reads as two."""
    try:
        with open(os.path.join(proc, str(pid), 'cmdline'), 'rb') as f:
            return [a.decode(errors='replace') for a in f.read().split(b'\0') if a]
    except OSError:
        if os.path.isdir(os.path.join(proc, 'self')):
            return None     # a /proc that does not show this pid: gone, or hidden
    try:
        r = subprocess.run(['ps', '-ww', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.split() if r.returncode == 0 and r.stdout.strip() else None


def is_supervisor(pid, proc='/proc', script='launch.py'):
    """Whether pid runs a gr8 supervisor (`launch.py supervise` or `launch.py up --foreground`; a KV node's is
    kvnode.py's), from its command line. A probe cannot use sv.recorded(): its start time against the pid file's mtime
    moves when NTP steps the clock after boot, and the probe would call a serving box unready for the rest of that
    boot. Where neither /proc nor ps can say, a live pid is taken as the supervisor."""
    if not pid or not sv.alive(pid):
        return False
    argv = cmdline(pid, proc)
    if argv is None:
        return not os.path.isdir(os.path.join(proc, 'self'))
    return any(os.path.basename(a) == script for a in argv) and ('supervise' in argv or '--foreground' in argv)


# The probe asks 127.0.0.1 directly: an HTTP_PROXY that docker or a cluster sets in the container must not take it.
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health(out, port, timeout=2, proc='/proc'):
    """(ok, line): whether this box's part of the cluster serves, read from this box alone, so the same command is a
    readiness check on every rank. Serving means out/gr8.state says `serving`, the supervisor that wrote it still runs,
    and on the head (rank 0) the front end's /health answers 200. It takes no lock: a probe must not make a supervisor
    that starts beside it think another one runs. Under load the front end can answer /health late: a probe that
    times out (2 s) calls the head unready while it serves."""
    st = read_state(out)
    state, rank = st.get('state') or 'none', st.get('rank')
    who = f'rank {rank}' if rank is not None else 'this box'
    pid = sv.read_pid(os.path.join(out, 'supervisor.pid'))
    if not pid or pid != st.get('pid') or not is_supervisor(pid, proc):
        return False, f'{who}: not serving, no supervisor runs (last state {state})'
    if state != 'serving':
        return False, f'{who}: {state}, supervisor pid {pid}'
    if rank != 0:
        return True, f'{who}: serving, supervisor pid {pid}'
    try:
        with NO_PROXY.open(f'http://127.0.0.1:{port}/health', timeout=timeout) as r:
            code = r.status
    except OSError as e:
        code = getattr(e, 'code', 0)
        if hasattr(e, 'close'):
            e.close()
    except (http.client.HTTPException, ValueError) as e:     # a bad answer, or a PORT that makes no URL
        return False, f'{who}: serving, supervisor pid {pid}, but the API /health on port {port} failed: ' \
            f'{type(e).__name__}: {e}'.splitlines()[0]
    if code != 200:
        return False, f'{who}: serving, supervisor pid {pid}, but the API /health ' + \
            (f'answers {code}' if code else f'on port {port} answers nothing')
    return True, f'{who}: serving, supervisor pid {pid}, API /health 200'


def cmd_health(argv):
    """Exit 0 or 1 only: docker reserves a healthcheck's exit 2, which bad options would otherwise give."""
    try:
        o = opt.parse(argv)
    except opt.OptionError as e:
        print(f'gr8: {e}')    # stdout, not stderr: docker keeps a healthcheck's stdout in `docker inspect`
        return 1
    out = out_dir(o)
    # the head's API port as the boot agreed it (the handshake), so a probe without the boot's options still finds it
    ok, line = health(out, (read_handshake(out) or {}).get('port') or o.get('PORT'))
    print(line)
    return 0 if ok else 1


def cmd_logs(argv):
    follow = '-f' in argv
    words = [a for a in argv if a != '-f']
    o = parse(words, check=False)
    which = o.args[0] if o.args else None
    m = members_of(o)
    me = m[2] if m else None
    if which is None or which == str(me):
        which = 'engine'
    if which == 'all':
        if not m:
            print('logs all: no handshake on this box', file=sys.stderr)
            return 2
        for r, addr in enumerate(m[0]):
            for w in ('engine', 'shim', 'front') if r == 0 else ('engine',):
                try:
                    data = rpc(m, addr, {'t': 'logs', 'which': w, 'offset': -4096})['data']
                except OSError as e:
                    data = f'no supervisor answers ({e})'
                print(f'== {"rank " + str(r) if w == "engine" else w} ({addr})')
                print('\n'.join(data.splitlines()[-10:]))
        return 0
    if which.isdigit():
        if not m or int(which) >= len(m[0]):
            print(f'logs: no rank {which} in the last handshake', file=sys.stderr)
            return 2
        host, msg = m[0][int(which)], {'t': 'logs', 'which': 'engine', 'offset': -8192}
    elif which in ('shim', 'front') and m and me != 0:   # they run on the head
        host, msg = m[0][0], {'t': 'logs', 'which': which, 'offset': -8192}
    else:
        host, msg = None, {'t': 'logs', 'which': which, 'offset': -8192}
    s = sv.Supervisor(o)
    s.rank = me
    while True:
        try:
            r = rpc(m, host, msg) if host else s.read_log(msg['which'], msg['offset'])
        except OSError as e:
            print(f'logs: {host}: {e}', file=sys.stderr)
            return 1
        sys.stdout.write(r.get('data', ''))
        sys.stdout.flush()
        if not follow:
            return 0
        msg['offset'] = r.get('offset', 0)
        time.sleep(1)


def local_rank(o):
    if opt.mode(o) != 'peers':
        return None
    from fabric import own_index
    return own_index(opt.resolve_peers(o), o.get('NODE_IP'))


def cmd_show_config(argv):
    o = parse(argv)
    ms, variant, err = sv.resolve_model(o)
    plan = opt.engine_plan(o)
    members = opt.resolve_peers(o) if opt.mode(o) == 'peers' else None
    c, text, digest = opt.canonical(o, plan, members, variant)
    print(f'options ({o.file or "no config file"}; command line > file > environment > default):')
    print(opt.summary(o))
    for line in opt.kv_describe(o):
        print(line)
    for w in o.warnings:
        print(f'warning: {w}')
    print('\ncluster options (the handshake compares these):')
    sys.stdout.write(''.join(f'  {line}\n' for line in text.splitlines()))
    print(f'digest    {digest}')
    if o.get('SHAPE') == 'ring' and opt.mode(o) == 'mentat':
        print('          (with --shape ring, each box adds ring=<the ring it found> when it joins, so the boot digest differs)')
    if err:
        print(f'model set none: {err}')
    else:
        print(f'model set {ms.id} ({ms.source})')
        if ms.release:
            print(f'release   {ms.release.label} at commit {ms.release.commit}, in {ms.release.dir}')
        if ms.drafter:
            print(f'drafter   {ms.drafter.label} at commit {ms.drafter.commit}, converted in {ms.drafter.out}')
        print(f'variant   {variant}')
        if variant.path:
            print(f'          about {modelset.expert_gb(variant.path, int(o.get("TP"))):.1f} GB of experts on the busiest '
                  f'rank at TP={o.get("TP")}')
        print(f'model id  {modelset.model_id(ms, variant)} (the handshake and the engine compare this)')
    print(f'engine    {sv.engine_build(os.path.abspath(o.get("GR8_WORK_DIR")), sv.engine_bin(o))} (bin/{sv.engine_bin(o)} and bin/lib)')
    if members:
        try:
            r = local_rank(o)
            print(f'this box  rank {r} of {",".join(members)}')
        except RuntimeError as e:
            print(f'this box  {e}')
    else:
        print(f'this box  rank decided in mentat group {o.get("GROUP")}')
    return 0


def cmd_check(argv):
    o = parse(argv)
    s = sv.Supervisor(o)
    s.ms, s.variant, err = sv.resolve_model(o)
    s.tmpl = opt.engine_plan(o)
    try:
        if err:
            raise RuntimeError(err)
        s.rank = local_rank(o)
    except RuntimeError as e:
        print(f'FAIL  {e}')
        return 1
    s.build = sv.engine_build(s.home, sv.engine_bin(o))
    print(f'model set {s.ms.id[:16]} from {s.ms.source}, variant {s.variant}; ' +
          (f'rank {s.rank}' if s.rank is not None else 'rank unknown until the mentat election: rank files not checked'))
    fails = s.checks(copy=False)
    for f in fails:
        print(f'FAIL  {f}')
    print(f'check: {len(fails)} failed')
    return 1 if fails else 0


def cmd_prepare(argv):
    o = parse(argv)
    rank = int(o.get('RANK')) if o.get('RANK') else next((int(a) for a in o.args if a.isdigit()), None)
    if rank is None:
        try:
            rank = local_rank(o)
        except RuntimeError as e:
            print(f'gr8: {e}', file=sys.stderr)
            return 2
    if rank is None:
        print('gr8: name the rank (./gr8 glm53 prepare --rank 2 ...), or give --peers: with mentat it is decided at '
              'boot', file=sys.stderr)
        return 2
    if not o.get('MODEL_CACHE'):
        print('gr8: prepare fetches into --model-cache; there is none', file=sys.stderr)
        return 2
    ms, variant, err = sv.resolve_model(o, refresh=True)
    if err:
        print(f'gr8: {err}', file=sys.stderr)
        return 1
    tp, bad = int(o.get('TP')), []
    try:
        if ms.release:
            if getattr(ms.release, 'pins', False) or ms.release.check(tp, rank, variant):
                ms.release.fetch_rank(tp, rank, variant)
            bad += ms.release.check(tp, rank, variant)
        if ms.drafter:
            if getattr(ms.drafter, 'pins', False) or ms.drafter.check(rank):
                ms.drafter.fetch_rank(rank)
            bad += ms.drafter.check(rank)
    except (modelset.ModelError, OSError, RuntimeError) as e:
        print(f'gr8: {e}', file=sys.stderr)
        return 1
    print('\n'.join(bad) or f'rank {rank} of model {modelset.model_id(ms, variant)[:16]} is ready '
          f'({ms.source}, variant {variant.name})')
    return 1 if bad else 0


def knob_lines(key, flag, scope, values, meaning):
    head = f'{key}  {flag}' + (f'  {scope}' if scope else '')
    return [head, *(['    ' + values] if values else []),
            *textwrap.wrap(meaning, 116, initial_indent='    ', subsequent_indent='    '), '']


def default_text(default, meaning, engine=False):
    """The default line, or '' when the meaning says it ('Default: ...')."""
    if default:
        return f'default: {default}'
    if 'Default: ' in meaning:
        return ''
    return 'default: per model and TP (scripts/engine_args.sh)' if engine else 'default: unset'


def engine_value(plan, key):
    """KEY's value in a plan: the option's, else GR8_KEY's in the engine environment (EXTRA_ENV can set it)."""
    if plan['knobs'].get(key):
        return plan['knobs'][key]
    return next((e.partition('=')[2] for e in reversed(plan['env']) if e.startswith(f'GR8_{key}=')), '')


def cmd_knobs(argv):
    """Without GR8_MODEL: every option of ./gr8 <model> and ./gr8 kv with its default. With it: the value each of the
    model's TPs (or --tp's alone) runs with, from scripts/engine_args.sh and these options."""
    o = parse(argv, check=False)
    words = [w.upper() for w in o.args]

    def wanted(k):
        return not words or any(w in k for w in words)
    model = models.of(o) if o.get('GR8_MODEL') else None
    tps = [o.get('TP')] if o.origin['TP'] != 'default' else list(model.tp) if model else []
    plans = {}
    for tp in tps:
        try:
            plans[tp] = opt.engine_plan(opt.parse(argv + ['--tp', tp]))
        except opt.OptionError as e:
            print(f'gr8: TP={tp}: {e}', file=sys.stderr)
            return 2
    intro = [f'./gr8 {model.name} options, and the value each TP runs with: ./gr8 {model.name} up --<flag> VALUE, '
             f'a config file or the environment ({o.file or "no config file"}). - = unset: the engine\'s own default '
             '(docs/ENV.md)' if model else
             './gr8 <model> options: --<flag> VALUE, KEY=value in a config file, or KEY in the environment. '
             './gr8 glm53 knobs shows what each TP runs with']
    lines = textwrap.wrap(intro[0], 120) + ['']
    for k, (scope, default, meaning) in opt.KEYS.items():
        if not wanted(k):
            continue
        if k == opt.ENGINE_KNOBS[0]:
            lines += ['-- engine knobs: scripts/engine_args.sh turns these into the engine\'s environment and flags', '']
        set_by = '' if o.origin[k] == 'default' else f' (set by the {o.origin[k]})'
        if model and k in opt.ENGINE_KNOBS:
            values = '  '.join(f'TP={tp} {engine_value(plans[tp], k) or "-"}' for tp in tps) + set_by
        elif model:
            values = (o.get(k) or 'unset') + set_by
        else:
            values = default_text(default, meaning, k in opt.ENGINE_KNOBS)
        lines += knob_lines(k, opt.flag_of(k), scope, values, meaning)
    if not model:
        kv = [k for k in kvnode.KEYS if wanted(k)]
        if kv:
            lines += ['-- ./gr8 kv up options (a KV node)', '']
        for k in kv:
            default, meaning = kvnode.KEYS[k]
            lines += knob_lines(k, kvnode.flag_of(k), '', default_text(default, meaning), meaning)
    lines.append('docs/ENV.md lists the engine\'s own GR8_ variables. Pass one with --extra-env "NAME=value ...".')
    print('\n'.join(lines))
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'supervise':
        sv.main(parse(rest))
    if cmd == 'get':   # one effective option: KV_FORMAT for ./gr8 glm53 build, PORT for smoke and bench
        o = parse(rest[1:], check=rest[0] == 'TP')
        print(o.get(rest[0]) or '')
        return 0 if o.get(rest[0]) else 1
    if cmd == 'restart':
        if os.environ.get('GR8_CONTAINER') == '1':
            # down ends this container's supervisor, and with it the container this restart runs in
            print('gr8: restart does not work inside the container: run down, then `docker compose up -d` on every '
                  'box', file=sys.stderr)
            return 2
        cmd_down(rest)
        return cmd_up(rest)
    f = {'up': cmd_up, 'down': cmd_down, 'status': cmd_status, 'health': cmd_health, 'logs': cmd_logs,
         'show-config': cmd_show_config,
         'check': cmd_check, 'prepare': cmd_prepare, 'knobs': cmd_knobs}.get(cmd)
    if not f:
        print(f'gr8: unknown command {cmd}\n{__doc__}', file=sys.stderr)
        return 2
    try:
        return f(rest)
    except opt.OptionError as e:    # show-config and check plan the engine line outside parse(): a refusal is a message, not a traceback
        print(f'gr8: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
