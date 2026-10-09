#!/usr/bin/env python3
"""./gr8 glm53 <command> for the symmetric launch: the same command on every box. Python 3 standard library only.

    up [OPTIONS] [@FILE]       start this box's supervisor in the background and follow it until the cluster serves
    down                       stop every box of the last boot (or this box alone)
    restart [OPTIONS]          down, then up
    status                     every box: its state, the engine, MemAvailable; the head's shim, front end and /health
    logs [RANK|shim|front|supervisor] [-f]   a log, from any box
    check [OPTIONS]            this box's checks
    show-config [OPTIONS]      the effective options, the cluster options the handshake compares, their digest, the
                               model set id
    prepare [RANK] [OPTIONS]   copy this box's shard of the model set into --model-cache, without booting

OPTIONS are --key value (or KEY=value lines in a config file, or KEY in the environment); options.py lists them.
--foreground runs the supervisor in this process. Exit status: 0 serving (or done), 1 failed, 2 bad options, 4 the
handshake refused the boot.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handshake  # noqa: E402
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
    return os.path.join(os.path.abspath(o.get('G6_HOME')), 'out')


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
    out, home = out_dir(o), os.path.abspath(o.get('G6_HOME'))
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
    for w in o.warnings:
        print(f'warning: {w}')
    print('\ncluster options (the handshake compares these):')
    sys.stdout.write(''.join(f'  {line}\n' for line in text.splitlines()))
    print(f'digest    {digest}')
    if err:
        print(f'model set none: {err}')
    else:
        print(f'model set {ms.id} ({ms.source})')
        print(f'variant   {variant}')
        if variant.path:
            print(f'          about {modelset.expert_gb(variant.path, int(o.get("TP"))):.1f} GB of experts on the busiest '
                  f'rank at TP={o.get("TP")}')
        print(f'model id  {modelset.model_id(ms, variant)} (the handshake and the engine compare this)')
    print(f'engine    {sv.engine_build(os.path.abspath(o.get("G6_HOME")))} (bin/g6m4 and bin/lib)')
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
    s.build = sv.engine_build(s.home)
    print(f'model set {s.ms.id[:16]} from {s.ms.source}, variant {s.variant}; ' +
          (f'rank {s.rank}' if s.rank is not None else 'rank unknown until the mentat election: rank files not checked'))
    fails = s.checks(copy=False)
    for f in fails:
        print(f'FAIL  {f}')
    print(f'check: {len(fails)} failed')
    return 1 if fails else 0


def cmd_prepare(argv):
    o = parse(argv)
    rank = next((int(a) for a in o.args if a.isdigit()), None)
    if rank is None:
        try:
            rank = local_rank(o)
        except RuntimeError as e:
            print(f'gr8: {e}', file=sys.stderr)
            return 2
    if rank is None:
        print('gr8: name the rank (./gr8 glm53 prepare 2 ...): with mentat it is decided at boot', file=sys.stderr)
        return 2
    if not o.get('MODEL_CACHE'):
        print('gr8: prepare copies into --model-cache; there is none', file=sys.stderr)
        return 2
    try:
        ms = modelset.resolve(o.get('MODEL'), o.get('MODEL_CACHE'), int(o.get('TP')), o.get('MODEL_DIR'))
        modelset.prepare(ms, rank, opt.engine_plan(o), log=sv.log)
        bad = modelset.validate_rank(ms, rank, opt.engine_plan(o))
    except modelset.ModelError as e:
        print(f'gr8: {e}', file=sys.stderr)
        return 1
    print('\n'.join(bad) or f'rank {rank} of set {ms.id[:16]} is ready in {ms.rank_dir(rank)}')
    return 1 if bad else 0


def main(argv):
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'supervise':
        sv.main(parse(rest))
    if cmd == 'get':   # one effective option, for ./gr8 glm53 build
        o = parse(rest[1:])
        print(o.get(rest[0]) or '')
        return 0 if o.get(rest[0]) else 1
    if cmd == 'restart':
        cmd_down(rest)
        return cmd_up(rest)
    f = {'up': cmd_up, 'down': cmd_down, 'status': cmd_status, 'logs': cmd_logs, 'show-config': cmd_show_config,
         'check': cmd_check, 'prepare': cmd_prepare}.get(cmd)
    if not f:
        print(f'gr8: unknown command {cmd}\n{__doc__}', file=sys.stderr)
        return 2
    return f(rest)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
