"""Launch options: the command line, one config file and the environment, in that order of precedence.

Every option has a KEY (the name in a config file and the environment) and a --flag. A config file is `@FILE` on the
command line, else $G6_CLUSTER, else ./cluster.env, else <repo>/config/cluster.env, else none (--no-config skips
it). The file is parsed, not sourced: `KEY=value` lines, `#` comments, optional `export`, quotes stripped, no `$`.

Scopes decide what the handshake compares. `cluster` options must be equal on every box (they reach every rank's
engine, or the cluster's shape). `box` options are this box's own (paths, the proxy CPU), and `run` options only change
how this launcher waits or checks.
"""
import hashlib
import ipaddress
import os
import re
import socket
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# key: (scope, default, meaning). A default of None means unset, or the engine's default for the TP
# (scripts/engine_args.sh). The flag is --<key in lower case, _ as ->, except where FLAGS says otherwise.
KEYS = {
    'TP': ('cluster', None, 'tensor-parallel size: 4, 6 or 3'),
    'SHAPE': ('cluster', 'mesh', "fabric shape: mesh (a switch), or ring (TP=3 or 4, no switch: each box's f0 port "
                                 "cabled to the f1 port of the next box in --peers order, the last to the first)"),
    'PEERS': ('cluster', None, 'LAN addresses of every box in rank order; the first is the head'),
    'MENTAT': ('box', None, "the local mentat daemon (localhost) to find the other boxes through"),
    'GROUP': ('cluster', 'gr8-glm53', 'mentat group the boxes join'),
    'HEAD': ('cluster', None, 'mentat mode: the head LAN address, instead of the lowest one'),
    'IPS': ('cluster', None, 'rail-0 fabric address of every rank (skips fabric discovery)'),
    'IPS2': ('cluster', None, 'rail-1 fabric address of every rank'),
    'CONTROL_PORT': ('cluster', '7340', 'the supervisors\' TCP port on the LAN'),
    'PORT': ('cluster', '8100', 'the OpenAI API port on the head'),
    'S2PORT': ('cluster', None, 'the S2 control port on the head (7341)'),
    'MODEL': ('box', None, 'the exlr8 release (https://huggingface.co/<org>/<repo>[@rev], or a local copy), or a '
                           'prepared pack directory'),
    'BASE_PACKS': ('box', None, 'with an https:// --model: take the non-expert weights from this prepared pack directory '
                                'instead of the release'),
    'MODEL_CACHE': ('box', None, "this box's local directory for its shard of the model set"),
    'MODEL_DIR': ('box', None, 'tokenizer directory, when the model directory has none'),
    'G6_HOME': ('box', REPO, 'working directory: bin/, out/, s2/, serve/, run/'),
    'PROXY_CPU': ('box', '19', "CPU core for this rank's fabric proxy thread"),
    'RING_DEVS': ('box', None, "--shape ring: this box's RDMA devices as relay-out,next,relay-in,prev (default "
                               'rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1)'),
    'NODE_IP': ('box', None, "this box's LAN address, when the peer list cannot tell"),
    'PY': ('box', None, 'python for the shim (default $G6_HOME/venv/bin/python)'),
    'FRONT': ('box', None, 'front end: rs or py'),
    'FRONT_BIN': ('box', None, 'the Rust front end binary (default bin/g6front-rs)'),
    'DEFAULT_MAX_TOKENS': ('box', '0', 'max_tokens for a chat request that names none; 0 = what keeps the request '
                                       'within SHARE_LIMIT, but at least 8192'),
    'RANK': ('run', None, 'prepare: the rank to fetch, when this box cannot tell'),
    'JOIN_TIMEOUT': ('run', '600', 'seconds to wait for the other boxes; on the head, from the first hello (0 = forever)'),
    'READY_TMO': ('run', None, 'seconds to wait for the engines to load'),
    'CHECK': ('run', '1', '0 = skip the local checks'),
    'CAP': ('cluster', None, 'KV pool in tokens (default: what SHARE_LIMIT leaves of the default pool memory)'),
    'SHARE_LIMIT': ('cluster', None, 'requests up to this many tokens (prompt + max_tokens) share decode steps; a '
                                     'longer one runs alone. Each rank holds the first SHARE_LIMIT tokens of a session'),
    'SESS': ('cluster', None, 'S2 sessions'),
    'MTP': ('cluster', None, 'MTP draft count'),
    'DFLASH': ('cluster', None, 'DFlash2 drafter'),
    'DFLASH_SEL': ('cluster', None, 'DFlash2 selector walk'),
    'DFLASH_K': ('cluster', None, 'DFlash2 drafts verified'),
    'SPEC_ADAPT': ('cluster', None, 'adaptive draft count'),
    'SPEC_LEVELS': ('cluster', None, 'draft counts SPEC_ADAPT may choose'),
    'SPEC_ACCEPT': ('cluster', None, 'draft acceptance'),
    'REUSE': ('cluster', None, 'prefix reuse'),
    'PF_OWN': ('cluster', None, 'prefill attention ownership'),
    'QP_ROLL': ('cluster', None, 'PF_OWN=4 rolling pair slots'),
    'QP_CTX': ('cluster', None, 'PF_OWN=4 QO key table size'),
    'QP_EARLY': ('cluster', None, 'PF_OWN=4 early counterpart partial'),
    'NOHASH': ('cluster', None, 'skip the pack tensor hashes at load'),
    'G6ENV_X': ('cluster', None, 'more engine environment for every rank'),
    'G6ARGS_X': ('cluster', None, 'more engine arguments for every rank'),
    'V2': ('box', None, 'exlr8 v2 width files directory'),
    'V2_SLICES': ('box', None, 'exlr8 v2 index slices'),
    'V2_WIDTHS': ('box', None, 'exlr8 v2 width manifest; --variant names one instead'),
    'VARIANT': ('cluster', None, 'exlr8 v2 variant: home (the default; down1 at TP=3), down1, up1, or a width manifest path'),
    'V2_DENSE': ('box', None, 'the exlr8 release directory whose dense/ files give every non-expert weight (no rank packs)'),
    'V2_DENSE_FORM': ('cluster', None, 'with V2_DENSE: trellis (the release\'s entries, the default) or rounded (MXFP8 / '
                                       'int4, the fallback)'),
    'DRAFTER': ('box', None, 'TP=4: the DFlash2 drafter, https://huggingface.co/<org>/<repo>[@rev], fetched and '
                             'converted on this box'),
    'DFLASH_DIR': ('box', None, 'the directory holding dflash.r<r>.*, when it is not the model set\'s'),
}
FLAGS = {'S2PORT': '--s2-port', 'G6_HOME': '--g6-home'}
# Read from old config files and ignored here, so they do not draw a warning.
IGNORED = {'HOSTS', 'PACKS', 'PACK_DIR', 'BIN_DIR', 'G6_CLUSTER', 'G6_NR', 'BUNDLE_LIBS', 'ARCH', 'EXTRA', 'SHARED',
           'RANK_HEALTH', 'PROBE_ARGS', 'QP_PACK', 'PLAIN'}
SWITCHES = {'--no-config', '--no-check', '--foreground'}
ENGINE_KNOBS = [k for k in list(KEYS)[list(KEYS).index('CAP'):]]
FABRIC_PORT = 18600   # engine/comms/fabric.h: the fabric bootstrap and per-collective ports start here


class OptionError(Exception):
    pass


def flag_of(key):
    return FLAGS.get(key, '--' + key.lower().replace('_', '-'))


KEY_OF_FLAG = {flag_of(k): k for k in KEYS}


def parse_file(path):
    """{KEY: value} from a cluster.env-style file. Raises OptionError naming the line."""
    out = {}
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].lstrip()
            m = re.match(r'([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
            if not m:
                raise OptionError(f'{path}:{n}: not KEY=value: {raw.rstrip()}')
            key, v = m.group(1), m.group(2)
            if v[:1] in ('"', "'"):
                end = v.find(v[0], 1)
                if end < 0:
                    raise OptionError(f'{path}:{n}: unterminated quote')
                rest = v[end + 1:].strip()
                if rest and not rest.startswith('#'):
                    raise OptionError(f'{path}:{n}: text after the closing quote')
                v = v[1:end]
            else:
                v = re.split(r'\s+#', v, maxsplit=1)[0].strip()
            if '$' in v or '`' in v:
                raise OptionError(f'{path}:{n}: {key}: the file is parsed, not sourced, so $ and ` are not expanded')
            out[key] = v
    return out


def default_file(environ):
    for p in (environ.get('G6_CLUSTER'), 'cluster.env', os.path.join(REPO, 'config', 'cluster.env')):
        if p and os.path.isfile(p):
            return p
    return None


class Options:
    """values: {KEY: value}; origin: {KEY: 'command line' | '@file' | 'environment' | 'default'}."""

    def __init__(self, values, origin, file, args, switches, warnings):
        self.values, self.origin, self.file, self.args, self.switches, self.warnings = \
            values, origin, file, args, switches, warnings

    def get(self, key, default=None):
        v = self.values.get(key)
        return default if v is None or v == '' else v

    def int(self, key):
        v = self.get(key)
        try:
            return int(v)
        except (TypeError, ValueError):
            raise OptionError(f'{key}={v!r}: want a whole number')


def parse(argv, environ=None):
    """argv after the command. Returns Options; positional words that are not options stay in .args."""
    environ = os.environ if environ is None else environ
    cmd, at, args, switches = {}, None, [], set()
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('@'):
            if at:
                raise OptionError('one @FILE only')
            at = a[1:]
        elif a in SWITCHES:
            switches.add(a)
        elif re.match(r'[A-Z][A-Z0-9_]*=', a):
            key, _, val = a.partition('=')
            if key not in KEYS:
                raise OptionError(f'unknown option {key} (in {a}): KNOBS.md lists the options')
            cmd[key] = val
        elif a.startswith('--'):
            name, eq, val = a.partition('=')
            key = KEY_OF_FLAG.get(name)
            if key is None:
                raise OptionError(f'unknown option {name}')
            if not eq:
                i += 1
                if i >= len(argv):
                    raise OptionError(f'{name} needs a value')
                val = argv[i]
            cmd[key] = val
        else:
            args.append(a)
        i += 1
    warnings = []
    path = at or (None if '--no-config' in switches else default_file(environ))
    filev = {}
    if path:
        if not os.path.isfile(path):
            raise OptionError(f'no config file {path}')
        filev = parse_file(path)
        # An old file names the boxes in HOSTS, by ssh name. HOSTS stands for PEERS when PEERS is absent.
        if 'HOSTS' in filev and 'PEERS' not in filev:
            filev['PEERS'] = ','.join(filev['HOSTS'].split())
        for k in sorted(filev):
            if k not in KEYS and k not in IGNORED:
                warnings.append(f'{path}: unknown key {k}, ignored')
    values, origin = {}, {}
    for k, (_, dflt, _) in KEYS.items():
        for src, d in (('command line', cmd), ('@file' if at else 'config file', filev), ('environment', environ)):
            if k in d:
                values[k], origin[k] = d[k], src
                break
        else:
            values[k], origin[k] = dflt, 'default'
    # The mode comes from the strongest source that names one.
    for src, d in (('command line', cmd), ('file', filev), ('environment', environ)):
        has_p, has_m = bool(d.get('PEERS')), bool(d.get('MENTAT'))
        if has_p and has_m:
            raise OptionError(f'the {src} sets both PEERS (--peers) and MENTAT (--mentat): choose one')
        if has_p:
            values['MENTAT'] = None
            break
        if has_m:
            values['PEERS'] = None
            break
    else:
        values['MENTAT'], origin['MENTAT'] = 'localhost', 'default'
    if '--no-check' in switches:
        values['CHECK'], origin['CHECK'] = '0', 'command line'
    opts = Options(values, origin, path, args, switches, warnings)
    opts.plain = filev.get('PLAIN', environ.get('PLAIN', ''))   # PLAIN=1 is retired. validate() refuses it
    opts.g6env = sorted(k for k in environ if k.startswith('G6_') and k not in KEYS and k not in IGNORED)
    for k in ('MODEL', 'MODEL_CACHE', 'MODEL_DIR', 'G6_HOME', 'V2', 'V2_SLICES', 'V2_WIDTHS', 'VARIANT', 'BASE_PACKS',
              'DFLASH_DIR', 'V2_DENSE'):
        if not values.get(k) or (k == 'MODEL' and looks_remote(values[k])) or (k == 'VARIANT' and '/' not in values[k]
                                                                               and not values[k].endswith('.json')):
            continue
        values[k] = os.path.abspath(values[k])   # the engine runs in G6_HOME
    return opts


def looks_remote(model):
    """A Hugging Face source is an https:// URL. Anything else is a local path."""
    return model.startswith('https://')


def mode(o):
    return 'peers' if o.get('PEERS') else 'mentat'


def validate(o):
    """Refuse what this version cannot boot, with the reason."""
    tp = o.get('TP')
    if tp is None:
        if mode(o) == 'peers':
            tp = str(len(split_list(o.get('PEERS'))))
            o.values['TP'], o.origin['TP'] = tp, 'the number of peers'
        else:
            raise OptionError('--tp is needed with --mentat (4, 6 or 3)')
    if o.g6env:
        raise OptionError(f'{", ".join(o.g6env)} set in the environment: the engine does not inherit G6_* variables. Pass '
                          'them with --g6env-x "NAME=value ..." so every box compares them')
    if o.get('VARIANT') and o.get('V2_WIDTHS'):
        raise OptionError('--variant and --v2-widths both name the width manifest: give one')
    if o.get('DRAFTER') and not o.get('DRAFTER').startswith('https://'):
        raise OptionError(f'--drafter {o.get("DRAFTER")}: want https://huggingface.co/<org>/<repo>[@rev]')
    if o.get('DRAFTER') and tp != '4':
        raise OptionError(f'--drafter: the DFlash2 drafter runs at TP=4 only (its 8 KV heads do not split {tp} ways)')
    if not str(o.get('DEFAULT_MAX_TOKENS')).isdigit():
        raise OptionError(f'--default-max-tokens {o.get("DEFAULT_MAX_TOKENS")}: want a whole number (0 = fit SHARE_LIMIT)')
    share = o.get('SHARE_LIMIT')
    if share is not None and not (str(share).isdigit() and int(share) >= 4096 and int(share) & (int(share) - 1) == 0):
        # the engine needs the limit to be one of its decode buckets (2048 doubling)
        raise OptionError(f'--share-limit {share}: want a power of two from 4096')
    if o.get('V2') and not o.get('V2_SLICES'):
        raise OptionError('--v2 needs --v2-slices')
    if o.get('V2_DENSE_FORM') not in (None, 'trellis', 'rounded'):
        raise OptionError(f'--v2-dense-form {o.get("V2_DENSE_FORM")}: want trellis or rounded')
    if o.plain == '1':
        raise OptionError('PLAIN=1 (the one-request loop) is retired at every TP: unset it')
    if tp not in ('3', '4', '6'):
        raise OptionError(f'--tp {tp}: want 4, 6 or 3')
    if o.get('SHAPE') not in ('mesh', 'ring'):
        raise OptionError(f'--shape {o.get("SHAPE")}: want mesh or ring')
    if o.get('SHAPE') == 'ring':
        if tp not in ('3', '4'):
            raise OptionError(f'--shape ring: TP=3 or 4, not {tp}')
        if mode(o) != 'peers':
            raise OptionError('--shape ring needs --peers: their order is the cable order')
        if o.get('IPS'):
            raise OptionError('--shape ring takes no --ips/--ips2: the engine finds its devices by name (--ring-devs) '
                              'and bootstraps over the LAN')
    if o.get('RING_DEVS') and len(split_list(o.get('RING_DEVS'))) != 4:
        raise OptionError(f'--ring-devs {o.get("RING_DEVS")}: want four devices')
    if mode(o) == 'peers' and len(split_list(o.get('PEERS'))) != int(tp):
        raise OptionError(f'--peers names {len(split_list(o.get("PEERS")))} boxes for --tp {tp}')
    for k in ('IPS', 'IPS2'):
        if o.get(k) and len(split_list(o.get(k))) != int(tp):
            raise OptionError(f'{k} has {len(split_list(o.get(k)))} addresses for --tp {tp}')
    if bool(o.get('IPS')) != bool(o.get('IPS2')):
        raise OptionError('IPS and IPS2 go together')
    for k in ('CONTROL_PORT', 'PORT', 'PROXY_CPU', 'JOIN_TIMEOUT'):
        o.int(k)


def split_list(v):
    return [x for x in re.split(r'[,\s]+', v or '') if x]


def resolve_peers(o):
    """The peer list as IPv4 addresses, resolved on this box."""
    out = []
    for name in split_list(o.get('PEERS')):
        try:
            addr = socket.getaddrinfo(name, None, socket.AF_INET)[0][4][0]
        except OSError:
            raise OptionError(f'cannot resolve peer {name!r}. PEERS (or HOSTS) must be LAN addresses or names every box '
                              'resolves, not ssh aliases; literal addresses are best')
        if ipaddress.ip_address(addr).is_loopback and name != addr and len(split_list(o.get('PEERS'))) > 1:
            raise OptionError(f'peer {name!r} resolves to {addr} here; name the LAN address instead')
        out.append(addr)
    if len(set(out)) != len(out):
        raise OptionError(f'the peer list names one box twice: {",".join(out)}')
    return out


# ---- the engine plan: scripts/engine_args.sh, so the ssh launchers and the supervisor start the same engine ---------
def engine_plan(o, rank='@R', rank_dir='@RANKDIR', aux_dir='@AUXDIR', home='@HOME', epoch='@EPOCH', set_id=''):
    """{'knobs': {K: resolved value}, 'env': [K=V..], 'args': [..], 's2': bool} from scripts/engine_args.sh.
    The defaults give the canonical template, with every per-box value as a placeholder."""
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'TP': o.get('TP'), 'SET_ID': set_id}
    for k in ENGINE_KNOBS + ['S2PORT', 'READY_TMO']:
        if o.get(k) is not None:
            env[k] = o.get(k)
    if rank == '@R':   # the template: v2 paths are per box
        for k in ('V2', 'V2_SLICES', 'V2_WIDTHS', 'DFLASH_DIR', 'V2_DENSE'):
            if env.get(k):
                env[k] = '@' + k
    names = ' '.join(ENGINE_KNOBS + ['S2PORT', 'READY_TMO', 'S2'])
    script = ('set -e; . "$0"; engine_defaults; engine_env; for k in ' + names + '; do echo "knob $k=${!k-}"; done; '
              'echo "env $ENGINE_ENV"; echo "args $(engine_args "$@")"')
    r = subprocess.run(['bash', '-c', script, os.path.join(REPO, 'scripts', 'engine_args.sh'),
                        str(rank), rank_dir, aux_dir, home, str(epoch)], env=env, capture_output=True, text=True)
    if r.returncode:
        raise OptionError(f'scripts/engine_args.sh: {r.stderr.strip()}')
    plan = {'knobs': {}, 'env': [], 'args': []}
    for line in r.stdout.splitlines():
        kind, _, rest = line.partition(' ')
        if kind == 'knob':
            k, _, v = rest.partition('=')
            plan['knobs'][k] = v
        elif kind == 'env':
            plan['env'] = rest.split()
        elif kind == 'args':
            plan['args'] = rest.split()
    plan['s2'] = plan['knobs'].pop('S2') == '1'
    plan['drafter_dir'] = o.get('DFLASH_DIR')
    return plan


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(o, plan, members, variant=None):
    """The cluster options as sorted key=value lines, and their sha256. members: the peer list (peers mode) or None.
    variant: the resolved exlr8 v2 variant (modelset.Variant), whose name and manifest sha256 count."""
    c = {'tp': o.get('TP'), 'shape': o.get('SHAPE'), 'mode': mode(o), 'fabric_port': str(FABRIC_PORT)}
    if members:
        c['peers'] = ','.join(members)
    else:
        c['group'] = o.get('GROUP')
        c['head'] = o.get('HEAD', '')
    for k in ('IPS', 'IPS2', 'CONTROL_PORT', 'PORT'):
        c[k] = o.get(k) or ''
    # Every knob that reaches the engine is in engine.env and engine.args below. These are the ones the supervisor
    # reads itself: the shim's context limit, and which pack files a rank needs.
    for k in ('CAP', 'DFLASH', 'PF_OWN'):
        c[k] = plan['knobs'].get(k, '')
    if variant:
        c['variant'] = variant.name
        c['variant.sha256'] = variant.sha256 or ''
    c['engine.env'] = ' '.join(sorted(plan['env']))
    c['engine.args'] = ' '.join(plan['args'])
    text = ''.join(f'{k}={c[k]}\n' for k in sorted(c))
    return c, text, hashlib.sha256(text.encode()).hexdigest()


def summary(o):
    """The effective options, one line each, with where each value came from."""
    lines = []
    for k, (scope, _, _) in KEYS.items():
        v = o.values.get(k)
        if v is None or v == '':
            continue
        lines.append(f'  {k:<13} {v:<28} {scope:<8} ({o.origin.get(k, "default")})')
    return '\n'.join(lines)
