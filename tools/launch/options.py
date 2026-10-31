"""Launch options: the command line, one config file and the environment, in that order of precedence.

Every option (knobs/launch.yaml) has a KEY (the name in a config file and the environment) and a --flag. A config file
is `@FILE` on the command line, else $GR8_CLUSTER, else ./cluster.env, else <repo>/config/cluster.env, else none
(--no-config skips it). The file is parsed, not sourced: `KEY=value` lines, `#` comments, optional `export`, quotes
stripped, no `$`.

Scopes decide what the handshake compares. `cluster` options must be equal on every box (they reach every rank's
engine, or the cluster's shape). `box` options are this box's own (paths, the proxy CPU), and `run` options only change
how this launcher waits or checks.
"""
import hashlib
import ipaddress
import os
import re
import socket
import struct
import subprocess

import knobs
import models

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# key: (scope, default, meaning), from knobs/launch.yaml through the knob library (knobs.py). A default of None means
# unset, or the engine's default for the TP (scripts/engine_args.sh). The flag is --<key in lower case, _ as ->, except
# where the file names one.
_OPTIONS = knobs.options('launch')
KEYS = {k: (o['scope'], REPO if o['default'] == '{repo}' else o['default'], o['doc']) for k, o in _OPTIONS.items()}
FLAGS = {k: o['flag'] for k, o in _OPTIONS.items() if o['flag']}
# Old names, still read from the command line, a config file and the environment, with a warning.
RENAMED = {'G6_HOME': 'GR8_WORK_DIR', 'G6ENV_X': 'EXTRA_ENV', 'G6ARGS_X': 'EXTRA_ARGS', 'G6_CLUSTER': 'GR8_CLUSTER'}
RENAMED_FLAGS = {'--g6-home': '--work-dir', '--g6env-x': '--extra-env',
                 '--g6args-x': '--extra-args'}
# Read from old config files and ignored here, so they do not draw a warning.
IGNORED = {'HOSTS', 'PACKS', 'PACK_DIR', 'BIN_DIR', 'GR8_CLUSTER', 'G6_NR', 'G6_WIDTHS', 'G6_TESTS', 'GR8_NR', 'GR8_WIDTHS',
           'GR8_TESTS', 'BUNDLE_LIBS', 'ARCH', 'EXTRA', 'SHARED', 'RANK_HEALTH', 'PROBE_ARGS', 'QP_PACK', 'PLAIN', 'FRONT'}
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


def renamed_argv(argv, warnings):
    """argv with each old flag and old KEY=value word under its new name."""
    out = []
    for a in argv:
        name, eq, val = a.partition('=')
        new = RENAMED_FLAGS.get(name) or (eq and RENAMED.get(name))
        if new:
            warnings.append(f'{name} is the old name of {new}: rename it, the old name will go away')
            a = new + eq + val
        out.append(a)
    return out


def renamed_keys(d, where, warnings):
    """d (a config file's or the environment's values) with each old key under its new name; the new name wins."""
    d = dict(d)
    for old, new in RENAMED.items():
        if old in d:
            warnings.append(f'{where}: {old} is the old name of {new}: rename it, the old name will go away')
            d.setdefault(new, d.pop(old))
    return d


def default_file(environ):
    for p in (environ.get('GR8_CLUSTER') or environ.get('G6_CLUSTER'), 'cluster.env',
              os.path.join(REPO, 'config', 'cluster.env')):
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
    warnings = []
    argv = renamed_argv(argv, warnings)
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
                raise OptionError(f'unknown option {key} (in {a}): ./gr8 knobs lists the options')
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
            if key in ('KV_STORE', 'KV_TIER') and cmd.get(key):   # one store per --kv-store
                val = cmd[key] + ';' + val
            cmd[key] = val
        else:
            args.append(a)
        i += 1
    path = at or (None if '--no-config' in switches else default_file(environ))
    filev = {}
    if path:
        if not os.path.isfile(path):
            raise OptionError(f'no config file {path}')
        filev = renamed_keys(parse_file(path), path, warnings)
        # An old file names the boxes in HOSTS, by ssh name. HOSTS stands for PEERS when PEERS is absent.
        if 'HOSTS' in filev and 'PEERS' not in filev:
            filev['PEERS'] = ','.join(filev['HOSTS'].split())
        for k in sorted(filev):
            if k not in KEYS and k not in IGNORED:
                warnings.append(f'{path}: unknown key {k}, ignored')
    environ = renamed_keys(environ, 'the environment', warnings)
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
    # A box that names its model-loader fetches through it: model-loader's own deploy trees (deploy/compose/gr8.yml,
    # deploy/spark-os/gr8.override.yml, deploy/k8s/examples/gr8) set MODEL_LOADER and MODEL_CACHE, not FETCH.
    if origin['FETCH'] == 'default' and values['MODEL_LOADER']:
        values['FETCH'], origin['FETCH'] = 'model-loader', f'MODEL_LOADER, from the {origin["MODEL_LOADER"]}'
    deprecated_kv(values, origin, warnings)
    kv_store_files(values, origin, path)
    model = values['GR8_MODEL']
    if model and model not in models.MODELS:
        raise OptionError(f'GR8_MODEL={model}, from the {origin["GR8_MODEL"]}: not a model gr8 serves. Want '
                          f'{models.CHOICES} (./gr8 models lists them)')
    if model and not values['GROUP']:
        values['GROUP'] = models.MODELS[model].group
    for k, v in models.MODELS[model].defaults if model else ():
        if origin[k] == 'default':
            values[k], origin[k] = v, f'{model} default'
    opts = Options(values, origin, path, args, switches, warnings)
    opts.plain = filev.get('PLAIN', environ.get('PLAIN', ''))   # PLAIN=1 is retired. validate() refuses it
    opts.g6env = sorted(k for k in environ if k.startswith('G6_') and k not in KEYS and k not in IGNORED)
    for k in ('MODEL', 'MODEL_CACHE', 'MODEL_DIR', 'GR8_WORK_DIR', 'V2', 'V2_SLICES', 'V2_WIDTHS', 'VARIANT',
              'DFLASH_DIR', 'V2_DENSE', 'STEERING_PROFILE', 'STEERING_CHAT_TEMPLATE', 'LAYOUT', 'GRAMMAR_DIR'):
        if not values.get(k) or (k == 'MODEL' and looks_remote(values[k])) or \
                (k == 'STEERING_PROFILE' and values[k].startswith('hf:')) or \
                (k == 'VARIANT' and '/' not in values[k] and not values[k].endswith('.json')):
            continue
        # the engine runs in GR8_WORK_DIR. STEERING_PROFILE may list several directories
        values[k] = ','.join(d if not d or d.startswith('hf:') else os.path.abspath(d) for d in values[k].split(',')) \
            if k == 'STEERING_PROFILE' else os.path.abspath(values[k])
    return opts


def looks_remote(model):
    """A Hugging Face source is an https:// URL. Anything else is a local path."""
    return model.startswith('https://')


def mode(o):
    return 'peers' if o.get('PEERS') else 'mentat'


def validate(o):
    """Refuse what this version cannot boot, with the reason."""
    if not o.get('GR8_MODEL'):
        raise OptionError(f'no model: set GR8_MODEL to {models.CHOICES}, in the config file or the environment, or '
                          'run ./gr8 <model> (./gr8 models lists them)')
    model = models.of(o)
    tp = o.get('TP')
    if tp is None:
        if mode(o) == 'peers':
            tp = str(len(split_list(o.get('PEERS'))))
            o.values['TP'], o.origin['TP'] = tp, 'the number of peers'
        else:
            raise OptionError(f'--tp is needed with --mentat ({model.tp_sizes})')
    if o.g6env:
        raise OptionError(f'{", ".join(o.g6env)} set in the environment: the engine does not inherit G6_* variables. Pass '
                          'them as GR8_ names with --extra-env "NAME=value ..." so every box compares them')
    for name in (w.partition('=')[0] for w in (o.get('EXTRA_ENV') or '').split()):
        k = knobs.knob_of(name)
        if k and k['scope'] == 'test':
            raise OptionError(f'--extra-env {name}: a test knob (knobs/test.yaml), which a serving engine refuses')
        if not k and name.startswith(('GR8_', 'G6_')):
            raise OptionError(f'--extra-env {name}: not an engine variable (docs/ENV.md)')
    if o.get('VARIANT') and o.get('V2_WIDTHS'):
        raise OptionError('--variant and --v2-widths both name the width manifest: give one')
    if o.get('DRAFTER') and not o.get('DRAFTER').startswith('https://'):
        raise OptionError(f'--drafter {o.get("DRAFTER")}: want https://huggingface.co/<org>/<repo>[@rev]')
    if not model.drafter and (o.get('DRAFTER') or o.get('DFLASH') == '1'):
        raise OptionError(f'--drafter, --dflash 1: {model.name} has no DFlash2 drafter')
    if o.get('DRAFTER') and tp not in ('4', '6'):
        raise OptionError(f'--drafter: the DFlash2 drafter runs at TP=4 (split) and TP=6 (replicated) only, not TP={tp}')
    if not str(o.get('DEFAULT_MAX_TOKENS')).isdigit():
        raise OptionError(f'--default-max-tokens {o.get("DEFAULT_MAX_TOKENS")}: want a whole number (0 = fit SHARE_LIMIT)')
    effort = o.get('DEFAULT_REASONING_EFFORT')
    if model.dialect == 'ds41':
        # the release's encoder takes an integer 1..100 (what the effort prefix prints); the names are the front end's map
        if effort is not None and effort not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max') and \
                not (effort.isdigit() and 1 <= int(effort) <= 100):
            raise OptionError(f'--default-reasoning-effort {effort}: want a whole number 1 to 100, or none, minimal, low, '
                              'medium, high, xhigh or max')
    elif effort == 'none' and not model.thinking_off:
        raise OptionError('--default-reasoning-effort none: GLM 5.3 cannot turn thinking off (use low, or '
                          '--default-thinking-budget)')
    elif effort not in (None, 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise OptionError(f'--default-reasoning-effort {effort}: want minimal, low, medium, high, xhigh or max')
    if o.get('CHUNK') is not None and model.name != 'ds41':
        raise OptionError(f'--chunk: ds41 only (the prompt rows a step of the DeepSeek serve path); ./gr8 {model.name} sets its own')
    if o.get('EAGER') is not None and model.name != 'ds41':
        raise OptionError(f'--eager: ds41 only (the DeepSeek serve path runs eagerly without --graph); ./gr8 {model.name} always captures')
    if any(w.startswith(('GR8_DS_', 'GR8_DS41_')) for w in (o.get('EXTRA_ENV') or '').split()) and model.name != 'ds41':
        raise OptionError(f'--extra-env names a GR8_DS_* variable: those are DeepSeek-V4.1 engine variables, and '
                          f'./gr8 {model.name} serves another model')
    if o.get('GRAMMAR') not in ('0', '1'):
        raise OptionError(f'--grammar {o.get("GRAMMAR")}: want 0 or 1')
    if o.get('ALLOW_THINKING_OFF') not in ('0', '1'):
        raise OptionError(f'--allow-thinking-off {o.get("ALLOW_THINKING_OFF")}: want 0 or 1')
    if o.get('UNDECLARED_TOOL_CALLS') not in ('text', 'emit'):
        raise OptionError(f'--undeclared-tool-calls {o.get("UNDECLARED_TOOL_CALLS")}: want text or emit')
    if not str(o.get('DEFAULT_THINKING_BUDGET')).isdigit():
        raise OptionError(f'--default-thinking-budget {o.get("DEFAULT_THINKING_BUDGET")}: want a whole number '
                          '(0 = no limit)')
    share = o.get('SHARE_LIMIT')
    if share is not None and not (str(share).isdigit() and int(share) >= 4096 and int(share) % 64 == 0):
        # the engine takes any multiple of a KV page (64 tokens) and makes it a decode bucket boundary (context_buckets.h)
        raise OptionError(f'--share-limit {share}: want a multiple of 64 from 4096')
    busy = o.get('SHARE_LIMIT_BUSY')
    if busy is not None:
        if not (str(busy).isdigit() and int(busy) >= 4096 and int(busy) % 64 == 0 and int(busy) < int(share or 32768)):
            raise OptionError(f'--share-limit-busy {busy}: want a multiple of 64 from 4096, below SHARE_LIMIT')
        if 'local=' not in str(o.get('KV_ROWS') or ''):
            raise OptionError('--share-limit-busy needs KV_ROWS with local= (the rows past it come from the rows store)')
    big = o.get('SHARE_BIG')
    if big is not None and not (str(big).isdigit() and busy is not None):
        raise OptionError(f'--share-big {big}: want a whole number, with SHARE_LIMIT_BUSY')
    if o.get('V2') and not o.get('V2_SLICES'):
        raise OptionError('--v2 needs --v2-slices')
    if o.get('V2_DENSE_FORM') not in (None, 'trellis', 'rounded'):
        raise OptionError(f'--v2-dense-form {o.get("V2_DENSE_FORM")}: want trellis or rounded')
    if o.get('KVB') not in (None, 'bf16', 'fp8'):
        raise OptionError(f'--kvb {o.get("KVB")}: want bf16 or fp8')
    # a release boot sets V2_DENSE, or --model names a release (resolve_model sets V2_DENSE after this); the engine refuses
    # KVB=fp8 on a pack boot at start
    if o.get('KVB') == 'fp8' and not (o.get('V2_DENSE') or o.get('MODEL')):
        raise OptionError('--kvb fp8 needs a release boot (--model <release> or --v2-dense): packs hold kv_b in BF16')
    if o.get('LAYOUT') and not os.path.isfile(o.get('LAYOUT')):
        raise OptionError(f'--layout {o.get("LAYOUT")}: no such file')
    if o.get('FETCH') not in ('legacy', 'model-loader'):
        raise OptionError(f'--fetch {o.get("FETCH")}: want legacy or model-loader')
    if o.get('FETCH') == 'legacy' and o.get('MODEL_LOADER'):
        raise OptionError(f'--fetch legacy with --model-loader {o.get("MODEL_LOADER")}: gr8 fetches its own weights then, '
                          'and does not ask model-loader. Unset one')
    if o.plain == '1':
        raise OptionError('PLAIN=1 (the one-request loop) is retired at every TP: unset it')
    steering = o.get('STEERING_PROFILE')
    if o.get('STEERING_FORCE') and not steering:
        raise OptionError('--steering-force needs --steering-profile')
    if steering:
        if o.get('NOHASH') == '1':
            raise OptionError('--steering-profile requires weight hash checks (unset NOHASH)')
        bundled = steering.startswith('hf:')
        if bundled and not re.fullmatch(r'hf:[A-Za-z0-9][A-Za-z0-9._-]{0,127}', steering):
            raise OptionError('--steering-profile: want profile directories, hf:default or hf:NAME')
        dirs = steering.split(',')
        if len(dirs) > 1 and (any(d.startswith('hf:') or not d for d in dirs) or len(dirs) > 8):
            raise OptionError('--steering-profile: a list holds up to 8 profile directories (hf: selects one profile)')
        if len(dirs) > 1 and o.get('STEERING_FORCE'):
            raise OptionError('--steering-force needs exactly one --steering-profile')
        missing = [flag_of(k) for k in ('STEERING_MODEL_REVISION', 'STEERING_TOKENIZER_REVISION',
                                        'STEERING_CHAT_TEMPLATE_SHA256') if not o.get(k)]
        if missing and not bundled:
            raise OptionError('--steering-profile needs ' + ', '.join(missing))
        for key in ('STEERING_CHAT_TEMPLATE_SHA256', 'STEERING_WEIGHT_SET'):
            if o.get(key) and not re.fullmatch(r'[0-9a-f]{64}', o.get(key)):
                raise OptionError(flag_of(key) + ': want 64 lowercase hex digits')
        if o.get('STEERING_FORCE'):
            try:
                force = float(o.get('STEERING_FORCE'))
            except ValueError:
                force = 0.0
            if not (0.0 < force <= 4.0) or struct.unpack('<f', struct.pack('<f', force))[0] == 0:
                raise OptionError('--steering-force: want a finite number in (0, 4]')
    if tp not in model.tp:
        raise OptionError(f'--tp {tp}: want {model.tp_sizes} ({model.name})')
    if o.get('SHAPE') not in ('mesh', 'ring'):
        raise OptionError(f'--shape {o.get("SHAPE")}: want mesh or ring')
    if o.get('SHAPE') == 'ring':
        if tp not in ('3', '4'):
            raise OptionError(f'--shape ring: TP=3 or 4, not {tp}')
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
    stores = kv_stores(o)
    cache = o.get('FLASH_CACHE')
    if cache not in (None, 'off', 'aligned', 'terminal'):
        raise OptionError(f'--flash-cache {cache}: want off, aligned or terminal')
    if cache not in (None, 'off') and model.name != 'glm53f':
        raise OptionError(f'--flash-cache: Flash session checkpoints are glm53f only, not {model.name}')
    budget = o.get('FLASH_STATE_BUDGET_MIB')
    if budget is not None and not (str(budget).isdigit() and int(budget) >= 1):
        raise OptionError(f'--flash-state-budget-mib {budget}: want a whole number of MiB from 1')
    if model.name == 'glm53f' and stores:
        if cache in (None, 'off'):
            raise OptionError('--kv-store on glm53f holds Flash session checkpoints: it needs --flash-cache aligned or terminal')
        if len(stores) > 1:
            raise OptionError('--flash-cache keeps its checkpoints in one store: give one --kv-store')
        if any(t['persist'] for t in stores):
            raise OptionError('--flash-cache: Flash checkpoints do not persist across boots (persist=on)')
    elif stores and o.get('REUSE') == '0':
        raise OptionError('--kv-store keeps the runs prefix reuse retains: it needs REUSE (on by default)')
    if stores:
        if any(t['host'] == 'auto' for t in stores) and mode(o) == 'peers':
            raise OptionError('--kv-store remote:auto finds a KV node through mentat: with --peers, name it (remote:HOST:PORT)')
    rows = kv_rows(o)
    if rows and rows['host'] == 'auto' and mode(o) == 'peers':
        raise OptionError('--kv-rows remote:auto finds a KV node through mentat: with --peers, name it (remote:HOST:PORT)')
    if rows and rows['fmt'] == 'int4' and o.get('KV_FORMAT') == 'int4':
        raise OptionError('--kv-rows fmt=int4: the int4 KV build\'s rows are INT4 already')
    if o.get('KV_FORMAT') == 'int4' and not model.int4_kv:
        raise OptionError(f'--kv-format int4: {model.name} has an fp8 KV build only')
    if o.get('KV_FORMAT') not in (None, 'fp8', 'int4'):
        raise OptionError(f'--kv-format {o.get("KV_FORMAT")}: want fp8 or int4')
    if o.get('KV_SHARE_WHEN') not in (None, 'idle', 'always'):
        raise OptionError(f'--kv-share-when {o.get("KV_SHARE_WHEN")}: want idle or always')
    if o.get('KV_KEEP') not in (None, 'system', 'sessions', 'all'):
        raise OptionError(f'--kv-keep {o.get("KV_KEEP")}: want system, sessions or all')
    sa = o.get('KV_SYSTEM_AFTER')
    if sa is not None and not (str(sa).isdigit() and int(sa) >= 1):
        raise OptionError(f'--kv-system-after {sa}: want a whole number from 1')
    wl = o.get('KV_WRITE_LIMIT')
    if wl is not None and not re.fullmatch(r'\d+(\.\d+)?', str(wl)):
        raise OptionError(f'--kv-write-limit {wl}: want a number of page blocks per page computed, such as 0.25')


def deprecated_kv(values, origin, warnings):
    """KV_TIER and KV_SHARED_AFTER (one release, deprecated) become KV_STORE, KV_KEEP and KV_SYSTEM_AFTER."""
    old = values.get('KV_TIER')
    if old not in (None, 'off') and values.get('KV_STORE') is None:
        texts = [t.strip() for t in old.split(';') if t.strip()]
        keeps = [m.group(1) if (m := re.search(r',keep=(\w+)', t)) else 'evicted' for t in texts]   # the old default
        values['KV_STORE'] = ';'.join(re.sub(r',keep=\w+', '', t) for t in texts)
        origin['KV_STORE'] = origin['KV_TIER']
        if values.get('KV_KEEP') is None:
            values['KV_KEEP'] = 'all' if any(k in ('evicted', 'all') for k in keeps) else 'system'
            origin['KV_KEEP'] = origin['KV_TIER']
        warnings.append(f'KV_TIER (--kv-tier) is deprecated: KV_STORE={values["KV_STORE"]} KV_KEEP={values["KV_KEEP"]}')
    if values.get('KV_SHARED_AFTER') is not None:
        if values.get('KV_SYSTEM_AFTER') is None:
            values['KV_SYSTEM_AFTER'], origin['KV_SYSTEM_AFTER'] = values['KV_SHARED_AFTER'], origin['KV_SHARED_AFTER']
        warnings.append(f'KV_SHARED_AFTER (--kv-shared-after) is deprecated: KV_SYSTEM_AFTER={values["KV_SYSTEM_AFTER"]}')
    values['KV_TIER'] = values['KV_SHARED_AFTER'] = None


# KV_STORE's one-line form: entries separated by ;, each a head then ,key=value settings:
#   cache:NAME,kind=spare_ram[,floor=N(M|G)]
#   ssd:PATH,size=N(M|G|T)[,persist=on|off][,keep=RULE][,name=NAME][,cache=NAME]
#   remote:HOST:PORT|remote:auto,size=N(M|G|T)[,keep=RULE][,name=NAME][,cache=NAME][,group=NAME[,cap=N(M|G|T)][,weight=N]]
# A YAML store file says the same things (kv_store_file), with these key names: {kind: {one-line key: YAML key}}
KV_KEYS = {'cache': {'kind': 'kind', 'floor': 'floor'},
           'ssd': {'size': 'size_per_rank', 'persist': 'persist', 'keep': 'keep', 'name': 'name', 'cache': 'cache'},
           'remote': {'size': 'size_per_rank', 'keep': 'keep', 'name': 'name', 'cache': 'cache', 'group': 'group',
                      'cap': 'cap_per_rank', 'weight': 'weight'}}
CACHE_KINDS = ('spare_ram',)
SIZE = r'\d+(\.\d+)?[MGT]'
WORD = r'[A-Za-z0-9_.-]+'
# in the design but not built: refused by name so a file written for them fails loudly
NOT_BUILT = {'caches': ('takes', 'holds', 'size', 'overflow'), 'tiers': ('transforms', 'tag_restored'),
             'top': ('runs',)}
TYPE_WORDS = {str: 'a string', bool: 'true or false', int: 'a whole number', dict: 'a mapping', list: 'a list'}


class KvError(OptionError):
    """A bad KV_STORE. `where` is ('cache', NAME), ('tier', INDEX) or None, so a store file can name its key."""

    def __init__(self, msg, where=None):
        super().__init__(msg)
        self.where = where


def kv_parse(v):
    """KV_STORE's one-line form parsed and checked: (caches {NAME: {'kind', 'floor'}}, stores in cascade order
    [{'text', 'store', 'kind': 'ssd' | 'remote', 'path', 'host', 'port', 'size', 'persist', 'keep', 'name', 'cache',
    'group', 'cap', 'weight', 'settings': [(key, value) as written]}]). keep None = KV_KEEP; name defaults to the kind.
    Raises KvError naming the entry."""
    caches, stores = {}, []
    for text in [t.strip() for t in (v or '').split(';') if t.strip()]:
        head, *kvs = text.split(',')
        if re.search(r'\s', text):
            raise KvError(f'--kv-store {text}: no spaces')
        settings = [kv.partition('=')[::2] for kv in kvs]
        keys = [k for k, _ in settings]
        if len(set(keys)) != len(keys):
            raise KvError(f'--kv-store {text}: a setting given twice')
        if head.startswith('cache:'):
            name, where = head[6:], ('cache', head[6:])
            if not re.fullmatch(WORD, name):
                raise KvError(f'--kv-store {text}: cache:NAME, NAME of letters, digits and _.-', where)
            if name in caches:
                raise KvError(f'--kv-store {text}: a second cache {name}', where)
            c = dict(settings)
            for k in c:
                if k not in KV_KEYS['cache']:
                    raise KvError(f'--kv-store {text}: {k}: a cache takes kind= and floor=', where)
            if c.get('kind') not in CACHE_KINDS:
                raise KvError(f'--kv-store {text}: kind={" or ".join(CACHE_KINDS)}', where)
            if 'floor' in c and not (re.fullmatch(r'\d+(\.\d+)?[MG]', c['floor']) and float(c['floor'][:-1]) > 0):
                raise KvError(f'--kv-store {text}: floor=N(M|G)', where)
            caches[name] = {'kind': c['kind'], 'floor': c.get('floor')}
            continue
        where = ('tier', len(stores))
        m = re.fullmatch(r'ssd:(.+)', head) or re.fullmatch(r'remote:([^:,]+)(?::(\d+))?', head)
        if not m or (head.startswith('remote:auto') and m.group(2)):
            raise KvError(f'--kv-store {text}: the store is ssd:PATH, remote:HOST:PORT or remote:auto', where)
        t = {'text': text, 'store': head, 'kind': 'ssd' if head.startswith('ssd:') else 'remote', 'path': None,
             'host': None, 'port': None, 'size': None, 'persist': False, 'keep': None, 'name': None, 'cache': None,
             'group': None, 'cap': None, 'weight': None, 'settings': settings}
        if t['kind'] == 'ssd':
            t['path'] = m.group(1)
        else:
            t['host'], t['port'] = m.group(1), m.group(2)
        for k, x in settings:
            if k == 'persist' and t['kind'] != 'ssd':
                raise KvError(f'--kv-store {text}: persist= is for ssd: stores', where)
            if k in ('group', 'cap', 'weight') and t['kind'] != 'remote':
                raise KvError(f'--kv-store {text}: {k}= is for remote: stores', where)
            if k in ('spare_ram_cache', 'spare_ram_floor'):
                raise KvError(f'--kv-store {text}: {k}= is the engine\'s; name a cache: cache:NAME,kind=spare_ram '
                              'and ,cache=NAME on the store', where)
            if k in ('size', 'cap') and re.fullmatch(SIZE, x) and float(x[:-1]) > 0:
                t[k] = x
            elif k == 'persist' and x in ('on', 'off'):
                t['persist'] = x == 'on'
            elif k in ('group', 'name', 'cache') and re.fullmatch(WORD, x):
                t[k] = x
            elif k == 'weight' and re.fullmatch(r'[1-9]\d*', x) and int(x) <= 1000000:
                t['weight'] = int(x)
            elif k == 'keep':
                if x not in ('system', 'sessions', 'all'):
                    raise KvError(f'--kv-store {text}: keep=system, sessions or all (this store\'s KV_KEEP)', where)
                t['keep'] = x
            else:
                raise KvError(f'--kv-store {text}: {k}={x}: want size=N(M|G|T), persist=on|off (ssd:), keep=RULE, '
                              'name=NAME, cache=NAME, or group=NAME[,cap=N(M|G|T)][,weight=N] (remote:)', where)
        if not t['size']:
            raise KvError(f'--kv-store {text}: size=N(G|T) is needed (per rank)', where)
        if (t['cap'] or t['weight']) and not t['group']:
            raise KvError(f'--kv-store {text}: cap= and weight= need group=', where)
        stores.append(t)
    # an unnamed store is named for its kind, or kind-N when its kind repeats
    unnamed = {k: [t for t in stores if t['kind'] == k and not t['name']] for k in ('ssd', 'remote')}
    for k, ts in unnamed.items():
        for n, t in enumerate(ts, 1):
            t['auto'] = k if len(ts) == 1 else f'{k}-{n}'
    names = {}
    for i, t in enumerate(stores):
        name = t['name'] or t['auto']
        if name in names:
            raise KvError(f'--kv-store {t["text"]}: the name {name} is also store {names[name] + 1}\'s', ('tier', i))
        names[name] = i
        t['name'] = name
        if t['cache']:
            if t['cache'] not in caches:
                raise KvError(f'--kv-store {t["text"]}: no cache {t["cache"]} (caches: {", ".join(caches) or "none"})',
                              ('tier', i))
            if t['group']:
                raise KvError(f'--kv-store {t["text"]}: a cache needs a fixed lease (no group=)', ('tier', i))
    for name in caches:
        if not any(t['cache'] == name for t in stores):
            raise KvError(f'--kv-store cache:{name}: no store uses it', ('cache', name))
    if caches and not stores:
        raise KvError('--kv-store: caches but no store')
    return caches, stores


def kv_stores(o):
    """KV_STORE's stores in cascade order, as kv_parse gives them ([] when unset or off)."""
    v = o.get('KV_STORE')
    return [] if v in (None, 'off') else kv_parse(v)[1]


def kv_line(caches, stores):
    """kv_parse's result as KV_STORE's one-line form: caches first by name, then the stores, settings in KV_KEYS
    order."""
    def store(t):
        given = dict(t['settings'])
        return ','.join([t['store']] + [f'{k}={given[k]}' for k in KV_KEYS[t['kind']] if k in given])

    return ';'.join([f'cache:{n},kind={c["kind"]}' + (f',floor={c["floor"]}' if c['floor'] else '') for n, c in sorted(caches.items())] +
                    [store(t) for t in stores])


def kv_yaml(caches, stores):
    """kv_parse's result as a YAML store file (kv_store_text reads it back to the same result)."""
    import yaml

    def q(x):   # a YAML scalar, quoted where YAML would read it as something else
        return yaml.safe_dump(x, default_flow_style=True, width=1 << 20).strip().removesuffix('\n...').removesuffix('...').strip()

    def flow(d):
        return '{' + ', '.join(f'{k}: {q(x)}' for k, x in d.items()) + '}'

    def value(k, x):
        return x == 'on' if k == 'persist' else int(x) if k == 'weight' else x

    lines = []
    if caches:
        lines.append('caches:')
        lines += [f'  {q(n)}: ' + flow({k: c[k] for k in KV_KEYS['cache'] if c[k]}) for n, c in sorted(caches.items())]
    lines.append('tiers:')
    for t in stores:
        given = dict(t['settings'])
        tier = {KV_KEYS[t['kind']][k]: given[k] for k in ('keep', 'name', 'cache') if k in given}
        store = {('path' if t['kind'] == 'ssd' else 'address'): t['store'].split(':', 1)[1]}
        store.update({KV_KEYS[t['kind']][k]: value(k, x) for k, x in t['settings'] if k not in ('keep', 'name', 'cache')})
        head = '- '
        for k, x in tier.items():
            lines.append(f'{head}{k}: {q(x)}')
            head = '  '
        lines += [f'{head}store:', f'    {t["kind"]}: {flow(store)}']
    return '\n'.join(lines) + '\n'


def kv_describe(o):
    """Log lines for KV_STORE: the one-line form, then the same stores as a YAML store file."""
    v = o.get('KV_STORE')
    if v in (None, 'off'):
        return []
    caches, stores = kv_parse(v)
    out = [f'KV_STORE={kv_line(caches, stores)}']
    try:
        out += ['KV_STORE as YAML:'] + ['  ' + line for line in kv_yaml(caches, stores).splitlines()]
    except ImportError:
        out.append('KV_STORE as YAML: needs PyYAML (python3-yaml)')
    return out


def kv_store_with(o, target):
    """KV_STORE with each remote:auto naming the KV node at `target`."""
    caches, stores = kv_parse(o.get('KV_STORE'))
    for t in stores:
        if t['host'] == 'auto':
            t['store'] = f'remote:{target}'
    return kv_line(caches, stores)


def kv_engine(v):
    """KV_STORE as the engine takes it (--kv-store each, g6m4.cu): a store's cache becomes spare_ram_cache=on and
    spare_ram_floor=, and name= and the cache entries go. Settings come in KV_KEYS order."""
    if v in (None, 'off'):
        return v
    caches, stores = kv_parse(v)
    out = []
    for t in stores:
        given = dict(t['settings'])
        parts = [t['store']] + [f'{k}={given[k]}' for k in KV_KEYS[t['kind']] if k in given and k not in ('name', 'cache')]
        if t['cache']:
            floor = caches[t['cache']]['floor']
            parts += ['spare_ram_cache=on'] + ([f'spare_ram_floor={floor}'] if floor else [])
        out.append(','.join(parts))
    return ';'.join(out)


def kv_store_files(values, origin, config):
    """Each @FILE in KV_STORE replaced by the stores it holds, in the one-line form. A relative FILE is relative to the
    config file that names it, else to the working directory."""
    v = values.get('KV_STORE')
    if not v or '@' not in v:
        return
    base = os.path.dirname(config) if config and origin['KV_STORE'] in ('config file', '@file') else os.getcwd()
    texts, files = [], []
    for t in [t.strip() for t in v.split(';') if t.strip()]:
        if t.startswith('@'):
            files.append(os.path.join(base, t[1:]))
            t = kv_store_file(files[-1])
        texts.append(t)
    values['KV_STORE'] = ';'.join(texts)
    origin['KV_STORE'] += ' via ' + ', '.join(files)


def _yaml_loader():
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node):   # PyYAML keeps the last of two equal keys: refuse them
        seen = set()
        for k, _ in node.value:
            key = loader.construct_object(k)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f'{key} given twice', k.start_mark)
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node)

    def integer(loader, node):   # YAML 1.1 reads 0777 as octal and 1:20 as base 60: plain decimal only
        if not re.fullmatch(r'-?(0|[1-9]\d*)', node.value):
            raise yaml.constructor.ConstructorError(None, None, f'{node.value}: write a whole number in plain decimal',
                                                    node.start_mark)
        return int(node.value)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    Loader.add_constructor('tag:yaml.org,2002:int', integer)
    return yaml, Loader


def kv_store_text(text, path='-'):
    """A YAML store file's text as KV_STORE's one-line form (kv_line). Shapes and types are checked here; everything
    else by kv_parse, so the two forms refuse the same things. Raises OptionError naming the file and the key.

        caches:
          spare: {kind: spare_ram, floor: 2G}      # memory the kernel may take back
        tiers:                                     # tried in order: a full tier passes entries to the next
          - keep: system
            cache: spare
            store:
              ssd: {path: /nvme/gr8kv, size_per_rank: 4G, persist: true}
          - keep: all
            cache: spare
            store:
              remote: {address: auto, size_per_rank: 4G}
    """
    def fail(where, msg):
        raise OptionError(f'KV_STORE=@{path}: {where}: {msg}')

    def want(where, x, t):
        if type(x) is not t:   # a YAML 1.1 on/yes is a bool, a bare 4 an int, 1:20 a base-60 int
            fail(where, f'want {TYPE_WORDS[t]}, got {x!r}')
        return x

    def keys(where, d, allowed, not_built=()):
        for k in d:
            if k in not_built:
                fail(f'{where}.{k}', 'not built yet')
            if k not in allowed:
                fail(f'{where}.{k}', f'want {", ".join(allowed)}')

    def one(x):   # a YAML value in the one-line form
        return ('on' if x else 'off') if isinstance(x, bool) else str(x)

    try:
        yaml, loader = _yaml_loader()
    except ImportError:
        raise OptionError(f'KV_STORE=@{path}: needs PyYAML (python3-yaml)') from None
    try:
        doc = yaml.load(text, Loader=loader)
    except yaml.YAMLError as e:
        raise OptionError(f'KV_STORE=@{path}: {e}') from None
    want('the file', doc, dict)
    keys('the file', doc, ('caches', 'tiers'), NOT_BUILT['top'])
    caches = want('caches', doc.get('caches', {}), dict)
    entries, tier_keys = [], {}
    for name, c in caches.items():
        where = f'caches.{name}'
        want(where, name, str)
        keys(where, want(where, c, dict), tuple(KV_KEYS['cache'].values()), NOT_BUILT['caches'])
        for k, x in c.items():
            want(f'{where}.{k}', x, str)
            if re.search(r'[,;=\s]', x):
                fail(f'{where}.{k}', f'{x!r}: no commas, semicolons, = or spaces')
        if re.search(r'[,;=\s]', name):
            fail(where, f'{name!r}: no commas, semicolons, = or spaces in a cache name')
        entries.append(','.join([f'cache:{name}'] + [f'{k}={c[k]}' for k in KV_KEYS['cache'] if k in c]))
    if 'tiers' not in doc:
        fail('tiers', 'no tiers')
    tiers = want('tiers', doc['tiers'], list)
    if not tiers:
        fail('tiers', 'no tiers')
    for i, t in enumerate(tiers):
        where = f'tiers[{i}]'
        keys(where, want(where, t, dict), ('name', 'keep', 'cache', 'store'), NOT_BUILT['tiers'])
        if 'store' not in t:
            fail(where, 'no store')
        store = want(f'{where}.store', t['store'], dict)
        if len(store) != 1 or next(iter(store)) not in ('ssd', 'remote'):
            fail(f'{where}.store', 'want one of ssd: or remote:')
        kind, s = next(iter(store.items()))
        where_s = f'{where}.store.{kind}'
        address = 'path' if kind == 'ssd' else 'address'
        on_store = {y: k for k, y in KV_KEYS[kind].items() if y not in ('keep', 'name', 'cache')}
        keys(where_s, want(where_s, s, dict), (address, *on_store))
        if address not in s:
            fail(where_s, f'no {address}')
        settings = [(f'{where}.{y}', k, t[y]) if y in ('keep', 'name', 'cache') else (f'{where_s}.{y}', k, s[y])
                    for k, y in KV_KEYS[kind].items() if y in (t if y in ('keep', 'name', 'cache') else s)]
        for w, k, x in [(f'{where_s}.{address}', None, s[address])] + settings:
            want(w, x, {'persist': bool, 'weight': int}.get(k, str))
            if isinstance(x, str) and re.search(r'[,;=\s]', x):
                fail(w, f'{x!r}: no commas, semicolons, = or spaces')
        entries.append(','.join([f'{kind}:{s[address]}'] + [f'{k}={one(x)}' for _, k, x in settings]))
        tier_keys[i] = where
    try:
        return kv_line(*kv_parse(';'.join(entries)))
    except KvError as e:
        where = {'cache': lambda n: f'caches.{n}', 'tier': lambda i: tier_keys.get(i, 'tiers')}[e.where[0]](e.where[1]) \
            if e.where else 'the file'
        fail(where, str(e))


def kv_store_file(path):
    """kv_store_text of the file at `path`."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        raise OptionError(f'KV_STORE=@{path}: {e.strerror}') from None
    return kv_store_text(text, path)


def kv_rows(o):
    """KV_ROWS parsed: {'text', 'store', 'kind': 'remote' | 'mem', 'host', 'port', 'size', 'local', 'pool', 'place',
    'fmt'}, or None when unset or off. Raises OptionError naming the bad part."""
    v = o.get('KV_ROWS')
    if v in (None, 'off'):
        return None
    text = v.strip()
    store, *kvs = text.split(',')
    m = re.fullmatch(r'remote:([^:,]+)(?::(\d+))?', store)
    if store != 'mem:' and (not m or (m.group(1) == 'auto' and m.group(2))):
        raise OptionError(f'--kv-rows {text}: the store is remote:auto, remote:HOST:PORT or mem:')
    t = {'text': text, 'store': store, 'kind': 'mem' if store == 'mem:' else 'remote', 'host': m.group(1) if m else None,
         'port': m.group(2) if m else None, 'size': None, 'local': None, 'pool': None, 'place': None, 'fmt': None}
    for kv in kvs:
        k, _, x = kv.partition('=')
        if k == 'size' and re.fullmatch(r'\d+(\.\d+)?[MGT]', x) and float(x[:-1]) > 0:
            t['size'] = x
        elif k in ('local', 'pool', 'place') and x.isdigit() and int(x) > 0:
            t[k] = int(x)
        elif k == 'fmt' and x in ('fp8', 'int4'):
            t['fmt'] = x
        else:
            raise OptionError(f'--kv-rows {text}: {kv}: want size=N(M|G|T), local=PAGES, pool=SLOTS, place=PAGES or fmt=fp8|int4')
    if not t['size']:
        raise OptionError(f'--kv-rows {text}: size=N(M|G|T) is needed (each rank\'s lease)')
    if t['place'] and not t['local']:
        raise OptionError(f'--kv-rows {text}: place= needs local= (the pages past it are reached through the rows store only)')
    if t['kind'] == 'remote' and t['host'] != 'auto' and not t['port']:
        raise OptionError(f'--kv-rows {text}: remote:HOST:PORT names the port')
    return t


def kv_rows_with(o, target):
    """KV_ROWS with remote:auto replaced by the KV node host:port this box uses (None: as written)."""
    t = kv_rows(o)
    if not t or t['host'] != 'auto' or not target:
        return t and t['text']
    return t['text'].replace('remote:auto', f'remote:{target}', 1)


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


# ---- the engine plan: scripts/engine_args.sh --------------------------------------------------------------------------
def engine_plan(o, rank='@R', rank_dir='@RANKDIR', aux_dir='@AUXDIR', home='@HOME', epoch='@EPOCH', set_id='',
                kv_store=None, kv_rows=None):
    """{'knobs': {K: resolved value}, 'env': [K=V..], 'args': [..]} from scripts/engine_args.sh.
    The defaults give the canonical template, with every per-box value as a placeholder. kv_store, kv_rows: this box's
    KV_STORE and KV_ROWS with the KV node resolved (remote:GB names none)."""
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'GR8_MODEL': models.of(o).name, 'TP': o.get('TP'),
           'SET_ID': set_id}
    for k in ENGINE_KNOBS + ['S2PORT', 'READY_TMO']:
        if o.get(k) is not None:
            env[k] = o.get(k)
    if o.get('STEERING_PROFILE'):
        for key in ('STEERING_PROFILE', 'STEERING_CHAT_TEMPLATE', 'STEERING_MODEL_REVISION', 'STEERING_TOKENIZER_REVISION'):
            if any(c.isspace() for c in o.get(key) or ''):
                raise OptionError(flag_of(key) + ': whitespace is not supported by the engine launcher')
    if o.get('STEERING_PROFILE') and not env.get('STEERING_CHAT_TEMPLATE'):
        env['STEERING_CHAT_TEMPLATE'] = '@STEERING_CHAT_TEMPLATE'
    if kv_store:
        env['KV_STORE'] = kv_store
    if env.get('KV_STORE'):
        env['KV_STORE'] = kv_engine(env['KV_STORE'])
    if kv_rows:
        env['KV_ROWS'] = kv_rows
    if rank == '@R':   # the template: v2 paths are per box
        for k in ('V2', 'V2_SLICES', 'V2_WIDTHS', 'DFLASH_DIR', 'V2_DENSE', 'STEERING_PROFILE', 'STEERING_CHAT_TEMPLATE', 'LAYOUT',
                  'GRAMMAR_DIR'):
            if env.get(k):
                env[k] = '@' + k
    names = ' '.join(ENGINE_KNOBS + ['S2PORT', 'READY_TMO'])
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
    plan['drafter_dir'] = o.get('DFLASH_DIR')
    return plan


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(o, plan, members, variant=None, ring=''):
    """The cluster options as sorted key=value lines, and their sha256. members: the peer list (peers mode) or None.
    variant: the resolved exlr8 v2 variant (modelset.Variant), whose name and manifest sha256 count. ring: --shape ring
    with mentat, the ring this box planned from mentat's claim (mentat.ring_text), so every box must have the same."""
    c = {'tp': o.get('TP'), 'shape': o.get('SHAPE'), 'mode': mode(o), 'fabric_port': str(FABRIC_PORT)}
    if models.of(o) is not models.GLM53:   # GLM-5.3's lines stay as they were, so a mixed upgrade still agrees
        c['model'] = models.of(o).name
    if members:
        c['peers'] = ','.join(members)
    else:
        c['group'] = o.get('GROUP')
        c['head'] = o.get('HEAD', '')
    for k in ('IPS', 'IPS2', 'CONTROL_PORT', 'PORT'):
        c[k] = o.get(k) or ''
    # Every knob that reaches the engine is in engine.env and engine.args below. CAP is also here by name, so a refusal
    # names it. DFLASH and PF_OWN decide which pack files a rank needs.
    for k in ('CAP', 'DFLASH', 'PF_OWN'):
        c[k] = plan['knobs'].get(k, '')
    if variant:
        c['variant'] = variant.name
        c['variant.sha256'] = variant.sha256 or ''
    if ring:
        c['ring'] = ring
    if o.get('LAYOUT'):   # a path per box: its bytes are what every rank must agree on
        c['layout.sha256'] = file_sha256(o.get('LAYOUT'))
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
