#!/usr/bin/env python3
"""The DeepSeek serve path accepts one small argument set (reports/ds-in-gr8-build/b9-serve-path.md, items 1 and 2, accepted
by the legate 10-25 6:55 PM PT): ONE session, position 0, no prefill, MTP 0, no KV store, no DFlash drafter,
no grammar, G6_DEC_SPLIT 0, Engram off unless G6_DS_ENGRAM=1 comes with its G6_DS_ENGRAM_PREFIX. g6m4_ds41 dies at parse time on
the rest. This test holds the launcher's default
`./gr8 ds41` engine line to that set, so a bad argument fails here, in CPU CI, and not at the end of a boot.

SERVE_FLAGS is the list B9's item 2 names, with the rule for each value. It is a copy of the engine's checks, not a read of
them: when the engine's die() lines change, this table changes with them."""
import hashlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402
import options as opt  # noqa: E402
import supervisor as S  # noqa: E402

# flag: (number of values, rule over the value list or None)
SERVE_FLAGS = {
    '--image': (0, None),
    '--layers': (1, lambda v: v == ['all']),                     # a partial layer list is an m4sim first-light run
    '--v2-dense': (1, lambda v: v[0].startswith(('/', '@'))),    # the checkpoint directory (a '@V2_DENSE' placeholder in a plan)
    '--chunk': (1, lambda v: v[0].isdigit() and 1 <= int(v[0]) <= 8),   # M <= 8 until B6's MoE slots allow more
    '--temp': (1, None),
    '--top-p-cands': (1, lambda v: v[0].isdigit()),              # kept (B7, row 5): the head returns the candidate list
    '--cap': (1, lambda v: v[0].isdigit() and int(v[0]) > 0),
    '--graph': (0, None),
    '--reuse-prefix': (0, None),   # opt-in DS41 snapshot cache (S2 only)
    '--serve': (1, None),
    '--s2-port': (1, lambda v: v[0].isdigit() and 1 <= int(v[0]) <= 65535),
    '--s2-sessions': (1, lambda v: v[0].isdigit() and 1 <= int(v[0]) <= 32),  # SESS 2..32: one attention / index / Engram slab per session
    '--s2-epoch': (1, None),                                     # '@EPOCH' in a plan; > 0 at a real boot
    '--mtp': (1, lambda v: v == ['0']),                          # MTP >= 1 is DSpark, serve v2
    '--no-hash': (0, None),
    '--set-id': (1, None),
}
GR8 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'gr8')
PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'
# What ./gr8 ds41 refuses before it starts anything (scripts/engine_args.sh ds41_defaults): an argument set the engine would
# die on at boot. The message is a substring of the one the launcher prints.
REFUSED = (
    (['--sess', '2'], 'G6_DS_PREFILL_MIN>=2'),
    (['--mtp', '1'], 'serves with MTP=0'),
    (['--g6env-x', 'G6_DEC_SPLIT=3'], 'G6_DEC_SPLIT=0'),
    (['--g6env-x', 'G6_DS_ENGRAM=1'], 'needs G6_DS_ENGRAM_PREFIX'),
    (['--kv-store', 'ssd:/tmp/kv,size=1G'], 'has no KV store'),
    (['--reuse', '2'], 'needs REUSE=0 or 1'),
    (['--dflash', '1'], 'no DFlash2 drafter'),
)
REQUIRED = ('--image', '--layers', '--v2-dense', '--graph', '--serve', '--s2-port', '--s2-sessions', '--s2-epoch', '--mtp')


def serve_problems(args, env, eager=False):
    """What g6m4_ds41 --serve would refuse in `args` (a token list) and `env` (['K=V', ..]); [] when it takes both.
    eager: the line runs each step without --graph (the engine takes it: --graph is optional)."""
    out, seen, i = [], [], 0
    table = SERVE_FLAGS
    while i < len(args):
        flag = args[i]
        if not flag.startswith('--') or flag not in table:
            out.append(f'{flag}: not in the DeepSeek serve argument set')
            i += 1
            continue
        n, rule = table[flag]
        vals = args[i + 1:i + 1 + n]
        if len(vals) != n or any(v.startswith('--') for v in vals):
            out.append(f'{flag}: wants {n} value(s), got {vals}')
        elif rule is not None and not rule(vals):
            out.append(f'{flag} {" ".join(vals)}: value outside the DeepSeek serve rule')
        seen.append(flag)
        i += 1 + n
    out += [f'{f}: required' for f in REQUIRED if f not in seen and not (eager and f == '--graph')]
    kv = dict(e.split('=', 1) for e in env if '=' in e)
    if kv.get('G6_DEC_SPLIT', '0') != '0':
        out.append(f'G6_DEC_SPLIT={kv["G6_DEC_SPLIT"]}: v1 serves at split 0')
    if kv.get('G6_DS_ENGRAM', '0') != '0' and not kv.get('G6_DS_ENGRAM_PREFIX'):
        out.append('G6_DS_ENGRAM=1 without G6_DS_ENGRAM_PREFIX: no sidecar')
    sess = args[args.index('--s2-sessions') + 1:][:1] if '--s2-sessions' in args else []
    if sess not in (['1'], []) and (not kv.get('G6_DS_PREFILL_MIN', '0').isdigit() or int(kv.get('G6_DS_PREFILL_MIN', '0')) < 2):
        out.append('--s2-sessions > 1 needs G6_DS_PREFILL_MIN >= 2 (g6m4.cu: the same rows per session per step)')
    if sess not in (['1'], []) and kv.get('G6_DS_MTP', '0') != '0' and min(int(sess[0]), 8) * (int(kv['G6_DS_MTP']) + 1) > 48:
        out.append('min(--s2-sessions, 8) x (G6_DS_MTP + 1) > 48 (g6m4.cu: a DSpark round of up to 8 sessions is one step of G6_MX rows)')
    if kv.get('G6_DS_ENGRAM', '0') != '0' and kv.get('G6_DS_MTP', '0') != '0' and kv.get('G6_DS_ENGRAM_MTP') != '1':
        out.append('G6_DS_ENGRAM=1 with G6_DS_MTP needs G6_DS_ENGRAM_MTP=1 (g6m4.cu:541: not validated yet)')
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.t)
        for n in ('config.json', 'model.safetensors.index.json', *models.TOKENIZER_DS):
            with open(os.path.join(self.t, n), 'w') as f:
                f.write('{}')

    def plan(self, *argv):
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t, *argv], {'GR8_MODEL': 'ds41'})
        opt.validate(o)
        S.resolve_model(o)          # V2_DENSE, GRAMMAR_DIR as a real `up` leaves them
        p = opt.engine_plan(o)
        return p['args'], p['env']

class Ds41ServeArgsTests(Base):
    def test_the_default_engine_line_is_in_the_serve_set(self):
        args, env = self.plan()
        self.assertEqual(serve_problems(args, env), [], ' '.join(args))
        self.assertEqual(args[args.index('--v2-dense') + 1], '@V2_DENSE')   # the plan is a template: the box fills it in

    def test_sessions_with_the_w_prefill_are_in_the_serve_set(self):
        """SESS 2..8 (one attention / index / Engram slab per session): the line carries --s2-sessions SESS when G6ENV_X has
        G6_DS_PREFILL_MIN >= 2 (every prompt of 2+ tokens on the W-row prefill, so a batched step has one row per session)."""
        args, env = self.plan('--sess', '8', '--g6env-x', 'G6_DS_PREFILL_MIN=2 G6_DS_PREFILL_W=1024')
        self.assertEqual(args[args.index('--s2-sessions') + 1], '8')
        self.assertEqual(serve_problems(args, env), [], ' '.join(args) + ' ' + ' '.join(env))

    def test_sessions_with_dspark_and_engram_are_in_the_serve_set(self):
        """SESS 8 and SESS 32 with the DSpark draft (G6_DS_MTP=5) and Engram ON: up to 8 x 6 = 48 verify rows and 8 x 5 = 40 draft rows a round (G6_MX 48); above 8 active sessions MTP is dynamically off."""
        x = 'G6_DS_PREFILL_MIN=2 G6_DS_PREFILL_W=1024 G6_DS_MTP=5 G6_DS_ENGRAM=1 G6_DS_ENGRAM_PREFIX=/x/ds41 G6_DS_ENGRAM_MTP=1'
        for sess in ('8', '16', '32'):
            with self.subTest(sess=sess):
                args, env = self.plan('--sess', sess, '--g6env-x', x)
                self.assertEqual(args[args.index('--s2-sessions') + 1], sess)
                self.assertIn('G6_DS_MTP=5', env)
                self.assertEqual(serve_problems(args, env), [], ' '.join(args) + ' ' + ' '.join(env))

    def test_the_gemv_splits_are_off_in_the_env_of_every_rank(self):
        """g6m4 defaults G6_QA_SPLIT=2 and G6_IDXQ_SPLIT=1 (G6_MTP_SPLIT=1 with --mtp) at NR > 1 on any non-Flash build, and on DeepSeek
        TP 4 that asserts in Fabric::init after 'layers bound'. The plan's env, which every rank's engine gets, must carry all three at 0."""
        want = ['G6_QA_SPLIT=0', 'G6_IDXQ_SPLIT=0', 'G6_MTP_SPLIT=0']
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t], {'GR8_MODEL': 'ds41'})
        opt.validate(o)
        S.resolve_model(o)
        plans = {'template': opt.engine_plan(o)}
        for r in range(4):
            plans[f'rank {r}'] = opt.engine_plan(o, rank=str(r), rank_dir=self.t, aux_dir=self.t, home=self.t, epoch='1', set_id='5e7b1d0c')
        for name, p in plans.items():
            with self.subTest(plan=name):
                self.assertEqual([e for e in p['env'] if e.split('=')[0] in ('G6_QA_SPLIT', 'G6_IDXQ_SPLIT', 'G6_MTP_SPLIT')], want, p['env'])
        # the supervisor hands a rank's engine this env over its allowlist, whatever the box's own environment says
        for r in range(4):
            with self.subTest(engine_env=r), unittest.mock.patch.dict(os.environ, {'G6_QA_SPLIT': '2', 'G6_IDXQ_SPLIT': '1'}):
                sv = types.SimpleNamespace(home=self.t, o=o)
                sv.borrow_idle = lambda knobs: False
                env = S.Supervisor.engine_env(sv, plans[f'rank {r}'])
                self.assertEqual([env.get(k) for k in ('G6_QA_SPLIT', 'G6_IDXQ_SPLIT', 'G6_MTP_SPLIT')], ['0', '0', '0'])
        # a set value is the user's: it goes through unchanged
        o2 = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t, '--qa-split', '2'], {'GR8_MODEL': 'ds41'})
        opt.validate(o2)
        self.assertIn('G6_QA_SPLIT=2', opt.engine_plan(o2)['env'])

    def test_chunk_defaults_to_eight_and_the_engine_line_follows_it(self):
        """CHUNK is the prompt rows a step (serve v1 runs the prompt as decode steps): 8 until set (served-v3 ring pass), 1 to 8 as the engine takes."""
        args, env = self.plan()
        self.assertEqual(args[args.index('--chunk') + 1], '8')
        self.assertEqual(serve_problems(args, env), [])
        for n in range(1, 9):
            with self.subTest(chunk=n):
                args, env = self.plan('--chunk', str(n))
                self.assertEqual(args[args.index('--chunk') + 1], str(n))
                self.assertEqual(serve_problems(args, env), [])
        for bad in ('0', '9', '16', 'x', '1.5'):
            with self.subTest(bad=bad):
                with self.assertRaises(opt.OptionError) as e:
                    self.plan('--chunk', bad)
                self.assertIn('CHUNK', str(e.exception))
        o = opt.parse(['--no-config', '--peers', PEERS, '--chunk', '4'], {'GR8_MODEL': 'glm53'})
        with self.assertRaises(opt.OptionError) as e:       # ds41 only: GLM sets its own
            opt.validate(o)
        self.assertIn('--chunk', str(e.exception))

    def test_each_argument_the_engine_refuses_is_caught(self):
        args, env = self.plan()
        for extra, env_extra, want in (
                (['--prefill', '4096'], [], '--prefill'),
                (['--pf-lanes', '2'], [], '--pf-lanes'),
                (['--wide'], [], '--wide'),
                (['--pfv2'], [], '--pfv2'),
                (['--kv-store', '/x'], [], '--kv-store'),
                (['--dflash-manifest', '/x'], [], '--dflash-manifest'),
                (['--grammar-dir', '/x'], [], '--grammar-dir'),
                (['--spec-adaptive'], [], '--spec-adaptive'),
                ([], ['G6_DEC_SPLIT=3'], 'G6_DEC_SPLIT=3'),
                ([], ['G6_DS_ENGRAM=1'], 'G6_DS_ENGRAM=1 without G6_DS_ENGRAM_PREFIX')):
            with self.subTest(want=want):
                got = serve_problems(args + extra, env + env_extra)
                self.assertTrue(any(want in p for p in got), got)
        for flag, bad in (('--mtp', '1'), ('--s2-sessions', '33'), ('--chunk', '9'), ('--s2-port', '70000'), ('--layers', '0,2')):
            with self.subTest(flag=flag, bad=bad):
                i = args.index(flag)
                got = serve_problems(args[:i + 1] + [bad] + args[i + 2:], env)
                self.assertTrue(any(p.startswith(f'{flag} {bad}') for p in got), got)
        for flag in REQUIRED:
            with self.subTest(missing=flag):
                i = args.index(flag)
                n = SERVE_FLAGS[flag][0]
                got = serve_problems(args[:i] + args[i + 1 + n:], env)
                self.assertIn(f'{flag}: required', got)

    def test_eager_drops_graph_and_nothing_else(self):
        """EAGER=1 (--eager 1): the engine line is the default line without --graph, for the first light of an op that has never been
        captured (Engram ON). Default 0 keeps the graph."""
        base, env = self.plan()
        eager, eenv = self.plan('--eager', '1')
        self.assertIn('--graph', base)
        self.assertEqual([a for a in base if a != '--graph'], eager)
        self.assertEqual(env, eenv)
        self.assertEqual(serve_problems(eager, eenv, eager=True), [])
        self.assertIn('--graph: required', serve_problems(eager, eenv))
        self.assertEqual(self.plan('--eager', '0'), (base, env))
        for bad in ('2', 'x', '-1'):
            with self.subTest(bad=bad):
                with self.assertRaises(opt.OptionError) as e:
                    self.plan('--eager', bad)
                self.assertIn('EAGER', str(e.exception))
        o = opt.parse(['--no-config', '--peers', PEERS, '--eager', '1'], {'GR8_MODEL': 'glm53'})
        with self.assertRaises(opt.OptionError) as e:       # ds41 only
            opt.validate(o)
        self.assertIn('--eager', str(e.exception))

    def test_g6_ds_engram_mtp_is_the_opt_in_for_engram_with_dspark_and_reaches_every_rank(self):
        """Engram ON + DSpark (G6_DS_MTP) is refused by the engine unless G6_DS_ENGRAM_MTP=1 (g6m4.cu:541). The knob is a G6_DS_*
        variable like the others: --g6env-x puts it in every rank's engine env, ./gr8 ds41 takes the whole line (MTP stays 0, split 0)."""
        base = 'G6_DS_ENGRAM=1 G6_DS_ENGRAM_PREFIX=/x/ds41 G6_DEC_SPLIT=0 G6_DS_MTP=5'
        args, env = self.plan('--g6env-x', base)
        got = serve_problems(args, env)
        self.assertTrue(any('G6_DS_ENGRAM_MTP=1' in g for g in got), got)         # without the knob the copy of the engine rule refuses
        args, env = self.plan('--g6env-x', base + ' G6_DS_ENGRAM_MTP=1')
        self.assertEqual(serve_problems(args, env), [])
        self.assertEqual(args[args.index('--mtp') + 1], '0')
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t, '--g6env-x', base + ' G6_DS_ENGRAM_MTP=1'], {'GR8_MODEL': 'ds41'})
        opt.validate(o)
        S.resolve_model(o)
        for r in range(4):
            plan = opt.engine_plan(o, rank=str(r), rank_dir=self.t, aux_dir=self.t, home=self.t, epoch='1', set_id='5e7b1d0c')
            sv = types.SimpleNamespace(home=self.t, o=o)
            sv.borrow_idle = lambda knobs: False
            with self.subTest(engine_env=r):
                e = S.Supervisor.engine_env(sv, plan)
                self.assertEqual([e.get(k) for k in ('G6_DS_ENGRAM', 'G6_DS_MTP', 'G6_DS_ENGRAM_MTP')], ['1', '5', '1'])
        r = subprocess.run([GR8, 'ds41', 'show-config', '--no-config', '--peers', PEERS, '--model', self.t,
                            '--g6env-x', base + ' G6_DS_ENGRAM_MTP=1'], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn('Traceback', r.stderr)

    def test_the_launcher_refuses_what_serve_v1_does_not_take(self):
        for argv, want in REFUSED:
            with self.subTest(argv=argv):
                with self.assertRaises(opt.OptionError) as e:
                    self.plan(*argv)
                self.assertIn(want, str(e.exception))

    def test_every_refusal_exits_nonzero_with_a_message(self):
        for argv, want in REFUSED:
            with self.subTest(argv=argv):
                r = subprocess.run([GR8, 'ds41', 'show-config', '--no-config', '--peers', PEERS, '--model', self.t, *argv],
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn(want, r.stderr)
                self.assertNotIn('Traceback', r.stderr)

    def test_the_serve_v1_settings_are_accepted(self):
        for argv in ([], ['--sess', '1', '--mtp', '0'], ['--g6env-x', 'G6_DS_ENGRAM=0 G6_DEC_SPLIT=0'], ['--kv-store', 'off'],
                     ['--g6env-x', 'G6_DS_ENGRAM=1 G6_DS_ENGRAM_PREFIX=/x/ds41 G6_DEC_SPLIT=0']):
            with self.subTest(argv=argv):
                args, env = self.plan(*argv)
                self.assertEqual(serve_problems(args, env), [])

    def test_prefix_reuse_is_opt_in(self):
        cold, env = self.plan()
        warm, warm_env = self.plan('--reuse', '1')
        self.assertNotIn('--reuse-prefix', cold)
        self.assertEqual(warm, cold + ['--reuse-prefix'])
        self.assertEqual(warm_env, env)
        self.assertEqual(serve_problems(warm, warm_env), [])


def engine_binary():
    """g6m4_ds41 built from a tree that has --check-args and the f1669d4 serve checks (B9): G6M4_DS41=/path, else bin/g6m4_ds41 in the checkout."""
    for p in (os.environ.get('G6M4_DS41'), os.path.join(os.path.dirname(GR8), 'bin', 'g6m4_ds41')):
        if p and os.access(p, os.X_OK):
            return p
    return None


def last_line(err):
    """The engine's last stderr line for a progress print ('' when it printed nothing, as /bin/false does)."""
    return ((err.strip().splitlines() or [''])[-1])[:110]


def with_flag(args, flag, *vals):
    """args with `flag vals` set (the value of an existing flag replaced, a missing flag appended)."""
    n = len(vals)
    if flag in args:
        i = args.index(flag)
        return args[:i] + [flag, *vals] + args[i + 1 + n:]
    return args + [flag, *vals]


class Ds41EngineParseTests(Base):
    """The same engine line through g6m4_ds41 itself: --check-args runs every parse-time check and exits 0 before CUDA, the
    fabric or any weights, so this needs no GPU (CUDA_VISIBLE_DEVICES is empty for every run). It does NOT skip when the
    binary is missing: a skip reads as green and proves nothing about the engine. It fails, naming how to supply one."""
    CHECKED = 'the arguments pass the parse-time checks'

    @classmethod
    def setUpClass(cls):
        cls.binary = engine_binary()
        if not cls.binary:
            raise AssertionError('NO g6m4_ds41 BINARY, so the engine parse checks did not run: set G6M4_DS41=/path/to/g6m4_ds41 '
                                 '(built from a tree with the f1669d4 checks, B9: MODEL=ds41 G6_WIDTHS=4 scripts/build.sh DIR) '
                                 'or build bin/g6m4_ds41')
        with open(cls.binary, 'rb') as f:
            data = f.read()
        print(f'\nengine binary: {cls.binary}  {len(data)} B  md5 {hashlib.md5(data).hexdigest()}', file=sys.stderr)

    def run_engine(self, args, env=(), binary=None):
        a = ['--rank', '0', '--ips', PEERS, '--pf-ips2', PEERS, '--proxy-cpu', '0', *args, '--check-args']
        e = {**os.environ, **dict(x.split('=', 1) for x in env), 'CUDA_VISIBLE_DEVICES': ''}
        r = subprocess.run([binary or self.binary, *a], capture_output=True, text=True, timeout=60, env=e)
        return r.returncode, r.stderr

    def line(self, *argv):
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t, *argv], {'GR8_MODEL': 'ds41'})
        opt.validate(o)
        S.resolve_model(o)
        p = opt.engine_plan(o, rank='0', rank_dir=self.t, aux_dir=self.t, home=self.t, epoch='1', set_id='5e7b1d0c')    # the engine wants 1..64 lowercase hex digits (the model id)
        return p['args'], p['env']

    def passes(self, args, env, binary=None):
        rc, err = self.run_engine(args, env, binary)
        return rc == 0 and self.CHECKED in err, rc, err

    def test_the_default_engine_line_passes_the_engines_parse_checks(self):
        args, env = self.line()
        ok, rc, err = self.passes(args, env)
        print(f'  default line: rc {rc}: {err.strip()}', file=sys.stderr)
        self.assertTrue(ok, f'rc {rc}: {err}')

    def test_prefix_reuse_line_passes_and_kv_store_still_refuses(self):
        args, env = self.line('--reuse', '1')
        self.assertIn('--reuse-prefix', args)
        ok, rc, err = self.passes(args, env)
        self.assertTrue(ok, f'rc {rc}: {err}')
        rc, err = self.run_engine(args + ['--kv-store', 'ssd:/tmp/kv,size=1G'], env)
        self.assertNotEqual(rc, 0, err)
        self.assertIn('DS41: --kv-store is not supported', err)

    def test_each_argument_the_engine_cannot_take_dies_at_parse_with_DS41(self):
        args, env = self.line()
        for name, bad in (('prefill', args + ['--prefill', '4096']), ('mtp 1', with_flag(args, '--mtp', '1')),
                          ('sessions 2', with_flag(args, '--s2-sessions', '2')), ('port 70000', with_flag(args, '--s2-port', '70000')), ('epoch 0', with_flag(args, '--s2-epoch', '0')),
                          # the engine's own generic checks fire first for --kv-store and --dflash-manifest alone (g6m4.cu:323, :494);
                          # these sets get past them, to the DS41 refusal (:524, :528)
                          ('kv-store', args + ['--reuse-prefix', '--kv-store', 'ssd:/tmp/kv,size=1G']),
                          ('dflash-manifest', args + ['--dflash-pack', '/x', '--dflash-manifest', '/x']),
                          ('grammar-dir', args + ['--grammar-dir', '/x'])):
            with self.subTest(bad=name):
                rc, err = self.run_engine(bad, env)
                print(f'  {name}: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertNotEqual(rc, 0, err)
                self.assertIn('DS41', err)

    def test_every_chunk_the_launcher_allows_passes_the_engines_parse_checks(self):
        for n in range(1, 9):
            with self.subTest(chunk=n):
                args, env = self.line('--chunk', str(n))
                self.assertEqual(args[args.index('--chunk') + 1], str(n))
                ok, rc, err = self.passes(args, env)
                self.assertTrue(ok, f'--chunk {n}: rc {rc}: {last_line(err)}')

    def test_the_bare_kv_store_and_dflash_manifest_still_die(self):
        args, env = self.line()
        for name, bad in (('kv-store alone', args + ['--kv-store', 'ssd:/tmp/kv,size=1G']),
                          ('dflash-manifest alone', args + ['--dflash-manifest', '/x'])):
            with self.subTest(bad=name):
                rc, err = self.run_engine(bad, env)
                print(f'  {name}: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertNotEqual(rc, 0, err)

    def test_serve_refuses_a_split_graph_and_takes_engram_with_its_sidecar_prefix(self):
        """B9 f1669d4: --serve with G6_DEC_SPLIT != 0 dies at parse (the launcher's ds41_defaults refuses it earlier; this is the
        engine's own check). B11 e465198: G6_DS_ENGRAM=1 + G6_DS_ENGRAM_PREFIX passes (one session, MTP 0, the token ring); without a
        PREFIX, or with a DSpark draft (G6_DS_MTP), it dies."""
        args, env = self.line()
        prefix = 'G6_DS_ENGRAM_PREFIX=/nonexistent/engram'      # --check-args reads no file: the sidecar is opened at boot
        for name, extra in (('G6_DEC_SPLIT=3', ['G6_DEC_SPLIT=3']), ('G6_DEC_SPLIT=1', ['G6_DEC_SPLIT=1']),
                            ('G6_DS_ENGRAM=1 with G6_DS_MTP=2', ['G6_DS_ENGRAM=1', prefix, 'G6_DS_MTP=2'])):
            with self.subTest(bad=name):
                rc, err = self.run_engine(args, [*env, *extra])
                print(f'  {name}: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertNotEqual(rc, 0, err)
                self.assertIn('DS41', err)
        with self.subTest(bad='G6_DS_ENGRAM=1 without a PREFIX'):
            rc, err = self.run_engine(args, [*env, 'G6_DS_ENGRAM=1'])
            print(f'  G6_DS_ENGRAM=1, no PREFIX: rc {rc}: {last_line(err)}', file=sys.stderr)
            self.assertNotEqual(rc, 0, err)
        for name, extra in (('G6_DEC_SPLIT=0', ['G6_DEC_SPLIT=0']), ('G6_DS_ENGRAM=0', ['G6_DS_ENGRAM=0']),
                            ('G6_DS_ENGRAM=1 + PREFIX', ['G6_DS_ENGRAM=1', prefix])):
            with self.subTest(ok=name):
                ok, rc, err = self.passes(args, [*env, *extra])
                print(f'  {name}: rc {rc}', file=sys.stderr)
                self.assertTrue(ok, f'rc {rc}: {err}')

    def test_the_eager_line_and_the_eager_engram_line_pass_the_engines_parse_checks(self):
        """The first-light line: no --graph, Engram ON with its sidecar PREFIX (the line the Engram first-light window runs)."""
        for name, argv in (('eager', ['--eager', '1']),
                           ('eager + Engram ON', ['--eager', '1', '--g6env-x', 'G6_DS_ENGRAM=1 G6_DS_ENGRAM_PREFIX=/root/ds41-engram/ds41']),
                           ('graph + Engram ON', ['--g6env-x', 'G6_DS_ENGRAM=1 G6_DS_ENGRAM_PREFIX=/root/ds41-engram/ds41'])):
            with self.subTest(line=name):
                args, env = self.line(*argv)
                self.assertEqual('--graph' in args, name.startswith('graph'))
                ok, rc, err = self.passes(args, env)
                print(f'  {name}: rc {rc}', file=sys.stderr)
                self.assertTrue(ok, f'rc {rc}: {err}')
                self.assertEqual(serve_problems(args, env, eager='--graph' not in args), [])

    def test_g6_ds_mtp_from_g6env_x_reaches_every_rank_engine_and_the_engine_takes_it(self):
        """The DSpark draft is G6_DS_MTP=1..5 with --mtp 0 (g6m4.cu:528); without the variable the engine stays at MTP 0 without a word.
        The launcher has no DSpark option: --g6env-x G6_DS_MTP=k is the way, so the variable must be in every rank's engine env,
        --mtp stays 0, --s2-sessions stays 1, and the engine's parse checks pass (0 = off, 6 = refused)."""
        for k in range(1, 6):
            with self.subTest(k=k):
                args, env = self.line('--g6env-x', f'G6_DS_MTP={k}')
                self.assertIn(f'G6_DS_MTP={k}', env)
                self.assertEqual(args[args.index('--mtp') + 1], '0')
                self.assertEqual(args[args.index('--s2-sessions') + 1], '1')
                ok, rc, err = self.passes(args, env)
                print(f'  G6_DS_MTP={k}: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertTrue(ok, f'rc {rc}: {err}')
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t, '--g6env-x', 'G6_DS_MTP=5'], {'GR8_MODEL': 'ds41'})
        opt.validate(o)
        S.resolve_model(o)
        for r in range(4):      # the supervisor's allowlist keeps the plan's G6_* variables, and drops the box's own
            plan = opt.engine_plan(o, rank=str(r), rank_dir=self.t, aux_dir=self.t, home=self.t, epoch='1', set_id='5e7b1d0c')
            sv = types.SimpleNamespace(home=self.t, o=o)
            sv.borrow_idle = lambda knobs: False
            with self.subTest(engine_env=r):
                self.assertEqual(S.Supervisor.engine_env(sv, plan).get('G6_DS_MTP'), '5')
        o0 = opt.parse(['--no-config', '--peers', PEERS, '--model', self.t], {'GR8_MODEL': 'ds41'})
        opt.validate(o0)
        S.resolve_model(o0)
        plan0 = opt.engine_plan(o0, rank='0', rank_dir=self.t, aux_dir=self.t, home=self.t, epoch='1', set_id='5e7b1d0c')
        sv0 = types.SimpleNamespace(home=self.t, o=o0)
        sv0.borrow_idle = lambda knobs: False
        with unittest.mock.patch.dict(os.environ, {'G6_DS_MTP': '5'}):     # a shell variable is not the way: the engine stays at MTP 0
            self.assertNotIn('G6_DS_MTP', S.Supervisor.engine_env(sv0, plan0))
        args, env = self.line()
        self.assertFalse([e for e in env if e.startswith('G6_DS_MTP')], env)    # default: off
        rc, err = self.run_engine(args, [*env, 'G6_DS_MTP=6'])
        print(f'  G6_DS_MTP=6: rc {rc}: {last_line(err)}', file=sys.stderr)
        self.assertNotEqual(rc, 0, err)
        self.assertIn('G6_DS_MTP', err)

    def test_engram_with_dspark_dies_at_parse_unless_g6_ds_engram_mtp_is_1(self):
        """g6m4.cu:541: G6_DS_ENGRAM=1 with G6_DS_MTP=1..5 keeps its parse-time die (same message) unless G6_DS_ENGRAM_MTP=1.
        Only the value 1 lifts it; Engram off, or MTP off, never needed it."""
        args, env = self.line()
        base = [*env, 'G6_DS_ENGRAM=1', 'G6_DS_ENGRAM_PREFIX=/nonexistent/engram']
        for name, extra in (('unset', []), ('G6_DS_ENGRAM_MTP=0', ['G6_DS_ENGRAM_MTP=0']), ('G6_DS_ENGRAM_MTP=2', ['G6_DS_ENGRAM_MTP=2']),
                            ('G6_DS_ENGRAM_MTP=', ['G6_DS_ENGRAM_MTP='])):
            with self.subTest(refused=name):
                rc, err = self.run_engine(args, [*base, 'G6_DS_MTP=5', *extra])
                print(f'  Engram + MTP 5, {name}: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertNotEqual(rc, 0, err)
                self.assertIn('G6_DS_ENGRAM=1 with G6_DS_MTP is not validated yet (Engram ON serves at MTP 0)', err)
        for k in range(1, 6):
            with self.subTest(taken_k=k):
                ok, rc, err = self.passes(args, [*base, f'G6_DS_MTP={k}', 'G6_DS_ENGRAM_MTP=1'])
                print(f'  Engram + MTP {k}, G6_DS_ENGRAM_MTP=1: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertTrue(ok, f'rc {rc}: {err}')
        for name, extra in (('Engram ON, MTP off', [*base]), ('Engram off, MTP 5', [*env, 'G6_DS_MTP=5']),
                            ('Engram off, MTP 5, knob 1', [*env, 'G6_DS_MTP=5', 'G6_DS_ENGRAM_MTP=1'])):
            with self.subTest(unaffected=name):
                ok, rc, err = self.passes(args, extra)
                self.assertTrue(ok, f'rc {rc}: {err}')

    def test_s32_64k_mtp8_engine_line_passes_the_engines_parse_checks(self):
        """Plan A (32 users at 64K context, MTP=5 up to 8 users / MTP=0 above 8 users): --sess 32 --cap 65536 --reuse 1 with G6_DS_MTP=5 passes --check-args; --s2-sessions 33 is refused."""
        x = 'G6_DS_PREFILL_MIN=2 G6_DS_PREFILL_W=4096 G6_DS_MTP=5 G6_DS_ENGRAM=1 G6_DS_ENGRAM_PREFIX=/nonexistent/engram G6_DS_ENGRAM_MTP=1'
        for sess in ('8', '16', '32'):
            with self.subTest(sess=sess):
                args, env = self.line('--sess', sess, '--cap', '65536', '--reuse', '1', '--g6env-x', x)
                self.assertEqual(args[args.index('--s2-sessions') + 1], sess)
                self.assertEqual(args[args.index('--cap') + 1], '65536')
                self.assertIn('--reuse-prefix', args)
                ok, rc, err = self.passes(args, env)
                print(f'  SESS {sess} CAP 65536 MTP 5: rc {rc}: {last_line(err)}', file=sys.stderr)
                self.assertTrue(ok, f'rc {rc}: {err}')
        args, env = self.line('--sess', '32', '--g6env-x', x)
        rc, err = self.run_engine(with_flag(args, '--s2-sessions', '33'), env)
        self.assertNotEqual(rc, 0, err)
        self.assertIn('--s2-sessions 1..32', err)

    def test_negative_control_a_binary_that_does_not_check_fails_the_check(self):
        """The check must not pass for a program that merely exits 0, or that exits nonzero on --check-args."""
        args, env = self.line()
        for exe in ('/bin/false', '/bin/true'):
            with self.subTest(exe=exe):
                ok, rc, err = self.passes(args, env, binary=exe)
                print(f'  control {exe}: rc {rc}, check passed: {ok}', file=sys.stderr)
                self.assertFalse(ok)


if __name__ == '__main__':
    unittest.main()
