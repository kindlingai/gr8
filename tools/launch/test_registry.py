"""Registry consumers and frozen pre-refactor engine command parity (CPU only)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models
import options
ROOT = Path(__file__).resolve().parents[2]
# SHA256 of normalized ENGINE_BIN, ENGINE_ENV and engine_args on the original implementation.
BASELINES = [
    ('glm53', '4', 'fp8', '', '7669284c0a4438c0b7cb208b07b947de06b2a4c63c6a85589bc1b080c7809abb'),
    ('glm53', '4', 'int4', '', '038f739a35ce9794da46bcfbc9409959604b230b50d6e8791d1dff684c9a2205'),
    ('glm53', '6', 'fp8', '', 'e5d813a60b12a5bf6c928bcb271e5e0240547c157940b9dd8d013fe6ae3f686c'),
    ('glm53', '6', 'int4', '', 'aad025af5325d509797adc325282d678f8695d4277131dfd04a773b649da681a'),
    ('glm53', '3', 'fp8', '', 'd5fe690d5714f5f0416af0a6aa157421e46e4b8fc975e0c38673b668c1dcb3b4'),
    ('glm53', '3', 'int4', '', 'd0285c27e92b63bd9b6731b01847c5f7f9e0880293f0045bcf8be219ba6f78e7'),
    ('glm53', '5', 'fp8', '', '8b7b3f1c1256a019108f16afc77d465e33c0615257ba1d3d48e2a66e11566ebd'),
    ('glm53', '5', 'int4', '400000', '05d8dbc910cf666bd3dcda829b4db408a03906476e1e4deaad8c4b4db7a7b8f0'),
    ('glm53f', '4', 'fp8', '', '58a085a8bd65fae9c82c681fb1494c3b521275936ace0a4e7f16316370785519'),
    ('glm53f', '4', 'int4', '', '2d9690f48c9fe061f85026df70324f996cb8c454205ec5692606c77b2799e7d8'),
    ('glm53f', '3', 'fp8', '', '8e82e379c4423912e3ef44c5edeb102e1f1c167ef99eb13e3f8663726e1b560c'),
    ('glm53f', '3', 'int4', '', 'fa9a22fb2b1b0f6fe8d9ec54eb2697458942ab1c653840dcb5883850892715d0'),
    ('glm53f', '2', 'fp8', '', '8e65b6e0542fae9977a6341b70f822371c7c9b0432ac8dff086c1ba7930d5f83'),
    ('glm53f', '2', 'int4', '', '9d164756e667cdfdff7f7184eb7f31b34f80c98931dd2f6b56ff2033e0bbae74'),
    ('glm53f', '1', 'fp8', '', '0fe0f8d5676effcb5d41571becaa710730acb0ae90ecd3d7177d68d922a2a661'),
    ('glm53f', '1', 'int4', '', '60f4509f961494b35ddb0744001c46e99c6c7e929b73bd676962c4a263304b34'),
    ('ds41', '4', 'fp8', '', '87d29e890f4cd1b40c4bb5972d7aaa603ebbdf8c1b7986a9734b32b6959128fb'),
    ('pseudo', '1', 'fp8', '', '4da55a9cecc1bde601e0afb471417298eabc014fe5e58cd04be2b7838377f50a'),
    ('pseudo', '2', 'fp8', '', '4da55a9cecc1bde601e0afb471417298eabc014fe5e58cd04be2b7838377f50a'),
    ('pseudo', '3', 'fp8', '', '4da55a9cecc1bde601e0afb471417298eabc014fe5e58cd04be2b7838377f50a'),
    ('pseudo', '4', 'fp8', '', '4da55a9cecc1bde601e0afb471417298eabc014fe5e58cd04be2b7838377f50a'),
]

class RegistryTests(unittest.TestCase):
    def shell(self, name, tp, fmt, cap='', extra=''):
        env = {'PATH': os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': '', 'GR8_MODEL': name, 'TP': tp,
               'KV_FORMAT': fmt, 'EXTRA_ARGS': extra}
        if cap:
            env['CAP'] = cap
        script = '. "$1"; engine_defaults || exit; engine_env || exit; echo "$ENGINE_BIN"; echo "$ENGINE_ENV"; engine_args 0 /packs /aux /work 7'
        return subprocess.check_output(['bash', '-c', script, 'test', str(ROOT / 'scripts/engine_args.sh')], env=env, text=True)

    def test_all_builtin_commands_match_frozen_baseline(self):
        for name, tp, fmt, cap, digest in BASELINES:
            with self.subTest(model=name, tp=tp, fmt=fmt):
                text = self.shell(name, tp, fmt, cap)
                normalized = '\n'.join(' '.join(l.split()) for l in text.splitlines())
                self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(), digest)

    def test_extra_args_reach_every_model_and_tp_once(self):
        for name, tp, fmt, cap, _ in BASELINES:
            with self.subTest(model=name, tp=tp, fmt=fmt):
                args = self.shell(name, tp, fmt, cap, '--top 5').splitlines()[-1].split()
                self.assertEqual(args.count('--top'), 1)
                self.assertEqual(args[args.index('--top') + 1], '5')

    def test_tp_defaults_origin_and_explicit_precedence(self):
        env = {'GR8_MODEL':'glm53', 'TP':'4'}
        o = options.parse(['--no-config'], env)
        self.assertEqual((o.get('KV_FORMAT'), o.origin['KV_FORMAT']), ('int4', 'glm53 TP=4 default'))
        for args, changes, origin in [(['--kv-format', 'fp8'], {}, 'command line'), ([], {'KV_FORMAT':'fp8'}, 'environment')]:
            o = options.parse(['--no-config', *args], {**env, **changes})
            self.assertEqual((o.get('KV_FORMAT'), o.origin['KV_FORMAT']), ('fp8', origin))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'cluster.env'
            p.write_text('KV_FORMAT=fp8\n')
            o = options.parse(['@'+str(p)], env)
            self.assertEqual((o.get('KV_FORMAT'), o.origin['KV_FORMAT']), ('fp8', '@file'))
        inferred = options.parse(['--no-config', '--peers', '192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4'], {'GR8_MODEL':'glm53'})
        self.assertEqual(inferred.origin['KV_FORMAT'], 'glm53 TP=4 default')
        # The bare shell path reads exactly the same default table.
        output = subprocess.check_output(['bash', '-c', '. "$1"; engine_defaults; echo "$KV_FORMAT"', 'test', str(ROOT/'scripts/engine_args.sh')], env={'PATH':os.environ['PATH'], 'TP':'4', 'CUDA_VISIBLE_DEVICES':''}, text=True)
        self.assertEqual(output.strip(), 'int4')

    def test_unknown_tp_default_is_an_option_error(self):
        with mock.patch.dict(models.GLM53.tp_defaults, {'4': {'NOT_A_KNOB': '1'}}):
            with self.assertRaisesRegex(options.OptionError, 'unknown option NOT_A_KNOB'):
                options.parse(['--no-config'], {'GR8_MODEL': 'glm53', 'TP': '4'})

    def test_registry_queries_and_image_layout(self):
        self.assertEqual(models.query(['glm53', 'build']), 'glm')
        self.assertEqual(models.query(['glm53', 'default-kv-format']), 'int4')
        self.assertEqual(models.query(['glm53', 'default-kv-format', '6']), 'fp8')
        for name in ('g6m4_flash', 'g6m4_flash-int4', 'm4sim_flash', 'kvman-remote'):
            self.assertIn(name, models.image_binaries())
        self.assertNotIn('g6pseudo', models.image_binaries())
        docker = (ROOT/'Dockerfile').read_text()
        for name in models.image_binaries():
            self.assertIn('/'+name, docker)
        for name in models.MODELS:
            self.assertTrue((ROOT / models.query([name, 'smoke_script'])).is_file())
        with self.assertRaises(ValueError): models.query(['missing','build'])

    def test_build_shell_is_quoted_and_model_specific(self):
        for name, model in models.MODELS.items():
            shell = models.query([model.build, 'build-shell'])
            out = subprocess.check_output(['bash', '-c', shell + '; printf "%s\n" "$BUILD_ENGINE" "$BUILD_DEFINE" "$BUILD_FC2GL"'], text=True).splitlines()
            self.assertEqual(out, [model.engine, model.build_define, str(model.build_fc2gl)])


    def test_shipped_build_metadata_matches_registry(self):
        source = (ROOT / 'scripts/build.sh').read_text()
        function = source[source.index('build_meta() {'):source.index('\nbuild_meta "')]
        # Compare every BUILD_* variable, including future additions and multiline values.
        dump = '\nfor key in ${!BUILD_@}; do printf "%s\\0%s\\0" "$key" "${!key}"; done'
        for model in models.SERVED:
            with self.subTest(model=model.build):
                env = {'PATH': '/nonexistent', 'MODEL': model.build, 'REPO': '/nonexistent'}
                actual = subprocess.check_output(['/bin/bash', '-ec', function + '\nbuild_meta "$MODEL"' + dump], env=env)
                expected = subprocess.check_output(['/bin/bash', '-ec', models.query([model.build, 'build-shell']) + dump], env=env)
                self.assertEqual(actual, expected)

    def test_build_dispatch_without_cuda(self):
        import shutil
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as t:
            work = Path(t)
            repo = work / 'repo'
            scripts = repo / 'scripts'
            runtime = repo / 'engine/runtime'
            bin_dir = work / 'bin'
            for directory in (scripts, runtime, bin_dir):
                directory.mkdir(parents=True)
            for name in ('build.sh', 'ds41_build_flags.sh'):
                shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
            # Match engine-src's script isolation, with only the compiler/runtime mocked.
            tool_paths = {
                name: shutil.which(name, path=os.defpath)
                for name in ('bash', 'dirname', 'mkdir', 'mv', 'rm', 'grep', 'md5sum',
                             'strings', 'basename', 'cat', 'tr', 'seq', 'chmod', 'wc',
                             'touch', 'ls', 'sed', 'tail')
            }
            missing = [name for name, path in tool_paths.items() if path is None]
            if missing:
                self.skipTest('engine-src build tools unavailable: ' + ', '.join(missing))
            for name, path in tool_paths.items():
                (bin_dir / name).symlink_to(path)
            self.assertIsNone(shutil.which('python3', path=str(bin_dir)))
            self.assertIsNone(shutil.which('python', path=str(bin_dir)))
            self.assertFalse((repo / 'tools/launch/models.py').exists())
            (runtime / 'build_m4.sh').write_text('''#!/bin/bash
set -eu
mkdir -p "$OUT"
for binary in g6m4 m4sim; do
  cat > "$OUT/$binary" <<'BIN'
#!/bin/bash
# int4g32h int4nope G6_CP_PF_HASH2 build
# flash kv: borrow-idle, %u of
# flash ckpt share: now
n=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tp) n=$2; shift ;;
    --ips) ips=$2; commas=${ips//[^,]/}; n=$((${#commas}+1)); shift ;;
  esac
  shift
done
echo "--top-p-cands must be 0..$((2048 / n))"
BIN
  printf '# %s\n' "$WIDTHS|$EXTRA|$2" >> "$OUT/$binary"
  chmod +x "$OUT/$binary"
done
''')
            nvcc = bin_dir / 'nvcc'
            nvcc.write_text('''#!/bin/bash
set -eu
while [ "$1" != -o ]; do shift; done
printf '#!/bin/bash\nexit 0\n' > "$2"
chmod +x "$2"
''')
            nvcc.chmod(0o755)
            cases = [(models.GLM53, 'fp8', {}), (models.GLM53, 'int4', {}),
                     (models.GLM53, 'fp8', {'GR8_TESTS': 'only'}),
                     (models.GLM53F, 'fp8', {}), (models.GLM53F, 'int4', {}),
                     (models.DS41, 'fp8', {}), (models.DS41, 'fp8', {'DS41_WHAT': 'sim'}),
                     (models.DS41, 'int4', {})]

            def run_case(model, fmt, changes, index):
                out = work / ('out-' + str(index))
                env = {'PATH': str(bin_dir), 'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': t,
                       'MODEL': model.build, 'KV_FORMAT': fmt, 'GR8_TESTS': '0',
                       'GR8_XGRAMMAR': '0', 'GR8_NVCC_CACHE': '0', 'JOBS': '2',
                       'EXTRA': '-DUSER=1', 'MOCK_NVCC': str(nvcc), **changes}
                # Functions outrank /usr/local/cuda/bin, which build.sh prepends.
                wrapper = ('nvcc() { "$MOCK_NVCC" "$@"; }; '
                           'nvidia-smi() { return 1; }; '
                           'export -f nvcc nvidia-smi; exec bash "$@"')
                result = subprocess.run(['/bin/bash', '-c', wrapper, 'test',
                                         str(scripts / 'build.sh'), str(out)],
                                        env=env, capture_output=True, text=True)
                if fmt == 'int4' and not model.int4_kv:
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn('fp8 only', result.stdout)
                    return
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if changes.get('GR8_TESTS') == 'only':
                    self.assertFalse((out / model.engine).exists())
                    self.assertTrue((out / 'sampler_u01_test').exists())
                    return
                sim_only = changes.get('DS41_WHAT') == 'sim'
                binary = model.simulator if sim_only else model.engine + ('-int4' if fmt == 'int4' else '')
                content = (out / binary).read_text()
                self.assertIn(' '.join(model.build_widths) + '|', content)
                self.assertIn('-DPF_FC2GL=' + str(model.build_fc2gl), content)
                self.assertIn('-DUSER=1', content)
                if model.build_define:
                    self.assertIn('-D' + model.build_define + '=1', content)
                if model.build_flags_script:
                    self.assertIn('-DDS_MHC_TRUNK_BF16', content)
                self.assertEqual((out / model.simulator).exists(), fmt == 'fp8')
                if sim_only:
                    self.assertFalse((out / model.engine).exists())

            for index, (model, fmt, changes) in enumerate(cases):
                with self.subTest(model=model.build, fmt=fmt, changes=changes):
                    run_case(model, fmt, changes, index)

            # Unknown targets must fail clearly without a checkout registry/Python.
            result = subprocess.run(['/bin/bash', str(scripts / 'build.sh'), str(work / 'unknown')],
                                    env={'PATH': str(bin_dir), 'CUDA_VISIBLE_DEVICES': '',
                                         'MODEL': 'qwen', 'GR8_XGRAMMAR': '0'},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('MODEL=qwen: glm, flash or ds41', result.stderr)

            # Full-checkout fallback: use the real query implementation with a fixture registration.
            registry = repo / 'tools/launch/models.py'
            registry.parent.mkdir(parents=True)
            source = (ROOT / 'tools/launch/models.py').read_text()
            registration = "\nfrom dataclasses import replace\nMODELS['qwen'] = replace(GLM53F, name='qwen', build='qwen', engine='g6m4_qwen', simulator='m4sim_qwen', build_define='G6_MODEL_QWEN', build_widths=('2',))\n"
            registry.write_text(source.replace("if __name__ == '__main__':", registration + "\nif __name__ == '__main__':"))
            (bin_dir / 'python3').symlink_to(sys.executable)
            custom = replace(models.GLM53F, name='qwen', build='qwen', engine='g6m4_qwen',
                             simulator='m4sim_qwen', build_define='G6_MODEL_QWEN', build_widths=('2',))
            for fmt in ('fp8', 'int4'):
                with self.subTest(model='qwen', fmt=fmt):
                    run_case(custom, fmt, {}, 'qwen-' + fmt)
