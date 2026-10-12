"""Steering release selection against the fake hub: tiny metadata/artifacts, no weights or GPU."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import hfsource
import modelset
import options
import steering_hf
import supervisor
from test_hf import FakeHub, RELEASE, C1, C2

PEERS = '192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4'
WEIGHTS = '5e' * 32


class SteeringHFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'release'
        self.cache = self.root / 'cache'
        self.template = b'{{ messages }}\n'
        for name in hfsource.META_FILES:
            self.put(name, self.template if name == 'chat_template.jinja' else b'{}')
        self.put('config.json', json.dumps({'hidden_size': 6144, 'num_hidden_layers': 78,
                                           '_commit_hash': 'model-rev'}).encode())
        self.put('tokenizer_config.json', b'{"_commit_hash":"token-rev"}')
        self.put('manifests/exlr8-manifest-home.json', json.dumps({'set_id': WEIGHTS, 'layers': {}}).encode())
        self.payload = struct.pack('<6144f', 1., *([0.] * 6143))
        self.manifest = {'schema_version': 1, 'profile_name': 'example', 'scope': 'shared',
                         'layers': [3], 'alphas': [.75],
                         'model': {'name': 'glm-5.3', 'revision': 'model-rev', 'weight_set': WEIGHTS,
                                   'tokenizer_revision': 'token-rev', 'hidden_size': 6144, 'layer_count': 78,
                                   'chat_template_sha256': hashlib.sha256(self.template).hexdigest()},
                         'directions': {'file': 'directions.f32', 'dtype': 'float32-le',
                                        'byte_count': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest()},
                         'provenance': {'extraction_tool': 'unit', 'extraction_commit': 'unit',
                                        'dataset_manifests': {'train': '1' * 64, 'validation': '2' * 64, 'final': '3' * 64}}}
        self.index = {'schema_version': 1, 'default_profile': 'example', 'profiles': []}
        self.pack()
        self.hub = FakeHub({RELEASE: {C1: str(self.model)}}, {RELEASE: {'main': C1}})
        self.addCleanup(self.hub.close)
        self.env = mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': f'http://127.0.0.1:{self.hub.port}', 'HF_TOKEN': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def put(self, name, raw):
        path = self.model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def save_index(self):
        self.put(steering_hf.INDEX, json.dumps(self.index).encode())

    def pack(self, name='example'):
        profile = self.root / ('input-' + name)
        profile.mkdir(exist_ok=True)
        manifest = dict(self.manifest, profile_name=name)
        # Noncanonical input bytes must survive materialization exactly.
        self.raw_manifest = (json.dumps(manifest, indent=2) + '\n').encode()
        (profile / 'manifest.json').write_bytes(self.raw_manifest)
        (profile / 'directions.f32').write_bytes(self.payload)
        file = f'steering/{name}.safetensors'
        steering_hf.release.pack(str(profile), str(self.model / file))
        raw = (self.model / file).read_bytes()
        rec = {'name': name, 'file': file, 'sha256': hashlib.sha256(raw).hexdigest(), 'byte_count': len(raw)}
        self.index['profiles'] = [r for r in self.index['profiles'] if r['name'] != name] + [rec]
        self.save_index()

    def source(self, selector='default', rev='main'):
        source = hfsource.Release(str(self.cache), RELEASE, rev, log=lambda _: None)
        source.fetch_meta(selector)
        return source

    def opts(self, *extra, selector='hf:default'):
        argv = ['--no-config', '--peers', PEERS, '--model', f'https://huggingface.co/{RELEASE}',
                '--model-cache', str(self.cache), '--variant', 'home', '--dflash', '0']
        if selector:
            argv += ['--steering-profile', selector]
        o = options.parse([*argv, *extra], {})
        options.validate(o)
        return o

    def test_default_and_named_exact_bytes_only_selected_artifact(self):
        self.pack('other')
        source = self.source('example')
        directory, manifest = source.steering_profile
        self.assertEqual(manifest['profile_name'], 'example')
        self.assertEqual((Path(directory) / 'manifest.json').read_bytes(),
                         (self.root / 'input-example/manifest.json').read_bytes())
        self.assertEqual((Path(directory) / 'directions.f32').read_bytes(), self.payload)
        self.assertFalse(self.hub.requests('/steering/other'))
        self.assertEqual(self.source().steering_profile[1]['profile_name'], 'example')
        self.assertEqual(self.source('other').steering_profile[1]['profile_name'], 'other')

    def test_stock_ignores_broken_or_absent_steering(self):
        self.put(steering_hf.INDEX, b'bad json')
        o = self.opts(selector=None)
        ms, _, error = supervisor.resolve_model(o, log=lambda _: None)
        self.assertFalse(error, error)
        self.assertIsNone(ms.release.steering_profile)
        self.assertNotIn('--steering-profile', options.engine_plan(o)['args'])
        self.assertFalse(self.hub.requests('/resolve/.*/steering/'))
        shutil.rmtree(self.model / 'steering')
        self.assertIsNone(self.source(None).steering_profile)

    def test_reused_release_without_selection_drops_previous_profile(self):
        source = self.source()
        with mock.patch.object(steering_hf, 'fetch', side_effect=AssertionError('profile accessed')):
            source.fetch_meta()
        self.assertIsNone(source.steering_profile)

    def test_download_errors_are_coordinated_model_errors(self):
        source = self.source(None)
        with mock.patch.object(source.repo, 'fetch', side_effect=RuntimeError('short read')):
            with self.assertRaisesRegex(modelset.ModelError, 'hf:default: short read'):
                steering_hf.fetch(source, 'default')

    def test_actual_identity_and_template_reach_engine(self):
        o = self.opts('--steering-force', '1.25')
        ms, _, error = supervisor.resolve_model(o, log=lambda _: None)
        self.assertFalse(error, error)
        self.assertEqual(o.get('STEERING_WEIGHT_SET'), WEIGHTS)
        self.assertEqual(o.get('STEERING_MODEL_REVISION'), 'model-rev')
        self.assertEqual(o.get('STEERING_TOKENIZER_REVISION'), 'token-rev')
        self.assertEqual(o.get('STEERING_CHAT_TEMPLATE'), str(Path(ms.tokenizer) / 'chat_template.jinja'))
        args = options.engine_plan(o, rank=1, set_id=ms.id)['args']
        for flag, value in (('--steering-weight-set', WEIGHTS), ('--steering-force', '1.25'),
                            ('--steering-profile', o.get('STEERING_PROFILE'))):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_cached_restart_and_rematerialization_without_network(self):
        o = self.opts()
        _, _, error = supervisor.resolve_model(o, log=lambda _: None)
        self.assertFalse(error, error)
        directory = Path(o.get('STEERING_PROFILE'))
        shutil.rmtree(directory)
        with mock.patch('urllib.request.urlopen', side_effect=AssertionError('network used')):
            for opts in (o, self.opts()):
                _, _, error = supervisor.resolve_model(opts, log=lambda _: None)
                self.assertFalse(error, error)
        self.assertEqual((directory / 'manifest.json').read_bytes(), self.raw_manifest)

    def test_refresh_reselects_for_new_commit_on_same_options(self):
        o = self.opts()
        self.assertFalse(supervisor.resolve_model(o, log=lambda _: None)[2])
        other = self.root / 'release2'
        shutil.copytree(self.model, other)
        # A different default at a new commit; reuse the same options after they contain derived local paths.
        self.pack('other')
        shutil.copytree(self.model / 'steering', other / 'steering', dirs_exist_ok=True)
        index = copy.deepcopy(self.index)
        index['default_profile'] = 'other'
        (other / steering_hf.INDEX).write_text(json.dumps(index))
        self.hub.repos[RELEASE][C2] = str(other)
        self.hub.refs[RELEASE]['main'] = C2
        ms, _, error = supervisor.resolve_model(o, refresh=True, log=lambda _: None)
        self.assertFalse(error, error)
        self.assertEqual(ms.release.commit, C2)
        self.assertTrue(o.get('STEERING_PROFILE').endswith('/profiles/other'))

    def test_explicit_identity_mismatch_is_not_overwritten(self):
        for key, wrong in (('model-revision', 'wrong'), ('tokenizer-revision', 'wrong'),
                           ('weight-set', 'a' * 64), ('chat-template-sha256', 'a' * 64)):
            with self.subTest(key=key):
                _, _, error = supervisor.resolve_model(self.opts('--steering-' + key, wrong), log=lambda _: None)
                self.assertIn('explicit STEERING_', error)

    def test_actual_model_identity_mismatches_fail_before_drafter(self):
        original = copy.deepcopy(self.manifest)
        for key, wrong in (('weight_set', 'a' * 64), ('chat_template_sha256', 'a' * 64),
                           ('revision', 'wrong'), ('tokenizer_revision', 'wrong')):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as cache:
                self.manifest = copy.deepcopy(original)
                self.manifest['model'][key] = wrong
                self.pack()
                o = self.opts('--model-cache', cache)
                with mock.patch.object(supervisor, 'drafter_of', side_effect=AssertionError('drafter used')):
                    self.assertTrue(supervisor.resolve_model(o, log=lambda _: None)[2])

    def test_missing_unknown_or_local_source_refused(self):
        with self.assertRaisesRegex(modelset.ModelError, 'no steering profile named'):
            self.source('missing')
        (self.model / steering_hf.INDEX).unlink()
        shutil.rmtree(self.cache)
        with self.assertRaisesRegex(modelset.ModelError, 'needs steering/index.json'):
            self.source()
        for extra in (['--base-packs', str(self.model)], ['--model', str(self.model)]):
            self.assertIn('requires an HF release without --base-packs',
                          supervisor.resolve_model(self.opts(*extra), log=lambda _: None)[2])

    def test_strict_index_schema_and_paths(self):
        original = copy.deepcopy(self.index)
        cases = [[], dict(original, schema_version=True), dict(original, default_profile=[]),
                 dict(original, profiles=[original['profiles'][0]] * 2)]
        for key, value in (('file', '../outside'), ('name', '.hidden'), ('sha256', 1),
                           ('byte_count', True), ('byte_count', steering_hf.MAX_ARTIFACT + 1)):
            candidate = copy.deepcopy(original)
            candidate['profiles'][0][key] = value
            cases.append(candidate)
        for candidate in cases:
            with self.subTest(index=candidate):
                with self.assertRaises(ValueError):
                    steering_hf.parse_index(json.dumps(candidate).encode())
        with self.assertRaises(ValueError):
            steering_hf.parse_index(b'{"schema_version":1,"schema_version":1}')

    def test_bounds_and_published_size_checked_before_download(self):
        self.put(steering_hf.INDEX, b'x' * (steering_hf.release.MAX_HEADER + 1))
        with self.assertRaisesRegex(modelset.ModelError, 'too large'):
            self.source()
        self.assertFalse(self.hub.requests('/resolve/.*/steering/index'))
        self.index['profiles'][0]['byte_count'] += 1
        self.save_index()
        with self.assertRaisesRegex(modelset.ModelError, 'published size'):
            self.source()
        self.assertFalse(self.hub.requests('/steering/example'))

    def test_artifact_index_digest_and_embedded_name_mismatch(self):
        self.index['profiles'][0]['sha256'] = 'a' * 64
        self.save_index()
        with self.assertRaisesRegex(modelset.ModelError, 'size or digest'):
            self.source()
        shutil.rmtree(self.cache)
        self.pack('other')
        raw = (self.model / 'steering/other.safetensors').read_bytes()
        self.put('steering/example.safetensors', raw)
        self.index['profiles'][0].update(sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw))
        self.save_index()
        with self.assertRaisesRegex(modelset.ModelError, 'embedded profile name'):
            self.source()

    def test_cached_index_artifact_and_materialized_corruption_refused(self):
        source = self.source()
        root = Path(source.dir)
        for path in (root / steering_hf.INDEX, root / 'steering/example.safetensors',
                     Path(source.steering_profile[0]) / 'directions.f32'):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b'bad')
                with mock.patch('urllib.request.urlopen', side_effect=AssertionError('network used')):
                    with self.assertRaises(modelset.ModelError):
                        self.source()
                path.write_bytes(original)

    def test_cached_index_without_receipt_is_reverified(self):
        source = self.source()
        receipt = Path(source.dir) / 'steering/.index-identity.json'
        receipt.unlink()
        self.source()
        self.assertTrue(receipt.is_file())
        receipt.unlink()
        (Path(source.dir) / steering_hf.INDEX).write_text('{}')
        with self.assertRaisesRegex(modelset.ModelError, 'pinned release'):
            self.source()

    def test_corrupt_index_receipt_is_not_trusted(self):
        source = self.source()
        path = Path(source.dir) / 'steering/.index-identity.json'
        original = json.loads(path.read_text())
        for changes in ({'commit': C2}, {'sha256': []}, {'byte_count': True}, {'byte_count': 0}):
            with self.subTest(changes=changes):
                path.write_text(json.dumps(dict(original, **changes)))
                with self.assertRaisesRegex(modelset.ModelError, 'cached index identity'):
                    self.source()

    def test_concurrent_selection_publishes_one_identical_directory(self):
        source = self.source(None)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: steering_hf.fetch(source, 'default'), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(self.hub.requests('/steering/example\\.safetensors$')), 1)

    def test_selected_artifact_resume_and_token_boundary(self):
        with mock.patch.dict(os.environ, {'HF_TOKEN': 'fixture-token'}):
            source = self.source(None)
            steering_hf._index(source)
            artifact = Path(source.dir) / 'steering/example.safetensors'
            raw = (self.model / 'steering/example.safetensors').read_bytes()
            Path(str(artifact) + '.part').write_bytes(raw[:2048])
            steering_hf.fetch(source, 'default')
        self.assertEqual(artifact.read_bytes(), raw)
        gets = self.hub.requests('/steering/example\\.safetensors$')
        self.assertEqual(len(gets), 1)
        self.assertEqual(gets[0][2], f'bytes=2048-{len(raw) - 1}')
        self.assertIsNone(gets[0][3])   # LFS payload on the CDN never receives the token.
        heads = self.hub.requests('/steering/example\\.safetensors$', method='HEAD')
        self.assertEqual(heads[0][3], 'Bearer fixture-token')

    def test_valid_index_hash_cannot_hide_a_tampered_payload(self):
        artifact = self.model / 'steering/example.safetensors'
        raw = bytearray(artifact.read_bytes())
        n, = struct.unpack('<Q', raw[:8])
        struct.pack_into('<f', raw, 8 + n, 2.)
        artifact.write_bytes(raw)
        self.index['profiles'][0]['sha256'] = hashlib.sha256(raw).hexdigest()
        self.save_index()
        with self.assertRaises(modelset.ModelError):
            self.source()
        self.assertFalse((self.cache / 'hf/fakeorg--glm-exlr8' / C1 / 'steering/profiles/example').exists())

    @unittest.skipUnless(os.environ.get('STEERING_BOOT_CHECK'), 'build the C++ boot bridge')
    def test_downloaded_profile_enters_actual_cpp_boot_receipt(self):
        o = self.opts()
        self.assertFalse(supervisor.resolve_model(o, log=lambda _: None)[2])
        expected = self.root / 'expected.json'
        expected.write_text(json.dumps(self.manifest['model']))
        command = [os.environ['STEERING_BOOT_CHECK'], o.get('STEERING_PROFILE'), str(expected),
                   o.get('STEERING_CHAT_TEMPLATE'), '3', '-']
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith('G6S2 4 '), result.stdout)
        Path(o.get('STEERING_CHAT_TEMPLATE')).write_text('changed')
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_hf_nohash_refusal_and_box_path_independent_plan(self):
        with self.assertRaisesRegex(options.OptionError, 'weight hash checks'):
            self.opts('--nohash', '1')
        a = self.opts()
        b = self.opts('--model-cache', str(self.root / 'other-cache'))
        for o in (a, b):
            self.assertFalse(supervisor.resolve_model(o, log=lambda _: None)[2])
        digest = lambda o: options.canonical(o, options.engine_plan(o), options.split_list(PEERS))[2]
        self.assertNotEqual(a.get('STEERING_PROFILE'), b.get('STEERING_PROFILE'))
        self.assertEqual(digest(a), digest(b))


if __name__ == '__main__':
    unittest.main()
