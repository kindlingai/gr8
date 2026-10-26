#!/usr/bin/env python3
"""#179: the DFlash2 drafter and the steering artifacts with --fetch model-loader, against fake_model_loader.py. The
drafter's own catalog entry in model-loader, converted on the box with a stub in place of bin/g6load dflash2 (as
test_hf.DrafterTests); the steering index and artifact from the release's catalog entry (as test_steering_hf).
    python3 -m unittest discover -s tools/launch"""
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake_model_loader  # noqa: E402
import hfsource  # noqa: E402
import mlsource  # noqa: E402
import modelset  # noqa: E402
import options as opt  # noqa: E402
import steering_hf  # noqa: E402
import supervisor as S  # noqa: E402
import test_hf as T  # noqa: E402
from test_modelset import make_release as local_release  # noqa: E402

PEERS4 = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'
PEERS6 = ','.join(f'10.0.0.{i}' for i in range(1, 7))
DRAFTER = 'https://huggingface.co/fakeorg/drafter'
WEIGHTS = '5e' * 32


def ro(tree):
    """Make a directory tree read-only, as the store mounted :ro is; returns a function that undoes it."""
    for d, dirs, files in os.walk(tree):
        for n in files:
            if not n.endswith('.sock'):          # a :ro mount still lets gr8 open model-loader's socket
                os.chmod(os.path.join(d, n), 0o444)
        os.chmod(d, 0o555)

    def undo():
        for d, dirs, files in os.walk(tree):
            os.chmod(d, 0o755)
    return undo


class DrafterTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix='gr8mld-')
        self.addCleanup(shutil.rmtree, self.t, True)
        self.src = os.path.join(self.t, 'drafter')
        os.makedirs(self.src)
        with open(os.path.join(self.src, 'config.json'), 'w') as f:
            f.write('{"hidden_size": 6144}')
        with open(os.path.join(self.src, 'model.safetensors'), 'wb') as f:
            f.write(os.urandom(50_000))
        with open(os.path.join(self.src, 'README.md'), 'w') as f:
            f.write('the drafter\n')
        self.store, self.home = os.path.join(self.t, 'store'), os.path.join(self.t, 'home')
        self.ml = fake_model_loader.FakeModelLoader(self.store, {'drafter': {'main': (self.src, T.C1)}},
                                                    repos={'drafter': 'fakeorg/drafter'})
        self.addCleanup(self.ml.close)
        stub = os.path.join(self.t, 'convert.py')
        with open(stub, 'w') as f:
            f.write(T.STUB_CONVERT)
        for p in (mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': 'http://127.0.0.1:9'}),   # no hub: gr8 fetches nothing
                  mock.patch.object(hfsource, 'converter', lambda: [sys.executable, stub])):
            p.start()
            self.addCleanup(p.stop)
        self.rel = local_release(os.path.join(self.t, 'local'))
        self.logs = []

    def boot(self, *argv, peers=PEERS4):
        o = opt.parse(['--no-config', '--peers', peers, '--model', self.rel, '--fetch', 'model-loader',
                       '--model-cache', self.store, '--g6-home', self.home, '--drafter', DRAFTER, *argv], T.GLM)
        opt.validate(o)
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertEqual(err, '')
        return o, ms, v

    def test_fetch_through_model_loader_and_convert_on_the_box(self):
        o, ms, v = self.boot()
        d = ms.drafter
        self.assertIsInstance(d, mlsource.Drafter)
        self.assertEqual((d.id, str(d)), (f'fakeorg/drafter@{T.C1}', f'fakeorg/drafter@{T.C1[:12]}'))
        self.assertEqual(o.get('DFLASH_DIR'), d.out)
        self.assertTrue(d.out.startswith(os.path.join(self.home, 'run', 'model-loader')))
        self.assertTrue(d.check(2))
        undo = ro(self.store)               # the store is read-only to gr8: the conversion writes under G6_HOME
        self.addCleanup(undo)
        d.fetch_rank(2)
        self.assertEqual(d.check(2), [])
        self.assertEqual(sorted(os.listdir(d.out)), ['dflash.r2.manifest.json', 'dflash.r2.pack', 'rank2.COMPLETE'])
        name, body = self.ml.puts[-1]
        self.assertEqual((name, body['model'], sorted(body['include'])),
                         ('gr8-glm53-drafter', 'drafter', ['config.json', 'model.safetensors']))
        with open(os.path.join(self.t, 'convert.args'), 'w'):
            pass
        undo()
        # model-loader holds the drafter's own files and nothing converted
        held = self.ml.files(os.path.join(self.store, 'models'))
        self.assertEqual(sorted(held), ['drafter/main/config.json', 'drafter/main/model.safetensors'])
        self.assertFalse([p for p in self.ml.files(self.store) if 'dflash' in p])
        plan = opt.engine_plan(o, rank=2, rank_dir='/r', aux_dir='/a', home='/h', epoch=1)
        self.assertIn(os.path.join(d.out, 'dflash.r2.pack'), plan['args'])

    def test_the_converter_reads_the_store_copy(self):
        o, ms, v = self.boot()
        ms.drafter.fetch_rank(1)
        with open(os.path.join(ms.drafter.out, 'rank1.COMPLETE')) as f:
            self.assertEqual(json.load(f)['drafter'], ms.drafter.id)

    def test_a_second_boot_converts_nothing_and_repins(self):
        o, ms, v = self.boot()
        ms.drafter.fetch_rank(3)
        o, ms, v = self.boot()
        puts = len(self.ml.puts)
        with mock.patch.object(hfsource, 'convert_drafter', side_effect=AssertionError('converted again')):
            ms.drafter.fetch_rank(3)
        self.assertGreater(len(self.ml.puts), puts)

    def test_a_conversion_from_another_commit_is_redone(self):
        o, ms, v = self.boot()
        ms.drafter.fetch_rank(0)
        m = os.path.join(ms.drafter.out, 'rank0.COMPLETE')
        with open(m) as f:
            c = json.load(f)
        with open(m, 'w') as f:
            json.dump(dict(c, drafter='fakeorg/drafter@' + T.C2), f)
        self.assertIn('was converted from', ms.drafter.check(0)[0])
        ms.drafter.fetch_rank(0)
        self.assertEqual(ms.drafter.check(0), [])

    def test_tp6_replicated(self):
        o, ms, v = self.boot('--dflash', '1', peers=PEERS6)
        self.assertTrue(ms.drafter.out.endswith('tp6'))
        with mock.patch.dict(os.environ, {'STUB_WANT_NR': '6'}):
            ms.drafter.fetch_rank(5)
        self.assertEqual(ms.drafter.check(5), [])

    def test_a_failed_conversion_leaves_nothing(self):
        o, ms, v = self.boot()
        with mock.patch.dict(os.environ, {'STUB_CONVERT_FAIL': '1'}):
            with self.assertRaises(modelset.ModelError) as e:
                ms.drafter.fetch_rank(1)
        self.assertIn('hidden_size = 1', str(e.exception))
        self.assertFalse(os.path.exists(ms.drafter.out))
        self.assertEqual(os.listdir(os.path.dirname(ms.drafter.out)), [])
        self.assertTrue(ms.drafter.check(1))

    def test_drafter_catalog_name_and_ids_agree_with_legacy(self):
        self.ml.catalog['glm-drafter'] = {'v1': (self.src, T.C2)}
        self.ml.repos['glm-drafter'] = 'fakeorg/drafter'
        # two entries for the repo: the @rev picks one, with no name given
        o, ms, v = self.boot('--drafter', DRAFTER + '@v1')
        self.assertEqual(ms.drafter.model, 'glm-drafter')
        o, ms, v = self.boot('--model-loader-drafter-model', 'glm-drafter', '--drafter', DRAFTER + '@v1')
        self.assertEqual((ms.drafter.revision, ms.drafter.commit), ('v1', T.C2))
        self.assertEqual(ms.drafter.id, f'fakeorg/drafter@{T.C2}')        # as hfsource.Drafter's at that commit

    def test_missing_drafter_files_are_named(self):
        os.unlink(os.path.join(self.src, 'model.safetensors'))
        o = opt.parse(['--no-config', '--peers', PEERS4, '--model', self.rel, '--fetch', 'model-loader',
                       '--model-cache', self.store, '--g6-home', self.home, '--drafter', DRAFTER], T.GLM)
        opt.validate(o)
        ms, v, err = S.resolve_model(o, log=self.logs.append)
        self.assertIn('the drafter fakeorg/drafter@', err)
        self.assertIn('has no model.safetensors', err)


@unittest.skipUnless(T.HAVE, 'needs numpy to write the synthetic release')
class SteeringTests(unittest.TestCase):
    """--steering-profile hf:NAME with --fetch model-loader: the index and the artifact come through the release's
    request, are checked as the legacy fetch checks them, and are unpacked under G6_HOME."""

    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.mkdtemp(prefix='gr8mls-')
        cls.rel = os.path.join(cls.t, 'release')
        T.make_release(cls.rel)
        template = open(os.path.join(cls.rel, 'chat_template.jinja'), 'rb').read()
        with open(os.path.join(cls.rel, 'config.json'), 'w') as f:
            json.dump({'hidden_size': 6144, 'num_hidden_layers': 78, '_commit_hash': 'model-rev'}, f)
        with open(os.path.join(cls.rel, 'tokenizer_config.json'), 'w') as f:
            json.dump({'_commit_hash': 'token-rev'}, f)
        cls.payload = struct.pack('<6144f', 1., *([0.] * 6143))
        cls.index = {'schema_version': 1, 'default_profile': 'example', 'profiles': []}
        for name in ('example', 'other'):
            prof = os.path.join(cls.t, 'in-' + name)
            os.makedirs(prof)
            man = {'schema_version': 1, 'profile_name': name, 'scope': 'shared', 'layers': [3], 'alphas': [.75],
                   'model': {'name': 'glm-5.3', 'revision': 'model-rev', 'weight_set': WEIGHTS,
                             'tokenizer_revision': 'token-rev', 'hidden_size': 6144, 'layer_count': 78,
                             'chat_template_sha256': hashlib.sha256(template).hexdigest()},
                   'directions': {'file': 'directions.f32', 'dtype': 'float32-le', 'byte_count': len(cls.payload),
                                  'sha256': hashlib.sha256(cls.payload).hexdigest()},
                   'provenance': {'extraction_tool': 'unit', 'extraction_commit': 'unit',
                                  'dataset_manifests': {'train': '1' * 64, 'validation': '2' * 64, 'final': '3' * 64}}}
            with open(os.path.join(prof, 'manifest.json'), 'w') as f:
                json.dump(man, f)
            with open(os.path.join(prof, 'directions.f32'), 'wb') as f:
                f.write(cls.payload)
            art = os.path.join(cls.rel, 'steering', f'{name}.safetensors')
            os.makedirs(os.path.dirname(art), exist_ok=True)
            steering_hf.release.pack(prof, art)
            raw = open(art, 'rb').read()
            cls.index['profiles'].append({'name': name, 'file': f'steering/{name}.safetensors',
                                          'sha256': hashlib.sha256(raw).hexdigest(), 'byte_count': len(raw)})
        with open(os.path.join(cls.rel, steering_hf.INDEX), 'w') as f:
            json.dump(cls.index, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.t, ignore_errors=True)

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='gr8mlss-', dir=self.t)
        self.store, self.home = os.path.join(self.d, 'store'), os.path.join(self.d, 'home')
        self.ml = fake_model_loader.FakeModelLoader(self.store, {'glm-5.3-exlr8': {'main': (self.rel, T.C1)}},
                                                    repos={'glm-5.3-exlr8': T.RELEASE})
        self.addCleanup(self.ml.close)
        p = mock.patch.dict(os.environ, {'GR8_HF_ENDPOINT': 'http://127.0.0.1:9'})
        p.start()
        self.addCleanup(p.stop)

    def opts(self, selector='hf:default', *extra):
        o = opt.parse(['--no-config', '--peers', PEERS4, '--model', f'https://huggingface.co/{T.RELEASE}',
                       '--model-cache', self.store, '--fetch', 'model-loader', '--g6-home', self.home,
                       '--variant', 'home', '--dflash', '0', '--steering-profile', selector, *extra], T.GLM)
        opt.validate(o)
        return o

    def test_selected_profile_only_and_unpacked_under_home(self):
        o = self.opts('hf:other')
        ms, v, err = S.resolve_model(o, log=lambda _: None)
        self.assertEqual(err, '')
        directory, manifest = ms.release.steering_profile
        self.assertEqual(manifest['profile_name'], 'other')
        self.assertEqual(o.get('STEERING_PROFILE'), directory)
        self.assertTrue(directory.startswith(os.path.join(self.home, 'run', 'model-loader')))
        self.assertEqual(Path(directory, 'directions.f32').read_bytes(), self.payload)
        self.assertEqual(o.get('STEERING_WEIGHT_SET'), WEIGHTS)
        held = self.ml.files(os.path.join(self.store, 'models', 'glm-5.3-exlr8', 'main', 'steering'))
        self.assertEqual(sorted(held), ['index.json', 'other.safetensors'])   # not example's
        # fetch_rank keeps the steering files in the boot's request
        ms.release.fetch_rank(4, 1, v)
        inc = self.ml.puts[-1][1]['include']
        self.assertIn('steering/other.safetensors', inc)
        self.assertIn(steering_hf.INDEX, inc)

    def test_the_store_is_read_only(self):
        S.resolve_model(self.opts(), log=lambda _: None)             # model-loader fetches them once
        undo = ro(self.store)
        self.addCleanup(undo)
        shutil.rmtree(os.path.join(self.home, 'run'))                  # gr8's own copies: unpacked again
        o = self.opts()
        ms, v, err = S.resolve_model(o, log=lambda _: None)
        self.assertEqual(err, '')
        self.assertTrue(os.path.isfile(os.path.join(o.get('STEERING_PROFILE'), 'manifest.json')))

    def test_an_unknown_profile_and_a_bad_digest(self):
        ms, v, err = S.resolve_model(self.opts('hf:nope'), log=lambda _: None)
        self.assertIn('no steering profile named', err)
        self.assertEqual(S.resolve_model(self.opts(), log=lambda _: None)[2], '')
        self.ml.corrupt('glm-5.3-exlr8', 'main', 'steering/example.safetensors', 100)
        ms, v, err = S.resolve_model(self.opts(), log=lambda _: None)
        self.assertIn('size or digest mismatch', err)

    def test_a_changed_index_is_not_trusted(self):
        S.resolve_model(self.opts(), log=lambda _: None)
        p = os.path.join(self.store, 'models', 'glm-5.3-exlr8', 'main', steering_hf.INDEX)
        os.chmod(p, 0o644)
        with open(p, 'rb') as f:
            raw = f.read()
        new = json.dumps(dict(self.index, default_profile='other')).encode()
        with open(p, 'wb') as f:                  # the same length, so it parses and the sizes agree
            f.write(new + b' ' * (len(raw) - len(new)))
        ms, v, err = S.resolve_model(self.opts(), log=lambda _: None)
        self.assertIn('digest mismatch', err)                          # the receipt under G6_HOME catches it

    def test_no_steering_files_without_a_selection(self):
        o = opt.parse(['--no-config', '--peers', PEERS4, '--model', f'https://huggingface.co/{T.RELEASE}',
                       '--model-cache', self.store, '--fetch', 'model-loader', '--g6-home', self.home,
                       '--dflash', '0'], T.GLM)
        opt.validate(o)
        ms, v, err = S.resolve_model(o, log=lambda _: None)
        self.assertEqual((err, ms.release.steering_profile), ('', None))
        self.assertFalse(os.path.exists(os.path.join(self.store, 'models', 'glm-5.3-exlr8', 'main', 'steering')))


if __name__ == '__main__':
    unittest.main()
