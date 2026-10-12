"""Local profile identity, launcher options and the actual C++ boot preparation bridge. No models/GPU/network."""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelset
import options
import steering_local
import supervisor
from test_modelset import make_packs

PEERS = '192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4'


class FixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        packs = make_packs(str(self.root / 'packs'))
        self.ms = modelset.resolve(packs, None, 4)
        self.template = Path(self.ms.tokenizer, 'chat_template.jinja')
        self.digest = options.file_sha256(self.template)
        self.profile = self.root / 'profile'
        self.profile.mkdir()
        payload = struct.pack('<' + 'f' * 6144, 1.0, *([0.] * 6143)) * 2
        self.manifest = {'schema_version': 1, 'profile_name': 'example', 'scope': 'shared',
                         'layers': [0, 3], 'alphas': [.5, 1.],
                         'model': {'name': 'glm-5.3', 'revision': 'rev-1', 'weight_set': self.ms.id,
                                   'tokenizer_revision': 'tok-1', 'chat_template_sha256': self.digest,
                                   'hidden_size': 6144, 'layer_count': 78},
                         'directions': {'file': 'directions.f32', 'dtype': 'float32-le', 'byte_count': len(payload),
                                        'sha256': hashlib.sha256(payload).hexdigest()},
                         'provenance': {'extraction_tool': 'unit', 'extraction_commit': 'unit',
                                        'dataset_manifests': {'train': '1' * 64, 'validation': '2' * 64, 'final': '3' * 64}}}
        (self.profile / 'directions.f32').write_bytes(payload)
        self.save_manifest()

    def save_manifest(self):
        (self.profile / 'manifest.json').write_text(json.dumps(self.manifest))

    def opts(self, *extra):
        o = options.parse(['--no-config', '--peers', PEERS, '--model', str(self.root / 'packs'),
                           '--steering-profile', str(self.profile), '--steering-model-revision', 'rev-1',
                           '--steering-tokenizer-revision', 'tok-1', '--steering-chat-template-sha256', self.digest,
                           *extra], {})
        options.validate(o)
        return o

class LocalTests(FixtureTestCase):
    def test_actual_template_and_weight_identity_reach_engine(self):
        o = self.opts('--steering-force', '1.25')
        self.assertEqual(steering_local.prepare(o, self.ms), self.manifest)
        self.assertEqual(o.get('STEERING_WEIGHT_SET'), self.ms.id)
        args = options.engine_plan(o, rank=1, set_id=self.ms.id)['args']
        for flag, value in (('--steering-profile', str(self.profile)), ('--steering-chat-template', str(self.template)),
                            ('--steering-weight-set', self.ms.id), ('--steering-force', '1.25')):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_no_profile_does_no_profile_io(self):
        o = options.parse(['--no-config', '--peers', PEERS], {})
        shutil.rmtree(self.profile)
        options.validate(o)
        self.assertIsNone(steering_local.prepare(o, self.ms))
        self.assertNotIn('--steering-profile', options.engine_plan(o)['args'])

    def test_tampered_template_and_payload_refused(self):
        o = self.opts()
        self.template.write_text('changed')
        with self.assertRaisesRegex(modelset.ModelError, 'served chat template'):
            steering_local.prepare(o, self.ms)
        self.template.write_text('{"chat_template.jinja": 1}')
        (self.profile / 'directions.f32').write_bytes(b'bad')
        with self.assertRaises(modelset.ModelError):
            steering_local.prepare(o, self.ms)

    def test_wrong_weight_set_and_missing_profile_refused(self):
        with self.assertRaisesRegex(modelset.ModelError, 'resolved weight set'):
            steering_local.prepare(self.opts('--steering-weight-set', 'a' * 64), self.ms)
        shutil.rmtree(self.profile)
        with self.assertRaises(modelset.ModelError):
            steering_local.prepare(self.opts(), self.ms)

    def test_release_weights_use_published_content_identity(self):
        weights = 'a' * 64
        self.ms.set['source'] = {'kind': 'exlr8-release', 'weights': weights}
        self.manifest['model']['weight_set'] = weights
        self.save_manifest()
        o = self.opts()
        steering_local.prepare(o, self.ms)
        self.assertEqual(o.get('STEERING_WEIGHT_SET'), weights)

    def test_actual_config_dimensions_and_revisions_checked(self):
        config = self.template.parent / 'config.json'
        for body in ([], {'hidden_size': 1, 'num_hidden_layers': 78},
                     {'hidden_size': 6144, 'num_hidden_layers': 46},
                     {'hidden_size': 6144, 'num_hidden_layers': 78, '_commit_hash': 'wrong'}):
            config.write_text(json.dumps(body))
            with self.assertRaises(modelset.ModelError):
                steering_local.prepare(self.opts(), self.ms)
        config.write_text(json.dumps({'hidden_size': 6144, 'num_hidden_layers': 78, '_commit_hash': 'rev-1'}))
        token = self.template.parent / 'tokenizer_config.json'
        token.write_text('[]')
        with self.assertRaisesRegex(modelset.ModelError, 'must be an object'):
            steering_local.prepare(self.opts(), self.ms)
        token.write_text('{"_commit_hash":"wrong"}')
        with self.assertRaisesRegex(modelset.ModelError, 'tokenizer revision'):
            steering_local.prepare(self.opts(), self.ms)
        token.write_text('{"_commit_hash":"tok-1"}')
        steering_local.prepare(self.opts(), self.ms)
        root_config = self.root / 'config.json'
        root_config.write_text('{"hidden_size":1,"num_hidden_layers":78}')
        o = self.opts()
        o.values['V2_DENSE'] = str(self.root)
        self.ms.set['source'] = {'kind': 'exlr8-release', 'weights': self.ms.id}
        with self.assertRaises(modelset.ModelError):
            steering_local.prepare(o, self.ms)

    def test_mixed_external_weights_refused(self):
        for key in ('BASE_PACKS', 'V2', 'V2_DENSE'):
            o = self.opts()
            o.values[key] = '/external/weights'
            with self.assertRaisesRegex(modelset.ModelError, 'mixed external weights'):
                steering_local.prepare(o, self.ms)

    def test_force_and_selector_validation(self):
        with self.assertRaisesRegex(options.OptionError, 'weight hash checks'):
            self.opts('--nohash', '1')
        for force in ('0', '-1', '4.1', 'nan', 'inf', 'word', '1e-50'):
            with self.assertRaises(options.OptionError, msg=force):
                self.opts('--steering-force', force)
        for force in ('1e-45', '4'):
            self.opts('--steering-force', force)
        self.opts('--steering-profile', 'hf:default')
        for selector in ('hf:', 'hf:../bad', 'hf:.bad', 'hf:a/b', 'hf:' + 'a' * 129):
            with self.assertRaises(options.OptionError):
                self.opts('--steering-profile', selector)
        with self.assertRaisesRegex(options.OptionError, 'needs --steering-profile'):
            o = options.parse(['--no-config', '--peers', PEERS, '--steering-force', '1'], {})
            options.validate(o)

    def test_profile_paths_differ_but_identity_and_force_agree(self):
        a = self.opts()
        steering_local.prepare(a, self.ms)
        other = self.root / 'other'
        shutil.copytree(self.profile, other)
        b = self.opts('--steering-profile', str(other))
        steering_local.prepare(b, self.ms)
        digest = lambda o: options.canonical(o, options.engine_plan(o), options.split_list(PEERS))[2]
        self.assertEqual(digest(a), digest(b))
        b.values['STEERING_FORCE'] = '2'
        self.assertNotEqual(digest(a), digest(b))

    def test_resolve_validates_before_drafter(self):
        ms, _, error = supervisor.resolve_model(self.opts(), log=lambda _: None)
        self.assertFalse(error, error)
        self.assertIsNotNone(ms)


@unittest.skipUnless(os.environ.get('STEERING_BOOT_CHECK'), 'build the C++ boot bridge')
class BridgeTests(FixtureTestCase):
    def run_bridge(self, force='-', layers='0,3', profile=None, template=None, expected_model=None):
        expected = self.root / 'expected.json'
        expected.write_text(json.dumps(expected_model or self.manifest['model']))
        return subprocess.run([os.environ['STEERING_BOOT_CHECK'], str(profile or self.profile), str(expected),
                               str(template or self.template), layers, force], text=True, capture_output=True)

    def test_loaded_receipt_and_rank_disagreement(self):
        a = self.run_bridge()
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertTrue(a.stdout.startswith('G6S2 4 '), a.stdout)
        self.manifest['alphas'][0] = .75
        self.save_manifest()
        schedule = self.run_bridge()
        self.assertEqual(schedule.returncode, 0, schedule.stderr)
        self.assertNotEqual(a.stdout, schedule.stdout)
        self.assertNotEqual(self.run_bridge(expected_model=dict(self.manifest['model'], revision='wrong')).returncode, 0)
        self.manifest['profile_name'] = 'other'
        self.save_manifest()
        b = self.run_bridge()
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertNotEqual(a.stdout, b.stdout)
        self.assertNotEqual(b.stdout, self.run_bridge(force='2').stdout)

    def test_missing_layers_duplicates_template_and_payload_fail_closed(self):
        for layers in ('0', '0,3,3'):
            self.assertNotEqual(self.run_bridge(layers=layers).returncode, 0)
        self.template.write_text('changed')
        self.assertNotEqual(self.run_bridge().returncode, 0)
        self.template.write_text('{"chat_template.jinja": 1}')
        (self.profile / 'directions.f32').write_bytes(b'bad')
        self.assertNotEqual(self.run_bridge().returncode, 0)

    def test_force_and_off_mode(self):
        for force in ('0', '-1', '4.1', 'nan', 'inf', 'word', '1e-50'):
            self.assertNotEqual(self.run_bridge(force=force).returncode, 0, force)
        for force in ('1e-45', '4'):
            r = self.run_bridge(force=force)
            self.assertEqual(r.returncode, 0, r.stderr)
        r = self.run_bridge(profile='-', template='/missing')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith('G6S2 3 '))
        self.assertNotEqual(self.run_bridge(profile='-', force='1').returncode, 0)
