import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from steering import release


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        payload = struct.pack('<6f', 1, 0, 0, 0, 1, 0)
        manifest = {'schema_version': 1, 'profile_name': 'test-v1',
                    'model': {'name': 'glm-5.3', 'revision': 'rev', 'weight_set': 'a' * 64,
                              'tokenizer_revision': 'tok', 'chat_template_sha256': 'b' * 64,
                              'hidden_size': 3, 'layer_count': 5},
                    'scope': 'full', 'layers': [1, 2], 'alphas': [1.0, 1.0],
                    'directions': {'file': 'directions.f32', 'dtype': 'float32-le', 'byte_count': len(payload),
                                   'sha256': hashlib.sha256(payload).hexdigest()},
                    'provenance': {'extraction_tool': 'fixture', 'extraction_commit': 'fixture',
                                   'dataset_manifests': {'train': '1' * 64, 'validation': '2' * 64,
                                                         'final': '3' * 64}}}
        with open(os.path.join(root, 'manifest.json'), 'w') as f: json.dump(manifest, f)
        with open(os.path.join(root, 'directions.f32'), 'wb') as f: f.write(payload)
        return manifest, payload

    def test_deterministic_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, payload = self.fixture(d)
            a, b, out = (os.path.join(d, x) for x in ('a.safetensors', 'b.safetensors', 'out'))
            release.pack(d, a); release.pack(d, b)
            with open(a, 'rb') as fa, open(b, 'rb') as fb:
                self.assertEqual(fa.read(), fb.read())
            got = release.unpack(a, out)
            self.assertEqual(got, manifest)
            self.assertEqual(release.validate_directory(out, manifest['model']), manifest)
            with open(os.path.join(out, 'directions.f32'), 'rb') as f:
                self.assertEqual(f.read(), payload)

    def test_rejects_payload_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d); p = os.path.join(d, 'p.safetensors'); release.pack(d, p)
            with open(p, 'rb') as f: raw = bytearray(f.read())
            raw[-1] ^= 1
            bad = os.path.join(d, 'bad.safetensors')
            with open(bad, 'wb') as f: f.write(raw)
            with self.assertRaises(release.ProfileError): release.read(bad)

    def test_directory_validation_rejects_identity_and_non_unit_rows(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, _ = self.fixture(d)
            with self.assertRaisesRegex(release.ProfileError, 'model identity mismatch'):
                release.validate_directory(d, dict(manifest['model'], revision='other'))
            with open(os.path.join(d, 'directions.f32'), 'wb') as f:
                f.write(struct.pack('<6f', 1, 1, 0, 0, 1, 0))
            manifest['directions']['sha256'] = hashlib.sha256(struct.pack('<6f', 1, 1, 0, 0, 1, 0)).hexdigest()
            with open(os.path.join(d, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(release.ProfileError, 'direction norm'):
                release.validate_directory(d)



    @staticmethod
    def artifact(header, payload):
        raw = release.canonical(header).encode()
        raw += b' ' * (-len(raw) % 8)
        return struct.pack('<Q', len(raw)) + raw + payload

    def test_exact_manifest_identity_and_repack(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, payload = self.fixture(d)
            manifest['provenance']['extraction_tool'] = 'unicode-\u00e9-\U0001f525'
            raw = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8') + b'\n\n'
            Path(d, 'manifest.json').write_bytes(raw)
            artifact, out = Path(d, 'profile.safetensors'), Path(d, 'out')
            release.pack(d, artifact)
            release.unpack(artifact, out)
            self.assertEqual(Path(out, 'manifest.json').read_bytes(), raw)
            self.assertEqual(Path(out, 'directions.f32').read_bytes(), payload)
            self.assertEqual(release.unpack(artifact, out), manifest)  # idempotent restart
            packed = artifact.read_bytes()
            n, = struct.unpack('<Q', packed[:8])
            meta = json.loads(packed[8:8+n])['__metadata__']
            self.assertEqual(meta['gr8_manifest'].encode('utf-8'), raw)
            self.assertEqual(meta['manifest_sha256'], hashlib.sha256(raw).hexdigest())
            release.pack(out, Path(d, 'repacked.safetensors'))
            self.assertEqual(Path(d, 'repacked.safetensors').read_bytes(), packed)

    def test_shared_contract_cases(self):
        table = Path(__file__).resolve().parents[3] / 'engine/runtime/tests/fixtures/steering_contract.json'
        cases = json.loads(table.read_text())
        for text in cases['json_valid']:
            with self.subTest(valid_json=text):
                release.parse_json(text, 'fixture')
        for text in cases['json_invalid']:
            with self.subTest(invalid_json=text), self.assertRaises(release.ProfileError):
                release.parse_json(text, 'fixture')
        with tempfile.TemporaryDirectory() as d:
            manifest, payload = self.fixture(d)
            for case in cases['profile_names']:
                manifest['profile_name'] = case['input']
                with self.subTest(profile_name=case['input']):
                    if case['valid']:
                        release.validate_manifest(manifest, payload)
                    else:
                        with self.assertRaises(release.ProfileError):
                            release.validate_manifest(manifest, payload)
            manifest['profile_name'] = 'test-v1'
            # The manifest alpha bound is the same pre-FP32 bound as request scales.
            for case in cases['scales']:
                manifest['alphas'][0] = float(case['input'])
                with self.subTest(alpha=case['input']):
                    if case['valid']:
                        release.validate_manifest(manifest, payload)
                    else:
                        with self.assertRaises(release.ProfileError):
                            release.validate_manifest(manifest, payload)

    def test_strict_json_encoding_and_depth(self):
        for raw in (b'"\xc0\x80"', b'"\xed\xa0\x80"', b'"\xf4\x90\x80\x80"', b'"\xc2"',
                    b'\xef\xbb\xbf{}', '{}'.encode('utf-16'), '"\\ud800"', '{"\\udc00":0}',
                    '[' * 66 + '0' + ']' * 66, '{"a":0,"a":1}'):
            with self.subTest(raw=raw), self.assertRaises(release.ProfileError):
                release.parse_json(raw, 'fixture')
        self.assertEqual(release.parse_json('"\\ud83d\\udd25"', 'fixture'), '\U0001f525')

    def test_rejects_malformed_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            _, payload = self.fixture(d)
            good = Path(d, 'good.safetensors')
            release.pack(d, good)
            raw = good.read_bytes()
            n, = struct.unpack('<Q', raw[:8])
            header = json.loads(raw[8:8+n])
            variants = [None, [], dict(header, extra=header['directions'])]
            def changed(section, field, value):
                h = copy.deepcopy(header)
                h[section][field] = value
                return h
            variants += [changed('directions', 'dtype', 'F16'),
                         changed('directions', 'shape', [2.0, 3]),
                         changed('directions', 'shape', [True, 3]),
                         changed('directions', 'shape', [3, 2]),
                         changed('directions', 'data_offsets', [0.0, len(payload)]),
                         changed('directions', 'data_offsets', [False, len(payload)]),
                         changed('directions', 'data_offsets', [0, -1]),
                         changed('directions', 'data_offsets', [0, len(payload) + 4]),
                         changed('__metadata__', 'format', 'other'),
                         changed('__metadata__', 'manifest_sha256', '0' * 64),
                         changed('__metadata__', 'gr8_manifest', '{')]
            bad = Path(d, 'bad.safetensors')
            for h in variants:
                with self.subTest(header=h), self.assertRaises(release.ProfileError):
                    bad.write_bytes(self.artifact(h, payload))
                    release.read(bad)
            malformed = [b'', b'1234567', struct.pack('<Q', 0), struct.pack('<Q', release.MAX_HEADER + 8),
                         struct.pack('<Q', 3) + b'{} ', raw[:8+n-1], raw[:-1], raw + b'tail']
            for data in malformed:
                with self.subTest(data=data[:8]), self.assertRaises(release.ProfileError):
                    bad.write_bytes(data)
                    release.read(bad)
            # The metadata digest is over exact bytes; formatting is allowed, changing it without rehashing is not.
            h = changed('__metadata__', 'gr8_manifest', header['__metadata__']['gr8_manifest'] + '\n')
            bad.write_bytes(self.artifact(h, payload))
            with self.assertRaisesRegex(release.ProfileError, 'digest'):
                release.read(bad)

    def test_limits_and_failed_pack_preserve_output(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d)
            output = Path(d, 'profile.safetensors')
            output.write_bytes(b'previous')
            with mock.patch.object(release, 'MAX_PAYLOAD', 4), self.assertRaises(release.ProfileError):
                release.pack(d, output)
            self.assertEqual(output.read_bytes(), b'previous')
            with mock.patch.object(release, 'MAX_HEADER', 8), self.assertRaises(release.ProfileError):
                release.pack(d, output)
            self.assertEqual(output.read_bytes(), b'previous')
            with mock.patch.object(release.os, 'replace', side_effect=OSError('publication failed')):
                with self.assertRaises(OSError):
                    release.pack(d, output)
            self.assertEqual(output.read_bytes(), b'previous')
            self.assertEqual(list(Path(d).glob('.steering-*')), [])

    def test_concurrent_pack_and_unpack_publication(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, payload = self.fixture(d)
            artifact = Path(d, 'p.safetensors')
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: release.pack(d, artifact), range(8)))
            self.assertEqual(release.read(artifact), (manifest, payload))
            self.assertEqual(list(Path(d).glob('.steering-*')), [])
            out = Path(d, 'out')
            out.mkdir()
            real_replace = release.os.replace
            def fail_publication(src, dest):
                if Path(dest) == out:
                    raise OSError('publication failed')
                return real_replace(src, dest)
            with mock.patch.object(release.os, 'replace', side_effect=fail_publication):
                with self.assertRaises(OSError):
                    release.unpack(artifact, out)
            self.assertEqual(list(out.iterdir()), [])
            self.assertEqual(list(Path(d).glob('.steering-*')), [])
            release.unpack(artifact, out)
            Path(out, 'manifest.json').write_bytes(b'previous')
            with self.assertRaisesRegex(release.ProfileError, 'different profile'):
                release.unpack(artifact, out)
            self.assertEqual(Path(out, 'manifest.json').read_bytes(), b'previous')
            self.assertEqual(Path(out, 'directions.f32').read_bytes(), payload)
            link = Path(d, 'link')
            link.symlink_to(out, target_is_directory=True)
            with self.assertRaisesRegex(release.ProfileError, 'symlink'):
                release.unpack(artifact, link)

    def test_cli_roundtrip_and_clean_errors(self):
        tool = Path(release.__file__)
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d)
            artifact, out = Path(d, 'p.safetensors'), Path(d, 'out')
            for command in [('pack', d, artifact), ('unpack', artifact, out)]:
                result = subprocess.run([sys.executable, tool, *map(str, command)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, tool, 'pack', str(Path(d, 'missing')), str(artifact)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Traceback', result.stderr)

    @unittest.skipUnless(os.environ.get('STEERING_PROFILE_CHECK'), 'set STEERING_PROFILE_CHECK for C++ parity')
    def test_cpp_loader_parity(self):
        checker = os.environ['STEERING_PROFILE_CHECK']
        with tempfile.TemporaryDirectory() as d:
            manifest, payload = self.fixture(d)
            expected = copy.deepcopy(manifest['model'])
            expected_path = Path(d, 'expected.json')
            def compare(raw, data=payload, model=expected, valid=False):
                Path(d, 'manifest.json').write_bytes(raw)
                Path(d, 'directions.f32').write_bytes(data)
                expected_path.write_text(json.dumps(model))
                accepted = True
                try:
                    release.validate_directory(d, model)
                except release.ProfileError:
                    accepted = False
                result = subprocess.run([checker, d, str(expected_path)], capture_output=True, text=True)
                for diagnostic in ('AddressSanitizer', 'UndefinedBehaviorSanitizer', 'runtime error:'):
                    self.assertNotIn(diagnostic, result.stderr)
                self.assertEqual(accepted, valid, raw)
                self.assertEqual(result.returncode, 0 if valid else 1, result.stderr)
                if valid:
                    self.assertEqual(result.stdout.strip(), hashlib.sha256(raw).hexdigest())
            raw = json.dumps(manifest).encode()
            compare(raw, valid=True)
            table = Path(__file__).resolve().parents[3] / 'engine/runtime/tests/fixtures/steering_contract.json'
            cases = json.loads(table.read_text())
            for case in cases['profile_names']:
                m = copy.deepcopy(manifest); m['profile_name'] = case['input']
                with self.subTest(name=case['input']):
                    compare(json.dumps(m).encode(), valid=case['valid'])
            for case in cases['scales']:
                m = copy.deepcopy(manifest); m['alphas'][0] = float(case['input'])
                with self.subTest(alpha=case['input']):
                    compare(json.dumps(m).encode(), valid=case['valid'])
            for field in expected:
                model = dict(expected)
                model[field] = model[field] + 1 if isinstance(model[field], int) else model[field] + '-other'
                with self.subTest(identity=field):
                    compare(raw, model=model)
            edits = [('schema_version', 1.0), ('schema_version', True), ('scope', 'other'),
                     ('layers', [1, 1]), ('layers', [2, 1]), ('layers', [1, 5]), ('layers', [True, 2]),
                     ('layers', [1.0, 2]), ('layers', []), ('alphas', [1]), ('alphas', [True, 1]),
                     ('provenance', {}), ('extra', 1)]
            for key, value in edits:
                m = copy.deepcopy(manifest); m[key] = value
                with self.subTest(field=key, value=value):
                    compare(json.dumps(m).encode())
            nested = [('model', 'hidden_size', 3.0), ('model', 'hidden_size', 0),
                      ('model', 'layer_count', 2**32), ('model', 'layer_count', 2**53 + 1),
                      ('model', 'name', ''), ('model', 'weight_set', 'weights'),
                      ('model', 'chat_template_sha256', int('1' * 64)),
                      ('directions', 'byte_count', float(len(payload))),
                      ('directions', 'file', '../directions.f32'), ('directions', 'dtype', 'float16-le'),
                      ('directions', 'sha256', '0' * 64), ('provenance', 'family', 'other'),
                      ('provenance', 'direction_layer', True), ('provenance', 'fit_sha256', int('1' * 64))]
            for section, field, value in nested:
                m = copy.deepcopy(manifest); m[section][field] = value
                with self.subTest(section=section, field=field, value=value):
                    compare(json.dumps(m).encode())
            for text in ('[]', '{}', raw.decode().replace('"schema_version": 1', '"schema_version": 01'),
                         raw.decode().replace('"schema_version": 1', '"schema_version": 1e0'),
                         raw.decode().replace('"schema_version": 1', '"schema_version": 1,"schema_version":1'),
                         raw.decode().replace('"fixture"', '"\\ud800"')):
                compare(text.encode())
            for values in ([2, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [float('inf'), 0, 0, 0, 1, 0],
                           [float('nan'), 0, 0, 0, 1, 0]):
                data = struct.pack('<6f', *values)
                m = copy.deepcopy(manifest); m['directions']['sha256'] = hashlib.sha256(data).hexdigest()
                compare(json.dumps(m).encode(), data=data)
            compare(raw, data=payload[:-1])
            compare(raw, data=payload + b'tail')
            for scope in ('shared', 'full'):
                m = copy.deepcopy(manifest); m['scope'] = scope
                m['provenance'].update(family=None, direction_layer=None)
                compare(json.dumps(m).encode(), valid=True)
            m = copy.deepcopy(manifest)
            m['provenance']['extraction_tool'] = '\u00e9-\U0001f525'
            compare(json.dumps(m, ensure_ascii=False).encode('utf-8') + b'\n\n', valid=True)
            # A real pack/materialize cycle preserves the C++ loader's manifest digest.
            Path(d, 'manifest.json').write_bytes(raw); Path(d, 'directions.f32').write_bytes(payload)
            artifact, out = Path(d, 'p.safetensors'), Path(d, 'out')
            release.pack(d, artifact); release.unpack(artifact, out)
            result = subprocess.run([checker, str(out), str(expected_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), hashlib.sha256(raw).hexdigest())


if __name__ == '__main__': unittest.main()
