"""The retired Python frontend's steering cases, retained against the S2 shim and shared host contract."""
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g6s2shim import _steering_header, parse_ready, build_request  # noqa: E402


class SteeringTests(unittest.TestCase):
    cap = {'profile': 'example', 'digest': '0' * 64, 'scope': 'full', 'layers': 48,
           'forced': False, 'forced_scale': 0.0}

    def test_off_and_canonical_zero(self):
        self.assertEqual(_steering_header({}, self.cap), ('', None))
        self.assertEqual(_steering_header({'steering_scale': -0.0}, self.cap), ('', None))

    def test_profile_default_and_fp32_scale(self):
        self.assertEqual(_steering_header({'steering_profile': 'example'}, self.cap),
                         (' steer example 3f800000', None))
        self.assertEqual(_steering_header({'steering_profile': 'example', 'steering_scale': 1.1}, self.cap),
                         (' steer example 3f8ccccd', None))

    def test_null_fields_and_named_zero(self):
        self.assertEqual(_steering_header({'steering_profile': None, 'steering_scale': None}, self.cap), ('', None))
        self.assertEqual(_steering_header({'steering_profile': 'example', 'steering_scale': None}, self.cap),
                         (' steer example 3f800000', None))
        self.assertEqual(_steering_header({'steering_profile': 'example', 'steering_scale': -0.0}, self.cap),
                         (' steer example 00000000', None))
        self.assertIn('not enabled', _steering_header({'steering_profile': 'example', 'steering_scale': 0}, None)[1])
        for value in (True, '1', [], {}, 10**400):
            self.assertIsNotNone(_steering_header({'steering_profile': 'example', 'steering_scale': value}, self.cap)[1])

    def test_rejections(self):
        cases = [
            ({'steering_profile': 'other'}, 'unknown steering_profile'),
            ({'steering_profile': 7}, 'steering_profile must be a profile name'),
            ({'steering_profile': 'bad/name'}, 'steering_profile must be a profile name'),
            ({'steering_scale': 1}, 'steering_scale requires steering_profile'),
            ({'steering_profile': 'example', 'steering_scale': math.inf}, 'finite and in [0, 4]'),
            ({'steering_profile': 'example', 'steering_scale': 4.1}, 'finite and in [0, 4]'),
        ]
        for body, message in cases:
            with self.subTest(body=body):
                suffix, error = _steering_header(body, self.cap)
                self.assertIsNone(suffix)
                self.assertIn(message, error)
        self.assertIn('steering is not enabled', _steering_header({'steering_profile': 'example'}, None)[1])

    def test_forced_conflict(self):
        cap = dict(self.cap, forced=True, forced_scale=2.0)
        self.assertEqual(_steering_header({}, cap), ('', None))
        self.assertIn('conflicts with forced mode', _steering_header({'steering_profile': 'example'}, cap)[1])
        self.assertEqual(_steering_header({'steering_profile': 'example', 'steering_scale': 2}, cap),
                         (' steer example 40000000', None))

    @unittest.skipUnless(os.environ.get('STEERING_TRANSPORT_CHECK'), 'set STEERING_TRANSPORT_CHECK for C++/grant parity')
    def test_shim_to_cpp_grant_preserves_shared_scale_bits(self):
        cases = json.loads((Path(__file__).resolve().parents[2] /
                            'engine/runtime/tests/fixtures/steering_contract.json').read_text())
        for case in cases['scales']:
            if not case['valid']:
                continue
            f = {'prompt': [1, 2], 'max_tokens': 20, 'steering_profile': 'example',
                 'steering_scale': float(case['input'])}
            body, error = build_request(f, 7, self.cap)
            self.assertIsNone(error)
            with self.subTest(scale=case['input']):
                r = subprocess.run([os.environ['STEERING_TRANSPORT_CHECK'], '--request'], input=body,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self.assertEqual(r.stdout.decode().strip(), case['bits'])
                self.assertNotIn(b'AddressSanitizer', r.stderr)
                self.assertNotIn(b'runtime error:', r.stderr)

    def test_shared_contract(self):
        cases = json.loads((Path(__file__).resolve().parents[2] /
                            'engine/runtime/tests/fixtures/steering_contract.json').read_text())
        for case in cases['profile_names']:
            cap = dict(self.cap, profile=case['input'])
            body = {'steering_profile': case['input']}
            with self.subTest(profile=case['input']):
                self.assertEqual(_steering_header(body, cap)[1] is None, case['valid'])
                receipt = 'G6S2 4 1 4 1024 2 1 8 20 ' + case['input'] + ' ' + 'a' * 64 + ' full 2 0 0'
                if case['valid']:
                    self.assertEqual(parse_ready(receipt)['steering']['profile'], case['input'])
                else:
                    with self.assertRaises(ValueError):
                        parse_ready(receipt)
        for case in cases['scales']:
            body = {'steering_profile': 'example', 'steering_scale': float(case['input'])}
            with self.subTest(scale=case['input']):
                suffix, error = _steering_header(body, self.cap)
                self.assertEqual(error is None, case['valid'])
                if case['valid']:
                    self.assertEqual(suffix, ' steer example ' + case['bits'])


if __name__ == '__main__':
    unittest.main()
