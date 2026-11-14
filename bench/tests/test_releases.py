import copy
import importlib.util
import json
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('make_zip', REPO / 'scripts' / 'make_zip.py')
make_zip = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_zip)


def working_release_files():
    raw = {}
    cl = REPO / 'CHANGELOG.md'
    if cl.is_file():
        raw['CHANGELOG.md'] = cl.read_bytes()
    rel_dir = REPO / 'releases'
    if rel_dir.is_dir():
        for p in sorted(rel_dir.rglob('*')):
            if p.is_file():
                raw[p.relative_to(REPO).as_posix()] = p.read_bytes()
    return raw


class ReleaseRecordsTest(unittest.TestCase):
    def test_historical_tree_without_release_records_passes(self):
        self.assertEqual(make_zip.check_release_records({'README.md': b'# gr8\n'}), [])

    def test_live_changelog_and_specs_are_valid_and_synced(self):
        raw = working_release_files()
        self.assertIn('CHANGELOG.md', raw)
        self.assertIn('releases/README.md', raw)
        self.assertIn('releases/v1.0.json', raw)
        self.assertIn('releases/v1.0.1.json', raw)
        self.assertEqual(make_zip.check_release_records(raw), [])

        v10 = json.loads(raw['releases/v1.0.json'].decode('utf-8'))
        v101 = json.loads(raw['releases/v1.0.1.json'].decode('utf-8'))
        self.assertIsNone(v10['previous_tag'])
        self.assertEqual(v101['previous_tag'], 'v1.0')
        self.assertEqual(v10['commit'], 'db6f2c17c7b05303b69ce667e416d10340aeb9b7')
        self.assertEqual(v101['commit'], '6333c78a4ab4cfedc706793d689f54858c576d84')

    def test_releasing_doc_requires_changelog_and_spec_updates(self):
        releasing = (REPO / 'RELEASING.md').read_text(encoding='utf-8')
        self.assertIn('CHANGELOG.md', releasing)
        self.assertIn('releases/<tag>.json', releasing)

    def test_detects_missing_spec_or_mismatched_metadata(self):
        raw = working_release_files()

        missing_spec = dict(raw)
        del missing_spec['releases/v1.0.1.json']
        errs = make_zip.check_release_records(missing_spec)
        self.assertTrue(any('missing releases/<tag>.json: v1.0.1' in e for e in errs), errs)

        bad_spec = copy.deepcopy(json.loads(raw['releases/v1.0.1.json'].decode('utf-8')))
        bad_spec['breaking'] = True
        bad_spec['commit'] = '0' * 40
        tampered = dict(raw)
        tampered['releases/v1.0.1.json'] = json.dumps(bad_spec).encode('utf-8')
        errs2 = make_zip.check_release_records(tampered)
        self.assertTrue(any('breaking=True disagrees' in e for e in errs2), errs2)
        self.assertTrue(any('does not mention commit' in e for e in errs2), errs2)

        missing_heading = dict(raw)
        missing_heading['CHANGELOG.md'] = raw['CHANGELOG.md'].replace(b'### Toolchain / Dockerfile sync', b'### Other')
        errs3 = make_zip.check_release_records(missing_heading)
        self.assertTrue(any('missing "### Toolchain / Dockerfile sync"' in e for e in errs3), errs3)

        malformed_h2 = dict(raw)
        malformed_h2['CHANGELOG.md'] = raw['CHANGELOG.md'].replace(
            b'## [Unreleased]',
            b'## [Unreleased]\n\n## [v1.1.0] - 2026-11-20\n',
        )
        errs4 = make_zip.check_release_records(malformed_h2)
        self.assertTrue(any('malformed H2 heading' in e for e in errs4), errs4)

        type_tampered = dict(raw)
        bad_types = copy.deepcopy(json.loads(raw['releases/v1.0.json'].decode('utf-8')))
        bad_types['schema_version'] = True
        bad_types['version'] = 1.0
        bad_types['previous_tag'] = ['v0.9']
        type_tampered['releases/v1.0.json'] = json.dumps(bad_types).encode('utf-8')
        errs5 = make_zip.check_release_records(type_tampered)
        self.assertTrue(any('schema_version must be integer 1' in e for e in errs5), errs5)
        self.assertTrue(any('version must be a non-empty string' in e for e in errs5), errs5)
        self.assertTrue(any('previous_tag must be null or a valid v* tag string' in e for e in errs5), errs5)

        cycle = dict(raw)
        self_loop = copy.deepcopy(json.loads(raw['releases/v1.0.1.json'].decode('utf-8')))
        self_loop['previous_tag'] = 'v1.0.1'
        cycle['releases/v1.0.1.json'] = json.dumps(self_loop).encode('utf-8')
        errs6 = make_zip.check_release_records(cycle)
        self.assertTrue(any('cycle detected in previous_tag chain' in e for e in errs6), errs6)


if __name__ == '__main__':
    unittest.main()
