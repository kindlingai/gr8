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
            b'## [v1.2.0]',
            b'## [v1.1.0] - 2026-11-20\n\n## [v1.2.0]',
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

    def test_accepts_release_please_entries_without_spec_files(self):
        raw = working_release_files()

        rp_minor_and_patch = dict(raw)
        rp_block = (
            b'## [1.3.0](https://github.com/kindlingai/gr8/compare/v1.2.1...v1.3.0) (2026-11-25)\n\n'
            b'### Features\n\n'
            b'* **qw38:** add feature (#610)\n\n'
            b'### [1.2.1](https://github.com/kindlingai/gr8/compare/v1.2.0...v1.2.1) (2026-11-24)\n\n'
            b'### Bug Fixes\n\n'
            b'* **ci:** fix build (#605)\n\n'
            b'## [v1.2.0]'
        )
        rp_minor_and_patch['CHANGELOG.md'] = raw['CHANGELOG.md'].replace(b'## [v1.2.0]', rp_block, 1)
        self.assertEqual(make_zip.check_release_records(rp_minor_and_patch), [])

        dup_rp = dict(raw)
        dup_block = (
            b'## [1.2.0](https://github.com/kindlingai/gr8/compare/v1.1.1...v1.2.0) (2026-11-23)\n\n'
            b'### Features\n\n'
            b'* duplicate (#602)\n\n'
            b'## [v1.2.0]'
        )
        dup_rp['CHANGELOG.md'] = raw['CHANGELOG.md'].replace(b'## [v1.2.0]', dup_block, 1)
        errs = make_zip.check_release_records(dup_rp)
        self.assertTrue(any('duplicate section for v1.2.0' in e for e in errs), errs)

        optional_spec_bad_date = dict(rp_minor_and_patch)
        v130 = copy.deepcopy(json.loads(raw['releases/v1.2.0.json'].decode('utf-8')))
        v130['version'] = '1.3.0'
        v130['tag'] = 'v1.3.0'
        v130['previous_tag'] = 'v1.2.0'
        v130['date'] = '2026-11-26'
        v130['release_url'] = 'https://github.com/kindlingai/gr8/releases/tag/v1.3.0'
        v130['container']['image'] = 'ghcr.io/kindlingai/gr8:v1.3.0'
        v130['container']['pinned'] = f"ghcr.io/kindlingai/gr8:v1.3.0@{v130['container']['digest']}"
        optional_spec_bad_date['releases/v1.3.0.json'] = json.dumps(v130).encode('utf-8')
        errs_date = make_zip.check_release_records(optional_spec_bad_date)
        self.assertTrue(any('v1.3.0: CHANGELOG.md date 2026-11-25 != releases/v1.3.0.json date 2026-11-26' in e for e in errs_date), errs_date)

        v130['date'] = '2026-11-25'
        optional_spec_bad_date['releases/v1.3.0.json'] = json.dumps(v130).encode('utf-8')
        self.assertEqual(make_zip.check_release_records(optional_spec_bad_date), [])


if __name__ == '__main__':
    unittest.main()
