#!/usr/bin/env python3
"""Model sets: set.json and its id, the cache layout, COMPLETE and validation. Small fake packs, no GPU.
    python3 -m unittest discover -s tools/launch"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelset as M  # noqa: E402


def make_packs(d, tp=4, dflash=True, qo=False, tokenizer=True, size=4096):
    """A pack directory shaped like build_packs.sh output: rank<r>.{pack,manifest.json}, the drafter packs, and the
    tokenizer files in tokenizer/."""
    os.makedirs(d, exist_ok=True)
    stems = [f'rank{r}' for r in range(tp)] + ([f'dflash.r{r}' for r in range(tp)] if dflash else []) + (['qo'] if qo else [])
    for i, stem in enumerate(stems):
        n = size * (i + 1)
        with open(os.path.join(d, stem + '.pack'), 'wb') as f:
            f.write(bytes([i % 251]) * n)
        with open(os.path.join(d, stem + '.manifest.json'), 'w') as f:
            json.dump({'format': 'g6pack-v1', 'rank': i, 'align': 4096, 'pack_bytes': n,
                       'tensors': [{'name': f'{stem}.w', 'sha256': f'{i:064x}'}]}, f)
    if tokenizer:
        os.makedirs(os.path.join(d, 'tokenizer'), exist_ok=True)
        for n in M.TOKENIZER:
            with open(os.path.join(d, 'tokenizer', n), 'w') as f:
                f.write(f'{{"{n}": 1}}')
    return d


def plan(dflash='1', own='4'):
    return {'knobs': {'DFLASH': dflash, 'PF_OWN': own}, 'env': [], 'args': []}


class SetTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.src = make_packs(os.path.join(self.t, 'packs'))

    def test_id_is_path_free_and_content_bound(self):
        a, _ = M.describe(self.src, 4)
        other = shutil.copytree(self.src, os.path.join(self.t, 'elsewhere'))
        self.assertEqual(M.describe(other, 4)[0]['set_id'], a['set_id'])
        self.assertEqual(a['set_id'], M.compute_id(a))
        self.assertEqual(len(a['ranks']), 4)
        with open(os.path.join(other, 'rank2.manifest.json'), 'a') as f:
            f.write(' ')
        self.assertNotEqual(M.describe(other, 4)[0]['set_id'], a['set_id'])
        with open(os.path.join(self.src, 'tokenizer', 'chat_template.jinja'), 'a') as f:
            f.write('x')
        self.assertNotEqual(M.describe(self.src, 4)[0]['set_id'], a['set_id'])

    def test_refusals(self):
        for model, tp, want in ((self.src, 3, 'TP above 3'), (self.src, 6, 'lacks rank4.manifest.json'),
                                ('/no/such/org/repo', 4, 'no such directory'), (None, 4, 'no model')):
            with self.assertRaises(M.ModelError) as e:
                M.resolve(model, None, tp)
            self.assertIn(want, str(e.exception))
        raw = os.path.join(self.t, 'raw')
        os.makedirs(raw)
        open(os.path.join(raw, 'model-00001.safetensors'), 'w').close()
        with self.assertRaises(M.ModelError) as e:
            M.resolve(raw, None, 4)
        self.assertIn('raw checkpoint', str(e.exception))
        shutil.rmtree(os.path.join(self.src, 'tokenizer'))
        with self.assertRaises(M.ModelError) as e:
            M.resolve(self.src, None, 4)
        self.assertIn('no tokenizer files', str(e.exception))

    def test_set_json_on_disk_is_checked(self):
        s, _ = M.describe(self.src, 4)
        with open(os.path.join(self.src, 'set.json'), 'w') as f:
            json.dump({**s, 'tp': 4, 'source': {'kind': 'edited'}}, f)
        with self.assertRaises(M.ModelError) as e:
            M.resolve(self.src, None, 4)
        self.assertIn('does not match its contents', str(e.exception))

    def test_in_place(self):
        ms = M.resolve(self.src, None, 4)
        self.assertEqual(ms.rank_dir(2), self.src)
        self.assertEqual(M.validate_rank(ms, 2, plan()), [])
        os.truncate(os.path.join(self.src, 'rank2.pack'), 10)
        self.assertIn('partly copied', M.validate_rank(ms, 2, plan())[0])
        os.unlink(os.path.join(self.src, 'dflash.r1.pack'))
        self.assertEqual(M.validate_rank(ms, 1, plan(dflash='0')), [])
        self.assertIn('missing', M.validate_rank(ms, 1, plan())[0])
        with self.assertRaises(M.ModelError):
            M.needed(ms.set, 0, plan(own='3'))     # no qo pack in this set


class ReleaseTests(unittest.TestCase):
    """An exlr8 release as the model set: no rank packs, the id from the release's weights and the tokenizer."""

    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.rel = os.path.join(self.t, 'rel')
        os.makedirs(os.path.join(self.rel, 'dense'))
        os.makedirs(os.path.join(self.rel, 'manifests'))
        open(os.path.join(self.rel, 'dense', 'exlr8-vocab.safetensors'), 'w').close()
        with open(os.path.join(self.rel, 'manifests', 'exlr8-manifest-home.json'), 'w') as f:
            json.dump({'format': 'exlr8-v2-manifest', 'set_id': 'ab' * 32, 'layers': {}}, f)
        for n in M.TOKENIZER:
            with open(os.path.join(self.rel, n), 'w') as f:
                f.write(n)

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_release_set(self):
        self.assertTrue(M.is_release(self.rel))
        self.assertFalse(M.is_release(make_packs(os.path.join(self.t, 'packs'))))
        ms = M.release_set(self.rel, 4)
        self.assertEqual(ms.set['source'], {'kind': 'exlr8-release', 'weights': 'ab' * 32})
        self.assertEqual(ms.tokenizer, self.rel)
        # every rank's file list is empty, for PF_OWN=3 too
        self.assertEqual(M.needed(ms.set, 1, plan(dflash='0', own='3')), {})
        self.assertEqual(M.validate_rank(ms, 2, plan(dflash='0')), [])
        self.assertIsNone(M.offers(ms, plan(dflash='0')))
        self.assertEqual(M.check_tokenizer(ms), [])
        # the id follows the weights and the TP. A copy elsewhere has the same id
        other = shutil.copytree(self.rel, os.path.join(self.t, 'copy'))
        self.assertEqual(M.release_set(other, 4).id, ms.id)
        self.assertNotEqual(M.release_set(self.rel, 3).id, ms.id)
        with open(os.path.join(other, 'manifests', 'exlr8-manifest-home.json'), 'w') as f:
            json.dump({'set_id': 'cd' * 32}, f)
        self.assertNotEqual(M.release_set(other, 4).id, ms.id)

    def test_release_needs_a_set_id(self):
        os.unlink(os.path.join(self.rel, 'manifests', 'exlr8-manifest-home.json'))
        with self.assertRaises(M.ModelError):
            M.release_set(self.rel, 4)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.src = make_packs(os.path.join(self.t, 'packs'))
        self.cache = os.path.join(self.t, 'cache')
        self.logs = []

    def prepare(self, rank, p, model=True):
        ms = M.resolve(self.src if model else None, self.cache, 4)
        M.prepare(ms, rank, p, log=self.logs.append)
        return ms

    def test_copies_only_this_rank(self):
        ms = self.prepare(2, plan(dflash='0'))
        d = os.path.join(self.cache, 'sets', ms.id)
        self.assertEqual(sorted(os.listdir(d)), ['.lock', 'rank2', 'set.json', 'tokenizer'])
        self.assertEqual(sorted(os.listdir(os.path.join(d, 'rank2'))), ['COMPLETE', 'rank2.manifest.json', 'rank2.pack'])
        with open(os.path.join(d, 'rank2', 'COMPLETE')) as f:
            c = json.load(f)
        self.assertEqual((c['set_id'], c['rank']), (ms.id, 2))
        self.assertEqual(c['files']['rank2.pack']['bytes'], os.path.getsize(os.path.join(self.src, 'rank2.pack')))
        self.assertEqual(M.load_set(os.path.join(d, 'set.json'))['set_id'], ms.id)
        self.assertEqual(ms.tokenizer, os.path.join(d, 'tokenizer'))
        self.assertEqual(os.listdir(os.path.join(self.cache, 'tmp')), [])
        # a later boot wants the drafter: it is added, and the rank pack is hard-linked
        ino = os.stat(os.path.join(d, 'rank2', 'rank2.pack')).st_ino
        self.logs.clear()
        self.prepare(2, plan())
        self.assertIn('dflash.r2.pack', os.listdir(os.path.join(d, 'rank2')))
        self.assertEqual(os.stat(os.path.join(d, 'rank2', 'rank2.pack')).st_ino, ino)
        self.assertFalse(any('rank2.pack' in x for x in self.logs))

    def test_cache_alone(self):
        self.prepare(1, plan())
        ms = M.resolve(None, self.cache, 4)
        self.assertEqual(M.validate_rank(ms, 1, plan()), [])
        bad = M.validate_rank(ms, 3, plan())
        self.assertIn('no COMPLETE', bad[0])
        with self.assertRaises(M.ModelError) as e:
            M.prepare(ms, 3, plan())
        self.assertIn('Pass --model', str(e.exception))
        with self.assertRaises(M.ModelError) as e:
            M.resolve(None, self.cache, 6)
        self.assertIn('no model set for TP=6', str(e.exception))

    def test_two_sets_need_a_choice(self):
        self.prepare(0, plan())
        other = make_packs(os.path.join(self.t, 'other'), size=8192)
        M.prepare(M.resolve(other, self.cache, 4), 0, plan(), log=self.logs.append)
        with self.assertRaises(M.ModelError) as e:
            M.resolve(None, self.cache, 4)
        self.assertIn('holds 2 sets', str(e.exception))

    def test_complete_is_checked_every_boot(self):
        ms = self.prepare(0, plan())
        d = ms.rank_dir(0)
        os.truncate(os.path.join(d, 'rank0.pack'), 5)
        self.assertIn('not 4096 bytes', M.validate_rank(ms, 0, plan())[0])
        self.prepare(0, plan())                      # repaired from --model
        self.assertEqual(M.validate_rank(ms, 0, plan()), [])
        os.unlink(os.path.join(d, 'COMPLETE'))       # an interrupted copy: no COMPLETE
        self.assertIn('no COMPLETE', M.validate_rank(ms, 0, plan())[0])
        self.prepare(0, plan())
        self.assertEqual(M.validate_rank(ms, 0, plan()), [])
        with open(os.path.join(d, 'COMPLETE'), 'w') as f:
            json.dump({'set_id': 'f' * 64, 'rank': 0, 'files': {}}, f)
        self.assertIn('is for set ffffffffffff', M.validate_rank(ms, 0, plan())[0])

    def test_dead_copy_is_cleaned(self):
        os.makedirs(os.path.join(self.cache, 'tmp', 'rank0.999999999'))
        self.prepare(0, plan())
        self.assertEqual(os.listdir(os.path.join(self.cache, 'tmp')), [])


if __name__ == '__main__':
    unittest.main()
