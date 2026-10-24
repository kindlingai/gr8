#!/usr/bin/env python3
"""Model sets: an exlr8 release as the model, its id and its tokenizer. A small fake release, no GPU.
    python3 -m unittest discover -s tools/launch"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelset as M  # noqa: E402


def make_release(d, set_id='ab' * 32, variants=('home',), tokenizer=True):
    """A local exlr8 release with what the launcher reads: dense/exlr8-vocab.safetensors, a width manifest per variant
    carrying the weight set id, empty experts/ and index-slices/, and the tokenizer files at the root."""
    for sub in ('dense', 'manifests', 'experts', 'index-slices'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    open(os.path.join(d, 'dense', 'exlr8-vocab.safetensors'), 'w').close()
    for v in variants:
        with open(os.path.join(d, 'manifests', f'exlr8-manifest-{v}.json'), 'w') as f:
            json.dump({'format': 'exlr8-v2-manifest', 'name': v, 'set_id': set_id, 'layers': {}}, f)
    if tokenizer:
        for n in M.TOKENIZER:
            with open(os.path.join(d, n), 'w') as f:
                f.write(f'{{"{n}": 1}}')
    return d


class ReleaseTests(unittest.TestCase):
    """An exlr8 release as the model set: the id from the release's weights and the tokenizer."""

    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.rel = make_release(os.path.join(self.t, 'rel'))

    def tearDown(self):
        shutil.rmtree(self.t)

    def test_release_set(self):
        self.assertTrue(M.is_release(self.rel))
        self.assertFalse(M.is_release(self.t))
        ms = M.release_set(self.rel, 4)
        self.assertEqual(ms.set['source'], {'kind': 'exlr8-release', 'weights': 'ab' * 32})
        self.assertEqual((ms.tokenizer, ms.dir), (self.rel, self.rel))
        self.assertEqual(M.check_tokenizer(ms), [])
        # the id follows the weights, the tokenizer and the TP. A copy elsewhere has the same id
        other = shutil.copytree(self.rel, os.path.join(self.t, 'copy'))
        self.assertEqual(M.release_set(other, 4).id, ms.id)
        self.assertNotEqual(M.release_set(self.rel, 3).id, ms.id)
        with open(os.path.join(other, 'manifests', 'exlr8-manifest-home.json'), 'w') as f:
            json.dump({'set_id': 'cd' * 32}, f)
        self.assertNotEqual(M.release_set(other, 4).id, ms.id)
        with open(os.path.join(self.rel, 'chat_template.jinja'), 'a') as f:
            f.write('x')
        self.assertNotEqual(M.release_set(self.rel, 4).id, ms.id)
        self.assertIn('chat_template.jinja', M.check_tokenizer(ms)[0])

    def test_refusals(self):
        os.unlink(os.path.join(self.rel, 'manifests', 'exlr8-manifest-home.json'))
        with self.assertRaises(M.ModelError) as e:
            M.release_set(self.rel, 4)
        self.assertIn('no width manifest with a set_id', str(e.exception))
        bare = make_release(os.path.join(self.t, 'bare'), tokenizer=False)
        with self.assertRaises(M.ModelError) as e:
            M.release_set(bare, 4)
        self.assertIn('no tokenizer files', str(e.exception))
        self.assertEqual(M.release_set(bare, 4, model_dir=self.rel).tokenizer, self.rel)


if __name__ == '__main__':
    unittest.main()
