#!/usr/bin/env python3
"""--variant: resolving home / down1 / up1 or a manifest path, the refusal without v2 widths, the canonical options,
the model id, and the expert GB per rank.
    python3 -m unittest discover -s tools/launch"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modelset as M  # noqa: E402
import options as opt  # noqa: E402
import supervisor as S  # noqa: E402
from test_modelset import make_packs  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'


def manifest(path, name, width, layers=(0,), shared=4):
    """A v2 width manifest with every routed unit at one width."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'format': 'exlr8-v2-manifest', 'name': name,
                   'layers': {str(l): {'gate': [[width] * 16] * 256, 'up': [[width] * 16] * 256, 'down': [width] * 256,
                                       'shared': shared} for l in layers}}, f)
    return path


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.v2 = os.path.join(self.t, 'v2')
        manifest(os.path.join(self.v2, 'exlr8-manifest-home.json'), 'home', 3)
        manifest(os.path.join(self.v2, 'manifests', 'exlr8-manifest-down1.json'), 'down1', 2)
        self.model = os.path.join(self.t, 'packs')
        manifest(os.path.join(self.model, 'exlr8-manifest-up1.json'), 'up1', 4)

    def test_names_and_paths(self):
        for name, where in (('home', self.v2), ('down1', os.path.join(self.v2, 'manifests')), ('up1', self.model)):
            v = M.resolve_variant(name, self.v2, [self.model])
            self.assertEqual((v.name, v.path), (name, os.path.join(where, f'exlr8-manifest-{name}.json')))
            self.assertEqual(len(v.sha256), 64)
        self.assertEqual(M.resolve_variant(None, self.v2, []).name, 'home')
        p = manifest(os.path.join(self.t, 'mine.json'), 'route-budget-80', 3)
        v = M.resolve_variant(p, self.v2, [])
        self.assertEqual((v.name, v.path), ('route-budget-80', p))

    def test_refusals(self):
        with self.assertRaises(M.ModelError) as e:
            M.resolve_variant('up1', self.v2, [])
        self.assertIn('no exlr8-manifest-up1.json in ' + self.v2, str(e.exception))
        with self.assertRaises(M.ModelError) as e:
            M.resolve_variant('k5', self.v2, [])
        self.assertIn('not home, down1 or up1', str(e.exception))
        for spec in ('down1', os.path.join(self.t, 'mine.json')):
            with self.assertRaises(M.ModelError) as e:
                M.resolve_variant(spec, None, [self.model])
            self.assertIn('needs the exlr8 v2 width files', str(e.exception))
        v = M.resolve_variant('home', None, [])
        self.assertEqual((v.name, v.path, v.sha256), ('home', None, None))

    def test_model_id(self):
        ms = M.resolve(make_packs(os.path.join(self.t, 'set')), None, 4)
        home, down1 = M.resolve_variant('home', self.v2, []), M.resolve_variant('down1', self.v2, [])
        self.assertEqual(M.model_id(ms, M.Variant('home')), ms.id)            # no v2: the set id
        ids = {M.model_id(ms, home), M.model_id(ms, down1), ms.id}
        self.assertEqual(len(ids), 3)
        self.assertEqual(M.model_id(ms, home), M.model_id(ms, M.resolve_variant('home', self.v2, [])))

    def test_expert_gb(self):
        p = manifest(os.path.join(self.t, 'k4.json'), 'k4', 4)
        # 257 experts (the shared one included) x 16 halves x 3 projections at K4, a quarter on each rank
        self.assertAlmostEqual(M.expert_gb(p, 4), 257 * 16 * 3 * M.HALF_BYTES[4] / 4 / 1e9)
        q = manifest(os.path.join(self.t, 'k2.json'), 'k2', 2, shared=0)
        self.assertAlmostEqual(M.expert_gb(q, 4), 256 * 16 * 3 * M.HALF_BYTES[2] / 4 / 1e9)
        self.assertGreater(M.expert_gb(p, 3), M.expert_gb(p, 6))


class OptionTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.v2 = os.path.join(self.t, 'v2')
        for name, w in (('home', 3), ('down1', 2)):
            manifest(os.path.join(self.v2, f'exlr8-manifest-{name}.json'), name, w)
        self.packs = make_packs(os.path.join(self.t, 'packs'))

    def opts(self, *argv):
        o = opt.parse(['--no-config', '--peers', PEERS, '--model', self.packs, *argv], {})
        opt.validate(o)
        return o

    def test_conflicts(self):
        for argv, want in ((['--variant', 'down1', '--v2-widths', '/w.json'], 'give one'), (['--v2', '/v'], '--v2-slices')):
            with self.assertRaises(opt.OptionError) as e:
                self.opts(*argv)
            self.assertIn(want, str(e.exception))
        self.assertEqual(self.opts('--variant', 'down1').get('VARIANT'), 'down1')
        self.assertTrue(os.path.isabs(self.opts('--variant', 'x/mine.json').get('VARIANT')))

    def test_resolve_model_sets_the_widths_and_the_digest_differs(self):
        digests = {}
        for name in ('home', 'down1'):
            o = self.opts('--v2', self.v2, '--v2-slices', self.v2, '--variant', name)
            ms, v, err = S.resolve_model(o)
            self.assertEqual(err, '')
            self.assertEqual(o.get('V2_WIDTHS'), os.path.join(self.v2, f'exlr8-manifest-{name}.json'))
            c, _, digests[name] = opt.canonical(o, opt.engine_plan(o), opt.split_list(PEERS), v)
            self.assertEqual((c['variant'], c['variant.sha256']), (name, v.sha256))
        self.assertNotEqual(digests['home'], digests['down1'])
        o = self.opts('--v2', self.v2, '--v2-slices', self.v2, '--v2-widths', os.path.join(self.v2, 'exlr8-manifest-down1.json'))
        self.assertEqual(S.resolve_model(o)[1].name, 'down1')          # --v2-widths names the same manifest
        _, v, err = S.resolve_model(self.opts('--variant', 'up1'))
        self.assertIn('needs the exlr8 v2 width files', err)


if __name__ == '__main__':
    unittest.main()
