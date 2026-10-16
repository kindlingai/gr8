"""CPU checks of the G6_MOE_TOPK cut (flash-G4), no GPU:

    python3 engine/tests/test_moe_topk_source.py -v

1. OFF surface: the K1 / TR3 consumers, the plan layout and the GLM decode call site are a5edf0a's bytes; the cut's
   defaults are the model top-k (k_router_plan<K = TOPK>, Expert::topk = TOPK, moe_topk_cut unset -> model top-k); K is a
   template parameter, K = TOPK keeps the pre-cut source (SASS-identical, receipt /tmp/g4-sass2/cmp.log); the MTP drafter
   keeps the model's top-k (Block::topk set in prepare_flash only for the main model).
2. The routing rule the kernels implement (glue.cuh k_router_plan, bind_prefill.cuh k_pf_route), as a NumPy reference:
   K = TOPK is the unchanged rule; K < TOPK keeps the first K of the same selection order and renormalizes over them.
OFF byte identity of the built engine is a GPU check (m4sim stdout, base vs G4 build).
"""
import os
import subprocess
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = 'a5edf0a'
TOPK, NEXP, ROUTE_SCALE = 8, 288, 2.5


def src(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


class OffSurface(unittest.TestCase):
    UNTOUCHED = ['engine/kernels/k1.cuh', 'engine/kernels/k1args.cuh', 'engine/kernels/tr3.cuh', 'engine/kernels/bind.cuh',
                 'engine/runtime/rank.cuh', 'engine/model_geom.h']

    def test_consumers_are_base_bytes(self):
        for name in self.UNTOUCHED:
            base = subprocess.check_output(['git', '-C', ROOT, 'show', f'{BASE}:{name}']).decode()
            self.assertEqual(base, src(name), name)

    def test_defaults_are_model_topk(self):
        glue = src('engine/kernels/glue.cuh')
        self.assertIn('template <int K = TOPK> __global__ void __launch_bounds__(NEXP) k_router_plan(const float* logits, const float* bias, int M, RoutePlan* P)', glue)
        self.assertIn('int topk = TOPK;', src('engine/kernels/bind_prefill.cuh'))
        self.assertIn('template <int K = TOPK> __global__ void __launch_bounds__(256) k_pf_route(Expert A)', src('engine/kernels/bind_prefill.cuh'))
        self.assertIn('int topk = geom::TOPK;', src('engine/kernels/flash_mlp.cuh'))
        env = src('engine/runtime/envparse.h')
        self.assertIn('env_digit("G6_MOE_TOPK", 1, model_topk, model_topk, &err)', env)
        # the GLM decode call site passes no K (default TOPK)
        self.assertIn('glue::k_router_plan<<<1, glue::NEXP, 0, st>>>(m.logits, y.router_bias, M, m.plan);',
                      src('engine/runtime/rank.cuh'))

    def test_drafter_keeps_model_topk(self):
        step = src('engine/runtime/flash_step.cuh')
        self.assertIn('topk=m.flash_draft ? geom::TOPK : moe_topk_cut(geom::TOPK)', step)
        # the only other flash_mlp::launch caller is the main-model prefill P1 path
        self.assertIn('b.topk = moe_topk_cut(geom::TOPK);', src('engine/runtime/flash_prefill.cuh'))

    def test_expert_topk_is_last_member(self):
        pf = src('engine/kernels/bind_prefill.cuh')
        i = pf.index('struct Expert')
        j = pf.index('\n};', i)
        body = pf[i:j]
        self.assertLess(body.index('int nroutes'), body.index('int topk = TOPK'))
        self.assertLess(body.index('int f12'), body.index('int topk = TOPK'))      # layout of the pre-cut members unchanged

    def test_boot_validation_in_both_mains(self):
        for name in ('engine/runtime/g6m4.cu', 'engine/runtime/m4sim.cu'):
            s = src(name)
            self.assertIn('env_digit("G6_MOE_TOPK", 1, geom::TOPK, geom::TOPK, &e)', s, name)
            self.assertIn('if (tk < 0) die(e);', s, name)
            self.assertIn('die("G6_MOE_TOPK is a Flash-model option', s, name)

    def test_dropped_routes_skipped(self):
        pf = src('engine/kernels/bind_prefill.cuh')
        self.assertIn('A.tok_e[m * TOPK + l] = -1;', pf)          # k_pf_route: dropped route has no expert
        self.assertIn('A.pos[i] = -1;', pf)                        # k_pf_fill: no route index
        self.assertIn('if constexpr (CUT)', pf)                    # k_pf_fin / k_pf_fill: the cut is a separate instance
        self.assertIn('if (p[k] >= 0) add(A.yw', pf)               # k_pf_fin: skips it
        self.assertIn('A.nroutes = M * A.topk;', pf)               # route grids sized to the live routes
        glue = src('engine/kernels/glue.cuh')
        self.assertIn('if (k >= K) continue;', glue)               # decode scatter: tok_slot stays -1


def route(logits, bias, K):
    """The kernels' rule: key = sigmoid(logit) + bias, top-K by key (ties -> lower id, selection order),
    weights = sigmoid of the kept / their sum * ROUTE_SCALE."""
    s = 1 / (1 + np.exp(-logits.astype(np.float32)))
    key = s + bias
    order = np.lexsort((np.arange(NEXP), -key), axis=-1) if key.ndim == 1 else None
    sel = order[:K]
    w = s[sel] / (s[sel].sum() + np.float32(1e-20)) * ROUTE_SCALE
    return sel, w.astype(np.float32)


class Rule(unittest.TestCase):
    def test_cut_is_prefix_and_renormalized(self):
        rng = np.random.default_rng(5)
        for _ in range(64):
            lg = rng.normal(size=NEXP).astype(np.float32)
            b = (rng.normal(size=NEXP) * 0.1).astype(np.float32)
            e8, w8 = route(lg, b, TOPK)
            for K in range(1, TOPK + 1):
                eK, wK = route(lg, b, K)
                np.testing.assert_array_equal(eK, e8[:K])                 # same experts, same order
                self.assertAlmostEqual(float(wK.sum()), ROUTE_SCALE, places=5)
                s = 1 / (1 + np.exp(-lg[eK]))
                np.testing.assert_allclose(wK, s / s.sum() * ROUTE_SCALE, rtol=1e-6)

    def test_ties_to_lower_id(self):
        lg = np.zeros(NEXP, np.float32)
        b = np.zeros(NEXP, np.float32)
        e, _ = route(lg, b, 6)
        np.testing.assert_array_equal(e, np.arange(6))


if __name__ == '__main__':
    unittest.main()
