"""CPU source check of G6_PF_PREWARM: default ON at NR=2 (=0 kills it), call sites follow the lane builds, the helper makes only the lazy-state calls."""
from pathlib import Path
import re
import unittest

RT = Path(__file__).resolve().parents[1]


class PrewarmTest(unittest.TestCase):
    def test_helper_is_gated_and_minimal(self):
        src = (RT / 'flash_prefill.cuh').read_text()
        env = src.split('inline bool pf_prewarm_env()', 1)[1].split('\n}\n', 1)[0]
        self.assertIn('getenv("G6_PF_PREWARM")', env)
        self.assertIn('if (e && *e) return atoi(e) == 1;', env)   # set: =1 on, anything else off
        self.assertIn('return g6nr::NR == 2;', env)               # unset: on at TP2, off at TP4
        body = src.split('inline void pf_prewarm_flash', 1)[1].split('\n}\n', 1)[0]
        self.assertTrue(body.lstrip().startswith('(Model& m, FlashPrefill& P, cudaStream_t st)'))
        self.assertIn('if (!pf_prewarm_env() || !P.MP) return;', body)
        calls = re.findall(r'^\s+(\w+)\(', body.split('return;', 1)[1], re.M)
        self.assertEqual(calls, ['bind_dsa_prefill', 'bind_mlp_prefill', 'G6RT_CK', 'check_headroom', 'fprintf'])

    def test_call_sites_follow_the_lane_builds(self):
        src = (RT / 'g6m4.cu').read_text()
        a = src.index('build_flash_prefill(m, FP, MP')
        self.assertIn('pf_prewarm_flash(m, FP, st);', src[a:a + 200])
        b = src.index('build_flash_prefill(m, FP2, MP')
        self.assertIn('pf_prewarm_flash(m, FP2, st);', src[b:b + 200])
        self.assertEqual(src.count('pf_prewarm_flash('), 2)

    def test_prewarm_binds_like_the_first_chunk(self):
        # the first chunk binds MLP tensors with find = nullptr (pf_mlp_block_gemm); the prewarm must use the same path
        mlp = (RT / 'flash_mlp_pf.cuh').read_text()
        blk = mlp.split('inline void pf_mlp_block_gemm', 1)[1].split('}', 2)[0:2]
        self.assertIn('bind_layer_tensors(m, i, nullptr, st, S.T[i])', ''.join(blk))
        allv = mlp.split('inline void bind_mlp_prefill(Model& m, Prefill& P, cudaStream_t st)', 1)[1].split('\n}\n', 1)[0]
        self.assertIn('bind_layer_tensors(m, i, nullptr, st, S.T[i])', allv)
        # no non-test caller of the Find variant
        for p in list(RT.glob('*.cu')) + list(RT.glob('*.cuh')) + list(RT.glob('s2/*.cuh')):
            for ln in p.read_text().splitlines():
                if 'bind_mlp_prefill(' in ln and 'inline void' not in ln and not ln.lstrip().startswith('//'):
                    self.assertTrue(p.name == 'flash_prefill.cuh' and 'P, st)' in ln, (p.name, ln))

    def test_headroom_after_pool_and_nothing_later_sizes_from_free_memory(self):
        step = (RT / 'flash_step.cuh').read_text()
        self.assertIn('check_headroom(rank,capacity,true);', step)      # the KV pool gate, built before FP
        g = (RT / 'g6m4.cu').read_text()
        self.assertNotIn('MemAvailable', g)


if __name__ == '__main__':
    unittest.main()
