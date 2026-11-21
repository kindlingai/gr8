"""Regression gate: CP prefill staging must complete before pf_ix reads/appends index keys.

With GR8_PF_OWN=2 / GR8_PF_OWN=4 (P.ownix=true) and CP prefill staging active,
pf_ix(m, P, i - 1, ...) appends and scores layer i's index keys in staged pool[i % 2]
using layer-step ls + i - 1 (collective c4).

Therefore, in every prefill loop (pf_step, pf_step2, pf_step2s in prefill.cuh, and
enqueue_pf, enqueue_pf2 in m4sim.cu):
  1. cp_pf_pre(m, i, ...) / cp_gather(i, ...) MUST run BEFORE pf_ix(..., i - 1, ...),
     so peer index pages for layer i have arrived before index append/scoring reads them.
  2. pf_ix(..., i - 1, ...) MUST run BEFORE pf_set_step(..., ls + i, ...),
     so collective c4 still uses layer-step ls + i - 1.
  3. No pf_ix(..., i, ...) may run at the tail of iteration i (after cp_pf_prefetch(m, i)),
     before iteration i + 1's cp_pf_pre(m, i + 1, ...).
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _extract_body(text: str, anchor: str, end_anchor: str) -> str:
    start = text.index(anchor)
    end = text.index(end_anchor, start)
    return text[start:end]


class CpPfIxOrderTests(unittest.TestCase):
    def test_prefill_cuh_pf_step(self):
        src = (ROOT / 'engine/runtime/prefill.cuh').read_text()
        body = _extract_body(src, 'inline void pf_step(', 'inline void pf_step2(')
        loop = _extract_body(body, 'for (int i = 0; i < m.nl; ++i)', 'pf_finish(P, M, st);')
        p_pre = loop.index('cp_pf_pre(m, i,')
        p_ix = loop.index('if (i > 0) pf_ix(m, P, i - 1,')
        p_step = loop.index('pf_set_step(P, ls0 + i,')
        p_attn = loop.index('pf_attn_block(m, P, i,')
        self.assertLess(p_pre, p_ix, 'cp_pf_pre(i) must precede pf_ix(i - 1) in pf_step')
        self.assertLess(p_ix, p_step, 'pf_ix(i - 1) must precede pf_set_step(ls0 + i) in pf_step')
        self.assertLess(p_step, p_attn, 'pf_set_step(ls0 + i) must precede pf_attn_block(i) in pf_step')
        self.assertNotRegex(loop, r'pf_ix\s*\(\s*m\s*,\s*P\s*,\s*i\s*,',
                            'pf_step must not call pf_ix(..., i, ...) before next layer cp_pf_pre')

    def test_prefill_cuh_pf_step2(self):
        src = (ROOT / 'engine/runtime/prefill.cuh').read_text()
        body = _extract_body(src, 'inline void pf_step2(', 'struct PfLanes')
        loop = _extract_body(body, 'for (int i = 0; i < m.nl; ++i)', 'pf_finish_put(A, MA, st);')
        p_pre = loop.index('cp_pf_pre(m, i,')
        p_ixa = loop.index('if (i > 0) pf_ix(m, A, i - 1,')
        p_stepa = loop.index('pf_set_step(A, lsA + i,')
        p_ixb = loop.index('pf_ix(m, B, i - 1,')
        p_stepb = loop.index('pf_set_step(B, lsB + i,')
        self.assertLess(p_pre, p_ixa, 'cp_pf_pre(i) must precede pf_ix(A, i - 1) in pf_step2')
        self.assertLess(p_ixa, p_stepa, 'pf_ix(A, i - 1) must precede pf_set_step(A, lsA + i) in pf_step2')
        self.assertLess(p_pre, p_ixb, 'cp_pf_pre(i) must precede pf_ix(B, i - 1) in pf_step2')
        self.assertLess(p_ixb, p_stepb, 'pf_ix(B, i - 1) must precede pf_set_step(B, lsB + i) in pf_step2')
        self.assertNotRegex(loop, r'pf_ix\s*\(\s*m\s*,\s*A\s*,\s*i\s*,',
                            'pf_step2 must not call pf_ix(A, i, ...) before next layer cp_pf_pre')

    def test_prefill_cuh_pf_step2s(self):
        src = (ROOT / 'engine/runtime/prefill.cuh').read_text()
        body = _extract_body(src, 'inline void pf_step2s(', 'struct DraftPrefill')
        loop = _extract_body(body, 'for (int i = 0; i < m.nl; ++i)', 'pf_finish(A, MA, sa);')
        p_pre = loop.index('cp_pf_pre(m, i,')
        p_ixa = loop.index('if (i > 0) pf_ix(m, A, i - 1,')
        p_stepa = loop.index('pf_set_step(A, lsA + i,')
        p_attna = loop.index('pf_attn_block(m, A, i,')
        p_ixb = loop.index('if (i > 0) pf_ix(m, B, i - 1, MB, ctxB, sb, L.ea[i])')
        p_stepb = loop.index('pf_set_step(B, lsB + i,')
        self.assertLess(p_pre, p_ixa, 'cp_pf_pre(i) must precede pf_ix(A, i - 1) in pf_step2s')
        self.assertLess(p_ixa, p_stepa, 'pf_ix(A, i - 1) must precede pf_set_step(A, lsA + i) in pf_step2s')
        self.assertLess(p_stepa, p_attna)
        self.assertLess(p_attna, p_ixb, 'Lane B pf_ix must wait on L.ea[i] (recorded after cp_pf_pre(i))')
        self.assertLess(p_ixb, p_stepb, 'pf_ix(B, i - 1) must precede pf_set_step(B, lsB + i) in pf_step2s')

    def test_m4sim_enqueue_pf_and_pf2(self):
        src = (ROOT / 'engine/runtime/m4sim.cu').read_text()
        pf1 = _extract_body(src, 'auto enqueue_pf = ', 'auto enqueue_pf2 = ')
        p_g1 = pf1.index('cp_gather(i,')
        p_ix1 = pf1.index('if (i > 0) pf_deliver_ix(PF, loop_pf, i - 1,')
        p_s1 = pf1.index('pf_set_step(PF[0], ls0 + i,')
        self.assertLess(p_g1, p_ix1, 'cp_gather(i) must precede pf_deliver_ix(PF, i - 1) in enqueue_pf')
        self.assertLess(p_ix1, p_s1, 'pf_deliver_ix(PF, i - 1) must precede pf_set_step(ls0 + i) in enqueue_pf')
        self.assertNotRegex(pf1, r'pf_deliver_ix\s*\(\s*PF\s*,\s*loop_pf\s*,\s*i\s*,')

        pf2 = _extract_body(src, 'auto enqueue_pf2 = ', 'u64 ls0 = 1')
        loop2 = _extract_body(pf2, 'for (int i = 0; i < NL; ++i)', 'pf_fin_rs(PF,')
        p_g2 = loop2.index('cp_gather(i,')
        p_ixa2 = loop2.index('if (i > 0) pf_deliver_ix(PF, loop_pf, i - 1,')
        p_attna2 = loop2.index('attn(PF, loop_pf, i,')
        p_ixb2 = loop2.index('pf_deliver_ix(PF2, loop_pf2, i - 1,')
        p_attnb2 = loop2.index('attn(PF2, loop_pf2, i,')
        self.assertLess(p_g2, p_ixa2, 'cp_gather(i) must precede pf_deliver_ix(PF, i - 1) in enqueue_pf2')
        self.assertLess(p_ixa2, p_attna2, 'pf_deliver_ix(PF, i - 1) must precede attn(PF, i) in enqueue_pf2')
        self.assertLess(p_g2, p_ixb2, 'cp_gather(i) must precede pf_deliver_ix(PF2, i - 1) in enqueue_pf2')
        self.assertLess(p_ixb2, p_attnb2, 'pf_deliver_ix(PF2, i - 1) must precede attn(PF2, i) in enqueue_pf2')


if __name__ == '__main__':
    unittest.main()
