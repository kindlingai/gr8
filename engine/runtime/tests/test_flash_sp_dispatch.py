"""CPU source check of the Flash SP consume dispatch (G6_PF_RS=2): the owner mHC kernel is _rr_sp (the kernel of the STACK OFF path with
STAGE_ASYNC=1, wp == 0) unless G6_PF_MHC_RR=0 (then _pk_sp, the kernel of the OFF path with STAGE_ASYNC=0); FUSE 0 = out_inv_sp + prepare_range;
a config the SP path cannot run stops instead of falling back to the OFF consume."""
from pathlib import Path
import unittest

RT = Path(__file__).resolve().parents[1]
PF = (RT / 'flash_prefill.cuh').read_text()
G6 = (RT / 'g6m4.cu').read_text()


def body(src, head):
    return src.split(head, 1)[1].split('\n}\n', 1)[0]


class SpDispatchTest(unittest.TestCase):
    def test_kernel_select_rr_default_pk_with_rr0(self):
        b = body(PF, 'inline void pf_consume_flash_sp(')
        fused = b.split('if (pf_fuse >= 1)', 1)[1].split('k_mhc_out_inv_sp<<<', 1)[0]
        self.assertIn('if (pf_mhc_rr())', fused)                          # no FUSE level in the select: same kernel at FUSE 1 and 2
        self.assertNotIn('pf_fuse == 1', fused)
        self.assertLess(fused.index('k_mhc_out_inv_prepare_rr_sp'), fused.index('else'))
        self.assertIn('k_mhc_out_inv_prepare_pk_sp<MhcRsBlock, 2>', fused.split('else', 1)[1])   # G6_PF_MHC_RR=0
        self.assertEqual(b.count('k_mhc_out_inv_prepare_pk_sp'), 1)
        self.assertEqual(b.count('k_mhc_out_inv_prepare_rr_sp'), 1)
        self.assertIn('static const bool on = !(getenv("G6_PF_MHC_RR") && atoi(getenv("G6_PF_MHC_RR")) == 0);', body(PF, 'inline bool pf_mhc_rr()'))

    def test_startup_line_names_the_kernel(self):
        self.assertIn('two lanes" : "one lane",\n#if G6_MODEL_FLASH\n                       pf_fuse_level() < 1 ? "out_inv_sp" : pf_mhc_rr() ? "rr_sp" : "pk_sp");\n#else\n                       "none");\n#endif', G6)
        self.assertIn('5 collectives, slice %zu B, %s, mHC kernel %s', G6)

    def test_off_rule_unchanged(self):
        launch = body(PF, 'inline void pf_consume_launch')
        self.assertIn('const bool rr = !P.fnp.empty() && !wp && pf_mhc_rr();', launch)
        cons = body(PF, 'inline void pf_consume_flash(')
        self.assertIn('const size_t wp = pf_fuse >= 2 && !stage ? payload : 0;', cons)
        self.assertLess(cons.index('if (P.sp)'), cons.index('const size_t wp'))   # SP leaves before any OFF state

    def test_no_silent_fallback(self):
        # the SP dispatch is keyed on the build flag P.sp (g6m4 pf_sp = MP && G6_PF_RS=2), never on a runtime miss
        self.assertIn('if (P.sp) { pf_consume_flash_sp(m, P, i, c, M, st); return; }', PF)
        self.assertIn('const bool pf_sp = MP && pf_rs_env == 2;', G6)
        self.assertIn('build_flash_prefill(m, FP, MP, fp, dmalloc<u64>(1), st, pf_sp);', G6)
        self.assertIn('build_flash_prefill(m, FP2, MP, fp2, dmalloc<u64>(1), st, pf_sp);', G6)
        b = body(PF, 'inline void build_flash_prefill')
        gate = b.split('pf_stage_async_mode())', 1)[1].split('cudaStreamCreateWithPriority', 1)[0]
        self.assertIn('G6_PF_RS', gate)                                    # STAGE_ASYNC=1 + RS=2 dies; default at TP4 is off
        self.assertIn('if (no && mode == 2)', gate)
        self.assertIn('return g6nr::NR == 2 ? 1 : 0;', body(PF, 'inline int pf_stage_async_mode()'))
        self.assertIn('if (g6nr::NR != 4) die("G6_PF_RS=2', G6)


if __name__ == '__main__':
    unittest.main()
