"""Host-only check of the indexer's pipelined-logits wiring, and that the dispatch with both pipe knobs off is the plain one."""
import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def source(rel):
    return (ROOT / rel).read_text()


class IdxPipeDispatch(unittest.TestCase):
    def test_off_is_plain_dispatch_byte_identical(self):
        text = source("engine/attn/ds41_idx.cuh")
        start = text.index("inline void stage_index_dense(")
        end = text.index("// ST_CAND_BLOCKS", start)
        actual = text[start:end]
        q_new = (
            "    IdxQP q{in.q_b, in.wproj, in.inv, in.pos0, sc.q_pack, sc.q_exp, sc.w, in.dpos, sc.trig};\n"
            "    if (in.q_trig) launch_index_q_cached(q, in.R, s);\n"
            "    else launch_index_q(q, in.R, s);\n"
        )
        q_old = (
            "    IdxQP q{in.q_b, in.wproj, in.inv, in.pos0, sc.q_pack, sc.q_exp, sc.w, in.dpos};\n"
            "    launch_index_q(q, in.R, s);\n"
        )
        dispatch_new = (
            "    if (in.pipe_qr) launch_logits_pipe_qr(l, s);\n"
            "    else if (in.pipe_b) launch_logits_pipe(l, s, false);\n"
        )
        mma_new = "    else if (in.logits_mma) launch_logits_dense_mma(l, s, in.logits_mma == 1);"
        self.assertEqual(actual.count(q_new), 1)
        self.assertEqual(actual.count(dispatch_new), 1)
        self.assertEqual(actual.count(mma_new), 1)
        old = actual.replace(q_new, q_old, 1).replace(dispatch_new, "", 1).replace(
            mma_new, "    if (in.logits_mma) launch_logits_dense_mma(l, s, in.logits_mma == 1);", 1
        )
        # SHA256 of the stage_index_dense body without the pipe branches, including
        # query prep, both plain launch branches, and top-k dispatch.
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         "e6a07e8a1d22e0f7ce10d6790fa319621467fb0b73954d0f58a817ca481d1a6d")

    def test_pipe_qr_is_off_by_default(self):
        idx = source("engine/attn/ds41_idx.cuh")
        ops = source("engine/attn/ds41_idx_ops.cuh")
        main = source("engine/runtime/g6m4.cu")
        self.assertIn("bool pipe_qr = false;", idx)
        self.assertIn("bool pipe_qr = false;", ops)
        self.assertIn("S.pipe_qr = knob::ds41_idx_pipe_qr();", ops)
        self.assertIn("t.pipe_qr = 0;", ops)
        self.assertTrue("knob::ds41_idx_pipe_qr()" in main, "g6m4.cu does not read knob::ds41_idx_pipe_qr()")

    def test_only_four_source_layers_opt_in(self):
        ops = source("engine/attn/ds41_idx_ops.cuh")
        self.assertIn("t.pipe_b = S->pipe_b && !cand;", ops)
        self.assertIn("const bool cand = l > geom::CAND_SOURCE;", ops)
        self.assertIn("t.pipe_b = 0;", ops)
        self.assertIn("S.pipe_b = m.ds41_idx_pipe_b;", ops)
        self.assertIn("in.pipe_b = s->pipe_b != 0;", source("engine/attn/ds41_bind.cu"))

    def test_ica_slot_stride_and_all_users_share_padded_rpp(self):
        ops = source("engine/attn/ds41_idx_ops.cuh")
        bind = source("engine/attn/ds41_bind.cu")
        idx = source("engine/attn/ds41_idx.cuh")
        self.assertIn("S.rows_kv[kix] = ratio == 2 ? ds41i::ica_aligned_rows(nominal_kv_rows) : nominal_kv_rows;", ops)
        self.assertIn("S.slots * S.rows_kv[kix] * DS41_ICA_ROW", ops)
        self.assertIn("void* p = dmalloc<char>(bytes);", ops)
        self.assertIn("cudaMalloc(&p, n * sizeof(T))", source("engine/runtime/rank.cuh"))
        self.assertIn("pages_of(m, *S, row0), S->rows_kv[kix], q.len[j], stream", ops)
        self.assertIn("t.rpp = S.rows_kv[kix]", ops)
        self.assertIn("S->rows_kv[k] * DS41_ICA_ROW", ops)
        self.assertIn("p.ica = ica_of(ica, pages, rpp)", bind)
        self.assertIn("in.k = ica_of((void*)s->ica, s->pages, s->rpp)", bind)
        self.assertIn("(size_t)rpp * (IK_CODES + IK_SF)", bind)
        self.assertIn("pages[j / rows_per_page] * page_stride", idx)

    def test_boot_reads_the_knob(self):
        # GR8_DS41_IDX_PIPE_B is an agree knob: the knob table refuses a bad value and puts a set one in the runtime signature
        main = source("engine/runtime/g6m4.cu")
        self.assertTrue("knob::ds41_idx_pipe_b()" in main, "g6m4.cu does not read knob::ds41_idx_pipe_b()")
        self.assertTrue("m.ds41_idx_pipe_b =" in main, "g6m4.cu does not set m.ds41_idx_pipe_b")


if __name__ == "__main__":
    unittest.main()
