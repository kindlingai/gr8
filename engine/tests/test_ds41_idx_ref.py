"""engine/ref/ds41/idx_ref.py: the attention B reference (indexer, candidate blocks, row producers) on CPU, synthetic data.
python3 -m unittest -q test_ds41_idx_ref   (from engine/tests; no GPU)"""
import json
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ref', 'ds41'))
import idx_ref as R  # noqa: E402

CONFIGS = [os.path.join(HERE, '..', 'ref', 'ds41', 'config.json')]


def load_text():
    for p in CONFIGS:
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)['text_config']
    return None


class Topology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = load_text()

    def setUp(self):
        if self.text is None:
            self.skipTest('config.json not present')

    def test_roles(self):
        t = R.Topo(self.text)
        want = {0: 'none', 1: 'none', 2: 'dense', 3: 'reuse', 7: 'reuse', 8: 'dense', 14: 'dense', 15: 'reuse',
                20: 'dense+cand', 21: 'reuse', 23: 'reuse', 24: 'cand', 25: 'reuse', 28: 'cand', 32: 'cand', 36: 'cand', 39: 'reuse'}
        for l, role in want.items():
            self.assertEqual(t.role(l), role, f'layer {l}')
        self.assertEqual([t.kv_src[l] for l in (2, 7, 8, 13, 14, 19, 20, 39)], [2, 2, 8, 8, 14, 14, 20, 20])
        self.assertEqual([t.idx_src[l] for l in (21, 23, 24, 27, 36, 39)], [20, 20, 24, 24, 36, 36])
        self.assertEqual([t.idxk_src[l] for l in (24, 39)], [20, 20])   # index-K of a candidate consumer is layer 20's cache

    def test_validation_catches_breaks(self):
        for key, val in (('candidate_source_layer_id', 21), ('candidate_topk_blocks', 4), ('candidate_block_size', 32),
                         ('index_source_layer_ids', [2, 8, 14, 24, 28, 32, 36])):
            bad = dict(self.text)
            bad[key] = val
            with self.assertRaises(AssertionError, msg=key):
                R.Topo(bad)


class Casts(unittest.TestCase):
    def test_scale_exponent(self):
        x = torch.tensor([1.0, 1.0000001, 0.3, 5.0, 1e-30, 2.0 ** -126])
        e = R._log2_ceil(x)
        self.assertEqual(e[:4].tolist(), [0, 1, -1, 3])
        self.assertEqual(R._pow2(e[:4]).tolist(), [1.0, 2.0, 0.5, 8.0])

    def test_e2m1_midpoints_tie_to_even(self):
        x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0, -0.25, -0.75, -9.0])
        want = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -0.0, -1.0, -6.0]
        self.assertEqual(R.deq_e2m1(R.quant_e2m1(x)).tolist(), want)

    def test_ue8m0_scale_exact_cases(self):
        for amax, byte in ((6.0, 127), (6.5, 128), (3.0, 126), (0.75, 124)):
            x = torch.zeros(1, 32, dtype=torch.bfloat16)
            x[0, 0] = amax
            p, e = R.cast_fp4_ue8m0(x, 32)
            self.assertEqual(e.item(), byte, f'amax {amax}')
        x = torch.zeros(1, 32, dtype=torch.bfloat16)
        x[0, 0] = 6.0
        p, e = R.cast_fp4_ue8m0(x, 32)
        self.assertEqual(R.deq_fp4_ue8m0(p, e, 32)[0, 0].item(), 6.0)
        z = torch.zeros(1, 32, dtype=torch.bfloat16)
        pz, ez = R.cast_fp4_ue8m0(z, 32)
        self.assertEqual((ez.item(), pz.sum().item()), (1, 0))     # an all-zero group: scale 2^-126, codes 0

    def test_pack_roundtrip_and_scales(self):
        torch.manual_seed(1)
        x = torch.randn(5, 128).to(torch.bfloat16)
        p, e = R.cast_fp4_ue8m0(x, 32)
        self.assertEqual((tuple(p.shape), tuple(e.shape)), ((5, 64), (5, 4)))
        self.assertEqual(R.pack_nibbles(R.unpack_nibbles(p)).tolist(), p.tolist())
        d = R.deq_fp4_ue8m0(p, e, 32)
        self.assertLess(((d - x.float()).abs() / (x.float().abs().amax() + 1e-6)).max().item(), 0.35)
        p4, sf = R.cast_fp4_e4m3(x.repeat(1, 4), 16)
        self.assertEqual((tuple(p4.shape), tuple(sf.shape)), ((5, 256), (5, 32)))
        self.assertEqual(sf.dtype, torch.float8_e4m3fn)
        d4 = R.deq_fp4_e4m3(p4, sf, 16)
        self.assertLess((d4 - x.repeat(1, 4).float()).abs().max().item(), 0.5)
        z4, zs = R.cast_fp4_e4m3(torch.zeros(1, 16, dtype=torch.bfloat16), 16)
        self.assertEqual(zs.float().item(), 2.0 ** -9)             # an all-zero group keeps a nonzero scale

    def test_rms_norm_rounds_once(self):
        torch.manual_seed(7)
        x = torch.randn(3, 512).to(torch.bfloat16)
        w = (torch.rand(512) + 0.5).to(torch.bfloat16)
        got = R.rms_norm(x, w, 1e-6)
        xf = x.float()
        want = (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6))).to(torch.bfloat16)
        self.assertEqual(got.tolist(), want.tolist())
        two = R.rms_norm(x, torch.ones(512, dtype=torch.bfloat16), 1e-6) * w   # the DeepInfer double rounding differs
        self.assertGreater(int((two != got).sum()), 0)


class Rows(unittest.TestCase):
    def test_boundaries_and_klen(self):
        pos = torch.arange(8)
        b1, i1, r1 = R.row_positions(pos, 1)
        self.assertTrue(bool(b1.all()))
        self.assertEqual(i1.tolist(), r1.tolist())
        b2, i2, r2 = R.row_positions(pos, 2)
        self.assertEqual(b2.tolist(), [False, True] * 4)
        self.assertEqual(r2.tolist(), [0, 0, 0, 2, 0, 4, 0, 6])
        self.assertEqual(R.klen_of(pos, 1).tolist(), list(range(1, 9)))
        self.assertEqual(R.klen_of(pos, 2).tolist(), [0, 1, 1, 2, 2, 3, 3, 4])

    def test_rope(self):
        tab = R.rope_table(256)
        self.assertEqual(tab.shape, (256, 64))
        x = torch.randn(4, 64).to(torch.bfloat16)
        self.assertEqual(R.rope_interleaved(x, torch.zeros(4, dtype=torch.long), tab).tolist(), x.tolist())
        pos = torch.tensor([3, 50, 100, 255])
        y = R.rope_interleaved(x, pos, tab)
        back = R.rope_interleaved(y, pos, tab, conjugate=True)
        self.assertLess((back.float() - x.float()).abs().max().item(), 0.05)
        self.assertLess((y.float().norm(dim=-1) / x.float().norm(dim=-1) - 1).abs().max().item(), 0.02)

    def test_r1_ca_and_ica_rows(self):
        torch.manual_seed(2)
        T = 12
        tab = R.rope_table(64)
        pos = torch.arange(T)
        wkv = torch.randn(T, 512).to(torch.bfloat16)
        w = torch.ones(512, dtype=torch.bfloat16)
        pre, roped, (codes, sfs) = R.build_ca_row_r1(wkv, w, 1e-6, pos, tab)
        self.assertEqual(roped[:, :448].tolist(), pre[:, :448].tolist())   # only the trailing 64 dims rotate
        self.assertEqual(roped[0].tolist(), pre[0].tolist())               # position 0 is the identity
        self.assertEqual((tuple(codes.shape), tuple(sfs.shape)), ((T, 256), (T, 32)))
        kp = torch.randn(128, 512).to(torch.bfloat16)
        k, pk, ek = R.build_ica_row(pre, kp, torch.ones(128, dtype=torch.bfloat16), 1e-6, pos, tab)
        self.assertEqual((tuple(pk.shape), tuple(ek.shape)), ((T, 64), (T, 4)))


class Selection(unittest.TestCase):
    def test_dense_topk_sorted_and_padded(self):
        torch.manual_seed(3)
        p = torch.randn(3, 40).to(torch.bfloat16)
        klen = torch.tensor([40, 25, 5], dtype=torch.int32)
        out = R.topk_sorted(p, klen, 8)
        for r in range(3):
            n = min(8, int(klen[r]))
            row = out[r].tolist()
            self.assertEqual(row[:n], sorted(row[:n]))
            self.assertTrue(all(v == -1 for v in row[n:]))
            want = sorted(torch.topk(p[r, :int(klen[r])].float(), n).indices.tolist())
            self.assertEqual(row[:n], want)
        self.assertEqual(out[2, :5].tolist(), [0, 1, 2, 3, 4])   # fewer rows than k: every row selected

    def test_ties_take_lower_index(self):
        p = torch.zeros(1, 20, dtype=torch.bfloat16)
        p[0, 7] = 1.0
        out = R.topk_sorted(p, torch.tensor([20], dtype=torch.int32), 4)
        self.assertEqual(out[0].tolist(), [0, 1, 2, 7])

    def test_block_scores_last_block_inf(self):
        p = torch.randn(2, 24).to(torch.bfloat16)
        klen = torch.tensor([21, 8], dtype=torch.int32)
        sc, vb = R.block_scores(p, klen, 8)
        self.assertEqual(vb.tolist(), [3, 1])
        self.assertEqual(sc[0, 2].item(), float('inf'))
        self.assertEqual(sc[1, 0].item(), float('inf'))
        self.assertEqual(sc[0, 0].item(), p[0, :8].max().item())
        self.assertEqual(sc[0, 1].item(), p[0, 8:16].max().item())

    def test_candidate_pipeline_equals_bruteforce(self):
        torch.manual_seed(4)
        bs, nblk, topk = 8, 16, 12
        for levels in (None, 6):   # None: distinct scores; 6: heavy ties
            R_ = 5
            N = 200
            p = torch.randn(R_, N)
            if levels:
                p = (p * levels).round() / levels
            p = p.to(torch.bfloat16)
            klen = torch.tensor([200, 190, 131, 64, 9], dtype=torch.int32)
            p = p.masked_fill(torch.arange(N).unsqueeze(0) >= klen.long().unsqueeze(1), float('-inf'))
            sc, vb = R.block_scores(p, klen, bs)
            cand = R.topk_sorted(sc, vb, nblk)
            for r in range(R_):
                vals = cand[r][cand[r] >= 0].tolist()
                self.assertEqual(vals, sorted(vals))
                self.assertIn(int(vb[r]) - 1, vals)                     # the newest block is always a candidate
                self.assertTrue(all(v == -1 for v in cand[r, len(vals):].tolist()))
            cl = R.compact_lens(cand, klen, bs)
            cp = R.compact_psum(p, cand, klen, bs)
            for r in range(R_):
                fin = torch.isfinite(cp[r].float()) | (cp[r].float() == float('inf'))
                self.assertEqual(int(fin.sum()), int(cl[r]))             # valid compact columns are exactly a prefix
                self.assertTrue(bool(fin[:int(cl[r])].all()))
            got = R.indexer_candidate(p, cand, klen, topk, bs)
            for r in range(R_):
                toks = [b * bs + o for b in cand[r].tolist() if b >= 0 for o in range(bs) if b * bs + o < int(klen[r])]
                v = torch.tensor([p[r, t].float() for t in toks])
                order = torch.argsort(-v, stable=True)[:topk]
                want = sorted(toks[i] for i in order.tolist())
                n = min(topk, len(toks))
                self.assertEqual(got[r, :n].tolist(), want, f'row {r} levels {levels}')
                self.assertTrue(bool((got[r, n:] == -1).all()))

    def test_candidate_equals_dense_when_everything_is_a_candidate(self):
        torch.manual_seed(5)
        p = torch.randn(3, 96).to(torch.bfloat16)
        klen = torch.tensor([96, 70, 33], dtype=torch.int32)
        p = p.masked_fill(torch.arange(96).unsqueeze(0) >= klen.long().unsqueeze(1), float('-inf'))
        cand = R.candidate_blocks(p, klen, n_blocks=16, bs=8)   # 16 blocks of 8 cover 128 >= 96 tokens
        self.assertEqual(R.indexer_candidate(p, cand, klen, 20, 8).tolist(), R.indexer_dense(p, klen, 20).tolist())

    def test_end_to_end_small(self):
        torch.manual_seed(6)
        T, tab = 40, R.rope_table(64)
        pos = torch.arange(T)
        pre, _, _ = R.build_ca_row_r1(torch.randn(T, 512).to(torch.bfloat16), torch.ones(512, dtype=torch.bfloat16), 1e-6, pos, tab)
        _, pk, ek = R.build_ica_row(pre, torch.randn(128, 512).to(torch.bfloat16) * 0.05, torch.ones(128, dtype=torch.bfloat16), 1e-6, pos, tab)
        qp, qe = R.build_index_q(torch.randn(T, 4096).to(torch.bfloat16), pos, tab)
        w = torch.randn(T, 32).to(torch.bfloat16)
        klen = R.klen_of(pos, 1)
        p = R.mqa_psum(qp, qe, w, pk, ek, klen)
        self.assertTrue(bool(torch.isinf(p[0, 1:]).all()) and bool(torch.isfinite(p[0, 0])))
        self.assertTrue(bool(torch.isfinite(p[T - 1]).all()))
        top = R.indexer_dense(p, klen, 8)
        self.assertEqual(top[3, :4].tolist(), [0, 1, 2, 3])   # query 3 sees 4 rows, all selected
        self.assertTrue(bool((top[T - 1] >= 0).all()) and int(top[T - 1].max()) < T)


def official_fq_fp4(x, block, e4m3):
    """kernel.py fp4_act_quant(inplace=True) written as fake-quant: bf16 in, bf16 (dequantized) out."""
    n = x.shape[-1]
    v = x.float().reshape(*x.shape[:-1], n // block, block)
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3:
        s = (amax.clamp_min(6 * 2.0 ** -9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = 2.0 ** torch.ceil(torch.log2(amax.clamp_min(6 * 2.0 ** -126) * torch.tensor(1.0 / 6.0, dtype=torch.float32)))
    y = R.deq_e2m1(R.quant_e2m1((v / s).clamp(-6, 6))) * s
    return y.reshape(x.shape).to(torch.bfloat16)


def official_rope(x, freqs_cis):
    """apply_rotary_emb on [b, s, d]: complex product in fp32, rounded to bf16."""
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    return torch.view_as_real(z * freqs_cis).flatten(-2).to(torch.bfloat16)


def official_rms(x, w, eps):
    xf = x.float()
    return (w * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


class OfficialTranscription(unittest.TestCase):
    """The official decode path (model.py Compressor ratio 1 -> Indexer.forward -> select_candidate_blocks) with its own torch
    ops in bf16, against idx_ref's packed-code path. Same random inputs; the p_sum and the selections must be equal."""

    def test_ratio1_rows_logits_topk_and_candidates(self):
        torch.manual_seed(11)
        N, topk, nblk, bs = 150, 40, 5, 8
        tab = R.rope_table(256)
        freqs_cis = torch.complex(tab[:, :32], tab[:, 32:])
        pos = torch.arange(N)
        wkv_out = torch.randn(N, 512).to(torch.bfloat16)
        cw = (torch.rand(512) + 0.5).to(torch.bfloat16)
        wk = (torch.randn(128, 512) * 0.05).to(torch.bfloat16)
        kw = (torch.rand(128) + 0.5).to(torch.bfloat16)
        # official: Compressor (ratio 1) -> latent; Indexer owner: wk, k_norm, rope, fp4 (e8m0); Attention: rope, fp4 (e4m3/16)
        latent = official_rms(wkv_out, cw, 1e-6)
        k = official_rms((latent.float() @ wk.float().t()).to(torch.bfloat16), kw, 1e-6)
        k[..., -64:] = official_rope(k[..., -64:], freqs_cis[pos])
        k_cache = official_fq_fp4(k, 32, False)
        lat_roped = latent.clone()
        lat_roped[..., -64:] = official_rope(latent[..., -64:], freqs_cis[pos])
        ca_cache = official_fq_fp4(lat_roped, 16, True)
        # idx_ref
        pre, roped, (cp, cs) = R.build_ca_row_r1(wkv_out, cw, 1e-6, pos, tab)
        kk, pk, ek = R.build_ica_row(pre, wk, kw, 1e-6, pos, tab)
        self.assertEqual(pre.tolist(), latent.tolist())
        self.assertEqual(R.deq_fp4_ue8m0(pk, ek, 32).to(torch.bfloat16).tolist(), k_cache.tolist())
        self.assertEqual(R.deq_fp4_e4m3(cp, cs, 16).to(torch.bfloat16).tolist(), ca_cache.tolist())
        # decode one query at position N-1 (the last row is its own)
        x_w = torch.randn(1, 32).to(torch.bfloat16)                      # weights_proj(x)
        q_b = torch.randn(1, 4096).to(torch.bfloat16)
        at = N - 1
        q = q_b.reshape(1, 1, 32, 128).clone()
        q[..., -64:] = official_rope(q[..., -64:].reshape(1, 32, 64), freqs_cis[at:at + 1].expand(1, 32, 32)).reshape(1, 1, 32, 64)
        q = official_fq_fp4(q, 32, False)
        weights = x_w * (128 ** -0.5 * 32 ** -0.5)
        index_k = k_cache[None, :N]
        score = torch.einsum('bshd,btd->bsht', q, index_k)
        score = (score.relu_() * weights.reshape(1, 1, 32, 1)).sum(dim=2)[0]          # [1, N] bf16
        qp, qe = R.build_index_q(q_b, torch.tensor([at]), tab)
        p = R.mqa_psum(qp, qe, R.index_weights(x_w), pk, ek, torch.tensor([N], dtype=torch.int32))
        self.assertEqual(p.tolist(), score.tolist())
        # dense top-k, official form
        end = N
        idxs = score.topk(min(topk, end), dim=-1, sorted=False).indices.sort(dim=-1).values.int()
        want_dense = R.topk_sorted(p, torch.tensor([N], dtype=torch.int32), topk)
        if len(set(score[0].tolist())) == N:          # distinct scores: the selections are unique
            self.assertEqual(want_dense[0].tolist(), idxs[0].tolist())
        # candidate blocks, official form: block max over bf16, newest block pinned, keep mask, consumers mask and top-k
        width = score.size(-1)
        sc = torch.nn.functional.pad(score, (0, -width % bs), value=-torch.inf).unflatten(-1, (-1, bs)).amax(dim=-1)
        last = (end - 1) // bs
        sc = sc.masked_fill(torch.arange(sc.size(-1)) == last, torch.inf)
        top = sc.topk(min(nblk, sc.size(-1)), dim=-1)
        keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
        mask = keep.repeat_interleave(bs, dim=-1)[..., :width]
        cons_score = score.masked_fill(~mask, -torch.inf)
        official_cons = cons_score.topk(min(topk, end), dim=-1, sorted=False).indices.sort(dim=-1).values.int()
        klen = torch.tensor([N], dtype=torch.int32)
        cand = R.candidate_blocks(p, klen, nblk, bs)
        self.assertEqual(sorted(cand[0][cand[0] >= 0].tolist()), sorted(top.indices[0].tolist()) if len(set(sc[0].tolist())) == sc.size(-1) else sorted(cand[0][cand[0] >= 0].tolist()))
        got = R.indexer_candidate(p, cand, klen, topk, bs)
        if len(set(sc[0].tolist())) == sc.size(-1) and len(set(score[0].tolist())) == N:
            self.assertEqual(got[0].tolist(), official_cons[0].tolist())


if __name__ == '__main__':
    unittest.main()
