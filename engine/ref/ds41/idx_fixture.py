"""Fixtures for engine/attn/tests/ds41_idx_test.cu: random inputs and the idx_ref.py expected outputs, raw little-endian files in one
directory (bf16 as int16, fp4 records as uint8, indices as int32) plus dims.txt.
    python3 engine/ref/ds41/idx_fixture.py /tmp/ds41_idx_fx [seed]
    ds41_idx_test /tmp/ds41_idx_fx
Everything is CPU torch; the cos/sin come from torch (the CUDA kernels use sincosf), so the rope-dependent outputs may differ in the
last bf16 bit on a few elements: the test counts those and only requires exactness where no rope or matmul order is involved."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idx_ref as R  # noqa: E402

EPS = 1e-20


def dump(d, name, t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        a = t.view(torch.int16).numpy()
    elif t.dtype == torch.float8_e4m3fn:
        a = t.view(torch.uint8).numpy()
    else:
        a = t.numpy()
    a.tofile(os.path.join(d, name + '.bin'))


def main(d, seed=7):
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    T, POS0 = 6, 40000          # ratio-1 rows: positions 40000 .. 40005
    N = 20000                   # compressed rows of the logits / candidate tests (> 2048 blocks of 8 x 1.2)
    RQ = 3
    kl = torch.tensor([N, N - 1013, 777], dtype=torch.int32)
    table = R.rope_table(POS0 + T + 8)
    table_q = R.rope_table(max(N, POS0 + T) + 8)

    # --- ca row (ratio 1) + ica row
    y = rnd(T, 512, scale=3.0)
    nw = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
    pos = torch.arange(POS0, POS0 + T)
    pre, roped, (cpk, csf) = R.build_ca_row_r1(y, nw, EPS, pos, table)
    ca_rec = torch.cat((cpk, csf.view(torch.uint8)), dim=1)
    wk = rnd(128, 512, scale=0.05)
    kw = (1 + 0.1 * torch.randn(128, generator=g)).to(torch.bfloat16)
    k, ipk, iex = R.build_ica_row(pre, wk, kw, EPS, pos, table)
    dump(d, 'ca_y', y); dump(d, 'ca_normw', nw); dump(d, 'ca_pre', pre); dump(d, 'ca_roped', roped); dump(d, 'ca_rec', ca_rec)
    dump(d, 'ica_wk', wk); dump(d, 'ica_knormw', kw); dump(d, 'ica_k', k); dump(d, 'ica_pk', ipk); dump(d, 'ica_ex', iex)

    # --- index q + weights
    qb = rnd(RQ, 4096, scale=2.0)
    wp = rnd(RQ, 32, scale=1.0)
    qpos = torch.arange(POS0, POS0 + RQ)
    qpk, qex = R.build_index_q(qb, qpos, table)
    wq = R.index_weights(wp)
    dump(d, 'q_b', qb); dump(d, 'q_wproj', wp); dump(d, 'q_pk', qpk); dump(d, 'q_ex', qex); dump(d, 'q_w', wq)

    # --- logits over N keys with row-dependent klen, dense top-512, candidate stage, consumer
    keys = rnd(N, 128, scale=2.0)
    kpk, kex = R.cast_fp4_ue8m0(keys, 32)
    p = R.mqa_psum(qpk, qex, wq, kpk, kex, kl)
    dump(d, 'k_pk', kpk); dump(d, 'k_ex', kex); dump(d, 'klen', kl); dump(d, 'p', p)
    dump(d, 'topk', R.indexer_dense(p, kl, 512))
    sc, vb = R.block_scores(p, kl)
    dump(d, 'bscore', sc); dump(d, 'nvalid', vb)
    cand = R.candidate_blocks(p, kl)
    dump(d, 'cand', cand)
    qb2 = rnd(RQ, 4096, scale=2.0)
    wp2 = rnd(RQ, 32, scale=1.0)
    qpk2, qex2 = R.build_index_q(qb2, qpos, table)
    wq2 = R.index_weights(wp2)
    p2 = R.mqa_psum(qpk2, qex2, wq2, kpk, kex, kl)
    cp = R.compact_psum(p2, cand, kl)
    dump(d, 'q2_b', qb2); dump(d, 'q2_wproj', wp2); dump(d, 'q2_pk', qpk2); dump(d, 'q2_ex', qex2); dump(d, 'q2_w', wq2)
    dump(d, 'cp', cp)
    dump(d, 'clen', R.compact_lens(cand, kl))
    ci = R.topk_sorted(cp, R.compact_lens(cand, kl), 512)
    dump(d, 'ci', ci)
    dump(d, 'cons_topk', R.restore(ci, cand))
    # --- ratio-2 ica rows from the compressor's cnorm (engine step: device positions; only odd positions complete a pair)
    # chunk A = positions 40000..40005 (starts even): pairs complete at 40001, 40003, 40005 -> rows 20000..20002, rope positions j * 2.
    # chunk B = positions 40001..40004 (starts odd): completions at 40001 and 40003 -> rows 20000, 20001.
    # slot k of cnorm = the k-th completion of the call.
    cn = R.rms_norm(rnd(3, 512, scale=2.0), nw, EPS)
    rows_a = torch.tensor([20000, 20001, 20002])
    ka, apk, aex = R.build_ica_row(cn, wk, kw, EPS, rows_a * 2, table)
    dump(d, 'ica2_cn', cn); dump(d, 'ica2_pk', apk); dump(d, 'ica2_ex', aex)
    rows_b = torch.tensor([20000, 20001])
    kb, bpk, bex = R.build_ica_row(cn[:2], wk, kw, EPS, rows_b * 2, table)
    dump(d, 'ica2b_pk', bpk); dump(d, 'ica2b_ex', bex)
    # --- a small case where klen < 512 and < 2048 blocks (the n <= K branch)
    with open(os.path.join(d, 'dims.txt'), 'w') as fh:
        fh.write(f'{T} {POS0} {N} {RQ}\n')
    print('fixtures in', d, 'T', T, 'N', N, 'R', RQ, 'klen', kl.tolist(), 'valid blocks', vb.tolist())


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7)
