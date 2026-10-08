"""SPEC §9 rank ownership map + per-rank pack sizes. The one place the converter reads ownership from.

Ranks r = 0..NR-1 are g6 ranks. For each rule "rank order rho_j = (j + off) % NR, widths w_j assigned contiguously
in that order", rank r = rho_j holds chunk j, with j = (r - off) % NR.

NR is 6 (TP6) unless set_nr() picks another count. The widths split n units the way engine/g6nr.h does: the larger
chunks first, n // NR + (j < n % NR). TP6: heads (11,11,11,11,10,10), expert blocks (3,3,3,3,2,2), vocab tiles
(202,202,202,202,201,201), dense groups 64 each. TP4: heads 16, expert blocks 4, vocab tiles (303,303,302,302),
dense groups 96 (3072 channels).
"""
import numpy as np

H = 6144
L_MAIN = 78          # backbone layers 0..77
L_MTP = 78
N_EXP = 256
N_HEADS = 64         # attention heads = shared-expert g32 groups (I = 2048)
N_BLOCKS = 16        # 128-column blocks of a routed expert (I = 2048)
N_VTILES = 1210      # 128-row lm_head tiles
N_DENSE_G = 384      # g32 groups of the dense MLP (I = 12288)
VOCAB = 154880


def widths(n, nr):
    """n units over nr ranks, as engine/g6nr.h width(): the larger chunks first."""
    return tuple(n // nr + (1 if j < n % nr else 0) for j in range(nr))


def set_nr(nr):
    """Pick the rank count. The engine build must match (-DG6_NR=nr)."""
    global NR, W_HEADS, WB, VOCAB_TILES, W_DENSE
    if nr not in (3, 6) and (nr <= 0 or 64 % nr):
        raise ValueError(f'rank count {nr}: 6 or a divisor of 64 (engine/g6nr.h)')
    NR = nr
    W_HEADS = widths(N_HEADS, nr)        # heads / shared-expert groups
    WB = widths(N_BLOCKS, nr)            # expert 128-blocks
    VOCAB_TILES = widths(N_VTILES, nr)
    W_DENSE = widths(N_DENSE_G, nr)


set_nr(6)


def _chunks(widths, off):
    """rank -> (start, count) in units of the widths."""
    starts = np.concatenate([[0], np.cumsum(widths)[:-1]])
    return {r: (int(starts[(r - off) % NR]), int(widths[(r - off) % NR])) for r in range(NR)}


def expert_blocks(l, e):
    """rank -> (first 128-block, n blocks) of routed expert e in layer l (16 blocks)."""
    return _chunks(WB, e + l)


def heads(l):
    """rank -> (first head, n heads) of the 64 attention heads."""
    return _chunks(W_HEADS, l)


def shared_groups(l):
    """rank -> (first g32 group, n groups) of the shared expert's 64 intermediate groups (I=2048)."""
    return _chunks(W_HEADS, l)


def dense_groups(l):
    """rank -> (first g32 group, n groups) of the dense MLP's 384 groups (I=12288): 64 per rank at TP6, 96 at TP4."""
    return _chunks(W_DENSE, l)


def vocab_rows(r):
    """(first row, n rows) of lm_head owned by rank r; not rotated."""
    s, n = _chunks(VOCAB_TILES, 0)[r]
    return s * 128, n * 128


def has_indexer(l):
    """Full indexer (own weights) layers: max(l-3+1,0)%4==0, and MTP always."""
    return l == L_MTP or max(l - 2, 0) % 4 == 0


def is_moe(l):
    return l >= 3


def _mx(n, k):     # MXFP8 codes + E8M0 scales
    return n * k + n * (k // 32)


def _gq(n, k):     # GPTQ int4 g32 codes + 16-bit scales (row-tile record is the same byte count)
    return n * k // 2 + n * (k // 32) * 2


def rank_bytes(r, bits_of):
    """Bytes of every tensor class rank r holds. bits_of(l) -> list of 256 expert bit widths."""
    out = {}

    def add(k, v):
        out[k] = out.get(k, 0) + v
    for l in range(L_MTP + 1):
        mtp = l == L_MTP
        nh = heads(l)[r][1]
        add('bf16_norms', 2 * (2 * H + 2048 + 512))
        if not mtp:
            add('gptq_q_a', _gq(2048, H))
            add('gptq_o_proj', _gq(H, nh * 256))
        add('mxfp8_q_a', _mx(2048, H))
        add('mxfp8_o_proj', _mx(H, nh * 256))
        add('mxfp8_kv_a', _mx(576, H))
        add('mxfp8_q_b', _mx(nh * 256, 2048))
        add('bf16_kv_b', nh * 448 * 512 * 2)
        if has_indexer(l):
            add('mxfp8_indexer_wq_b', _mx(4096, 2048))
            add('bf16_indexer', 2 * (128 * H + 32 * H + 256))
        if is_moe(l):
            ng = shared_groups(l)[r][1]
            if not mtp:
                add('gptq_shared', 3 * _gq(H, ng * 32))
            add('mxfp8_shared', 3 * _mx(H, ng * 32))
            add('bf16_router', 256 * H * 2 + 256 * 4)
            bits = bits_of(l)
            for e in range(N_EXP):
                nb = expert_blocks(l, e)[r][1]
                add('tr3_codes', nb * 9216 * 32 * bits[e])
                add('tr3_vectors', nb * (2 * H + 3 * 128 + H) * 2)
        else:
            nd = dense_groups(l)[r][1] * 32
            add('gptq_dense', 3 * _gq(H, nd))
            add('mxfp8_dense', 3 * _mx(H, nd))
        if mtp:
            add('mxfp8_mtp_eh', _mx(H, 2 * H))
            add('bf16_norms', 3 * 2 * H)
    add('bf16_embedding', VOCAB * H * 2)
    add('mxfp8_lm_head', _mx(vocab_rows(r)[1], H))
    add('bf16_norms', 2 * H)
    return out


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) > 2:
        set_nr(int(sys.argv[2]))
    tb = json.load(open(sys.argv[1]))
    bits_of = lambda l: tb[str(l)]['k']
    rows = [rank_bytes(r, bits_of) for r in range(NR)]
    keys = sorted(rows[0])
    print('| class | ' + ' | '.join(f'r{r}' for r in range(NR)) + ' |')
    print('|---|' + '---|' * NR)
    for k in keys:
        print(f'| {k} | ' + ' | '.join(f'{rows[r][k] / 1e9:.3f}' for r in range(NR)) + ' |')
    print('| **total GB** | ' + ' | '.join(f'**{sum(rows[r].values()) / 1e9:.2f}**' for r in range(NR)) + ' |')
