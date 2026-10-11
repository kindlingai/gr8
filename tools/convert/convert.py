"""gr8 weight converter: TR3 checkpoint + the old-rank D8 GPTQ dirs -> per-rank packs.

python3 -m tools.convert.convert --tr3 DIR --d8 R0 R1 R2 R3 R4 R5 --out DIR [--layers 0,3,40,77,78]
        [--ranks 0,1,2,3,4,5] [--skip embed,head] [--drop-cache]

The number of --d8 dirs sets the rank count: six (TP6, the default build) or four (TP4, an engine built with
-DG6_NR=4). The GPTQ codes are solved per rank slice, so TP4 needs the TP4 codes (tp4/r0..r3), not the TP6 ones.

Nothing is re-quantized except the D2 MXFP8 copies, which are produced from the original BF16 tensors with the
same deterministic equation D2 runs at boot (tools/convert/mxfp8.py). TR3 trellis/suh/svh bytes and D8 q4 codes
are copied; D8 BF16 scales are converted to the FP16 the row-tile records carry, and every one is checked exact.
One pass over the sources writes all requested ranks; per-rank packs are sequential files (tools/convert/pack.py).

Routed experts are TR3 or exlr8-e4m3, read per layer from the layer file's safetensors metadata (`exlr8`). Both
use the same tensor names and layout and convert the same way. An exlr8-e4m3 layer's tr3.* entries carry
`"expert_format": "exlr8-e4m3"` in the manifest (the engine then decodes its codebook with E4M3 rounding), and the
manifest's top-level `expert_formats` maps those layers to their format. TR3 layers get neither, so a TR3 pack and its
manifest are the same as before the exlr8-e4m3 format.

exlr8 v2 expert files (tools/convert/v2.py): --v2 DIR takes the routed experts of every layer that has v2 files in
DIR (or of --v2-layers) from them instead of from --tr3, at the widths --v2-widths picks, and stores the shared expert
as trellis row 256 of tr3.table (--v2-shared). Everything else in those layers still comes from --tr3 and --d8, and the
shared expert's int4 / MXFP8 records are left out. Such a layer's tr3.* entries carry `"expert_source": "exlr8-v2"` and
add tr3.kmix; the manifest's `v2_layers` maps them to their widths.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'engine', 'ref'))
import g6ref  # noqa: E402  (record layouts: rt_record / rt_expert_record)

from tools.convert import mxfp8, plan, slim, v2  # noqa: E402
from tools.convert.pack import PackWriter  # noqa: E402
from tools.convert.st import ST  # noqa: E402

H = plan.H
MCG = 0xCBAC1FED
RT_INT4 = g6ref.RT('int4')
VERSION = 'g6convert-v1'
# safetensors metadata `exlr8` of a layer file -> expert format (no key: TR3)
EXPERT_FORMATS = {None: 'tr3', 'exlr8-e4m3-v1': 'exlr8-e4m3'}
GV = False  # --gv-records: also write the decode GEMV record streams (g6ref.mxfp8_record / bf16_record)
V2 = None   # --v2: dict(dir, layers, widths, shared, shared_width)


# ---------------------------------------------------------------------------------------------- D8 GPTQ
def s4_to_f16(s4_bf16, what):
    """BF16 scale bits -> FP16, raising unless every value converts exactly."""
    f32 = mxfp8.bf16_to_f32(s4_bf16)
    f16 = f32.astype(np.float16)
    if not np.array_equal(f16.astype(np.float32), f32):
        bad = int((f16.astype(np.float32) != f32).sum())
        raise ValueError(f'{what}: {bad} BF16 scales not exactly representable in FP16')
    return f16


def _check_zero_pad(q, s, what):
    """Trimmed padding must decode to zero weights (code 8 or scale 0)."""
    if q.size and not ((q == 0x88).all() or (mxfp8.bf16_to_f32(s) == 0).all()):
        raise ValueError(f'{what}: non-zero padding in old D8 layout')


def d8_global(d8, l, module):
    """Reassemble the logical (unpadded) q4/s4 of one D8 module from the old ranks' files (six or four)."""
    name = f'model.layers.{l}.{module}'
    qs = [(f.get(name + '.q4'), f.get(name + '.s4')) for f in d8]
    if module.endswith('fused_qkv_a_proj'):
        for q, s in qs[1:]:
            if not (np.array_equal(q, qs[0][0]) and np.array_equal(s, qs[0][1])):
                raise ValueError(f'L{l} q_a: old replicas differ')
        if qs[0][0].shape != (2048, H // 2):
            raise ValueError(f'L{l} q_a: unexpected shape {qs[0][0].shape}')
        return qs[0]
    inter = 12288 if l < 3 else 2048
    if module.endswith('gate_up_proj'):
        out = []
        for j in (0, 1):   # j=0 codes, j=1 scales
            gate = np.concatenate([p[j][:p[j].shape[0] // 2] for p in qs], 0)
            up = np.concatenate([p[j][p[j].shape[0] // 2:] for p in qs], 0)
            out.append((gate, up))
        (gq, uq), (gs, us) = out
        _check_zero_pad(gq[inter:], gs[inter:], f'L{l} {module} gate')
        _check_zero_pad(uq[inter:], us[inter:], f'L{l} {module} up')
        return (gq[:inter], gs[:inter]), (uq[:inter], us[:inter])
    k = 16384 if module.endswith('o_proj') else inter
    q = np.concatenate([p[0] for p in qs], 1)
    s = np.concatenate([p[1] for p in qs], 1)
    _check_zero_pad(q[:, k // 2:], s[:, k // 32:], f'L{l} {module}')
    return q[:, :k // 2], s[:, :k // 32]


def cut_in(q, s, a, b):
    """Input-column cut [a, b) on whole g32 groups."""
    assert a % 32 == 0 and b % 32 == 0
    return q[:, a // 2:b // 2], s[:, a // 32:b // 32]


# ---------------------------------------------------------------------------------------------- helpers
def put_mx(w, name, bf16, gv=False, **info):
    put_mxq(w, name, mxfp8.quantize(bf16), gv, **info)


def put_mxq(w, name, cs, gv=False, **info):
    """Row-major MXFP8 (prefill); gv adds the decode GEMV record stream of the same codes/scales (densev.cuh)."""
    w.put(name + '.mx.codes', cs[0], 'u8', layout='mxfp8_e4m3fn_rowmajor', **info)
    w.put(name + '.mx.scales', cs[1], 'u8', layout='e8m0_biased127_rowmajor_g32', **info)
    if gv and GV:
        w.put(name + '.mx.gv', g6ref.mxfp8_record(*cs), 'u8', layout='mxfp8_gv_record[O/16][K/32][528]',
              rows=int(cs[0].shape[0]), cols=int(cs[0].shape[1]), **info)


def put_bf16(w, name, a, layout, **info):
    """Row-major BF16 + (with --gv-records) its GEMV record stream; leading dims fold into rows, as stored."""
    w.put(name, a, 'bf16', layout=layout, **info)
    if GV:
        a2 = np.ascontiguousarray(a).reshape(-1, a.shape[-1])
        w.put(name + '.gv', g6ref.bf16_record(a2), 'u8', layout='bf16_gv_record[O/16][K/16][512]',
              rows=int(a2.shape[0]), cols=int(a2.shape[1]), **info)


def put_rt(w, name, q, s, what, **info):
    rec = g6ref.rt_record(RT_INT4, np.ascontiguousarray(q), s4_to_f16(np.ascontiguousarray(s), what))
    w.put(name + '.gptq', rec, 'u8', layout='rt_int4_g32_record[O/16][K/32][288]', rows=int(q.shape[0]),
          cols=int(q.shape[1] * 2), **info)


def put_rt_expert(w, name, gate, up, down, what, **info):
    (gq, gs), (uq, us), (dq, ds) = gate, up, down
    f1, f2 = g6ref.rt_expert_record(RT_INT4, np.ascontiguousarray(gq), s4_to_f16(np.ascontiguousarray(gs), what),
                                    np.ascontiguousarray(uq), s4_to_f16(np.ascontiguousarray(us), what),
                                    np.ascontiguousarray(dq), s4_to_f16(np.ascontiguousarray(ds), what))
    w.put(name + '.gptq.fc1', f1, 'u8', layout='rt_int4_expert_fc1[I/16][192][gate288|up288]', **info)
    w.put(name + '.gptq.fc2', f2, 'u8', layout='rt_int4_expert_fc2[384][I/32][288]', **info)


def rng(t):
    return [int(t[0]), int(t[0] + t[1])]


# ---------------------------------------------------------------------------------------------- one layer
def convert_layer(l, src, d8, writers):
    p = f'model.layers.{l}.'
    mtp = l == plan.L_MTP
    moe = plan.is_moe(l)
    P = f'L{l:03d}'
    hd, sg, dg = plan.heads(l), plan.shared_groups(l), plan.dense_groups(l)
    g = lambda n: src.get(p + n)

    # ---- attention (replicated parts first, then head-split parts)
    attn_norm, mlp_norm = g('input_layernorm.weight'), g('post_attention_layernorm.weight')
    q_a_norm, kv_a_norm = g('self_attn.q_a_layernorm.weight'), g('self_attn.kv_a_layernorm.weight')
    q_a = g('self_attn.q_a_proj.weight')
    q_a_mx = mxfp8.quantize(q_a)
    kv_a_mx = mxfp8.quantize(g('self_attn.kv_a_proj_with_mqa.weight'))
    q_b = g('self_attn.q_b_proj.weight').reshape(64, 256, 2048)
    kv_b = g('self_attn.kv_b_proj.weight').reshape(64, 448, 512)
    o_proj = g('self_attn.o_proj.weight')
    if not mtp:
        qa_q, qa_s = d8_global(d8, l, 'self_attn.fused_qkv_a_proj')
        o_q, o_s = d8_global(d8, l, 'self_attn.o_proj')
    idx = None
    if plan.has_indexer(l):
        idx = dict(wq_b=mxfp8.quantize(g('self_attn.indexer.wq_b.weight')), wk=g('self_attn.indexer.wk.weight'),
                   kn_w=g('self_attn.indexer.k_norm.weight'), kn_b=g('self_attn.indexer.k_norm.bias'),
                   wp=g('self_attn.indexer.weights_proj.weight'))
    for r, w in writers.items():
        h0, nh = hd[r]
        hi = dict(heads=rng(hd[r]))
        w.put(f'{P}.attn_norm', attn_norm, 'bf16', layout='bf16_vector')
        if not mtp:
            put_rt(w, f'{P}.q_a', qa_q, qa_s, f'{P} q_a')
        put_mxq(w, f'{P}.q_a', q_a_mx, gv=mtp)   # gv=mtp: no int4 twin in the MTP layer, its decode path reads MXFP8
        w.put(f'{P}.q_a_norm', q_a_norm, 'bf16', layout='bf16_vector')
        put_mxq(w, f'{P}.kv_a', kv_a_mx, gv=True)
        w.put(f'{P}.kv_a_norm', kv_a_norm, 'bf16', layout='bf16_vector')
        put_mx(w, f'{P}.q_b', q_b[h0:h0 + nh].reshape(nh * 256, 2048), gv=True, **hi)
        put_bf16(w, f'{P}.kv_b.uk', np.ascontiguousarray(kv_b[h0:h0 + nh, :192]), 'bf16_rowmajor[h][192][512]', **hi)
        put_bf16(w, f'{P}.kv_b.uv', np.ascontiguousarray(kv_b[h0:h0 + nh, 192:]), 'bf16_rowmajor[h][256][512]', **hi)
        if not mtp:
            q, s = cut_in(o_q, o_s, h0 * 256, (h0 + nh) * 256)
            put_rt(w, f'{P}.o_proj', q, s, f'{P} o_proj', **hi)
        put_mx(w, f'{P}.o_proj', np.ascontiguousarray(o_proj[:, h0 * 256:(h0 + nh) * 256]), gv=mtp, **hi)
        if idx:
            put_mxq(w, f'{P}.idx.wq_b', idx['wq_b'], gv=True)
            put_bf16(w, f'{P}.idx.wk', idx['wk'], 'bf16_rowmajor')
            w.put(f'{P}.idx.k_norm.w', idx['kn_w'], 'bf16', layout='bf16_vector')
            w.put(f'{P}.idx.k_norm.b', idx['kn_b'], 'bf16', layout='bf16_vector')
            put_bf16(w, f'{P}.idx.weights_proj', idx['wp'], 'bf16_rowmajor')
        w.put(f'{P}.mlp_norm', mlp_norm, 'bf16', layout='bf16_vector')
    del q_a, q_a_mx, kv_a_mx, q_b, kv_b, o_proj, idx

    # ---- dense MLP (layers 0-2) / shared expert
    if moe:
        base, groups, inter = 'mlp.shared_experts.', sg, 2048
    else:
        base, groups, inter = 'mlp.', dg, 12288
    gate_b, up_b, down_b = g(base + 'gate_proj.weight'), g(base + 'up_proj.weight'), g(base + 'down_proj.weight')
    if not mtp:
        gq, uq = d8_global(d8, l, base + 'gate_up_proj')
        dq = d8_global(d8, l, base + 'down_proj')
    tag = 'shared' if moe else 'dense'
    if moe:
        router_w, router_b = g('mlp.gate.weight'), g('mlp.gate.e_score_correction_bias')
    # exlr8 v2 layer with the shared expert as trellis row 256: the engine runs it from there (prefill from an FP8 copy it
    # decodes at load), so its int4 / MXFP8 records are left out
    records = not (moe and V2 and l in V2['layers'] and V2['shared'] == 'trellis')
    for r, w in writers.items():
        a, b = groups[r][0] * 32, (groups[r][0] + groups[r][1]) * 32
        gi = dict(channels=[a, b])
        if moe:
            put_bf16(w, f'{P}.router.w', router_w, 'bf16_rowmajor')
            w.put(f'{P}.router.bias', router_b, 'f32', layout='f32_vector')
        if not records:
            continue
        if not mtp:
            put_rt_expert(w, f'{P}.{tag}', (gq[0][a:b], gq[1][a:b]), (uq[0][a:b], uq[1][a:b]),
                          cut_in(dq[0], dq[1], a, b), f'{P} {tag}', **gi)
        put_mx(w, f'{P}.{tag}.gate', gate_b[a:b], gv=mtp and moe, **gi)
        put_mx(w, f'{P}.{tag}.up', up_b[a:b], gv=mtp and moe, **gi)
        put_mx(w, f'{P}.{tag}.down', np.ascontiguousarray(down_b[:, a:b]), gv=mtp and moe, **gi)
    del gate_b, up_b, down_b

    # ---- routed experts (TR3 / exlr8 v1, or exlr8 v2)
    if moe and V2 and l in V2['layers']:
        convert_experts_v2(l, src, writers)
    elif moe:
        convert_experts(l, src, writers)

    # ---- MTP extras
    if mtp:
        eh = mxfp8.quantize(g('eh_proj.weight'))
        en, hn, sh = g('enorm.weight'), g('hnorm.weight'), g('shared_head.norm.weight')
        for r, w in writers.items():
            put_mxq(w, f'{P}.eh', eh, gv=True)
            w.put(f'{P}.enorm', en, 'bf16', layout='bf16_vector')
            w.put(f'{P}.hnorm', hn, 'bf16', layout='bf16_vector')
            w.put(f'{P}.shared_head_norm', sh, 'bf16', layout='bf16_vector')


def expert_format(l, src):
    """'tr3' or 'exlr8-e4m3' from the layer file's metadata. A sanity check on the label, not a proof of the format:
    exlr8-e4m3 uses one hidden-side sign vector per layer, the same in every expert and rank slice, and TR3 draws them
    per expert. Probed on experts 0-3: gate input signs of the first and last slice, down output signs of the first.
    It cannot see whether an encode rounded its codebook to E4M3. A file whose signs disagree with its label is refused."""
    tag = src.meta.get('exlr8')
    if tag not in EXPERT_FORMATS:
        raise ValueError(f'L{l}: unknown exlr8 metadata {tag!r}')
    fmt = EXPERT_FORMATS[tag]
    p = f'model.layers.{l}.mlp.experts.'
    last = sum(1 for k in src.h if k.startswith(f'{p}0.gate_proj.rank') and k.endswith('.suh')) - 1
    groups = [[('gate', r, 'suh') for r in (0, last)], [('down', 0, 'svh')]]
    shared = True
    for g in groups:
        signs = [np.signbit(src.get(f'{p}{e}.{m}_proj.rank{r}.{v}')) for m, r, v in g for e in range(4)]
        shared &= all(np.array_equal(signs[0], x) for x in signs[1:])
    if shared != (fmt == 'exlr8-e4m3'):
        raise ValueError(f'L{l}: metadata says {fmt}, but the hidden-side signs of experts 0-3 are '
                         f'{"shared" if shared else "not shared"}')
    return fmt


def convert_experts(l, src, writers):
    """Pass 1 streams fc1 (gate/up tiles), pass 2 streams fc2 (down tiles); vectors are buffered (<= 20 MB/rank)."""
    P = f'L{l:03d}'
    p = f'model.layers.{l}.mlp.experts.'
    fmt = expert_format(l, src)
    xf = {} if fmt == 'tr3' else dict(expert_format=fmt)   # TR3 entries stay as before
    if xf:
        for w in writers.values():
            w.meta.setdefault('expert_formats', {})[str(l)] = fmt
    bits = []
    for e in range(plan.N_EXP):
        k = src.shape(f'{p}{e}.gate_proj.rank0.trellis')[-1] // 16
        if k not in (3, 4):
            raise ValueError(f'{P} expert {e}: bits {k}')
        bits.append(k)
    own = {r: [plan.expert_blocks(l, e)[r] for e in range(plan.N_EXP)] for r in writers}
    table = {r: np.zeros((plan.N_EXP, 6), np.int64) for r in writers}
    nblk = {r: sum(n for _, n in own[r]) for r in writers}
    v1 = {r: np.empty((nblk[r], 2, H), np.float16) for r in writers}
    v2 = {r: np.empty((nblk[r], 3, 128), np.float16) for r in writers}
    v3 = {r: np.empty((nblk[r], H), np.float16) for r in writers}
    fc1_bytes = {r: sum(n * 2 * 384 * 8 * 32 * bits[e] for e, (_, n) in enumerate(own[r])) for r in writers}
    fc2_bytes = {r: sum(n * 384 * 8 * 32 * bits[e] for e, (_, n) in enumerate(own[r])) for r in writers}

    def frag(e, mat, f):
        n = f'{p}{e}.{mat}_proj.rank{f}.'
        mcg = int(src.get(n + 'mcg').astype(np.int64)) & 0xffffffff
        if mcg != MCG:
            raise ValueError(f'{n}mcg = {mcg:#x}')
        t = src.get(n + 'trellis')
        if t.shape[-1] != 16 * bits[e]:
            raise ValueError(f'{n}: bits differ inside expert')
        return t, src.get(n + 'suh'), src.get(n + 'svh')

    # pass 1: fc1 = gate/up tiles, v1 (suh gate/up), v2[0:2] (svh gate/up slices)
    for r, w in writers.items():
        w.begin(f'{P}.tr3.fc1', 'u8', (fc1_bytes[r],), layout='tr3_fc1 per expert per block [2][384][8][TB], TB=32*bits', **xf)
    fc1_off = {r: 0 for r in writers}
    row = {r: 0 for r in writers}
    for e in range(plan.N_EXP):
        TB = 32 * bits[e]
        need = set().union(*(_frags(own[r][e]) for r in writers))
        gu = {f: (frag(e, 'gate', f), frag(e, 'up', f)) for f in sorted(need)}
        for r, w in writers.items():
            b0, n = own[r][e]
            table[r][e] = (fc1_off[r], 0, row[r], n, bits[e], b0)
            for b in range(b0, b0 + n):
                f, lb = divmod(b, 4)
                (gt, gsuh, gsvh), (ut, usuh, usvh) = gu[f]
                for t in (gt, ut):
                    if t.shape[:2] != (384, 32):
                        raise ValueError(f'{P} e{e} gate/up trellis shape {t.shape}')
                    w.write(t[:, lb * 8:(lb + 1) * 8, :])
                v1[r][row[r]] = (gsuh, usuh)
                v2[r][row[r], 0] = gsvh[lb * 128:(lb + 1) * 128]
                v2[r][row[r], 1] = usvh[lb * 128:(lb + 1) * 128]
                row[r] += 1
            fc1_off[r] += n * 2 * 384 * 8 * TB
    for w in writers.values():
        w.end()

    # pass 2: fc2 = down tiles transposed to [n-tile][k-tile], v2[2] (suh down slice), v3 (svh down)
    for r, w in writers.items():
        w.begin(f'{P}.tr3.fc2', 'u8', (fc2_bytes[r],), layout='tr3_fc2 per expert per block [384][8][TB]', **xf)
    fc2_off = {r: 0 for r in writers}
    row = {r: 0 for r in writers}
    for e in range(plan.N_EXP):
        TB = 32 * bits[e]
        need = set().union(*(_frags(own[r][e]) for r in writers))
        dn = {f: frag(e, 'down', f) for f in sorted(need)}
        for r, w in writers.items():
            b0, n = own[r][e]
            table[r][e][1] = fc2_off[r]
            for b in range(b0, b0 + n):
                f, lb = divmod(b, 4)
                dt, dsuh, dsvh = dn[f]
                if dt.shape[:2] != (32, 384):
                    raise ValueError(f'{P} e{e} down trellis shape {dt.shape}')
                w.write(np.ascontiguousarray(dt[lb * 8:(lb + 1) * 8].transpose(1, 0, 2)))
                v2[r][row[r], 2] = dsuh[lb * 128:(lb + 1) * 128]
                v3[r][row[r]] = dsvh
                row[r] += 1
            fc2_off[r] += n * 384 * 8 * TB
    for r, w in writers.items():
        w.end()
        w.put(f'{P}.tr3.v1', v1[r], 'f16', layout='[blk][2][6144] suh gate, suh up (block fragment)', **xf)
        w.put(f'{P}.tr3.v2', v2[r], 'f16', layout='[blk][3][128] svh gate, svh up, suh down (block slice)', **xf)
        w.put(f'{P}.tr3.v3', v3[r], 'f16', layout='[blk][6144] svh down (block fragment)', **xf)
        w.put(f'{P}.tr3.table', table[r], 'i64',
              layout='[256][6] fc1_byte_off, fc2_byte_off, first_blk_row, n_blocks, bits, first_block(0..15)', **xf)


def convert_experts_v2(l, src, writers):
    """Routed (and shared) experts of layer l from the v2 files; the v1 layer file only gives the home split."""
    bits = v2.v1_bits_of(src, l)
    sw = V2['shared_width'] if V2['shared'] == 'trellis' else 0
    wd, prov = v2.load_widths(V2['widths'], l, bits, sw)
    lay = v2.V2Layer(V2['dir'], l)
    for w in writers.values():
        w.meta.setdefault('expert_formats', {})[str(l)] = 'exlr8-e4m3'
        w.meta.setdefault('v2_layers', {})[str(l)] = prov
    v2.convert_experts_v2(l, lay, wd, writers, plan.NR, P=f'L{l:03d}',
                          info=dict(expert_format='exlr8-e4m3', expert_source='exlr8-v2'), shared=V2['shared'] == 'trellis')
    lay.close()


def _frags(owned):
    b0, n = owned
    return {b // 4 for b in range(b0, b0 + n)}


# ---------------------------------------------------------------------------------------------- globals
def convert_embed(src_embed, writers, chunk=8192):
    name = 'model.embed_tokens.weight'
    n, k = src_embed.shape(name)
    for w in writers.values():
        w.begin('G.embed', 'bf16', (n, k), layout='bf16_rowmajor[vocab][6144]')
    for a in range(0, n, chunk):
        rows = src_embed.get(name, rows=(a, min(n, a + chunk)))
        for w in writers.values():
            w.write(rows)
    for w in writers.values():
        w.end()


def convert_head(src_head, writers):
    norm = src_head.get('model.norm.weight')
    for r, w in writers.items():
        a, n = plan.vocab_rows(r)
        w.put('G.final_norm', norm, 'bf16', layout='bf16_vector')
        lm = src_head.get('lm_head.weight', rows=(a, a + n))
        put_mx(w, 'G.lm_head', lm, gv=True, vocab_rows=[a, a + n])


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tr3', required=True, help='TR3 or exlr8-e4m3 checkpoint dir (model-layer-NNN / model-embed / model-head)')
    ap.add_argument('--d8', nargs='+', required=True,
                    help='old D8 gptq-w32 dirs in old-rank order: six (r0..r5, TP6) or four (r0..r3, TP4)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nr', type=int, default=0, help='rank count of the new packs (default: the number of --d8 dirs); 3 = TP3')
    ap.add_argument('--layers', default='all')
    ap.add_argument('--ranks', default='all')
    ap.add_argument('--skip', default='', help='comma list of globals to leave out: embed, head')
    ap.add_argument('--drop-cache', action='store_true', help='fdatasync + FADV_DONTNEED written and read pages')
    ap.add_argument('--slim-qo', metavar='QO_MANIFEST', help='write slim packs (tools/convert/slim.py): leave out the dead kv_b .gv and, in the layers '
                    'the QO pack holds, the q_b / o_proj .mx.codes + .scales. The QO pack must exist already (qopack.py reads those tensors from a full pack); '
                    '`all` = layers 0-77')
    ap.add_argument('--gv-records', action='store_true', help='add .mx.gv / .gv GEMV record tensors (densev.cuh)')
    ap.add_argument('--v2', help='exlr8 v2 expert files dir: routed + shared experts of its layers come from it')
    ap.add_argument('--v2-layers', default='auto', help='comma list, or auto = every layer with v2 files in --v2')
    ap.add_argument('--v2-widths', default='home', help='home | down1 | up1 | k2 | k3 | k4 | a width manifest file')
    ap.add_argument('--v2-shared', default='trellis', choices=('trellis', 'off'),
                    help='store the shared expert as trellis row 256 (then its int4 / MXFP8 records are left out)')
    ap.add_argument('--v2-shared-width', type=int, default=None, choices=(3, 4),
                    help="the shared expert's width (default: the manifest's; K4 for home and up1, K3 for down1)")
    a = ap.parse_args(argv)
    if len(a.d8) not in (4, 6):
        ap.error(f'--d8 takes six dirs (TP6) or four (TP4), got {len(a.d8)}')
    # d8_global rebuilds each dense matrix from all old ranks, so the new rank count need not match the old one
    plan.set_nr(a.nr or len(a.d8))
    global GV, V2
    GV = a.gv_records
    layers = list(range(plan.L_MTP + 1)) if a.layers == 'all' else [int(x) for x in a.layers.split(',')]
    if a.v2:
        have = [l for l in range(plan.L_MTP + 1) if os.path.exists(os.path.join(a.v2, v2.INDEX_FILE.format(l=l)))]
        vl = have if a.v2_layers == 'auto' else [int(x) for x in a.v2_layers.split(',')]
        if not set(vl) <= set(have) or not vl:
            ap.error(f'--v2: layers {sorted(set(vl) - set(have)) or vl} have no v2 files in {a.v2}')
        V2 = dict(dir=a.v2, layers=set(vl), widths=a.v2_widths, shared=a.v2_shared, shared_width=a.v2_shared_width)
    ranks = list(range(plan.NR)) if a.ranks == 'all' else [int(x) for x in a.ranks.split(',')]
    if any(r < 0 or r >= plan.NR for r in ranks):
        ap.error(f'--ranks must be 0..{plan.NR - 1}')
    os.makedirs(a.out, exist_ok=True)
    skip = set(filter(None, a.skip.split(',')))
    meta = dict(converter=VERSION, layers=layers, globals=sorted({'embed', 'head'} - skip), spec='SPEC.md §9 rank map')
    if plan.NR != 6:
        meta['nr'] = plan.NR   # TP6 manifests stay as they were
    if GV:
        meta['gv_records'] = True
    if a.slim_qo:
        qo = set(range(78)) if a.slim_qo == 'all' else slim.qo_layers(a.slim_qo)
        if not qo:
            ap.error(f'--slim-qo {a.slim_qo}: no layer with all of the QO tensors')
        writers = {r: slim.SlimWriter(os.path.join(a.out, f'rank{r}.pack'), r, a.drop_cache, meta, qo=qo) for r in ranks}
    else:
        writers = {r: PackWriter(os.path.join(a.out, f'rank{r}.pack'), r, a.drop_cache, meta) for r in ranks}
    if 'embed' not in skip:
        e = ST(os.path.join(a.tr3, 'model-embed.safetensors'))
        convert_embed(e, writers)
        if a.drop_cache:
            e.drop_cache()
        e.close()
    t0 = time.time()
    for l in layers:
        tl = time.time()
        src = ST(os.path.join(a.tr3, f'model-layer-{l:03d}.safetensors'))
        d8 = [] if l == plan.L_MTP else [ST(os.path.join(d, f'layer-{l:03d}.safetensors')) for d in a.d8]
        convert_layer(l, src, d8, writers)
        for f in [src] + d8:
            if a.drop_cache:
                f.drop_cache()
            f.close()
        print(f'layer {l} done in {time.time() - tl:.1f} s (total {time.time() - t0:.0f} s)', flush=True)
    if 'head' not in skip:
        hsrc = ST(os.path.join(a.tr3, 'model-head.safetensors'))
        convert_head(hsrc, writers)
        if a.drop_cache:
            hsrc.drop_cache()
        hsrc.close()
    for r, w in writers.items():
        m = w.close()
        print(f'rank {r}: {m["pack_bytes"]} bytes, {len(m["tensors"])} tensors', flush=True)


if __name__ == '__main__':
    main()
