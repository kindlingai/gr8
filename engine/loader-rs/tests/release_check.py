"""Checks a rank image planned from the exlr8 release (g6plandump, src/release.rs) against an independent derivation:
the reference loader's decode (loader/exlr8.py), the converter's record layouts (engine/ref/g6ref.py), its MXFP8
equation (tools/convert/mxfp8.py, here on fp32 inputs), a NumPy int4 round-to-nearest and the converter's ownership
rules (tools/convert/plan.py). Every non-expert tensor must match byte for byte, and the planned image must hold
exactly the names a converted pack has (tools/convert/convert.py), less the ones no engine path reads.

    python3 engine/loader-rs/tests/release_check.py --release DIR --widths FILE --ref EXLR8_REPO \
        --dump engine/loader-rs/target/release/g6plandump --tp 4 --ranks 0,1,2,3 --layers 0,3,4,6,78 [--qo]

Run from the repo root. Needs torch and NumPy.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'engine', 'ref')]
import g6ref  # noqa: E402
from tools.convert import mxfp8, plan  # noqa: E402

H = 6144


def mx_quantize(w):
    """tools/convert/mxfp8.quantize's equation on fp32 rows [n][k]."""
    n, k = w.shape
    x = w.astype(np.float32).reshape(n, k // 32, 32)
    amax = np.maximum(np.abs(x).max(-1), np.finfo(np.float32).tiny)
    ratio = (amax / np.float32(448.0)).astype(np.float32)
    m, ex = np.frexp(ratio.astype(np.float64))
    s = np.clip(np.where(m == 0.5, ex - 1, ex) + 127, 0, 254).astype(np.int64)
    xs = (x.astype(np.float64) * np.ldexp(1.0, 127 - s)[..., None]).astype(np.float32)
    return mxfp8.e4m3fn_encode(np.clip(xs, -448, 448)).reshape(n, k), s.astype(np.uint8)


def int4_quantize(w):
    """scale = fp16(amax / 7) per 32 inputs, code = clip(rint(x / scale), -8, 7) + 8; code 8 where the scale is 0."""
    n, k = w.shape
    x = w.astype(np.float32).reshape(n, k // 32, 32)
    s = (np.abs(x).max(-1) / np.float32(7.0)).astype(np.float16)
    sf = s.astype(np.float32)[..., None]
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(sf == 0, 8, np.clip(np.rint(x / sf), -8, 7) + 8).astype(np.uint8).reshape(n, k)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8), s


def np_of(t):
    """A torch tensor as NumPy, BF16 as its uint16 bits."""
    return t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.numpy()


def bf16(w):
    return torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


class Expect:
    """The tensors one rank's image should hold, layer by layer, from the reference decode."""

    def __init__(self, X, release, r, trellis):
        self.X, self.release, self.r, self.trellis = X, release, r, trellis

    def layer(self, l, qo=False):
        X, r = self.X, self.r
        f = X.SafeFile(os.path.join(self.release, 'dense', f'exlr8-dense-{l:03d}.safetensors'))
        rec = {m[:-7]: f.get(m) for m in f.names() if m.endswith('.record')}
        raw = lambda m, s: np_of(f.get(f'{m}.{s}'))
        dec = lambda m: X.decode_dense(rec[m], {s: f.get(f'{m}.{s}') for s in X.DATA[int(rec[m][0])]}).numpy()
        P, mtp = f'L{l:03d}', l == plan.L_MTP
        h0, nh = plan.heads(l)[r]
        T = {}

        def mx_copy(name, m, gv=True, copy=True):
            c, s = raw(m, 'fp8'), raw(m, 'e8m0')
            if copy:
                T[f'{P}.{name}.mx.codes'], T[f'{P}.{name}.mx.scales'] = c, s
            if gv:
                T[f'{P}.{name}.mx.gv'] = g6ref.mxfp8_record(c, s)

        def mx_round(name, w, gv):
            c, s = mx_quantize(w)
            T[f'{P}.{name}.mx.codes'], T[f'{P}.{name}.mx.scales'] = c, s
            if gv:
                T[f'{P}.{name}.mx.gv'] = g6ref.mxfp8_record(c, s)

        def with_gv(name, w):
            b = bf16(w)
            T[f'{P}.{name}'], T[f'{P}.{name}.gv'] = b, g6ref.bf16_record(b)

        vec = lambda name, m: T.__setitem__(f'{P}.{name}', np_of(f.get(m)))

        def entries(name, m, ch):
            """the dense trellis kernels' form: the rank's entries as stored, suh / svh sliced on the sharded side"""
            side = int(rec[m][4])
            T[f'{P}.{name}.trellis'] = raw(m, 'trellis')[ch.start // 128:ch.stop // 128]
            T[f'{P}.{name}.suh'] = raw(m, 'suh')[ch] if side == X.IN else raw(m, 'suh')
            T[f'{P}.{name}.svh'] = raw(m, 'svh') if side == X.IN else raw(m, 'svh')[ch]
        tr = lambda m: self.trellis and int(rec[m][0]) == X.TRELLIS
        vec('attn_norm', 'input_layernorm.weight')
        mx_copy('q_a', 'self_attn.q_a_proj')
        vec('q_a_norm', 'self_attn.q_a_layernorm.weight')
        mx_copy('kv_a', 'self_attn.kv_a_proj_with_mqa')
        vec('kv_a_norm', 'self_attn.kv_a_layernorm.weight')
        qb = dec('self_attn.q_b_proj')
        heads = slice(h0 * 256, (h0 + nh) * 256)
        if tr('self_attn.q_b_proj'):
            entries('q_b', 'self_attn.q_b_proj', heads)
        else:
            mx_round('q_b', qb[heads], True)
            if int(rec['self_attn.q_b_proj'][0]) == X.INT4:
                T[f'{P}.q_b.gptq'] = g6ref.rt_record(g6ref.RT('int4'), raw('self_attn.q_b_proj', 'q4')[heads],
                                                     raw('self_attn.q_b_proj', 's4')[heads])
        kvb = bf16(dec('self_attn.kv_b_proj')).reshape(64, 448, 512)
        T[f'{P}.kv_b.uk'], T[f'{P}.kv_b.uv'] = kvb[h0:h0 + nh, :192], kvb[h0:h0 + nh, 192:]
        op = dec('self_attn.o_proj')[:, heads]
        if tr('self_attn.o_proj'):
            entries('o_proj', 'self_attn.o_proj', heads)
        elif not mtp:
            if int(rec['self_attn.o_proj'][0]) == X.INT4:   # the release's own codes, side-2 storage transposed back
                q4 = raw('self_attn.o_proj', 'q4')[2 * h0:2 * (h0 + nh)].transpose(1, 0, 2).reshape(H, -1)
                s4 = raw('self_attn.o_proj', 's4')[2 * h0:2 * (h0 + nh)].transpose(1, 0, 2).reshape(H, -1)
            else:
                q4, s4 = int4_quantize(op)
            T[f'{P}.o_proj.gptq'] = g6ref.rt_record(g6ref.RT('int4'), q4, s4)
        if not tr('self_attn.o_proj'):
            mx_round('o_proj', op, mtp)
        if 'self_attn.indexer.wq_b' in rec:
            mx_copy('idx.wq_b', 'self_attn.indexer.wq_b')
            with_gv('idx.wk', dec('self_attn.indexer.wk'))
            vec('idx.k_norm.w', 'self_attn.indexer.k_norm.weight')
            vec('idx.k_norm.b', 'self_attn.indexer.k_norm.bias')
            with_gv('idx.weights_proj', dec('self_attn.indexer.weights_proj'))
        vec('mlp_norm', 'post_attention_layernorm.weight')
        if plan.is_moe(l):
            with_gv('router.w', dec('mlp.gate'))
            vec('router.bias', 'mlp.gate.e_score_correction_bias')
        else:
            g0, ng = plan.dense_groups(l)[r]
            a, b = g0 * 32, (g0 + ng) * 32
        if not plan.is_moe(l) and all(tr(f'mlp.{n}_proj') for n in ('gate', 'up', 'down')):
            for n in ('gate', 'up', 'down'):
                entries(f'dense.{n}', f'mlp.{n}_proj', slice(a, b))
        elif not plan.is_moe(l):
            g, u, d = dec('mlp.gate_proj')[a:b], dec('mlp.up_proj')[a:b], dec('mlp.down_proj')[:, a:b]
            f1, f2 = g6ref.rt_expert_record(g6ref.RT('int4'), *int4_quantize(g), *int4_quantize(u), *int4_quantize(d))
            T[f'{P}.dense.gptq.fc1'], T[f'{P}.dense.gptq.fc2'] = f1, f2
            for n, w in (('gate', g), ('up', u), ('down', d)):
                mx_round(f'dense.{n}', w, False)
        if qo and not mtp:
            T[f'{P}.qo.q_b.mx.codes'], T[f'{P}.qo.q_b.mx.scales'] = mx_quantize(qb)
            T[f'{P}.qo.o_proj.mx.codes'], T[f'{P}.qo.o_proj.mx.scales'] = mx_quantize(dec('self_attn.o_proj'))
            T[f'{P}.qo.kv_b.uk'], T[f'{P}.qo.kv_b.uv'] = kvb[:, :192], kvb[:, 192:]
        if mtp:
            mx_copy('eh', 'eh_proj', copy=False)
            vec('enorm', 'enorm.weight')
            vec('hnorm', 'hnorm.weight')
            vec('shared_head_norm', 'shared_head.norm.weight')
        return T

    def globals(self):
        X = self.X
        f = X.SafeFile(os.path.join(self.release, 'dense', 'exlr8-vocab.safetensors'))
        dec = lambda m: X.decode_dense(f.get(f'{m}.record'), {s: f.get(f'{m}.{s}') for s in ('fp8', 'scale')}).numpy()
        a, n = plan.vocab_rows(self.r)
        c, s = mx_quantize(dec('lm_head')[a:a + n])
        return {'G.embed': bf16(dec('embed_tokens')), 'G.final_norm': np_of(f.get('norm.weight')),
                'G.lm_head.mx.gv': g6ref.mxfp8_record(c, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--widths', required=True)
    ap.add_argument('--ref', required=True, help='a kindlingai/exlr8 checkout')
    ap.add_argument('--dump', required=True, help='the g6plandump binary')
    ap.add_argument('--tp', type=int, required=True)
    ap.add_argument('--ranks', required=True)
    ap.add_argument('--layers', required=True)
    ap.add_argument('--qo', action='store_true', help='plan and check the G6_PF_OWN=3 full-head tensors too')
    ap.add_argument('--rounded', action='store_true', help='plan and check the rounded form of the trellis matrices')
    ap.add_argument('--work', default='/tmp/release-check')
    a = ap.parse_args()
    sys.path.insert(0, a.ref)
    from loader import exlr8 as X
    plan.set_nr(a.tp)
    os.makedirs(a.work, exist_ok=True)
    layers = [int(x) for x in a.layers.split(',')]
    fails, n_t, n_b = [], 0, 0
    for r in [int(x) for x in a.ranks.split(',')]:
        out = os.path.join(a.work, f'tp{a.tp}.r{r}.safetensors')
        rel = a.release
        subprocess.run([a.dump, rel, os.path.join(rel, 'experts'), os.path.join(rel, 'index-slices'), a.widths, str(a.tp),
                        str(r), out, '--layers', a.layers, '--globals', '--experts'] + (['--qo'] if a.qo else [])
                       + (['--rounded'] if a.rounded else []),
                       check=True, stderr=subprocess.DEVNULL)
        got = X.SafeFile(out)
        e = Expect(X, a.release, r, not a.rounded)
        want = e.globals()
        for l in layers:
            want.update(e.layer(l, a.qo))
        names = set(got.names())
        experts = {n for n in names if n[5:].startswith('tr3.')}
        for l in layers:
            if plan.is_moe(l) and not {f'L{l:03d}.tr3.{t}' for t in ('fc1', 'fc2', 'table', 'kmix', 'v2')} <= experts:
                fails.append((r, f'L{l:03d}', 'expert tensors missing'))
        if names - experts != set(want):
            fails.append((r, 'names', sorted((names - experts) ^ set(want))))
            print(f'rank {r}: names differ: {sorted((names - experts) ^ set(want))}', flush=True)
        for k in sorted(set(want) & names):
            w = np.ascontiguousarray(want[k])
            g = np_of(got.get(k))
            if g.tobytes() != w.tobytes() or g.size != w.size:
                fails.append((r, k))
                print(f'rank {r}: {k} differs ({g.shape} {g.dtype} vs {w.shape} {w.dtype})', flush=True)
            else:
                n_t += 1
                n_b += w.nbytes
        os.unlink(out)
        print(f'rank {r}: {len(want)} tensors checked', flush=True)
    print(f'{n_t} tensors, {n_b / 1e9:.2f} GB equal byte for byte' + ('' if not fails else f'; {len(fails)} FAILED'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
