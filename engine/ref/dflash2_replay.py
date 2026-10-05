"""Replay a G6_DFL_DUMP of one request through the NumPy reference drafter (dflash2_ref.py).

python3 engine/ref/dflash2_replay.py DUMP --src DRAFTER --glm GLM_DIR [--quant mxfp8|bf16] [--rounds N] [--out FILE]

For every draft step of the dump: the reference's drafts (argmax and selector walk) from the same context features and
bonus token, against the engine's drafts and against the committed tokens (acceptance = leading drafts equal to them).
The context K/V of a position is computed once; the block of every round runs layer by layer over all rounds together,
so each weight is read once.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash2_ref as R  # noqa: E402

B, NT, HID, NH, NKV, HD = 8, 6, 6144, 64, 8, 128


def load_dump(path):
    raw = open(path, 'rb').read()
    off, steps, feats, tokens = 0, [], {}, None
    while off < len(raw):
        h = np.frombuffer(raw, np.int32, 12, off)
        off += 48
        if h[0] == 2:
            tokens = np.frombuffer(raw, np.int32, h[1], off).copy()
            off += 4 * h[1]
            continue
        kind, pos0, rows, q0, tok0 = (int(x) for x in h[:5])
        f = np.frombuffer(raw, np.float16, rows * NT * HID, off).reshape(rows, NT * HID)
        off += f.nbytes
        for r in range(rows):
            feats[pos0 + r] = f[r]
        if kind == 1:
            steps.append(dict(q0=q0, tok0=tok0, drafts=[int(x) for x in h[5:12]]))
    return steps, feats, tokens


def accept(d, actual):
    a = 0
    while a < len(d) and a < len(actual) and d[a] == actual[a]:
        a += 1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--src', required=True)
    ap.add_argument('--glm', required=True)
    ap.add_argument('--quant', default='mxfp8')
    ap.add_argument('--rounds', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--feats', default=None, help='take the context features from this dump instead (e.g. the same text '
                                                  'sent again as a prompt, so its features come from the prefill path)')
    a = ap.parse_args()
    steps, feats, tokens = load_dump(a.dump)
    if a.feats:
        _, f2, _ = load_dump(a.feats)
        missing = [p for p in feats if p not in f2]
        print(f'features from {a.feats}: {len(f2)} positions, {len(missing)} of the original missing (kept)')
        feats = {p: f2.get(p, v) for p, v in feats.items()}
    if a.rounds:
        steps = steps[:a.rounds]
    D = R.Drafter(a.src, None if a.quant == 'bf16' else a.quant)
    pos = np.array(sorted(feats))
    F = np.stack([feats[p] for p in pos]).astype(np.float32) * 64          # the engine stores residual / 64
    if a.noise:                       # sensitivity: Gaussian noise, std = noise x the RMS of each row's tap layer
        Fl = F.reshape(len(F), NT, HID)
        Fl += np.random.default_rng(0).standard_normal(Fl.shape).astype(np.float32) * a.noise * np.sqrt((Fl ** 2).mean(-1, keepdims=True))
    kv, _ = D.context_kv(F, pos)
    emb_st = R.ST(os.path.join(a.glm, 'model-embed.safetensors'))
    mask = D.d['mask_token_id']
    erow = lambda t: R.bf(emb_st.get('model.embed_tokens.weight', rows=(t, t + 1))[0])
    me = erow(mask)
    n = len(steps)
    h = np.stack([np.stack([erow(s['tok0'])] + [me] * 7) for s in steps])  # [n, 8, 6144]
    qpos = np.array([np.arange(s['q0'], s['q0'] + B) for s in steps])
    NG = HID // D.G

    def prep(x, j, which):
        """conv prepare of every round at once: (conv'd rows [n, 8, 6144], (delta side 1, base side 1))"""
        p = f'layers.{j}.{which}.'
        dl = D.lin(x.reshape(-1, HID), p + 'kernel_projection.weight').reshape(n, B, 2, 2, NG)
        base = D.w(p + 'base_kernel')
        return np.stack([D.conv(x[i], dl[i, :, 0], base[0]) for i in range(n)]), (dl[:, :, 1], base[1])

    def fin(y, f):
        return np.stack([D.conv(y[i], f[0][i], f[1]) for i in range(n)])

    for j in range(D.L):
        p = f'layers.{j}.self_attn.'
        x = R.rms(h, D.w(f'layers.{j}.input_layernorm.weight'), D.eps)
        xc, f1 = prep(x, j, 'attention_conv')
        X = xc.reshape(-1, HID)
        q = R.rms(D.lin(X, p + 'q_proj.weight').reshape(n, B, NH, HD), D.w(p + 'q_norm.weight'), D.eps)
        k = R.rms(D.lin(X, p + 'k_proj.weight').reshape(n, B, NKV, HD), D.w(p + 'k_norm.weight'), D.eps)
        v = D.lin(X, p + 'v_proj.weight').reshape(n, B, NKV, HD)
        out = np.zeros((n, B, NH, HD), np.float32)
        for i, s in enumerate(steps):
            qi = R.rope_neox(q[i].transpose(1, 0, 2), qpos[i], D.theta).transpose(1, 0, 2)
            ki = R.rope_neox(k[i].transpose(1, 0, 2), qpos[i], D.theta).transpose(1, 0, 2)
            keep = (pos < s['q0']) & (pos > s['q0'] - D.win)
            kp = np.concatenate([pos[keep], qpos[i]])
            K = np.concatenate([kv[j][0][keep], ki])
            V = np.concatenate([kv[j][1][keep], v[i]])
            for r in range(B):
                vis = (qpos[i][r] - kp < D.win) & (kp - qpos[i][r] < D.win)
                Kv, Vv = K[vis], V[vis]                                     # [n_vis, 8, 128]
                sc = np.einsum('kgd,ghd->ghk', Kv, qi[r].reshape(NKV, NH // NKV, HD)) / np.sqrt(HD)
                e = np.exp(sc - sc.max(-1, keepdims=True))
                out[i, r] = (np.einsum('ghk,kgd->ghd', e / e.sum(-1, keepdims=True), Vv)).reshape(NH, HD)
        att = D.lin(out.reshape(n * B, NH * HD), p + 'o_proj.weight').reshape(n, B, HID)
        h = h + fin(att, f1)
        x2 = R.rms(h, D.w(f'layers.{j}.post_attention_layernorm.weight'), D.eps)
        xc2, f2 = prep(x2, j, 'mlp_conv')
        m = D.mlp(xc2.reshape(-1, HID), j).reshape(n, B, HID)
        h = h + fin(m, f2)
        print(f'layer {j} done', flush=True)
    x = R.rms(h, D.w('norm.weight'), D.eps)
    head = R.ST(os.path.join(a.glm, 'model-head.safetensors'))
    cand, logit = D.top(x[:, 1:].reshape(-1, HID), head)
    cand, logit = cand.reshape(n, B - 1, -1), logit.reshape(n, B - 1, -1)
    rows = []
    for i, s in enumerate(steps):
        walk, _ = D.select(x[i, 1:], cand[i], logit[i], s['tok0'])
        actual = tokens[s['q0'] + 1:s['q0'] + B] if tokens is not None else []
        rows.append(dict(q0=s['q0'], eng=s['drafts'], ref_walk=walk.tolist(), ref_argmax=cand[i][:, 0].tolist(),
                         actual=[int(t) for t in actual], acc_eng=accept(s['drafts'], actual),
                         acc_walk=accept(walk.tolist(), actual), acc_argmax=accept(cand[i][:, 0].tolist(), actual),
                         same_walk=int(sum(x_ == y_ for x_, y_ in zip(s['drafts'], walk.tolist())))))
    tot = lambda k: sum(r[k] for r in rows) / max(len(rows), 1) + 1
    print(json.dumps(dict(rounds=len(rows), tokens_per_step_engine=tot('acc_eng'), tokens_per_step_ref_walk=tot('acc_walk'),
                          tokens_per_step_ref_argmax=tot('acc_argmax'),
                          engine_equals_ref_walk=sum(r['eng'] == r['ref_walk'] for r in rows))))
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=0)


if __name__ == '__main__':
    main()
