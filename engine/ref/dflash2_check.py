"""Drafter step check: engine/runtime/dfltest against the NumPy reference (dflash2_ref.py).

python3 engine/ref/dflash2_check.py gen DIR --glm GLM_DIR [--n 20] [--p0 1000] [--bonus 13] [--seed 1] [--scale 4]
    case.bin: n context rows of random target features (normal, std --scale) at positions p0.., the bonus token, and the
    embedding rows of [bonus, mask x 7] from the GLM checkpoint
python3 engine/ref/dflash2_check.py ref DIR --src DRAFTER --glm GLM_DIR [--quant mxfp8|bf16]
    ref.<quant>.npz: the reference's final-norm rows, context K/V of the last chunk, top-16 and the walk
python3 engine/ref/dflash2_check.py cmp DIR [--quant mxfp8] [--sel 1]
    compares out.<sel>.bin (dfltest) with ref.<quant>.npz
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash2_ref as R  # noqa: E402

B, NT, HID, NKV, HD = 8, 6, 6144, 8, 128


def gen(a):
    rng = np.random.default_rng(a.seed)
    feat = (rng.standard_normal((a.n, NT * HID)) * a.scale).astype(np.float32)
    cfg = json.load(open(os.path.join(a.src, 'config.json'))) if a.src else {'dflash_config': {'mask_token_id': 154856}}
    mask = cfg['dflash_config']['mask_token_id']
    st = R.ST(os.path.join(a.glm, 'model-embed.safetensors'))
    emb = np.stack([R.bf(st.get('model.embed_tokens.weight', rows=(t, t + 1))[0]) for t in [a.bonus] + [mask] * 7])
    with open(os.path.join(a.dir, 'case.bin'), 'wb') as f:
        f.write(struct.pack('<4i', a.n, a.p0, a.bonus, a.p0))
        f.write(feat.tobytes())
        f.write(emb.astype(np.float32).tobytes())
    print(f'case: {a.n} rows at {a.p0}, bonus {a.bonus}, mask {mask}')


def load_case(d):
    raw = open(os.path.join(d, 'case.bin'), 'rb').read()
    n, p0, bonus, lo = struct.unpack('<4i', raw[:16])
    feat = np.frombuffer(raw, np.float32, n * NT * HID, 16).reshape(n, NT * HID)
    emb = np.frombuffer(raw, np.float32, B * HID, 16 + feat.nbytes).reshape(B, HID)
    return n, p0, bonus, lo, feat, emb


def ref(a):
    n, p0, bonus, lo, feat, emb = load_case(a.dir)
    D = R.Drafter(a.src, None if a.quant == 'bf16' else a.quant)
    # the engine stores the features as fp16 of residual / 64 (dflash.cuh FEAT_SCALE); fc is linear
    f16 = (feat / 64).astype(np.float16).astype(np.float32) * 64
    pos = np.arange(p0, p0 + n)
    kv, ctx = D.context_kv(f16, pos)
    caches = [(pos, k, v) for k, v in kv]
    q0 = p0 + n
    x = D.block(emb, np.arange(q0, q0 + B), caches)
    head = R.ST(os.path.join(a.glm, 'model-head.safetensors'))
    cand, logit = D.top(x[1:], head)
    walk, score = D.select(x[1:], cand, logit, bonus)
    last = n - (n - 1) // B * B
    np.savez(os.path.join(a.dir, f'ref.{a.quant}.npz'), x=x, cand=cand, logit=logit, walk=walk, score=score,
             k=np.stack([k[-last:] for k, _ in kv]), v=np.stack([v[-last:] for _, v in kv]))
    print('ref argmax', cand[:, 0].tolist(), 'walk', walk.tolist())


def cmp(a):
    n, p0, bonus, lo, feat, emb = load_case(a.dir)
    r = np.load(os.path.join(a.dir, f'ref.{a.quant}.npz'))
    raw = open(os.path.join(a.dir, f'out.{a.sel}.bin'), 'rb').read()
    tok = np.frombuffer(raw, np.int32, B)
    x = np.frombuffer(raw, np.float32, B * HID, 4 * B).reshape(B, HID)
    last = r['k'].shape[1]
    kv = np.frombuffer(raw, np.float32, 6 * 2 * NKV * last * HD,
                       4 * B + x.nbytes).reshape(6, 2, NKV, last, HD)
    rel = lambda e, g: float(np.linalg.norm(e - g) / max(np.linalg.norm(g), 1e-30))
    cos = lambda e, g: float((e * g).sum() / max(np.linalg.norm(e) * np.linalg.norm(g), 1e-30))
    out = dict(x_rel=rel(x, r['x']), x_cos_min=min(cos(x[i], r['x'][i]) for i in range(B)))
    out['k_rel'] = rel(kv[:, 0].transpose(0, 2, 1, 3), r['k'])
    out['v_rel'] = rel(kv[:, 1].transpose(0, 2, 1, 3), r['v'])
    want = r['walk'] if a.sel == 1 else r['cand'][:, 0]
    out['drafts'] = tok[1:].tolist()
    out['ref'] = want.tolist()
    out['match'] = int((tok[1:] == want).sum())
    out['in_ref_top16'] = int(sum(t in c for t, c in zip(tok[1:], r['cand'])))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['gen', 'ref', 'cmp'])
    ap.add_argument('dir')
    ap.add_argument('--glm')
    ap.add_argument('--src')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--p0', type=int, default=1000)
    ap.add_argument('--bonus', type=int, default=13)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--scale', type=float, default=4.0)
    ap.add_argument('--quant', default='mxfp8')
    ap.add_argument('--sel', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    dict(gen=gen, ref=ref, cmp=cmp)[a.cmd](a)


if __name__ == '__main__':
    main()
