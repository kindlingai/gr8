"""m4sim_flash (decode only) against flash_ref.FlashRef on the same weights: the W7 first-token check.

    G6_CP=0 G6_RESID_DUMP=resid.bin G6_LOGIT_DUMP=logits.bin m4sim_flash --v2 REL/experts ... --v2-dense REL \
        --v2-partial --layers 0,1,2,3,4,5 --prompt @prompt.ids --gen 8 --temp 0 > steps.txt
    python3 engine/ref/flash/flash_forward.py --release REL --loader EXLR8_LOADER --config config.json --K 4 \
        --layers 0,1,2,3,4,5 --ids prompt.ids --m4 steps.txt --resid resid.bin --logits logits.bin

The weights: the release's dense/ files (the lossless BF16 export) and its routed + shared experts decoded at width
--K by the exlr8 reference loader (the width manifest m4sim ran must be uniform K). The tokens: the prompt ids m4sim
read, then its generated tokens but the last (never fed).

Checks, all fp32 on CPU:
  sublayer   each attention and MLP block (with its mHC) runs on gr8's own four input streams (the residual dump),
             over every position from a fresh state, and its output streams are compared with gr8's next dump slot:
             rel-L2 of the streams (gate) and of the block's update out - in (diagnostic; stricter)
  head       final mean -> norm -> lm_head on gr8's last streams vs gr8's logits (the head alone)
  chain      the whole cut model on the token ids vs gr8's logits: rel-L2, top-1 agreement, KL(ref || gr8), and gr8's
             greedy tokens vs the reference's argmax at the same positions
  MTP        (--mtp-in / --mtp-logits, the first draft pass per step) flash_ref.mtp on gr8's own final-normed hidden
             and fed tokens, positions 0.. in order: draft logits top-1 and KL; gr8's hidden vs the chain's (reported)
Gates (W7-PLAN.md, ENGINE-PLAN.md section 4): sublayer streams <= 2e-2, top-1 >= 99%, mean KL <= 1e-3, greedy equal;
the same top-1 and KL for the MTP draft logits.
Long prompts: both dumps are read a slot / a row range at a time (no G6_LOGIT_DUMP_FROM: it also cuts the residual
dump, and the sublayer checks need every position). The report counts, per DSA layer of the reference chain, the
causal keys the indexer drops (none before position 2,051: index_topk 2048 keys = 512 pools of 4).
Routed experts are decoded when a token picks one, and each layer's weights are dropped after its checks, so the CPU
run fits an 8 GiB memory cap. Many layers: --device cuda --cache-experts on a GPU that is not running the engine.
"""
import argparse
import copy
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_ref import Cfg, FlashRef, SeqState  # noqa: E402

GATE_STREAM, GATE_TOP1, GATE_KL = 2e-2, 0.99, 1e-3


def read_ints(path):
    return [int(x) for x in re.split(r'[,\s]+', open(path).read().strip()) if x]


class ResidDump:
    """G6_RESID_DUMP: per step int32 s, M, slots, width, then slots x M x width fp32. Read lazily: dump[k] is slot k as
    [T, hc, H] (a 9k-token dump of 8 layers is 10 GB)."""

    def __init__(self, path, hc, H):
        self.path, self.hc, self.H, self.steps = path, hc, H, []
        with open(path, 'rb') as f:
            end = f.seek(0, 2)
            f.seek(0)
            while f.tell() < end:
                h = f.read(16)
                if len(h) != 16:
                    raise ValueError('truncated residual header')
                s, M, slots, width = (int(v) for v in np.frombuffer(h, '<i4'))
                if width != hc * H:
                    raise ValueError(f'step {s}: width {width}, expected {hc} x {H} (a Flash four-stream dump)')
                self.steps.append((f.tell(), M, slots))
                if f.seek(4 * slots * M * width, 1) > end:
                    raise ValueError('truncated residual payload')
        if not self.steps:
            raise ValueError('empty residual dump')
        if len({st[2] for st in self.steps}) != 1:
            raise ValueError('residual dump: the slot count changes between steps')
        self.shape = (self.steps[0][2], sum(st[1] for st in self.steps), hc, H)

    def __getitem__(self, k):
        out = np.empty(self.shape[1:], np.float32)
        w, t = self.hc * self.H, 0
        with open(self.path, 'rb') as f:
            for off, M, _ in self.steps:
                f.seek(off + 4 * k * M * w)
                out[t:t + M] = np.fromfile(f, '<f4', M * w).reshape(M, self.hc, self.H)
                t += M
        return out


class LogitDump:
    """G6_LOGIT_DUMP: per step int32 s, M, then per rank int32 vn and M x vn fp32 (vocab rows in rank order). Read
    lazily: dump[a:b] is rows a..b-1 as [b - a, V] (a 9k-token dump is 5.7 GB)."""

    def __init__(self, path, vocab):
        self.path, self.steps = path, []              # (first row, M, [(offset, first vocab row, vn)])
        with open(path, 'rb') as f:
            end = f.seek(0, 2)
            f.seek(0)
            t = 0
            while f.tell() < end:
                s, M = (int(v) for v in np.frombuffer(f.read(8), '<i4'))
                parts, n = [], 0
                while n < vocab:
                    vn = int(np.frombuffer(f.read(4), '<i4')[0])
                    parts.append((f.tell(), n, vn))
                    n += vn
                    if f.seek(4 * M * vn, 1) > end:
                        raise ValueError('truncated logit dump')
                if n != vocab:
                    raise ValueError(f'step {s}: {n} vocab rows, expected {vocab}')
                self.steps.append((t, M, parts))
                t += M
        if not self.steps:
            raise ValueError('empty logit dump')
        self.shape = (t, vocab)

    def __getitem__(self, sl):
        a, b, _ = sl.indices(self.shape[0])
        out = np.empty((max(b - a, 0), self.shape[1]), np.float32)
        with open(self.path, 'rb') as f:
            for t, M, parts in self.steps:
                lo, hi = max(a, t), min(b, t + M)
                if lo >= hi:
                    continue
                for off, v0, vn in parts:
                    f.seek(off + 4 * (lo - t) * vn)
                    out[lo - a:hi - a, v0:v0 + vn] = np.fromfile(f, '<f4', (hi - lo) * vn).reshape(hi - lo, vn)
        return out


def read_draft_in(path, H):
    """G6_DRAFT_IN_DUMP: per first draft pass int32 s, M, pos[M], tok[M], then fp32 hidden[M][H] -> (pos, tok, hidden)"""
    pos, tok, hid = [], [], []
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if not h:
                break
            s, M = np.frombuffer(h, '<i4')
            pos.append(np.frombuffer(f.read(4 * M), '<i4'))
            tok.append(np.frombuffer(f.read(4 * M), '<i4'))
            x = np.frombuffer(f.read(4 * M * H), '<f4')
            if x.size != M * H:
                raise ValueError('truncated draft input payload')
            hid.append(x.reshape(M, H))
    if not pos:
        raise ValueError('empty draft input dump')
    return np.concatenate(pos), np.concatenate(tok), np.concatenate(hid)


def generated(m4_out):
    """m4sim's 'generated t0 t1 ...' line (one sequence)"""
    lines = [ln.split()[1:] for ln in open(m4_out) if ln.startswith('generated')]
    if len(lines) != 1:
        raise ValueError(f'{m4_out}: {len(lines)} "generated" lines, expected one sequence')
    return [int(x) for x in lines[0]]


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


class DeviceWeights:
    """Weights on a device, layer by layer: each tensor is moved when read; with cache, a layer's decoded experts are
    kept until evict(L) (fp32, about 29 GB for a K4 layer: GPU only). evict(L) also drops the sources' copies of L."""

    def __init__(self, base, dev, cache=False):
        self.base, self.dev, self.cache, self.experts = base, dev, cache, {}

    def has(self, name):
        return self.base.has(name)

    def get(self, name):
        return self.base.get(name).to(self.dev)

    def expert(self, L, e):
        if not self.cache:
            return tuple(t.to(self.dev) for t in self.base.expert(L, e))
        lay = self.experts.setdefault(L, {})
        if e not in lay:
            lay[e] = tuple(t.to(self.dev, torch.float32) for t in self.base.expert(L, e))
        return lay[e]

    def evict(self, L):
        self.experts.pop(L, None)
        for s in getattr(self.base, 'sources', [self.base]):
            if isinstance(getattr(s, 'layers', None), dict):
                s.layers.pop(L, None)


def logit_stats(ref, streams, gl, dev, chunk=512):
    """the head on streams [T, hc, H] vs gr8's logits gl (LogitDump or array), chunk rows at a time: rel-L2, top-1,
    KL rows, the reference's argmax"""
    d2 = r2 = 0.0
    top1, kl, arg, miss = 0, [], [], []
    for i in range(0, streams.shape[0], chunk):
        lg = ref.head(streams[i:i + chunk])[1]
        g = torch.from_numpy(np.ascontiguousarray(gl[i:i + chunk])).to(dev)
        d2 += float(((g.double() - lg.double()) ** 2).sum())
        r2 += float((lg.double() ** 2).sum())
        a = lg.argmax(-1)
        top1 += int((g.argmax(-1) == a).sum())
        for r in (g.argmax(-1) != a).nonzero().flatten().tolist():  # top-1 disagreements: both top-2 gaps
            gaps = [float(t[0] - t[1]) for t in (lg[r].float().topk(2).values, g[r].float().topk(2).values)]
            miss.append({'row': i + r, 'ref_gap': gaps[0], 'gr8_gap': gaps[1]})
        kl.append(kl_rows(lg, g).cpu())
        arg.append(a.cpu())
    return {'rel': (d2 / max(r2, 1e-60)) ** 0.5, 'top1': top1, 'misses': miss}, torch.cat(kl), torch.cat(arg)


def idx_counter(out):
    """FlashRef rec hook: per DSA layer, the causal keys the indexer drops (a prompt from position 0)"""
    def rec(name, x):
        if name.endswith('.idx_mask'):
            T, N = x.shape
            d = torch.arange(N - T + 1, N + 1, device=x.device) - x.sum(1)
            out[name.split('.')[0]] = {'queries_with_drops': int((d > 0).sum()), 'keys_dropped': int(d.sum())}
    return rec


def bf16_round(name, x):
    return x.to(torch.bfloat16).to(x.dtype)


def compare(ref, layers, ids, gen, g6, g6l, dev='cpu', mtp=None, floor=False):
    """ref: FlashRef (fp32); g6 [2 len(layers) + 1, T, hc, H] gr8 streams; g6l [T, V] gr8 logits (dumps or arrays,
    read a slot / a row range at a time). Returns a report.
    One pass over the layers: layer L's two sublayer checks and the reference chain's layer L, then L is evicted.
    mtp: (pos, tok, hidden, draft logits) of gr8's first draft passes: flash_ref.mtp on gr8's own hidden and tokens.
    floor: also run a second reference chain rounded to BF16 at the engine's rounding points (block inputs, block
    outputs, DSA latent row; FlashRef.z) and report it against the fp32 chain: the rounding-only noise floor."""
    T = len(ids)
    if g6.shape[0] != 2 * len(layers) + 1 or g6.shape[1] != T or g6l.shape[0] != T:
        raise ValueError(f'dump shapes {g6.shape} / {g6l.shape} for {len(layers)} layers and {T} tokens '
                         '(both dumps must start at step 0: no G6_LOGIT_DUMP_FROM)')
    g = lambda k: torch.from_numpy(np.ascontiguousarray(g6[k])).to(dev)
    evict = getattr(ref.w, 'evict', lambda L: None)
    rep = {'tokens': T, 'layers': list(layers), 'sublayers': [], 'indexer': {}}
    with torch.no_grad():
        chain = ref.emb(ids).unsqueeze(1).expand(-1, ref.c.hc, -1).contiguous()
        rep['embed'] = rel(g(0), chain)
        st = SeqState()
        if floor:
            fref = copy.copy(ref)
            fref.qz, fref.rec, fst, fchain = bf16_round, None, SeqState(), chain
        for i, L in enumerate(layers):
            for k, what in ((2 * i, 'attention'), (2 * i + 1, 'MLP')):
                x, nxt = g(k), g(k + 1)
                y = ref.attn_block(x, L, SeqState()) if k % 2 == 0 else ref.mlp_block(x, L)
                rep['sublayers'].append({'layer': L, 'block': what, 'stream': rel(nxt, y), 'update': rel(nxt - x, y - x)})
                del x, nxt, y
            ref.rec = idx_counter(rep['indexer'])
            chain = ref.layer(chain, L, st)
            ref.rec = None
            rep.setdefault('drift', []).append({'layer': L, 'rel': rel(g(2 * i + 2), chain)})
            if floor:
                fchain = fref.layer(fchain, L, fst)
                rep.setdefault('floor_drift', []).append({'layer': L, 'rel': rel(fchain, chain)})
            evict(L)
        rep['head'] = logit_stats(ref, g(2 * len(layers)), g6l, dev)[0]
        rep['chain'], kl, top = logit_stats(ref, chain, g6l, dev)
        if floor:                                           # the rounded chain's logits play gr8's part
            flog = torch.cat([ref.head(fchain[i:i + 512])[1].float().cpu() for i in range(0, T, 512)]).numpy()
            rep['floor'], fkl, _ = logit_stats(ref, chain, flog, dev)
            rep['floor'].update(kl_mean=float(fkl.mean()), kl_max=float(fkl.max()))
            del flog, fchain
        if mtp is not None:
            rep['mtp'] = compare_mtp(ref, ids, gen, ref.final_hidden(chain), mtp, dev)
    P = T - len(gen) + 1                                    # prompt length: gen[j] was sampled at row P - 1 + j
    rep['chain'].update(kl_mean=float(kl.mean()), kl_max=float(kl.max()), greedy_gr8=gen,
                        greedy_ref=top[P - 1:].tolist())
    c = rep['chain']
    rep['gates'] = {
        'sublayer streams <= 2e-2': all(s['stream'] <= GATE_STREAM for s in rep['sublayers']),
        'top-1 >= 99%': c['top1'] >= GATE_TOP1 * T,
        'mean KL <= 1e-3': c['kl_mean'] <= GATE_KL,
        'greedy equal': c['greedy_gr8'] == c['greedy_ref'],
    }
    if 'mtp' in rep:
        m = rep['mtp']
        rep['gates']['MTP top-1 >= 99%'] = m['top1'] >= GATE_TOP1 * m['rows']
        rep['gates']['MTP mean KL <= 1e-3'] = m['kl_mean'] <= GATE_KL
    rep['pass'] = all(rep['gates'].values())
    return rep


def kl_rows(p, q):
    """KL(p || q) per row, nats, from logits"""
    lp, lq = torch.log_softmax(p.double(), -1), torch.log_softmax(q.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def compare_mtp(ref, ids, gen, hidden, mtp, dev):
    """The MTP layer on gr8's own final-normed hidden and fed tokens, rows in position order from 0, fresh state."""
    pos, tok, hid, dl = mtp
    n = len(pos)
    if not np.array_equal(pos, np.arange(n)) or dl.shape[0] != n:
        raise ValueError(f'draft rows: positions must be 0..{n - 1} in order, once each ({dl.shape[0]} logit rows)')
    want = (list(ids) + [gen[-1]])[1:n + 1]
    if tok.tolist() != want:
        raise ValueError('draft tokens are not the tokens after each position')
    h = torch.from_numpy(np.ascontiguousarray(hid)).to(dev)
    with torch.no_grad():
        logits = ref.mtp(h, torch.as_tensor(tok.astype(np.int64), device=dev), SeqState())
    g = torch.from_numpy(np.ascontiguousarray(dl[0:n])).to(dev)
    kl = kl_rows(logits, g)
    m = min(n, hidden.shape[0])
    return {'rows': n, 'hidden_rel': rel(h[:m], hidden[:m]), 'rel': rel(g, logits),
            'top1': int((g.argmax(-1) == logits.argmax(-1)).sum()), 'kl_mean': float(kl.mean()), 'kl_max': float(kl.max())}


def print_report(rep):
    print(f"{rep['tokens']} positions, layers {rep['layers']}; embedding rel-L2 {rep['embed']:.2e}")
    print('| layer | block, on gr8\'s input | streams rel-L2 | update rel-L2 |\n|---|---|---:|---:|')
    for s in rep['sublayers']:
        print(f"| {s['layer']} | {s['block']} | {s['stream']:.2e} | {s['update']:.2e} |")
    for L, d in rep['indexer'].items():
        print(f"indexer {L} (reference chain): {d['keys_dropped']} causal keys dropped, {d['queries_with_drops']} queries")
    h, c, T = rep['head'], rep['chain'], rep['tokens']
    print(f"head on gr8's streams: logits rel-L2 {h['rel']:.2e}, top-1 {h['top1']}/{T}")
    print(f"chain: logits rel-L2 {c['rel']:.2e}, top-1 {c['top1']}/{T}, KL(ref || gr8) mean {c['kl_mean']:.2e} "
          f"max {c['kl_max']:.2e} nats")
    print(f"greedy gr8 {c['greedy_gr8']}\ngreedy ref {c['greedy_ref']}")
    if 'floor' in rep:
        f = rep['floor']
        print(f"noise floor (BF16-rounded reference chain vs fp32 chain, not a gate): logits rel-L2 {f['rel']:.2e}, "
              f"top-1 {f['top1']}/{T}, KL mean {f['kl_mean']:.2e} max {f['kl_max']:.2e} nats")
        print('floor drift (rounded vs fp32 chain after each layer, rel-L2): ' +
              ' '.join(f"L{d['layer']} {d['rel']:.1e}" for d in rep['floor_drift']))
    if rep.get('drift'):
        print('chain drift (gr8 stream vs reference chain after each layer, rel-L2): ' +
              ' '.join(f"L{d['layer']} {d['rel']:.1e}" for d in rep['drift']))
    for m in c.get('misses', []):
        print(f"chain top-1 differs at row {m['row']}: reference top-2 gap {m['ref_gap']:.3f}, gr8 {m['gr8_gap']:.3f}")
    if 'mtp' in rep:
        m = rep['mtp']
        print(f"MTP ({m['rows']} rows, on gr8's hidden): gr8 hidden vs reference chain hidden rel-L2 {m['hidden_rel']:.2e}; "
              f"draft logits rel-L2 {m['rel']:.2e}, top-1 {m['top1']}/{m['rows']}, KL mean {m['kl_mean']:.2e} "
              f"max {m['kl_max']:.2e}")
    for k, v in rep['gates'].items():
        print(f"{'PASS' if v else 'FAIL'}: {k}")
    print('PASS' if rep['pass'] else 'FAIL')


def main(argv=None, weights=None, cfg=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', help='exlr8 release root (dense/ + experts/)')
    ap.add_argument('--dense', help='release root with the dense/ files, if not --release')
    ap.add_argument('--loader', help='exlr8-loader repo root')
    ap.add_argument('--config', help='the model config.json')
    ap.add_argument('--K', type=int, default=4, help='expert width m4sim ran (uniform manifest)')
    ap.add_argument('--layers', required=True, help='the m4sim --layers list, e.g. 0,1,2,3,4,5')
    ap.add_argument('--ids', required=True, help='the prompt ids file m4sim read (--prompt @FILE)')
    ap.add_argument('--m4', required=True, help='m4sim stdout (the "generated" line)')
    ap.add_argument('--resid', required=True)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--json', help='write the report here too')
    ap.add_argument('--mtp-in', help='G6_DRAFT_IN_DUMP of the same run: also check the MTP layer')
    ap.add_argument('--mtp-logits', help='G6_DRAFT_LOGIT_DUMP of the same run')
    ap.add_argument('--device', default='cpu', help='cuda: run the reference on the GPU (one that is not running the engine)')
    ap.add_argument('--floor', action='store_true', help='also report a BF16-rounded reference chain vs the fp32 '
                    'chain (the rounding-only noise floor; not a gate)')
    ap.add_argument('--cache-experts', action='store_true', help='decode each expert once per layer (GPU memory)')
    a = ap.parse_args(argv)
    layers = [int(x) for x in a.layers.split(',')]
    if weights is None:
        from weights import Exlr8Dense, Exlr8Experts, Overlay
        cfg = Cfg.load(a.config)
        weights = Overlay(Exlr8Experts(a.release, a.loader, a.K, layers=layers + ([cfg.mtp] if a.mtp_in else [])),
                          Exlr8Dense(a.dense or a.release, a.loader))
    torch.set_num_threads(os.cpu_count())
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = generated(a.m4)
    ids = read_ints(a.ids) + gen[:-1]
    mtp = (*read_draft_in(a.mtp_in, cfg.H), LogitDump(a.mtp_logits, cfg.vocab)) if a.mtp_in else None
    rep = compare(FlashRef(cfg, DeviceWeights(weights, a.device, a.cache_experts), dev=a.device), layers, ids, gen,
                  ResidDump(a.resid, cfg.hc, cfg.H), LogitDump(a.logits, cfg.vocab), a.device, mtp, a.floor)
    print_report(rep)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(rep, f, indent=1)
    return 0 if rep['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
