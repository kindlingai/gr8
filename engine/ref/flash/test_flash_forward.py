"""flash_forward.py on synthetic m4sim dumps of a small random model (test_flash_ref's config), CPU, ~10 s.

    python3 engine/ref/flash/test_flash_forward.py      # needs transformers >= 5.19 (glm5_next)

The "engine" here is flash_ref itself: its streams after the embedding and after every sublayer, and its logits, are
written in m4sim's G6_RESID_DUMP / G6_LOGIT_DUMP framing (steps of up to 5 rows, logits split over 4 ranks), with a
'generated' line of its greedy tokens. Checks: (1) the clean dumps PASS with errors at fp32 rounding; (2) one
sublayer's output scaled by 5% (later sublayers computed from it) fails that sublayer only; (3) a wrong greedy
token fails the greedy gate; (4) --cache-experts gives the same report as (1);
(5) MTP draft dumps of the reference's own draft logits PASS; (6) one wrong draft top-1 fails the MTP top-1 gate;
(7) --floor reports a BF16-rounded chain (small, nonzero) and leaves the gates and the reference unchanged. Case 1 also checks that the reference indexer drops keys.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flash_forward as FF  # noqa: E402
from flash_ref import Cfg, FlashRef, SeqState  # noqa: E402
from test_flash_ref import checkpoint_dict, hf_model  # noqa: E402
from weights import DictWeights  # noqa: E402

LAYERS = [0, 1, 2, 3]       # KDA + dense x 3, DSA + MoE


def engine_run(ref, ids):
    """[slots, T, hc, H] streams (embedding, then after each sublayer) and [T, V] logits, as one chain"""
    st = SeqState()
    x = ref.emb(ids).unsqueeze(1).expand(-1, ref.c.hc, -1).contiguous()
    slots = [x]
    for L in LAYERS:
        x = ref.attn_block(x, L, st)
        slots.append(x)
        x = ref.mlp_block(x, L)
        slots.append(x)
    return torch.stack(slots).numpy().astype('<f4'), ref.head(x)[1].numpy().astype('<f4')


def write_dumps(d, g6, logits, prompt, gen, steps=5, ranks=4):
    T = g6.shape[1]
    with open(os.path.join(d, 'resid.bin'), 'wb') as r, open(os.path.join(d, 'logits.bin'), 'wb') as lg:
        for s, a in enumerate(range(0, T, steps)):
            blk = g6[:, a:a + steps]
            M = blk.shape[1]
            r.write(np.array([s, M, g6.shape[0], g6.shape[2] * g6.shape[3]], '<i4').tobytes())
            r.write(np.ascontiguousarray(blk).tobytes())
            lg.write(np.array([s, M], '<i4').tobytes())
            for part in np.array_split(logits[a:a + M], ranks, 1):
                lg.write(np.array([part.shape[1]], '<i4').tobytes())
                lg.write(np.ascontiguousarray(part).tobytes())
    open(os.path.join(d, 'prompt.ids'), 'w').write(','.join(map(str, prompt)) + '\n')
    open(os.path.join(d, 'steps.txt'), 'w').write('step 0 M 5 tok 1 lp 0\ngenerated ' + ' '.join(map(str, gen)) + '\n')


def write_draft(d, ref, ids, gen, hidden, steps=5, ranks=4, bump=None):
    """first-pass draft dumps: rows 0..T-1 (the last row's next token is the last generated one), gr8's own hidden"""
    tok = (list(ids) + [gen[-1]])[1:]
    with torch.no_grad():
        dl = ref.mtp(hidden, torch.tensor(tok), SeqState()).numpy().astype('<f4')
    if bump is not None:                                      # make another token the top-1 of row bump
        dl[bump, (dl[bump].argmax() + 1) % dl.shape[1]] = dl[bump].max() + 5
    h = hidden.numpy().astype('<f4')
    with open(os.path.join(d, 'din.bin'), 'wb') as fi, open(os.path.join(d, 'dlog.bin'), 'wb') as fl:
        for s, a in enumerate(range(0, len(tok), steps)):
            M = len(tok[a:a + steps])
            fi.write(np.array([s, M], '<i4').tobytes() + np.arange(a, a + M, dtype='<i4').tobytes()
                     + np.array(tok[a:a + M], '<i4').tobytes() + np.ascontiguousarray(h[a:a + M]).tobytes())
            fl.write(np.array([s, M], '<i4').tobytes())
            for part in np.array_split(dl[a:a + M], ranks, 1):
                fl.write(np.array([part.shape[1]], '<i4').tobytes() + np.ascontiguousarray(part).tobytes())


def run(d, ref, cfg, extra=()):
    argv = list(extra) + ['--layers', ','.join(map(str, LAYERS)), '--ids', f'{d}/prompt.ids', '--m4', f'{d}/steps.txt',
            '--resid', f'{d}/resid.bin', '--logits', f'{d}/logits.bin', '--json', f'{d}/rep.json']
    rc = FF.main(argv, weights=ref.w, cfg=cfg)
    import json
    return rc, json.load(open(f'{d}/rep.json'))


def main():
    hcfg, m = hf_model()
    cfg = Cfg(dict(hcfg.to_dict(), num_nextn_predict_layers=1))
    ref = FlashRef(cfg, DictWeights(checkpoint_dict(hcfg, m)))
    torch.manual_seed(3)
    prompt = torch.randint(0, cfg.vocab, (13,)).tolist()
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(4):                                   # greedy continuation, the reference's own
            ids.append(int(ref.forward(ids, SeqState(), layers=LAYERS)['logits'][-1].argmax()))
        gen, ids = ids[len(prompt):], ids[:-1]
        g6, logits = engine_run(ref, ids)
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        write_dumps(d, g6, logits, prompt, gen)
        rc, rep = run(d, ref, cfg)
        rep_clean = rep
        worst = max(max(s['stream'], s['update']) for s in rep['sublayers'])
        drops = {L: v['keys_dropped'] for L, v in rep['indexer'].items()}
        ok = rc == 0 and worst < 1e-5 and rep['chain']['kl_mean'] < 1e-9 and rep['chain']['top1'] == len(ids) \
            and len(drops) == sum(not cfg.kda[L] for L in LAYERS) and all(drops.values())
        print(f'1 clean: rc {rc}, worst sublayer {worst:.1e}, KL {rep["chain"]["kl_mean"]:.1e}, indexer drops {drops}',
              'OK' if ok else 'BAD')
        bad += not ok

        k = 4                                                 # the output of layer 1's MLP (sublayer 3) 5% off,
        g6p = g6.copy()                                       # and the engine's later sublayers run on it
        g6p[k] = 1.05 * g6[k]
        with torch.no_grad():
            x = torch.from_numpy(g6p[k].copy())
            for j in range(k, 2 * len(LAYERS)):
                L = LAYERS[j // 2]
                x = ref.attn_block(x, L, SeqState()) if j % 2 == 0 else ref.mlp_block(x, L)
                g6p[j + 1] = x.numpy()
        write_dumps(d, g6p, logits, prompt, gen)
        rc, rep = run(d, ref, cfg)
        fails = [i for i, s in enumerate(rep['sublayers']) if s['stream'] > FF.GATE_STREAM]
        ok = rc == 1 and fails == [k - 1]
        print(f'2 perturbed slot {k}: rc {rc}, failing sublayers {fails}', 'OK' if ok else 'BAD')
        bad += not ok

        write_dumps(d, g6, logits, prompt, gen)
        rc4, rep4 = run(d, ref, cfg, ['--cache-experts'])
        ok = rc4 == 0 and rep4['sublayers'] == rep_clean['sublayers'] and rep4['chain'] == rep_clean['chain']
        print(f'4 clean with --cache-experts: rc {rc4}, report equal to case 1', 'OK' if ok else 'BAD')
        bad += not ok

        rc7, rep7 = run(d, ref, cfg, ['--floor'])             # the floor is a report, not a gate: same gates
        f = rep7['floor']
        ok = rc7 == 0 and rep7['gates'] == rep_clean['gates'] and rep7['chain'] == rep_clean['chain'] \
            and len(rep7['floor_drift']) == len(LAYERS) and 1e-5 < f['rel'] < 5e-2 and ref.qz is None
        print(f"7 clean with --floor: rc {rc7}, floor logits rel-L2 {f['rel']:.1e}, top-1 {f['top1']}/{len(ids)}, "
              f"KL {f['kl_mean']:.1e}", 'OK' if ok else 'BAD')
        bad += not ok

        with torch.no_grad():
            hidden = ref.head(torch.from_numpy(g6[-1].copy()))[0]
        mt = ['--mtp-in', f'{d}/din.bin', '--mtp-logits', f'{d}/dlog.bin']
        write_draft(d, ref, ids, gen, hidden)
        rc5, rep5 = run(d, ref, cfg, mt)
        m = rep5['mtp']
        ok = rc5 == 0 and m['rows'] == len(ids) and m['top1'] == m['rows'] and m['kl_mean'] < 1e-9
        print(f"5 MTP clean: rc {rc5}, rows {m['rows']}, top-1 {m['top1']}, KL {m['kl_mean']:.1e}", 'OK' if ok else 'BAD')
        bad += not ok
        write_draft(d, ref, ids, gen, hidden, bump=2)
        rc6, rep6 = run(d, ref, cfg, mt)
        ok = rc6 == 1 and not rep6['gates']['MTP top-1 >= 99%'] and rep6['gates']['greedy equal']
        print(f"6 MTP wrong draft row: rc {rc6}, top-1 {rep6['mtp']['top1']}/{rep6['mtp']['rows']}", 'OK' if ok else 'BAD')
        bad += not ok

        write_dumps(d, g6, logits, prompt, gen[:-1] + [(gen[-1] + 1) % cfg.vocab])
        rc, rep = run(d, ref, cfg)
        ok = rc == 1 and not rep['gates']['greedy equal'] and rep['gates']['top-1 >= 99%']
        print(f'3 wrong greedy token: rc {rc}', 'OK' if ok else 'BAD')
        bad += not ok
    print('FAIL' if bad else 'PASS')
    return int(bool(bad))


if __name__ == '__main__':
    raise SystemExit(main())
