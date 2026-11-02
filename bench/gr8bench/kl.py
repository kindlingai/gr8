"""KL drift: score fixed token sequences with prompt_logprobs (top-K at every position) and compare with a reference
taken from a recorded build. KL(reference || new), the tail lumped into one bucket, and top-1 agreement."""
import json
import math

TINY = 1e-12


def prompts(path):
    with open(path) as f:
        return json.load(f)


def score(client, ids, k):
    """prompt_logprobs of a token-id sequence -> per position [(id, logprob), ...] sorted by logprob descending
    (ranks 1..k only), None at position 0."""
    r = client.complete(ids, max_tokens=1, temperature=0, prompt_logprobs=k)
    plp = r['choices'][0]['prompt_logprobs']
    if plp is None or len(plp) != len(ids):
        raise RuntimeError(f'server returned {None if plp is None else len(plp)} prompt_logprobs for {len(ids)} tokens')
    out = []
    for e in plp:
        if e is None:
            out.append(None)
            continue
        top = [(int(i), v['logprob']) for i, v in e.items() if v.get('rank') is not None and v['rank'] <= k]
        top.sort(key=lambda t: -t[1])
        out.append(top)
    return out


def make_reference(client, prompt_texts, n_tokens=256, k=20):
    """Greedy continuations of the fixed prompts on this build, scored on the same build.
    -> list of {'prompt_len', 'ids', 'pos'} (pos = top-k lists for the continuation positions only)."""
    seqs = []
    for text in prompt_texts:
        p = client.tokenize_chat([{'role': 'user', 'content': text}])
        r = client.complete(p, max_tokens=n_tokens, min_tokens=n_tokens, temperature=0, ignore_eos=True,
                            return_token_ids=True)
        gen = r['choices'][0]['token_ids']
        ids = p + gen
        top = score(client, ids, k)
        seqs.append({'prompt_len': len(p), 'ids': ids, 'pos': [[[i, v] for i, v in t] for t in top[len(p):]]})
    return seqs


def kl_position(ref, new):
    """KL(ref || new) at one position. ref, new: [(id, logprob), ...] top-K lists.
    Bucket A = the ref's top-K ids, bucket R = everything else. A token of A missing from new's top-K gets new's
    K-th logprob (an upper bound on its true value), so the result is a lower bound for those terms."""
    p = {i: math.exp(v) for i, v in ref}
    q_top = {i: math.exp(v) for i, v in new}
    floor = min(q_top.values()) if q_top else TINY
    q = {i: q_top.get(i, floor) for i in p}
    p_tail = max(1.0 - sum(p.values()), TINY)
    q_tail = max(1.0 - sum(q.values()), TINY)
    kl = sum(pv * math.log(pv / max(q[i], TINY)) for i, pv in p.items())
    kl += p_tail * math.log(p_tail / q_tail)
    return max(kl, 0.0)  # KL is not negative; below zero is rounding


def compare(client, reference, k=20):
    """Score every reference sequence on `client`'s build. -> dict of the KL numbers."""
    per_seq, all_kl, agree, n = [], [], 0, 0
    for s in reference['sequences']:
        top = score(client, s['ids'], k)[s['prompt_len']:]
        kls, ag = [], 0
        for r, t in zip(s['pos'], top):
            kls.append(kl_position([(i, v) for i, v in r], t))
            ag += bool(r and t and r[0][0] == t[0][0])
        per_seq.append({'kl_mean': sum(kls) / len(kls), 'top1': ag / len(kls)})
        all_kl += kls
        agree += ag
        n += len(kls)
    all_kl.sort()
    return {'positions': n, 'sequences': len(per_seq), 'mean': sum(all_kl) / n, 'p95': all_kl[int(0.95 * (n - 1))],
            'max': all_kl[-1], 'top1': agree / n, 'per_sequence': per_seq,
            'reference_commit': reference.get('commit'), 'reference_topology': reference.get('topology')}
