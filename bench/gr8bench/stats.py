"""Error bars and paired comparisons. The method follows the glm-5.3-flash-gx10 recipe's gate/bench."""
import math
import random
import statistics

BOOT = 2000
Z95, Z80 = 1.96, 0.8416


def bootstrap(xs, stat, seed=0):
    """95% interval of stat over resamples of xs."""
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def modified_z(xs):
    med = statistics.median(xs)
    mad = statistics.median(abs(x - med) for x in xs)
    return [0.0 if mad == 0 else 0.6745 * (x - med) / mad for x in xs]


def summarize(runs):
    """Runs of one workload (tokens, decode_s, tps, tok_per_step, ms_per_step, ttft) -> the workload's numbers.
    tok/s is all tokens over all decode time, so a long run weighs more than a short one."""
    rate = lambda rs: sum(r['tokens'] for r in rs) / sum(r['decode_s'] for r in rs)
    rates = [r['tps'] for r in runs]
    cv = statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else None
    lo, hi = bootstrap(runs, rate) if len(runs) > 1 else (None, None)
    med = lambda k: statistics.median(r[k] for r in runs if r.get(k) is not None) \
        if any(r.get(k) is not None for r in runs) else None
    return {'runs': len(runs), 'tps': rate(runs), 'ci': [lo, hi], 'cv': cv,
            # the smallest difference an unpaired comparison of this many runs finds (95% confidence, 80% power)
            'detects': (Z95 + Z80) * math.sqrt(2) * cv / math.sqrt(len(runs)) if cv is not None else None,
            'tok_per_step': med('tok_per_step'), 'ms_per_step': med('ms_per_step'), 'accept': med('accept'),
            'ttft_ms': 1000 * med('ttft') if med('ttft') is not None else None,
            'outliers': [r['i'] for r, z in zip(runs, modified_z(rates)) if abs(z) > 3.5] if len(rates) > 2 else []}


def wilcoxon(ds):
    """Two-sided exact Wilcoxon signed-rank p-value. Zero differences are dropped."""
    ds = [d for d in ds if d != 0]
    n = len(ds)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda k: abs(ds[k]))
    ranks = [0] * n
    k = 0
    while k < n:  # ties share the average rank; doubled ranks stay integers
        j = k
        while j + 1 < n and abs(ds[order[j + 1]]) == abs(ds[order[k]]):
            j += 1
        for m in range(k, j + 1):
            ranks[order[m]] = k + j + 2
        k = j + 1
    w = sum(r for r, d in zip(ranks, ds) if d > 0)
    counts = {0: 1}  # sign patterns giving each positive-rank sum
    for r in ranks:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    total = 2 ** n
    below = sum(c for s, c in counts.items() if s <= w) / total
    above = sum(c for s, c in counts.items() if s >= w) / total
    return min(1.0, 2 * min(below, above))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, worst = [0.0] * len(ps), 0.0
    for rank, k in enumerate(order):
        worst = max(worst, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = worst
    return adj


def paired(a, b, key='tps'):
    """Runs of one workload on two builds, paired by run index -> B / A as a geometric mean with a bootstrap 95%
    interval and an exact Wilcoxon p-value, and the same ratio for tokens per step and ms per step. None with fewer
    than two pairs."""
    ra = {r['i']: r for r in a if not r.get('problem')}
    rb = {r['i']: r for r in b if not r.get('problem')}
    idx = sorted(set(ra) & set(rb))

    def logs(k):
        return [math.log(rb[i][k] / ra[i][k]) for i in idx if rb[i].get(k) and ra[i].get(k)]
    ds = logs(key)
    if len(ds) < 2:
        return None
    lo, hi = bootstrap(ds, statistics.mean)
    gm = lambda xs: math.exp(statistics.mean(xs)) if xs else None
    return {'pairs': len(ds), 'ratio': gm(ds), 'ci': [math.exp(lo), math.exp(hi)], 'p': wilcoxon(ds),
            'tok_per_step': gm(logs('tok_per_step')), 'ms_per_step': gm(logs('ms_per_step'))}
