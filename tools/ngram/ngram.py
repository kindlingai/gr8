#!/usr/bin/env python3
"""S5 (#56) step 2-4: a GLM-distilled n-gram table as a draft source, tested offline by replay. Stdlib only, CPU only.

A trace is one GLM-5.3 generation with the model's top-k log-probabilities at every generated position (capture.py
records them through the front end; `import-kl-ref` turns a bench/gate KL reference into traces). One JSON line each:
    {"workload": "code", "prompt": [ids], "gen": [ids], "top": [[[id, logprob], ...] for each gen position]}

The table maps the last n tokens (several n, longest first, with backoff) to the top 2-4 next tokens, each with the
number of positions seen and GLM's mean probability for it there (a token outside a position's top-k counts as 0).
Entries below a count floor or a probability floor are dropped. Lookups chain: a proposed token extends the context
for the next lookup, so one hit can propose a run.

Replay walks each held-out trace as greedy speculative decoding would: at each round the source proposes a chain or a
small tree, the round accepts the longest proposed path that equals GLM's own tokens, and commits that plus one token.
A round with d draft tokens costs d + 1 verify rows (engine/runtime/spec_len.h). Tokens per ms use spec_len.h's TP4
verify times by row count (2200 MHz, DFlash2 verify steps); a lookup costs no GPU time. Trees need tree verification
(#57); chains run on today's verify path.

    python3 tools/ngram/ngram.py import-kl-ref bench/gate/refs/tp4.json --out traces.jsonl
    python3 tools/ngram/ngram.py replay traces.jsonl --folds 5 [--json out.json]
    python3 tools/ngram/ngram.py sizes traces.jsonl --floors 0.3,0.5,0.7,0.9
    python3 tools/ngram/ngram.py export traces.jsonl --out ngram.tab     # the engine's table (GR8_NGRAM)
"""
import argparse
import heapq
import json
import math
import random
import sys

# spec_len.h SpecLen::table: verify ms by rows 1..8 at TP4 (2200 MHz, median DFlash2 verify steps). Row 1 = no drafts.
VERIFY_MS = [0, 43.0, 52.4, 60.4, 68.4, 76.1, 83.7, 91.1, 97.4]
BUDGETS = (3, 5, 8)


def verify_ms(rows):
    return VERIFY_MS[min(max(rows, 1), len(VERIFY_MS) - 1)]


# ---------------------------------------------------------------- traces

def trace_key(t):
    """Traces of the same prompt fall in the same fold (greedy reruns of a prompt are near copies)."""
    return tuple(t['prompt'])


def load_traces(path):
    """One trace per line; an exact repeat of an earlier line (prompt and generation) is dropped."""
    out, seen = [], set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if len(t['top']) != len(t['gen']):
                raise ValueError(f'{path}:{ln}: {len(t["gen"])} gen tokens but {len(t["top"])} top lists')
            k = (tuple(t['prompt']), tuple(t['gen']))
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
    return out


def save_traces(path, traces):
    with open(path, 'w') as f:
        for t in traces:
            f.write(json.dumps(t, separators=(',', ':')) + '\n')


# bench/gate/data/kl_prompts.json, by index: what kind of answer each prompt asks for
KL_WORKLOAD = {0: 'code', 1: 'prose', 2: 'code', 3: 'prose', 4: 'json', 5: 'code', 6: 'prose', 7: 'code', 8: 'prose',
               9: 'prose', 10: 'code', 11: 'prose', 12: 'json', 13: 'prose', 14: 'code', 15: 'json'}


def from_kl_ref(ref):
    """A gate.py kl-ref file (greedy continuations with top-k at each position, taken on one build) -> traces."""
    out = []
    for i, s in enumerate(ref['sequences']):
        L = s['prompt_len']
        out.append({'workload': KL_WORKLOAD.get(i, 'other'), 'prompt': s['ids'][:L], 'gen': s['ids'][L:],
                    'top': [[[int(a), float(b)] for a, b in p] for p in s['pos']],
                    'source': f'kl-ref {ref.get("topology")} {ref.get("commit")} #{i}'})
    return out


# ---------------------------------------------------------------- the table

class Table:
    """ns: context lengths, longest first. width: candidates kept per entry. floor: least mean probability kept.
    min_count: least positions behind an entry."""

    def __init__(self, ns=(8, 6, 4, 3), width=4, floor=0.3, min_count=2):
        self.ns = tuple(sorted(set(ns), reverse=True))
        self.width, self.floor, self.min_count = width, floor, min_count
        self.raw = {n: {} for n in self.ns}     # n -> key -> [positions, {token: sum of p}]
        self.t = {n: {} for n in self.ns}       # n -> key -> ((token, mean p, count), ...) by mean p descending

    def add(self, trace):
        ctx = list(trace['prompt'])
        for tok, top in zip(trace['gen'], trace['top']):
            for n in self.ns:
                if len(ctx) < n:
                    continue
                k = tuple(ctx[-n:])
                e = self.raw[n].get(k)
                if e is None:
                    e = self.raw[n][k] = [0, {}]
                e[0] += 1
                for i, lp in top:
                    e[1][i] = e[1].get(i, 0.0) + math.exp(lp)
            ctx.append(tok)

    def finalize(self):
        for n in self.ns:
            out = {}
            for k, (cnt, sums) in self.raw[n].items():
                if cnt < self.min_count:
                    continue
                c = sorted(((i, s / cnt) for i, s in sums.items()), key=lambda x: (-x[1], x[0]))
                c = tuple((i, p, cnt) for i, p in c[:self.width] if p >= self.floor)
                if c:
                    out[k] = c
            self.t[n] = out
        return self

    def lookup(self, ctx):
        """-> (n, candidates) for the longest context length with an entry, or (0, ())."""
        for n in self.ns:
            if len(ctx) >= n:
                c = self.t[n].get(tuple(ctx[-n:]))
                if c:
                    return n, c
        return 0, ()

    def entries(self):
        return {n: len(self.t[n]) for n in self.ns}

    def bytes(self):
        """Host bytes in a packed layout: an 8-byte key hash and a 1-byte candidate count per entry, then 8 bytes per
        candidate (a 4-byte token id, a 2-byte probability, a 2-byte saturating position count). An estimate for
        sizing, not this object's size."""
        return sum(9 * len(t) + 8 * sum(len(c) for c in t.values()) for t in self.t.values())


def build(traces, **kw):
    t = Table(**kw)
    for tr in traces:
        t.add(tr)
    return t.finalize()


# ---------------------------------------------------------------- proposals
# A proposal is a list of nodes (parent index or -1, token, probability of the path from the root), in an order where
# every parent comes before its children. Rows = len(nodes) + 1.

def chain(table, ctx, k):
    nodes, c, p = [], list(ctx[-table.ns[0]:]), 1.0
    while len(nodes) < k:
        _, cand = table.lookup(c)
        if not cand:
            break
        tok, q, _ = cand[0]
        p *= q
        nodes.append((len(nodes) - 1, tok, p))
        c.append(tok)
    return nodes


def tree(table, ctx, k):
    """Best-first: the k nodes with the highest path probabilities. Path probability falls along every path, so every
    prefix of this order is a tree, and for the stored probabilities it is the tree of k nodes with the most expected
    accepted tokens (each node adds its path probability)."""
    nodes, heap, seq, tail = [], [], 0, list(ctx[-table.ns[0]:])

    def expand(parent, path, p):
        nonlocal seq
        _, cand = table.lookup(tail + path)
        for tok, q, _ in cand:
            heapq.heappush(heap, (-p * q, seq, parent, tok, path))
            seq += 1

    expand(-1, [], 1.0)
    while heap and len(nodes) < k:
        negp, _, parent, tok, path = heapq.heappop(heap)
        nodes.append((parent, tok, -negp))
        if len(nodes) < k:
            expand(len(nodes) - 1, path + [tok], -negp)
    return nodes


def prompt_lookup(ctx, k, nmax=4, nmin=2):
    """#34's draft source: the tokens after the latest earlier occurrence of the context's last n tokens (n from nmax
    down to nmin), in the prompt and the generation so far. It has no probabilities: path probability is None."""
    L = len(ctx)
    for n in range(nmax, nmin - 1, -1):
        if L <= n:
            continue
        pat = ctx[-n:]
        for s in range(L - n - 1, -1, -1):
            if ctx[s:s + n] == pat:
                run = ctx[s + n:s + n + k]
                return [(i - 1, t, None) for i, t in enumerate(run)]
    return []


def trim(nodes, rows_max):
    """Adaptive row count: the prefix of nodes (r - 1 of them, r <= rows_max) with the most expected tokens per ms."""
    best, br, e = 0, 1.0 / verify_ms(1), 1.0
    for j, (_, _, p) in enumerate(nodes[:rows_max - 1]):
        e += p
        r = e / verify_ms(j + 2)
        if r > br * (1 + 1e-9):
            best, br = j + 1, r
    return nodes[:best]


def accept(nodes, truth):
    """Longest path of nodes equal to truth[0], truth[1], ... -> its length."""
    best, depth = 0, {}
    for i, (parent, tok, _) in enumerate(nodes):
        d = 0 if parent < 0 else depth.get(parent)
        if d is None or d >= len(truth) or truth[d] != tok:
            continue
        depth[i] = d + 1
        best = max(best, d + 1)
    return best


# ---------------------------------------------------------------- the engine's table (GR8_NGRAM, engine/runtime/ngram_lookup.h)
# The engine proposes chains: only each entry's top token is exported. Its stored probability is calibrated: cal[b] is
# the share of held-out chain tokens whose stored probability fell in bucket b that GLM produced, given the tokens before
# them in the chain were right (the replay showed the raw means overestimate acceptance by 30-60%).

CAL_EDGES = (0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99)     # = spec_len.h SpecLen::bucket, ngram_lookup.h NgramTable::bucket


def bucket(p):
    b = 0
    while b < len(CAL_EDGES) and p >= CAL_EDGES[b]:
        b += 1
    return b


def stored(p):
    """A probability as the exported file holds it (6 places; the engine parses it to a double)."""
    return float(f'{min(max(p, 1e-6), 1.0):.6f}')


def calibrate(traces, k=5, seed=0, kmax=7, prior=4.0, **table_kw):
    """Held out by prompt as replay(): per bucket of a chain token's stored probability, (offered, accepted) counts and
    the calibrated acceptance (shrunk toward the bucket's lower edge by `prior` pseudo-counts, so a thin bucket errs
    low). Rounds walk as replay's chain source with every row filled."""
    n, a = [0] * 8, [0] * 8
    for train, test in folds(traces, k, seed):
        t = build(train, **table_kw)
        for tr in test:
            gen, ctx, pos = tr['gen'], list(tr['prompt']), 0
            while pos < len(gen):
                acc, w = 0, list(ctx[-t.ns[0]:])
                for j in range(kmax):
                    _, cand = t.lookup(w)
                    if not cand or pos + j >= len(gen):
                        break
                    tok, q, _ = cand[0]
                    w.append(tok)
                    b = bucket(stored(q))        # the probability as export writes it and the engine reads it
                    n[b] += 1
                    if gen[pos + j] != tok:
                        break
                    a[b] += 1
                    acc += 1
                commit = min(acc + 1, len(gen) - pos)
                ctx.extend(gen[pos:pos + commit])
                pos += commit
    lo = (0.0,) + CAL_EDGES
    cal = [(a[b] + prior * lo[b]) / (n[b] + prior) for b in range(8)]
    return n, a, cal


def export(table, cal, path):
    """Write the engine's table: top token per entry, entries sorted (the file's bytes, and so its digest, depend only
    on the table)."""
    with open(path, 'w') as f:
        f.write('# GLM-distilled n-gram table (tools/ngram/ngram.py export); engine/runtime/ngram_lookup.h reads it\n')
        f.write('g6ngram 1\n')
        f.write('ns ' + ' '.join(map(str, table.ns)) + '\n')
        f.write('cal ' + (' '.join(f'{c:.4f}' for c in cal) if cal else 'none') + '\n')
        for n in table.ns:
            for key in sorted(table.t[n]):
                tok, p, _ = table.t[n][key][0]
                f.write(f'e {n} ' + ' '.join(map(str, key)) + f' {tok} {stored(p):.6f}\n')


def read_export(path):
    """-> (ns, cal or None, {(n, key): (tok, p)}): the file as ngram_lookup.h reads it (tests)."""
    ns, cal, t = None, None, {}
    with open(path) as f:
        lines = [x.split() for x in f if x.strip() and not x.startswith('#')]
    if lines[0] != ['g6ngram', '1']:
        raise ValueError('not a g6ngram 1 file')
    for x in lines[1:]:
        if x[0] == 'ns':
            ns = tuple(int(v) for v in x[1:])
        elif x[0] == 'cal':
            cal = None if x[1] == 'none' else [float(v) for v in x[1:]]
        elif x[0] == 'e':
            n = int(x[1])
            t[(n, tuple(int(v) for v in x[2:2 + n]))] = (int(x[2 + n]), float(x[3 + n]))
    return ns, cal, t


def engine_chain(ns, cal, t, ctx, kmax):
    """ngram_lookup.h NgramTable::chain on a read_export() table: tokens and calibrated conditionals."""
    out, c, w = [], [], list(ctx[-ns[0]:])
    while len(out) < kmax:
        hit = None
        for n in ns:
            if len(w) >= n and (n, tuple(w[-n:])) in t:
                hit = t[(n, tuple(w[-n:]))]
                break
        if not hit:
            break
        out.append(hit[0])
        c.append(cal[bucket(hit[1])] if cal else hit[1])
        w.append(hit[0])
    return out, c


# ---------------------------------------------------------------- replay

SOURCES = ('prompt', 'chain', 'tree', 'chain+prompt', 'tree+prompt')


def propose(source, table, ctx, rows, adaptive):
    """-> (offered, nodes): whether the source had a proposal for this context (the table an entry, prompt lookup a
    match), and the nodes to verify (after the adaptive trim; the +prompt sources fall back to prompt lookup)."""
    k = rows - 1
    if source == 'prompt':
        nodes = prompt_lookup(ctx, k)
        return bool(nodes), nodes
    fn = chain if source.startswith('chain') else tree
    nodes = fn(table, ctx, k)
    offered = bool(nodes)
    if adaptive:
        nodes = trim(nodes, rows)
    if not nodes and source.endswith('+prompt'):
        nodes = prompt_lookup(ctx, k)
        offered = offered or bool(nodes)
    return offered, nodes


class Stats:
    """hits: rounds where the source had a proposal (before the row trim). used: rounds that verified drafts.
    t_*: the rounds whose verified nodes all carry the table's probabilities (every round of chain and tree; prompt
    lookup's nodes have none), for the calibration columns."""
    FIELDS = ('rounds', 'hits', 'used', 'tokens', 'acc', 'rows', 'drafted', 'ms',
              't_rounds', 't_acc', 't_rows', 't_tokens', 't_expected')

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)

    def add(self, offered, nodes, a, commit):
        self.rounds += 1
        self.hits += bool(offered)
        self.used += bool(nodes)
        self.drafted += len(nodes)
        self.acc += a
        self.tokens += commit
        self.rows += len(nodes) + 1
        self.ms += verify_ms(len(nodes) + 1)
        if all(p is not None for _, _, p in nodes):
            self.t_rounds += 1
            self.t_acc += a
            self.t_tokens += commit
            self.t_rows += len(nodes) + 1
            self.t_expected += sum(p for _, _, p in nodes)

    def merge(self, o):
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f) + getattr(o, f))

    def row(self):
        r = max(self.rounds, 1)
        tr = self.t_rounds
        return {'tokens': self.tokens, 'rounds': self.rounds, 'hit_rate': self.hits / r, 'used_rate': self.used / r,
                'acc_per_used': self.acc / max(self.used, 1), 'acc_per_round': self.acc / r,
                'tokens_per_round': self.tokens / r, 'rows_per_round': self.rows / r,
                'tokens_per_row': self.tokens / max(self.rows, 1), 'draft_precision': self.acc / max(self.drafted, 1),
                # calibration, over the rounds whose nodes came from the table: predicted against actual
                'table_rounds': tr,
                'expected_acc_per_round': self.t_expected / tr if tr else None,
                'actual_acc_per_round': self.t_acc / tr if tr else None,
                'expected_tokens_per_row': (tr + self.t_expected) / self.t_rows if tr else None,
                'tok_per_s': 1000.0 * self.tokens / max(self.ms, 1e-9),
                'speedup': (self.tokens / max(self.ms, 1e-9)) * verify_ms(1)}


def replay_trace(trace, table, source, rows, adaptive):
    st = Stats()
    gen, ctx, pos = trace['gen'], list(trace['prompt']), 0
    while pos < len(gen):
        offered, nodes = propose(source, table, ctx, rows, adaptive)
        a = accept(nodes, gen[pos:])
        commit = min(a + 1, len(gen) - pos)
        st.add(offered, nodes, a, commit)
        ctx.extend(gen[pos:pos + commit])
        pos += commit
    return st


def folds(traces, k, seed=0):
    """k splits by prompt: each holds out about 1/k of the prompts with all their traces (k=5: build on 80%, replay
    20%). A prompt's traces never sit on both sides."""
    groups = {}
    for i, t in enumerate(traces):
        groups.setdefault(trace_key(t), []).append(i)
    keys = sorted(groups, key=lambda g: groups[g][0])
    random.Random(seed).shuffle(keys)
    for f in range(k):
        test = {i for g in keys[f::k] for i in groups[g]}
        yield [t for i, t in enumerate(traces) if i not in test], [t for i, t in enumerate(traces) if i in test]


def replay(traces, k=5, seed=0, budgets=BUDGETS, sources=SOURCES, adaptive=True, **table_kw):
    """-> {(source, rows, workload): Stats}, workload 'all' included, and the tables' sizes per fold."""
    res, sizes = {}, []
    for train, test in folds(traces, k, seed):
        t = build(train, **table_kw)
        sizes.append({'entries': sum(t.entries().values()), 'bytes': t.bytes()})
        for tr in test:
            for src in sources:
                for b in budgets:
                    s = replay_trace(tr, t, src, b, adaptive)
                    for w in (tr.get('workload', 'other'), 'all'):
                        res.setdefault((src, b, w), Stats()).merge(s)
    return res, sizes


def table_md(res, budgets=BUDGETS, sources=SOURCES):
    ws = sorted({w for _, _, w in res}, key=lambda w: (w == 'all', w))
    lines = ['| workload | source | rows | hit % | used % | accepted / used | tokens / round | rows / round '
             '| tokens / row | expected tokens / row | table acc / round: expected (actual) | speedup vs 1 row |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for w in ws:
        for src in sources:
            for b in budgets:
                s = res.get((src, b, w))
                if not s:
                    continue
                r = s.row()
                if src == 'prompt' or r['table_rounds'] == 0:
                    ept = exp = '-'
                else:
                    ept = f'{r["expected_tokens_per_row"]:.3f}'
                    exp = f'{r["expected_acc_per_round"]:.3f} ({r["actual_acc_per_round"]:.3f})'
                lines.append(f'| {w} | {src} | {b} | {100 * r["hit_rate"]:.1f} | {100 * r["used_rate"]:.1f} | '
                             f'{r["acc_per_used"]:.2f} | {r["tokens_per_round"]:.3f} | {r["rows_per_round"]:.2f} | '
                             f'{r["tokens_per_row"]:.3f} | {ept} | {exp} | {r["speedup"]:.3f} |')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('import-kl-ref', help='a bench/gate kl-ref file -> traces')
    p.add_argument('ref')
    p.add_argument('--out', required=True)
    for name in ('replay', 'sizes', 'export'):
        p = sub.add_parser(name)
        p.add_argument('traces', nargs='+')
        p.add_argument('--ns', default='8,6,4,3', help='context lengths, with backoff from the longest')
        p.add_argument('--width', type=int, default=4)
        p.add_argument('--floor', type=float, default=0.3)
        p.add_argument('--min-count', type=int, default=2)
        p.add_argument('--folds', type=int, default=5)
        p.add_argument('--seed', type=int, default=0)
    p = sub.choices['replay']
    p.add_argument('--budgets', default='3,5,8')
    p.add_argument('--sources', default=','.join(SOURCES))
    p.add_argument('--fill', action='store_true', help='always use the whole row budget (default: adaptive rows)')
    p.add_argument('--json', help='also write the results here')
    sub.choices['sizes'].add_argument('--floors', default='0.1,0.3,0.5,0.7,0.9')
    sub.choices['sizes'].add_argument('--widths', default='', help='also sweep these widths (at --floor)')
    sub.choices['sizes'].add_argument('--ns-list', default='', help='also sweep these ns sets, e.g. "8,6,4,3;4,3;3"')
    sub.choices['export'].add_argument('--out', required=True, help="the engine's table file (GR8_NGRAM)")
    sub.choices['export'].add_argument('--no-cal', action='store_true', help='write the raw mean probabilities (cal none)')
    a = ap.parse_args(argv)

    if a.cmd == 'import-kl-ref':
        with open(a.ref) as f:
            tr = from_kl_ref(json.load(f))
        save_traces(a.out, tr)
        print(f'{len(tr)} traces, {sum(len(t["gen"]) for t in tr)} positions -> {a.out}')
        return 0

    traces = [t for p in a.traces for t in load_traces(p)]
    ns = tuple(int(x) for x in a.ns.split(','))
    kw = dict(ns=ns, width=a.width, min_count=a.min_count)
    print(f'{len(traces)} traces, {sum(len(t["gen"]) for t in traces)} generated positions, {a.folds} folds')
    if a.cmd == 'export':
        table = build(traces, ns=ns, width=a.width, floor=a.floor, min_count=a.min_count)
        cal = None
        if not a.no_cal:
            n, acc, cal = calibrate(traces, a.folds, a.seed, ns=ns, width=a.width, floor=a.floor, min_count=a.min_count)
            print('| bucket from | chain tokens | accepted | calibrated |')
            print('|---|---|---|---|')
            for b, lo in enumerate((0.0,) + CAL_EDGES):
                print(f'| {lo} | {n[b]} | {acc[b]} | {cal[b]:.3f} |')
        export(table, cal, a.out)
        print(f'{sum(table.entries().values())} entries -> {a.out}')
        return 0
    if a.cmd == 'sizes':
        print('| ns | width | floor | entries (mean per fold) | packed MB | chain hit % (8 rows) '
              '| tokens / round (chain, 8 rows) |')
        print('|---|---|---|---|---|---|---|')
        points = [(ns, a.width, float(x)) for x in a.floors.split(',')]
        points += [(ns, int(w), a.floor) for w in a.widths.split(',') if w]
        points += [(tuple(int(x) for x in n.split(',')), a.width, a.floor) for n in a.ns_list.split(';') if n]
        for pns, pw, fl in points:
            res, sizes = replay(traces, a.folds, a.seed, budgets=(8,), sources=('chain',), ns=pns, width=pw,
                                floor=fl, min_count=a.min_count)
            r = res[('chain', 8, 'all')].row()
            ent = sum(s['entries'] for s in sizes) / len(sizes)
            mb = sum(s['bytes'] for s in sizes) / len(sizes) / 1e6
            print(f'| {",".join(map(str, pns))} | {pw} | {fl} | {ent:.0f} | {mb:.3f} | {100 * r["hit_rate"]:.1f} | '
                  f'{r["tokens_per_round"]:.3f} |')
        return 0

    budgets = tuple(int(x) for x in a.budgets.split(','))
    sources = tuple(a.sources.split(','))
    res, sizes = replay(traces, a.folds, a.seed, budgets, sources, adaptive=not a.fill, floor=a.floor, **kw)
    print(f'table per fold: {sum(s["entries"] for s in sizes) / len(sizes):.0f} entries, '
          f'{sum(s["bytes"] for s in sizes) / len(sizes) / 1e6:.3f} MB packed; rows {"filled" if a.fill else "adaptive"}')
    print(table_md(res, budgets, sources))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'args': vars(a), 'sizes': sizes,
                       'results': [dict(source=s, rows=b, workload=w, **st.row()) for (s, b, w), st in res.items()]},
                      f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
