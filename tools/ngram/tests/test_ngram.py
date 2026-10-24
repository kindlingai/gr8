#!/usr/bin/env python3
"""Desk tests for the n-gram draft source replay (no GPU, no network). Run: python3 -m unittest discover -s tools/ngram"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ngram as ng  # noqa: E402


def lp(p):
    return math.log(p)


def trace(prompt, gen, tops=None, workload='code'):
    """tops: per position [(id, p), ...]; default: the generated token at 0.99."""
    if tops is None:
        tops = [[(t, 0.99)] for t in gen]
    return {'workload': workload, 'prompt': list(prompt), 'gen': list(gen),
            'top': [[[i, lp(p)] for i, p in tp] for tp in tops]}


class TableTest(unittest.TestCase):
    def test_counts_mean_probability_and_floor(self):
        # context (1, 2, 3) -> 4 at 0.9 and 5 at 0.1 in one trace; 4 at 0.5, 6 at 0.5 in another
        a = trace([1, 2, 3], [4], [[(4, 0.9), (5, 0.1)]])
        b = trace([1, 2, 3], [4], [[(4, 0.5), (6, 0.5)]])
        t = ng.build([a, b], ns=(3,), width=4, floor=0.2, min_count=2)
        n, c = t.lookup([9, 1, 2, 3])
        self.assertEqual(n, 3)
        self.assertEqual([x[0] for x in c], [4, 6])          # 5: mean 0.05, below the floor
        self.assertAlmostEqual(c[0][1], 0.7)
        self.assertAlmostEqual(c[1][1], 0.25)
        self.assertEqual(c[0][2], 2)

    def test_min_count_and_backoff(self):
        tr = [trace([7, 1, 2], [3]), trace([8, 1, 2], [3])]
        t = ng.build(tr, ns=(3, 2), floor=0.5, min_count=2)
        self.assertEqual(t.entries(), {3: 0, 2: 1})            # each 3-gram seen once
        self.assertEqual(t.lookup([5, 1, 2])[0], 2)
        self.assertEqual(t.lookup([5, 6, 2]), (0, ()))

    def test_width(self):
        tr = [trace([1, 2], [3], [[(3, 0.4), (4, 0.3), (5, 0.2), (6, 0.1)]])] * 2
        t = ng.build(tr, ns=(2,), width=2, floor=0.0)
        self.assertEqual([x[0] for x in t.lookup([1, 2])[1]], [3, 4])
        self.assertEqual(t.bytes(), 9 + 16)


class ProposalTest(unittest.TestCase):
    def setUp(self):
        # a loop 1 2 | 3 4 1 2 3 4 ... seen twice, so every 2-gram predicts the next token at 0.99
        g = [3, 4, 1, 2] * 4
        self.t = ng.build([trace([1, 2], g), trace([1, 2], g)], ns=(2,), floor=0.5)

    def test_chain_runs_until_k(self):
        nodes = ng.chain(self.t, [1, 2], 5)
        self.assertEqual([x[1] for x in nodes], [3, 4, 1, 2, 3])
        self.assertEqual([x[0] for x in nodes], [-1, 0, 1, 2, 3])
        self.assertAlmostEqual(nodes[-1][2], 0.99 ** 5)

    def test_accept(self):
        nodes = ng.chain(self.t, [1, 2], 5)
        self.assertEqual(ng.accept(nodes, [3, 4, 1, 9]), 3)
        self.assertEqual(ng.accept(nodes, [9]), 0)
        self.assertEqual(ng.accept(nodes, [3, 4]), 2)           # the trace ends

    def test_tree_branches_on_the_second_option(self):
        a = trace([1, 2], [3, 5], [[(3, 0.6), (4, 0.4)], [(5, 0.99)]])
        b = trace([1, 2], [3, 5], [[(3, 0.6), (4, 0.4)], [(5, 0.99)]])
        c = trace([2, 4], [6], [[(6, 0.99)]])
        d = trace([2, 4], [6], [[(6, 0.99)]])
        t = ng.build([a, b, c, d], ns=(2,), floor=0.3)
        nodes = ng.tree(t, [1, 2], 3)
        # root options 3 (0.6) and 4 (0.4); then 3 -> 5 (0.594) beats 4 -> 6 (0.396)
        self.assertEqual([(p, x) for p, x, _ in nodes], [(-1, 3), (0, 5), (-1, 4)])
        self.assertEqual(ng.accept(nodes, [4, 6]), 1)
        self.assertEqual(ng.accept(nodes, [3, 5]), 2)
        nodes4 = ng.tree(t, [1, 2], 4)
        self.assertEqual(nodes4[3][:2], (2, 6))
        self.assertEqual(ng.accept(nodes4, [4, 6]), 2)
        for i, (par, _, p) in enumerate(nodes4):               # parents first, path probabilities fall
            self.assertLess(par, i)
            if par >= 0:
                self.assertLessEqual(p, nodes4[par][2])

    def test_trim_keeps_rows_that_pay(self):
        sure = [(i - 1, 0, 0.99) for i in range(7)]
        self.assertEqual(len(ng.trim(sure, 8)), 7)
        self.assertEqual(len(ng.trim(sure, 3)), 2)
        unsure = [(-1, 0, 0.1)]
        self.assertEqual(ng.trim(unsure, 8), [])                # 1.1 tokens / 52.4 ms < 1 / 43 ms

    def test_prompt_lookup(self):
        ctx = [5, 6, 7, 8, 9, 1, 6, 7]
        self.assertEqual([x[1] for x in ng.prompt_lookup(ctx, 3)], [8, 9, 1])
        self.assertEqual(ng.prompt_lookup([1, 2, 3], 3), [])


class ReplayTest(unittest.TestCase):
    def test_no_source_commits_one_token_per_round(self):
        tr = trace([1], [2, 3, 4, 5])
        t = ng.build([], ns=(2,))
        st = ng.replay_trace(tr, t, 'chain', 8, True)
        self.assertEqual((st.tokens, st.rounds, st.hits, st.rows), (4, 4, 0, 4))
        self.assertAlmostEqual(st.row()['speedup'], 1.0)

    def test_perfect_table_commits_runs(self):
        g = [3, 4, 1, 2] * 8
        t = ng.build([trace([1, 2], g)] * 2, ns=(2,), floor=0.5)
        st = ng.replay_trace(trace([1, 2], g), t, 'chain', 8, False)
        self.assertEqual(st.tokens, len(g))
        self.assertEqual(st.rounds, 4)                          # 8 tokens a round
        self.assertGreater(st.row()['speedup'], 3)

    def test_replay_is_held_out_and_reports_workloads(self):
        g = [3, 4, 1, 2] * 8
        traces = [trace([1, 2], g, workload='code') for _ in range(4)] + \
                 [trace([5, 6], [7, 8, 9] * 5, workload='prose') for _ in range(4)]
        res, sizes = ng.replay(traces, k=4, budgets=(3,), sources=('chain', 'prompt'), ns=(2,), min_count=1)
        self.assertEqual(len(sizes), 4)
        self.assertEqual(res[('chain', 3, 'all')].tokens, sum(len(t['gen']) for t in traces))
        self.assertIn(('chain', 3, 'prose'), res)
        self.assertGreater(res[('chain', 3, 'code')].row()['tokens_per_round'], 2.5)
        md = ng.table_md(res, (3,), ('chain', 'prompt'))
        self.assertIn('| all | chain | 3 |', md)

    def test_folds_partition(self):
        traces = [{'i': i} for i in range(11)]
        seen = []
        for train, test in ng.folds(traces, 5):
            self.assertEqual(len(train) + len(test), 11)
            seen += [t['i'] for t in test]
        self.assertEqual(sorted(seen), list(range(11)))


class CliTest(unittest.TestCase):
    def test_import_kl_ref_and_replay(self):
        ref = {'topology': 'tp4', 'commit': 'abc', 'sequences': [
            {'prompt_len': 2, 'ids': [1, 2] + [3, 4] * 6, 'pos': [[[t, -0.01], [9, -5.0]] for t in [3, 4] * 6]}
            for _ in range(5)]}
        with tempfile.TemporaryDirectory() as d:
            rp, tp, jp = (os.path.join(d, x) for x in ('ref.json', 't.jsonl', 'r.json'))
            with open(rp, 'w') as f:
                json.dump(ref, f)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ng.main(['import-kl-ref', rp, '--out', tp]), 0)
                self.assertEqual(ng.main(['replay', tp, '--ns', '2', '--json', jp]), 0)
                self.assertEqual(ng.main(['sizes', tp, '--ns', '2', '--floors', '0.5']), 0)
            self.assertIn('| all | chain | 8 |', out.getvalue())
            tr = ng.load_traces(tp)
            self.assertEqual(len(tr), 5)
            self.assertEqual(tr[0]['workload'], 'code')
            with open(jp) as f:
                out = json.load(f)
            row = [r for r in out['results'] if r['source'] == 'chain' and r['rows'] == 8 and r['workload'] == 'all'][0]
            self.assertGreater(row['tokens_per_round'], 3)

    def test_bad_trace(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            with open(p, 'w') as f:
                f.write(json.dumps({'prompt': [1], 'gen': [2, 3], 'top': [[[2, 0.0]]]}) + '\n')
            with self.assertRaises(ValueError):
                ng.load_traces(p)


if __name__ == '__main__':
    unittest.main()
