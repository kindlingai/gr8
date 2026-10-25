#!/usr/bin/env python3
"""ab.py against a stub server with the front end's response shapes (no GPU, no model)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ab  # noqa: E402


class Stub(BaseHTTPRequestHandler):
    seen = []
    flip = None         # a prompt length whose reply changes one token (a broken draft source)
    fail = None         # a prompt length whose request gets a 500
    short = None        # a prompt length whose reply is 10 tokens (not timed)
    steps = 20
    model = 'glm-5.3'

    def do_GET(self):
        b = json.dumps({'object': 'list', 'data': [{'id': Stub.model}]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Stub.seen.append((self.path, body))
        if self.path == '/tokenize':
            n = len(json.dumps(body['messages'])) % 7 + 3 + bool(body.get('tools'))
            out = {'count': n, 'tokens': list(range(1, n + 1))}
        else:
            p = body['prompt']
            if Stub.fail == len(p):
                self.send_error(500)
                return
            gen = [(len(p) * 31 + j) % 97 for j in range(min(body['max_tokens'], 10 if Stub.short == len(p) else 60))]
            if Stub.flip == len(p):
                gen[40] += 1
            out = {'choices': [{'index': 0, 'text': 'x', 'token_ids': gen, 'finish_reason': 'length'}],
                   'usage': {'prompt_tokens': len(p), 'completion_tokens': len(gen),
                             'step_stats': {'steps': Stub.steps, 'draft_tokens': 60, 'accepted_tokens': 40,
                                            'accepted_per_pos': [20], 'generated_tokens': len(gen),
                                            'prefill_s': 0.01, 'decode_s': 1.18}}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)


ITEMS = [{'workload': 'tool', 'tools': [{'type': 'function'}], 'messages': [{'role': 'user', 'content': 'a'}]},
         {'workload': 'json', 'messages': [{'role': 'user', 'content': 'bb'}]},
         {'workload': 'json', 'messages': [{'role': 'user', 'content': 'cccc'}]}]


class AbTest(unittest.TestCase):
    def setUp(self):
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.srv.server_address[1]}'
        self.d = tempfile.TemporaryDirectory()
        self.pp = os.path.join(self.d.name, 'p.json')
        with open(self.pp, 'w') as f:
            json.dump(ITEMS, f)
        Stub.seen, Stub.flip, Stub.fail, Stub.short, Stub.steps, Stub.model = [], None, None, None, 20, 'glm-5.3'

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.d.cleanup()

    def run_ab(self, label, *extra):
        out = os.path.join(self.d.name, label + '.json')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = ab.main(['run', '--url', self.url, '--prompts', self.pp, '--label', label, '--out', out,
                          '--max-tokens', '80', *extra])
        return rc, out

    def compare(self, a, b):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ab.main(['compare', a, b])
        return rc, buf.getvalue()

    def test_identical_runs_pass_and_report_by_workload(self):
        rc, a = self.run_ab('off')
        self.assertEqual(rc, 0)
        Stub.steps = 10
        rc, b = self.run_ab('on')
        self.assertEqual(rc, 0)
        comp = [body for p, body in Stub.seen if p == '/v1/completions']
        self.assertTrue(all(x['temperature'] == 0 and x['return_token_ids'] and x['return_step_stats']
                            and 'logprobs' not in x and not x.get('stream') for x in comp))
        with open(a) as f:
            r = json.load(f)['rows'][0]
        self.assertEqual(r['tokens'], 60)
        self.assertAlmostEqual(r['tok_s'], 59 / 1.18, places=2)
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 0, out)
        self.assertIn('outputs: 3 of 3 prompts identical', out)
        self.assertIn('| json | 2 | 2 / 2 |', out)
        self.assertIn('| all | 3 | 3 / 3 |', out)
        self.assertIn('| 2.950 | 5.900 |', out)          # decode tokens per step: 59/20 off, 59/10 on

    def test_a_changed_output_fails(self):
        _, a = self.run_ab('off')
        with open(a) as f:
            Stub.flip = json.load(f)['rows'][1]['prompt_tokens']
        _, b = self.run_ab('on')
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 1)
        self.assertIn('outputs differ at token 40', out)

    def prompt_len(self, out, i):
        with open(out) as f:
            return json.load(f)['rows'][i]['prompt_tokens']

    def test_a_failed_request_is_an_error_row_and_fails_compare(self):
        rc, a = self.run_ab('off')
        Stub.fail = self.prompt_len(a, 0)
        rc, b = self.run_ab('on')
        self.assertEqual(rc, 0)                          # the run goes on
        with open(b) as f:
            rows = json.load(f)['rows']
        self.assertEqual(len(rows), 3)
        self.assertIn('error', rows[0])
        self.assertEqual(rows[1]['tokens'], 60)
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 1)
        self.assertIn('prompt 0 (tool): request failed on on', out)

    def test_short_replies_are_counted_but_not_timed(self):
        rc, a = self.run_ab('off')
        Stub.short = self.prompt_len(a, 0)
        _, a = self.run_ab('off')
        _, b = self.run_ab('on')
        with open(a) as f:
            self.assertIsNone(json.load(f)['rows'][0]['tok_s'])
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 0, out)
        self.assertIn('| tool | 1 | 0 / 0 | - | - | - |', out)
        self.assertIn('| all | 3 | 2 / 2 |', out)

    def test_model_mismatch_fails(self):
        _, a = self.run_ab('off')
        Stub.model = 'other'
        _, b = self.run_ab('on')
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 1)
        self.assertIn('model differs', out)

    def test_prompt_sets_differ_fails(self):
        _, a = self.run_ab('off')
        _, b = self.run_ab('on')
        with open(b) as f:
            B = json.load(f)
        B['rows'] = B['rows'][:2]
        with open(b, 'w') as f:
            json.dump(B, f)
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 1)
        self.assertIn('prompt sets differ: [2]', out)

    def test_settings_mismatch_fails(self):
        _, a = self.run_ab('off')
        _, b = self.run_ab('on', '--effort', 'low')
        rc, out = self.compare(a, b)
        self.assertEqual(rc, 1)
        self.assertIn('settings differs', out)

    def test_refuses_a_prompt_in_the_traces(self):
        _, a = self.run_ab('off')
        with open(a) as f:
            n = json.load(f)['rows'][2]['prompt_tokens']
        tp = os.path.join(self.d.name, 't.jsonl')
        with open(tp, 'w') as f:
            f.write(json.dumps({'workload': 'json', 'prompt': list(range(1, n + 1)), 'gen': [], 'top': []}) + '\n')
        Stub.seen = []
        rc, _ = self.run_ab('on', '--traces', tp)
        self.assertEqual(rc, 2)
        self.assertFalse([p for p, _ in Stub.seen if p == '/v1/completions'])   # refused before any generation


class BundledSetsTest(unittest.TestCase):
    def test_ab_prompts_are_held_out_of_the_table_sets(self):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

        def load(n):
            with open(os.path.join(d, n)) as f:
                return json.load(f)
        table = load('prompts.json') + load('prompts_tool2.json')
        held = load('ab_prompts.json')
        key = (lambda i: json.dumps([i['messages'], i.get('tools')], sort_keys=True))
        self.assertFalse({key(i) for i in table} & {key(i) for i in held})
        for i in table + held:
            self.assertIn(i['workload'], ('code', 'edit', 'tool', 'json', 'prose'))
            if i.get('tools'):     # every tool call has its result, so the next turn is the model's
                names = {t['function']['name'] for t in i['tools']}
                for m in i['messages']:
                    for c in m.get('tool_calls') or []:
                        self.assertIn(c['function']['name'], names)
                        self.assertIsInstance(json.loads(c['function']['arguments']), dict)
                ids = [c['id'] for m in i['messages'] for c in m.get('tool_calls') or []]
                res = [m['tool_call_id'] for m in i['messages'] if m['role'] == 'tool']
                self.assertEqual(ids, res)
                self.assertNotEqual(i['messages'][-1]['role'], 'assistant')


if __name__ == '__main__':
    unittest.main()
