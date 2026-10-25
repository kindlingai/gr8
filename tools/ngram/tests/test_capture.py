#!/usr/bin/env python3
"""capture.py against a stub server with the front end's response shapes (no GPU, no model)."""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import capture  # noqa: E402
import ngram as ng  # noqa: E402

GEN = [11, 12, 13, 11, 12, 13]


class Stub(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Stub.seen.append((self.path, body))
        if self.path == '/tokenize':
            out = {'count': 3, 'tokens': [1, 2, 3 + bool(body.get('tools'))], 'max_model_len': 1000}
        else:
            k = body['logprobs']
            n = min(body['max_tokens'], len(GEN))
            gen = GEN[:n]
            # vLLM's per-position dict: the sampled token first, then the top-k (k + 1 entries at most)
            tops = [{f'token_id:{t}': math.log(0.8), 'token_id:99': math.log(0.15), 'token_id:98': math.log(0.05)}
                    for t in gen]
            tops = [dict(list(d.items())[:k + 1]) for d in tops]
            out = {'choices': [{'index': 0, 'token_ids': gen, 'finish_reason': 'length',
                                'logprobs': {'token_logprobs': [math.log(0.8)] * n, 'top_logprobs': tops}}]}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class CaptureTest(unittest.TestCase):
    def test_capture_writes_replayable_traces(self):
        srv = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f'http://127.0.0.1:{srv.server_address[1]}'
            items = [{'workload': 'code', 'messages': [{'role': 'user', 'content': 'x'}]},
                     {'workload': 'tool', 'tools': [{'type': 'function'}], 'messages': [{'role': 'user', 'content': 'y'}]}]
            with tempfile.TemporaryDirectory() as d:
                pp, op = os.path.join(d, 'p.json'), os.path.join(d, 't.jsonl')
                with open(pp, 'w') as f:
                    json.dump(items, f)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = capture.main(['--url', url, '--prompts', pp, '--out', op, '--max-tokens', '5', '--top', '2'])
                self.assertEqual(rc, 0)
                tr = ng.load_traces(op)
        finally:
            srv.shutdown()
        self.assertEqual(len(tr), 2)
        self.assertEqual(tr[0]['prompt'], [1, 2, 3])
        self.assertEqual(tr[1]['prompt'], [1, 2, 4])            # tools reach the template
        self.assertEqual(tr[0]['gen'], GEN[:5])
        self.assertEqual([t[0] for t in tr[0]['top'][0]], [11, 99])
        self.assertAlmostEqual(math.exp(tr[0]['top'][0][0][1]), 0.8)
        comp = [b for p, b in Stub.seen if p == '/v1/completions']
        self.assertTrue(all(b['temperature'] == 0 and b['return_tokens_as_token_ids'] for b in comp))


if __name__ == '__main__':
    unittest.main()
