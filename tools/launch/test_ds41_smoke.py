#!/usr/bin/env python3
"""smoketest/ds41.sh against a stand-in server that answers the way the ds41 front end does (a reply shaped like
frontend/g6front-rs/src/api.rs writes it): the script's requests and jq checks hold, and each wrong answer fails its case.
It tests the script, not the model; ./gr8 ds41 smoke runs the same cases against a booted serve."""
import json
import os
import shutil
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'smoketest', 'ds41.sh')
NAME = 'deepseek-v4.1-flash'


def call(name, args):
    return {'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def reply(content=None, reasoning=None, calls=None, ids=None, finish='stop'):
    msg = {'role': 'assistant', 'content': content}
    if reasoning is not None:
        msg['reasoning'] = reasoning
    if calls:
        msg['tool_calls'] = calls
    return {'object': 'chat.completion', 'model': NAME,
            'choices': [{'index': 0, 'message': msg, 'finish_reason': 'tool_calls' if calls else finish,
                         'token_ids': ids}],
            'usage': {'prompt_tokens': 9, 'completion_tokens': 7, 'total_tokens': 16}}


class Fake(BaseHTTPRequestHandler):
    """lie: {case: wrong answer}. A case in lie is answered wrongly."""
    lie = ()

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            return self.send(200, b'')
        if self.path == '/v1/models':
            return self.send(200, {'data': [{'id': NAME, 'max_model_len': 262144}]})
        self.send(404, {})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        lie = self.lie
        kw = b.get('chat_template_kwargs') or {}
        text = b['messages'][-1]['content']
        if b.get('stream'):
            chunks = [{'choices': [{'delta': {'content': c}}]} for c in ('Par', 'is')]
            data = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + ('' if 'stream' in lie else 'data: [DONE]\n\n')
            return self.send(200, data.encode(), 'text/event-stream')
        if 'effort' in lie and b.get('reasoning_effort') == 101:
            return self.send(200, reply('hi'))
        if b.get('reasoning_effort') == 101:
            return self.send(400, {'error': {'message': 'reasoning_effort must be an integer from 1 to 100, or none, ...',
                                             'param': 'reasoning_effort'}})
        if b.get('tools') and b.get('tool_choice') in ('required', {'type': 'function', 'function': {'name': 'get_weather'}}):
            return self.send(200, reply(calls=[call('get_weather', {'city': 'Paris'})],
                                        reasoning='thinking' if 'forced' in lie else None))
        if b.get('tools'):
            return self.send(200, reply(calls=[call('get_weather', {'city': 'Paris'})]) if 'tool' not in lie
                             else reply('It is sunny.'))
        if 'temperature' not in b:
            self.missing.append(text)
        if '19 + 23' in text:
            self.seen.append(b.get('temperature'))
            # the 'drift' lie: the second greedy reply of t_greedy_sampled_greedy (the 4th request with this prompt) differs
            return self.send(200, reply(' 42\n' if not ('smoke' in lie and len(self.seen) == 1) else '41') if not ('drift' in lie and len(self.seen) == 4)
                             else reply(' 43\n'))
        if kw.get('thinking') is True:
            ids = [5, 6, 128822, 7] if 'close' not in lie else [5, 6, 7]
            return self.send(200, reply('391', reasoning='17*23 = 391', ids=ids))
        if kw.get('thinking') is False or kw.get('enable_thinking') is False or b.get('reasoning_effort') == 'none':
            return self.send(200, reply('Hi!', reasoning='x' if 'off' in lie else None))
        self.send(400, {'error': {'message': 'unexpected request'}})


@unittest.skipUnless(shutil.which('curl') and shutil.which('jq'), 'smoketest/ds41.sh needs curl and jq')
class Ds41SmokeTests(unittest.TestCase):
    def run_script(self, *lie):
        handler = type('H', (Fake,), {'lie': lie, 'seen': [], 'missing': []})
        self.handler = handler
        srv = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        r = subprocess.run(['bash', SCRIPT, f'http://127.0.0.1:{srv.server_port}'], capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout

    def test_a_server_that_answers_as_the_front_end_does_passes_every_case(self):
        code, out = self.run_script()
        self.assertEqual(code, 0, out)
        self.assertIn('smoketest deepseek-v4.1-flash against', out)
        self.assertIn('11/11 passed', out)           # ten cases, the forced one twice
        self.assertEqual(self.handler.missing, [], 'every request must carry its temperature')
        self.assertEqual(self.handler.seen, [0, 0, 0.7, 0], 'smoke_42 greedy, then greedy, T 0.7, greedy')

    def test_each_wrong_answer_fails_its_case(self):
        for lie, case in (('smoke', 'smoke_42'), ('close', 'thinking_on'), ('off', 'thinking_off'),
                          ('effort', 'reasoning_effort_range'), ('stream', 'stream'), ('tool', 'tool_call_parsed'),
                          ('forced', 'tool_choice_forced'), ('drift', 'greedy_sampled_greedy')):
            with self.subTest(lie=lie):
                code, out = self.run_script(lie)
                self.assertEqual(code, 1, out)
                self.assertIn(f'FAIL {case}:', out)


if __name__ == '__main__':
    unittest.main()
