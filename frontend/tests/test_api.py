"""HTTP-level tests against a running front end + mock engine (tests/serve_mock.sh start).
Run: URL=http://127.0.0.1:8100 python -m unittest tests.test_api -v
/health?deep needs the front end started with FRONT_ARGS='--supervisor 127.0.0.1:8198' (SUP_PORT): the test plays the
supervisor there."""
import http.client
import json
import os
import socketserver
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

URL = os.environ.get('URL', 'http://127.0.0.1:8100')


def post(path, body, raw=False):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            data = f.read()
            return f.status, (data if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def sse(path, body):
    status, data = post(path, dict(body, stream=True), raw=True)
    assert status == 200, (status, data)
    lines = [ln[6:] for ln in data.decode().split('\n') if ln.startswith('data: ')]
    assert lines[-1] == '[DONE]', lines[-3:]
    return [json.loads(ln) for ln in lines[:-1]]


def metric(name):
    t = urllib.request.urlopen(URL + '/metrics').read().decode()
    return sum(float(ln.split()[-1]) for ln in t.splitlines() if ln.startswith(name))


CHAT = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'Tell me about harbors.'}],
        'max_tokens': 60, 'temperature': 0}
TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather',
                                           'parameters': {'type': 'object', 'properties': {
                                               'city': {'type': 'string'}, 'days': {'type': 'integer'}}}}}]


class Chat(unittest.TestCase):
    def test_stream_equals_nonstream(self):
        s, full = post('/v1/chat/completions', CHAT)
        self.assertEqual(s, 200)
        chunks = sse('/v1/chat/completions', dict(CHAT, stream_options={'include_usage': True}))
        self.assertEqual(chunks[0]['choices'][0]['delta'], {'role': 'assistant', 'content': ''})
        deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
        msg = full['choices'][0]['message']
        self.assertEqual(''.join(d.get('reasoning', '') for d in deltas), msg['reasoning'])
        self.assertNotIn('reasoning_content', msg)  # default --reasoning-fields reasoning (vLLM 0.30 parity)
        self.assertEqual(''.join(d.get('content') or '' for d in deltas), msg['content'] or '')
        self.assertEqual(chunks[-1]['choices'], [])
        self.assertEqual(chunks[-1]['usage'], {k: full['usage'][k] for k in
                                               ('prompt_tokens', 'total_tokens', 'completion_tokens',
                                                'completion_tokens_details')})
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], full['choices'][0]['finish_reason'])
        self.assertEqual(full['usage']['completion_tokens'], 60)

    def test_no_usage_chunk_without_include_usage(self):
        chunks = sse('/v1/chat/completions', CHAT)
        self.assertTrue(all('usage' not in c for c in chunks))

    def test_quality_py_shape(self):
        # quality.py-style client: model "glm53", chat_template_kwargs {"thinking": False}, seed, reads message.content
        body = {'model': 'glm53', 'messages': [{'role': 'user', 'content': "Solve. Answer as '#### <number>'."}],
                'max_tokens': 1024, 'temperature': 0, 'seed': 1234, 'chat_template_kwargs': {'thinking': False}}
        s, r = post('/v1/chat/completions', body)
        self.assertEqual(s, 200)
        self.assertTrue(r['choices'][0]['message']['content'].endswith('#### 42'))
        self.assertEqual(r['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(r['choices'][0]['stop_reason'], 154827)

    def test_tool_call_stream_equals_nonstream(self):
        body = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'weather?'}], 'tools': TOOLS,
                'max_tokens': 200}
        s, full = post('/v1/chat/completions', body)
        call = full['choices'][0]['message']['tool_calls'][0]
        self.assertEqual(json.loads(call['function']['arguments']), {'city': 'Paris', 'days': 3})
        self.assertEqual(full['choices'][0]['finish_reason'], 'tool_calls')
        chunks = sse('/v1/chat/completions', body)
        args = ''.join(tc['function'].get('arguments', '') for c in chunks for tc in
                       c['choices'][0]['delta'].get('tool_calls', []))
        self.assertEqual(args, call['function']['arguments'])
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')

    def test_prose_about_a_tool_call_is_not_a_call(self):
        # the reply writes about the call format: none of it may become a tool_calls entry, and none may be lost
        body = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'describe the call format'}],
                'tools': TOOLS, 'max_tokens': 200}
        s, r = post('/v1/chat/completions', body)
        self.assertEqual(s, 200, r)
        msg = r['choices'][0]['message']
        self.assertNotIn('tool_calls', msg)
        self.assertIn('<tool_call>get_weather', msg['content'])
        self.assertIn('in your code.', msg['content'])
        self.assertEqual(r['choices'][0]['finish_reason'], 'stop')

    def test_tool_choice_required_forces_a_call(self):
        # `required` and a named function force the opener with the engine's insert rule, so the reply is a call even
        # though the messages ask for an answer and not the weather
        for choice, name in (('required', 'get_weather'),
                             ({'type': 'function', 'function': {'name': 'get_weather'}}, 'get_weather')):
            body = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'Say hello in French.'}],
                    'tools': TOOLS, 'tool_choice': choice, 'max_tokens': 200}
            s, r = post('/v1/chat/completions', body)
            self.assertEqual(s, 200, r)
            msg = r['choices'][0]['message']
            self.assertEqual(msg['tool_calls'][0]['function']['name'], name)
            self.assertEqual(json.loads(msg['tool_calls'][0]['function']['arguments']), {'city': 'Paris', 'days': 3})
            self.assertEqual(r['choices'][0]['finish_reason'], 'tool_calls')
            # the reply's first token is the opener, so there is no think block and no reasoning
            self.assertEqual(''.join(msg.get('reasoning') or ''), '')
        # the named choice is the first token (opener) plus the name, and the opener is not in content
        s, r = post('/v1/chat/completions', {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'hi'}],
                                            'tools': TOOLS,
                                            'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}},
                                            'max_tokens': 200})
        self.assertNotIn('<tool_call>', r['choices'][0]['message'].get('content') or '')

    def test_tool_choice_errors(self):
        forced = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': TOOLS,
                  'tool_choice': 'required'}
        self.assertEqual(post('/v1/chat/completions', dict(forced, logprobs=True))[0], 400)
        self.assertEqual(post('/v1/chat/completions', dict(forced, thinking_budget=8))[0], 400)
        self.assertEqual(post('/v1/chat/completions', {'model': 'glm-5.3', 'messages': [{'role': 'user',
                                                                                         'content': 'hi'}],
                                                       'tool_choice': 'required'})[0], 400)
        self.assertEqual(post('/v1/chat/completions', dict(forced, tool_choice={
            'type': 'function', 'function': {'name': 'get_rain'}}))[0], 400)
        self.assertEqual(post('/v1/chat/completions', dict(forced, tool_choice='require'))[0], 400)

    def test_tool_result_turn_renders(self):
        body = {'model': 'glm-5.3', 'tools': TOOLS, 'max_tokens': 100, 'messages': [
            {'role': 'user', 'content': 'weather?'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                'name': 'get_weather', 'arguments': '{"city": "Paris", "days": 3}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'}]}
        s, r = post('/v1/chat/completions', body)
        self.assertEqual(s, 200, r)
        self.assertNotIn('tool_calls', r['choices'][0]['message'])

    def test_chat_logprobs(self):
        s, r = post('/v1/chat/completions', dict(CHAT, max_tokens=5, logprobs=True, top_logprobs=3))
        c = r['choices'][0]['logprobs']['content']
        self.assertEqual(len(c), 5)
        self.assertEqual(len(c[0]['top_logprobs']), 3)

    def test_errors(self):
        self.assertEqual(post('/v1/chat/completions', dict(CHAT, model='nope'))[0], 404)
        self.assertEqual(post('/v1/chat/completions', dict(CHAT, n=2))[0], 400)
        self.assertEqual(post('/v1/chat/completions', dict(CHAT, max_tokens=10 ** 7))[0], 400)
        s, e = post('/v1/chat/completions', dict(CHAT, messages='x'))
        self.assertEqual((s, e['error']['type']), (400, 'BadRequestError'))

    def test_client_disconnect_cancels(self):
        before = metric('vllm:request_success_total{engine="0",model_name="glm-5.3",finished_reason="abort"}')
        u = urllib.parse.urlparse(URL)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        conn.request('POST', '/v1/chat/completions', json.dumps(dict(CHAT, max_tokens=5000, stream=True)),
                     {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        resp.read1(2000)
        conn.sock.close()
        conn.close()
        for _ in range(50):
            time.sleep(0.1)
            if metric('vllm:request_success_total{engine="0",model_name="glm-5.3",finished_reason="abort"}') > before:
                break
        self.assertGreater(
            metric('vllm:request_success_total{engine="0",model_name="glm-5.3",finished_reason="abort"}'), before)


def reasoning_tokens(r):
    return r['usage']['completion_tokens_details']['reasoning_tokens']


def bad(path, body):
    """-> the error message of a request that must be a 400"""
    s, e = post(path, body)
    assert s == 400, (s, e)
    return e['error']['message']


class Thinking(unittest.TestCase):
    """reasoning_effort, thinking_budget and the refusals (GLM 5.3's template cannot turn thinking off)."""

    def test_reasoning_tokens_count_through_think_end(self):
        s, r = post('/v1/chat/completions', CHAT)
        n = post('/tokenize', {'model': 'glm-5.3', 'prompt': r['choices'][0]['message']['reasoning'],
                               'add_special_tokens': False})[1]['count']
        self.assertEqual(reasoning_tokens(r), n + 1)    # the mock's reasoning, then </think>

    def test_budget_closes_the_think_block_and_the_answer_follows(self):
        body = dict(CHAT, messages=[{'role': 'user', 'content': "Solve. Answer as '#### <number>'."}],
                    max_tokens=200, thinking_budget=5)
        s, r = post('/v1/chat/completions', body)
        self.assertEqual(s, 200, r)
        msg = r['choices'][0]['message']
        self.assertTrue(msg['content'].endswith('#### 42'), msg)
        self.assertEqual(reasoning_tokens(r), 6)        # 4 tokens, then the close's "\n" and </think>
        self.assertEqual(r['choices'][0]['finish_reason'], 'stop')
        chunks = sse('/v1/chat/completions', dict(body, stream_options={'include_usage': True}))
        deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
        self.assertEqual(''.join(d.get('reasoning', '') for d in deltas), msg['reasoning'])
        self.assertEqual(''.join(d.get('content') or '' for d in deltas), msg['content'])
        self.assertEqual(chunks[-1]['usage'], {k: r['usage'][k] for k in
                                               ('prompt_tokens', 'total_tokens', 'completion_tokens',
                                                'completion_tokens_details')})
        # every reply token once: the ids returned are the completion tokens
        s, r = post('/v1/chat/completions', dict(body, return_token_ids=True))
        self.assertEqual(len(r['choices'][0]['token_ids']), r['usage']['completion_tokens'])
        self.assertEqual(r['choices'][0]['token_ids'][4:7], [198, 154842, 271])   # the 5th token is the close

    def test_budget_counts_against_max_tokens(self):
        s, r = post('/v1/chat/completions', dict(CHAT, thinking_budget=5))
        self.assertEqual((r['usage']['completion_tokens'], reasoning_tokens(r)), (60, 6))
        self.assertEqual(r['choices'][0]['finish_reason'], 'length')
        self.assertTrue(r['choices'][0]['message']['content'])

    def test_budget_zero_null_or_negative_is_no_limit_without_a_default(self):
        s, plain = post('/v1/chat/completions', CHAT)
        for b in (0, None, -3):
            s, r = post('/v1/chat/completions', dict(CHAT, thinking_budget=b))
            self.assertEqual(r['choices'][0]['message'], plain['choices'][0]['message'], b)

    def test_efforts_render_the_nearest_named_level(self):
        def ids(**kw):
            return post('/tokenize', dict({'model': 'glm-5.3', 'messages': CHAT['messages']}, **kw))[1]['tokens']
        default, low, high = ids(), ids(reasoning_effort='low'), ids(reasoning_effort='high')
        self.assertEqual(ids(reasoning_effort='max'), default)
        self.assertEqual(ids(reasoning_effort='xhigh'), default)
        self.assertEqual(ids(reasoning_effort='minimal'), low)
        self.assertEqual(ids(reasoning_effort='medium'), high)
        self.assertEqual(ids(chat_template_kwargs={'reasoning_effort': 'minimal'}), low)
        self.assertEqual(ids(chat_template_kwargs={'enable_thinking': True}), default)
        self.assertNotEqual(low, default)
        self.assertNotEqual(high, default)
        s, r = post('/v1/chat/completions', dict(CHAT, reasoning_effort='medium'))
        self.assertEqual(r['usage']['prompt_tokens'], len(high))

    def test_refusals(self):
        for path in ('/v1/chat/completions', '/tokenize'):
            m = bad(path, dict(CHAT, chat_template_kwargs={'enable_thinking': False}))
            self.assertIn('thinking cannot be disabled', m)
            self.assertIn('reasoning_effort "low"', m)
            self.assertIn('"none"', bad(path, dict(CHAT, reasoning_effort='none')))
            for e in ('auto', 'LOW', 3):
                self.assertEqual(bad(path, dict(CHAT, reasoning_effort=e)),
                                 'reasoning_effort must be none, minimal, low, medium, high, xhigh or max')
        for b in (True, 1.5, 'abc', [5]):
            self.assertEqual(bad('/v1/chat/completions', dict(CHAT, thinking_budget=b)),
                             'thinking_budget must be an integer or null')
        self.assertIn('logprobs', bad('/v1/chat/completions', dict(CHAT, thinking_budget=5, logprobs=True)))

    def test_response_format(self):
        self.assertEqual(post('/v1/chat/completions', dict(CHAT, response_format={'type': 'text'}))[0], 200)
        for rf in ({'type': 'json_object'}, {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {}}}):
            for path, body in (('/v1/chat/completions', CHAT),
                               ('/v1/completions', {'model': 'glm-5.3', 'prompt': 'x', 'max_tokens': 4})):
                self.assertIn('is not supported by the gr8 front end yet', bad(path, dict(body, response_format=rf)))


class Completions(unittest.TestCase):
    def test_stop_string(self):
        s, r = post('/v1/completions', {'model': 'glm-5.3', 'prompt': 'Tell me a story.', 'max_tokens': 200,
                                        'stop': ['keeper']})
        ch = r['choices'][0]
        self.assertEqual((ch['finish_reason'], ch['stop_reason']), ('stop', 'keeper'))
        self.assertNotIn('keeper', ch['text'])
        chunks = sse('/v1/completions', {'model': 'glm-5.3', 'prompt': 'Tell me a story.', 'max_tokens': 200,
                                         'stop': ['keeper']})
        self.assertEqual(''.join(c['choices'][0]['text'] for c in chunks if c['choices']), ch['text'])

    def test_d8_kl_gen_shape(self):
        s, t = post('/tokenize', {'model': 'glm-5.3', 'add_generation_prompt': True,
                                  'messages': [{'role': 'user', 'content': 'hi'}]})
        s, r = post('/v1/completions', {'model': 'glm-5.3', 'prompt': t['tokens'], 'max_tokens': 8,
                                        'temperature': 0, 'return_tokens_as_token_ids': True, 'logprobs': 1})
        toks = [int(x.split(':')[1]) for x in r['choices'][0]['logprobs']['tokens']]
        self.assertEqual(len(toks), 8)
        self.assertEqual(r['usage']['completion_tokens'], 8)

    def test_prompt_logprobs_shape(self):
        ids = [154822, 154824, 9703, 1052, 1052, 13]
        s, r = post('/v1/completions', {'model': 'glm-5.3', 'prompt': ids, 'max_tokens': 1, 'temperature': 0,
                                        'prompt_logprobs': 20})
        pl = r['choices'][0]['prompt_logprobs']
        self.assertEqual(len(pl), len(ids))
        self.assertIsNone(pl[0])
        for j in range(1, len(ids)):
            actual = pl[j][str(ids[j])]  # vLLM: the scored token first, then the top-k
            self.assertEqual(next(iter(pl[j])), str(ids[j]))
            others = sorted(v['rank'] for k, v in pl[j].items() if k != str(ids[j]))
            if actual['rank'] <= 20:
                self.assertEqual(len(pl[j]), 20)
                self.assertEqual(others, [r for r in range(1, 21) if r != actual['rank']])
            else:
                self.assertEqual((len(pl[j]), others), (21, list(range(1, 21))))

    def test_stream_equals_nonstream(self):
        body = {'model': 'glm-5.3', 'prompt': 'Once upon a time', 'max_tokens': 40, 'temperature': 0}
        s, r = post('/v1/completions', body)
        chunks = sse('/v1/completions', dict(body, stream_options={'include_usage': True}))
        self.assertEqual(''.join(c['choices'][0]['text'] for c in chunks if c['choices']), r['choices'][0]['text'])
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 40)

    def test_default_max_tokens_16(self):
        s, r = post('/v1/completions', {'model': 'glm-5.3', 'prompt': 'x'})
        self.assertEqual(r['usage']['completion_tokens'], 16)


class Misc(unittest.TestCase):
    def test_tokenize_text_count(self):
        s, r = post('/tokenize', {'model': 'glm-5.3', 'prompt': 'hello world'})
        self.assertEqual(r['count'], len(r['tokens']))

    def test_detokenize_roundtrip(self):
        s, r = post('/tokenize', {'model': 'glm-5.3', 'prompt': 'héllo wörld 你好'})
        s, d = post('/detokenize', {'model': 'glm-5.3', 'tokens': r['tokens']})
        self.assertEqual(d['prompt'], 'héllo wörld 你好')

    def test_models_health_metrics(self):
        m = json.loads(urllib.request.urlopen(URL + '/v1/models').read())
        self.assertEqual(m['data'][0]['id'], 'glm-5.3')
        self.assertEqual(urllib.request.urlopen(URL + '/health').status, 200)
        self.assertGreaterEqual(metric('vllm:spec_decode_num_drafts_total'), 0)


class FakeSupervisor(socketserver.StreamRequestHandler):
    answer = {}

    def handle(self):
        self.server.asked.append(json.loads(self.rfile.readline()))
        self.wfile.write((json.dumps(FakeSupervisor.answer) + '\n').encode())


class DeepHealth(unittest.TestCase):
    def get(self, path):
        try:
            with urllib.request.urlopen(URL + path, timeout=30) as f:
                return f.status, json.loads(f.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_the_supervisor_decides_the_status(self):
        socketserver.TCPServer.allow_reuse_address = True
        try:
            srv = socketserver.TCPServer(('127.0.0.1', int(os.environ.get('SUP_PORT', 8198))), FakeSupervisor)
        except OSError as e:
            self.skipTest(f'cannot play the supervisor: {e}')
        srv.asked = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            FakeSupervisor.answer = {'t': 'check', 'ok': True, 'ranks': [{'rank': 0, 'ok': True}]}
            s, r = self.get('/health?deep')
            if s == 503 and 'no supervisor' in r.get('error', ''):
                self.skipTest("the front end has no --supervisor (FRONT_ARGS='--supervisor 127.0.0.1:8198')")
            self.assertEqual((s, r['ranks'][0]['ok']), (200, True))
            self.assertEqual(srv.asked[-1], {'t': 'check_all'})
            FakeSupervisor.answer = {'t': 'check', 'ok': False, 'ranks': [{'rank': 0, 'ok': False}]}
            self.assertEqual(self.get('/health?deep')[0], 503)
            self.assertEqual(urllib.request.urlopen(URL + '/health').status, 200)     # plain /health asks nobody
            self.assertEqual(len(srv.asked), 2)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == '__main__':
    unittest.main()
