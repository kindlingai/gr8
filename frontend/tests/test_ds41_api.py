"""DeepSeek HTTP regressions against a controlled PROTOCOL.md engine (no GPU).

Build g6front-rs first, then run: python3 -m unittest discover -s frontend/tests -p test_ds41_api.py
Uses a tiny byte tokenizer by default. GR8_TEST_DS41_TOKENIZER_DIR runs the same tests on the real tokenizer.
The mock deliberately withholds Finish until Cancel, and can hold that acknowledgement behind an event.
"""
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / 'frontend/g6front-rs/target/release/g6front-rs'
TOOLS = [{'type': 'function', 'function': {'name': 'f', 'parameters': {
    'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}}]
NAMED = {'type': 'function', 'function': {'name': 'f'}}
OPENER = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n'
TAIL = '<｜DSML｜ parameter name="text" string="true">ok</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'


def tiny_tokenizer(path):
    # A byte BPE with no merges: synthetic prompt/insert wiring coverage, not model-output quality.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs.copy()
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    vocab = {chr(c): i for i, c in enumerate(cs)}
    specials = ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>',
                '<｜System｜>', '<think>', '</think>']
    added = []
    for text in specials:
        i = len(vocab)
        vocab[text] = i
        added.append(dict(id=i, content=text, single_word=False, lstrip=False, rstrip=False,
                          normalized=False, special=text not in ('<think>', '</think>')))
    tk = dict(version='1.0', truncation=None, padding=None, added_tokens=added, normalizer=None,
              pre_tokenizer=dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=True),
              post_processor=None,
              decoder=dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=True),
              model=dict(type='BPE', vocab=vocab, merges=[], dropout=None, unk_token=None,
                         continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False,
                         byte_fallback=False, ignore_merges=False))
    (path / 'tokenizer.json').write_text(json.dumps(tk))
    (path / 'tokenizer_config.json').write_text(json.dumps({'eos_token': specials[1]}))
    (path / 'config.json').write_text(json.dumps({'architectures': ['DeepseekV41ForCausalLM'],
                                                'eos_token_id': vocab[specials[1]]}))


class MockEngine:
    def __init__(self, path):
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(str(path))
        self.listener.listen(1)
        self.requests = queue.Queue()
        self.cancelled = threading.Event()
        self.release_finish = threading.Event()
        self.release_finish.set()
        self.finished = threading.Event()
        self.tokens = []
        self.finish_on_tokens = False
        self.busy = None          # Retry-After seconds: refuse every request as the busy guard does
        self.hold_first = False   # send nothing for a request until its cancel
        self.conn = None
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def send(self, obj):
        b = json.dumps(obj).encode()
        self.conn.sendall(struct.pack('>I', len(b)) + b)

    def run(self):
        try:
            self.conn, _ = self.listener.accept()
            self.send(dict(type='hello', version=1, max_model_len=1048576, insert=True))
            with self.conn.makefile('rb') as f:
                while hdr := f.read(4):
                    msg = json.loads(f.read(struct.unpack('>I', hdr)[0]))
                    if msg['type'] == 'generate':
                        self.requests.put(msg)
                        if self.busy is not None:
                            self.send(dict(type='finish', id=msg['id'], reason='error', message='busy',
                                           status=429, retry_after=self.busy))
                            continue
                        if self.hold_first:
                            continue
                        ids = msg.get('insert', [{}])[0].get('tokens', []) + self.tokens
                        self.send(dict(type='tokens', id=msg['id'], ids=ids))
                        if self.finish_on_tokens:
                            self.send(dict(type='finish', id=msg['id'], reason='stop', cached_tokens=200))
                            self.finished.set()
                    elif msg['type'] == 'cancel':
                        self.cancelled.set()
                        if not self.release_finish.wait(10):
                            raise AssertionError('test did not release the engine Finish')
                        self.send(dict(type='finish', id=msg['id'], reason='abort', cached_tokens=200,
                                       completion_tokens=len(self.tokens),
                                       spec=dict(drafts=1, draft_tokens=2, accepted=1, accepted_per_pos=[1]),
                                       timing=dict(prefill_s=0.5, decode_s=0.25)))
                        self.finished.set()
        except (BrokenPipeError, ConnectionResetError):
            pass  # front end closed during cleanup / the disconnect test
        except Exception as e:
            self.error = e

    def close(self):
        self.release_finish.set()
        if self.conn:
            self.conn.close()
        self.listener.close()
        self.thread.join(timeout=5)
        if self.error:
            raise self.error


class Front:
    def __init__(self, *flags):
        self.tmp = tempfile.TemporaryDirectory(prefix='gr8-dsapi-')
        self.home = Path(self.tmp.name)
        model = os.environ.get('GR8_TEST_DS41_TOKENIZER_DIR')
        if not model:
            tiny_tokenizer(self.home)
            model = str(self.home)
        self.engine = MockEngine(self.home / 'e.sock')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{self.port}'
        self.log = open(self.home / 'front.log', 'w+')
        self.proc = subprocess.Popen([str(BIN), '--model-dir', model, '--engine-sock', str(self.home / 'e.sock'),
                                      '--host', '127.0.0.1', '--port', str(self.port), '--dialect', 'ds41',
                                      '--default-reasoning-effort', 'low', '--enable-prompt-tokens-details', *flags],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                if self.proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(self.url + '/health', timeout=0.2) as f:
                        if json.load(f)['ready']:
                            return
                except (OSError, urllib.error.HTTPError):
                    pass
                time.sleep(0.02)
            self.log.seek(0)
            raise AssertionError(self.log.read())
        except BaseException:
            self.close()
            raise

    def close(self):
        self.engine.release_finish.set()
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.engine.close()
        self.log.close()
        self.tmp.cleanup()

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as f:
            data = f.read()
        if body.get('stream'):
            lines = [ln[6:] for ln in data.decode().splitlines() if ln.startswith('data: ')]
            assert lines[-1] == '[DONE]', lines
            return [json.loads(ln) for ln in lines[:-1]]
        return json.loads(data)

    def encode(self, text):
        return self.post('/tokenize', {'prompt': text, 'add_special_tokens': False})['tokens']

    def decode(self, ids):
        return self.post('/detokenize', {'tokens': ids})['prompt']

    def reply(self, choice=NAMED, thinking=False):
        if choice == 'required':
            text = ' name="f">\n' + TAIL
        elif choice == NAMED:
            text = TAIL
        else:
            text = ('</think>' if thinking else '') + OPENER + TAIL
        self.engine.tokens = self.encode(text)
        self.engine.finish_on_tokens = choice is None
        self.engine.cancelled.clear()
        self.engine.finished.clear()


@unittest.skipUnless(BIN.exists(), 'build frontend/g6front-rs in release mode first')
class Ds41Api(unittest.TestCase):
    def front(self, *args):
        f = Front(*args)
        self.addCleanup(f.close)
        return f

    def body(self, choice=NAMED, **kw):
        return dict(messages=[{'role': 'user', 'content': 'call f'}], tools=TOOLS,
                    tool_choice=choice, max_tokens=1024, temperature=0, **kw)

    def assert_prompt(self, front, body, thinking, effort=50):
        front.reply(body.get('tool_choice'), thinking=thinking)
        response = front.post('/v1/chat/completions', body)
        req = front.engine.requests.get(timeout=2)
        prompt = front.decode(req['prompt'])
        self.assertTrue(prompt.endswith('<｜Assistant｜>' + ('<think>' if thinking else '</think>')), prompt[-100:])
        self.assertEqual(f'Reasoning Effort: {effort}' in prompt, thinking)
        if body.get('tool_choice') in ('required', NAMED):
            insert = front.decode(req['insert'][0]['tokens'])
            self.assertTrue(insert.startswith(('</think>' if thinking else '') + '\n\n<｜DSML｜ calls>'), insert)
            self.assertEqual('</think>' in insert, thinking)
        else:
            self.assertNotIn('insert', req)
        tokens = front.post('/tokenize', body)['tokens']
        self.assertEqual(tokens, req['prompt'])
        usage = response['usage']
        self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], 200)
        self.assertEqual(usage['completion_tokens_details']['reasoning_tokens'], 1 if thinking else 0)
        return response, req, prompt

    def test_forced_calls_native_chat_mode_and_prefix_continuity(self):
        f = self.front()
        for choice in ('required', NAMED):
            with self.subTest(choice=choice):
                body = self.body(choice)
                response, req, _ = self.assert_prompt(f, body, False)
                generated = req['insert'][0]['tokens'] + f.engine.tokens
                msg = response['choices'][0]['message']
                self.assertEqual(json.loads(msg['tool_calls'][0]['function']['arguments']), {'text': 'ok'})
                body['messages'] += [msg, {'role': 'tool', 'tool_call_id': msg['tool_calls'][0]['id'], 'content': 'ok'},
                                     {'role': 'user', 'content': 'again'}]
                _, req2, _ = self.assert_prompt(f, body, False)
                previous = req['prompt'] + generated
                self.assertEqual(req2['prompt'][:len(previous)], previous)

    def test_mixed_history_explicit_thinking_auto_and_plain_chat(self):
        f = self.front()
        for key in ('reasoning_content', 'reasoning'):
            body = self.body()
            body['messages'] = [{'role': 'user', 'content': 'first'},
                                {'role': 'assistant', 'content': 'ok', key: 'prior thought'}, *body['messages']]
            _, _, prompt = self.assert_prompt(f, body, True)
            self.assertIn('<think>prior thought</think>', prompt)
        for kw, effort in [({'reasoning_effort': 'high'}, 75),
                           ({'chat_template_kwargs': {'thinking': True}}, 50),
                           ({'chat_template_kwargs': {'enable_thinking': True}}, 50)]:
            self.assert_prompt(f, self.body(**kw), True, effort)
        self.assert_prompt(f, self.body('auto'), True)
        body = self.body('auto')
        del body['tools'], body['tool_choice']
        self.assert_prompt(f, body, True)

    def test_continue_final_message_and_tokenize_choice_validation(self):
        f = self.front()
        body = self.body(add_generation_prompt=False, continue_final_message=True)
        body['messages'].append({'role': 'assistant', 'content': 'partial', 'reasoning_content': 'why'})
        forced = f.post('/tokenize', body)['tokens']
        auto = f.post('/tokenize', dict(body, tool_choice='auto'))['tokens']
        self.assertEqual(forced, auto)
        self.assertIn('Reasoning Effort: 50', f.decode(forced))
        f.reply()
        f.post('/v1/chat/completions', body)
        self.assertEqual(f.engine.requests.get(timeout=2)['prompt'], forced)
        for choice in ('bogus', {'type': 'function', 'function': {'name': 'missing'}}):
            with self.assertRaises(urllib.error.HTTPError) as e:
                f.post('/tokenize', self.body(choice))
            self.assertEqual(e.exception.code, 400)

    def test_rollback_flag_keeps_thinking_and_explicit_off_still_wins(self):
        f = self.front('--forced-tool-thinking', '1')
        for choice in ('required', NAMED):
            self.assert_prompt(f, self.body(choice), True)
        self.assert_prompt(f, self.body(reasoning_effort='none'), False)
        f0 = self.front('--forced-tool-thinking', '0')
        self.assert_prompt(f0, self.body(), False)
        bad = subprocess.run([str(BIN), '--forced-tool-thinking', 'yes'], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn('--forced-tool-thinking: want 0 or 1', bad.stderr)

    def test_server_default_none_boots_and_requests_can_enable_thinking(self):
        f = self.front('--default-reasoning-effort', 'none')
        self.assert_prompt(f, self.body('auto'), False)
        self.assert_prompt(f, self.body(reasoning_effort='low'), True)
        self.assert_prompt(f, self.body(chat_template_kwargs={'thinking': True}), True, 75)
        self.assert_prompt(f, self.body(reasoning_effort='low', chat_template_kwargs={'thinking': False}), False)

    def test_closed_tool_block_reports_cache_hits_streaming_and_nonstreaming(self):
        f = self.front()
        for stream in (False, True):
            for stats in (False, True):
                with self.subTest(stream=stream, stats=stats):
                    f.reply()
                    body = self.body(stream=stream, return_step_stats=stats)
                    if stream:
                        body['stream_options'] = {'include_usage': True}
                    r = f.post('/v1/chat/completions', body)
                    usage = r[-1]['usage'] if stream else r['usage']
                    self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], 200)
                    self.assertEqual(usage['completion_tokens_details']['reasoning_tokens'], 0)
                    self.assertTrue(f.engine.finished.wait(2))

    def test_streams_without_usage_wait_for_finish_before_done(self):
        f = self.front()
        for path in ('/v1/chat/completions', '/v1/completions'):
            with self.subTest(path=path):
                if path.endswith('/chat/completions'):
                    f.reply()
                    body = self.body(stream=True)
                else:
                    f.engine.tokens = f.encode('aSTOPb')
                    body = dict(prompt='x', max_tokens=1024, stop=['STOP'], stream=True)
                f.engine.cancelled.clear()
                f.engine.finished.clear()
                f.engine.release_finish.clear()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    response = pool.submit(f.post, path, body)
                    try:
                        self.assertTrue(f.engine.cancelled.wait(5), 'frontend must cancel the early stop')
                        with self.assertRaises(concurrent.futures.TimeoutError):
                            response.result(timeout=0.1)
                    finally:
                        f.engine.release_finish.set()
                    chunks = response.result(timeout=5)
                    self.assertTrue(f.engine.finished.wait(2))
                    self.assertFalse(any(c.get('usage') for c in chunks))

    def test_disconnected_stream_does_not_wait_for_finish(self):
        f = self.front()
        # No tool closer: the generation loop waits for another engine frame until the HTTP client leaves.
        f.engine.tokens = f.encode('a')
        f.engine.release_finish.clear()
        conn = http.client.HTTPConnection('127.0.0.1', f.port, timeout=5)
        conn.request('POST', '/v1/completions', json.dumps(dict(prompt='x', max_tokens=1024, stream=True)),
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(response.readline().startswith(b'data: '))
        response.close()
        conn.close()
        self.assertTrue(f.engine.cancelled.wait(5))
        self.assertFalse(f.engine.finished.is_set())
        f.engine.release_finish.set()

    def test_a_busy_refusal_is_http_429_with_retry_after(self):
        f = self.front()
        f.engine.busy = 7
        for path, body in (('/v1/chat/completions', dict(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=8)),
                           ('/v1/completions', dict(prompt='x', max_tokens=8))):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    req = urllib.request.Request(f.url + path, json.dumps(dict(body, stream=stream)).encode(),
                                                 {'Content-Type': 'application/json'})
                    with self.assertRaises(urllib.error.HTTPError) as e:
                        urllib.request.urlopen(req, timeout=10)
                    self.assertEqual((e.exception.code, e.exception.headers['Retry-After']), (429, '7'))
                    self.assertEqual(json.loads(e.exception.read())['error']['type'], 'RateLimitError')

    def test_a_stream_dropped_before_its_first_token_cancels_the_engine_request(self):
        # The stream's response waits for the engine's first frame. A client that leaves before it still cancels.
        f = self.front()
        f.engine.hold_first = True
        body = json.dumps(dict(prompt='x', max_tokens=8, stream=True)).encode()
        with socket.create_connection(('127.0.0.1', f.port), timeout=5) as s:
            s.sendall(b'POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
                      b'Content-Length: %d\r\n\r\n' % len(body) + body)
            f.engine.requests.get(timeout=5)
        self.assertTrue(f.engine.cancelled.wait(5))


@unittest.skipUnless(BIN.exists(), 'build frontend/g6front-rs in release mode first')
class Hold(unittest.TestCase):
    """/internal/hold: a scale's cutover holds new requests until the new engine serves (tools/launch `scale`)."""

    def hold(self, f, seconds=None):
        q = '' if seconds is None else f'?seconds={seconds}'
        req = urllib.request.Request(f.url + '/internal/hold' + q, method='GET' if seconds is None else 'POST')
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)

    def stop_engine(self, f):
        f.engine.conn.shutdown(socket.SHUT_RDWR)   # the front end sees the engine go
        f.engine.close()
        os.unlink(f.home / 'e.sock')

    def chat(self, f, out):
        body = json.dumps(dict(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=8)).encode()
        req = urllib.request.Request(f.url + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out.append((r.status, None))
        except urllib.error.HTTPError as e:
            out.append((e.code, e.headers['Retry-After']))

    def test_a_request_waits_through_the_cutover(self):
        f = Front()
        self.addCleanup(f.close)
        self.assertEqual(self.hold(f, 30), {'hold': True, 'running': 0, 'waiting': 0})
        self.stop_engine(f)
        out = []
        t = threading.Thread(target=self.chat, args=(f, out))
        t.start()
        for _ in range(100):
            if self.hold(f)['waiting'] == 1:
                break
            time.sleep(0.05)
        self.assertEqual(self.hold(f)['waiting'], 1)
        f.engine = MockEngine(f.home / 'e.sock')
        f.engine.tokens = f.encode('ok')
        f.engine.finish_on_tokens = True
        time.sleep(0.6)       # the front end reconnects (every 0.25 s under a hold) but the request still waits
        self.assertEqual(out, [])
        self.assertEqual(self.hold(f, 0)['hold'], False)
        t.join(10)
        self.assertEqual(out, [(200, None)])

    def test_a_hold_that_runs_out_answers_503_with_retry_after(self):
        f = Front()
        self.addCleanup(f.close)
        self.stop_engine(f)
        self.hold(f, 1)
        out = []
        t0 = time.monotonic()
        self.chat(f, out)
        self.assertGreater(time.monotonic() - t0, 0.5)
        self.assertEqual(out, [(503, '5')])
        f.engine.close = lambda: None   # already closed


if __name__ == '__main__':
    unittest.main()
