#!/usr/bin/env python3
"""Inkling-2-Small in the launcher: the model set of a safetensors checkpoint, the engine command line, and
smoketest/ink2.sh against a stand-in server that answers the way the inkling front end does (the script's requests and
jq checks hold, and each wrong answer fails its case). It tests the launcher and the script, not the model;
./gr8 ink2 smoke runs the same cases against a booted serve."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import models      # noqa: E402
import modelset    # noqa: E402

SCRIPT = os.path.join(ROOT, 'smoketest', 'ink2.sh')
NAME = 'inkling-2-small'


def safetensors(path, header, payload=b'\0' * 16):
    body = json.dumps(header).encode()
    with open(path, 'wb') as f:
        f.write(len(body).to_bytes(8, 'little') + body + payload)


def checkpoint(d, sharded=True, hidden=64):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump({'architectures': ['Inkling2ForConditionalGeneration'], 'text_config': {'hidden_size': hidden}}, f)
    if sharded:
        with open(os.path.join(d, 'model.safetensors.index.json'), 'w') as f:
            json.dump({'weight_map': {'a': 'model-00001-of-00001.safetensors'}}, f)
    else:
        safetensors(os.path.join(d, 'model.safetensors'), {'a': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}})
    for name in models.TOKENIZER_INK2:
        with open(os.path.join(d, name), 'w') as f:
            f.write('{}')
    return d


class Ink2ModelSetTests(unittest.TestCase):
    def test_the_registration(self):
        m = models.MODELS['ink2']
        self.assertIs(m, models.INK2)
        self.assertIn(m, models.SERVED)
        self.assertEqual((m.engine, m.simulator, m.tp, m.dialect), ('g6m4_ink2', 'm4sim_ink2', ('6', '4'), 'inkling'))
        self.assertEqual(m.smoke_script, 'smoketest/ink2.sh')
        self.assertIn(NAME, m.served)

    def test_a_checkpoint_directory_is_recognised(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(modelset.is_ink2(checkpoint(os.path.join(t, 'a'))))
            self.assertTrue(modelset.is_ink2(checkpoint(os.path.join(t, 'b'), sharded=False)))
            os.makedirs(os.path.join(t, 'c'))
            self.assertFalse(modelset.is_ink2(os.path.join(t, 'c')))
            self.assertFalse(modelset.is_ink2(''))

    def test_the_set_id_names_the_checkpoint_and_the_tp(self):
        with tempfile.TemporaryDirectory() as t:
            a = checkpoint(os.path.join(t, 'a'))
            b = checkpoint(os.path.join(t, 'b'))
            c = checkpoint(os.path.join(t, 'c'), hidden=128)
            sa = modelset.ink2_set(a, 6)
            self.assertEqual(sa.set['source']['kind'], 'hf-safetensors')
            self.assertEqual(sa.set['set_id'], modelset.ink2_set(b, 6).set['set_id'], 'the same files, the same id')
            self.assertNotEqual(sa.set['set_id'], modelset.ink2_set(a, 4).set['set_id'], 'another TP, another id')
            self.assertNotEqual(sa.set['set_id'], modelset.ink2_set(c, 6).set['set_id'], 'another config, another id')
            with open(os.path.join(b, 'hf_quant_config.json'), 'w') as f:
                f.write('{"quantization": {}}')
            self.assertNotEqual(sa.set['set_id'], modelset.ink2_set(b, 6).set['set_id'], 'the quant config counts')

    def test_one_weights_file_is_named_by_its_header(self):
        with tempfile.TemporaryDirectory() as t:
            a = checkpoint(os.path.join(t, 'a'), sharded=False)
            b = checkpoint(os.path.join(t, 'b'), sharded=False)
            ida = modelset.ink2_set(a, 6).set['set_id']
            safetensors(os.path.join(b, 'model.safetensors'), {'a': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}},
                        b'\1' * 16)
            self.assertEqual(ida, modelset.ink2_set(b, 6).set['set_id'], 'the header names the file, not the weights')
            safetensors(os.path.join(b, 'model.safetensors'), {'b': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}})
            self.assertNotEqual(ida, modelset.ink2_set(b, 6).set['set_id'])
            with open(os.path.join(b, 'model.safetensors'), 'wb') as f:
                f.write(b'\0\0\0')
            with self.assertRaises(modelset.ModelError):
                modelset.ink2_set(b, 6)


class Ink2EngineArgsTests(unittest.TestCase):
    def shell(self, **env):
        e = {'PATH': os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': '', 'GR8_MODEL': 'ink2', 'TP': '6', **env}
        script = '. "$1"; engine_defaults || exit; engine_env || exit; echo "$ENGINE_BIN"; engine_args 0 /packs /aux /work 7'
        return subprocess.run(['bash', '-c', script, 'test', os.path.join(ROOT, 'scripts', 'engine_args.sh')], env=e,
                              capture_output=True, text=True)

    def test_the_command_line(self):
        r = self.shell(V2_DENSE='/ckpt')
        self.assertEqual(r.returncode, 0, r.stderr)
        binary, args = r.stdout.splitlines()
        self.assertEqual(binary, 'g6m4_ink2')
        self.assertEqual(args, '--layers all --v2-dense /ckpt --top-p-cands 256 --chunk 8 --temp 0.7 --cap 262144 --graph '
                               '--mtp 0 --serve /work/s2 --s2-port 7341 --s2-sessions 4 --s2-epoch 7')

    def test_eager_extra_args_and_the_set_id(self):
        args = self.shell(V2_DENSE='/ckpt', EAGER='1', EXTRA_ARGS='--top 5', SET_ID='abc', SESS='8', CHUNK='4').stdout.splitlines()[-1].split()
        self.assertNotIn('--graph', args)
        self.assertEqual(args.count('--top'), 1)
        self.assertEqual(args[-2:], ['--set-id', 'abc'])
        self.assertEqual(args[args.index('--s2-sessions') + 1], '8')
        self.assertEqual(args[args.index('--chunk') + 1], '4')

    def test_the_drafts_of_a_round(self):
        for mtp in ('0', '3', '7'):
            with self.subTest(mtp=mtp):
                r = self.shell(V2_DENSE='/ckpt', MTP=mtp)
                self.assertEqual(r.returncode, 0, r.stderr)
                args = r.stdout.splitlines()[-1].split()
                self.assertEqual(args[args.index('--mtp') + 1], mtp)

    def test_reuse_and_kv_store(self):
        r = self.shell(V2_DENSE='/ckpt', REUSE='1', KV_STORE='ssd:/kv1,size=8G;ssd:/kv2,size=16G',
                       KV_KEEP='all', KV_SYSTEM_AFTER='3', KV_WRITE_LIMIT='2.5')
        self.assertEqual(r.returncode, 0, r.stderr)
        args = r.stdout.splitlines()[-1].split()
        self.assertIn('--reuse-prefix', args)
        stores = [args[i + 1] for i, x in enumerate(args) if x == '--kv-store']
        self.assertEqual(stores, ['ssd:/kv1,size=8G', 'ssd:/kv2,size=16G'])
        self.assertEqual(args[args.index('--kv-keep') + 1], 'all')
        self.assertEqual(args[args.index('--kv-system-after') + 1], '3')
        self.assertEqual(args[args.index('--kv-write-limit') + 1], '2.5')

    def test_what_it_refuses(self):
        for env, text in (({'TP': '3'}, 'wants 6 or 4'), ({'MTP': '8'}, 'drafts 0 to 7 tokens'), ({'SESS': '9'}, '1 to 8 sessions'),
                          ({'DFLASH': '1'}, 'has no DFLASH'), ({'CHUNK': '9'}, '1 to 8 prompt rows')):
            with self.subTest(env=env):
                r = self.shell(**env)
                self.assertEqual(r.returncode, 2, r.stdout)
                self.assertIn(text, r.stderr)


def call(name, args):
    return {'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def reply(content=None, reasoning=None, calls=None, finish='stop'):
    msg = {'role': 'assistant', 'content': content}
    if reasoning is not None:
        msg['reasoning'] = reasoning
    if calls:
        msg['tool_calls'] = calls
    return {'object': 'chat.completion', 'model': NAME,
            'choices': [{'index': 0, 'message': msg, 'finish_reason': 'tool_calls' if calls else finish}],
            'usage': {'prompt_tokens': 9, 'completion_tokens': 7, 'total_tokens': 16}}


class Fake(BaseHTTPRequestHandler):
    """lie: the cases answered wrongly."""
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
            return self.send(503 if 'health' in self.lie else 200, b'')
        if self.path == '/v1/models':
            return self.send(200, {'data': [{'id': 'other' if 'models' in self.lie else NAME, 'max_model_len': 262144}]})
        self.send(404, {})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        lie = self.lie
        kw = b.get('chat_template_kwargs') or {}
        text = b['messages'][-1]['content']
        if 'temperature' not in b:
            self.missing.append(text)
        if b.get('stream'):
            chunks = [{'choices': [{'delta': {'content': c}}]} for c in ('Par', 'is')]
            data = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + ('' if 'stream' in lie else 'data: [DONE]\n\n')
            return self.send(200, data.encode(), 'text/event-stream')
        if b.get('tools'):
            return self.send(200, reply(calls=[call('get_weather', {'city': 'Paris'})]) if 'tool' not in lie
                             else reply('It is sunny.'))
        if '19 + 23' in text:
            self.seen.append(b.get('temperature'))
            if 'smoke' in lie and len(self.seen) == 1:
                return self.send(200, reply('41'))
            # the 'drift' lie: the second greedy reply of t_greedy_sampled_greedy (the 4th request with this prompt) differs
            return self.send(200, reply(' 43\n' if 'drift' in lie and len(self.seen) == 4 else ' 42\n'))
        if kw.get('thinking') is True:
            return self.send(200, reply('391', reasoning='17*23 = 391<|end_message|>' if 'tag' in lie else '17*23 = 391'))
        if kw.get('thinking') is False or kw.get('enable_thinking') is False or b.get('reasoning_effort') == 'none':
            return self.send(200, reply('Hi!', reasoning='x' if 'off' in lie else None))
        self.send(400, {'error': {'message': 'unexpected request'}})


@unittest.skipUnless(shutil.which('curl') and shutil.which('jq'), 'smoketest/ink2.sh needs curl and jq')
class Ink2SmokeTests(unittest.TestCase):
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
        self.assertIn(f'smoketest {NAME} against', out)
        self.assertIn('8/8 passed', out)
        self.assertEqual(self.handler.missing, [], 'every request must carry its temperature')
        self.assertEqual(self.handler.seen, [0, 0, 0.7, 0], 'smoke_42 greedy, then greedy, T 0.7, greedy')

    def test_each_wrong_answer_fails_its_case(self):
        for lie, case in (('health', 'health'), ('models', 'models_lists_served_name'), ('smoke', 'smoke_42'),
                          ('tag', 'thinking_on'), ('off', 'thinking_off'), ('stream', 'stream'),
                          ('tool', 'tool_call_parsed'), ('drift', 'greedy_sampled_greedy')):
            with self.subTest(lie=lie):
                code, out = self.run_script(lie)
                self.assertEqual(code, 1, out)
                self.assertIn(f'FAIL {case}:', out)


if __name__ == '__main__':
    unittest.main()
