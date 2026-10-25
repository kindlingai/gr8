#!/usr/bin/env python3
"""End to end on one machine with no GPU: `./gr8 pseudo up` at TP=1 boots the real g6pseudo, shim and front end from a
pseudo release, answers a chat request from the rules, and stops. It needs the release builds of both binaries
(cargo build --release in frontend/pseudo-engine and frontend/g6front-rs) and GLM-5.3's tokenizer files in
GR8_PSEUDO_TOKENIZER, and skips without them.
    GR8_PSEUDO_TOKENIZER=DIR python3 -m unittest discover -s tools/launch -p 'test_pseudo_e2e.py'"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ENGINE = os.path.join(REPO, 'frontend', 'pseudo-engine', 'target', 'release', 'g6pseudo')
FRONT = os.path.join(REPO, 'frontend', 'g6front-rs', 'target', 'release', 'g6front-rs')
TOK = os.environ.get('GR8_PSEUDO_TOKENIZER', '')
TOKENIZER = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@unittest.skipUnless(os.access(ENGINE, os.X_OK) and os.access(FRONT, os.X_OK), 'needs the release builds of g6pseudo '
                     'and g6front-rs')
@unittest.skipUnless(all(os.path.isfile(os.path.join(TOK, f)) for f in TOKENIZER), 'needs GR8_PSEUDO_TOKENIZER')
class PseudoEndToEnd(unittest.TestCase):
    def setUp(self):
        # a short path: the engine socket lives under G6_HOME, and macOS caps a socket path at 104 bytes
        self.t = tempfile.mkdtemp(prefix='gr8p-', dir='/tmp')
        self.rel = os.path.join(self.t, 'rel')
        os.makedirs(self.rel)
        for f in TOKENIZER:
            os.symlink(os.path.join(TOK, f), os.path.join(self.rel, f))
        os.symlink(os.path.join(REPO, 'frontend', 'pseudo-engine', 'model', 'responses.toml'),
                   os.path.join(self.rel, 'responses.toml'))
        self.home = os.path.join(self.t, 'h')
        os.makedirs(os.path.join(self.home, 'bin'))
        os.symlink(ENGINE, os.path.join(self.home, 'bin', 'g6pseudo'))
        self.api = free_port()
        self.env = {k: v for k, v in os.environ.items() if k not in ('CAP', 'SESS', 'PEERS', 'G6_CLUSTER')}
        self.env['GR8_MODEL'] = 'pseudo'

    def tearDown(self):
        self.gr8('down')
        shutil.rmtree(self.t, ignore_errors=True)

    def gr8(self, cmd, *argv):
        return subprocess.run([sys.executable, os.path.join(HERE, 'launch.py'), cmd, '--no-config', '--g6-home', self.home,
                               *argv], env=self.env, capture_output=True, text=True, timeout=120)

    def test_up_chat_down(self):
        r = self.gr8('up', '--peers', '127.0.0.1', '--model', self.rel, '--port', str(self.api),
                     '--control-port', str(free_port()), '--py', sys.executable, '--front-bin', FRONT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f'serving: http://127.0.0.1:{self.api}/v1', r.stdout)
        body = json.dumps({'model': 'glm-5.3', 'max_tokens': 200,
                           'messages': [{'role': 'user', 'content': 'What is the capital of Australia?'}]}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{self.api}/v1/chat/completions', body,
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            msg = json.load(resp)['choices'][0]['message']
        self.assertIn('Canberra', msg['content'])
        self.assertTrue(msg.get('reasoning_content') or msg.get('reasoning'))
        self.assertIn('stopped', self.gr8('down').stdout)


if __name__ == '__main__':
    unittest.main()
