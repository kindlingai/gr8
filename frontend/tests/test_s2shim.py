#!/usr/bin/env python3
"""g6s2shim against tests/fake_s2_engine.py: several streams at once over the PROTOCOL.md socket, no GPU.

usage: cd frontend && python3 -m unittest tests.test_s2shim -v      (stdlib only)
"""
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from g6front import wire  # noqa: E402
import g6s2shim  # noqa: E402
from fake_s2_engine import token_at, payload_hash as fake_hash  # noqa: E402


def gen(rid, prompt=None, **kw):
    f = {'type': 'generate', 'id': rid, 'prompt': prompt or [1, 2, 3, 4, 5], 'max_tokens': 32, 'min_tokens': 0,
         'temperature': 0.0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'seed': None, 'stop_ids': [],
         'logprobs': None, 'prompt_logprobs': None}
    f.update(kw)
    return f


class Client:
    def __init__(self, sock):
        self.sock = sock
        self.frames = {}
        self.done = {}
        self.metrics = asyncio.Queue()
        self.hello = None

    async def open(self):
        self.r, self.w = await asyncio.open_unix_connection(self.sock, limit=wire.MAX_FRAME)
        self.hello = await asyncio.wait_for(wire.read_frame(self.r), 20)
        self.task = asyncio.create_task(self.reader())
        return self

    async def reader(self):
        while True:
            f = await wire.read_frame(self.r)
            if f is None:
                return
            if f['type'] == 'metrics':
                await self.metrics.put(f)
                continue
            self.frames.setdefault(f['id'], []).append((time.monotonic(), f))
            if f['type'] == 'finish':
                self.done.setdefault(f['id'], asyncio.Event()).set()

    def send(self, f):
        if f['type'] == 'generate':
            self.done.setdefault(f['id'], asyncio.Event())
        wire.write_frame(self.w, f)

    async def wait(self, rid, t=20):
        await asyncio.wait_for(self.done[rid].wait(), t)
        fs = [f for _, f in self.frames[rid]]
        return [i for f in fs if f['type'] == 'tokens' for i in f['ids']], fs[-1], fs

    def close(self):
        self.task.cancel()
        self.w.close()


class UnitTests(unittest.TestCase):
    def test_payload_hash_matches_fnv1a(self):
        self.assertEqual(g6s2shim.payload_hash(b''), 14695981039346656037)
        self.assertEqual(g6s2shim.payload_hash(b'a'), 0xaf63dc4c8601ec8c)     # published FNV-1a 64 vectors
        self.assertEqual(g6s2shim.payload_hash(b'foobar'), 0x85944171f73967e8)
        self.assertEqual(g6s2shim.payload_hash(b'n_gen 4\n1,2\n'), fake_hash(b'n_gen 4\n1,2\n'))

    def test_parse_ready(self):
        c = g6s2shim.parse_ready('G6S2 1 7 6 20000 2 4 8\n')
        self.assertEqual(c, {'max_lp': 0, 'epoch': 7, 'ranks': 6, 'pool_rows': 20000, 'mtp': 2, 'guard': 4, 'max_rows': 8,
                             'insert': False})
        self.assertEqual(g6s2shim.parse_ready('G6S2 2 7 6 20000 2 4 8 20\n')['max_lp'], 20)
        self.assertFalse(g6s2shim.parse_ready('G6S2 2 7 6 20000 2 4 8 20\n')['insert'])
        self.assertTrue(g6s2shim.parse_ready('G6S2 3 7 6 20000 2 4 8 20\n')['insert'])
        for bad in ('G6S2 2 7 6 20000 2 4 8\n', 'G6S2 1 7 6 20000 2 4 8 20\n', 'G6S2 4 7 6 20000 2 4 8 20\n',
                    'G6S2 1 7 6\n', 'ready\n', 'G6S2 1 7 6 20000 2 4 x\n'):
            with self.assertRaises(ValueError):
                g6s2shim.parse_ready(bad)

    def test_build_request(self):
        body, err = g6s2shim.build_request({'prompt': [5, 6, 7], 'max_tokens': 9, 'temperature': 0.5, 'top_p': 0.9,
                                            'seed': 3, 'min_tokens': 4, 'stop_ids': [2, 2, 8]}, 11)
        self.assertIsNone(err)
        self.assertEqual(body, b'n_gen 9 temp 0.5 seed 3 min 4 stop 2,8 top_p 0.9\n5,6,7\n')
        body, _ = g6s2shim.build_request({'prompt': [1], 'max_tokens': 3, 'min_tokens': 50}, 11)
        self.assertIn(b' min 3 stop - ', body)           # min_tokens is capped to n_gen
        self.assertIn(b' seed 1245 ', body)              # no seed: derived from the ordinal
        V = g6s2shim.VOCAB
        self.assertEqual(V, 154880)
        self.assertIsNone(g6s2shim.build_request({'prompt': [0, V - 1], 'max_tokens': 4}, 0)[1])
        for bad_id in (V, V + 1, 2**31 - 1, -1, 1.5, True, '7'):
            err = g6s2shim.build_request({'prompt': [1, bad_id], 'max_tokens': 4}, 0)[1]
            self.assertIn('prompt token id', err or '', bad_id)
        for tp in (0, 0.0, -0.1, 1.0001, 2, float('nan'), float('inf'), 'x', [1]):
            err = g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, 'top_p': tp}, 0)[1]
            self.assertIsNotNone(err, tp)                     # no rewrite of 0 to 1.0
        for tp in (None, 1.0, 1, 0.01):
            self.assertIsNone(g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, 'top_p': tp}, 0)[1], tp)
        self.assertIn(b' top_p 1.0\n', g6s2shim.build_request({'prompt': [1], 'max_tokens': 4}, 0)[0])
        self.assertIn(b' top_p 1.0 lp 0\n', g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, 'logprobs': 0}, 0)[0])
        self.assertIn(b' top_p 1.0 lp 2 plp 3\n',
                      g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, 'logprobs': 2, 'prompt_logprobs': 3}, 0)[0])
        # the system span (PROTOCOL.md system_tokens) goes last, as sys <n>, and must lie within the prompt
        self.assertIn(b' top_p 1.0 sys 2\n', g6s2shim.build_request({'prompt': [1, 2, 3], 'max_tokens': 4, 'system_tokens': 2}, 0)[0])
        self.assertEqual(g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, 'system_tokens': 5}, 0)[1],
                         'system_tokens must be 0..len(prompt)')
        for lp in (-1, 21, 1.5, True, '3'):
            for key in ('logprobs', 'prompt_logprobs'):
                self.assertIsNotNone(g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, key: lp}, 0)[1], (key, lp))
        for kw in ({'temperature': 'hot'}, {'temperature': -1}, {'seed': 'x'}, {'min_tokens': 'x'}):
            self.assertIsNotNone(g6s2shim.build_request({'prompt': [1], 'max_tokens': 4, **kw}, 0)[1], kw)
        for bad in ({'prompt': [], 'max_tokens': 4}, {'prompt': [1], 'max_tokens': 0}, {'prompt': [-1], 'max_tokens': 4},
                    {'prompt': [1], 'max_tokens': 4, 'stop_ids': [-3]}, {'prompt': [1], 'max_tokens': 4, 'top_p': 2.0},
                    {'prompt': [1], 'max_tokens': 4, 'temperature': float('nan')}):
            self.assertIsNotNone(g6s2shim.build_request(bad, 0)[1], bad)

    def test_build_request_insert(self):
        rule = {'after_tokens': 7, 'unless_emitted': 154842, 'tokens': [198, 154842, 271]}
        body, err = g6s2shim.build_request({'prompt': [1], 'max_tokens': 9, 'prompt_logprobs': 1, 'insert': [rule]}, 0)
        self.assertIsNone(err)
        self.assertIn(b' top_p 1.0 plp 1 ins 7:154842:198,154842,271\n', body)
        body, _ = g6s2shim.build_request({'prompt': [1], 'max_tokens': 9, 'insert': [dict(rule, unless_emitted=None)]}, 0)
        self.assertIn(b' ins 7:-1:198,154842,271\n', body)
        self.assertNotIn(b'ins', g6s2shim.build_request({'prompt': [1], 'max_tokens': 9, 'insert': []}, 0)[0])
        V = g6s2shim.VOCAB
        for bad in (rule, [rule, rule], [dict(rule, after_tokens=-1)], [dict(rule, after_tokens=True)],
                    [dict(rule, unless_emitted=V)], [dict(rule, tokens=[])], [dict(rule, tokens=[V])],
                    [dict(rule, tokens=[1] * 65)], [dict(rule, tokens='198')], ['x']):
            self.assertIsNotNone(g6s2shim.build_request({'prompt': [1], 'max_tokens': 9, 'insert': bad}, 0)[1], bad)
        err = g6s2shim.build_request({'prompt': [1], 'max_tokens': 9, 'logprobs': 0, 'insert': [rule]}, 0)[1]
        self.assertIn('logprobs', err)

    def test_scan_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(g6s2shim.scan_numbers(d, 'cmd'), 0)
            for n in ('cmd.0', 'cmd.1', 'cmd.5', 'cmd.2.tmp', 'req.9', 'xcmd.40'):
                open(os.path.join(d, n), 'w').close()
            self.assertEqual(g6s2shim.scan_numbers(d, 'cmd'), 6)
            self.assertEqual(g6s2shim.scan_numbers(d, 'req'), 10)
            self.assertEqual(g6s2shim.scan_numbers(d + '/none', 'req'), 0)


class S2Base(unittest.IsolatedAsyncioTestCase):
    RANKS = 1
    FAKE = ()
    SHIM = ()

    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix='s2shim-')
        self.dirs = [os.path.join(self.tmp, f'r{i}') for i in range(self.RANKS)]
        for d in self.dirs:
            os.mkdir(d)
        self.sock = os.path.join(self.tmp, 'engine.sock')
        self.procs = []
        self.clients = []
        self.start_fake()
        self.start_shim()

    def start_fake(self):
        self.fake = subprocess.Popen([sys.executable, os.path.join(HERE, 'fake_s2_engine.py'), *self.dirs, *self.FAKE],
                                     stderr=subprocess.PIPE)
        self.procs.append(self.fake)

    def start_shim(self, *extra):
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        with open(os.path.join(self.tmp, 'shim.log'), 'a') as logf:
            self.shim = subprocess.Popen([sys.executable, os.path.join(ROOT, 'g6s2shim.py'), '--dir', self.dirs[0],
                                          '--sock', self.sock, '--poll', '0.002', *self.SHIM, *extra], stderr=logf)
        self.procs.append(self.shim)
        return self.shim

    async def asyncTearDown(self):
        for c in self.clients:
            c.close()
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
            if p.stderr:
                p.stderr.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def client(self):
        c = Client(self.sock)
        for _ in range(200):
            if os.path.exists(self.sock):
                break
            await asyncio.sleep(0.05)
        await c.open()
        self.clients.append(c)
        return c

    def fake_log(self):
        with open(os.path.join(self.dirs[0], 'fake.log')) as fh:
            return fh.read()

    def expect(self, prompt, n):
        return [token_at(prompt, i) for i in range(1, n + 1)]


class ConcurrentStreams(S2Base):
    async def test_hello(self):
        c = await self.client()
        self.assertEqual(c.hello['type'], 'hello')
        self.assertEqual(c.hello['version'], 1)
        self.assertEqual(c.hello['max_model_len'], 20000 - 4)    # pool rows - guard (2 * mtp)

    async def test_four_streams_concurrent_exact_tokens(self):
        c = await self.client()
        prompts = [[10 + i, 20 + i, 30 + i] for i in range(4)]
        n = 60
        t0 = time.monotonic()
        c.send(gen('solo', prompts[0], max_tokens=n))
        ids, fin, _ = await c.wait('solo')
        single = time.monotonic() - t0
        self.assertEqual(ids, self.expect(prompts[0], n))
        self.assertEqual((fin['reason'], fin['completion_tokens'], fin['prompt_tokens']), ('length', n, 3))

        t0 = time.monotonic()
        for i, p in enumerate(prompts):
            c.send(gen(f's{i}', p, max_tokens=n))
        res = [await c.wait(f's{i}') for i in range(4)]
        wall = time.monotonic() - t0
        for i, (ids, fin, fs) in enumerate(res):
            self.assertEqual(ids, self.expect(prompts[i], n), f'stream {i} cross-talk or loss')
            self.assertEqual(fin['reason'], 'length')
            self.assertEqual(fs[-1]['type'], 'finish')
        self.assertLess(wall, 2 * single, f'4 streams {wall:.2f}s vs 1 stream {single:.2f}s: not concurrent')
        self.assertIn('RESIDENT 4', self.fake_log())
        c.send({'type': 'metrics', 'seq': 3})
        m = await asyncio.wait_for(c.metrics.get(), 5)
        self.assertEqual(m['seq'], 3)
        s = m['spec']
        self.assertGreater(s['drafts'], 0)
        self.assertEqual(s['draft_tokens'], 2 * s['drafts'])
        self.assertEqual(len(s['accepted_per_pos']), 2)
        self.assertEqual(m['requests']['finished'], {'length': 5})
        self.assertEqual(m['requests']['running'], 0)
        self.assertEqual(m['tokens'], {'prompt': 3 * 5, 'generation': 5 * n})
        self.assertEqual(m['s2']['epoch'], 7)

    async def test_frames_are_multi_token(self):
        c = await self.client()
        c.send(gen('a', max_tokens=40))
        ids, fin, fs = await c.wait('a')
        nt = sum(f['type'] == 'tokens' for f in fs)
        self.assertEqual(len(ids), 40)
        self.assertLess(nt, 40)

    async def test_cancel_resident_is_real(self):
        c = await self.client()
        ps = [[1, 1, i] for i in range(3)]
        for i, p in enumerate(ps):
            c.send(gen(f'k{i}', p, max_tokens=80))
        await asyncio.sleep(0.4)
        t0 = time.monotonic()
        c.send({'type': 'cancel', 'id': 'k1'})
        ids, fin, _ = await c.wait('k1')
        self.assertEqual(fin['reason'], 'abort')
        self.assertLess(time.monotonic() - t0, 0.3)
        self.assertEqual(fin['completion_tokens'], len(ids))
        self.assertTrue(0 < len(ids) < 80)
        self.assertRegex(self.fake_log(), r'CANCEL \d+ resident')
        for i in (0, 2):
            ids, fin, _ = await c.wait(f'k{i}')
            self.assertEqual((fin['reason'], ids), ('length', self.expect(ps[i], 80)), f'k{i} disturbed by the cancel')

    async def test_connection_drop_cancels(self):
        c = await self.client()
        c.send(gen('x', max_tokens=500))
        await asyncio.sleep(0.3)
        c.close()
        for _ in range(100):
            if 'CANCEL' in self.fake_log():
                break
            await asyncio.sleep(0.05)
        self.assertRegex(self.fake_log(), r'CANCEL \d+ resident')
        self.assertRegex(self.fake_log(), r'END \d+ abort')
        c2 = await self.client()                 # engine and shim carry on
        c2.send(gen('y', max_tokens=6))
        self.assertEqual((await c2.wait('y'))[1]['reason'], 'length')

    async def test_validation_errors_keep_engine_alive(self):
        c = await self.client()
        bad = [gen('big', max_tokens=30000), gen('empty', prompt=[1], max_tokens=0), gen('lp', logprobs=21),
               gen('plp', prompt_logprobs=21), gen('plplong', list(range(1, 1100)), prompt_logprobs=1)]
        for f in bad:
            c.send(f)
        for f in bad:
            _, fin, _ = await c.wait(f['id'])
            self.assertEqual(fin['reason'], 'error', f['id'])
        c.send(gen('neg', prompt=[1, -5], max_tokens=4))
        self.assertEqual((await c.wait('neg'))[1]['reason'], 'error')
        for rid, kw in (('oov', {'prompt': [1, 154880]}), ('oov2', {'prompt': [2**31 - 1]}), ('tp0', {'top_p': 0.0}),
                        ('tp2', {'top_p': 1.5})):
            c.send(gen(rid, max_tokens=4, **kw))
            _, fin, _ = await c.wait(rid)
            self.assertEqual(fin['reason'], 'error', rid)
            self.assertTrue(fin['message'], rid)
        c.send(gen('top', [154879, 0], max_tokens=4, top_p=0.5))   # last valid id, a real top_p
        self.assertEqual((await c.wait('top'))[1]['reason'], 'length')
        c.send(gen('ok', max_tokens=5))
        ids, fin, _ = await c.wait('ok')
        self.assertEqual((fin['reason'], len(ids)), ('length', 5))
        self.assertIsNone(self.fake.poll())
        self.assertNotIn('EPOCH FAILED', self.fake_log())
        self.assertFalse(os.path.exists(os.path.join(self.dirs[0], 'abort.s2')))

    async def test_logprobs(self):
        c = await self.client()
        self.assertEqual(c.hello['max_logprobs'], 20)
        c.send(gen('lp', [7, 8], max_tokens=30, logprobs=3))
        c.send(gen('plain', [7, 8], max_tokens=30))
        ids, fin, fs = await c.wait('lp')
        self.assertEqual((fin['reason'], ids), ('length', self.expect([7, 8], 30)))
        lps = [e for f in fs if f['type'] == 'tokens' for e in f['logprobs']]
        self.assertEqual(lps, [{'lp': -0.25, 'rank': 1, 'top': [[t + i, -0.25 - i] for i in range(3)]} for t in ids])
        ids, _, fs = await c.wait('plain')
        self.assertEqual(ids, self.expect([7, 8], 30))
        self.assertFalse(any('logprobs' in f for f in fs if f['type'] == 'tokens'))

    async def test_prompt_logprobs(self):
        c = await self.client()
        prompt = [7, 8, 9, 10, 11]
        c.send(gen('plp', prompt, max_tokens=6, prompt_logprobs=2, logprobs=1))
        ids, fin, fs = await c.wait('plp')
        self.assertEqual((fin['reason'], ids), ('length', self.expect(prompt, 6)), fin)
        self.assertEqual([f['type'] for f in fs][0], 'prompt_logprobs')
        self.assertEqual(fs[0]['lps'], [None] + [{'lp': -0.25, 'rank': 1, 'top': [[t, -0.25], [t + 1, -1.25]]} for t in prompt[1:]])
        self.assertEqual(sum(f['type'] == 'prompt_logprobs' for f in fs), 1)

    async def test_duplicate_id(self):
        c = await self.client()
        c.send(gen('d', max_tokens=40))
        c.send(gen('d', max_tokens=40))
        await asyncio.sleep(0.05)
        first = c.frames['d'][0][1]
        self.assertEqual(first['type'], 'finish')
        self.assertEqual(first['reason'], 'error')
        self.assertIn('duplicate', first['message'])
        await asyncio.sleep(1.5)
        last = c.frames['d'][-1][1]
        self.assertEqual((last['type'], last['reason'], last['completion_tokens']), ('finish', 'length', 40))

    async def test_sigterm_writes_quit(self):
        c = await self.client()
        c.send(gen('q', max_tokens=4))
        await c.wait('q')
        self.shim.send_signal(signal.SIGTERM)
        self.assertEqual(self.shim.wait(timeout=10), 0)
        self.assertTrue(os.path.exists(os.path.join(self.dirs[0], 'quit')))
        self.assertEqual(self.fake.wait(timeout=10), 0)

    async def test_abort_file_fails_requests(self):
        c = await self.client()
        c.send(gen('r', max_tokens=500))
        await asyncio.sleep(0.2)
        with open(os.path.join(self.dirs[0], 'abort.s2'), 'w') as fh:
            fh.write('boom\n')
        _, fin, _ = await c.wait('r')
        self.assertEqual(fin['reason'], 'error')
        self.assertIn('abort', fin['message'])
        c.send(gen('after', max_tokens=4))
        self.assertEqual((await c.wait('after'))[1]['reason'], 'error')

    async def test_shim_restart_continues_numbering(self):
        c = await self.client()
        for i in range(3):
            c.send(gen(f'a{i}', max_tokens=4))
        for i in range(3):
            await c.wait(f'a{i}')
        c.close()
        self.shim.kill()                                    # no quit file: the engine keeps running
        self.shim.wait()
        self.start_shim()
        c2 = await self.client()
        c2.send(gen('b', [9, 9], max_tokens=10))
        ids, fin, _ = await c2.wait('b')
        self.assertEqual((fin['reason'], ids), ('length', self.expect([9, 9], 10)))
        self.assertIsNone(self.fake.poll())
        self.assertNotIn('EPOCH FAILED', self.fake_log())
        self.assertTrue(os.path.exists(os.path.join(self.dirs[0], 'cmd.3')))


class InsertTests(S2Base):
    async def test_insert_replaces_the_token_after_the_rule(self):
        c = await self.client()
        self.assertTrue(c.hello['insert'])
        p, ins = [3, 1, 4], [198, 154842, 271]
        c.send(gen('i', p, max_tokens=20, insert=[{'after_tokens': 5, 'unless_emitted': None, 'tokens': ins}]))
        ids, fin, _ = await c.wait('i')
        self.assertEqual(ids[:5], self.expect(p, 5))
        self.assertEqual(ids[5:8], ins)
        self.assertEqual((len(ids), fin['completion_tokens'], fin['reason']), (20, 20, 'length'))
        # the unless token among the first five: no insert
        c.send(gen('u', p, max_tokens=20, insert=[{'after_tokens': 5, 'unless_emitted': self.expect(p, 3)[2], 'tokens': ins}]))
        ids, _, _ = await c.wait('u')
        self.assertEqual(ids, self.expect(p, 20))
        # an invalid rule is the request's error, and the engine lives on
        c.send(gen('bad', p, insert=[{'after_tokens': 1, 'tokens': []}]))
        _, fin, _ = await c.wait('bad')
        self.assertEqual(fin['reason'], 'error')
        c.send(gen('after', p, max_tokens=4))
        ids, _, _ = await c.wait('after')
        self.assertEqual(ids, self.expect(p, 4))


class NoInsertEngine(S2Base):
    FAKE = ('--ready-version', '2')

    async def test_an_engine_without_insert(self):
        c = await self.client()
        self.assertNotIn('insert', c.hello)
        c.send(gen('i', [1, 2], insert=[{'after_tokens': 1, 'unless_emitted': None, 'tokens': [5]}]))
        _, fin, _ = await c.wait('i')
        self.assertEqual((fin['reason'], fin['message']), ('error', 'not supported by this engine: insert'))
        c.send(gen('ok', [1, 2], max_tokens=3))
        ids, _, _ = await c.wait('ok')
        self.assertEqual(ids, self.expect([1, 2], 3))


class QueueTests(S2Base):
    FAKE = ('--sessions', '2')

    async def test_queue_depth_and_cancel_queued(self):
        c = await self.client()
        ps = [[2, 2, i] for i in range(3)]
        for i, p in enumerate(ps):
            c.send(gen(f'q{i}', p, max_tokens=60))
        await asyncio.sleep(0.25)
        c.send({'type': 'cancel', 'id': 'q2'})            # third one still waits for a slot
        ids, fin, _ = await c.wait('q2')
        self.assertEqual((fin['reason'], ids, fin['completion_tokens']), ('abort', [], 0))
        self.assertRegex(self.fake_log(), r'CANCEL \d+ queued')
        for i in (0, 1):
            ids, fin, _ = await c.wait(f'q{i}')
            self.assertEqual((fin['reason'], ids), ('length', self.expect(ps[i], 60)))
        self.assertNotIn('RESIDENT 3', self.fake_log())

    async def test_queued_request_runs_after_a_slot_frees(self):
        c = await self.client()
        ps = [[3, i] for i in range(3)]
        for i, p in enumerate(ps):
            c.send(gen(f'w{i}', p, max_tokens=20))
        for i in range(3):
            ids, fin, _ = await c.wait(f'w{i}')
            self.assertEqual((fin['reason'], ids), ('length', self.expect(ps[i], 20)))
        self.assertIn('RESIDENT 2', self.fake_log())
        self.assertNotIn('RESIDENT 3', self.fake_log())


class StopTests(S2Base):
    FAKE = ('--stop-at', '10')

    async def test_stop_and_min_tokens(self):
        c = await self.client()
        c.send(gen('s', max_tokens=64, stop_ids=[5, 6]))
        ids, fin, _ = await c.wait('s')
        self.assertEqual((fin['reason'], fin['stop_id'], len(ids), ids[-1]), ('stop', 5, 10, 5))
        c.send(gen('m', max_tokens=64, stop_ids=[5], min_tokens=20))
        ids, fin, _ = await c.wait('m')
        self.assertEqual((fin['reason'], fin['stop_id'], len(ids)), ('stop', 5, 21))
        c.send(gen('n', max_tokens=64))                    # no stop ids: runs to length
        self.assertEqual((await c.wait('n'))[1]['reason'], 'length')


class MaxLenTests(S2Base):
    SHIM = ('--max-model-len', '200')

    async def test_prompt_plus_max_tokens_over_limit(self):
        c = await self.client()
        self.assertEqual(c.hello['max_model_len'], 200)
        c.send(gen('long', prompt=list(range(150)), max_tokens=100))
        _, fin, _ = await c.wait('long')
        self.assertEqual(fin['reason'], 'error')
        self.assertIn('too long', fin['message'])
        c.send(gen('fit', prompt=list(range(100)), max_tokens=100))
        self.assertEqual((await c.wait('fit'))[1]['reason'], 'length')


class OneDir(S2Base):
    async def test_requests_commands_and_quit_stay_in_dir(self):
        c = await self.client()
        c.send(gen('t', [4, 5, 6], max_tokens=12))
        ids, fin, _ = await c.wait('t')
        self.assertEqual((fin['reason'], ids), ('length', self.expect([4, 5, 6], 12)))
        names = sorted(f for f in os.listdir(self.dirs[0]) if f.startswith(('req.', 'cmd.')))
        self.assertEqual(names, ['cmd.0', 'req.0'])
        self.shim.send_signal(signal.SIGTERM)
        self.assertEqual(self.shim.wait(timeout=10), 0)
        self.assertTrue(os.path.exists(os.path.join(self.dirs[0], 'quit')))

    def test_dir_must_be_local(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'g6s2shim.py'), '--dir', 'box2:/g6/s2'], capture_output=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn(b'local directory', r.stderr)


class SlowBoot(S2Base):
    FAKE = ('--ready-delay', '0.8')

    async def test_clients_wait_for_ready(self):
        t0 = time.monotonic()
        c = await self.client()
        self.assertGreater(time.monotonic() - t0, 0.5)    # hello only after ready.s2
        c.send(gen('z', max_tokens=4))
        self.assertEqual((await c.wait('z'))[1]['reason'], 'length')


if __name__ == '__main__':
    unittest.main()
