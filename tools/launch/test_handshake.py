#!/usr/bin/env python3
"""The pre-engine handshake over real TCP on localhost: a head and several fake workers, agreeing or not.
    python3 -m unittest discover -s tools/launch"""
import json
import os
import socket
import socketserver
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handshake as H  # noqa: E402

LAN = '127.0.0.1'     # every fake box is this machine, so the member list repeats one address
POLL = 0.05


def hello(rank, **kw):
    opts = {'tp': '4', 'CAP': '393216', 'engine.args': '--cap 393216 --s2-sessions 8'}
    opts.update(kw.pop('opts', {}))
    h = {'t': 'hello', 'proto': H.PROTO, 'rank': rank, 'lan': LAN, 'members': [LAN] * 4,
         'rails': [f'10.1.0.{rank + 1}', f'10.2.0.{rank + 1}'], 'home': f'/home{rank}', 'opts': opts,
         'opts_digest': json.dumps(opts, sort_keys=True), 'set_id': 'a' * 64, 'set_source': 'packs',
         'engine_build': 'b' * 64, 'fails': [], 'version': 'test'}
    h.update(kw)
    return h


class Fixture:
    """A head on a free localhost port, answering the way the supervisor's control server does: `retry` until the
    head has finished its own checks (ready), then Head.rpc."""

    def __init__(self, own=None, ready=True, drop_s=H.DROP_S):
        self.head = H.Head(own or hello(0), drop_s=drop_s)
        self.ready = threading.Event()
        if ready:
            self.ready.set()
        fx = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                m = json.loads(self.rfile.readline())
                v = fx.head.rpc(m, self.client_address[0]) if fx.ready.is_set() else \
                    {'t': 'retry', 'detail': 'the head has not finished its checks'}
                self.wfile.write((json.dumps(v) + '\n').encode())
        self.srv = socketserver.ThreadingTCPServer((LAN, 0), Handler)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def send_all(self, hellos, timeout=20, drop_s=H.DROP_S):
        out = {}

        def one(h):
            out[h['rank']] = H.send_hello(LAN, self.port, h, None, timeout, lambda s: None, poll_s=POLL, drop_s=drop_s)
        ts = [threading.Thread(target=one, args=(h,)) for h in hellos]
        for t in ts:
            t.start()
        return ts, out

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


OK = {'t': 'ok', 'epoch': 123, 'ips': ['10.1.0.1', '10.1.0.2', '10.1.0.3', '10.1.0.4']}


class HandshakeTests(unittest.TestCase):
    def run_case(self, workers, own=None, ok=None):
        f = Fixture(own)
        try:
            ts, out = f.send_all(workers)
            if f.head.wait_all(20) and ok:
                f.head.decide(ok)
            for t in ts:
                t.join(20)
            return f.head.verdict, out
        finally:
            f.close()

    def test_all_agree(self):
        v, out = self.run_case([hello(1), hello(2), hello(3)], ok=OK)
        self.assertEqual(v, OK)
        self.assertEqual([out[r]['epoch'] for r in (1, 2, 3)], [123] * 3)

    def assert_refused(self, workers, field, own=None, values=None):
        v, out = self.run_case(workers, own)
        self.assertEqual(v['t'], 'refused')
        self.assertEqual(v['field'], field)
        for r in out:
            self.assertEqual(out[r], v, f'rank {r} got another verdict')
        if values:
            self.assertEqual(v['values'], values)
        return v

    def test_option_mismatch_names_the_key_and_every_value(self):
        v = self.assert_refused([hello(1), hello(2, opts={'CAP': '262144'}), hello(3)], 'option CAP')
        self.assertEqual(v['values']['rank 0 (127.0.0.1)'], '393216')
        self.assertEqual(v['values']['rank 2 (127.0.0.1)'], '262144')
        self.assertIn('option CAP', H.describe(v))

    def test_engine_args_mismatch_shows_the_differing_words(self):
        v = self.assert_refused([hello(1, opts={'engine.args': '--cap 393216 --s2-sessions 4'})], 'option engine.args')
        self.assertEqual(v['values'], {'rank 0 (127.0.0.1)': '8', 'rank 1 (127.0.0.1)': '4'})

    def test_other_fields(self):
        self.assert_refused([hello(1), hello(2, engine_build='c' * 64), hello(3)], 'engine build (sha256 of bin/g6m4 and bin/lib)')
        v = self.assert_refused([hello(1, set_id='d' * 64, set_source='/nfs/other'), hello(2), hello(3)], 'model set id')
        self.assertIn('/nfs/other', v['values']['rank 1 (127.0.0.1)'])
        self.assert_refused([hello(1), hello(1), hello(3)], 'rank')
        self.assert_refused([hello(1, proto=99), hello(2), hello(3)], 'handshake protocol')
        self.assert_refused([hello(1, members=[LAN] * 3), hello(2), hello(3)], 'member list')
        self.assert_refused([hello(1), hello(2, fails=['missing /c/rank2.pack']), hello(3)], 'checks',
                            values={'rank 2 (127.0.0.1)': 'missing /c/rank2.pack'})
        self.assert_refused([hello(7)], 'rank')
        self.assert_refused([hello(1, lan='127.0.0.9')], 'rank')

    def test_every_failing_box_is_named(self):
        v = self.assert_refused([hello(1), hello(2, fails=['no /dev/infiniband']), hello(3, fails=['missing a.pack', 'b'])],
                                'checks', own=hello(0, fails=['an engine is already running (pid 7)']))
        self.assertEqual(v['values'], {'rank 0 (127.0.0.1)': 'an engine is already running (pid 7)',
                                       'rank 2 (127.0.0.1)': 'no /dev/infiniband', 'rank 3 (127.0.0.1)': 'missing a.pack; b'})
        self.assertEqual(v['detail'], '3 of 4 boxes failed their checks')

    def test_failed_checks_win_over_a_missing_box(self):
        f = Fixture()
        try:
            ts, out = f.send_all([hello(1, fails=['no /dev/infiniband']), hello(2)])
            time.sleep(0.5)
            self.assertFalse(f.head.wait_all(1.0))
            ts[0].join(20)
            self.assertEqual(out[1]['field'], 'checks')
            self.assertIn('no hello within 1s from rank 3 (127.0.0.1)', out[1]['detail'])
        finally:
            f.close()

    def test_crossed_rails(self):
        self.assert_refused([hello(1, rails=['10.2.0.2', '10.1.0.2']), hello(2), hello(3)], 'rails')

    def test_head_fails_its_checks(self):
        self.assert_refused([hello(1), hello(2), hello(3)], 'checks', own=hello(0, fails=['no /dev/infiniband']))

    def test_late_hello_gets_the_same_refusal(self):
        f = Fixture()
        try:
            ts, out = f.send_all([hello(1, opts={'CAP': '4'})])
            ts[0].join(20)
            ts2, out2 = f.send_all([hello(2)])
            ts2[0].join(20)
            self.assertEqual(out[1]['field'], 'option CAP')
            self.assertEqual(out2[2], out[1])
        finally:
            f.close()

    def test_hello_after_the_boot_is_refused(self):
        f = Fixture()
        try:
            ts, out = f.send_all([hello(1), hello(2), hello(3)])
            self.assertTrue(f.head.wait_all(20))
            f.head.decide(OK)
            for t in ts:
                t.join(20)
            ts, late = f.send_all([hello(2)])
            ts[0].join(20)
            self.assertEqual(late[2]['field'], 'booted')
            self.assertIn('booted at epoch 123: ./gr8 glm53 down first', late[2]['detail'])
            self.assertEqual(f.head.verdict, OK)        # the cluster's verdict stands
        finally:
            f.close()

    def test_join_timeout_runs_from_the_first_hello(self):
        f = Fixture()
        try:
            t0 = time.monotonic()
            self.assertFalse(f.head.wait_all(1.0, stop=_after(1.5)))   # no hello yet: no clock, only the stop
            self.assertIsNone(f.head.verdict)
            self.assertGreater(time.monotonic() - t0, 1.4)
            ts, out = f.send_all([hello(1)])
            self.assertFalse(f.head.wait_all(1.0))
            ts[0].join(20)
            self.assertEqual(out[1]['field'], 'join')
            self.assertIn('rank 2 (127.0.0.1), rank 3 (127.0.0.1)', out[1]['detail'])
            self.assertEqual(out[1]['exit'], 1)
        finally:
            f.close()

    def test_a_late_worker_waits_for_a_slow_head(self):
        """The worker's own timeout (1 s) is shorter than the head's checks (2.5 s), and the head's verdict comes
        later still: the worker keeps asking, because the head keeps answering."""
        f = Fixture(ready=False)
        try:
            ts, out = f.send_all([hello(1), hello(2), hello(3)], timeout=1)
            time.sleep(2.5)
            f.ready.set()
            self.assertTrue(f.head.wait_all(20))
            time.sleep(1.5)
            f.head.decide(OK)
            for t in ts:
                t.join(20)
            self.assertEqual([out[r] for r in (1, 2, 3)], [OK] * 3)
        finally:
            f.close()

    def test_a_rank_that_stops_asking_is_dropped(self):
        f = Fixture(drop_s=1.0)
        try:
            self.assertEqual(f.head.rpc(hello(1), LAN), {'t': 'wait'})     # a hello, then silence
            ts, out = f.send_all([hello(2)])
            self.assertFalse(f.head.wait_all(20))
            ts[0].join(20)
            self.assertEqual(out[2]['field'], 'join')
            self.assertIn('rank 1 (127.0.0.1) stopped asking', out[2]['detail'])
        finally:
            f.close()

    def test_the_worker_gives_up_on_a_head_that_stops_answering(self):
        f = Fixture()
        ts, out = f.send_all([hello(1)], drop_s=1.0)
        time.sleep(0.5)
        f.close()
        ts[0].join(20)
        self.assertIn('stopped answering', out[1]['detail'])

    def test_a_refusing_head_answers_hello_and_poll(self):
        v = {'t': 'refused', 'field': 'head', 'values': {}, 'detail': 'the head stopped: boom', 'exit': 1}
        head = H.Head.refusing(v)
        self.assertEqual(head.rpc(hello(1), LAN), v)
        self.assertEqual(head.rpc({'t': 'verdict', 'rank': 1}, LAN), v)


def _after(s):
    e = threading.Event()
    threading.Timer(s, e.set).start()
    return e


if __name__ == '__main__':
    unittest.main()
