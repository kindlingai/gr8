#!/usr/bin/env python3
"""g6shim.py's request checks, without an engine (stdlib only).
    cd frontend && python3 -m unittest tests.test_g6shim"""
import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import g6shim  # noqa: E402

V = g6shim.VOCAB


class BuildRequest(unittest.TestCase):
    def test_a_good_request(self):
        body, err = g6shim.build_request({'prompt': [5, 6], 'max_tokens': 4, 'stop_ids': [7, V - 1], 'seed': 3}, 9)
        self.assertIsNone(err)
        self.assertEqual(body, f'n_gen 4 temp 0.0 seed 3 min 0 stop 7,{V - 1} top_p 1.0\n5,6\n'.encode())

    def test_bad_stop_lists_are_refused(self):
        for stops in ([V], [-1], [True], ['5'], [1] * (g6shim.MAX_STOP_IDS + 1)):
            body, err = g6shim.build_request({'prompt': [5], 'max_tokens': 4, 'stop_ids': stops}, 0)
            self.assertIsNone(body, stops if len(stops) < 5 else len(stops))
            self.assertIn('bad stop ids', err)

    def test_a_prompt_id_outside_the_vocabulary_is_refused(self):
        self.assertIn('bad prompt token id', g6shim.build_request({'prompt': [5, V], 'max_tokens': 4}, 0)[1])


class One(unittest.TestCase):
    def test_a_refused_request_writes_nothing_and_keeps_its_number(self):
        shim = g6shim.Shim.__new__(g6shim.Shim)
        shim.n = 7
        shim.write_req = mock.AsyncMock()
        shim.finish = mock.Mock()
        r = mock.Mock(f={'prompt': [5], 'max_tokens': 4, 'stop_ids': [1] * 70000})
        asyncio.run(shim.one(r))
        shim.write_req.assert_not_called()
        self.assertEqual(shim.n, 7)
        self.assertEqual(shim.finish.call_args.args[1], 'error')


if __name__ == '__main__':
    unittest.main()
