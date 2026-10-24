"""./gr8 <model> resize (launch.cmd_resize): refusals, the wait for no running request, then down and up."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch  # noqa: E402

PEERS = '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4'
LEASE = ['--kv-store', 'remote:192.0.2.5:7350,size=8G,persist=lease']


class ResizeTests(unittest.TestCase):
    def run_resize(self, argv, counts):
        seen, calls = list(counts), []
        with mock.patch.object(launch, 'cmd_down', lambda a: calls.append('down')), \
                mock.patch.object(launch, 'cmd_up', lambda a: calls.append('up') or 0), \
                mock.patch('sys.stderr'):
            rc = launch.cmd_resize(['--no-config', '--peers', PEERS, *argv], wait=lambda s: None,
                                   running=lambda port: seen.pop(0) if seen else 0)
        return rc, calls

    def test_off_by_default_and_checkpoint_not_built(self):
        self.assertEqual(self.run_resize(LEASE, []), (2, []))
        self.assertEqual(self.run_resize(['--restart', 'checkpoint', *LEASE], []), (2, []))

    def test_needs_a_lease_store(self):
        self.assertEqual(self.run_resize(['--restart', 'idle'], []), (2, []))
        self.assertEqual(self.run_resize(['--restart', 'idle', '--kv-store', 'remote:192.0.2.5:7350,size=8G'], []),
                         (2, []))

    def test_waits_for_two_idle_polls_then_down_and_up(self):
        self.assertEqual(self.run_resize(['--restart', 'idle', *LEASE], [3, 1, 0, 2, 0, 0]), (0, ['down', 'up']))

    def test_gives_up_when_busy_past_the_wait_or_no_front_end(self):
        self.assertEqual(self.run_resize(['--restart', 'idle', '--resize-wait', '3', *LEASE], [1] * 10), (1, []))
        self.assertEqual(self.run_resize(['--restart', 'idle', *LEASE], [None]), (1, []))

    def test_reads_the_running_gauge(self):
        class R:
            def __init__(self, text):
                self.text = text

            def read(self):
                return self.text.encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        body = '# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model_name="glm"} 2\n'
        with mock.patch('urllib.request.urlopen', lambda u, timeout: R(body)):
            self.assertEqual(launch.running_requests('8100'), 2)
        with mock.patch('urllib.request.urlopen', mock.Mock(side_effect=OSError('refused'))):
            self.assertIsNone(launch.running_requests('8100'))


if __name__ == '__main__':
    unittest.main()
