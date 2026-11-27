"""Unit tests for bench/tts_bench.py against a mock streaming HTTP server."""
import http.server
import json
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR))

import tts_bench  # noqa: E402


class MockSpeechServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler_cls):
        super().__init__(addr, handler_cls)
        self.lock = threading.Lock()
        self.requests_seen = []
        self.active = 0
        self.max_active = 0
        self.ttfa_delay_s = 0.03
        self.chunk_delay_s = 0.01
        self.num_chunks = 4
        # 4 * 22050 = 88200 bytes = 1.000 s of 44.1 kHz s16le mono PCM
        self.chunk_bytes = 22050
        self.fail_status = None
        self.return_empty = False
        self.use_chunked = True


class MockSpeechHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        srv = self.server
        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length) if length else b''
        body = json.loads(raw.decode('utf-8')) if raw else {}

        with srv.lock:
            srv.active += 1
            if srv.active > srv.max_active:
                srv.max_active = srv.active
            srv.requests_seen.append({'path': self.path, 'body': body})

        try:
            if self.path != '/v1/audio/speech':
                msg = b'not found'
                self.send_response(404)
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return

            if srv.fail_status is not None:
                msg = b'upstream error'
                self.send_response(srv.fail_status)
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return

            if srv.return_empty:
                self.send_response(200)
                self.send_header('Content-Type', 'audio/pcm')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return

            self.send_response(200)
            self.send_header('Content-Type', 'audio/pcm')
            if srv.use_chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            else:
                self.send_header('Content-Length', str(srv.num_chunks * srv.chunk_bytes))
            self.end_headers()

            if srv.ttfa_delay_s > 0:
                time.sleep(srv.ttfa_delay_s)

            payload = b'\x01\x00' * (srv.chunk_bytes // 2)
            for idx in range(srv.num_chunks):
                if idx > 0 and srv.chunk_delay_s > 0:
                    time.sleep(srv.chunk_delay_s)
                if srv.use_chunked:
                    self.wfile.write(f'{len(payload):x}\r\n'.encode('ascii'))
                    self.wfile.write(payload)
                    self.wfile.write(b'\r\n')
                else:
                    self.wfile.write(payload)
                self.wfile.flush()

            if srv.use_chunked:
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
        finally:
            with srv.lock:
                srv.active -= 1


class TestPromptsFile(unittest.TestCase):
    def test_default_prompts_en_jsonl(self):
        path = BENCH_DIR / 'data' / 'tts_prompts_en.jsonl'
        prompts = tts_bench.load_prompts(path)
        self.assertEqual(len(prompts), 64)
        texts = [p['text'] for p in prompts]
        self.assertEqual(len(set(texts)), 64, 'prompts should all be distinct')
        for p in prompts:
            words = p['text'].split()
            self.assertGreaterEqual(len(words), 8, f'too short ({len(words)} words): {p}')
            self.assertLessEqual(len(words), 40, f'too long ({len(words)} words): {p}')


class TestMetricsMath(unittest.TestCase):
    def test_percentile_interpolation(self):
        self.assertAlmostEqual(tts_bench.percentile([10.0], 50), 10.0)
        self.assertAlmostEqual(tts_bench.percentile([10.0, 20.0, 30.0, 40.0], 50), 25.0)
        self.assertAlmostEqual(tts_bench.percentile([10.0, 20.0, 30.0, 40.0], 95), 38.5)
        self.assertIsNone(tts_bench.percentile([], 50))

    def test_summarize_level_exact_math(self):
        rounds = [
            {'run': 0, 'wall_s': 0.5, 'audio_s': 2.0, 'audio_s_per_s': 4.0, 'frames_per_s': 4.0 * 21.533,
             'requests_ok': 2, 'requests_total': 2},
            {'run': 1, 'wall_s': 0.5, 'audio_s': 2.0, 'audio_s_per_s': 4.0, 'frames_per_s': 4.0 * 21.533,
             'requests_ok': 2, 'requests_total': 2},
        ]
        reqs = [
            {'ok': True, 'audio_s': 1.0, 'ttfa_s': 0.10, 'gen_s': 0.40, 'rtf': 0.40},
            {'ok': True, 'audio_s': 1.0, 'ttfa_s': 0.20, 'gen_s': 0.50, 'rtf': 0.50},
            {'ok': True, 'audio_s': 1.0, 'ttfa_s': 0.30, 'gen_s': 0.45, 'rtf': 0.45},
            {'ok': True, 'audio_s': 1.0, 'ttfa_s': 0.40, 'gen_s': 0.45, 'rtf': 0.45},
        ]
        s = tts_bench.summarize_level(concurrency=2, runs=2, rounds=rounds, requests=reqs)
        self.assertEqual(s['concurrency'], 2)
        self.assertEqual(s['requests_ok'], 4)
        self.assertEqual(s['requests_total'], 4)
        self.assertAlmostEqual(s['ttfa_p50_ms'], 250.0, places=6)
        self.assertAlmostEqual(s['ttfa_p95_ms'], 385.0, places=6)
        self.assertAlmostEqual(s['mean_rtf'], 0.45, places=6)
        self.assertAlmostEqual(s['audio_s_per_s'], 4.0, places=6)
        self.assertAlmostEqual(s['frames_per_s'], 4.0 * 21.533, places=6)


class TestStreamingBenchE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = MockSpeechServer(('127.0.0.1', 0), MockSpeechHandler)
        cls.port = cls.srv.server_address[1]
        cls.url = f'http://127.0.0.1:{cls.port}'
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.requests_seen.clear()
        self.srv.active = 0
        self.srv.max_active = 0
        self.srv.ttfa_delay_s = 0.03
        self.srv.chunk_delay_s = 0.01
        self.srv.num_chunks = 4
        self.srv.chunk_bytes = 22050  # total 88200 bytes = 1.0 s PCM
        self.srv.fail_status = None
        self.srv.return_empty = False
        self.srv.use_chunked = True

    def test_cli_metrics_concurrency_and_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / 'out' / 'report.json'
            cmd = [
                sys.executable,
                str(BENCH_DIR / 'tts_bench.py'),
                '--url', self.url,
                '--model', 'drama-3',
                '--prompts', str(BENCH_DIR / 'data' / 'tts_prompts_en.jsonl'),
                '--concurrency', '1,3',
                '--runs', '2',
                '--max-new-tokens', '500',
                '--temperature', '1.0',
                '--top-p', '0.98',
                '--top-k', '30',
                '--voice', 'default',
                '--json', str(out_json),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue(out_json.exists())

            data = json.loads(out_json.read_text(encoding='utf-8'))
            meta = data['meta']
            self.assertEqual(meta['model'], 'drama-3')
            self.assertEqual(meta['concurrency'], [1, 3])
            self.assertEqual(meta['runs'], 2)
            self.assertEqual(meta['max_new_tokens'], 500)
            self.assertAlmostEqual(meta['temperature'], 1.0)
            self.assertAlmostEqual(meta['top_p'], 0.98)
            self.assertEqual(meta['top_k'], 30)
            self.assertEqual(meta['voice'], 'default')
            self.assertEqual(meta['response_format'], 'pcm')
            self.assertTrue(meta['stream'])
            self.assertEqual(meta['pcm_bytes_per_sec'], 88200)
            self.assertAlmostEqual(meta['frames_per_audio_sec'], 21.533)

            # Verify exact request count: 1*2 + 3*2 = 8 requests, and true concurrency >= 2
            self.assertEqual(len(self.srv.requests_seen), 8)
            self.assertGreaterEqual(self.srv.max_active, 2)
            first_req = self.srv.requests_seen[0]
            self.assertEqual(first_req['path'], '/v1/audio/speech')
            self.assertEqual(first_req['body']['model'], 'drama-3')
            self.assertEqual(first_req['body']['response_format'], 'pcm')
            self.assertTrue(first_req['body']['stream'])
            self.assertEqual(first_req['body']['voice'], 'default')
            self.assertEqual(first_req['body']['max_new_tokens'], 500)
            self.assertEqual(first_req['body']['top_k'], 30)

            c1 = data['concurrency']['1']
            c3 = data['concurrency']['3']
            self.assertEqual(c1['requests_ok'], 2)
            self.assertEqual(c1['requests_total'], 2)
            self.assertEqual(c3['requests_ok'], 6)
            self.assertEqual(c3['requests_total'], 6)

            for row in (c1, c3):
                # TTFA should reflect the 30ms server delay (and be strictly less than total gen_s)
                self.assertGreaterEqual(row['ttfa_p50_ms'], 20.0)
                self.assertGreaterEqual(row['ttfa_p95_ms'], row['ttfa_p50_ms'])
                self.assertGreater(row['mean_rtf'], 0.0)
                self.assertLess(row['mean_rtf'], 0.5)
                self.assertAlmostEqual(
                    row['frames_per_s'],
                    row['audio_s_per_s'] * 21.533,
                    places=6,
                )
                for req in row['requests']:
                    self.assertTrue(req['ok'])
                    self.assertEqual(req['bytes'], 88200)
                    self.assertAlmostEqual(req['audio_s'], 1.0, places=6)
                    self.assertLess(req['ttfa_s'], req['gen_s'])

            # Concurrency 3 should achieve noticeably higher aggregate audio-s/s than concurrency 1
            self.assertGreater(c3['audio_s_per_s'], c1['audio_s_per_s'] * 1.8)

    def test_error_response_returns_nonzero(self):
        self.srv.fail_status = 500
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / 'err.json'
            rc = tts_bench.main([
                '--url', self.url,
                '--model', 'drama-3',
                '--prompts', str(BENCH_DIR / 'data' / 'tts_prompts_en.jsonl'),
                '--concurrency', '1',
                '--runs', '1',
                '--json', str(out_json),
            ])
            self.assertEqual(rc, 1)
            data = json.loads(out_json.read_text(encoding='utf-8'))
            c1 = data['concurrency']['1']
            self.assertEqual(c1['requests_ok'], 0)
            self.assertEqual(c1['requests_total'], 1)
            self.assertIn('HTTP 500', c1['requests'][0]['error'])


if __name__ == '__main__':
    unittest.main()
