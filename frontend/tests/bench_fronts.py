#!/usr/bin/env python3
"""F1 gate numbers: what each front end adds on the head node, Python g6front vs g6front-rs.

Each side runs alone (its own mock engine, then torn down) so they never share the CPU. The mock prefills instantly
(--prefill-tps 1e15), decodes as fast as it can (--rounds-per-s 0) and skips its prompt-dependent scripting
(--plain), so what is left is front-end time plus the mock reading one generate frame:
  ttft_<N>   chat request with ~N prompt words*1.25 -> first SSE content byte (median of --reps)
  stream_tps tokens/s one stream carries end to end (2000 tokens)
  cpu_us_tok front-end CPU time per streamed token, over --conc concurrent 2000-token streams
  rss_idle / rss_load  front-end resident memory after start-up / at the end of the concurrent run (MiB)

usage: bench_fronts.py --model-dir DIR --rust-bin PATH [--py PYTHON] [--sizes 1024,8192,32768,131072] [--reps 5]
"""
import argparse
import http.client
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.dirname(HERE)
WORDS = ('river stone morning harbor lantern quiet market winter orchard letter village engine garden thunder '
         'copper meadow window valley story candle bridge north silver forest ladder kitchen shadow evening').split()


def proc_stats(pid):
    """(rss MiB, cpu seconds) of one process: /proc on Linux (tick resolution), else ps (macOS)."""
    if os.path.exists(f'/proc/{pid}/stat'):
        with open(f'/proc/{pid}/stat') as f:
            fields = f.read().rsplit(')', 1)[1].split()
        cpu = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
        with open(f'/proc/{pid}/status') as f:
            rss = next(int(ln.split()[1]) for ln in f if ln.startswith('VmRSS:')) / 1024
        return rss, cpu
    out = subprocess.run(['ps', '-o', 'rss=,time=', '-p', str(pid)], capture_output=True, text=True).stdout.split()
    rss = int(out[0]) / 1024
    t = out[1]
    days = 0
    if '-' in t:
        d, t = t.split('-')
        days = int(d)
    parts = [float(x) for x in t.split(':')]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return rss, secs + days * 86400


def post_stream(port, body, first_only=False):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    t0 = time.perf_counter()
    c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse()
    ttft, n = None, 0
    buf = b''
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            ev, buf = buf.split(b'\n\n', 1)
            if not ev.startswith(b'data: ') or ev == b'data: [DONE]':
                continue
            d = json.loads(ev[6:])
            if d.get('choices') and d['choices'][0].get('delta', {}).get('role') is None:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    if first_only:
                        c.close()
                        return ttft, 0, time.perf_counter() - t0
                n += 1
    c.close()
    return ttft, n, time.perf_counter() - t0


def prompt_of(tokens_approx, rng):
    # ~1.25 tokens per word for this vocabulary; exact size is reported from usage
    return ' '.join(rng.choice(WORDS) for _ in range(int(tokens_approx / 1.25)))


def wait_health(port):
    for _ in range(600):
        try:
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
            c.request('GET', '/health')
            if c.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.1)
    return False


def tokenize_count(port, text):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    c.request('POST', '/tokenize', json.dumps({'messages': [{'role': 'user', 'content': text}]}),
              {'Content-Type': 'application/json'})
    return json.loads(c.getresponse().read())['count']


def run_side(name, cmd, a, work, port):
    sock = os.path.join(work, f'mock-{name}.sock')
    mock = subprocess.Popen([a.py, os.path.join(FRONT, 'mock_engine.py'), '--model-dir', a.model_dir, '--sock', sock,
                             '--rounds-per-s', '0', '--prefill-tps', '1e15', '--max-model-len', '400000', '--plain'],
                            stdout=open(os.path.join(work, f'mock-{name}.log'), 'w'), stderr=subprocess.STDOUT)
    for _ in range(200):
        if os.path.exists(sock):
            break
        time.sleep(0.1)
    front = subprocess.Popen(cmd + ['--engine-sock', sock, '--host', '127.0.0.1', '--port', str(port)], cwd=FRONT,
                             stdout=open(os.path.join(work, f'front-{name}.log'), 'w'), stderr=subprocess.STDOUT)
    res = {}
    try:
        if not wait_health(port):
            raise RuntimeError(f'{name} did not start; see {work}')
        time.sleep(1.0)
        res['rss_idle'] = round(proc_stats(front.pid)[0], 1)
        rng = random.Random(1)
        for size in [int(s) for s in a.sizes.split(',')]:
            times = []
            for rep in range(a.reps + 1):  # first rep warms caches
                text = prompt_of(size, rng)
                body = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': text}], 'max_tokens': 4,
                        'stream': True}
                ttft, _, _ = post_stream(port, body, first_only=True)
                if rep:
                    times.append(ttft)
            res[f'ttft_{size}_ms'] = round(statistics.median(times) * 1e3, 2)
        res['prompt_tokens'] = {s: tokenize_count(port, prompt_of(int(s), random.Random(2)))
                                for s in a.sizes.split(',')}
        body = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'Tell me about harbors.'}],
                'max_tokens': 2000, 'stream': True}
        post_stream(port, body)  # warm
        _, n, dt = post_stream(port, body)
        res['stream_tps'] = round(n / dt, 1) if n else None
        _, cpu0 = proc_stats(front.pid)
        tok = [0]
        lock = threading.Lock()

        def worker():
            _, n, _ = post_stream(port, body)
            with lock:
                tok[0] += n

        ths = [threading.Thread(target=worker) for _ in range(a.conc)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        rss, cpu1 = proc_stats(front.pid)
        res['conc'] = a.conc
        res['conc_chunks_per_s'] = round(tok[0] / wall, 1)
        res['cpu_us_per_chunk'] = round((cpu1 - cpu0) / max(tok[0], 1) * 1e6, 1)
        res['rss_load'] = round(rss, 1)
    finally:
        for p in (front, mock):
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--rust-bin', required=True)
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--sizes', default='1024,8192,32768,131072')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--conc', type=int, default=16)
    ap.add_argument('--port', type=int, default=18111)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix='g6bench-')
    out = {
        'python': run_side('py', [a.py, '-m', 'g6front', '--model-dir', a.model_dir], a, work, a.port),
        'rust': run_side('rs', [a.rust_bin, '--model-dir', a.model_dir], a, work, a.port),
    }
    keys = [k for k in out['python'] if k != 'prompt_tokens']
    print(f"{'metric':<22}{'python':>12}{'rust':>12}")
    for k in keys:
        print(f"{k:<22}{out['python'][k]!s:>12}{out['rust'][k]!s:>12}")
    print('prompt tokens per size:', out['rust']['prompt_tokens'])
    print(json.dumps(out))


if __name__ == '__main__':
    main()
