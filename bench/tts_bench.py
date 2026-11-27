#!/usr/bin/env python3
"""Streaming TTS benchmark for /v1/audio/speech (Python stdlib only).

POSTs /v1/audio/speech with response_format="pcm" and stream=true across one or
more concurrency levels, measuring per-request TTFA (time to first audio byte),
generation time, audio duration (bytes / 88200), RTF, aggregate audio-s/s, and
aggregate frames/s (audio-s * 21.533).

Example:
  python3 bench/tts_bench.py \\
    --url http://127.0.0.1:8100 --model drama-3 \\
    --prompts bench/data/tts_prompts_en.jsonl \\
    --concurrency 1,2,4,8 --runs 4 \\
    --max-new-tokens 500 --temperature 1.0 --top-p 0.98 --top-k 30 \\
    --voice default --json report.json
"""
import argparse
import concurrent.futures
import http.client
import json
import math
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

PCM_SAMPLE_RATE = 44100
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SEC = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_BYTES_PER_SAMPLE  # 88200
FRAMES_PER_AUDIO_SEC = 21.533

DEFAULT_PROMPTS = Path(__file__).resolve().parent / 'data' / 'tts_prompts_en.jsonl'


def percentile(values, p):
    """Linear-interpolated percentile in [0, 100] over a non-empty sequence."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    frac = k - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def load_prompts(path):
    """Load prompts from a JSONL file (or plain-text fallback).

    Each JSONL line may be an object with 'text', 'input', or 'prompt', or a
    JSON string. Blank lines are skipped.
    """
    prompts = []
    with Path(path).open('r', encoding='utf-8') as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                prompts.append({'id': f'line-{lineno}', 'text': line})
                continue
            if isinstance(obj, str):
                prompts.append({'id': f'line-{lineno}', 'text': obj})
            elif isinstance(obj, dict):
                text = obj.get('text') or obj.get('input') or obj.get('prompt')
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f'{path}:{lineno}: missing non-empty text/input/prompt field')
                prompts.append({'id': str(obj.get('id', f'line-{lineno}')), 'text': text})
            else:
                raise ValueError(f'{path}:{lineno}: unsupported JSONL entry type {type(obj).__name__}')
    if not prompts:
        raise ValueError(f'{path}: no prompts loaded')
    return prompts


def parse_concurrency(spec):
    levels = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError(f'concurrency must be positive, got {n}')
        levels.append(n)
    if not levels:
        raise ValueError('at least one concurrency level is required')
    return levels


def stream_speech_request(url, payload, timeout=300.0, chunk_size=8192):
    """POST /v1/audio/speech and stream PCM bytes, returning timing metrics."""
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme or 'http'
    host = parsed.hostname
    if not host:
        raise ValueError(f'invalid URL (missing host): {url!r}')
    port = parsed.port or (443 if scheme == 'https' else 80)
    base_path = parsed.path.rstrip('/')
    if base_path.endswith('/v1/audio/speech'):
        path = base_path
    else:
        path = f'{base_path}/v1/audio/speech'
    if parsed.query:
        path = f'{path}?{parsed.query}'

    body = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'audio/pcm, application/octet-stream, */*',
        'Content-Length': str(len(body)),
        'Connection': 'close',
    }

    conn_cls = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
    try:
        conn.connect()
        t0 = time.monotonic()
        conn.request('POST', path, body=body, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            err_body = resp.read(1024).decode('utf-8', errors='replace').strip()
            t_end = time.monotonic()
            return {
                'ok': False,
                'status': status,
                'error': f'HTTP {status}: {err_body}' if err_body else f'HTTP {status}',
                'bytes': 0,
                'audio_s': 0.0,
                'frames': 0.0,
                'ttfa_s': None,
                'ttfa_ms': None,
                'gen_s': t_end - t0,
                'rtf': None,
            }

        first = resp.read(1)
        if not first:
            t_end = time.monotonic()
            return {
                'ok': False,
                'status': status,
                'error': 'empty audio body',
                'bytes': 0,
                'audio_s': 0.0,
                'frames': 0.0,
                'ttfa_s': None,
                'ttfa_ms': None,
                'gen_s': t_end - t0,
                'rtf': None,
            }
        t_first = time.monotonic()
        total_bytes = len(first)
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
        t_end = time.monotonic()

        ttfa_s = t_first - t0
        gen_s = t_end - t0
        audio_s = total_bytes / float(PCM_BYTES_PER_SEC)
        frames = audio_s * FRAMES_PER_AUDIO_SEC
        rtf = (gen_s / audio_s) if audio_s > 0.0 else None
        return {
            'ok': True,
            'status': status,
            'error': None,
            'bytes': total_bytes,
            'audio_s': audio_s,
            'frames': frames,
            'ttfa_s': ttfa_s,
            'ttfa_ms': ttfa_s * 1000.0,
            'gen_s': gen_s,
            'rtf': rtf,
        }
    except Exception as exc:
        return {
            'ok': False,
            'status': None,
            'error': f'{type(exc).__name__}: {exc}',
            'bytes': 0,
            'audio_s': 0.0,
            'frames': 0.0,
            'ttfa_s': None,
            'ttfa_ms': None,
            'gen_s': None,
            'rtf': None,
        }
    finally:
        conn.close()


def build_payload(model, text, voice, max_new_tokens, temperature, top_p, top_k):
    return {
        'model': model,
        'input': text,
        'voice': voice,
        'response_format': 'pcm',
        'stream': True,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
    }


def summarize_level(concurrency, runs, rounds, requests):
    ok_reqs = [r for r in requests if r['ok']]
    ttfas_s = [r['ttfa_s'] for r in ok_reqs if r['ttfa_s'] is not None]
    rtfs = [r['rtf'] for r in ok_reqs if r['rtf'] is not None]

    total_audio_s = sum(r['audio_s'] for r in ok_reqs)
    total_wall_s = sum(rd['wall_s'] for rd in rounds)
    audio_s_per_s = (total_audio_s / total_wall_s) if total_wall_s > 0.0 else 0.0
    frames_per_s = audio_s_per_s * FRAMES_PER_AUDIO_SEC

    ttfa_p50_s = percentile(ttfas_s, 50.0)
    ttfa_p95_s = percentile(ttfas_s, 95.0)
    mean_rtf = statistics.fmean(rtfs) if rtfs else None

    return {
        'concurrency': concurrency,
        'runs': runs,
        'requests_ok': len(ok_reqs),
        'requests_total': len(requests),
        'ttfa_p50_s': ttfa_p50_s,
        'ttfa_p95_s': ttfa_p95_s,
        'ttfa_p50_ms': (ttfa_p50_s * 1000.0) if ttfa_p50_s is not None else None,
        'ttfa_p95_ms': (ttfa_p95_s * 1000.0) if ttfa_p95_s is not None else None,
        'mean_rtf': mean_rtf,
        'audio_s_per_s': audio_s_per_s,
        'frames_per_s': frames_per_s,
        'total_audio_s': total_audio_s,
        'total_wall_s': total_wall_s,
        'rounds': rounds,
        'requests': requests,
    }


def run_concurrency_level(
    url,
    model,
    prompts,
    concurrency,
    runs,
    voice,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    timeout=300.0,
    prompt_offset=0,
):
    rounds = []
    all_requests = []
    idx = prompt_offset

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for run_idx in range(runs):
            batch_prompts = []
            for slot in range(concurrency):
                p_idx = idx % len(prompts)
                idx += 1
                batch_prompts.append((slot, p_idx, prompts[p_idx]))

            t_round_start = time.monotonic()
            futs = []
            for slot, p_idx, prompt in batch_prompts:
                payload = build_payload(
                    model=model,
                    text=prompt['text'],
                    voice=voice,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                fut = pool.submit(stream_speech_request, url, payload, timeout)
                futs.append((slot, p_idx, prompt['id'], fut))

            round_reqs = []
            for slot, p_idx, prompt_id, fut in futs:
                res = fut.result()
                res['run'] = run_idx
                res['slot'] = slot
                res['prompt_index'] = p_idx
                res['prompt_id'] = prompt_id
                round_reqs.append(res)
            t_round_end = time.monotonic()

            wall_s = t_round_end - t_round_start
            round_audio_s = sum(r['audio_s'] for r in round_reqs if r['ok'])
            round_audio_per_s = (round_audio_s / wall_s) if wall_s > 0.0 else 0.0
            rounds.append({
                'run': run_idx,
                'wall_s': wall_s,
                'audio_s': round_audio_s,
                'audio_s_per_s': round_audio_per_s,
                'frames_per_s': round_audio_per_s * FRAMES_PER_AUDIO_SEC,
                'requests_ok': sum(1 for r in round_reqs if r['ok']),
                'requests_total': len(round_reqs),
            })
            all_requests.extend(round_reqs)

    return summarize_level(concurrency, runs, rounds, all_requests), idx


def run_benchmark(args):
    prompts = load_prompts(args.prompts)
    levels = parse_concurrency(args.concurrency)
    if args.runs <= 0:
        raise ValueError(f'--runs must be positive, got {args.runs}')

    t_start = time.monotonic()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    if args.warmup > 0:
        for w in range(args.warmup):
            prompt = prompts[w % len(prompts)]
            payload = build_payload(
                model=args.model,
                text=prompt['text'],
                voice=args.voice,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            stream_speech_request(args.url, payload, timeout=args.timeout)

    by_concurrency = {}
    prompt_cursor = 0
    for c in levels:
        level_summary, prompt_cursor = run_concurrency_level(
            url=args.url,
            model=args.model,
            prompts=prompts,
            concurrency=c,
            runs=args.runs,
            voice=args.voice,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout=args.timeout,
            prompt_offset=prompt_cursor,
        )
        by_concurrency[str(c)] = level_summary

    took_s = time.monotonic() - t_start
    return {
        'meta': {
            'url': args.url,
            'model': args.model,
            'prompts': str(args.prompts),
            'prompt_count': len(prompts),
            'concurrency': levels,
            'runs': args.runs,
            'warmup': args.warmup,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'voice': args.voice,
            'response_format': 'pcm',
            'stream': True,
            'pcm_bytes_per_sec': PCM_BYTES_PER_SEC,
            'frames_per_audio_sec': FRAMES_PER_AUDIO_SEC,
            'started': started,
            'took_s': round(took_s, 3),
        },
        'concurrency': by_concurrency,
    }


def format_table(report):
    header = (
        f'{"C":>4}  {"ok/tot":>7}  {"TTFA p50":>10}  {"TTFA p95":>10}  '
        f'{"mean RTF":>9}  {"audio-s/s":>10}  {"frames/s":>10}'
    )
    sep = '-' * len(header)
    lines = [
        f'model={report["meta"]["model"]}  voice={report["meta"]["voice"]}  '
        f'runs={report["meta"]["runs"]}  max_new_tokens={report["meta"]["max_new_tokens"]}',
        header,
        sep,
    ]
    for c in report['meta']['concurrency']:
        row = report['concurrency'][str(c)]
        p50 = f'{row["ttfa_p50_ms"]:.1f} ms' if row['ttfa_p50_ms'] is not None else 'n/a'
        p95 = f'{row["ttfa_p95_ms"]:.1f} ms' if row['ttfa_p95_ms'] is not None else 'n/a'
        rtf = f'{row["mean_rtf"]:.3f}' if row['mean_rtf'] is not None else 'n/a'
        aud = f'{row["audio_s_per_s"]:.2f}'
        fps = f'{row["frames_per_s"]:.1f}'
        ok_tot = f'{row["requests_ok"]}/{row["requests_total"]}'
        lines.append(f'{c:>4}  {ok_tot:>7}  {p50:>10}  {p95:>10}  {rtf:>9}  {aud:>10}  {fps:>10}')
    return '\n'.join(lines)


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--url', default='http://127.0.0.1:8100', help='Base URL of the TTS server')
    ap.add_argument('--model', default='drama-3', help='Served model name')
    ap.add_argument('--prompts', default=str(DEFAULT_PROMPTS), help='Path to prompts JSONL file')
    ap.add_argument('--concurrency', default='1,2,4,8', help='Comma-separated concurrency levels')
    ap.add_argument('--runs', type=int, default=4, help='Number of concurrent rounds per level')
    ap.add_argument('--max-new-tokens', type=int, default=500, help='Maximum audio frames to generate')
    ap.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    ap.add_argument('--top-p', type=float, default=0.98, help='Nucleus sampling top-p')
    ap.add_argument('--top-k', type=int, default=30, help='Top-k sampling limit')
    ap.add_argument('--voice', default='default', help='Voice name')
    ap.add_argument('--warmup', type=int, default=0, help='Optional unmeasured warmup requests before timing')
    ap.add_argument('--timeout', type=float, default=300.0, help='Per-request HTTP socket timeout in seconds')
    ap.add_argument('--json', dest='json_out', help='Output path for the JSON report')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(format_table(report))
    if args.json_out:
        print(f'JSON report: {args.json_out}')

    total_ok = sum(v['requests_ok'] for v in report['concurrency'].values())
    total_all = sum(v['requests_total'] for v in report['concurrency'].values())
    return 0 if total_ok == total_all and total_all > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
