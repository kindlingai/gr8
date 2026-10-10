#!/usr/bin/env python3
"""Engine-side S2 shim: speaks PROTOCOL.md v1 on the Unix socket and drives g6m4 --serve DIR (a DEC_S2 build) over the
S2 file protocol, so several requests run at the same time. Stdlib only (plus g6front.wire, and EOS_IDS from g6shim).

The engine side is the code, not a document: engine/runtime/s2/{ingress.h,sessions.h,service.h,bootstrap.h,scheduler.h}.
File protocol, in the head rank's DIR (the shim runs on the head), one engine epoch per start of g6m4 --serve (every
rank needs a FRESH DIR). The head passes each request's bytes and the quit to the other ranks over its control links.
  engine rank 0 -> DIR/ready.s2        "G6S2 2 <epoch> <ranks> <pool_rows> <mtp> <guard> <max_rows> <max_lp>" once all ranks
                                       booted (version 1 has no <max_lp>: no logprobs)
  engine rank 0 -> DIR/status.s2       JSON (resident, queued, free_pages, ...), rewritten on queue/lifecycle changes
  shim -> DIR/req.<id>                 written as req.<id>.tmp, then renamed. Two lines:
        n_gen <max_tokens> temp <float> seed <uint32> min <min_tokens> stop <id,id,... or -> top_p <float> [lp <k>] [plp <k>]
        <prompt id,id,...>
      <id> is the request ordinal: unique, strictly increasing in submit order (gaps are fine).
  shim -> rank 0 DIR/cmd.<k>           k = 0, 1, 2, ... with no gap, written the same way, one line:
        ADMIT <epoch> <id> <fnv1a64 of the req.<id> bytes>      (req.<id> is already written)
        CANCEL <epoch> <id> 0                                   (a real cancel: queued, prefilling or decoding)
      Requests wait in the engine's FIFO until a session slot and a KV run are free (several decode in one step).
  engine rank 0 -> DIR/resp.<id>       one line per event:
        a <epoch> <slot> <generation> <base_page> <pages>   admitted (the request got a slot and its KV run)
        p <j> <lp> <rank> <n> <id>:<lp> ...                  with plp: prompt token j (1 <= j < prompt length), before the
                                                             first t line; the shim sends one prompt_logprobs frame then
        l <lp> <rank> <n> <id>:<lp> ...                      with lp: one per token of the next t line (as g6shim.py)
        t <id> <id> ...                                      newly committed tokens -> tokens frame (+ logprobs)
        m <rounds> <draft_tokens> <accepted> <acc_pos0> ...  this request's spec counters, just before end
        end <stop|length|error|abort> <stop_id or -1> <completion_tokens>   -> finish frame
      A request cancelled while still queued gets only "end abort -1 0".
A request the engine cannot parse or fit ends the whole epoch (the engine writes DIR/abort.s2 and exits), so the shim
checks every field before it writes a file. SIGTERM/SIGINT writes DIR/quit (quit an idle engine).
If the shim restarts against a running engine it continues the cmd/req numbering from the files in DIR.

usage: g6s2shim.py --dir /run/g6/s2 [--sock /run/g6/engine.sock] [--max-model-len N]
"""
import argparse
import asyncio
import collections
import json
import logging
import math
import os
import re
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g6front import wire  # noqa: E402
from g6shim import EOS_IDS, VOCAB, Shim as PlainShim  # noqa: E402

log = logging.getLogger('g6s2shim')

INT32_MAX = 2**31 - 1
READY_FIELDS = {'1': ('epoch', 'ranks', 'pool_rows', 'mtp', 'guard', 'max_rows'),
                '2': ('epoch', 'ranks', 'pool_rows', 'mtp', 'guard', 'max_rows', 'max_lp')}


def payload_hash(data: bytes) -> int:
    """FNV-1a 64, the engine's payload_hash (s2/ingress.h)."""
    h = 14695981039346656037
    for c in data:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def parse_ready(text: str) -> dict:
    w = text.split()
    names = READY_FIELDS.get(w[1]) if len(w) > 1 and w[0] == 'G6S2' else None
    if not names or len(w) != 2 + len(names) or not all(x.isdigit() for x in w[2:]):
        raise ValueError(f'bad ready.s2: {text!r}')
    return {'max_lp': 0, **dict(zip(names, map(int, w[2:])))}    # version 1: no logprobs


def build_request(f: dict, ordinal: int):
    """-> (req.<id> bytes, error message or None). Validates what g6s2::parse_request / reserve_rows would reject."""
    prompt = f.get('prompt') or []
    mt = f.get('max_tokens')
    if not prompt or not isinstance(mt, int) or isinstance(mt, bool) or mt < 1 or mt > INT32_MAX:
        return None, 'bad prompt/max_tokens'
    # the engine indexes the embedding table with a prompt id and does not bound-check it
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= VOCAB for i in prompt):
        return None, f'bad prompt token id (must be 0..{VOCAB - 1})'
    stops = []
    for i in f.get('stop_ids') or []:
        if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i > INT32_MAX:
            return None, 'bad stop id'
        if i not in stops:
            stops.append(i)
    try:
        temp = 0.0 if f.get('temperature') is None else float(f['temperature'])
        top_p = 1.0 if f.get('top_p') is None else float(f['top_p'])    # 0 is an error, not "off" (as the HTTP front end)
        seed = f.get('seed')
        seed = (1234 + ordinal) if seed is None else int(seed) & 0xFFFFFFFF
        mn = min(int(f.get('min_tokens') or 0), mt)   # the engine refuses min_tokens > n_gen
    except (TypeError, ValueError, OverflowError):
        return None, 'bad temperature/top_p/seed/min_tokens'
    if not (math.isfinite(temp) and temp >= 0):
        return None, 'bad temperature'
    if not (math.isfinite(top_p) and 0 < top_p <= 1):
        return None, 'top_p must be in (0, 1]'
    lps = {}
    for key, word in (('logprobs', 'lp'), ('prompt_logprobs', 'plp')):
        v = f.get(key)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 20):
            return None, f'{key} must be 0..20'
        if v is not None:
            lps[word] = v
    body = (f"n_gen {mt} temp {temp!r} seed {seed} min {max(mn, 0)} stop {','.join(map(str, stops)) or '-'} top_p {top_p!r}"
            + ''.join(f' {w} {v}' for w, v in lps.items()) + '\n' + ','.join(map(str, prompt)) + '\n').encode()
    return body, None


def scan_numbers(d: str, prefix: str) -> int:
    """Next free number for files named <prefix>.<n> in d (0 for a fresh dir)."""
    top = -1
    try:
        for name in os.listdir(d):
            m = re.fullmatch(re.escape(prefix) + r'\.(\d+)', name)
            if m:
                top = max(top, int(m.group(1)))
    except FileNotFoundError:
        pass
    return top + 1


class Req:
    def __init__(self, f, writer):
        self.f, self.writer = f, writer
        self.rid = f['id']
        self.ordinal = None      # engine request id once the ADMIT command is written
        self.admitted = False    # an "a" line was seen: the engine gave it a slot
        self.cancelled = False   # the front end cancelled it
        self.cancel_at = 0.0     # when CANCEL went to the engine
        self.done = False        # a finish frame went out
        self.emitted = 0
        self.m = None            # this request's last "m" line
        self.lps = []            # "l" entries for the next "t" line
        self.plps = {}           # "p" entries by prompt position
        self.plp_sent = False    # the prompt_logprobs frame went out
        self.fh = None
        self.buf = b''
        self.last = time.monotonic()


class Shim:
    def __init__(self, a):
        self.a = a
        self.cfg = None                      # ready.s2 fields
        self.ready = asyncio.Event()
        self.dead = None                     # reason string once the engine epoch is lost
        self.live: dict[str, Req] = {}       # every unfinished request, by front-end id
        self.active: dict[int, Req] = {}     # ADMIT written, no "end" yet, by engine ordinal
        self.wlock = asyncio.Lock()          # serializes ordinal + cmd number allocation with the file writes
        self.next_id = scan_numbers(a.dir, 'req')
        self.next_cmd = scan_numbers(a.dir, 'cmd')
        self.wake = asyncio.Event()
        self.c = {'rounds': 0, 'draft_tokens': 0, 'accepted': 0, 'per_pos': [], 'prompt': 0, 'gen': 0,
                  'fin': collections.Counter(), 'max_active': 0}

    # ---- frames -------------------------------------------------------------------------------------------------
    def send(self, r: Req, frame: dict):
        if r.writer.is_closing():
            return
        try:
            wire.write_frame(r.writer, frame)
        except Exception as e:  # a dead front end must not kill the engine loop
            log.warning('write to front end failed: %r', e)

    def finish(self, r: Req, reason, ct=0, stop_id=None, message=None):
        if r.done:
            return
        r.done = True
        self.live.pop(r.rid, None)
        if r.ordinal is not None:
            self.active.pop(r.ordinal, None)
        if r.fh:
            r.fh.close()
            r.fh = None
        self.c['fin'][reason] += 1
        f = {'type': 'finish', 'id': r.rid, 'reason': reason, 'stop_id': stop_id,
             'prompt_tokens': len(r.f.get('prompt') or []), 'cached_tokens': 0, 'completion_tokens': ct}
        if message:
            f['message'] = message
        self.send(r, f)

    def status(self):
        try:
            with open(f'{self.a.dir}/status.s2') as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def metrics(self, seq):
        c = self.c
        running = sum(1 for r in self.live.values() if r.admitted)
        m = {'type': 'metrics', 'seq': seq,
             'spec': {'drafts': c['rounds'], 'draft_tokens': c['draft_tokens'], 'accepted': c['accepted'],
                      'accepted_per_pos': list(c['per_pos']) or [0] * (self.cfg['mtp'] if self.cfg else 0)},
             'prefix_cache': {'queries': 0, 'hits': 0},
             'requests': {'running': running, 'waiting': len(self.live) - running, 'finished': dict(c['fin'])},
             'tokens': {'prompt': c['prompt'], 'generation': c['gen']}}
        st = self.status()
        if st:
            m['s2'] = st    # extra field (ignored by the front end): the engine's own queue/pool status
        return m

    # ---- socket side --------------------------------------------------------------------------------------------
    async def serve(self, reader, writer):
        await self.ready.wait()
        wire.write_frame(writer, {'type': 'hello', 'version': 1, 'model': 'glm-5.3-g6',
                                  'max_model_len': self.max_model_len(), 'vocab_size': VOCAB,
                                  'eos_ids': EOS_IDS, 'max_logprobs': self.max_logprobs()})
        mine: set[str] = set()
        try:
            while True:
                f = await wire.read_frame(reader)
                if f is None:
                    break
                t = f['type']
                if t == 'generate':
                    self.submit(f, writer, mine)
                elif t == 'cancel':
                    await self.cancel(f.get('id'))
                elif t == 'metrics':
                    wire.write_frame(writer, self.metrics(f.get('seq')))
                await writer.drain()
        except (wire.FrameError, ConnectionError) as e:
            log.warning('front end connection closed: %r', e)
        finally:
            for rid in list(mine):  # connection gone: every request of it is dead
                await self.cancel(rid, notify=False)
            writer.close()

    def max_logprobs(self):
        return min(self.a.max_logprobs, self.cfg['max_lp'])

    def max_model_len(self):
        room = self.cfg['pool_rows'] - self.cfg['guard']
        return min(self.a.max_model_len, room) if self.a.max_model_len else room

    def submit(self, f, writer, mine):
        r = Req(f, writer)
        prompt, mt = f.get('prompt') or [], f.get('max_tokens', 0)
        if r.rid in self.live:
            return self.send(r, {'type': 'finish', 'id': r.rid, 'reason': 'error', 'stop_id': None,
                                 'prompt_tokens': len(prompt), 'cached_tokens': 0, 'completion_tokens': 0,
                                 'message': 'duplicate id'})
        if self.dead:
            return self.finish(r, 'error', message=self.dead)
        for k in ('logprobs', 'prompt_logprobs'):
            v = f.get(k)
            if v is None:
                continue
            if not self.cfg['max_lp']:      # a version-1 engine refuses any lp key and would end the epoch
                return self.finish(r, 'error', message=f'not supported by this engine: {k}')
            if not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= self.max_logprobs()):
                return self.finish(r, 'error', message=f'{k} must be 0..{self.max_logprobs()}')
        if f.get('prompt_logprobs') is not None and len(prompt) > self.a.max_plp_prompt:
            return self.finish(r, 'error', message=f'prompt too long for prompt_logprobs (max {self.a.max_plp_prompt})')
        if not isinstance(mt, int) or not prompt or mt < 1 or len(prompt) + mt > self.max_model_len():
            return self.finish(r, 'error', message='prompt too long' if prompt and isinstance(mt, int) and mt >= 1
                               else 'bad prompt/max_tokens')
        for k, off in (('top_k', 0), ('min_p', 0.0)):   # top_p goes to the engine
            v = f.get(k)
            if v is not None and v != off and float(f.get('temperature') or 0) > 0:
                log.warning('%s: %s=%s not supported, ignored', r.rid, k, v)
        self.live[r.rid] = r
        mine.add(r.rid)
        asyncio.get_running_loop().create_task(self.admit(r))

    async def cancel(self, rid, notify=True):
        r = self.live.get(rid)
        if r is None:
            return
        r.cancelled = True
        if r.ordinal is None:           # ADMIT not written yet (or being written): the admit task sees the flag
            if not r.done:
                self.finish(r, 'abort', ct=0) if notify else self.drop(r)
            return
        if r.cancel_at:
            return
        r.cancel_at = time.monotonic()
        if not notify:
            r.writer = _Closed()        # nobody to tell; the engine's end line still retires it
        await self.send_cancel(r)

    def drop(self, r):
        r.done = True
        self.live.pop(r.rid, None)

    # ---- engine side: files ---------------------------------------------------------------------------------------
    async def write_all(self, name: str, body: bytes):
        """req.<id> in DIR (atomic). -> list of error strings."""
        try:
            await asyncio.to_thread(write_atomic, f'{self.a.dir}/{name}', body)
        except OSError as e:
            return [f'{self.a.dir}: {e!r}']
        return []

    async def command(self, text: str):
        """Next cmd.<k> on rank 0. Caller holds wlock."""
        k = self.next_cmd
        await asyncio.to_thread(write_atomic, f'{self.a.dir}/cmd.{k}', text.encode())
        self.next_cmd += 1

    async def admit(self, r: Req):
        try:
            async with self.wlock:
                if r.cancelled or r.done:
                    return
                ordinal = self.next_id
                self.next_id += 1
                body, err = build_request(r.f, ordinal)
                if err:
                    return self.finish(r, 'error', message=err)
                errs = await self.write_all(f'req.{ordinal}', body)
                if errs:  # a request missing on one rank would stall the epoch at the fence: do not ADMIT it
                    log.error('req.%d write failed: %s', ordinal, errs)
                    return self.finish(r, 'error', message='req write failed: ' + '; '.join(errs))
                if r.cancelled or r.done:
                    return
                # from here a cancel() sends CANCEL; it queues behind this lock, so it lands after the ADMIT
                r.ordinal = ordinal
                r.last = time.monotonic()
                self.active[ordinal] = r
                await self.command(f"ADMIT {self.cfg['epoch']} {ordinal} {payload_hash(body)}\n")
                self.c['prompt'] += len(r.f['prompt'])
                self.c['max_active'] = max(self.c['max_active'], len(self.active))
                log.info('req.%d %s: %d prompt, n_gen %s admitted to the queue (%d in flight)', ordinal, r.rid,
                         len(r.f['prompt']), r.f['max_tokens'], len(self.active))
            self.wake.set()
        except Exception as e:
            log.exception('request %s failed', r.rid)
            self.fatal(f'shim error: {e!r}')

    async def send_cancel(self, r: Req):
        async with self.wlock:
            await self.command(f"CANCEL {self.cfg['epoch']} {r.ordinal} 0\n")
        log.info('req.%d %s: CANCEL sent', r.ordinal, r.rid)

    def fatal(self, why):
        if self.dead:
            return
        self.dead = why
        log.error('engine epoch lost: %s', why)
        for r in list(self.live.values()):
            self.finish(r, 'error', ct=r.emitted, message=why)

    # ---- engine side: responses ---------------------------------------------------------------------------------
    def line(self, r: Req, w):
        if not w:
            return
        if w[0] == 'a':
            r.admitted = True
        elif w[0] == 'l':
            r.lps.append(PlainShim.entry(w[1:]))
        elif w[0] == 'p':
            r.plps[int(w[1])] = PlainShim.entry(w[2:])
        elif w[0] == 't':
            ids = [int(x) for x in w[1:]]
            r.emitted += len(ids)
            self.c['gen'] += len(ids)
            lps, r.lps = r.lps, []
            if ids and r.f.get('prompt_logprobs') is not None and not r.plp_sent:
                r.plp_sent = True
                n = len(r.f['prompt'])
                if any(j not in r.plps for j in range(1, n)):
                    log.error('req.%s: engine sent %d of %d prompt logprobs', r.ordinal, len(r.plps), n - 1)
                    return self.finish(r, 'error', ct=r.emitted, message='engine returned incomplete prompt logprobs')
                self.send(r, {'type': 'prompt_logprobs', 'id': r.rid, 'lps': [None] + [r.plps[j] for j in range(1, n)]})
            if ids:
                fr = {'type': 'tokens', 'id': r.rid, 'ids': ids}
                if r.f.get('logprobs') is not None:
                    if len(lps) != len(ids):
                        log.error('req.%s: %d tokens with %d logprob entries', r.ordinal, len(ids), len(lps))
                        return self.finish(r, 'error', ct=r.emitted, message='engine returned mismatched logprobs')
                    fr['logprobs'] = lps
                self.send(r, fr)
        elif w[0] == 'm':
            r.m = [int(x) for x in w[1:]]
        elif w[0] == 'end':
            reason, sid, ct = w[1], int(w[2]), int(w[3])
            if ct != r.emitted:
                log.warning('req.%s: end says %d tokens, saw %d', r.ordinal, ct, r.emitted)
            if r.m:   # per-request counters, added once
                v = r.m
                self.c['rounds'] += v[0]
                self.c['draft_tokens'] += v[1]
                self.c['accepted'] += v[2]
                cur, pp = self.c['per_pos'], v[3:]
                self.c['per_pos'] = [(cur[i] if i < len(cur) else 0) + (pp[i] if i < len(pp) else 0)
                                     for i in range(max(len(cur), len(pp)))]
            if reason not in ('stop', 'length', 'error', 'abort'):
                reason = 'error'
            self.finish(r, reason, ct=ct, stop_id=sid if sid >= 0 else None,
                        message='engine error' if reason == 'error' else None)
            log.info('req.%s end %s %d tokens', r.ordinal, reason, ct)
        else:
            log.warning('req.%s: unknown resp line %r', r.ordinal, ' '.join(w))

    def pump(self, r: Req):
        path = f'{self.a.dir}/resp.{r.ordinal}'
        if r.fh is None:
            if not os.path.exists(path):
                return
            r.fh = open(path, 'rb')
        chunk = r.fh.read()
        if chunk:
            r.last = time.monotonic()
            r.buf += chunk
            *lines, r.buf = r.buf.split(b'\n')
            for ln in lines:
                self.line(r, ln.decode().split())
                if r.done:
                    return

    async def run(self):
        """Poll the resp files of every request in flight (one loop for all of them)."""
        next_abort_check = 0.0
        while True:
            if not self.active:
                self.wake.clear()
                await self.wake.wait()
            now = time.monotonic()
            for r in list(self.active.values()):
                if r.done:
                    continue
                try:
                    self.pump(r)
                except Exception as e:
                    log.exception('resp read of %s failed', r.rid)
                    self.finish(r, 'error', ct=r.emitted, message=repr(e))
                    continue
                if r.done:
                    continue
                if r.cancel_at and now - r.cancel_at > self.a.cancel_timeout:
                    log.error('req.%s: no end line %.0fs after CANCEL; answering abort', r.ordinal, self.a.cancel_timeout)
                    self.finish(r, 'abort', ct=r.emitted)
                elif r.admitted and now - r.last > self.a.stall_timeout:
                    self.finish(r, 'error', ct=r.emitted, message=f'engine silent {self.a.stall_timeout:.0f}s on req.{r.ordinal}')
            if now >= next_abort_check:
                next_abort_check = now + 0.1
                if os.path.exists(f'{self.a.dir}/abort.s2'):
                    self.fatal('engine S2 epoch aborted (abort.s2)')
            await asyncio.sleep(self.a.poll)

    async def wait_ready(self):
        path = f'{self.a.dir}/ready.s2'
        t0 = time.monotonic()
        while True:
            if os.path.exists(path):
                try:
                    with open(path) as fh:
                        text = fh.read()
                    if text.endswith('\n'):
                        self.cfg = parse_ready(text)
                        break
                except ValueError as e:
                    log.error('%s', e)
                    raise
            if os.path.exists(f'{self.a.dir}/abort.s2'):
                raise RuntimeError('engine aborted before ready')
            if time.monotonic() - t0 > self.a.ready_timeout:
                raise RuntimeError(f'no ready.s2 in {self.a.dir} after {self.a.ready_timeout:.0f}s')
            await asyncio.sleep(0.05)
        c = self.cfg
        log.info('engine ready: epoch %d, %d ranks, pool %d rows, mtp %d, guard %d, max rows %d; next req %d, next cmd %d',
                 c['epoch'], c['ranks'], c['pool_rows'], c['mtp'], c['guard'], c['max_rows'], self.next_id, self.next_cmd)
        self.ready.set()

    def quit(self):
        """DIR/quit: the head engine passes it to the other ranks."""
        try:
            open(f'{self.a.dir}/quit', 'w').close()
            log.info('quit written to %s', self.a.dir)
        except Exception as e:
            log.error('quit to %s failed: %r', self.a.dir, e)


class _Closed:
    """Stands in for the writer of a connection that is gone."""
    def is_closing(self):
        return True


def write_atomic(path: str, body: bytes):
    with open(path + '.tmp', 'wb') as fh:
        fh.write(body)
    os.rename(path + '.tmp', path)


async def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help="the head rank's engine --serve DIR (local)")
    ap.add_argument('--sock', default='/run/g6/engine.sock')
    ap.add_argument('--max-model-len', type=int, default=0,
                    help='cap on prompt + max_tokens (0 = what the KV pool allows: pool_rows - guard, from ready.s2)')
    ap.add_argument('--max-logprobs', type=int, default=20, help='largest k of logprobs (also capped by the engine ready line)')
    ap.add_argument('--max-plp-prompt', type=int, default=1024,
                    help='longest prompt of a request with prompt_logprobs: it runs as decode steps (~13 ms per token at '
                         'chunk 8, TP4) and every other session waits until it is done')
    ap.add_argument('--poll', type=float, default=0.002, help='resp poll interval, s')
    ap.add_argument('--ready-timeout', type=float, default=3600.0, help='wait this long for DIR/ready.s2')
    ap.add_argument('--cancel-timeout', type=float, default=60.0, help='answer abort if the engine sends no end line this long after CANCEL')
    ap.add_argument('--stall-timeout', type=float, default=900.0, help='error an admitted request after this long with no resp')
    a = ap.parse_args(argv)
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if ':' in a.dir:
        ap.error('--dir is the head rank\'s local directory')
    shim = Shim(a)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    os.makedirs(os.path.dirname(a.sock) or '.', exist_ok=True)
    server = await asyncio.start_unix_server(shim.serve, a.sock, limit=wire.MAX_FRAME)
    log.info('g6s2shim on %s -> %s, waiting for ready.s2', a.sock, a.dir)
    ready = asyncio.create_task(shim.wait_ready())
    worker = asyncio.create_task(shim.run())
    waiter = asyncio.create_task(stop.wait())
    await asyncio.wait([ready, waiter], return_when=asyncio.FIRST_COMPLETED)
    if ready.done() and ready.exception():
        log.error('startup failed: %r', ready.exception())
    else:
        await waiter
    ready.cancel()
    worker.cancel()
    server.close()
    shim.quit()
    return 1 if ready.done() and ready.exception() else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
