#!/usr/bin/env python3
"""Engine-side shim: speaks PROTOCOL.md v1 on the Unix socket and drives the real engine (g6m4 / m4sim --serve DIR)
through files. Stdlib only (plus g6front.wire).

File protocol, per request n = 0, 1, 2, ... (one request at a time, the rest queue):
  shim -> every rank DIR/req.<n> (written as req.<n>.tmp then renamed; "host:/path" entries over ssh):
      n_gen <max_tokens> temp <float> seed <uint32> min <min_tokens> stop <id,id,... or -> top_p <float> [lp <k>] [plp <k>]
      <prompt id,id,...>
  rank 0 -> DIR/resp.<n> (the first --rank-dirs entry, local), one line per event:
      l <lp> <rank> <n> <id>:<lp> ...                   (request with lp) one per token of the next t line, before it
      p <j> <lp> <rank> <n> <id>:<lp> ...               (request with plp) prompt token j, before the first t line
      t <id> <id> ...                                   newly committed tokens -> tokens frame (+ logprobs entries)
      m <rounds> <draft_tokens> <accepted> <acc_pos0> ...  cumulative spec counters -> metrics frame
      end <stop|length|error> <stop_id or -1> <completion_tokens>  -> finish frame
cancel: the engine cannot stop mid-request, so the shim answers finish/abort at once and swallows the rest.
A request with prompt_logprobs runs its whole prompt as decode steps (no prefill path): --max-plp-prompt bounds it.
SIGTERM/SIGINT: writes DIR/quit to every rank dir, then exits.

usage: g6shim.py --rank-dirs /run/g6/r0 node2:/run/g6/r1 ... [--sock /run/g6/engine.sock] [--max-model-len 2048]
"""
import argparse
import asyncio
import collections
import logging
import math
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g6front import wire  # noqa: E402

log = logging.getLogger('g6shim')

EOS_IDS = [154820, 154827, 154829]
# one persistent ssh connection per host: a fresh ssh per request costs ~0.25 s of TTFT
# (masters are opened at start with stdio closed; a master forked from a request would hold its stderr pipe open)
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlPath=/tmp/g6shim-ssh-%C', '-o', 'ControlMaster=no']
VOCAB = 154880
ENGINE_MAX_LP = 20  # the engine's top lists hold 20 (serve.h LP_MAX = glue::LP_K); it clamps silently above
FLOOR = -9999.0  # logprob sent for -inf / nan (the front end clamps at the same value)


class Req:
    def __init__(self, f, writer):
        self.f, self.writer = f, writer
        self.rid = f['id']
        self.cancelled = False  # finish/abort already sent; swallow the rest
        self.done = False       # a finish frame went out
        self.lps = []           # engine "l" entries not yet sent (logprobs)
        self.plps = {}          # engine "p" entries by prompt position
        self.plp_sent = False


class Shim:
    def __init__(self, a):
        self.a = a
        self.n = a.first_n
        self.queue: collections.deque[Req] = collections.deque()
        self.wake = asyncio.Event()
        self.live: dict[str, Req] = {}  # queued + running, by id
        self.running: Req | None = None
        self.c = {'rounds': 0, 'draft_tokens': 0, 'accepted': 0, 'per_pos': [], 'prompt': 0, 'gen': 0,
                  'fin': collections.Counter()}

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
        self.c['fin'][reason] += 1
        f = {'type': 'finish', 'id': r.rid, 'reason': reason, 'stop_id': stop_id,
             'prompt_tokens': len(r.f.get('prompt') or []), 'cached_tokens': 0, 'completion_tokens': ct}
        if message:
            f['message'] = message
        self.send(r, f)

    def metrics(self, seq):
        c = self.c
        return {'type': 'metrics', 'seq': seq,
                'spec': {'drafts': c['rounds'], 'draft_tokens': c['draft_tokens'], 'accepted': c['accepted'],
                         'accepted_per_pos': list(c['per_pos'])},
                'prefix_cache': {'queries': 0, 'hits': 0},
                'requests': {'running': int(self.running is not None and not self.running.done),
                             'waiting': len(self.queue), 'finished': dict(c['fin'])},
                'tokens': {'prompt': c['prompt'], 'generation': c['gen']}}

    # ---- socket side --------------------------------------------------------------------------------------------
    async def serve(self, reader, writer):
        wire.write_frame(writer, {'type': 'hello', 'version': 1, 'model': 'glm-5.3-g6',
                                  'max_model_len': self.a.max_model_len, 'vocab_size': VOCAB,
                                  'eos_ids': EOS_IDS, 'max_logprobs': self.a.max_logprobs})
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
                    self.cancel(f.get('id'))
                elif t == 'metrics':
                    wire.write_frame(writer, self.metrics(f.get('seq')))
                await writer.drain()
        except (wire.FrameError, ConnectionError) as e:
            log.warning('front end connection closed: %r', e)
        finally:
            for rid in list(mine):  # connection gone: every request of it is dead
                self.cancel(rid, notify=False)
            writer.close()

    def submit(self, f, writer, mine):
        r = Req(f, writer)
        prompt, mt = f.get('prompt') or [], f.get('max_tokens', 0)
        if r.rid in self.live:
            return self.send(r, {'type': 'finish', 'id': r.rid, 'reason': 'error', 'stop_id': None,
                                 'prompt_tokens': len(prompt), 'cached_tokens': 0, 'completion_tokens': 0,
                                 'message': 'duplicate id'})
        for k in ('logprobs', 'prompt_logprobs'):
            v = f.get(k)
            if v is not None and not (isinstance(v, int) and 0 <= v <= self.a.max_logprobs):
                return self.finish(r, 'error', message=f'{k} must be 0..{self.a.max_logprobs}')
        if f.get('prompt_logprobs') is not None and len(prompt) > self.a.max_plp_prompt:
            return self.finish(r, 'error', message=f'prompt too long for prompt_logprobs (max {self.a.max_plp_prompt})')
        if not prompt or mt < 1 or len(prompt) + mt > self.a.max_model_len:
            return self.finish(r, 'error', message='bad prompt/max_tokens')
        for k, off in (('top_k', 0), ('min_p', 0.0)):   # top_p goes to the engine (--top-p-cands)
            v = f.get(k)
            if v is not None and v != off and float(f.get('temperature') or 0) > 0:
                log.warning('%s: %s=%s not supported, ignored', r.rid, k, v)
        self.live[r.rid] = r
        mine.add(r.rid)
        self.queue.append(r)
        self.wake.set()

    def cancel(self, rid, notify=True):
        r = self.live.get(rid)
        if r is None:
            return
        r.cancelled = True
        if r in self.queue:
            self.queue.remove(r)
        if notify:
            self.finish(r, 'abort', ct=getattr(r, 'emitted', 0))
        else:
            r.done = True
            self.live.pop(rid, None)

    # ---- engine side --------------------------------------------------------------------------------------------
    async def write_req(self, n, body: bytes):
        procs = []
        for d in self.a.rank_dirs:
            if ':' in d:
                host, path = d.split(':', 1)
                cmd = f'cat > {path}/req.{n}.tmp && mv {path}/req.{n}.tmp {path}/req.{n}'
                procs.append((d, await asyncio.create_subprocess_exec(
                    *SSH, host, cmd, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)))
            else:
                with open(f'{d}/req.{n}.tmp', 'wb') as fh:
                    fh.write(body)
                os.rename(f'{d}/req.{n}.tmp', f'{d}/req.{n}')
        errs = []
        for d, p in procs:
            _, err = await p.communicate(body)
            if p.returncode:
                errs.append(f'{d}: rc {p.returncode} {err.decode(errors="replace").strip()}')
        return errs

    async def run(self):
        while True:
            while not self.queue:
                self.wake.clear()
                await self.wake.wait()
            r = self.queue.popleft()
            self.running = r
            try:
                await self.one(r)
            except Exception as e:
                log.exception('request %s failed', r.rid)
                self.finish(r, 'error', message=repr(e))
            finally:
                self.running = None

    async def one(self, r: Req):
        f, n = r.f, self.n
        self.n += 1
        seed = f.get('seed')
        seed = (1234 + n) if seed is None else int(seed) & 0xFFFFFFFF
        stop = ','.join(str(i) for i in (f.get('stop_ids') or [])) or '-'
        body = (f"n_gen {f['max_tokens']} temp {float(f.get('temperature') or 0.0)} seed {seed} "
                f"min {int(f.get('min_tokens') or 0)} stop {stop} top_p {float(f.get('top_p') or 1.0)}"
                + (f" lp {int(f['logprobs'])}" if f.get('logprobs') is not None else '')
                + (f" plp {int(f['prompt_logprobs'])}" if f.get('prompt_logprobs') is not None else '')
                + '\n' + ','.join(str(i) for i in f['prompt']) + '\n').encode()
        resp = f'{self.a.rank_dirs[0]}/resp.{n}'
        if os.path.exists(resp):
            log.warning('stale %s removed', resp)
            os.unlink(resp)
        t0 = time.monotonic()
        errs = await self.write_req(n, body)
        if errs:
            # a partial write leaves the ranks out of step; nothing sane to do but report it loudly
            log.error('req.%d write failed: %s', n, errs)
            return self.finish(r, 'error', message='req write failed: ' + '; '.join(errs))
        log.info('req.%d %s: %d prompt, n_gen %s, written in %.1f ms', n, r.rid, len(f['prompt']),
                 f['max_tokens'], (time.monotonic() - t0) * 1e3)
        self.c['prompt'] += len(f['prompt'])
        r.emitted = 0
        fh, buf, last = None, b'', time.monotonic()
        try:
            while True:
                if fh is None and os.path.exists(resp):
                    fh = open(resp, 'rb')
                chunk = fh.read() if fh else b''
                if not chunk:
                    if time.monotonic() - last > self.a.stall_timeout:
                        return self.finish(r, 'error', ct=r.emitted,
                                           message=f'engine silent {self.a.stall_timeout:.0f}s on req.{n}')
                    await asyncio.sleep(self.a.poll)
                    continue
                last = time.monotonic()
                buf += chunk
                *lines, buf = buf.split(b'\n')
                for ln in lines:
                    if self.line(r, n, ln.decode().split()):
                        return
        finally:
            if fh:
                fh.close()

    @staticmethod
    def entry(w):
        """engine logprob fields <lp> <rank> <n> <id>:<lp> ... -> a PROTOCOL.md entry {lp, rank, top}"""
        def num(x):
            x = float(x)
            return x if math.isfinite(x) else FLOOR
        rank, k = int(w[1]), int(w[2])
        top = []
        for tok in w[3:3 + k]:
            i, v = tok.split(':')
            top.append([int(i), num(v)])
        return {'lp': num(w[0]), 'rank': rank if rank > 0 else None, 'top': top}

    def prompt_frame(self, r: Req):
        """the prompt_logprobs frame, once, before the first tokens; None if some position is missing"""
        n = len(r.f['prompt'])
        if any(j not in r.plps for j in range(1, n)):
            return None
        return {'type': 'prompt_logprobs', 'id': r.rid, 'lps': [None] + [r.plps[j] for j in range(1, n)]}

    def line(self, r: Req, n, w) -> bool:
        """Handle one resp line; True when the request ended."""
        if not w:
            return False
        if w[0] == 'l':
            r.lps.append(self.entry(w[1:]))
            return False
        if w[0] == 'p':
            r.plps[int(w[1])] = self.entry(w[2:])
            return False
        if w[0] == 't':
            ids = [int(x) for x in w[1:]]
            r.emitted += len(ids)
            self.c['gen'] += len(ids)
            if not r.cancelled and ids:
                if r.f.get('prompt_logprobs') is not None and not r.plp_sent:
                    r.plp_sent = True
                    pf = self.prompt_frame(r)
                    if pf is None:
                        log.error('req.%d: engine sent %d of %d prompt logprobs', n, len(r.plps), len(r.f['prompt']) - 1)
                        self.finish(r, 'error', ct=r.emitted, message='engine returned incomplete prompt logprobs')
                        return True
                    self.send(r, pf)
                fr = {'type': 'tokens', 'id': r.rid, 'ids': ids}
                if r.f.get('logprobs') is not None:
                    if len(r.lps) != len(ids):
                        log.error('req.%d: %d tokens with %d logprob entries', n, len(ids), len(r.lps))
                        self.finish(r, 'error', ct=r.emitted, message='engine returned mismatched logprobs')
                        return True
                    fr['logprobs'] = r.lps
                self.send(r, fr)
            r.lps = []
            return False
        if w[0] == 'm':
            v = [int(x) for x in w[1:]]
            self.c['rounds'], self.c['draft_tokens'], self.c['accepted'] = v[0], v[1], v[2]
            self.c['per_pos'] = v[3:]
            return False
        if w[0] == 'end':
            reason, sid, ct = w[1], int(w[2]), int(w[3])
            if ct != r.emitted:
                log.warning('req.%d: end says %d tokens, saw %d', n, ct, r.emitted)
            if reason not in ('stop', 'length', 'error'):
                reason = 'error'
            if not r.cancelled:
                self.finish(r, reason, ct=ct, stop_id=sid if sid >= 0 else None,
                            message='engine error' if reason == 'error' else None)
            log.info('req.%d end %s %d tokens', n, reason, ct)
            return True
        log.warning('req.%d: unknown resp line %r', n, ' '.join(w))
        return False

    def open_masters(self):
        for host in sorted({d.split(':', 1)[0] for d in self.a.rank_dirs if ':' in d}):
            rc = subprocess.call(SSH[:-1] + ['ControlMaster=yes', '-o', 'ControlPersist=yes', '-fN', host],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info('ssh master to %s: rc %d', host, rc)

    def quit(self):
        for d in self.a.rank_dirs:
            try:
                if ':' in d:
                    host, path = d.split(':', 1)
                    os.spawnvp(os.P_WAIT, 'ssh', SSH + [host, f'touch {path}/quit'])
                else:
                    open(f'{d}/quit', 'w').close()
            except Exception as e:
                log.error('quit to %s failed: %r', d, e)
        log.info('quit written to %d rank dirs', len(self.a.rank_dirs))
        for host in {d.split(':', 1)[0] for d in self.a.rank_dirs if ':' in d}:
            subprocess.call(SSH + ['-O', 'exit', host], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank-dirs', nargs='+', required=True,
                    help='engine --serve DIR per rank, rank 0 first (local); "host:/path" goes over ssh')
    ap.add_argument('--sock', default='/run/g6/engine.sock')
    ap.add_argument('--max-model-len', type=int, default=2048)
    ap.add_argument('--first-n', type=int, default=0, help='request number to start at (engine restart = 0)')
    ap.add_argument('--max-logprobs', type=int, default=20, help=f'largest k of logprobs / prompt_logprobs, 0..{ENGINE_MAX_LP} (the engine top lists hold {ENGINE_MAX_LP})')
    ap.add_argument('--max-plp-prompt', type=int, default=4096,
                    help='longest prompt of a request with prompt_logprobs (it runs as decode steps, 8 rows each)')
    ap.add_argument('--poll', type=float, default=0.001, help='resp poll interval, s')
    ap.add_argument('--stall-timeout', type=float, default=900.0, help='error a request after this long with no resp')
    a = ap.parse_args()
    if not 0 <= a.max_logprobs <= ENGINE_MAX_LP:
        ap.error(f'--max-logprobs must be 0..{ENGINE_MAX_LP} (the engine returns at most {ENGINE_MAX_LP} top entries)')
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if ':' in a.rank_dirs[0]:
        ap.error('the first --rank-dirs entry (rank 0) must be a local path')
    shim = Shim(a)
    shim.open_masters()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    os.makedirs(os.path.dirname(a.sock) or '.', exist_ok=True)
    server = await asyncio.start_unix_server(shim.serve, a.sock, limit=wire.MAX_FRAME)
    worker = asyncio.create_task(shim.run())
    log.info('g6shim on %s -> %s (max_model_len %d)', a.sock, ' '.join(a.rank_dirs), a.max_model_len)
    await stop.wait()
    worker.cancel()
    server.close()
    shim.quit()


if __name__ == '__main__':
    asyncio.run(main())
