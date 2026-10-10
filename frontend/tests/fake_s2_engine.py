#!/usr/bin/env python3
"""Fake engine for g6s2shim tests: the file side of `g6m4 --serve DIR` (a DEC_S2 build), no model, no GPU.

It follows the engine code (engine/runtime/s2/ingress.h, sessions.h, service.h, scheduler.h):
  - writes DIR/ready.s2 (after --ready-delay), publishes DIR/status.s2
  - rank 0 reads DIR/cmd.0, cmd.1, ... in order (ADMIT <epoch> <id> <hash> | CANCEL <epoch> <id> 0); the request ids must
    increase; req.<id> must exist on EVERY rank dir with the announced FNV-1a hash when the request is admitted
  - a request waits in a FIFO until a session slot (--sessions) and a KV run (--pool-rows) are free, then gets
    an "a" line, then decodes together with the other sessions: every round (--round-ms, the same for any number of
    sessions) each session commits 1 + accepted tokens ("t" lines); at the end an "m" line and "end"
  - CANCEL: queued -> only "end abort -1 0"; resident -> "m ..." + "end abort -1 <emitted>", slot and run freed;
    unknown (already ended) -> nothing
  - anything the real engine would reject (bad header, epoch, duplicate/decreasing id, hash mismatch, request over the
    pool) ends the epoch: DIR/abort.s2 on every rank dir, exit 3. DIR/quit (any rank dir) = clean exit 0.
Output tokens are a pure function of the prompt and the position, so a stream can be checked for cross-talk.

usage: fake_s2_engine.py DIR0 [DIR1 ...] [--epoch 7] [--sessions 4] [--pool-rows 20000] [--mtp 2] [--round-ms 20]
"""
import argparse
import collections
import json
import os
import sys
import time

VOCAB = 154880
GUARD_FACTOR = 2   # guard = 2 * mtp (Config::guard)
ACC = [2, 0, 1, 2, 1, 0]


def payload_hash(data: bytes) -> int:
    h = 14695981039346656037
    for c in data:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def token_at(prompt, made):
    """The made-th generated token (1-based) of a request: depends on the prompt only."""
    return 1000 + (sum(prompt) * 7 + made * 13) % 5000


def lp_line(t, k, tag='l'):
    """the engine's "l" line for sampled token t with k top entries (deterministic stand-in values); tag 'p <j>' for
    prompt token j"""
    return f'{tag} -0.25 1 {k}' + ''.join(f' {t + i}:{-0.25 - i}' for i in range(k)) + '\n'


def number(s):
    if not s.isdigit():
        raise ValueError(f'bad number {s!r}')
    return int(s)


def parse_request(data: bytes):
    line, _, rest = data.partition(b'\n')
    if not _:
        raise ValueError('request missing prompt line')
    w = line.decode().split()
    if len(w) not in (10, 12, 14, 16) or w[0] != 'n_gen' or w[2] != 'temp' or w[4] != 'seed' or w[6] != 'min' or w[8] != 'stop':
        raise ValueError(f'invalid request header {line!r}')
    r = {'n_gen': number(w[1]), 'temp': float(w[3]), 'seed': number(w[5]), 'min': number(w[7]),
         'stop': [] if w[9] == '-' else [number(x) for x in w[9].split(',')], 'top_p': 1.0, 'lp': None, 'plp': None}
    if len(w) >= 12:
        if w[10] != 'top_p':
            raise ValueError('invalid top_p header')
        r['top_p'] = float(w[11])
    for i in range(12, len(w), 2):     # lp then plp, each optional
        if w[i] not in ('lp', 'plp') or r[w[i]] is not None or (w[i] == 'lp' and r['plp'] is not None) \
                or not 0 <= number(w[i + 1]) <= 20:
            raise ValueError('invalid lp/plp header')
        r[w[i]] = number(w[i + 1])
    return r, rest


class Fake:
    def __init__(self, a):
        self.a = a
        self.d0 = a.dirs[0]
        self.guard = GUARD_FACTOR * a.mtp
        self.queue = collections.deque()     # {'id', 'prompt', 'req'}
        self.res = {}                        # slot -> session dict
        self.pages_used = 0
        self.last_id = -1
        self.next_cmd = 0
        self.gen = [0] * 8
        self.log = open(os.path.join(self.d0, 'fake.log'), 'a', buffering=1)
        self.max_resident = 0
        self.dirty = True

    def say(self, s):
        self.log.write(f'{time.monotonic():.4f} {s}\n')

    def die(self, why):
        self.say(f'EPOCH FAILED: {why}')
        for d in self.a.dirs:
            open(os.path.join(d, 'abort.s2'), 'w').write('fake engine epoch failed\n')
        sys.exit(3)

    # ---- control side ---------------------------------------------------------------------------------------------
    def poll_commands(self):
        while True:
            p = os.path.join(self.d0, f'cmd.{self.next_cmd}')
            if not os.path.exists(p):
                return
            w = open(p).read().split()
            try:
                if len(w) != 4 or w[0] not in ('ADMIT', 'CANCEL'):
                    raise ValueError(f'invalid ingress command {w}')
                epoch, rid, h = number(w[1]), number(w[2]), number(w[3])
                if epoch != self.a.epoch or (w[0] == 'CANCEL' and h != 0):
                    raise ValueError('ingress epoch/cancel mismatch')
                self.next_cmd += 1
                (self.admit if w[0] == 'ADMIT' else self.cancel)(rid, h)
            except ValueError as e:
                self.die(str(e))

    def admit(self, rid, h):
        if rid <= self.last_id:
            self.die('native request ordinals must increase')
        self.last_id = rid
        datas = []
        for d in self.a.dirs:   # rank 0 reads it at ADMIT, the others at the fence: both must find the same bytes
            p = os.path.join(d, f'req.{rid}')
            if not os.path.exists(p):
                self.die(f'req.{rid} missing in {d} when the ADMIT arrived')
            datas.append(open(p, 'rb').read())
        if any(payload_hash(x) != h for x in datas):
            self.die('published request payload mismatch')
        try:
            req, rest = parse_request(datas[0])
            prompt = [number(x) for x in rest.decode().strip().split(',')]
            if not (req['n_gen'] > 0 and req['min'] <= req['n_gen'] and prompt):
                raise ValueError('invalid request sampling/length')
            if max(prompt) >= VOCAB:   # the real engine indexes the embedding table with it, unchecked
                raise ValueError('prompt token id outside the vocabulary')
        except ValueError as e:
            self.die(str(e))
        rows = len(prompt) + req['n_gen'] + self.guard
        if rows > self.a.pool_rows:
            self.die('request exceeds pool including guard')
        self.say(f'ADMIT {rid} prompt {len(prompt)} n_gen {req["n_gen"]} min {req["min"]} stop {req["stop"]}')
        self.queue.append({'id': rid, 'prompt': prompt, 'req': req, 'pages': (rows + 63) // 64})
        self.dirty = True

    def cancel(self, rid, _):
        for q in self.queue:
            if q['id'] == rid:
                self.queue.remove(q)
                tmp = os.path.join(self.d0, f'resp.{rid}.tmp')
                open(tmp, 'w').write('end abort -1 0\n')
                os.rename(tmp, os.path.join(self.d0, f'resp.{rid}'))
                self.say(f'CANCEL {rid} queued')
                self.dirty = True
                return
        for slot, s in list(self.res.items()):
            if s['id'] == rid:
                self.say(f'CANCEL {rid} resident after {s["emitted"]} tokens')
                self.terminal(slot, 'abort')
                return
        self.say(f'CANCEL {rid} unknown')

    # ---- scheduler ------------------------------------------------------------------------------------------------
    def try_admit(self):
        while self.queue and len(self.res) < self.a.sessions:
            q = self.queue[0]
            if self.pages_used + q['pages'] > self.a.pool_rows // 64:
                return
            self.queue.popleft()
            slot = min(set(range(8)) - set(self.res))
            self.gen[slot] += 1
            self.pages_used += q['pages']
            f = open(os.path.join(self.d0, f'resp.{q["id"]}'), 'w', buffering=1)
            f.write(f'a {self.a.epoch} {slot} {self.gen[slot]} {self.pages_used - q["pages"]} {q["pages"]}\n')
            self.res[slot] = dict(q, slot=slot, f=f, emitted=0, rounds=0, acc=[0] * (self.a.mtp + 1), stop_id=-1,
                                  ready_at=time.monotonic() + self.a.prefill_ms / 1e3, done=False)
            self.max_resident = max(self.max_resident, len(self.res))
            self.say(f'RESIDENT {len(self.res)} (admitted {q["id"]} slot {slot})')
            self.dirty = True

    def terminal(self, slot, reason=None):
        s = self.res.pop(slot)
        f, m = s['f'], self.a.mtp
        f.write(f'm {s["rounds"]} {s["rounds"] * m} {s["emitted"] - s["rounds"]}')
        for i in range(m):
            f.write(' %d' % sum(s['acc'][i + 1:]))
        f.write(f'\nend {reason or ("stop" if s["stop_id"] >= 0 else "length")} {s["stop_id"]} {s["emitted"]}\n')
        f.close()
        self.pages_used -= s['pages']
        self.say(f'END {s["id"]} {reason or "natural"} {s["emitted"]} tokens (resident {len(self.res)})')
        self.dirty = True

    def round(self):
        now = time.monotonic()
        for slot, s in list(self.res.items()):
            if s['ready_at'] > now:
                continue
            acc = ACC[(s['rounds'] + s['id']) % len(ACC)]
            req, ids, done = s['req'], [], False
            for _ in range(1 + acc):
                n = s['emitted'] + len(ids) + 1
                stop_here = req['stop'] and self.a.stop_at and n == max(self.a.stop_at, req['min'] + 1)
                ids.append(req['stop'][0] if stop_here else token_at(s['prompt'], n))
                if stop_here:
                    s['stop_id'] = ids[-1]
                    done = True
                if n >= req['n_gen']:
                    done = True
                if done:
                    break
            s['emitted'] += len(ids)
            s['rounds'] += 1
            s['acc'][len(ids) - 1] += 1
            if req['plp'] is not None and s['rounds'] == 1:     # the first round (counted above)
                s['f'].write(''.join(lp_line(t, req['plp'], f'p {j}') for j, t in enumerate(s['prompt'][1:], 1)))
            if req['lp'] is not None:
                s['f'].write(''.join(lp_line(t, req['lp']) for t in ids))
            s['f'].write('t ' + ' '.join(map(str, ids)) + '\n')
            if done:
                self.terminal(slot)

    def status(self):
        free = self.a.pool_rows // 64 - self.pages_used
        body = json.dumps({'epoch': self.a.epoch, 'resident': len(self.res), 'queued': len(self.queue), 'retiring': 0,
                           'reserved_pages': self.pages_used, 'free_pages': free, 'largest_free_run': free,
                           'admission_stalls': 0, 'fragmentation_stalls': 0}) + '\n'
        tmp = os.path.join(self.d0, 'status.s2.tmp')
        open(tmp, 'w').write(body)
        os.rename(tmp, os.path.join(self.d0, 'status.s2'))

    def run(self):
        a = self.a
        for d in a.dirs:   # the real engine insists on fresh dirs
            for f in ('ready', 'ready.s2', 'abort.s2', 'quit', 'cmd.0', 'req.0'):
                if os.path.exists(os.path.join(d, f)):
                    self.die(f'S2 directory {d} is not fresh')
        time.sleep(a.ready_delay)
        tmp = os.path.join(self.d0, 'ready.s2.tmp')
        open(tmp, 'w').write(f'G6S2 2 {a.epoch} {len(a.dirs)} {a.pool_rows} {a.mtp} {self.guard} {a.max_rows} 20\n')
        os.rename(tmp, os.path.join(self.d0, 'ready.s2'))
        self.say('ready')
        next_round = time.monotonic()
        while True:
            if any(os.path.exists(os.path.join(d, 'quit')) for d in a.dirs):
                self.say(f'quit (max resident {self.max_resident})')
                return 0
            if any(os.path.exists(os.path.join(d, 'abort.s2')) for d in a.dirs):
                self.say('abort.s2 seen')
                return 3
            self.poll_commands()
            self.try_admit()
            now = time.monotonic()
            if now >= next_round:
                if self.res:
                    self.round()
                    self.try_admit()
                next_round = now + a.round_ms / 1e3
            if self.dirty:
                self.status()
                self.dirty = False
            time.sleep(0.0005)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--epoch', type=int, default=7)
    ap.add_argument('--sessions', type=int, default=4)
    ap.add_argument('--pool-rows', type=int, default=20000)
    ap.add_argument('--mtp', type=int, default=2)
    ap.add_argument('--max-rows', type=int, default=8)
    ap.add_argument('--round-ms', type=float, default=20.0)
    ap.add_argument('--prefill-ms', type=float, default=30.0)
    ap.add_argument('--ready-delay', type=float, default=0.0)
    ap.add_argument('--stop-at', type=int, default=0, help='first stop id is token #N of a request that has stop ids (0 = never)')
    sys.exit(Fake(ap.parse_args()).run())
