"""The pre-engine handshake, and the line-JSON RPC the supervisors use on the LAN control port.

Every worker sends the head one `hello`: its rank, LAN address, rails, G6_HOME, the canonical option set and its
digest, the model set id, the engine build and the failures of its own checks. The head compares the hello with its
own and answers at once: `refused` at the first difference, with the field and every box's value, else `wait`. The
worker then asks for the `verdict` about once a second. Once every rank has said hello and the rails answer, the
verdict is `ok` with the boot epoch and the assembled IPS/IPS2. A refusal goes to every box, and every box exits 4.

The head answers `retry` while it has not finished its own checks, so a worker waits for a slow head for as long as
the head answers. The head drops a rank that stops asking for 30 s and refuses the boot, naming that rank.
"""
import json
import socket
import threading
import time

from fabric import same_subnet

PROTO = 1
POLL_S = 1.0
DROP_S = 30.0


def label(h):
    return f'rank {h["rank"]} ({h["lan"]})'


def refusal(field, have, value, detail='', code=4):
    """A refusal naming field, with value(h) for every box that has said hello."""
    vals = {label(h): value(h) for h in sorted(have.values(), key=lambda h: h['rank'])}
    return {'t': 'refused', 'field': field, 'values': vals, 'detail': detail, 'exit': code}


def word_diff(a, b):
    """The words of a that b lacks, for an engine.env or engine.args that differs."""
    other = set(b.split())
    return ' '.join(w for w in a.split() if w not in other) or '(nothing more)'


def compare(ref, h, peer, have):
    """None when hello h agrees with the head's hello ref, else a refusal. have: {rank: hello} so far, ref included;
    peer: the address the hello came from. Checks run in a fixed order and the first failure is the one reported."""
    tp = int(ref['opts']['tp'])
    if h.get('proto') != PROTO:
        return refusal('handshake protocol', {**have, -1: h}, lambda x: str(x.get('proto')),
                       'the boxes run different gr8 versions')
    if not isinstance(h.get('rank'), int) or not 0 <= h['rank'] < tp:
        return refusal('rank', {**have, -1: h}, lambda x: str(x.get('rank')), f'a rank must be 0..{tp - 1}')
    both = {**have, h['rank'] if h['rank'] not in have else -1: h}
    if h['opts'].get('tp') != ref['opts']['tp']:
        return refusal('tp', both, lambda x: x['opts'].get('tp'))
    if h['members'] != ref['members']:
        return refusal('member list', both, lambda x: ','.join(x['members']),
                       'every box must list the same peers, in the same order')
    if h['rank'] in have:
        return refusal('rank', both, lambda x: str(x['rank']), f'two boxes claim rank {h["rank"]}')
    if ref['members'][h['rank']] != h['lan'] or peer != h['lan']:
        return refusal('rank', both, lambda x: f'{x["rank"]} from {x["lan"]}',
                       f'rank {h["rank"]} is {ref["members"][h["rank"]]} in the member list, but the hello came from {peer}')
    if h.get('fails') or ref.get('fails'):   # Head.wait_all lists every failing box once all have said hello
        return None
    if h['engine_build'] != ref['engine_build']:
        return refusal('engine build (sha256 of bin/g6m4 and bin/lib)', both, lambda x: x['engine_build'][:16],
                       'build the engine once and copy bin/ to every box')
    if h['set_id'] != ref['set_id']:
        return refusal('model set id', both, lambda x: f'{x["set_id"][:16]} ({x["set_source"]})',
                       'the boxes hold different models (pack manifests, drafter or tokenizer differ)')
    if h['opts_digest'] != ref['opts_digest']:
        keys = sorted(set(h['opts']) | set(ref['opts']))
        k = next(k for k in keys if h['opts'].get(k) != ref['opts'].get(k))
        if k.startswith('engine.'):
            return refusal(f'option {k}', both, lambda x: word_diff(x['opts'].get(k, ''), (h if x is ref else ref)['opts'].get(k, '')),
                           'each box shows the words the other box lacks')
        return refusal(f'option {k}', both, lambda x: x['opts'].get(k, '(unset)'))
    a, b = ref['rails'], h['rails']
    if not ref['opts'].get('IPS') and same_subnet(a[0], b[1]) and same_subnet(a[1], b[0]) and not same_subnet(a[0], b[0]):
        return refusal('rails', both, lambda x: ','.join(x['rails']),
                       f'{label(h)} has its rails the other way round: rail 0 is the lowest RoCE v2 address on every '
                       'box, so each PCIe root must sit in the same subnet on every box')
    return None


class Head:
    """The head's side: takes hellos and verdict polls from the control server's threads."""

    def __init__(self, own, drop_s=DROP_S, clock=time.monotonic):
        self.own, self.have, self.verdict = own, {0: own}, None
        self.seen, self.first, self.drop_s, self.clock = {}, None, drop_s, clock
        self.cv = threading.Condition()
        self.tp = int(own['opts']['tp']) if own else 0

    @classmethod
    def refusing(cls, verdict):
        """A head that failed before it could compare anything: every hello and poll gets this verdict."""
        h = cls(None)
        h.have, h.verdict = {}, verdict
        return h

    def rpc(self, m, peer):
        """A hello or a verdict poll. Never blocks."""
        with self.cv:
            v = self.verdict
            if m.get('t') == 'hello':
                if v and v['t'] == 'ok':
                    return {'t': 'refused', 'field': 'booted', 'values': {}, 'exit': 4,
                            'detail': f'the cluster booted at epoch {v["epoch"]}: ./gr8 glm53 down first'}
                if v:
                    return v
                bad = compare(self.own, m, peer, self.have)
                if bad:
                    self.verdict = bad
                    self.cv.notify_all()
                    return bad
                self.have[m['rank']] = m
                self.seen[m['rank']] = self.clock()
                self.first = self.first or self.clock()
                self.cv.notify_all()
                return {'t': 'wait'}
            r = m.get('rank')
            if r in self.have and r != 0 and self.have[r]['lan'] == peer:
                self.seen[r] = self.clock()
                return v or {'t': 'wait'}
            return v or {'t': 'retry', 'detail': f'no hello from rank {r} at {peer}'}

    def failed_checks(self, why=''):
        """A refusal listing every box that failed its checks, or None."""
        bad = {r: h for r, h in self.have.items() if h.get('fails')}
        if not bad:
            return None
        return refusal('checks', bad, lambda x: '; '.join(x['fails']),
                       f'{len(bad)} of {self.tp} boxes failed their checks' + (f'. {why}' if why else ''))

    def wait_all(self, timeout, stop=None):
        """Wait until every rank has said hello (True), or a verdict, a stop, a silent rank or the timeout (False).
        The timeout runs from the first hello. A box that failed its checks waits too, so that the refusal names
        every failing box."""
        with self.cv:
            while self.verdict is None and len(self.have) < self.tp:
                now = self.clock()
                if stop is not None and stop.is_set():
                    return False
                gone = [r for r, t in self.seen.items() if now - t > self.drop_s]
                if gone:
                    why = (f'{", ".join(label(self.have[r]) for r in gone)} stopped asking for the verdict for '
                           f'{self.drop_s:.0f}s')
                    self.verdict = self.failed_checks(why) or refusal(
                        'join', {r: self.have[r] for r in gone}, lambda x: 'stopped asking', why)
                elif timeout and self.first is not None and now - self.first >= timeout:
                    missing = [f'rank {r} ({self.own["members"][r]})' for r in range(self.tp) if r not in self.have]
                    why = f'no hello within {timeout:.0f}s from {", ".join(missing)}'
                    self.verdict = self.failed_checks(why) or refusal('join', self.have, lambda x: 'said hello', why,
                                                                      code=1)
                else:
                    self.cv.wait(0.5)
            if self.verdict is None:
                self.verdict = self.failed_checks()
            self.cv.notify_all()
            return self.verdict is None

    def decide(self, verdict):
        with self.cv:
            if self.verdict is None:
                self.verdict = verdict
            self.cv.notify_all()


def describe(v):
    """A refusal as the lines every box prints."""
    out = [f'handshake refused: {v["field"]}' + (f': {v["detail"]}' if v.get('detail') else '')]
    out += [f'  {k}: {val}' for k, val in v.get('values', {}).items()]
    return '\n'.join(out)


# ---- the line-JSON RPC ---------------------------------------------------------------------------------------------
def call(host, port, msg, timeout=5.0, bind=None):
    """Send one JSON object, read one back."""
    with socket.create_connection((host, port), timeout=timeout, source_address=(bind, 0) if bind else None) as s:
        s.sendall((json.dumps(msg) + '\n').encode())
        f = s.makefile('rb')
        line = f.readline(1 << 24)
        if not line:
            raise ConnectionError(f'{host}:{port} closed the connection without an answer')
        return json.loads(line)


def send_hello(head, port, hello, bind, timeout, log, stop=None, poll_s=POLL_S, drop_s=DROP_S):
    """The worker's side: send the hello, then ask for the verdict until there is one. timeout bounds the wait for a
    head that does not answer at all. Once the head answers, the worker waits as long as the head keeps answering."""
    t0, said, answered, ask = time.monotonic(), -60.0, None, hello
    while True:
        if stop is not None and stop.is_set():
            return {'t': 'refused', 'field': 'stopped', 'values': {}, 'detail': 'down while waiting for the head',
                    'exit': 1}
        now = time.monotonic()
        try:
            r = call(head, port, ask, timeout=5, bind=bind)
            answered = now
        except (OSError, ValueError) as e:
            if answered is None and timeout and now - t0 >= timeout:
                return {'t': 'refused', 'field': 'join', 'values': {}, 'exit': 1,
                        'detail': f'the head {head}:{port} did not answer within {timeout:.0f}s'}
            if answered is not None and now - answered >= drop_s:
                return {'t': 'refused', 'field': 'join', 'values': {}, 'exit': 1,
                        'detail': f'the head {head}:{port} stopped answering for {drop_s:.0f}s'}
            r = {'t': 'retry', 'detail': str(e)}
        if r.get('t') == 'wait':
            ask = {'t': 'verdict', 'rank': hello['rank']}
        elif r.get('t') == 'retry':
            ask = hello
            if now - t0 - said >= 60:
                log(f'waiting for the head {head}:{port}: {r.get("detail", "")}')
                said = now - t0
        else:
            return r
        time.sleep(poll_s)
