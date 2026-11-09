#!/usr/bin/env python3
"""One chat prompt, streamed: thinking dim, then the answer, as the tokens arrive.

    ./gr8 query "Why is the sky blue?"                       against this head's API (http://127.0.0.1:8100)
    ./gr8 query --api http://HEAD:8100 --effort low "..."
    ./gr8 query --steer en-zh-v1:0.65 "..."                  select the loaded steering profile at scale 0.65
    ./gr8 query --record why.json "..."                      also save every delta with its arrival time
    ./gr8 query --replay why.json --window 40x10             play a recording back in real time
    ./gr8 query --window 40x10 --title "stock" "..."         draw into a fixed COLSxROWS box (a title row, then the
                                                             tail of the wrapped text), for side-by-side recordings

The model defaults to the first one /v1/models lists. Only the standard library is used.
"""
import argparse
import json
import sys
import time
import unicodedata
import urllib.request

DIM, BOLD, OFF = '\x1b[2m', '\x1b[1m', '\x1b[0m'


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=900)


def first_model(api):
    with urllib.request.urlopen(api + '/v1/models', timeout=30) as r:
        return json.load(r)['data'][0]['id']


def stream(a):
    """(seconds since the request, 'think' | 'answer', text) for every delta of the reply"""
    body = {'model': a.model or first_model(a.api), 'temperature': a.temperature, 'max_tokens': a.max_tokens,
            'stream': True, 'messages': [{'role': 'user', 'content': a.prompt}]}
    if a.system:
        body['messages'].insert(0, {'role': 'system', 'content': a.system})
    if a.effort:
        body['chat_template_kwargs'] = {'reasoning_effort': a.effort}
    if a.steer:
        name, _, scale = a.steer.partition(':')
        body.update(steering_profile=name, steering_scale=float(scale or 1))
    t0 = time.monotonic()
    with post(a.api + '/v1/chat/completions', body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            choices = json.loads(line[6:]).get('choices') or [{}]
            d = choices[0].get('delta') or {}
            t = round(time.monotonic() - t0, 3)
            if d.get('reasoning_content') or d.get('reasoning'):
                yield t, 'think', d.get('reasoning_content') or d.get('reasoning')
            if d.get('content'):
                yield t, 'answer', d['content']


def replay(path):
    """a recording's deltas, paced as they arrived"""
    t0 = time.monotonic()
    for t, kind, text in json.load(open(path))['events']:
        wait = t - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)
        yield t, kind, text


class Plain:
    """the deltas as they come: thinking dim, a blank line, then the answer"""

    def __init__(self):
        self.kind = None

    def add(self, kind, text):
        if kind != self.kind:
            sys.stdout.write(OFF + ('\n\n' if self.kind else '') + (DIM if kind == 'think' else ''))
            self.kind = kind
        sys.stdout.write(text)
        sys.stdout.flush()

    def done(self):
        sys.stdout.write(OFF + '\n')


class Window:
    """a COLS x ROWS box redrawn from its top left corner: the title, then the last rows of the wrapped text. A wide
    (CJK) character takes two columns; bold markers are dropped to save width."""

    def __init__(self, cols, rows, title):
        self.cols, self.rows, self.title = cols, rows, title
        self.spans = []
        sys.stdout.write('\x1b[?25l\x1b[2J')
        self.draw()

    def add(self, kind, text):
        style = DIM if kind == 'think' else ''
        if self.spans and self.spans[-1][0] != style:
            self.spans.append((DIM, '\n'))
        if self.spans and self.spans[-1][0] == style:
            self.spans[-1] = (style, self.spans[-1][1] + text)
        else:
            self.spans.append((style, text))
        self.draw()

    def wrap(self):
        """rows of (style, char); a row breaks at its last space when a Latin word would cross the edge"""
        rows, row, w = [], [], 0
        for style, text in self.spans:
            for c in text.replace('**', ''):
                if c == '\n':
                    rows.append(row)
                    row, w = [], 0
                    continue
                cw = 2 if unicodedata.east_asian_width(c) in 'WF' else 1
                if w + cw > self.cols:
                    cut = len(row)
                    if c != ' ' and c.isascii():
                        k = next((k for k in range(len(row) - 1, -1, -1) if row[k][1] == ' '), -1)
                        if k > 0 and all(ch.isascii() for _, ch in row[k + 1:]):
                            cut = k + 1
                    rows.append(row[:cut])
                    row = [x for x in row[cut:] if rows[-1] or x[1] != ' ']
                    w = sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for _, ch in row)
                    if c == ' ' and not row:
                        continue
                row.append((style, c))
                w += cw
        return rows + [row]

    def draw(self):
        body = self.wrap()[-(self.rows - 1):]
        out = ['\x1b[H', BOLD, self.title[:self.cols].ljust(self.cols), OFF, '\x1b[K']
        for i in range(self.rows - 1):
            out.append('\r\n')
            if i < len(body):
                out += [s + c + (OFF if s else '') for s, c in body[i]]
            out.append('\x1b[K')
        sys.stdout.write(''.join(out))
        sys.stdout.flush()

    def done(self):
        sys.stdout.write('\x1b[?25h')
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(prog='./gr8 query', description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[2:]))
    ap.add_argument('prompt', nargs='?')
    ap.add_argument('--api', default='http://127.0.0.1:8100')
    ap.add_argument('--model', help='default: the first model /v1/models lists')
    ap.add_argument('--system')
    ap.add_argument('--effort', help='chat_template_kwargs reasoning_effort (default: the template\'s)')
    ap.add_argument('--steer', metavar='PROFILE[:SCALE]')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--record', metavar='FILE')
    ap.add_argument('--replay', metavar='FILE')
    ap.add_argument('--window', metavar='COLSxROWS')
    ap.add_argument('--title', default='')
    ap.add_argument('--hold', type=float, default=0.0, help='--window: seconds to keep the last frame')
    a = ap.parse_args()
    if bool(a.prompt) == bool(a.replay):
        ap.error('give a prompt or --replay FILE')
    a.api = a.api.rstrip('/')
    if a.window:
        cols, _, rows = a.window.partition('x')
        view = Window(int(cols), int(rows), a.title)
    else:
        view = Plain()
    events = []
    try:
        for t, kind, text in (replay(a.replay) if a.replay else stream(a)):
            events.append([t, kind, text])
            view.add(kind, text)
    except KeyboardInterrupt:
        pass
    finally:
        if a.window and a.hold:
            time.sleep(a.hold)
        view.done()
        if a.record:
            with open(a.record, 'w') as f:
                json.dump({'prompt': a.prompt, 'steer': a.steer, 'effort': a.effort, 'events': events}, f,
                          ensure_ascii=False)


if __name__ == '__main__':
    main()
