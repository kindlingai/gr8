"""--record: the raw streamed output of the concurrency and agentic stages, saved with arrival times for a replay
(bench/gif.py draws one as an animated GIF).

An event is [t, kind, value]: t is seconds since that request was sent. The kinds:
  think    reasoning text
  answer   content text
  tool     a tool-call delta: {"index": i} plus whichever of "id", "name" and "arguments" (text) the delta carries
  finish   the finish reason
  usage    the usage object the server sends last
think and answer match the events of tools/query/query.py --record."""
import json
import time
import urllib.error
import urllib.request

FORMAT = 'gr8bench-record-1'


def add(events, t, chunk):
    """Append the events of one parsed SSE chunk to events."""
    t = round(t, 3)
    for ch in chunk.get('choices') or []:
        d = ch.get('delta') or {}
        think = (d.get('reasoning') or '') + (d.get('reasoning_content') or '')
        if think:
            events.append([t, 'think', think])
        text = d.get('content') or ch.get('text') or ''
        if text:
            events.append([t, 'answer', text])
        for tc in d.get('tool_calls') or []:
            fn = tc.get('function') or {}
            ev = {'index': tc.get('index', 0)}
            if tc.get('id'):
                ev['id'] = tc['id']
            if fn.get('name'):
                ev['name'] = fn['name']
            if fn.get('arguments'):
                ev['arguments'] = fn['arguments']
            events.append([t, 'tool', ev])
        if ch.get('finish_reason'):
            events.append([t, 'finish', ch['finish_reason']])
    if chunk.get('usage'):
        events.append([t, 'usage', chunk['usage']])


def _get(url):
    """JSON (or text) from a GET, or None when the server does not answer it."""
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            body = r.read().decode(errors='replace')
    except (urllib.error.URLError, OSError, ValueError):
        return None
    try:
        return json.loads(body)
    except ValueError:
        return body.strip()[:500]


def context(cl, label, hosts, effort):
    """What a recording was made against: kept at the top of every file."""
    ctx = {'model': cl.model, 'models': [m.get('id') for m in cl.models()], 'label': label, 'hosts': hosts,
           'effort': effort, 'recorded': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    for path in ('/health', '/version'):
        v = _get(cl.url + path)
        if v is not None:
            ctx[path.strip('/')] = v
    return ctx


class Recorder:
    """Writes DIR/NAME.json, each holding the run's context and one stage's streams."""

    def __init__(self, path, meta):
        self.dir, self.meta = path, meta
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, name, body):
        out = self.dir / f'{name}.json'
        tmp = out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'format': FORMAT, 'meta': self.meta, **body}, ensure_ascii=False))
        tmp.replace(out)
        return out
