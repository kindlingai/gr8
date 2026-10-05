"""HTTP client for the front end (OpenAI/vLLM shapes). Stdlib only."""
import json
import time
import urllib.request

MODEL = 'glm-5.3'


class Client:
    def __init__(self, url, model=MODEL, timeout=3600):
        self.url, self.model, self.timeout = url.rstrip('/'), model, timeout

    def _post(self, path, body):
        r = urllib.request.Request(self.url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        return urllib.request.urlopen(r, timeout=self.timeout)

    def tokenize(self, text):
        """-> list of token ids (the server's own tokenizer, no special tokens)."""
        with self._post('/tokenize', {'model': self.model, 'prompt': text, 'add_special_tokens': False}) as r:
            return json.load(r)['tokens']

    def tokenize_chat(self, messages, enable_thinking=False):
        """Chat-templated prompt ids (generation prompt added). -> list of token ids."""
        body = {'model': self.model, 'messages': messages, 'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': enable_thinking}}
        with self._post('/tokenize', body) as r:
            return json.load(r)['tokens']

    def count(self, text):
        with self._post('/tokenize', {'model': self.model, 'prompt': text, 'add_special_tokens': False}) as r:
            d = json.load(r)
        return d.get('count', len(d.get('tokens', [])))

    def complete(self, prompt, **kw):
        """Non-streaming /v1/completions. prompt = text or list of token ids. -> response dict."""
        with self._post('/v1/completions', dict(model=self.model, prompt=prompt, **kw)) as r:
            return json.load(r)

    def metrics(self):
        with urllib.request.urlopen(self.url + '/metrics', timeout=60) as r:
            return parse_metrics(r.read().decode())

    def chat(self, messages, max_tokens, temperature=0.7, effort='nothink', min_p=0.0):
        """Streamed /v1/chat/completions. Times the first generated token (reasoning or content).
        -> dict: ttft, decode_s, tokens (completion_tokens from usage), prompt_tokens, cached_tokens, content,
        reasoning, finish_reason."""
        kwargs = {'enable_thinking': False} if effort == 'nothink' else {'reasoning_effort': effort}
        body = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens, 'stream': True,
                'temperature': temperature, 'min_p': min_p, 'stream_options': {'include_usage': True},
                'chat_template_kwargs': kwargs}
        return self._stream('/v1/chat/completions', body)

    def complete_stream(self, prompt_ids, max_tokens, temperature=0.0):
        """Streamed /v1/completions on exact prompt token ids (no chat template). Same fields as chat()."""
        body = {'model': self.model, 'prompt': prompt_ids, 'max_tokens': max_tokens, 'stream': True,
                'temperature': temperature, 'stream_options': {'include_usage': True}}
        return self._stream('/v1/completions', body)

    def _stream(self, path, body):
        t0 = time.time()
        tfirst = None
        usage = None
        content, reasoning = [], []
        finish = None
        with self._post(path, body) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b'data: ') or line == b'data: [DONE]':
                    continue
                d = json.loads(line[6:])
                if d.get('usage'):
                    usage = d['usage']
                ch = d.get('choices') or []
                if not ch:
                    continue
                dl = ch[0].get('delta') or {}
                c = dl.get('content') or ch[0].get('text') or ''
                rs = (dl.get('reasoning') or '') + (dl.get('reasoning_content') or '')
                if c or rs:
                    if tfirst is None:
                        tfirst = time.time()
                    if c:
                        content.append(c)
                    if rs:
                        reasoning.append(rs)
                finish = ch[0].get('finish_reason') or finish
        t1 = time.time()
        usage = usage or {}
        details = usage.get('prompt_tokens_details') or {}
        n = usage.get('completion_tokens', 0)
        tfirst = tfirst or t1
        return {'t0': t0, 'ttft': tfirst - t0, 'decode_s': t1 - tfirst, 'total_s': t1 - t0, 'tokens': n,
                'prompt_tokens': usage.get('prompt_tokens', 0), 'cached_tokens': details.get('cached_tokens'),
                'content': ''.join(content), 'reasoning': ''.join(reasoning), 'finish_reason': finish}


def parse_metrics(text):
    """Prometheus text -> {metric name with labels: float}."""
    out = {}
    for ln in text.splitlines():
        if not ln or ln.startswith('#'):
            continue
        k, _, v = ln.rpartition(' ')
        try:
            out[k] = out.get(k, 0.0) + float(v)
        except ValueError:
            pass
    return out


def msum(m, name):
    """Sum of a metric over its label sets (parse_metrics keys keep the labels)."""
    return sum(v for k, v in m.items() if k == name or k.startswith(name + '{'))


def spec_snapshot(m):
    """Drafter and prefix-cache counters from a parsed /metrics dict."""
    pos = {}
    for k, v in m.items():
        if k.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{') and 'position="' in k:
            i = int(k.split('position="')[1].split('"')[0])
            pos[i] = pos.get(i, 0.0) + v
    return {'drafts': msum(m, 'vllm:spec_decode_num_drafts_total'),
            'draft_tokens': msum(m, 'vllm:spec_decode_num_draft_tokens_total'),
            'accepted': msum(m, 'vllm:spec_decode_num_accepted_tokens_total'),
            'per_pos': [pos.get(i, 0.0) for i in range(max(pos) + 1)] if pos else [],
            'prefix_queries': msum(m, 'vllm:prefix_cache_queries_total'),
            'prefix_hits': msum(m, 'vllm:prefix_cache_hits_total')}


def spec_delta(a, b):
    """Counters between two snapshots -> drafts, draft_tokens, accepted, acceptance rate, tokens per step."""
    d = {k: b[k] - a[k] for k in ('drafts', 'draft_tokens', 'accepted', 'prefix_queries', 'prefix_hits')}
    n = max(len(a['per_pos']), len(b['per_pos']))
    pa = a['per_pos'] + [0.0] * (n - len(a['per_pos']))
    pb = b['per_pos'] + [0.0] * (n - len(b['per_pos']))
    drafts = d['drafts']
    d['per_pos_survival'] = [round((y - x) / drafts, 4) for x, y in zip(pa, pb)] if drafts > 0 else []
    d['acceptance'] = round(d['accepted'] / d['draft_tokens'], 4) if d['draft_tokens'] > 0 else None
    d['tokens_per_step'] = round(1 + d['accepted'] / drafts, 3) if drafts > 0 else None
    return d
