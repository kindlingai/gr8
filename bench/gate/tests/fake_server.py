"""A small stand-in for the front end (stdlib only): enough of /tokenize, /v1/completions, /v1/chat/completions
(streamed) and /metrics for the harness tests. Not a model: logprobs are a fixed function of the token ids."""
import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = 50


def toks(text):
    return re.findall(r'\w+|\S', text)


def ids_of(text):
    return [int(hashlib.md5(t.encode()).hexdigest(), 16) % VOCAB for t in toks(text)]


def dist(prev_ids, drift):
    """Next-token distribution after prev_ids: a softmax over VOCAB logits derived from the last 3 ids."""
    h = hashlib.md5(str(prev_ids[-3:]).encode()).digest()
    lg = [(h[i % 16] / 16.0) * 3 + (0.1 * drift * ((h[(i * 5) % 16] % 7) - 3)) for i in range(VOCAB)]
    m = max(lg)
    z = sum(math.exp(x - m) for x in lg)
    return [x - m - math.log(z) for x in lg]


class State:
    def __init__(self):
        self.drift = 0.0          # perturbs logprobs
        self.retrieve = True      # chat answers the needle question correctly
        self.cache_hit = 0        # prompt tokens reported as cached
        self.counters = {'drafts': 0, 'draft': 0, 'acc': 0, 'hits': 0, 'queries': 0}
        self.lock = threading.Lock()
        self.chat_prompts = []
        self.completion_prompts = []
        self.fail_nth = 0         # the Nth chat request from now (1-based) ends with finish_reason 'error'
        self.chat_calls = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *a):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        st = self.server.st
        c = st.counters
        lines = [f'vllm:spec_decode_num_drafts_total{{engine="0"}} {c["drafts"]}',
                 f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {c["draft"]}',
                 f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {c["acc"]}',
                 f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="0"}} {c["acc"] * 0.6}',
                 f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="1"}} {c["acc"] * 0.4}',
                 f'vllm:prefix_cache_queries_total{{engine="0"}} {c["queries"]}',
                 f'vllm:prefix_cache_hits_total{{engine="0"}} {c["hits"]}']
        data = '\n'.join(lines).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        st = self.server.st
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            if 'messages' in body:
                text = ''.join(m['content'] for m in body['messages']) + ' assistant'
            else:
                text = body['prompt']
            ids = ids_of(text)
            return self._json({'count': len(ids), 'tokens': ids})
        if self.path == '/v1/completions':
            return self._completions(st, body)
        if self.path == '/v1/chat/completions':
            return self._chat(st, body)
        self.send_error(404)

    def _completions(self, st, body):
        p = body['prompt']
        n = body.get('max_tokens', 16)
        if body.get('stream'):
            return self._stream_completions(st, p, n)
        gen = []
        for _ in range(n):
            lp = dist(p + gen, 0.0)
            gen.append(max(range(VOCAB), key=lambda i: lp[i]))
        choice = {'index': 0, 'text': ' '.join(map(str, gen)), 'token_ids': gen if body.get('return_token_ids') else None,
                  'prompt_logprobs': None}
        k = body.get('prompt_logprobs')
        if k is not None:
            out = [None]
            for j in range(1, len(p)):
                lp = dist(p[:j], st.drift * (1 + (j % 5)))
                order = sorted(range(VOCAB), key=lambda i: -lp[i])
                ent = {str(i): {'logprob': lp[i], 'rank': r + 1, 'decoded_token': str(i)} for r, i in enumerate(order[:k])}
                if str(p[j]) not in ent:
                    ent[str(p[j])] = {'logprob': lp[p[j]], 'rank': order.index(p[j]) + 1, 'decoded_token': str(p[j])}
                out.append(ent)
            choice['prompt_logprobs'] = out
        self._json({'choices': [choice], 'usage': {'prompt_tokens': len(p), 'completion_tokens': n}})

    def _stream_completions(self, st, p, n):
        """Streamed completions on token ids: n tokens, one text chunk per 4, drafter counters like chat."""
        with st.lock:
            st.completion_prompts.append(list(p))
            c = st.counters
            c['drafts'] += n // 4
            c['draft'] += n
            c['acc'] += n // 2
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for i in range(0, n, 4):
            chunk = {'choices': [{'index': 0, 'text': 'tok ' * min(4, n - i)}]}
            self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
            time.sleep(0.002)
        self.wfile.write(b'data: ' + json.dumps({'choices': [{'index': 0, 'text': '', 'finish_reason': 'length'}]}).encode()
                         + b'\n\n')
        usage = {'prompt_tokens': len(p), 'completion_tokens': n, 'total_tokens': len(p) + n}
        self.wfile.write(b'data: ' + json.dumps({'choices': [], 'usage': usage}).encode() + b'\n\ndata: [DONE]\n\n')

    def _chat(self, st, body):
        text = body['messages'][-1]['content']
        n_prompt = len(toks(text))
        st.chat_prompts.append(text)
        m = re.search(r'# PASSPHRASE: (\S+)', text)
        ans = (f'You asked about the passphrase. It is {m.group(1)}.' if m and st.retrieve
               else 'I could not find it.')
        n = body['max_tokens']
        words = (ans + ' ' + 'word ' * n).split()[:n]
        with st.lock:
            st.chat_calls += 1
            reason = 'error' if st.fail_nth and st.chat_calls == st.fail_nth else 'length'
            c = st.counters
            c['drafts'] += n // 3
            c['draft'] += n
            c['acc'] += n // 2
            c['queries'] += n_prompt
            c['hits'] += st.cache_hit
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for i in range(0, len(words), 4):
            chunk = {'choices': [{'index': 0, 'delta': {'content': ' '.join(words[i:i + 4]) + ' '}}]}
            self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
            time.sleep(0.002)
        end = {'choices': [{'index': 0, 'delta': {}, 'finish_reason': reason}]}
        self.wfile.write(b'data: ' + json.dumps(end).encode() + b'\n\n')
        usage = {'prompt_tokens': n_prompt, 'completion_tokens': n, 'total_tokens': n_prompt + n}
        if st.cache_hit:
            usage['prompt_tokens_details'] = {'cached_tokens': st.cache_hit}
        self.wfile.write(b'data: ' + json.dumps({'choices': [], 'usage': usage}).encode() + b'\n\ndata: [DONE]\n\n')


def start():
    srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    srv.st = State()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f'http://127.0.0.1:{srv.server_address[1]}'
