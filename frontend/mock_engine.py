#!/usr/bin/env python3
"""Mock gr8 engine speaking PROTOCOL.md, so the front end and the old rulers run with no model.

Deterministic: the output depends only on the prompt ids and request fields (never on temperature/seed).
It decodes the prompt to pick a canned reply:
  - prompt defines tools            -> short reasoning, then a GLM tool call for the first tool, stop on <|observation|>
  - prompt asks for '#### <number>' -> short reasoning, an answer line '#### 42', stop on <|user|>
  - prompt asks for a python block  -> short reasoning, a ```python block, stop on <|user|>
  - anything else                   -> short reasoning, then prose repeated until max_tokens (finish=length)
Spec decode is simulated: rounds of 1 + accepted tokens (k=3 drafts, fixed acceptance cycle) at --rounds-per-s,
after a prefill delay of uncached_prompt_tokens / --prefill-tps. Prefix cache: longest shared prefix with the last 64
prompts, in whole blocks of --reuse-block tokens (64 by default; S2 reuses whole 4,096-row prefill cells and, like this,
never the rows a request decoded). A prompt whose last assistant turn already closed its think block (a thinking
budget's continuation) gets its reply without the reasoning. A generate frame's insert rule (PROTOCOL.md; off with
--no-insert, which leaves `insert` out of the hello) puts its tokens in place of the reply's after_tokens+1-th, and the
reply goes on as the continuation of a prompt that holds them. Logprobs and prompt_logprobs are fake but deterministic,
proper (sum p < 1), and the scored token is not always the argmax.

usage: mock_engine.py --model-dir DIR [--sock /tmp/g6-mock.sock]
"""
import argparse
import asyncio
import collections
import json
import logging
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402
from tokenizer import GlmTokenizer  # noqa: E402

log = logging.getLogger('mock_engine')

USER, ASSISTANT, OBSERVATION = '<|user|>', '<|assistant|>', '<|observation|>'
REASONING = 'Let me think about how to answer this well. I will keep it short and clear.'
PROSE = ('The harbor lay quiet under a pale morning sky, and the old keeper walked the length of the stone pier '
         'as he had done for thirty years. Gulls turned over the water, the tide pulled at the pilings, and the '
         'town behind him woke slowly, one lamp and one chimney at a time. ')
ACCEPT_CYCLE = [3, 2, 3, 1, 0, 2, 3, 3, 1, 2]  # accepted drafts per round (k=3)
K_DRAFT = 3


class Mock:
    def __init__(self, a):
        self.a = a
        self.t = GlmTokenizer(a.model_dir)
        self.V = self.t.vocab_size
        self.stop_user = self.t.tok.token_to_id(USER)
        self.stop_obs = self.t.tok.token_to_id(OBSERVATION)
        self.think_end = self.t.encode('</think>', add_special_tokens=False)
        self.reasoning_ids = self.t.encode(' '.join([REASONING] * a.reasoning_repeat), add_special_tokens=False)
        self.prose_ids = self.t.encode(PROSE, add_special_tokens=False)
        self.recent = collections.deque(maxlen=64)
        self.c = {'drafts': 0, 'draft_tokens': 0, 'accepted': 0, 'per_pos': [0] * K_DRAFT, 'pc_q': 0, 'pc_h': 0,
                  'prompt': 0, 'gen': 0, 'fin': collections.Counter()}
        self.running = 0

    # ---- scripted output ----------------------------------------------------------------------------------------
    def script(self, prompt: list[int]):
        """-> (ids to emit, stop id or None = cycle prose forever)"""
        if self.a.plain:  # front-end benchmarks: no per-request work proportional to the prompt
            return self.reasoning_ids + self.think_end, None
        text = self.t.decode(prompt, skip_special_tokens=False)
        last_user = text.rsplit(USER, 1)[-1]
        head = [] if '</think>' in text.rsplit(ASSISTANT, 1)[-1] else self.reasoning_ids + self.think_end
        enc = lambda s: self.t.encode(s, add_special_tokens=False)  # noqa: E731
        if text.rfind(OBSERVATION) > text.rfind(USER):  # answering after a tool result
            return head + enc('The tool returned its result; here is the answer.'), self.stop_user
        if 'describe the call format' in last_user:  # a reply *about* a call: prose, not one (front end tests)
            return head + enc('Call it like <tool_call>get_weather<arg_key>city</arg_key>'
                              '<arg_value>Paris</arg_value></tool_call> in your code.'), self.stop_user
        blocks = re.findall(r'<tools>(.*?)</tools>', text, re.S)
        tool_lines = [ln for ln in (blocks[-1] if blocks else '').splitlines() if ln.strip().startswith('{')]
        if tool_lines:
            tool = json.loads(tool_lines[0])
            props = list(((tool.get('parameters') or {}).get('properties') or {}).keys())
            args = ''.join(f'<arg_key>{k}</arg_key><arg_value>{"Paris" if i == 0 else 3}</arg_value>'
                           for i, k in enumerate(props[:2]))
            return head + enc('I will call a tool.') + enc(f'<tool_call>{tool["name"]}{args}</tool_call>'), \
                self.stop_obs
        if '####' in last_user:
            return head + enc('Working: 6 * 7 = 42.\n#### 42'), self.stop_user
        if '```python' in last_user:
            fn = re.search(r'def (\w+)\(', last_user)
            name = fn.group(1) if fn else 'solution'
            return head + enc(f'```python\ndef {name}(*args, **kwargs):\n    return None\n```'), self.stop_user
        return head, None

    def stream(self, prompt, max_tokens, min_tokens, stop_ids, insert=None):
        out, stop = self.plain_stream(prompt, max_tokens, min_tokens, stop_ids)
        if not insert:
            return out, stop
        after, unless, ins = insert[0]['after_tokens'], insert[0]['unless_emitted'], insert[0]['tokens']
        if after >= len(out) or unless in out[:after]:
            return out, stop
        head = out[:after] + ins
        if len(head) >= max_tokens:
            return head[:max_tokens], None
        rest, stop = self.plain_stream(prompt + head, max_tokens - len(head), max(0, min_tokens - len(head)), stop_ids)
        return head + rest, stop

    def plain_stream(self, prompt, max_tokens, min_tokens, stop_ids):
        ids, stop = self.script(prompt)
        out = list(ids)
        if stop is None or stop not in stop_ids or len(out) + 1 < min_tokens:
            while len(out) < max_tokens:
                out += self.prose_ids
            return out[:max_tokens], None
        out.append(stop)
        if len(out) > max_tokens:
            return out[:max_tokens], None
        return out, stop

    # ---- fake logprobs ------------------------------------------------------------------------------------------
    def _entry(self, prev: int, tid: int, k: int, salt: int):
        h = (prev * 1000003 + tid * 7919 + salt) & 0x7FFFFFFF
        rank = 1 if h % 7 else 2 + h % 3
        # a proper distribution: p_i = 0.55 * 0.35^i (sum < 0.85), small deterministic jitter
        lps = [math.log(0.55 * 0.35 ** i) - (h >> (i % 16) & 3) * 0.01 for i in range(max(k, rank))]
        others = [(tid + 1 + (h >> 3) % 997 + 131 * i) % self.V for i in range(max(k, rank))]
        top = []
        for i in range(max(k, rank)):
            top.append([tid if i == rank - 1 else others[i], lps[i]])
        return {'lp': lps[rank - 1], 'rank': rank, 'top': top[:k]}

    def prompt_lps(self, prompt, k):
        return [None] + [self._entry(prompt[j - 1], prompt[j], k, 17) for j in range(1, len(prompt))]

    # ---- prefix cache -------------------------------------------------------------------------------------------
    def cached(self, prompt):
        if self.a.plain:
            return 0
        best = 0
        for p in self.recent:
            n = min(len(p), len(prompt) - 1)
            i = 0
            while i < n and p[i] == prompt[i]:
                i += 1
            best = max(best, i)
        self.recent.append(tuple(prompt))
        return best // self.a.reuse_block * self.a.reuse_block

    # ---- protocol -----------------------------------------------------------------------------------------------
    async def serve(self, reader, writer):
        hello = {'type': 'hello', 'version': 1, 'model': 'glm-5.3-mock', 'max_model_len': self.a.max_model_len,
                 'vocab_size': self.V, 'eos_ids': self.t.eos_ids, 'max_logprobs': 20}
        if not self.a.no_insert:
            hello['insert'] = True
        wire.write_frame(writer, hello)
        tasks: dict[str, asyncio.Task] = {}
        try:
            while True:
                f = await wire.read_frame(reader)
                if f is None:
                    break
                t = f['type']
                if t == 'generate':
                    rid = f['id']
                    if rid in tasks:
                        wire.write_frame(writer, self._finish(rid, 'error', 0, 0, 0, message='duplicate id'))
                        continue
                    tasks[rid] = asyncio.create_task(self.generate(f, writer))
                    tasks[rid].add_done_callback(lambda _t, r=rid: tasks.pop(r, None))
                elif t == 'cancel':
                    task = tasks.get(f.get('id'))
                    if task:
                        task.cancel()
                elif t == 'metrics':
                    wire.write_frame(writer, self.metrics(f.get('seq')))
                await writer.drain()
        except wire.FrameError as e:
            log.warning('bad frame, closing: %s', e)
        finally:
            for task in list(tasks.values()):
                task.cancel()
            writer.close()

    def _finish(self, rid, reason, pt, cached, ct, stop_id=None, message=None):
        self.c['fin'][reason] += 1
        f = {'type': 'finish', 'id': rid, 'reason': reason, 'stop_id': stop_id, 'prompt_tokens': pt,
             'cached_tokens': cached, 'completion_tokens': ct}
        if message:
            f['message'] = message
        return f

    async def generate(self, f, writer):
        rid, prompt = f['id'], f['prompt']
        max_tokens, k = f['max_tokens'], f.get('logprobs')
        if not prompt or len(prompt) + max_tokens > self.a.max_model_len or max_tokens < 1:
            wire.write_frame(writer, self._finish(rid, 'error', len(prompt), 0, 0, message='bad prompt/max_tokens'))
            return
        self.running += 1
        emitted, cached = 0, 0
        try:
            cached = self.cached(prompt)
            self.c['pc_q'] += len(prompt)
            self.c['pc_h'] += cached
            self.c['prompt'] += len(prompt) - cached
            await asyncio.sleep((len(prompt) - cached) / self.a.prefill_tps)
            if f.get('prompt_logprobs') is not None:
                wire.write_frame(writer, {'type': 'prompt_logprobs', 'id': rid,
                                          'lps': self.prompt_lps(prompt, f['prompt_logprobs'])})
            if f.get('insert') and self.a.no_insert:
                raise ValueError('insert sent to an engine without it')
            ids, stop = self.stream(prompt, max_tokens, f.get('min_tokens', 0), set(f.get('stop_ids') or []), f.get('insert'))
            rnd, prev = 0, prompt[-1]
            while emitted < len(ids):
                acc = ACCEPT_CYCLE[rnd % len(ACCEPT_CYCLE)]
                n = min(1 + acc, len(ids) - emitted)
                if len(ids) - emitted > 1:  # a verify round with drafts
                    self.c['drafts'] += 1
                    self.c['draft_tokens'] += K_DRAFT
                    self.c['accepted'] += n - 1
                    for i in range(n - 1):
                        self.c['per_pos'][i] += 1
                chunk = ids[emitted:emitted + n]
                fr = {'type': 'tokens', 'id': rid, 'ids': chunk}
                if k is not None:
                    fr['logprobs'] = []
                    for tid in chunk:
                        fr['logprobs'].append(self._entry(prev, tid, k, 5))
                        prev = tid
                else:
                    prev = chunk[-1]
                if rnd == 0 or self.a.rounds_per_s <= 0:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(1 / self.a.rounds_per_s)
                wire.write_frame(writer, fr)
                emitted += n
                self.c['gen'] += n
                rnd += 1
            wire.write_frame(writer, self._finish(rid, 'stop' if stop is not None else 'length', len(prompt), cached,
                                                  emitted, stop_id=stop))
        except asyncio.CancelledError:
            if not writer.is_closing():
                wire.write_frame(writer, self._finish(rid, 'abort', len(prompt), cached, emitted))
        except Exception as e:  # never leave a request without a finish frame
            log.exception('generate %s failed', rid)
            if not writer.is_closing():
                wire.write_frame(writer, self._finish(rid, 'error', len(prompt), cached, emitted, message=repr(e)))
        finally:
            self.running -= 1

    def metrics(self, seq):
        c = self.c
        return {'type': 'metrics', 'seq': seq,
                'spec': {'drafts': c['drafts'], 'draft_tokens': c['draft_tokens'], 'accepted': c['accepted'],
                         'accepted_per_pos': list(c['per_pos'])},
                'prefix_cache': {'queries': c['pc_q'], 'hits': c['pc_h']},
                'requests': {'running': self.running, 'waiting': 0, 'finished': dict(c['fin'])},
                'tokens': {'prompt': c['prompt'], 'generation': c['gen']}}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--sock', default='/tmp/g6-mock.sock')
    ap.add_argument('--max-model-len', type=int, default=200000)
    ap.add_argument('--prefill-tps', type=float, default=50000.0, help='simulated prefill tokens/s')
    ap.add_argument('--rounds-per-s', type=float, default=100.0, help='simulated verify rounds/s (0 = no delay)')
    ap.add_argument('--reuse-block', type=int, default=64,
                    help='prefix cache hits are whole blocks of this many tokens (S2: 4096)')
    ap.add_argument('--reasoning-repeat', type=int, default=1, help='the canned reasoning, this many times over')
    ap.add_argument('--no-insert', action='store_true', help='an engine without the insert rule (the hello says so)')
    ap.add_argument('--plain', action='store_true',
                    help='skip prompt-dependent scripting and the prefix cache (front-end overhead benchmarks)')
    a = ap.parse_args()
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    mock = Mock(a)
    if os.path.exists(a.sock):
        os.unlink(a.sock)
    server = await asyncio.start_unix_server(mock.serve, a.sock, limit=wire.MAX_FRAME)
    log.info('mock engine on %s (vocab %d)', a.sock, mock.V)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
