#!/usr/bin/env python3
"""F1 gate: the Rust front end (g6front-rs) answers a fixed request set byte-identically to the Python front end.

Starts two mock engines (one per front end, so prefix-cache and metrics state evolve the same way), the Python
front end on one and g6front-rs on the other, sends every case to both in the same order and diffs the raw
response bodies after normalizing only what is random by design: response/tool-call ids and `created`.

usage: diff_fronts.py --model-dir DIR --rust-bin PATH [--py PYTHON] [--keep] [-k SUBSTR] [-v]
Then replays every case that succeeded, all at once, against g6front-rs (frames of different requests interleave on
its one engine connection) and checks each answer equals its sequential answer.
Exit status 0 = all cases identical.
"""
import argparse
import concurrent.futures
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.dirname(HERE)

TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city',
                                           'parameters': {'type': 'object', 'properties': {
                                               'city': {'type': 'string'}, 'days': {'type': 'integer'}}}}}]
TOOLS_RICH = [
    {'type': 'function', 'function': {'name': 'search', 'description': 'Søk «quoted»   text', 'strict': True,
                                      'parameters': {'type': 'object', 'properties': {
                                          'q': {'type': 'string'}, 'limit': {'type': ['integer', 'null']},
                                          'ratio': {'type': 'number', 'default': 0.25},
                                          'opts': {'type': 'object'}, 'tags': {'type': 'array'},
                                          'flag': {'anyOf': [{'type': 'boolean'}, {'enum': [1, 2.5, 'x', None]}]}},
                                          'required': ['q']}}},
    {'type': 'function', 'function': {'name': 'hidden', 'description': None, 'parameters': None,
                                      'defer_loading': True}},
    {'type': 'function', 'defer_loading': False, 'function': {'name': 'noargs'}},
]
U = lambda s: {'role': 'user', 'content': s}  # noqa: E731
M = 'glm-5.3'


def chat(msgs, **kw):
    return ('POST', '/v1/chat/completions', dict({'model': M, 'messages': msgs, 'max_tokens': 40}, **kw))


def cmpl(prompt, **kw):
    return ('POST', '/v1/completions', dict({'model': M, 'prompt': prompt, 'max_tokens': 24}, **kw))


def tokz(**kw):
    return ('POST', '/tokenize', dict({'model': M}, **kw))


def big_text(n_lines, seed):
    import random
    r = random.Random(seed)
    words = ['harbor', 'Lantern', 'x', '42', '3.14', 'naïve', '你好', '🙂', 'code()', '{', '}', '  ', '\t', "'s", '--']
    return ''.join(' '.join(r.choice(words) for _ in range(r.randint(0, 14))) + r.choice(['\n', '\r\n', '\n\n', ' \n'])
                   for _ in range(n_lines))


BIG = big_text(12000, 7)  # ~250 KB: long enough for the parallel encode path

ASSIST_TC = {'role': 'assistant', 'content': None, 'tool_calls': [
    {'id': 'c1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris", "days": 3}'}},
    {'id': 'c2', 'type': 'function', 'function': {'name': 'get_weather',
                                                  'arguments': {'city': 'Zürich', 'days': 1.5, 'n': None,
                                                                'deep': {'a': [1, 2e-7, 1e16, True]}}}}]}

HISTORY = [
    {'role': 'system', 'content': 'You are terse.'},
    U('hi'),
    {'role': 'assistant', 'content': 'Hello!', 'reasoning': 'greet back'},
    U([{'type': 'text', 'text': 'part one '}, {'type': 'image_url', 'image_url': {'url': 'x'}},
       {'type': 'input_text', 'text': 'part two'}, 'bare string part']),
    {'role': 'assistant', 'content': '<think>old thoughts</think>  visible answer  '},
    U('and now?'),
]

CASES = {
    # ---- chat, plain
    'chat_basic': chat([U('Tell me about harbors.')], temperature=0),
    'chat_stream': chat([U('Tell me about harbors.')], stream=True),
    'chat_stream_usage': chat([U('Tell me about harbors.')], stream=True, stream_options={'include_usage': True}),
    'chat_stream_continuous': chat([U('Tell me about harbors.')], stream=True,
                                   stream_options={'include_usage': True, 'continuous_usage_stats': True}),
    'chat_history': chat(HISTORY),
    'chat_history_stream': chat(HISTORY, stream=True),
    'chat_answer_stop': chat([U("Solve. Answer as '#### <number>'.")], max_tokens=1024, seed=1234,
                             chat_template_kwargs={'thinking': False}),
    'chat_python': chat([U('Write ```python\ndef add(a, b):```')], max_tokens=200),
    'chat_no_reasoning': chat([U('Tell me about harbors.')], include_reasoning=False),
    'chat_no_reasoning_stream': chat([U('Tell me about harbors.')], include_reasoning=False, stream=True),
    'chat_details': chat([U('Tell me about harbors.')], max_completion_tokens=7),
    'chat_ret_ids': chat([U('Tell me about harbors.')], return_token_ids=True, max_tokens=8),
    'chat_ret_ids_stream': chat([U('Tell me about harbors.')], return_token_ids=True, max_tokens=8, stream=True),
    'chat_logprobs': chat([U('Tell me about harbors.')], max_tokens=6, logprobs=True, top_logprobs=3),
    'chat_logprobs0': chat([U('Tell me about harbors.')], max_tokens=6, logprobs=True),
    'chat_logprobs_stream': chat([U('Tell me about harbors.')], max_tokens=6, logprobs=True, top_logprobs=2,
                                 stream=True),
    'chat_logprobs_ids': chat([U('Tell me about harbors.')], max_tokens=6, logprobs=True, top_logprobs=2,
                              return_tokens_as_token_ids=True),
    'chat_prompt_logprobs': chat([U('hi')], max_tokens=2, prompt_logprobs=3),
    'chat_stop_str': chat([U('Tell me about harbors.')], max_tokens=200, stop=['keeper', 'zzz']),
    'chat_stop_str_stream': chat([U('Tell me about harbors.')], max_tokens=200, stop='keeper', stream=True),
    'chat_stop_incl': chat([U('Tell me about harbors.')], max_tokens=200, stop=['the old'],
                           include_stop_str_in_output=True, stream=True),
    'chat_stop_ids': chat([U('Tell me about harbors.')], max_tokens=100, stop_token_ids=[13]),
    'chat_min_tokens': chat([U("Solve. Answer as '#### <number>'.")], max_tokens=60, min_tokens=50),
    'chat_ignore_eos': chat([U("Solve. Answer as '#### <number>'.")], max_tokens=40, ignore_eos=True),
    'chat_reasoning_effort': chat([U('x')], reasoning_effort='low'),
    'chat_reasoning_effort_none': chat([U('x')], reasoning_effort='none'),
    'chat_developer': chat([{'role': 'developer', 'content': 'dev rules'}, U('hi'),
                            {'role': 'system', 'content': [{'type': 'text', 'text': 'late sys'}]}]),
    'chat_continue_final': chat([U('hi'), {'role': 'assistant', 'content': 'Sure, the answer is'}],
                                add_generation_prompt=False, continue_final_message=True),
    'chat_name_field': chat([{'role': 'user', 'content': 'hi', 'name': 'bob'}]),
    'chat_unicode': chat([U('héllo wörld 你好 🙂 \u0000 \t tab "quote" \\ back')]),
    # ---- tools
    'tool_call': chat([U('weather?')], tools=TOOLS, max_tokens=200),
    'tool_call_stream': chat([U('weather?')], tools=TOOLS, max_tokens=200, stream=True),
    'tool_call_rich': chat([U('search it')], tools=TOOLS_RICH, max_tokens=200),
    'tool_call_rich_stream': chat([U('search it')], tools=TOOLS_RICH, max_tokens=200, stream=True),
    'tool_choice_none': chat([U('weather?')], tools=TOOLS, tool_choice='none', max_tokens=200),
    'tool_choice_none_stream': chat([U('weather?')], tools=TOOLS, tool_choice='none', max_tokens=200, stream=True),
    'tool_result': chat([U('weather?'), ASSIST_TC,
                         {'role': 'tool', 'tool_call_id': 'c2', 'content': 'cloudy'},
                         {'role': 'tool', 'tool_call_id': 'c1', 'content': [{'type': 'text', 'text': 'sunny'},
                                                                            {'type': 'text', 'text': 'warm'}]}],
                        tools=TOOLS, max_tokens=100),
    'tool_result_unsorted': chat([U('weather?'), ASSIST_TC, {'role': 'tool', 'content': 'no id'}],
                                 tools=TOOLS, max_tokens=100),
    'tool_result_stream': chat([U('weather?'), ASSIST_TC, {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}],
                               tools=TOOLS, max_tokens=100, stream=True),
    # ---- completions
    'cmpl_text': cmpl('Once upon a time', temperature=0),
    'cmpl_stream': cmpl('Once upon a time', stream=True, stream_options={'include_usage': True}),
    'cmpl_default_max': ('POST', '/v1/completions', {'model': M, 'prompt': 'x'}),
    'cmpl_ids': cmpl([154822, 154824, 9703, 1052, 1052, 13], max_tokens=4),
    'cmpl_batch_of_one': cmpl(['Once upon a time']),
    'cmpl_logprobs': cmpl('Once upon a time', logprobs=3, max_tokens=8),
    'cmpl_logprobs_stream': cmpl('Once upon a time', logprobs=2, max_tokens=8, stream=True),
    'cmpl_kl_shape': cmpl([154822, 154824, 9703], max_tokens=8, temperature=0, return_tokens_as_token_ids=True,
                          logprobs=1),
    'cmpl_prompt_logprobs': cmpl([154822, 154824, 9703, 1052, 1052, 13], max_tokens=1, prompt_logprobs=20),
    'cmpl_prompt_logprobs_stream': cmpl([154822, 154824, 9703, 1052, 1052, 13], max_tokens=3, prompt_logprobs=2,
                                        stream=True, return_token_ids=True),
    'cmpl_prompt_logprobs_all': cmpl('The keeper walked', max_tokens=1, prompt_logprobs=-1),
    'cmpl_stop': cmpl('Tell me a story.', max_tokens=200, stop=['keeper']),
    'cmpl_stop_stream': cmpl('Tell me a story.', max_tokens=200, stop=['keeper', 'pier'], stream=True),
    'cmpl_skip_special_off': cmpl('Tell me a story.', max_tokens=20, skip_special_tokens=False),
    'cmpl_no_special': cmpl('Tell me a story.', add_special_tokens=False),
    'cmpl_ret_ids': cmpl('Tell me a story.', return_token_ids=True, max_tokens=5, stream=True),
    'cmpl_seed_float_int': cmpl('x', seed=5.0, top_k=-1, top_p=0.5, min_p=0.1, temperature=1),
    # ---- tokenize / detokenize / misc
    'tok_text': tokz(prompt='hello world'),
    'tok_text_strs': tokz(prompt='héllo wörld 你好', return_token_strs=True),
    'tok_chat': tokz(messages=[U('hi')], add_generation_prompt=True, chat_template_kwargs={'enable_thinking': False}),
    'tok_chat_history': tokz(messages=HISTORY),
    'tok_chat_tools': tokz(messages=[U('hi')], tools=TOOLS_RICH),
    'tok_chat_tool_result': tokz(messages=[U('weather?'), ASSIST_TC,
                                           {'role': 'tool', 'tool_call_id': 'c1', 'content': 'a'},
                                           {'role': 'tool', 'tool_call_id': 'c2', 'content': 'b'}], tools=TOOLS),
    'tok_chat_tool_ref': tokz(messages=[U('find tools'), {'role': 'tool', 'content': [
        {'type': 'tool_reference', 'name': 'hidden'}]}], tools=TOOLS_RICH),
    'tok_chat_clear_thinking': tokz(messages=HISTORY, chat_template_kwargs={'clear_thinking': True}),
    'tok_chat_effort': tokz(messages=[U('x')], chat_template_kwargs={'reasoning_effort': 'high'}),
    'tok_chat_no_gen': tokz(messages=[U('x')], add_generation_prompt=False),
    'tok_chat_continue': tokz(messages=[U('x'), {'role': 'assistant', 'content': ' leading space'}],
                              add_generation_prompt=False, continue_final_message=True),
    'tok_chat_add_special': tokz(messages=[U('x')], add_special_tokens=True),
    'tok_big_text': tokz(prompt=BIG),
    'tok_big_chat': tokz(messages=[U(BIG), {'role': 'assistant', 'content': BIG[:50000]}, U(BIG[1000:90000])]),
    'chat_big': chat([U(BIG)], max_tokens=8),
    'cmpl_big_stream': cmpl(BIG, max_tokens=8, stream=True),
    'detok': ('POST', '/detokenize', {'model': M, 'tokens': [154822, 154824, 9703, 1052, 13]}),
    'models': ('GET', '/v1/models', None),
    'health': ('GET', '/health', None),
    'metrics': ('GET', '/metrics', None),
    # ---- review regressions (#15 review): each of these once differed or crashed
    'rv_stop_behind_released': cmpl('hello there', max_tokens=30, min_tokens=8, stop=['e', 'q' * 20],
                                    include_stop_str_in_output=True),
    'rv_stop_behind_released_chat': chat([U('hi')], max_tokens=30, min_tokens=8, stop=['e', 'q' * 20],
                                         include_stop_str_in_output=True, stream=True),
    'rv_seed_2p63': chat([U('x')], seed=9223372036854775808),
    'rv_seed_neg': chat([U('x')], seed=-9223372036854775809),
    'rv_seed_max_ok': chat([U('x')], seed=9223372036854775807, max_tokens=2),
    'rv_top_k_huge': chat([U('x')], top_k=36893488147419103232, max_tokens=2),
    'rv_defer_null': tokz(messages=[U('x')], tools=[{'type': 'function', 'defer_loading': True,
                                                     'function': {'name': 'f', 'defer_loading': None}}]),
    'rv_kwargs_false': tokz(messages=[U('x')], chat_template_kwargs=False),
    'rv_text_numbers': tokz(messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 1e16}, {'type': 'text', 'text': 1.5e-7}, {'type': 'text', 'text': True},
        {'type': 'text', 'text': None}, {'type': 'text', 'text': {'a': [1, 2.5, 'it\'s', 'q"\n']}},
        {'type': 'text', 'text': 7}]}]),
    # ---- errors
    'err_model': chat([U('x')], model='nope'),
    'err_n': chat([U('x')], n=2),
    'err_n_ok_float': chat([U('x')], n=1.0, max_tokens=3),
    'err_max_tokens': chat([U('x')], max_tokens=10 ** 7),
    'err_max_tokens_zero': chat([U('x')], max_tokens=0),
    'err_max_tokens_neg': chat([U('x')], max_tokens=-1),
    'err_max_tokens_float': chat([U('x')], max_tokens=2.5),
    'err_max_tokens_bool': chat([U('x')], max_tokens=True),
    'err_temperature': chat([U('x')], temperature=3),
    'err_top_p_zero': chat([U('x')], top_p=0),
    'err_min_gt_max': chat([U('x')], max_tokens=3, min_tokens=5),
    'err_messages': chat('x'),
    'err_message_role': chat([{'content': 'x'}]),
    'err_content_type': chat([{'role': 'user', 'content': 5}]),
    'err_echo': chat([U('x')], echo=True),
    'err_tool_choice': chat([U('x')], tools=TOOLS, tool_choice='required'),
    'err_tool_choice_named': chat([U('x')], tools=TOOLS, tool_choice={'type': 'function',
                                                                      'function': {'name': 'get_weather'}}),
    'err_tools_shape': chat([U('x')], tools=[{'type': 'function'}]),
    'err_template_kwargs': chat([U('x')], chat_template_kwargs={'tokenize': True}),
    'err_custom_template': chat([U('x')], chat_template='{{ x }}'),
    'err_both_continue': chat([U('x')], continue_final_message=True),
    'err_top_logprobs_alone': chat([U('x')], top_logprobs=2),
    'err_logprobs_big': chat([U('x')], logprobs=True, top_logprobs=21),
    'err_penalty': chat([U('x')], presence_penalty=0.5),
    'ok_penalty_zero': chat([U('x')], presence_penalty=0, frequency_penalty=0.0, repetition_penalty=1, max_tokens=2),
    'err_rep_penalty': chat([U('x')], repetition_penalty=1.1),
    'err_stop_type': chat([U('x')], stop=[1]),
    'err_stop_ids_type': chat([U('x')], stop_token_ids=['a']),
    'err_seed_type': chat([U('x')], seed='1'),
    'err_best_of': cmpl('x', best_of=2),
    'err_cmpl_batch': cmpl(['a', 'b']),
    'err_cmpl_oov': cmpl([1, 10 ** 7]),
    'err_cmpl_empty': cmpl([]),
    'err_cmpl_echo': cmpl('x', echo=True),
    'err_cmpl_suffix': cmpl('x', suffix='y'),
    'err_cmpl_logprobs_bool': cmpl('x', logprobs=True),
    'err_plp_big': cmpl('x', prompt_logprobs=21),
    'err_tok_missing': tokz(),
    'err_tok_both': tokz(messages=[U('x')], continue_final_message=True),
    'err_detok': ('POST', '/detokenize', {'model': M, 'tokens': 'abc'}),
    'err_not_object': ('POST', '/v1/chat/completions', [1, 2]),
    'err_404': ('GET', '/nope', None),
}

ID_RES = [
    (re.compile(rb'"(chatcmpl-tool|chatcmpl|cmpl)-[0-9a-f]{32}"'), rb'"\1-ID"'),
    (re.compile(rb'"created":\d+'), rb'"created":0'),
]


# Counters that depend on how fast a front end cancels after a stop string matched (the mock keeps generating
# until the cancel lands), not on what it sent: masked in /metrics. Every other counter must match exactly.
TIMING_METRICS = re.compile(rb'^(vllm:(generation_tokens_total|spec_decode_[a-z_]+)\{[^}]*\}) .*$', re.M)


CACHED = re.compile(rb'"cached_tokens":\d+')


def normalize(b: bytes, path: str = '') -> bytes:
    for rx, rep in ID_RES:
        b = rx.sub(rep, b)
    if path == '/metrics':
        b = TIMING_METRICS.sub(rb'\1 <timing>', b)
    return b


def request(port, method, path, body):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    data = None if body is None else json.dumps(body).encode()
    c.request(method, path, data, {'Content-Type': 'application/json'} if data else {})
    r = c.getresponse()
    out = r.read()
    ctype = r.getheader('Content-Type', '')
    c.close()
    return r.status, ctype, out


def wait_http(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if request(port, 'GET', '/health', None)[0] == 200:
                return True
        except OSError:
            pass
        time.sleep(0.1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--rust-bin', required=True)
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--port-py', type=int, default=18101)
    ap.add_argument('--port-rs', type=int, default=18102)
    ap.add_argument('--front-args', default='', help='extra flags for both front ends')
    ap.add_argument('-k', default='', help='only cases containing this substring')
    ap.add_argument('-v', action='store_true', help='print both bodies for mismatches')
    a = ap.parse_args()

    work = tempfile.mkdtemp(prefix='g6diff-')
    procs = []

    def spawn(name, cmd, cwd=FRONT):
        log = open(os.path.join(work, name + '.log'), 'w')
        p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        return p

    try:
        socks = {}
        for side in ('py', 'rs'):
            socks[side] = os.path.join(work, f'mock-{side}.sock')
            spawn(f'mock-{side}', [a.py, os.path.join(FRONT, 'mock_engine.py'), '--model-dir', a.model_dir,
                                   '--sock', socks[side], '--rounds-per-s', '0'])
        for _ in range(100):
            if all(os.path.exists(s) for s in socks.values()):
                break
            time.sleep(0.2)
        extra = a.front_args.split()
        spawn('front-py', [a.py, '-m', 'g6front', '--model-dir', a.model_dir, '--engine-sock', socks['py'],
                           '--host', '127.0.0.1', '--port', str(a.port_py), *extra])
        spawn('front-rs', [a.rust_bin, '--model-dir', a.model_dir, '--engine-sock', socks['rs'],
                           '--host', '127.0.0.1', '--port', str(a.port_rs), *extra])
        if not (wait_http(a.port_py) and wait_http(a.port_rs)):
            print(f'front ends did not come up; logs in {work}')
            return 2
        fails = 0
        n = 0
        seq_rs = {}
        for name, (method, path, body) in CASES.items():
            if a.k not in name:
                continue
            n += 1
            sp, tp, bp = request(a.port_py, method, path, body)
            sr, tr, br = request(a.port_rs, method, path, body)
            bp, br = normalize(bp, path), normalize(br, path)
            if sr == 200 and path not in ('/metrics', '/health'):
                seq_rs[name] = br
            same = (sp, bp) == (sr, br) and tp.split(';')[0] == tr.split(';')[0]
            if not same:
                fails += 1
                print(f'DIFF {name}: status {sp} vs {sr}, ctype {tp!r} vs {tr!r}')
                if a.v:
                    print('  py:', bp[:3000].decode(errors='replace'))
                    print('  rs:', br[:3000].decode(errors='replace'))
                else:
                    i = next((i for i in range(min(len(bp), len(br))) if bp[i] != br[i]), min(len(bp), len(br)))
                    print(f'  first difference at byte {i}:')
                    print('  py:', bp[max(0, i - 120):i + 120].decode(errors='replace'))
                    print('  rs:', br[max(0, i - 120):i + 120].decode(errors='replace'))
        print(f'{n - fails}/{n} cases identical')
        cfails = 0
        with concurrent.futures.ThreadPoolExecutor(len(seq_rs) or 1) as ex:
            futs = {name: ex.submit(request, a.port_rs, *CASES[name]) for name in seq_rs}
            for name, f in futs.items():
                st, _, body = f.result()
                # the mock's prefix cache sees the prompts in a different order when they arrive together
                cached = lambda b: CACHED.sub(rb'"cached_tokens":N', b)  # noqa: E731
                if (st, cached(normalize(body))) != (200, cached(seq_rs[name])):
                    cfails += 1
                    print(f'CONCURRENT DIFF {name}: status {st}')
        print(f'{len(seq_rs) - cfails}/{len(seq_rs)} cases identical when replayed concurrently')
        return 1 if fails or cfails else 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        print(f'logs: {work}')


if __name__ == '__main__':
    sys.exit(main())
