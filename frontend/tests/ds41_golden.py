#!/usr/bin/env python3
"""Regenerate frontend/g6front-rs/src/ds41_golden.json: DeepSeek-V4.1 prompts rendered by the release's own encoder.

The Rust renderer (src/dschat.rs) is a port of that encoder; `cargo test` compares it byte for byte with this file.
The encoder is encoding.py from the DeepSeek-V4.1-Flash-1017 checkpoint directory (its `encoding/` folder). It is
not copied into this repo.

  python3 frontend/tests/ds41_golden.py --encoder /path/to/ds41-flash-1017/encoding/encoding.py

The official fixtures tests/test_input_{1,2}.json next to it are folded in when they exist (3-5 use the latest_reminder
role, tasks and images, which the front end refuses).
"""
import argparse
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'g6front-rs', 'src', 'ds41_golden.json')


def load_encoder(path):
    spec = importlib.util.spec_from_file_location('ds41_encoding', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fn(name, desc=None, params=None):
    f = {'name': name}
    if desc is not None:
        f['description'] = desc
    f['parameters'] = params if params is not None else {'type': 'object', 'properties': {}}
    return {'type': 'function', 'function': f}


def call(cid, name, args):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': args}}


WEATHER = fn('get_weather', 'Get the weather for a city', {
    'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}}, 'required': ['city']})
SEARCH = fn('web_search', 'Search the web', {'type': 'object', 'properties': {'q': {'type': 'string'}}})
NS_TOOL = fn('fs::read', 'Read a file')
NS_OBJ = dict(fn('write', 'Write a file'), namespace={'name': 'fs', 'description': 'Filesystem tools.'})
NS_STR = dict(fn('stat'), namespace='fs')
UNI = fn('查询', '查询天气 «ünï»', {'type': 'object', 'properties': {'城市': {'type': 'string', 'description': '城市名'}}})

U = lambda t: {'role': 'user', 'content': t}
A = lambda t, rc=None, calls=None: dict(
    {'role': 'assistant', 'content': t}, **({'reasoning_content': rc} if rc is not None else {}),
    **({'tool_calls': calls} if calls else {}))
S = lambda t: {'role': 'system', 'content': t}
T = lambda cid, t: {'role': 'tool', 'tool_call_id': cid, 'content': t}

CASES = []


def case(name, messages, tools=None, thinking=True, effort=None):
    CASES.append({'name': name, 'messages': messages, 'tools': tools or [], 'thinking': thinking, 'effort': effort})


for think in (True, False):
    tag = 'think' if think else 'chat'
    case(f'plain_{tag}', [U('What is 19 + 23? Reply only with the number.')], thinking=think)
    case(f'system_{tag}', [S('You are terse.'), U('hi')], thinking=think)
    case(f'developer_{tag}', [{'role': 'developer', 'content': 'dev rules'}, U('hi')], thinking=think)
    case(f'multi_turn_{tag}', [S('S'), U('a'), A('x', 'why x'), U('b'), A('y', 'why y'), U('c')], thinking=think)
    case(f'multi_turn_tools_{tag}', [U('a'), A('x', 'why x'), U('b'), A('y', 'why y'), U('c')], tools=[WEATHER], thinking=think)
    case(f'tools_no_system_{tag}', [U('weather in Paris?')], tools=[WEATHER, SEARCH], thinking=think)
    case(f'tools_with_system_{tag}', [S('Be brief.'), U('weather in Paris?')], tools=[WEATHER], thinking=think)
    case(f'tool_loop_{tag}', [
        S('S'), U('weather in Paris and Rome?'),
        A('Let me check.', 'need both', [call('c1', 'get_weather', '{"city": "Paris"}'), call('c2', 'get_weather', {'city': 'Rome', 'days': 3})]),
        T('c2', 'rain'), T('c1', 'sun'),
    ], tools=[WEATHER], thinking=think)
    case(f'tool_loop_answer_{tag}', [
        U('weather in Paris?'),
        A('', 'check it', [call('c1', 'get_weather', '{"city": "Paris"}')]),
        T('c1', '{"t": 22}'),
        A('It is 22.', 'report it'),
        U('and tomorrow?'),
    ], tools=[WEATHER], thinking=think)
    case(f'tool_result_then_text_{tag}', [
        U('go'), A('', 'r', [call('c1', 'web_search', '{"q": "x"}')]), T('c1', 'res'), U('now summarize'),
    ], tools=[SEARCH], thinking=think)
    case(f'user_then_tool_merge_{tag}', [
        U('go'), A('', None, [call('c1', 'web_search', '{"q": "x"}')]), U('extra note'), T('c1', 'res'),
    ], tools=[SEARCH], thinking=think)
    case(f'consecutive_users_{tag}', [U('one'), U('two'), A('ok', 'r'), U('three')], thinking=think)
    case(f'mid_system_{tag}', [S('S'), U('a'), A('x', 'r'), S('new rule'), A('y', 'r2'), U('b')], thinking=think)
    case(f'mid_system_last_{tag}', [S('S'), U('a'), A('x', 'r'), S('reminder')], thinking=think)
    case(f'only_system_{tag}', [S('only')], thinking=think)
    case(f'assistant_last_{tag}', [U('a'), A('partial', 'r')], thinking=think)
    case(f'unicode_{tag}', [S('你是助手 «ünï» 🚀'), U('请用中文回答：天气如何？\n第二行\t制表')], thinking=think)
    case(f'unicode_tools_{tag}', [U('查')], tools=[UNI], thinking=think)
    case(f'empty_content_{tag}', [U(''), A('', ''), U('x')], thinking=think)
    case(f'null_assistant_content_{tag}', [U('a'), {'role': 'assistant', 'content': None, 'reasoning_content': 'r'}, U('b')], thinking=think)

# effort: ints and the three names
for e in (1, 50, 75, 100):
    case(f'effort_{e}', [U('hi')], effort=e)
for e in ('low', 'high', 'max'):
    case(f'effort_{e}', [S('S'), U('hi')], effort=e)
case('effort_ignored_in_chat', [U('hi')], thinking=False, effort=10)

# tool-call argument forms
case('args_forms', [
    U('go'),
    A('', None, [
        call('c1', 'f', '{"s": "line1\\nline2 \\"q\\"", "n": 3, "f": 1.0, "g": 0.1, "b": true, "z": null, "l": [1, "a", {"k": "v"}], "o": {"x": 1, "y": [true]}, "u": "日本語"}'),
        call('c2', 'f', {'a': 1, 'b': 'two', 'c': [3]}),
        call('c3', 'f', '"{\\"k\\": 1}"'),
        call('c4', 'f', 'not json'),
        call('c5', 'f', '[1, 2]'),
        call('c6', 'f', '{}'),
        call('c7', 'f', ''),
        call('c8', 'f', '{"a": 1, "a": 2, "b": 0}'),
        call('c9', 'f', '{"big": 12345678901234, "neg": -0.5, "e": 1e-7, "t": "<｜DSML｜ calls>"}'),
    ]),
    T('c1', 'r1'), T('c2', 'r2'), T('c3', 'r3'), T('c4', 'r4'), T('c5', 'r5'), T('c6', 'r6'), T('c7', 'r7'), T('c8', 'r8'), T('c9', 'r9'),
], tools=[fn('f')])
case('result_order_unknown_ids', [
    U('go'),
    A('', None, [call('a', 'f', '{}'), call('b', 'f', '{}'), call('c', 'f', '{}')]),
    T('c', 'C'), T('zzz', 'Z'), T('a', 'A'), T('b', 'B'),
], tools=[fn('f')], thinking=False)
case('result_order_with_text', [
    U('go'),
    A('', None, [call('a', 'f', '{}'), call('b', 'f', '{}')]),
    T('b', 'B'), U('note'), T('a', 'A'),
], tools=[fn('f')], thinking=False)
case('call_with_content_and_reasoning', [U('go'), A('Checking.', 'plan', [call('c1', 'f', '{"x": 1}')]), T('c1', 'ok')], tools=[fn('f')])
case('tool_content_list', [
    U('go'), A('', None, [call('c1', 'f', '{}')]),
    {'role': 'tool', 'tool_call_id': 'c1', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}, {'type': 'weird'}]},
], tools=[fn('f')], thinking=False)

# namespaces
case('namespaces', [U('go'), A('', None, [call('c1', 'fs::read', '{"p": "/x"}'), {'id': 'c2', 'type': 'function', 'namespace': 'fs', 'function': {'name': 'stat', 'arguments': '{}'}}]), T('c1', 'r'), T('c2', 's')],
     tools=[NS_TOOL, NS_OBJ, NS_STR], thinking=False)

# content given as lists
case('list_content', [
    {'role': 'system', 'content': [{'type': 'text', 'text': 'sys a'}, {'type': 'text', 'text': 'sys b'}]},
    {'role': 'user', 'content': [{'type': 'text', 'text': 'u a'}, {'type': 'text', 'text': 'u b'}]},
    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'a a'}, {'type': 'text', 'text': 'a b'}], 'reasoning_content': 'r'},
    {'role': 'user', 'content': [{'type': 'text', 'text': 'u2'}, {'type': 'audio'}]},
], thinking=True)

# tools whose JSON has awkward values
case('tool_schema_awkward', [U('x')], tools=[fn('g', 'a "quoted" \\ back\nnew', {'type': 'object', 'properties': {'k': {'enum': [1, 2.5, None, True, '日本']}}})], thinking=False)


def to_encoder_input(c):
    msgs = copy.deepcopy(c['messages'])
    for m in msgs:
        if m['role'] == 'developer':
            m['role'] = 'system'
    if c['tools']:
        if msgs[0]['role'] != 'system':
            msgs.insert(0, {'role': 'system', 'content': ''})
        msgs[0]['tools'] = c['tools']
    return msgs


def official_fixtures(enc, enc_dir):
    """Fixtures 1 and 2 (text, tools, tool results). Each must reproduce its published output first."""
    out = []
    for i in (1, 2):
        p = os.path.join(enc_dir, 'tests', f'test_input_{i}.json')
        o = os.path.join(enc_dir, 'tests', f'test_output_{i}.txt')
        if not (os.path.exists(p) and os.path.exists(o)):
            continue
        d = json.load(open(p))
        want = open(o, encoding='utf-8').read()
        msgs = d['messages'] if isinstance(d, dict) else d
        tools = d.get('tools', []) if isinstance(d, dict) else []
        mode = (d.get('thinking_mode') if isinstance(d, dict) else None) or 'thinking'
        msgs = copy.deepcopy(msgs)
        if msgs and msgs[0].get('tools'):
            tools = msgs[0].pop('tools')
        got = enc.encode_messages(to_encoder_input({'messages': msgs, 'tools': tools}), mode, reasoning_effort=None)
        if got != want:
            for alt in ('thinking', 'chat'):
                if enc.encode_messages(to_encoder_input({'messages': msgs, 'tools': tools}), alt) == want:
                    mode = alt
                    break
            else:
                sys.exit(f'official fixture {i} does not reproduce its published output with this encoder')
        out.append({'name': f'official_{i}', 'messages': msgs, 'tools': tools, 'thinking': mode == 'thinking', 'effort': None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, help='path to the checkpoint\'s encoding/encoding.py')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    enc = load_encoder(a.encoder)
    cases = official_fixtures(enc, os.path.dirname(os.path.abspath(a.encoder))) + CASES
    golden = []
    for c in cases:
        effort = c['effort']
        prompt = enc.encode_messages(to_encoder_input(c), 'thinking' if c['thinking'] else 'chat', reasoning_effort=effort)
        n = enc.REASONING_EFFORT_MAPPINGS.get(effort, effort) if effort is not None else 75
        golden.append({
            'name': c['name'],
            'request': {'messages': c['messages'], 'tools': c['tools'], 'thinking': c['thinking'], 'effort': n},
            'prompt': prompt,
        })
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(f'{len(golden)} cases -> {a.out}')


if __name__ == '__main__':
    main()
