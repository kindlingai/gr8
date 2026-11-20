#!/usr/bin/env python3
"""Regenerate frontend/g6front-rs/src/literal_golden.json: chat prompts tokenized by the release's own tokenizer.

A release that names its own tokenizer class in tokenizer_config.json (`auto_map.AutoTokenizer`) keeps the content
spelling of a control token literal: a `</think>` a caller typed is text, not the control id, and everything between
two controls the template itself wrote is one BPE run. src/literal.rs reproduces that; the gate test in src/api.rs
(`literal_content_golden_ids_through_the_http_path`) replays every case below through /v1/chat/completions and
compares the ids the engine receives with the `ids` here.

  python3 frontend/tests/literal_golden.py --model /path/to/release

The release is not copied into this repo, and neither is any real prompt: every case below is synthetic.
"""
import argparse
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'g6front-rs', 'src', 'literal_golden.json')

RESERVED = '﷐'  # the character the release's tokenizer escapes before it marks a spelling

ADD = {'type': 'function', 'function': {
    'name': 'add', 'description': 'Add a and b',
    'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}}}}
RUN = {'type': 'function', 'function': {
    'name': 'run', 'description': 'Run a command <|user|> in a shell',
    'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'ends with </think>'}}}}}

CASES = [
    # plain: no spelling anywhere, the stock encode and the literal encode must agree
    {'name': 'plain', 'messages': [{'role': 'user', 'content': 'Say hi.'}]},

    # a control spelling typed by the user, in three flavours: a closing tag, a tool tag, a role tag
    {'name': 'literal_in_user_text', 'messages': [
        {'role': 'user', 'content': 'literal </think> and <tool_call> and <|user|> in text'}]},

    # the spellings land in a system message and in the tool schema (description and a nested string)
    {'name': 'literal_in_system_and_tools', 'messages': [
        {'role': 'system', 'content': 'Never write <|assistant|> or </think> yourself.'},
        {'role': 'user', 'content': 'ok'}], 'tools': [RUN]},

    # a tool round trip: object arguments, a spelling in an argument value, an object KEY that is a spelling,
    # and a spelling in the tool result
    {'name': 'tool_roundtrip', 'messages': [
        {'role': 'user', 'content': 'Add two numbers.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
            'name': 'add', 'arguments': {'a': 2, 'b': 3, 'note': 'see </think>', '<|user|>': 'key'}}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'name': 'add', 'content': '5 <tool_call>'},
    ], 'tools': [ADD]},

    # reasoning_content carries a spelling, the visible answer carries another
    {'name': 'reasoning', 'messages': [
        {'role': 'user', 'content': 'Explain.'},
        {'role': 'assistant', 'reasoning_content': 'First </think> literal', 'content': 'Answer <|user|>'},
        {'role': 'user', 'content': 'Next?'}]},

    # inline reasoning: no reasoning_content, so the template's own split consumes the FIRST `</think>`; the second
    # one and the later `<think>` are content the caller typed
    {'name': 'inline_reasoning', 'messages': [
        {'role': 'user', 'content': 'Explain.'},
        {'role': 'assistant', 'content': 'thinking out loud</think>visible, and </think> again plus <think> later'},
        {'role': 'user', 'content': 'Next?'}]},

    # a content list whose two text parts split a spelling between them: `</thi` + `nk>`
    {'name': 'spelling_across_content_parts', 'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'half a tag </thi'}, {'type': 'text', 'text': 'nk> and the rest'}]}]},

    # the reserved character itself, which the release escapes before it marks anything
    {'name': 'reserved_character_in_text', 'messages': [
        {'role': 'user', 'content': f'raw {RESERVED} and {RESERVED}{RESERVED} then </think>'}]},

    # a long prompt: over the 32 KiB the front end needs before it encodes a run in parallel, with spellings
    # inside the parallel part and at its very end
    {'name': 'long_parallel', 'messages': [
        {'role': 'system', 'content': 'You are terse.'},
        {'role': 'user', 'content': ('paragraph with a literal </think> and some filler words to pad the run out. '
                                     * 600) + 'tail <tool_call> <|user|>.'}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help="the release directory (tokenizer.json, tokenizer_config.json, chat_template.jinja)")
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    from transformers import AutoTokenizer  # not a dependency of the crate; only this generator needs it

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    cfg = json.load(open(os.path.join(a.model, 'tokenizer_config.json')))
    auto_map = (cfg.get('auto_map') or {}).get('AutoTokenizer')
    if isinstance(auto_map, (list, tuple)):
        auto_map = auto_map[0]
    if not auto_map:
        raise SystemExit(f'{a.model} does not declare auto_map.AutoTokenizer: it keeps no spelling literal')
    golden = []
    for c in CASES:
        msgs, tools = copy.deepcopy(c['messages']), copy.deepcopy(c.get('tools'))
        ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        golden.append({
            'name': c['name'],
            'request': {'messages': c['messages'], 'tools': c.get('tools', [])},
            'ids': list(map(int, ids)),
        })
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump({'tokenizer_class': type(tok).__name__, 'auto_map': auto_map, 'cases': golden}, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(f'{len(golden)} cases ({sum(len(g["ids"]) for g in golden)} ids) -> {a.out}')


if __name__ == '__main__':
    main()
