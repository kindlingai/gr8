#!/usr/bin/env python3
"""Regenerate frontend/g6front-rs/src/ds41_vision_golden.json: DeepSeek-V4.1 prompts with image blocks, rendered by the
release's own encoder (encode_messages with return_multi_modal_data=True). `cargo test` checks dschat::render_mm against
it: the prompt text byte for byte and the order of the image records (the encoder collects them after it sorts the tool
results). Each image's data is a tag (IMG<n>), so the order is readable.

  python3 frontend/tests/ds41_vision_golden.py --encoder /path/to/ds41-flash-1017/encoding/encoding.py
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'g6front-rs', 'src', 'ds41_vision_golden.json')


def U(n):
    return {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,IMG{n}'}}


def A(n):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': f'IMG{n}'}}


def T(s):
    return {'type': 'text', 'text': s}


CASES = {
    'user_two': [{'role': 'user', 'content': [T('compare'), U(0), A(1)]}],
    'system_and_assistant': [
        {'role': 'system', 'content': [U(0), T('sys')]},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': [T('look'), U(1)]},
        {'role': 'user', 'content': [U(2), T('what')]}],
    'tool_results_sorted': [
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}},
            {'id': 'b', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': [T('B'), U(1)]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [U(0), T('A')]}],
    'anthropic_tool_result_block': [
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': [A(0), T('r')]},
                                     T('and'), U(1)]}],
}


def image_tag(rec):
    if 'url' in rec:
        return rec['url'].split(',', 1)[1]
    return rec['source']['data']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, help="the release's encoding/encoding.py")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('ds41_encoding', a.encoder)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    out = []
    for name, msgs in CASES.items():
        for mode in ('chat', 'thinking'):
            prompt, mm = enc.encode_messages(msgs, thinking_mode=mode, return_multi_modal_data=True)
            out.append({'name': f'{name}/{mode}', 'messages': msgs, 'thinking': mode == 'thinking', 'effort': 75,
                        'prompt': prompt, 'order': [image_tag(r) for r in mm['images']]})
    with open(OUT, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(f'{len(out)} cases -> {os.path.normpath(OUT)}')


if __name__ == '__main__':
    main()
