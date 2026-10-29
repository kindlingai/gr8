#!/usr/bin/env python3
"""Contrastive pairs for an English -> Chinese profile, written by the served model itself.

Each question gets two answers at temperature 0: one with a system turn asking for English, one asking for Chinese.
The pair keeps only the question as its prompt, so the capture sees the same user turn on both sides and the answers
differ only in language. Questions after the first --train go to a held-out file for score.py.

    make_pairs.py questions.txt OUTDIR --api http://HEAD:8100 --model glm-5.3
"""
import argparse
import json
import os
import urllib.request

SYSTEM = {'positive': '请只用简体中文回答。', 'negative': 'Answer in English only.'}


def answer(api, model, system, question, max_tokens):
    body = {'model': model, 'temperature': 0, 'max_tokens': max_tokens,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
            'chat_template_kwargs': {'reasoning_effort': 'low'}}
    req = urllib.request.Request(api.rstrip('/') + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        c = json.load(r)['choices'][0]
    return c['message'].get('content') or '', c.get('finish_reason')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('questions')
    ap.add_argument('out')
    ap.add_argument('--api', default='http://127.0.0.1:8100')
    ap.add_argument('--model', default='glm-5.3')
    ap.add_argument('--train', type=int, default=48, help='questions used for pairs; the rest are held out')
    ap.add_argument('--max-tokens', type=int, default=1024)
    a = ap.parse_args()
    qs = [q.strip() for q in open(a.questions, encoding='utf-8') if q.strip()]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'heldout.txt'), 'w', encoding='utf-8') as o:
        o.write(''.join(q + '\n' for q in qs[a.train:]))
    with open(os.path.join(a.out, 'pairs.jsonl'), 'w', encoding='utf-8') as o:
        for i, q in enumerate(qs[:a.train]):
            pair = {'prompt': q}
            for k, system in SYSTEM.items():
                text, finish = answer(a.api, a.model, system, q, a.max_tokens)
                if finish != 'stop':
                    print(f'{i}: the {k} answer stopped on {finish}; kept as it is')
                pair[k] = text.strip()
            o.write(json.dumps(pair, ensure_ascii=False) + '\n')
            print(f'{i}: {q}  en {len(pair["negative"])} chars, zh {len(pair["positive"])} chars')


if __name__ == '__main__':
    main()
