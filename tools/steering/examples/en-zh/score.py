#!/usr/bin/env python3
"""The share of Chinese in answers to held-out English questions, at each steering scale.

The server must have the profile loaded (STEERING_PROFILE in cluster.env, without STEERING_FORCE). Scale 0 is the
stock answer. A working profile raises the share with the scale while the answers stay on topic, so read some.

    score.py heldout.txt --profile en-zh-v1 --scales 0,0.25,0.5,1 --api http://HEAD:8100 > score.json
"""
import argparse
import json
import re
import sys
import urllib.request


def cjk_share(text):
    """Chinese characters over Chinese characters + English words (a character is about a word)"""
    zh = len(re.findall('[一-鿿]', text))
    en = len(re.findall('[A-Za-z]+', text))
    return zh / (zh + en) if zh + en else 0.0


def answer(a, question, scale):
    body = {'model': a.model, 'temperature': 0, 'max_tokens': a.max_tokens,
            'messages': [{'role': 'user', 'content': question}],
            'chat_template_kwargs': {'reasoning_effort': a.effort}}
    if scale:
        body.update(steering_profile=a.profile, steering_scale=scale)
    req = urllib.request.Request(a.api.rstrip('/') + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        m = json.load(r)['choices'][0]['message']
    return m.get('reasoning_content') or m.get('reasoning') or '', m.get('content') or ''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('questions')
    ap.add_argument('--profile', required=True)
    ap.add_argument('--scales', default='0,0.25,0.5,1')
    ap.add_argument('--api', default='http://127.0.0.1:8100')
    ap.add_argument('--model', default='glm-5.3')
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--effort', default='low', help='reasoning_effort (low: answers without reasoning)')
    a = ap.parse_args()
    qs = [q.strip() for q in open(a.questions, encoding='utf-8') if q.strip()]
    rows = []
    for scale in map(float, a.scales.split(',')):
        think = content = 0.0
        for q in qs:
            r, c = answer(a, q, scale)
            rows.append({'scale': scale, 'question': q, 'reasoning': r, 'content': c})
            think += cjk_share(r)
            content += cjk_share(c)
        print(f'scale {scale:4}: Chinese share {content / len(qs):.2f} of answers, {think / len(qs):.2f} of '
              f'reasoning ({len(qs)} questions)', file=sys.stderr)
    json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
