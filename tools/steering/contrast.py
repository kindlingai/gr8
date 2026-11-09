#!/usr/bin/env python3
"""Make a steering profile from contrastive responses.

A pairs file (JSON lines) holds a prompt and two answers to it: `positive` shows the behaviour the profile should
push toward, `negative` the behaviour it moves away from:

    {"prompt": "Why is the sky blue?", "positive": "天空之所以是蓝色的……", "negative": "The sky is blue because ..."}

The steps (README.md has the whole run):

    contrast.py tokenize PAIRS OUT --api URL     the chat-template token ids of every (prompt, answer), and where the
                                                 answer starts, from a running front end's /tokenize
    contrast.py argv LOG OUT                     an offline g6m4 command for one rank: the serving command from a
                                                 supervisor log with the serve-only flags removed and OUT's prompts added
    contrast.py reduce OUT < DUMP                rank 0's GR8_RESID_DUMP stream -> per-layer sums of the attention and
                                                 MLP writer outputs over the answer tokens of each side (OUT/sums.npz)
    contrast.py fit OUT PROFILE --base M --name N   one direction and alpha per layer -> PROFILE (manifest.json and
                                                 directions.f32, checked with release.validate_directory)

The operator is a projection (docs/STEERING.md): y' = y - a * r (r . y) on each writer output y, so it scales the
part of y along r by (1 - a). The direction of layer l is the unit difference of the positive and negative mean
writer outputs (attention + MLP). With c+ and c- the two means' components along it, a = 1 - c+ / c- moves the
negative mean's component onto the positive one. The operator shrinks (0 < a < 1) or flips (a > 1) a component but
cannot grow it, and the schema caps a at 4: a layer where a falls outside [0, 4] is left out, and `fit` says so.

tokenize and argv need only the standard library; reduce and fit need numpy.
"""
import argparse
import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# the offline run generates one token per sequence; its decode step adds one dump row after the prompt's rows
GEN = 1


def die(msg):
    print(f'contrast.py: {msg}', file=sys.stderr)
    sys.exit(2)


def read_pairs(path):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            p = json.loads(line)
            for k in ('prompt', 'positive', 'negative'):
                if not isinstance(p.get(k), str) or not p[k]:
                    die(f'{path}:{n}: "{k}" must be a nonempty string')
            pairs.append(p)
    if not pairs:
        die(f'{path}: no pairs')
    return pairs


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------------------- tokenize
def tokenize_messages(api, model, messages):
    body = {'model': model, 'messages': messages, 'add_generation_prompt': False}
    req = urllib.request.Request(api.rstrip('/') + '/tokenize', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)['tokens']


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def cmd_tokenize(a):
    pairs = read_pairs(a.pairs)
    os.makedirs(os.path.join(a.out, 'prompts'), exist_ok=True)
    seqs = []
    for i, p in enumerate(pairs):
        user = ([{'role': 'system', 'content': p['system']}] if p.get('system') else []) + \
               [{'role': 'user', 'content': p['prompt']}]
        ids = {k: tokenize_messages(a.api, a.model, user + [{'role': 'assistant', 'content': p[k]}])
               for k in ('positive', 'negative')}
        # the answers start where the two renderings part; the closing tokens after them are shared too
        start = common_prefix(ids['positive'], ids['negative'])
        tail = common_prefix(ids['positive'][::-1], ids['negative'][::-1])
        for k, label in (('positive', 1), ('negative', 0)):
            end = len(ids[k]) - tail
            if end - start < 1:
                die(f'pair {i}: the {k} answer has no tokens of its own')
            f = os.path.join('prompts', f'{i:05d}.{k[:3]}.ids')
            with open(os.path.join(a.out, f), 'w') as o:
                o.write(','.join(map(str, ids[k])) + '\n')
            seqs.append({'file': f, 'label': label, 'tokens': len(ids[k]), 'start': start, 'end': end})
    with open(os.path.join(a.out, 'seqs.json'), 'w') as o:
        json.dump({'pairs': os.path.abspath(a.pairs), 'pairs_sha256': sha256_file(a.pairs), 'seqs': seqs}, o, indent=1)
    n = sum(s['end'] - s['start'] for s in seqs)
    print(f'{len(seqs)} sequences, {sum(s["tokens"] for s in seqs)} tokens, {n} answer tokens -> {a.out}/seqs.json')


# ----------------------------------------------------------------------------------------------------------- argv
# serving-only flags of g6m4 (with their value) and switches, dropped from the logged serving command
DROP_VAL = {'--serve', '--s2-port', '--s2-sessions', '--s2-epoch', '--kv-store', '--kv-keep', '--kv-system-after',
            '--kv-write-limit', '--dflash-manifest', '--dflash-pack', '--set-id', '--mtp', '--prefill', '--pf-lanes',
            '--temp', '--gen', '--top-p-cands', '--cap', '--steering-profile', '--steering-model-revision',
            '--steering-tokenizer-revision', '--steering-chat-template-sha256', '--steering-chat-template',
            '--steering-weight-set', '--steering-force'}
DROP_SW = {'--reuse-prefix', '--spec-adaptive', '--pfv2', '--graph'}
# the knobs of what the offline run leaves out (tensor prefill, MTP, the drafter, S2 serving, the rows tier, graphs):
# g6m4 refuses most of them without their flag. A log from before the GR8_ names has them under G6_.
DROP_ENV = tuple(p + k for p in ('GR8_', 'G6_') for k in ('PF_', 'MTP_', 'DRAFT_', 'DFL_', 'S2_', 'RT_', 'DEC_SHARED_SCRATCH=',
                                                         'DEC_SPLIT='))


def offline_argv(line, prompts, cap):
    """the env and argv of one rank's offline run, from the supervisor's `engine: ENV ./bin/g6m4 ARGS` log line"""
    words = shlex.split(line.split('engine: ', 1)[1])
    k = next(i for i, w in enumerate(words) if '=' not in w)
    env = [w for w in words[:k] if not w.startswith(DROP_ENV)]
    if not any(w.startswith('LD_LIBRARY_PATH=') for w in env):
        env.append('LD_LIBRARY_PATH=bin/lib')        # the supervisor adds it (bundled cuBLAS); run from the tree root
    args, it = [], iter(words[k + 1:])
    for w in it:
        if w in DROP_VAL:
            next(it, None)
        elif w not in DROP_SW:
            args.append(w)
    args += ['--cap', str(cap), '--gen', str(GEN), '--temp', '0']
    for p in prompts:
        args += ['--prompt', '@' + p]
    return env, [words[k]] + args


def cmd_argv(a):
    with open(a.log, encoding='utf-8', errors='replace') as f:
        lines = [x for x in f if ' engine: ' in x and '/g6m4' in x]
    if not lines:
        die(f'{a.log}: no "engine:" line (the supervisor log of a ./gr8 glm53 up)')
    seqs = json.load(open(os.path.join(a.out, 'seqs.json')))['seqs']
    prompts = [os.path.abspath(os.path.join(a.out, s['file'])) for s in seqs]
    cap = (max(s['tokens'] for s in seqs) + GEN + 63) // 64 * 64
    env, argv = offline_argv(lines[-1].rstrip('\n'), prompts, cap)
    print(' '.join(shlex.quote(w) for w in env + argv))


# --------------------------------------------------------------------------------------------------------- reduce
def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b


def cmd_reduce(a):
    import numpy as np
    seqs = json.load(open(os.path.join(a.out, 'seqs.json')))['seqs']
    total = sum(s['tokens'] + GEN for s in seqs)
    # the label of each row of the dump stream (in step order): 1 or 0 on answer rows [start, end), else -1
    keep = np.full(total, -1, np.int8)
    o = 0
    for s in seqs:
        keep[o + s['start']:o + s['end']] = s['label']
        o += s['tokens'] + GEN
    f = sys.stdin.buffer
    sums = cnt = None
    row = steps = 0
    while True:
        try:
            s, M, slots, hid = struct.unpack('<4i', read_exact(f, 16))
        except EOFError:
            break
        x = np.frombuffer(read_exact(f, slots * M * hid * 4), '<f4').reshape(slots, M, hid)
        if sums is None:
            nl = (slots - 1) // 2
            sums = np.zeros((2, 2 * nl, hid))    # [label][2 l = attention of layer l, 2 l + 1 = its MLP]
            cnt = np.zeros(2)
        if row + M > total:
            die(f'the dump has more rows than {a.out}/seqs.json ({total}): not this prompt set, or an extra step')
        lab = keep[row:row + M]
        w = np.diff(x, axis=0)                   # slot k + 1 - slot k: the writer output of sublayer k
        for v in (0, 1):
            m = lab == v
            if m.any():
                sums[v] += w[:, m, :].sum(axis=1, dtype=np.float64)
                cnt[v] += m.sum()
        row += M
        steps += 1
    if sums is None:
        die('no dump records on stdin')
    if row != total:
        die(f'the dump has {row} rows, {a.out}/seqs.json {total}: the run stopped early or the rows do not line up')
    np.savez(os.path.join(a.out, 'sums.npz'), sums=sums, count=cnt)
    print(f'{steps} steps, {row} rows: {int(cnt[1])} positive and {int(cnt[0])} negative answer tokens '
          f'-> {a.out}/sums.npz')


# ------------------------------------------------------------------------------------------------------------ fit
def parse_layers(text, nl):
    out = set()
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    bad = [l for l in out if not 0 <= l < nl]
    if bad:
        die(f'--layers: {bad} outside 0..{nl - 1}')
    return sorted(out)


def cmd_fit(a):
    import numpy as np
    sys.path.insert(0, HERE)
    import release
    z = np.load(os.path.join(a.out, 'sums.npz'))
    cnt = z['count']
    if min(cnt) < 1:
        die('sums.npz has no tokens on one side')
    mean = z['sums'] / cnt[:, None, None]        # [label][sublayer][hid]
    nl, hid = mean.shape[1] // 2, mean.shape[2]
    base = json.load(open(a.base))
    model = dict(base['model'])
    if model['hidden_size'] != hid or model['layer_count'] != nl:
        die(f'--base is a {model["hidden_size"]} x {model["layer_count"]} model, the dump {hid} x {nl}')
    if a.toward == 'negative' and not a.anchor:
        die('--toward negative needs --anchor: the default fit gives nearly the same profile both ways')
    # target and source means per layer: the writer outputs of attention + MLP
    tgt = 1 if a.toward == 'positive' else 0
    layers, alphas, dirs = [], [], []
    print(' layer    c src   c tgt   alpha  (c = the mean writer output along the direction)')
    for l in parse_layers(a.layers, nl):
        t, s = (mean[k, 2 * l] + mean[k, 2 * l + 1] for k in (tgt, 1 - tgt))
        d = t - s
        if a.anchor:                             # the part of the gap with no component along the target mean
            u = t / np.linalg.norm(t)
            d = (s - t) - ((s - t) @ u) * u
        d /= np.linalg.norm(d)
        ct, cs = float(t @ d), float(s @ d)
        alpha = a.strength if a.anchor else 1 - a.strength * ct / cs if cs else float('inf')
        ok = 0 <= alpha <= 4
        print(f'{l:6d} {cs:8.3f} {ct:7.3f} {alpha:7.3f}  {"kept" if ok else "left out"}')
        if ok:
            layers.append(l)
            alphas.append(round(alpha, 4))
            dirs.append(d)
    if not layers:
        die('no layer qualifies')
    os.makedirs(a.profile, exist_ok=True)
    payload = np.asarray(dirs, '<f4').tobytes()
    with open(os.path.join(a.profile, 'directions.f32'), 'wb') as o:
        o.write(payload)
    seqs = json.load(open(os.path.join(a.out, 'seqs.json')))
    commit = subprocess.run(['git', '-C', HERE, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    manifest = {
        'schema_version': 1, 'profile_name': a.name, 'model': model, 'scope': 'full', 'layers': layers,
        'alphas': alphas,
        'directions': {'file': 'directions.f32', 'dtype': 'float32-le', 'byte_count': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest()},
        # one pairs file trains the profile; it stands for all three sets
        'provenance': {'extraction_tool': 'tools/steering/contrast.py', 'extraction_commit': commit or 'unknown',
                       'dataset_manifests': {k: seqs['pairs_sha256'] for k in ('train', 'validation', 'final')},
                       'fit_sha256': sha256_file(os.path.join(a.out, 'sums.npz')), 'family': 'per-layer'},
    }
    with open(os.path.join(a.profile, 'manifest.json'), 'w') as o:
        json.dump(manifest, o, indent=1)
        o.write('\n')
    release.validate_directory(a.profile, expected=model)
    print(f'{len(layers)} layers -> {a.profile} (scope full, strength {a.strength})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('tokenize')
    p.add_argument('pairs')
    p.add_argument('out')
    p.add_argument('--api', default='http://127.0.0.1:8100')
    p.add_argument('--model', default='glm-5.3')
    p = sub.add_parser('argv')
    p.add_argument('log')
    p.add_argument('out')
    p = sub.add_parser('reduce')
    p.add_argument('out')
    p = sub.add_parser('fit')
    p.add_argument('out')
    p.add_argument('profile')
    p.add_argument('--base', required=True, help="a manifest.json of the same model, for its model identity")
    p.add_argument('--name', required=True)
    p.add_argument('--layers', default='0-77', help='candidate layers, e.g. 10-60 or 2,4,6')
    p.add_argument('--strength', type=float, default=1.0,
                   help='fraction of the way from the source to the target component (default 1)')
    p.add_argument('--anchor', action='store_true',
                   help='directions with the target mean at zero and alpha = strength: the profile erases what '
                        'sets the source apart and leaves target-like text nearly unchanged')
    p.add_argument('--toward', choices=('positive', 'negative'), default='positive',
                   help='the side to steer toward (negative needs --anchor)')
    a = ap.parse_args()
    {'tokenize': cmd_tokenize, 'argv': cmd_argv, 'reduce': cmd_reduce, 'fit': cmd_fit}[a.cmd](a)


if __name__ == '__main__':
    main()
