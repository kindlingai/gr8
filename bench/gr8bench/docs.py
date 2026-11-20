"""Synthetic long documents: seeded, varied Python-like code, sized to a token count with the server's tokenizer."""
import random

_NOUNS = ('buffer cursor packet frame ledger window shard record token stream queue index table cache page lane '
          'ticket socket digest ring slot batch chunk header').split()
_VERBS = ('merge split scan flush fold probe route drain mark seal pack unpack rotate trim sort count').split()
_OPS = ['+', '-', '*', '//', '%', '^', '&', '|']


def _fn(rng):
    name = f'{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}_{rng.randrange(10**6):06d}'
    a, b, c = (f'{rng.choice(_NOUNS)}{rng.randrange(100)}' for _ in range(3))
    k1, k2, k3 = (rng.randrange(1, 10**5) for _ in range(3))
    return [
        f'def {name}({a}, {b}, {c}={k1}):',
        f'    """{rng.choice(_VERBS).capitalize()} the {rng.choice(_NOUNS)} of {a} against {b} (limit {k2})."""',
        f'    acc = {a} {rng.choice(_OPS)} {k2}',
        f'    for i in range({k3 % 17 + 2}):',
        f'        acc = (acc {rng.choice(_OPS)} {b}) {rng.choice(_OPS)} (i + {rng.randrange(1, 999)})',
        f'        if acc > {k1}:',
        f'            acc -= {k2}',
        f'    return acc {rng.choice(_OPS)} {c}',
        '',
    ]


def make_lines(rng, n_functions):
    out = []
    for _ in range(n_functions):
        out += _fn(rng)
    return out


def passphrase(rng):
    w = ('amber basil cedar delta ember fjord grove harbor ivory juniper kestrel lumen maple nectar onyx pearl quartz '
         'raven sable tundra umber velvet willow xenon yarrow zephyr').split()
    return '-'.join(rng.choice(w) for _ in range(3)) + f'-{rng.randrange(1000, 9999)}'


def _tokens(client, lines):
    return client.count('\n'.join(lines))


def sized_doc(client, target_tokens, seed, tol=0.01):
    """Code lines whose server-side token count is within `tol` of target_tokens. -> list of lines.
    Calibrates tokens/line on a sample, builds, then trims or extends by whole functions."""
    rng = random.Random(seed)
    sample = make_lines(rng, 200)
    per_line = max(client.count('\n'.join(sample)) / len(sample), 1e-3)
    lines = make_lines(rng, max(1, int(target_tokens / per_line / 9)))
    for _ in range(8):
        n = _tokens(client, lines)
        if abs(n - target_tokens) <= tol * target_tokens:
            return lines
        per_line = n / len(lines)
        delta = int((target_tokens - n) / per_line)
        if delta > 0:
            lines += make_lines(rng, max(1, delta // 9))
        else:
            lines = lines[:max(9, len(lines) + delta)]
    return lines


def needle_prompt(client, target_tokens, depth, seed, secret):
    """Code document with a passphrase line at `depth` (0.15 = 15%) of its tokens, then the question.
    -> (messages, info). The document starts with a per-seed salt line so repeated runs do not hit the prefix cache."""
    lines = sized_doc(client, target_tokens, seed)
    idx = int(len(lines) * depth)
    while idx < len(lines) and lines[idx] != '':
        idx += 1  # land between functions
    needle = f'# PASSPHRASE: {secret}'
    lines = [f'# build {seed:08x}'] + lines[:idx] + [needle] + lines[idx:]
    doc = '\n'.join(lines)
    before = client.count('\n'.join(lines[:idx + 1]))
    total = client.count(doc)
    question = ('Somewhere in the code above there is one comment line that starts with "# PASSPHRASE:". '
                'Reply with the passphrase from that line.')
    info = {'doc_tokens': total, 'needle_depth': round(before / total, 4)}
    return [{'role': 'user', 'content': doc + '\n\n' + question}], info


def filler_prompt(client, target_tokens, seed, ask):
    """Salted code document of ~target_tokens followed by an instruction. -> messages."""
    lines = [f'# build {seed:08x}'] + sized_doc(client, target_tokens, seed)
    return [{'role': 'user', 'content': '\n'.join(lines) + '\n\n' + ask}]
