#!/usr/bin/env python3
"""Every refusal of g6m4's option parse reached by at least one case of options_cases.txt.

    scripts/check_options_coverage.py engine/scheduler-rs/src/options.rs DUMP...

DUMP: options_vectors_test's --dump output, one file per shape (scripts/host_tests.sh writes them). Each Err( of the
parse in options.rs (the file up to its C ABI) must match a "g6: " line of some dump: every string literal of its
message, in order, with a format string's {..} placeholders as gaps. A refusal that forwards a helper's message names
the messages expected below; the ones no case can reach are listed with the reason. Fails naming each refusal no
case reached, and each listed one the source no longer has.
"""
import re
import sys

# Err(ARG) with no literal: the helper's messages a case must reach
FORWARDED = {
    'err': ['GR8_DS_MTP must be 0..5', 'GR8_DS_BLOCK must be 5', 'GR8_DS_RELAXED_ACCEPT must be 0 / off',
                'GR8_DS_VERIFY_CAP must be a number', 'GR8_DS_CONF_TEMP must be positive', 'GR8_DS_CONF_TEMP: one value',
                'GR8_DS_PREFILL_KEEP must be 0'],                                                    # ds41_dspark.h parse_knobs
    'why': ['GR8_PF_QP_WIRE_E is set but GR8_PF_QP_ROLL is not', 'GR8_PF_QP_ROLL needs GR8_PF_OWN=4 and --prefill',
            'GR8_PF_QP_ROLL: no --qp-pack', 'GR8_PF_QP_ROLL needs GR8_PF_QP_CTX'],                     # roll_env_refusal
}
# keyed by the start of the first literal (or the argument): why no case reaches it
UNREACHABLE = {
    'Flash --layers all did not select every main layer': 'parse_ints("all") gives every main layer',
    'Flash --layers all selected an unexpected layer': 'parse_ints("all") gives every main layer in order',
    '--dflash: TP=': 'only a GLM build of width 3 or 5 refuses; the shapes are TP4 and TP6',
    'Flash serve with DFlash2 needs --mtp': 'the serve check refuses --mtp 0 first',
    'this engine is the int4 KV build (': 'no shape is an int4 KV build',
    'DS41: --s2-sessions x (GR8_DS_MTP + 1) verify rows': '8 sessions x 6 rows fit the DS41 shape\'s 48',
    'GR8_GLM53F_CACHE=': 'the knob table takes only the values flash_cache_mode takes',
}


def refusals(src):
    src = src[:src.index(' the C ABI\n')]
    out = []
    for m in re.finditer(r'\bErr\(', src):
        i, depth, quote = m.end(), 1, False
        while depth:
            c = src[i]
            if quote:
                if c == '\\':
                    i += 1
                elif c == '"':
                    quote = False
            elif c == '"':
                quote = True
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            i += 1
        arg = src[m.end():i - 1].strip()
        lits = [p for x in re.findall(r'"((?:[^"\\]|\\.)*)"', arg) for p in re.split(r'\{[^{}]*\}', x.replace('\\"', '"')) if p]
        out.append((src.count('\n', 0, m.start()) + 1, arg, lits))
    return out


def reached(pieces, lines):
    for line in lines:
        at = 0
        for p in pieces:
            at = line.find(p, at)
            if at < 0:
                break
            at += len(p)
        else:
            return True
    return False


def main():
    src = open(sys.argv[1]).read()
    lines = [x[4:] for f in sys.argv[2:] for x in open(f).read().splitlines() if x.startswith('g6: ')]
    bad, seen, n = [], set(), 0
    for line, arg, lits in refusals(src):
        first = lits[0] if lits else arg
        skip = [k for k in UNREACHABLE if first.startswith(k)]
        seen.update(skip)
        if skip:
            continue
        if lits:
            n += 1
            if not reached(lits, lines):
                bad.append(f'options.rs:{line}: Err({arg[:90]})')
        elif arg in FORWARDED:
            for msg in FORWARDED[arg]:
                n += 1
                if not reached([msg], lines):
                    bad.append(f'options.rs:{line}: Err({arg}): "{msg}"')
        else:
            bad.append(f'options.rs:{line}: Err({arg}): a forwarded message this script does not list')
    bad += [f'listed as unreachable, not in options.rs: {k}' for k in UNREACHABLE if k not in seen]
    for b in bad:
        print('options coverage: no case reaches ' + b if 'listed' not in b else 'options coverage: ' + b)
    print(f'options coverage: {n - len([b for b in bad if "listed" not in b])} of {n} refusals reached, '
          f'{len(UNREACHABLE)} unreachable')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
