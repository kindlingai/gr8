"""Merge g6m4 rank logits into m4sim framing, without float conversion.

python3 tools/cp/merge_logits.py -o logits.bin logits.bin.rank*
Input: G6LGR001 + int32 NR, rank, vocab, vlo, vn; then int32 step, M
and M*vn float32 per step. Output: int32 step, M, then int32 vn and
M*vn float32 for each part in vocabulary order. Integers are little-endian.
"""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import struct
import tempfile

MAGIC = b'G6LGR001'
HEADER = struct.Struct('<5i')
STEP = struct.Struct('<2i')


def exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f'{f.name}: truncated dump ({len(data)} of {n} bytes)')
    return data


def merge(paths, output):
    """Validate coverage and aligned records; atomically publish output."""
    output = Path(output)
    if any(Path(p).resolve() == output.resolve() for p in paths):
        raise ValueError('output must not overwrite an input')
    with ExitStack() as stack:
        ranks = []
        for path in paths:
            f = stack.enter_context(open(path, 'rb'))
            if exact(f, 8) != MAGIC:
                raise ValueError(f'{path}: bad rank dump magic/version')
            nr, rank, vocab, vlo, vn = HEADER.unpack(exact(f, HEADER.size))
            if nr < 1 or not 0 <= rank < nr or vocab < 1 or vlo < 0 or vn < 1 or vlo + vn > vocab:
                raise ValueError(f'{path}: invalid rank header')
            ranks.append((vlo, vn, nr, rank, vocab, f))
        if not ranks:
            raise ValueError('no input rank files')
        nr, vocab = ranks[0][2], ranks[0][4]
        if len(ranks) != nr or any(x[2] != nr or x[4] != vocab for x in ranks):
            raise ValueError('rank count or vocabulary mismatch')
        if {x[3] for x in ranks} != set(range(nr)):
            raise ValueError('duplicate or missing rank')
        ranks.sort(key=lambda x: x[0])
        next_row = 0
        for vlo, vn, *_ in ranks:
            if vlo != next_row:
                raise ValueError('overlapping or missing vocabulary rows')
            next_row += vn
        if next_row != vocab:
            raise ValueError('incomplete vocabulary')
        fd, tmp = tempfile.mkstemp(prefix=output.name + '.', dir=output.parent)
        count = 0
        try:
            with os.fdopen(fd, 'wb') as out:
                while True:
                    headers = [r[-1].read(STEP.size) for r in ranks]
                    if not any(headers):
                        break
                    if any(len(h) != STEP.size for h in headers):
                        raise ValueError('truncated step header or unequal step counts')
                    if any(h != headers[0] for h in headers):
                        raise ValueError('step number or row count mismatch')
                    step, rows = STEP.unpack(headers[0])
                    if step < 0 or rows < 1:
                        raise ValueError('invalid step number or row count')
                    out.write(headers[0])
                    for _, vn, _, _, _, f in ranks:
                        out.write(struct.pack('<i', vn))
                        remaining = rows * vn * 4
                        while remaining:
                            n = min(remaining, 1 << 20)
                            out.write(exact(f, n))
                            remaining -= n
                    count += 1
                if count == 0:
                    raise ValueError('empty dump')
            os.replace(tmp, output)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('ranks', nargs='+')
    a = ap.parse_args()
    try:
        count = merge(a.ranks, a.output)
    except (ValueError, OSError) as e:
        ap.exit(1, f'merge_logits: {e}\n')
    print(f'merged {len(a.ranks)} ranks, {count} steps: {a.output}')


if __name__ == '__main__':
    main()
