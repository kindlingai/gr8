#!/usr/bin/env python3
"""Read diagnostic G6_DRAFT_HEAD_DUMP files. No third-party dependencies.
Consume calls() fully: only EOF after a checked DONE footer is a complete capture.
The hidden is normalized draft FP32 xn, BEFORE the head's FP16 fragment conversion.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

HEADER = struct.Struct('<8s8I')
CALL = struct.Struct('<3Q4i')
ROW = struct.Struct('<4i2Ii3f')
DONE = struct.Struct('<2Q')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def exact(f, n):
    data = f.read(n)
    require(len(data) == n, 'truncated capture (missing data or DONE footer)')
    return data


class Capture:
    def __init__(self, path):
        self.path = Path(path)
        self.complete = False
        self.total_calls = self.total_rows = 0
        with self.path.open('rb') as f:
            h = HEADER.unpack(exact(f, HEADER.size))
        require(h[0] == b'G6DHD001', 'bad magic/version')
        self.nr, self.rank, self.hid, self.vocab, self.mx, self.k, self.adaptive, dtype = h[1:]
        require(1 <= self.nr <= 6 and 0 <= self.rank < self.nr, 'bad rank identity')
        require(self.hid == 4096 and self.vocab == 154880, 'not Flash hidden/vocab shape')
        require(2 <= self.mx <= 32 and 1 <= self.k < self.mx, 'bad row/k shape')
        require(self.adaptive in (0, 1) and dtype == 1, 'bad mode/dtype')
        self.header = h

    def calls(self):
        self.complete = False
        count = total = 0
        prev = None
        with self.path.open('rb') as f:
            require(HEADER.unpack(exact(f, HEADER.size)) == self.header, 'header changed')
            while True:
                tag = exact(f, 4)
                if tag == b'DONE':
                    require(DONE.unpack(exact(f, DONE.size)) == (count, total), 'footer totals mismatch')
                    require(not f.read(1), 'bytes after footer')
                    self.total_calls, self.total_rows = count, total
                    self.complete = True
                    return
                require(tag == b'CALL', 'bad record tag')
                call, seq, step, s, m, chain, context = CALL.unpack(exact(f, CALL.size))
                require(call == count and step > 0 and s >= 0, 'bad call/step identity')
                require(1 <= m <= self.mx and chain in (0, 1) and (not chain or m == 1), 'bad call shape')
                row_bytes = exact(f, m * ROW.size)
                rows = list(ROW.iter_unpack(row_bytes))
                for r, row in enumerate(rows):
                    pos, inp, out, pred, seed_lo, seed_hi, draw, inv_temp, top_p, lp = row
                    require(pos >= 0 and pred == pos + 2 and pos == rows[0][0] + r, 'bad row positions')
                    require(0 <= inp < self.vocab and 0 <= out < self.vocab, 'bad token id')
                    require(draw >= 0 and math.isfinite(inv_temp) and inv_temp >= 0 and math.isfinite(top_p), 'bad sampling fields')
                    require(math.isfinite(lp) and lp <= 1e-5, 'bad log probability')
                require(context == rows[-1][0] + 1, 'bad context')
                if prev:
                    require(seq >= prev['seq'] and step == prev['step'] + 1, 'non-monotone sequence/step')
                if chain:
                    require(prev is not None and seq == prev['seq'], 'chain without first form')
                    require(s == (prev['s'] + 1 if prev['chain'] else 1) and s < self.k, 'bad chain index')
                    require(rows[0][0] == prev['rows'][-1][0] + 1 and rows[0][1] == prev['rows'][-1][2], 'broken draft chain')
                hidden = exact(f, m * self.hid * 4)
                values = array.array('f')
                values.frombytes(hidden)
                if sys.byteorder != 'little':
                    values.byteswap()
                require(all(math.isfinite(v) for v in values), 'non-finite hidden')
                rec = dict(call=call, seq=seq, step=step, s=s, m=m, chain=chain,
                           context=context, rows=rows, row_bytes=row_bytes, hidden=hidden)
                prev = rec
                count += 1
                total += m
                yield rec


def compare_ranks(paths):
    """Require all ranks and identical decisions/LP *bits*, not rounded stdout.
    Hidden bytes are checked too: these ranks must see the same replicated xn.
    """
    from itertools import zip_longest
    caps = [Capture(p) for p in paths]
    require(caps and len(caps) == caps[0].nr, 'incomplete rank set')
    require(sorted(c.rank for c in caps) == list(range(len(caps))), 'duplicate/missing rank')
    base = caps[0].header
    require(all(c.header[:2] + c.header[3:] == base[:2] + base[3:] for c in caps), 'rank header mismatch')
    digest = hashlib.sha256()
    for records in zip_longest(*(c.calls() for c in caps)):
        require(all(r is not None for r in records), 'rank call counts differ')
        first = records[0]
        require(all(r == first for r in records), 'rank metadata/ids/LP bits/hidden mismatch')
        digest.update(CALL.pack(*(first[k] for k in ('call', 'seq', 'step', 's', 'm', 'chain', 'context'))))
        digest.update(first['row_bytes'])
        digest.update(first['hidden'])
    require(all(c.complete for c in caps), 'incomplete capture')
    return dict(calls=caps[0].total_calls, rows=caps[0].total_rows,
                ranks=len(caps), fixed_k=None if caps[0].adaptive else caps[0].k,
                logical_sha256=digest.hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ranks', nargs='+', type=Path)
    args = ap.parse_args()
    print(json.dumps(compare_ranks(args.ranks), indent=2))


if __name__ == '__main__':
    main()
