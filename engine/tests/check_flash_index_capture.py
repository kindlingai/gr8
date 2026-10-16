#!/usr/bin/env python3
"""Compare complete selected-pool captures from two ranks; refuse missing coverage.

check_flash_index_capture.py RANK0 RANK1 --layers 3,7 --prefill-rows 299999
No GPU, NumPy, or model weights required. A successful comparison proves only
these captured queries match; it does not establish prospective owner-path speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

HEADER = struct.Struct('<8s4I')
MAGIC = b'FPIX0001'
POOLS = 512
WORDS = 513


def compare(path0, path1, layers, prefill_rows):
    if sys.byteorder != 'little':
        raise ValueError('checker requires a little-endian host')
    if prefill_rows < 1 or not layers or len(set(layers)) != len(layers):
        raise ValueError('invalid expected coverage')
    next_row = {layer: 0 for layer in layers}
    counts = {layer: 0 for layer in layers}
    digest = hashlib.sha256()
    total_bytes = 0
    with open(path0, 'rb') as left, open(path1, 'rb') as right:
        while True:
            ha, hb = left.read(HEADER.size), right.read(HEADER.size)
            if not ha and not hb:
                break
            if len(ha) != HEADER.size or len(hb) != HEADER.size:
                raise ValueError('missing or truncated block header')
            if ha != hb:
                raise ValueError('rank block metadata differs')
            magic, layer, context, rows, pools = HEADER.unpack(ha)
            if magic != MAGIC or layer not in next_row or pools != POOLS:
                raise ValueError('invalid magic, layer, or pool count')
            if not 0 < rows <= context <= prefill_rows:
                raise ValueError('invalid block dimensions')
            first = context - rows
            if first != next_row[layer]:
                raise ValueError(f'layer {layer}: gap, duplicate, or reordered query range')
            size = rows * WORDS * 4
            a, b = left.read(size), right.read(size)
            if len(a) != size or len(b) != size:
                raise ValueError('truncated pool-ID payload')
            if a != b:
                av, bv = memoryview(a).cast('i'), memoryview(b).cast('i')
                word = next(i for i, (x, y) in enumerate(zip(av, bv)) if x != y)
                row, field = divmod(word, WORDS)
                raise ValueError(f'pool-ID mismatch layer={layer} query={first+row} field={field} '
                                 f'rank0={av[word]} rank1={bv[word]}')
            words = memoryview(a).cast('i')
            for r in range(rows):
                at, pos = r * WORDS, first + r
                if words[at] != pos:
                    raise ValueError('wrong query position')
                visible, previous = (pos + 1) // 4, -1
                keep = min(visible, POOLS)
                for j in range(POOLS):
                    pool = words[at + 1 + j]
                    if j < keep:
                        if not previous < pool < visible:
                            raise ValueError(f'invalid pool order/visibility at layer={layer} query={pos}')
                        previous = pool
                    elif pool != -1:
                        raise ValueError('invalid pool-ID padding')
            digest.update(ha)
            digest.update(a)
            total_bytes += HEADER.size + size
            next_row[layer] = context
            counts[layer] += 1
    if any(end != prefill_rows for end in next_row.values()):
        raise ValueError(f'incomplete coverage: {next_row}, expected {prefill_rows} per layer')
    return dict(status='PASS', exact_pool_id_bytes=True, layers=layers,
                prefill_rows_per_layer=prefill_rows, blocks_per_layer=counts,
                selected_pool_rows=prefill_rows * len(layers),
                bytes_per_rank=total_bytes, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rank0', type=Path)
    parser.add_argument('rank1', type=Path)
    parser.add_argument('--layers', required=True)
    parser.add_argument('--prefill-rows', required=True, type=int)
    args = parser.parse_args()
    try:
        result = compare(args.rank0, args.rank1, [int(x) for x in args.layers.split(',')], args.prefill_rows)
    except (OSError, ValueError) as error:
        print(json.dumps(dict(status='FAIL', error=str(error))))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
