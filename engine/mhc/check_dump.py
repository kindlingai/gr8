"""Check a Flash m4sim GR8_RESID_DUMP against the layers 0-3 prompt goldens.

Run m4sim with exactly the golden metadata's tokens, --gen 0 and --layers 0,1,2,3.
Pass --layer3 for matching exlr8 expert weights (for example layer3-K4.safetensors).
This checks residuals only, not logits or the token source of the binary dump.
"""
import argparse
import struct

import numpy as np


def read_dump(path):
    chunks = []
    with open(path, 'rb') as f:
        while True:
            header = f.read(16)
            if not header:
                break
            if len(header) != 16:
                raise ValueError('truncated residual header')
            step, rows, slots, width = struct.unpack('<4i', header)
            if step != len(chunks) or not 1 <= rows <= 32 or slots != 9 or width != 4 * 4096:
                raise ValueError(f'bad residual record: {step, rows, slots, width}')
            nbytes = rows * slots * width * 4
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError('truncated residual payload')
            x = np.frombuffer(raw, dtype='<f4').reshape(slots, rows, 4, 4096)
            if not np.isfinite(x).all():
                raise ValueError('non-finite residual')
            chunks.append(x)
    if not chunks:
        raise ValueError('empty residual dump')
    return np.concatenate(chunks, axis=1)


def check(actual, expected, gate=2e-2):
    if actual.shape != expected.shape:
        raise ValueError(f'residual shape {actual.shape} != golden {expected.shape}')
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if not np.isfinite(b).all():
        raise ValueError('non-finite golden')
    den = max(float(np.linalg.norm(b)), 1e-30)
    err = float(np.linalg.norm(a - b)) / den
    per_row = np.linalg.norm((a - b).reshape(len(a), -1), axis=1) / np.maximum(
        np.linalg.norm(b.reshape(len(b), -1), axis=1), 1e-30)
    return err <= gate, err, float(per_row.max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump')
    p.add_argument('golden')
    p.add_argument('--layer3', help='same-input layer-3 golden for the chosen expert width')
    p.add_argument('--gate', type=float, default=2e-2)
    a = p.parse_args()
    from safetensors import safe_open
    x = read_dump(a.dump)
    names = ['L00.in'] + [f'L{l:02d}.{stage}' for l in range(4) for stage in ('mid', 'out')]
    bad = 0
    with safe_open(a.golden, framework='numpy') as g:
        for slot, name in enumerate(names):
            if a.layer3 and name == 'L03.out':
                with safe_open(a.layer3, framework='numpy') as q:
                    ref = q.get_tensor(name)
            else:
                ref = g.get_tensor(name)
            ok, err, worst = check(x[slot], ref, a.gate)
            bad += not ok
            print(f'{name}: rel-L2={err:.8g}, worst-row={worst:.8g}, {"PASS" if ok else "FAIL"}')
    print(f'{"FAIL" if bad else "PASS"}: {len(names)} residual checks, {bad} failed, rows={x.shape[1]}')
    return int(bool(bad))


if __name__ == '__main__':
    raise SystemExit(main())
