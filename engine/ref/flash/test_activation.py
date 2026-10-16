"""Host test of the C++ helper against W0 (not a substitute for the GPU gate)."""
import os
from pathlib import Path
import subprocess
import tempfile
import torch
from flash_ref import swiglu


def main():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        for flash in (0, 1):
            exe = str(Path(tmp) / 'activation')
            subprocess.run(['g++', '-O2', '-std=c++17', f'-DG6_MODEL_FLASH={flash}',
                            str(root / 'engine/kernels/tests/activation_test.cpp'), '-o', exe], check=True)
            rows = [list(map(float, s.split())) for s in subprocess.check_output([exe], text=True).splitlines()]
            g, u, actual = torch.tensor(rows, dtype=torch.float32).T
            expected = swiglu(g, u, 10. if flash else None)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
            rel = (actual - expected).norm() / expected.norm()
            print(f'activation flash={flash}: {len(rows)} clamp-boundary cases PASS; W0 rel-L2={rel:.3g}')


if __name__ == '__main__':
    main()
