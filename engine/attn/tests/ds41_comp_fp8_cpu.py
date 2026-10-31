#!/usr/bin/env python3
"""CPU-only E4M3/UE8M0 compressor prepack oracle (synthetic BF16); no GPU kernel execution."""
import torch

N, K, M = 64, 5120, 16
# Each of ckv and csc must be finite, NRMSE <= 0.10,
# and max_abs_err <= 0.20 * max_abs_reference + 1e-5. wo_b stays bit-exact against the c2 GEMM.
NRMSE_LIMIT = 0.10
MAX_LIMIT = 0.20


def q32(x):
    blocks = x.float().reshape(*x.shape[:-1], K // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0))
    scale = torch.pow(torch.tensor(2.0), exp)
    code = (blocks / scale[..., None]).to(torch.float8_e4m3fn)
    return code.reshape(x.shape), (exp + 127).to(torch.uint8), (code.float() * scale[..., None]).reshape(x.shape)


if __name__ == '__main__':
    for seed in (3, 19, 97):
        torch.manual_seed(seed)
        x = (torch.randn(M, K) * 0.6).to(torch.bfloat16)
        _, sx, qx = q32(x)
        assert sx.shape == (M, K // 32)
        for role in ('ckv', 'csc'):
            w = (torch.randn(N, K) * (0.06 if role == 'ckv' else 0.04)).to(torch.bfloat16)
            codes, sw, qw = q32(w)
            assert codes.shape == (N, K) and sw.shape == (N, K // 32)
            assert torch.isfinite(qw).all() and int(sw.min()) > 0 and int(sw.max()) < 255
            ref = x.float() @ w.float().T
            cand = qx @ qw.T
            err = cand - ref
            nrmse = float(err.square().mean().sqrt() / ref.square().mean().sqrt())
            maxrel = float(err.abs().max() / ref.abs().max())
            assert torch.isfinite(cand).all() and nrmse <= NRMSE_LIMIT and maxrel <= MAX_LIMIT, (seed, role, nrmse, maxrel)
            print(f'COMP_FP8_CPU_SAMPLE seed={seed} role={role} nrmse={nrmse:.6f} max_abs_over_refmax={maxrel:.6f}')
    print(f'DS41_COMP_FP8_CPU_PASS samples=6 NRMSE_limit={NRMSE_LIMIT} max_ratio_limit={MAX_LIMIT} quality=UNMEASURED')
