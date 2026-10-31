#!/usr/bin/env python3
"""CPU/source gate for the default-off persistent wo_b and compressor FP8 arms; no CUDA execution."""
from pathlib import Path

D = Path(__file__).resolve().parents[1]
C = (D / 'ds41_attn.cuh').read_text()
O = (D / 'ds41_attn_ops.cuh').read_text()
T = (D / 'ds41_attn_types.h').read_text()
U = (D / 'ds41_attn.cu').read_text()


def coverage(mt, nt, grid):
    assert grid >= nt and mt > 0 and nt > 0
    got = []
    for block in range(grid):
        n = block % nt
        peer = block // nt
        peers = (grid + nt - 1 - n) // nt
        for m in range(peer, mt, peers):
            got.append((m, n))
    return got


def unique_coverage(mt, nt, grid):
    a = coverage(mt, nt, grid)
    return len(a) == mt * nt and len(set(a)) == mt * nt


def admit_smem(bm, bn, stages, static_bytes, cap=99 * 1024):
    dynamic = stages * (bm * 144 + bn * 144 + bm * 4 + bn * 4)
    total = dynamic + static_bytes
    if total > cap:
        raise ValueError(f'per-block shared memory {total} exceeds cap {cap}')
    return dynamic, total


def source_gate(c, o, t, u):
    assert 'static const bool on = knob::ds_wob_persist();' in c
    assert 'k_gemm_mx8t_c2<128, 128, 2>(p, 1, st); return;' not in c  # launcher handles the switch
    assert 'if (tc_ok(p)) { launch_tc_t_c2<128, 128, 2>(p, 1, st); return; }' in c
    assert 'const unsigned grid = sm < nt ? nt : (sm < mt * nt ? sm : mt * nt);' in c
    assert 'k_gemm_mx8t_c2_persist<128, 128, 2>' in c
    assert 'TcCfg<128, 128, 2>::SMEM' in c
    assert 'static_assert(C::SMEM + 1024 <= 99 * 1024' in c
    assert 'cudaFuncGetAttributes(&attr, k_gemm_mx8t_c2_persist<128, 128, 2>)' in c
    assert 'const int peers = ((int)gridDim.x + nt - 1 - n_tile) / nt;' in c
    assert '__syncthreads(); // all coalesced stores read sY' in c
    assert 'if (c2_trace_on()) ++c2_dispatch_count_ref();' in c
    assert 'ds41_attn_set_comp_fp8(knob::ds_comp_fp8());' in o
    assert 'prepack(W.c_wkv, W.c_wkv8)' in o and 'prepack(W.c_wgate, W.c_wgate8)' in o
    assert 'cudaStreamSynchronize((cudaStream_t)stream)' in o
    assert 'Fp8Mat c_wkv8, c_wgate8' in t
    assert 'launch_act_quant(src, ds41a::HID, codes' in u
    assert 'if (g_comp_fp8 && n >= 9)' in c and 'DS41 compressor FP8 prepack RED' in c
    assert 'launch_comp_fp8(S.yq, S.ys, W.c_wkv8, S.ckv, n, st)' in c
    assert 'launch_comp_fp8(S.yq, S.ys, W.c_wgate8, S.csc, n, st)' in c
    assert 'k_f32gemm2<<<dim3' in c and 'float* Y = Y32' in c
    assert 'Y[(size_t)row * p.ldy + col] = acc[mb][nt][0]' in c
    assert 'constexpr int BM = 128, BN = 64, ST = 3;' in c
    return True


if __name__ == '__main__':
    # M=4096, N=5120: 32x40 tiles; vary SM count and tails. Each tile must be issued exactly once.
    n = 0
    for mt in (1, 2, 31, 32, 33):
        for nt in (1, 8, 40, 41):
            for sm in (1, 16, 48, 132):
                grid = max(nt, min(sm, mt * nt))
                assert unique_coverage(mt, nt, grid), (mt, nt, grid)
                n += 1
    assert admit_smem(128, 128, 2, 1024) == (75776, 76800)
    for static_bytes in (0, 1024, 25600):
        assert admit_smem(128, 128, 2, static_bytes)[1] <= 99 * 1024
    over_limit_mutants = ((128, 128, 3, 1024), (128, 128, 2, 25601))
    for config in over_limit_mutants:
        try:
            admit_smem(*config)
        except ValueError:
            pass
        else:
            raise AssertionError(f'over-limit shared-memory config admitted: {config}')
    assert source_gate(C, O, T, U)
    # Each source gate has a deliberately faulty sample that must fail closed.
    mutants = [
        (C.replace('sm < nt ? nt :', 'sm < nt ? sm :', 1), O, T, U),
        (C.replace('k_gemm_mx8t_c2_persist<128, 128, 2>', 'k_gemm_mx8t_c2_persist<128, 128, 3>'), O, T, U),
        (C, O.replace('prepack(W.c_wgate, W.c_wgate8)', 'prepack(W.c_wkv, W.c_wgate8)'), T, U),
        (C.replace('float* Y = Y32', 'bf16* Y = p.Y', 1), O, T, U),
        (C.replace('if (c2_trace_on()) ++c2_dispatch_count_ref();', ''), O, T, U),
    ]
    for i, args in enumerate(mutants):
        try:
            source_gate(*args)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'mutant {i} passed')
    # A broken ownership map that launches fewer CTAs than N tiles fails, not a skipped tail.
    try:
        coverage(32, 40, 16)
    except AssertionError:
        pass
    else:
        raise AssertionError('grid-too-small mutant passed')
    print(f'DS41_DENSE_EARLY_CPU_PASS coverage_cases={n} dynamic=75776 static_budget=1024 total=76800 cc12x_limit=101376 over_limit_mutants={len(over_limit_mutants)}/RED source_gates=1 mutants={len(mutants)+1}/RED')
