#!/usr/bin/env python3
"""CPU proof of the private output layout and unchanged attention body."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE.parents[2] / "ds41_attn.cuh").read_text()
variant = (HERE / "kernel.cuh").read_text()
marker = "template <bool FL, bool CQ = false>\n__global__ void __launch_bounds__(AT_THREADS, 1) "
source = source[source.index(marker + "k_attn_cpa_t"):]
variant = variant[variant.index(marker + "k_attn_ostore_t"):]
variant = variant.replace("k_attn_ostore_t", "k_attn_cpa_t")
assert source[:source.index("        bf16* o0 = p.out")] == variant[:variant.index("        bf16* o0 = Qs")]
for line in (
    "*(__nv_bfloat162*)(o0 + d) = __floats2bfloat162_rn(acc[nt][0] / d0, acc[nt][1] / d0);",
    "*(__nv_bfloat162*)(o1 + d) = __floats2bfloat162_rn(acc[nt][2] / d1, acc[nt][3] / d1);",
):
    assert line in source and line in variant
print("PASS: attention body and output division/rounding unchanged")

writes = {}
for tid in range(256):
    lane, warp = tid % 32, tid // 32
    for nt in range(8):
        for row in (lane // 4, lane // 4 + 8):
            for j in (0, 1):
                col = warp * 64 + nt * 8 + 2 * (lane % 4) + j
                addr = row * 520 + col
                assert addr not in writes
                writes[addr] = (row, col)
reads = set()
for tid in range(256):
    for c in range(tid, 1024, 256):
        row, col = c // 64, (c % 64) * 8
        assert (row * 520 + col) * 2 % 16 == 0
        for j in range(8):
            addr = row * 520 + col + j
            assert writes[addr] == (row, col + j)
            assert addr not in reads
            reads.add(addr)
assert len(writes) == len(reads) == 8192
print("PASS: 8192 output elements covered once at identical coordinates; 16-byte vector alignment")
for warp in range(8):
    for nt in range(8):
        banks = [((lane // 4) * 520 + warp * 64 + nt * 8 + 2 * (lane % 4)) // 2 % 32
                 for lane in range(32)]
        assert len(set(banks)) == 32
print("PASS: each fragment-store instruction addresses 32 distinct shared banks per warp")
