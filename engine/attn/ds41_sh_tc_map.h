// Shared expert on tensor cores: the MXFP8 decode record's index map (k_kda_mx_record, engine/kda/kda.cuh = loader-rs records::mxfp8_record) as one host-and-device
// function, so k_mx_unrecord (ds41_sh_tc.cuh) and the host test engine/tests/test_ds41_sh_tc.cpp run the same lines.
// Record = 16 rows x 32 k = 528 B: byte h * 256 + i (h = 0, 1; i = 0..255) holds code (row, 16 h + col) of the record, with lane l = i / 8, e = i % 8,
// r = l / 4, c = l % 4, row = r + 8 (e & 2), col = 2 c + (e & 1) + 8 (e & 4); then 8 scale pairs: byte 512 + 2 j = row j, 513 + 2 j = row j + 8 (j = 0..7).
#pragma once
#if defined(__CUDACC__)
#define DS41_SH_HD __host__ __device__ __forceinline__
#else
#define DS41_SH_HD inline
#endif

namespace ds41a
{
constexpr int MX_REC_B = 528;   // bytes per record
// i = 0..255: the record byte h * 256 + i holds code (row, 16 h + col)
DS41_SH_HD void mx_rec_rc(int i, int& row, int& col)
{
    const int l = i >> 3, e = i & 7, r = l >> 2, c = l & 3;
    row = r + ((e & 2) ? 8 : 0);
    col = 2 * c + (e & 1) + ((e & 4) ? 8 : 0);
}
}   // namespace ds41a
