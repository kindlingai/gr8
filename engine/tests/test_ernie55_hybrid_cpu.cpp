// Host-only parity & DDR5 throughput test for engine/runtime/ernie55_hybrid_cpu.h
// Verifies:
//   1. 65,536 / 65,536 bit-exact match of MUL1_LUT and bsum_lut vs exact tr3 mul1 math.
//   2. Wide-order 16x128 tile GEMV parity for K=2, K=3, K=4 (both FC1 PLANE_STRIDE=2048 and FC2=256)
//      against an independent un-widened reference tile decoder, plus scalar vs AVX-512 VNNI_L1 & LUT_L2.
//   3. Full eval_cold_expert_cpu parity (< 1e-5 max abs diff) for K=2, K=3, K=4, and both kmix modes
//      (negative uniform gate/up kmix and positive per-unit kmix) vs independent un-widened reference.
//   4. Persistent CpuExpertPool multi-thread parity across n_exps in {1, 2, 4, 6} (block splitting).
//   5. Optional `--bench`: real out-of-L3 DDR5 throughput benchmark on 256 MiB ring of 85 K2 experts.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "runtime/ernie55_hybrid_cpu.h"

namespace {

using namespace ernie55cpu;

// Deterministic xorshift64* PRNG
struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1ull) {}
    uint64_t next_u64() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545f4914f6cdd1dull;
    }
    uint8_t next_u8() { return static_cast<uint8_t>(next_u64() >> 56); }
    float next_f32(float lo, float hi) {
        const double u = static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
        return static_cast<float>(lo + (hi - lo) * u);
    }
};

// Independent bit-exact mul1 reference (matching engine/tests/test_ernie55_math.cpp:522-537)
float ref_mul1_decode(uint32_t state16) {
    const uint32_t s = state16 & 0xffffu;
    const uint32_t prod = s * 0x83DCD12Du;
    const uint32_t bsum = (prod & 0xffu) + ((prod >> 8) & 0xffu) +
                          ((prod >> 16) & 0xffu) + ((prod >> 24) & 0xffu);
    const float x = f16_to_f32(static_cast<uint16_t>(0x6400u + bsum));
    const float c1 = f16_to_f32(0x1eee);
    const float c0 = f16_to_f32(0xc931);
    const float y = f16_to_f32(f32_to_f16_rn(std::fma(x, c1, c0)));
    return e4m3fn_to_f32(f32_to_e4m3fn_sat(y));
}

// Un-widened tile chunk extraction (exact mirror of ernie55::trw_tile_chunk in ernie55_moe.cuh:92)
template <int BITS>
uint32_t ref_trw_tile_chunk(const uint8_t* tile, int lane) {
    constexpr int CB = 8 * BITS, NW = 8 * BITS;
    const int bit = lane * CB, wi = bit >> 5, off = bit & 31;
    uint32_t t0, t1;
    std::memcpy(&t0, tile + wi * 4, 4);
    std::memcpy(&t1, tile + (wi + 1 < NW ? wi + 1 : NW - 1) * 4, 4);
    const uint64_t m = (static_cast<uint64_t>(t0) << 32) | t1;
    return static_cast<uint32_t>(m >> (64 - off - CB)) & static_cast<uint32_t>((1ull << CB) - 1);
}

template <int BITS>
void ref_trw_pack(uint64_t v[3], int p, uint32_t chunk) {
    constexpr int CB = 8 * BITS;
    const int bit = CB * p, n = bit >> 6, sh = bit & 63;
    v[n] |= static_cast<uint64_t>(chunk) << sh;
    if constexpr (CB > 16) {
        if (sh > 64 - CB) v[n + 1] |= static_cast<uint64_t>(chunk) >> (64 - sh);
    }
}

// Exact host wide-reorder of one FC1 unit [EHB, 2048 * BITS] (mirror of ernie55_hybrid.cuh:63)
void widen_fc1_mat_ref(uint8_t* mat, int bits) {
    const int GRP_B = 2048 * bits;
    std::vector<uint8_t> tmp(static_cast<size_t>(GRP_B));
    for (int hb = 0; hb < EHB; ++hb) {
        uint8_t* grp = mat + static_cast<size_t>(hb) * GRP_B;
        for (int j = 0; j < 8; ++j) {
            for (int lane = 0; lane < 32; ++lane) {
                if (bits == 2) {
                    uint64_t v[3] = {0, 0, 0};
                    for (int p = 0; p < 8; ++p)
                        ref_trw_pack<2>(v, p, ref_trw_tile_chunk<2>(grp + (p * 8 + j) * 64, lane));
                    for (int n = 0; n < 2; ++n) std::memcpy(tmp.data() + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
                } else if (bits == 3) {
                    uint64_t v[3] = {0, 0, 0};
                    for (int p = 0; p < 8; ++p)
                        ref_trw_pack<3>(v, p, ref_trw_tile_chunk<3>(grp + (p * 8 + j) * 96, lane));
                    for (int n = 0; n < 3; ++n) std::memcpy(tmp.data() + (n * 8 + j) * 256 + lane * 8, &v[n], 8);
                } else {
                    uint32_t w[8];
                    for (int p = 0; p < 8; ++p) std::memcpy(&w[p], grp + (p * 8 + j) * 128 + lane * 4, 4);
                    std::memcpy(tmp.data() + j * 1024 + lane * 16,       &w[0], 16);
                    std::memcpy(tmp.data() + j * 1024 + 512 + lane * 16, &w[4], 16);
                }
            }
        }
        std::memcpy(grp, tmp.data(), static_cast<size_t>(GRP_B));
    }
}

// Exact host wide-reorder of one FC2 block [EHB * 8, 256 * BITS] (mirror of ernie55_hybrid.cuh:116)
void widen_fc2_blk_ref(uint8_t* blk, int bits) {
    constexpr int N_TILES = EHB * 8;
    const int NT_B = 256 * bits;
    std::vector<uint8_t> tmp(static_cast<size_t>(NT_B));
    for (int nt = 0; nt < N_TILES; ++nt) {
        uint8_t* dst = blk + static_cast<size_t>(nt) * NT_B;
        for (int lane = 0; lane < 32; ++lane) {
            if (bits == 2) {
                uint64_t v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p) ref_trw_pack<2>(v, p, ref_trw_tile_chunk<2>(dst + p * 64, lane));
                for (int n = 0; n < 2; ++n) std::memcpy(tmp.data() + n * 256 + lane * 8, &v[n], 8);
            } else if (bits == 3) {
                uint64_t v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p) ref_trw_pack<3>(v, p, ref_trw_tile_chunk<3>(dst + p * 96, lane));
                for (int n = 0; n < 3; ++n) std::memcpy(tmp.data() + n * 256 + lane * 8, &v[n], 8);
            } else {
                uint32_t w[8];
                for (int p = 0; p < 8; ++p) std::memcpy(&w[p], dst + p * 128 + lane * 4, 4);
                std::memcpy(tmp.data() + lane * 16,       &w[0], 16);
                std::memcpy(tmp.data() + 512 + lane * 16, &w[4], 16);
            }
        }
        std::memcpy(dst, tmp.data(), static_cast<size_t>(NT_B));
    }
}

// Independent un-widened reference 16x128 tile matvec (reads ORIGINAL pre-widen tile bytes!)
template <int BITS>
void ref_unwidened_tile16x128(const uint8_t* orig_grp_or_nt, bool is_fc1, int warp,
                              const float x[128], float out[16]) {
    for (int r = 0; r < 16; ++r) out[r] = 0.0f;
    const int tile_bytes = 32 * BITS;
    constexpr int B0 = 7 * BITS;
    for (int p = 0; p < 8; ++p) {
        const uint8_t* tile_p = is_fc1
            ? (orig_grp_or_nt + (p * 8 + warp) * tile_bytes)
            : (orig_grp_or_nt + p * tile_bytes);
        uint32_t ch[32];
        for (int lane = 0; lane < 32; ++lane) {
            if constexpr (BITS == 4) {
                std::memcpy(&ch[lane], tile_p + lane * 4, 4);
            } else {
                ch[lane] = ref_trw_tile_chunk<BITS>(tile_p, lane);
            }
        }
        for (int lane = 0; lane < 32; ++lane) {
            const int g = lane >> 2;
            const int col = lane & 3;
            const int prev = (lane - 1) & 31;
            const uint64_t m = (static_cast<uint64_t>(ch[prev]) << (8 * BITS)) | ch[lane];
            const float x0 = x[p * 16 + 2 * col + 0];
            const float x1 = x[p * 16 + 2 * col + 1];
            const float x2 = x[p * 16 + 2 * col + 8];
            const float x3 = x[p * 16 + 2 * col + 9];
            double acc_lo = 0.0, acc_hi = 0.0;
            acc_lo += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 0 * BITS)))) * x0;
            acc_lo += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 1 * BITS)))) * x1;
            acc_lo += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 2 * BITS)))) * x2;
            acc_lo += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 3 * BITS)))) * x3;
            acc_hi += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 4 * BITS)))) * x0;
            acc_hi += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 5 * BITS)))) * x1;
            acc_hi += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 6 * BITS)))) * x2;
            acc_hi += double(ref_mul1_decode(static_cast<uint32_t>(m >> (B0 - 7 * BITS)))) * x3;
            out[g]     += static_cast<float>(acc_lo);
            out[g + 8] += static_cast<float>(acc_hi);
        }
    }
}

// Independent un-widened reference full expert forward (matches GPU k_ernie55_routed_fc1/fc2_grp)
void ref_eval_expert_unwidened(
    const uint8_t* orig_fc1, const uint8_t* orig_fc2,
    const size_t piece_fc1_off[EB_MAX], const size_t piece_fc2_off[EB_MAX],
    const uint16_t* v1b, const uint16_t* v2, const uint16_t* v3b, const uint16_t* sg_out,
    const uint16_t* x_had_r, size_t brow0, long long km, int nb, int dk, float topk_w,
    float out_2048[EH]) {
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(EHB) * 2048;
    float fc2_acc[EH] = {};

    for (int b = 0; b < nb; ++b) {
        const size_t brow = brow0 + static_cast<size_t>(b);
        const int kg = fc1_unit_width(km, dk, 2 * b + 0);
        const int ku = fc1_unit_width(km, dk, 2 * b + 1);
        const uint8_t* mw_g = orig_fc1 + piece_fc1_off[b];
        const uint8_t* mw_u = mw_g + static_cast<size_t>(kg) * UNIT_PER_BIT;

        float g_out[128] = {}, u_out[128] = {};
        for (int hb = 0; hb < EHB; ++hb) {
            const float bsg = f16_to_f32(v1b[(brow * 2 + 0) * EHB + hb]);
            const float bsu = f16_to_f32(v1b[(brow * 2 + 1) * EHB + hb]);
            float xg[128], xu[128];
            for (int c = 0; c < 128; ++c) {
                const float xh = f16_to_f32(x_had_r[hb * 128 + c]);
                // Note: GPU multiplies x_had * v1b in FP16; here we also test linear FP32 match.
                xg[c] = xh * bsg;
                xu[c] = xh * bsu;
            }
            const uint8_t* grpg = mw_g + static_cast<size_t>(hb) * (2048 * kg);
            const uint8_t* grpu = mw_u + static_cast<size_t>(hb) * (2048 * ku);
            for (int warp = 0; warp < 8; ++warp) {
                float tg[16], tu[16];
                if (kg == 2)      ref_unwidened_tile16x128<2>(grpg, true, warp, xg, tg);
                else if (kg == 3) ref_unwidened_tile16x128<3>(grpg, true, warp, xg, tg);
                else              ref_unwidened_tile16x128<4>(grpg, true, warp, xg, tg);
                if (ku == 2)      ref_unwidened_tile16x128<2>(grpu, true, warp, xu, tu);
                else if (ku == 3) ref_unwidened_tile16x128<3>(grpu, true, warp, xu, tu);
                else              ref_unwidened_tile16x128<4>(grpu, true, warp, xu, tu);
                for (int r = 0; r < 16; ++r) {
                    g_out[warp * 16 + r] += tg[r];
                    u_out[warp * 16 + r] += tu[r];
                }
            }
        }
        had128_cpu(g_out);
        had128_cpu(u_out);
        const uint16_t* sv_g = v2 + (brow * 3 + 0) * 128;
        const uint16_t* sv_u = v2 + (brow * 3 + 1) * 128;
        const uint16_t* su_d = v2 + (brow * 3 + 2) * 128;
        float va[128];
        for (int c = 0; c < 128; ++c) {
            const float gf = bf16_round_f32(g_out[c] * f16_to_f32(sv_g[c]));
            const float uf = bf16_round_f32(u_out[c] * f16_to_f32(sv_u[c]));
            const float act = bf16_round_f32(situ_mul_cpu(gf, uf));
            va[c] = act * f16_to_f32(su_d[c]);
        }
        had128_cpu(va);
        for (int c = 0; c < 128; ++c) va[c] = f16_round_f32(va[c]);

        const uint8_t* mw_fc2 = orig_fc2 + piece_fc2_off[b];
        const size_t nt_stride = static_cast<size_t>(256 * dk);
        for (int hb = 0; hb < EHB; ++hb) {
            const float sc = f16_to_f32(v3b[brow * EHB + hb]);
            for (int warp = 0; warp < 8; ++warp) {
                const int n_tile = hb * 8 + warp;
                const uint8_t* nt = mw_fc2 + static_cast<size_t>(n_tile) * nt_stride;
                float c16[16];
                if (dk == 2)      ref_unwidened_tile16x128<2>(nt, false, 0, va, c16);
                else if (dk == 3) ref_unwidened_tile16x128<3>(nt, false, 0, va, c16);
                else              ref_unwidened_tile16x128<4>(nt, false, 0, va, c16);
                for (int r = 0; r < 16; ++r) fc2_acc[hb * 128 + warp * 16 + r] += c16[r] * sc;
            }
        }
    }

    for (int hb = 0; hb < EHB; ++hb) {
        float* v = fc2_acc + hb * 128;
        had128_cpu(v);
        for (int c = 0; c < 128; ++c) {
            out_2048[hb * 128 + c] = v[c] * f16_to_f32(sg_out[hb * 128 + c]) * topk_w;
        }
    }
}

void test_mul1_lut_exact() {
    const Mul1Tables& tbl = mul1_tables();
    int exact = 0;
    for (uint32_t s = 0; s < 65536; ++s) {
        const float want = ref_mul1_decode(s);
        uint32_t u_got, u_want;
        std::memcpy(&u_got, &tbl.lut[s], 4);
        std::memcpy(&u_want, &want, 4);
        assert(u_got == u_want);
        const uint32_t prod = s * 0x83DCD12Du;
        const uint32_t bsum = (prod & 0xffu) + ((prod >> 8) & 0xffu) +
                              ((prod >> 16) & 0xffu) + ((prod >> 24) & 0xffu);
        uint32_t u_bsum;
        std::memcpy(&u_bsum, &tbl.bsum_lut[bsum], 4);
        assert(u_bsum == u_want);
        ++exact;
    }
    std::printf("test_mul1_lut_exact: %d/65536 BIT-EXACT (lut + bsum_lut)\n", exact);
}

void test_tile_gemv_parity() {
    Rng rng(12345);
    float x[128];
    for (int i = 0; i < 128; ++i) x[i] = f16_round_f32(rng.next_f32(-0.5f, 0.5f));
    PackedX128 px;
    pack_x128(x, px);

    for (int bits : {2, 3, 4}) {
        // Test FC1 layout (PLANE_STRIDE = 2048)
        std::vector<uint8_t> fc1_orig(static_cast<size_t>(EHB) * 2048 * bits);
        for (auto& b : fc1_orig) b = rng.next_u8();
        std::vector<uint8_t> fc1_wide = fc1_orig;
        widen_fc1_mat_ref(fc1_wide.data(), bits);

        float max_err_sc = 0.0f, max_err_v1 = 0.0f, max_err_v2 = 0.0f;
        for (int warp = 0; warp < 8; ++warp) {
            float ref[16], got_sc[16];
            if (bits == 2)      ref_unwidened_tile16x128<2>(fc1_orig.data(), true, warp, x, ref);
            else if (bits == 3) ref_unwidened_tile16x128<3>(fc1_orig.data(), true, warp, x, ref);
            else                ref_unwidened_tile16x128<4>(fc1_orig.data(), true, warp, x, ref);

            const uint8_t* tile = fc1_wide.data() + static_cast<size_t>(warp) * ((bits == 4) ? 1024u : 256u);
            if (bits == 2)      mul1_tile16x128_dot_scalar<2, 2048>(tile, px, got_sc);
            else if (bits == 3) mul1_tile16x128_dot_scalar<3, 2048>(tile, px, got_sc);
            else                mul1_tile16x128_dot_scalar<4, 2048>(tile, px, got_sc);

            for (int r = 0; r < 16; ++r) max_err_sc = std::max(max_err_sc, std::fabs(got_sc[r] - ref[r]));

#if ERNIE55_CPU_CAN_AVX512
            if (cpu_has_avx512_vnni()) {
                float got_v0[16], got_v1[16], got_v2[16];
                if (bits == 2) {
                    mul1_tile16x128_dot_k2_avx512<2048, Avx512DecodeMode::VNNI_REG>(tile, px, got_v0);
                    mul1_tile16x128_dot_k2_avx512<2048, Avx512DecodeMode::VNNI_L1>(tile, px, got_v1);
                    mul1_tile16x128_dot_k2_avx512<2048, Avx512DecodeMode::LUT_L2>(tile, px, got_v2);
                } else if (bits == 3) {
                    mul1_tile16x128_dot_k34_avx512<3, 2048, Avx512DecodeMode::VNNI_REG>(tile, px, got_v0);
                    mul1_tile16x128_dot_k34_avx512<3, 2048, Avx512DecodeMode::VNNI_L1>(tile, px, got_v1);
                    mul1_tile16x128_dot_k34_avx512<3, 2048, Avx512DecodeMode::LUT_L2>(tile, px, got_v2);
                } else {
                    mul1_tile16x128_dot_k34_avx512<4, 2048, Avx512DecodeMode::VNNI_REG>(tile, px, got_v0);
                    mul1_tile16x128_dot_k34_avx512<4, 2048, Avx512DecodeMode::VNNI_L1>(tile, px, got_v1);
                    mul1_tile16x128_dot_k34_avx512<4, 2048, Avx512DecodeMode::LUT_L2>(tile, px, got_v2);
                }
                for (int r = 0; r < 16; ++r) {
                    max_err_v1 = std::max(max_err_v1, std::fabs(got_v0[r] - ref[r]));
                    max_err_v2 = std::max(max_err_v2, std::fabs(got_v2[r] - ref[r]));
                    assert(got_v0[r] == got_v1[r]); // 100% BIT-IDENTICAL: register-only == L1 LUT!
                    assert(got_v1[r] == got_v2[r]); // 100% BIT-IDENTICAL: L1 LUT == L2 LUT!
                }
            }
#endif
        }
        assert(max_err_sc < 1e-5f);
        assert(max_err_v1 < 1e-5f);
        assert(max_err_v2 < 1e-5f);

        // Test FC2 layout (PLANE_STRIDE = 256)
        std::vector<uint8_t> fc2_orig(static_cast<size_t>(EHB) * 8 * 256 * bits);
        for (auto& b : fc2_orig) b = rng.next_u8();
        std::vector<uint8_t> fc2_wide = fc2_orig;
        widen_fc2_blk_ref(fc2_wide.data(), bits);
        float max_fc2 = 0.0f;
        for (int nt = 0; nt < 8; ++nt) {
            float ref[16], got[16];
            const size_t off = static_cast<size_t>(nt) * 256 * bits;
            if (bits == 2)      ref_unwidened_tile16x128<2>(fc2_orig.data() + off, false, 0, x, ref);
            else if (bits == 3) ref_unwidened_tile16x128<3>(fc2_orig.data() + off, false, 0, x, ref);
            else                ref_unwidened_tile16x128<4>(fc2_orig.data() + off, false, 0, x, ref);

            if (bits == 2)      mul1_tile16x128_dot_cpu<2, 256>(fc2_wide.data() + off, px, got);
            else if (bits == 3) mul1_tile16x128_dot_cpu<3, 256>(fc2_wide.data() + off, px, got);
            else                mul1_tile16x128_dot_cpu<4, 256>(fc2_wide.data() + off, px, got);

            for (int r = 0; r < 16; ++r) max_fc2 = std::max(max_fc2, std::fabs(got[r] - ref[r]));
        }
        assert(max_fc2 < 1e-5f);
        std::printf("test_tile_gemv_parity K=%d: FC1(sc=%.2e, vnni=%.2e, lut=%.2e) FC2(%.2e) OK\n",
                    bits, max_err_sc, max_err_v1, max_err_v2, max_fc2);
    }
}

struct SyntheticExpert {
    std::vector<uint8_t> orig_fc1, orig_fc2;
    std::vector<uint8_t> wide_fc1, wide_fc2;
    size_t piece_fc1_off[EB_MAX] = {};
    size_t piece_fc2_off[EB_MAX] = {};
    std::vector<uint16_t> v1b, v2, v3b, sg_out;
    long long km = 0;
    int nb = 0, dk = 2;
    float topk_w = 0.25f;
};

SyntheticExpert make_synthetic_expert(Rng& rng, int nb, int dk, long long km) {
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(EHB) * 2048;
    SyntheticExpert e;
    e.nb = nb; e.dk = dk; e.km = km;
    e.topk_w = rng.next_f32(0.05f, 0.35f);
    size_t f1_cur = 0, f2_cur = 0;
    for (int b = 0; b < nb; ++b) {
        e.piece_fc1_off[b] = f1_cur;
        e.piece_fc2_off[b] = f2_cur;
        const int kg = fc1_unit_width(km, dk, 2 * b + 0);
        const int ku = fc1_unit_width(km, dk, 2 * b + 1);
        f1_cur += static_cast<size_t>(kg + ku) * UNIT_PER_BIT;
        f2_cur += static_cast<size_t>(dk) * UNIT_PER_BIT;
    }
    e.orig_fc1.resize(f1_cur);
    e.orig_fc2.resize(f2_cur);
    for (auto& x : e.orig_fc1) x = rng.next_u8();
    for (auto& x : e.orig_fc2) x = rng.next_u8();
    e.wide_fc1 = e.orig_fc1;
    e.wide_fc2 = e.orig_fc2;
    for (int b = 0; b < nb; ++b) {
        const int kg = fc1_unit_width(km, dk, 2 * b + 0);
        const int ku = fc1_unit_width(km, dk, 2 * b + 1);
        uint8_t* p = e.wide_fc1.data() + e.piece_fc1_off[b];
        widen_fc1_mat_ref(p, kg);
        widen_fc1_mat_ref(p + static_cast<size_t>(kg) * UNIT_PER_BIT, ku);
        widen_fc2_blk_ref(e.wide_fc2.data() + e.piece_fc2_off[b], dk);
    }
    e.v1b.resize(static_cast<size_t>(nb) * 2 * EHB);
    e.v2.resize(static_cast<size_t>(nb) * 3 * 128);
    e.v3b.resize(static_cast<size_t>(nb) * EHB);
    e.sg_out.resize(EH);
    for (auto& h : e.v1b)    h = f32_to_f16_rn(rng.next_f32(0.02f, 0.08f));
    for (auto& h : e.v2)     h = f32_to_f16_rn(rng.next_f32(0.05f, 0.20f));
    for (auto& h : e.v3b)    h = f32_to_f16_rn(rng.next_f32(0.02f, 0.08f));
    for (auto& h : e.sg_out) h = f32_to_f16_rn(rng.next_f32(0.5f, 1.5f));
    return e;
}

void test_full_expert_and_pool_parity() {
    Rng rng(67890);
    std::vector<uint16_t> x_had(EH);
    for (auto& h : x_had) h = f32_to_f16_rn(rng.next_f32(-0.5f, 0.5f));

    struct Case { const char* name; int nb; int dk; long long km; };
    const Case cases[] = {
        {"K2_uniform",   4, 2, 0LL},
        {"K3_uniform",   4, 3, 0LL},
        {"K4_uniform",   4, 4, 0LL},
        {"kmix_neg_3_2", 4, 2, static_cast<long long>((1ull << 63) | (2ull << 4) | 3ull)}, // gk=3, uk=2, dk=2
        {"kmix_pos_mix", 4, 3, static_cast<long long>((2ull << 0) | (3ull << 4) | (4ull << 8) | (2ull << 12) |
                                                      (3ull << 16) | (4ull << 20) | (2ull << 24) | (3ull << 28))},
    };

    for (const auto& cs : cases) {
        auto se = make_synthetic_expert(rng, cs.nb, cs.dk, cs.km);
        float ref[EH], got[EH];
        ref_eval_expert_unwidened(se.orig_fc1.data(), se.orig_fc2.data(),
                                  se.piece_fc1_off, se.piece_fc2_off,
                                  se.v1b.data(), se.v2.data(), se.v3b.data(), se.sg_out.data(),
                                  x_had.data(), 0, se.km, se.nb, se.dk, se.topk_w, ref);
        CpuExpertDesc d{};
        d.exp_fc1 = se.wide_fc1.data();
        d.exp_fc2 = se.wide_fc2.data();
        d.piece_fc1_off = se.piece_fc1_off;
        d.piece_fc2_off = se.piece_fc2_off;
        d.v1b = se.v1b.data();
        d.v2  = se.v2.data();
        d.v3b = se.v3b.data();
        d.sg_out = se.sg_out.data();
        d.brow0 = 0; d.kmix = se.km; d.nb = se.nb; d.dk = se.dk; d.topk_w = se.topk_w;
        d.out_2048 = got;
        eval_cold_expert_cpu(d, x_had.data());

        float max_diff = 0.0f, max_mag = 0.0f;
        for (int i = 0; i < EH; ++i) {
            max_diff = std::max(max_diff, std::fabs(got[i] - ref[i]));
            max_mag  = std::max(max_mag,  std::fabs(ref[i]));
        }
        assert(max_mag > 1e-4f);
        assert(max_diff < 1e-5f);
        std::printf("test_full_expert [%s]: max_abs_diff=%.2e (max_mag=%.3e) OK\n",
                    cs.name, max_diff, max_mag);
    }

    // Multi-thread CpuExpertPool parity across n_exps = {1, 2, 4, 6} (exercises 8/4/2/1 splits!)
    CpuExpertPool pool;
    pool.start(8);
    std::vector<SyntheticExpert> exps;
    for (int i = 0; i < 6; ++i) exps.push_back(make_synthetic_expert(rng, 16, 2, 0LL));
    std::vector<std::vector<float>> ref_outs(6, std::vector<float>(EH));
    for (int i = 0; i < 6; ++i) {
        CpuExpertDesc d{};
        d.exp_fc1 = exps[i].wide_fc1.data();
        d.exp_fc2 = exps[i].wide_fc2.data();
        d.piece_fc1_off = exps[i].piece_fc1_off;
        d.piece_fc2_off = exps[i].piece_fc2_off;
        d.v1b = exps[i].v1b.data(); d.v2 = exps[i].v2.data();
        d.v3b = exps[i].v3b.data(); d.sg_out = exps[i].sg_out.data();
        d.brow0 = 0; d.kmix = 0; d.nb = 16; d.dk = 2; d.topk_w = exps[i].topk_w;
        d.out_2048 = ref_outs[i].data();
        eval_cold_expert_cpu(d, x_had.data());
    }
    for (int n : {1, 2, 4, 6}) {
        std::vector<std::vector<float>> got_outs(static_cast<size_t>(n), std::vector<float>(EH));
        std::vector<CpuExpertDesc> descs(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            auto& d = descs[static_cast<size_t>(i)];
            d.exp_fc1 = exps[i].wide_fc1.data();
            d.exp_fc2 = exps[i].wide_fc2.data();
            d.piece_fc1_off = exps[i].piece_fc1_off;
            d.piece_fc2_off = exps[i].piece_fc2_off;
            d.v1b = exps[i].v1b.data(); d.v2 = exps[i].v2.data();
            d.v3b = exps[i].v3b.data(); d.sg_out = exps[i].sg_out.data();
            d.brow0 = 0; d.kmix = 0; d.nb = 16; d.dk = 2; d.topk_w = exps[i].topk_w;
            d.out_2048 = got_outs[static_cast<size_t>(i)].data();
        }
        pool.run_batch(descs.data(), n, x_had.data());
        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < EH; ++c)
                max_diff = std::max(max_diff, std::fabs(got_outs[i][c] - ref_outs[i][c]));
        }
        assert(max_diff < 1e-6f);
        std::printf("test_pool_parity n=%d (8 threads): max_abs_diff=%.2e OK\n", n, max_diff);
    }
    pool.shutdown();
}

// Optional real out-of-L3 DDR5 throughput benchmark on a 256 MiB ring of 85 distinct 3.0 MiB K2 experts!
void run_ddr5_bench() {
    constexpr int N_RING = 85; // 85 * 3.0 MiB = 255 MiB >> 96 MiB L3 on Ryzen 7 9800X3D!
    constexpr size_t UNIT_PER_BIT = static_cast<size_t>(EHB) * 2048; // 32 KiB
    constexpr size_t FC1_BYTES = 16ull * 4 * UNIT_PER_BIT; // 2.0 MiB
    constexpr size_t FC2_BYTES = 16ull * 2 * UNIT_PER_BIT; // 1.0 MiB
    constexpr size_t EXP_BYTES = FC1_BYTES + FC2_BYTES;    // 3.0 MiB

    std::printf("\n=== Real Out-of-L3 DDR5 Benchmark (%d distinct 3.0 MiB K2 experts = %.1f MiB ring) ===\n",
                N_RING, (N_RING * EXP_BYTES) / double(1ull << 20));
    std::printf("AVX-512 VNNI active: %s\n",
#if ERNIE55_CPU_CAN_AVX512
                cpu_has_avx512_vnni() ? "YES" : "NO"
#else
                "NO (non-x86_64)"
#endif
    );

    // Allocate one contiguous 255 MiB arena and fill deterministically so OS pages are physically backed
    std::vector<uint8_t> arena(N_RING * EXP_BYTES);
    uint64_t* a64 = reinterpret_cast<uint64_t*>(arena.data());
    const size_t n64 = arena.size() / 8;
    for (size_t i = 0; i < n64; ++i) a64[i] = (i * 0x9e3779b97f4a7c15ull) ^ (i >> 17);

    size_t p1[EB_MAX], p2[EB_MAX];
    for (int b = 0; b < 16; ++b) {
        p1[b] = static_cast<size_t>(b) * 4 * UNIT_PER_BIT;
        p2[b] = static_cast<size_t>(b) * 2 * UNIT_PER_BIT;
    }

    Rng rng(99999);
    std::vector<uint16_t> v1b(16 * 2 * EHB), v2(16 * 3 * 128), v3b(16 * EHB), sg(EH), x_had(EH);
    for (auto& h : v1b)   h = f32_to_f16_rn(rng.next_f32(0.02f, 0.08f));
    for (auto& h : v2)    h = f32_to_f16_rn(rng.next_f32(0.05f, 0.20f));
    for (auto& h : v3b)   h = f32_to_f16_rn(rng.next_f32(0.02f, 0.08f));
    for (auto& h : sg)    h = f32_to_f16_rn(rng.next_f32(0.5f, 1.5f));
    for (auto& h : x_had) h = f32_to_f16_rn(rng.next_f32(-0.5f, 0.5f));

    std::vector<float> outs(10 * EH);

    // 1. Compare single-tile micro-throughput across decode implementations (1 thread, out-of-L3)
    {
        PackedX128 px;
        float xf[128];
        for (int c = 0; c < 128; ++c) xf[c] = f16_to_f32(x_had[c]);
        pack_x128(xf, px);
        constexpr int N_TILES = (N_RING * FC2_BYTES) / 512; // ~174,080 tiles = 85 MiB -> pure sequential stream
        float t16[16];
        volatile float sink = 0.0f;

        auto bench_tile = [&](const char* label, auto fn) {
            auto t0 = std::chrono::steady_clock::now();
            float acc = 0.0f;
            for (int i = 0; i < N_TILES; ++i) {
                const size_t e = static_cast<size_t>(i) / 2048;
                const size_t ti = static_cast<size_t>(i) % 2048;
                const uint8_t* ptr = arena.data() + e * EXP_BYTES + FC1_BYTES + ti * 512;
                fn(ptr, px, t16);
                acc += t16[0];
            }
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            sink += acc;
            double gbs = (static_cast<double>(N_TILES) * 512.0 / 1e9) / s;
            std::printf("  1-thread tile decode [%s]: %.2f GB/s (%.1f us / 3 MiB expert)\n",
                        label, gbs, (EXP_BYTES / 1e9) / gbs * 1e6);
        };
        bench_tile("scalar LUT256K", mul1_tile16x128_dot_scalar<2, 256>);
#if ERNIE55_CPU_CAN_AVX512
        if (cpu_has_avx512_vnni()) {
            bench_tile("AVX512 VNNI_REG (0-gather!)", mul1_tile16x128_dot_k2_avx512<256, Avx512DecodeMode::VNNI_REG>);
            bench_tile("AVX512 VNNI+L1_4K",           mul1_tile16x128_dot_k2_avx512<256, Avx512DecodeMode::VNNI_L1>);
            bench_tile("AVX512 LUT256K_L2",           mul1_tile16x128_dot_k2_avx512<256, Avx512DecodeMode::LUT_L2>);
        }
#endif
        (void)sink;
    }

    // 2. Full end-to-end multi-thread CpuExpertPool sweep over thread counts & batch sizes (out-of-L3 ring!)
    for (int nt : {4, 8, 12, 16}) {
        CpuExpertPool pool;
        pool.start(nt);
        for (int batch : {1, 2, 4, 6}) {
            const int iters = 84; // 2 full 42-layer token steps!
            int cursor = 0;
            // Warm up pool threads
            {
                std::vector<CpuExpertDesc> wd(static_cast<size_t>(batch));
                for (int i = 0; i < batch; ++i) {
                    const size_t ei = static_cast<size_t>(i % N_RING);
                    wd[i] = {arena.data() + ei * EXP_BYTES, arena.data() + ei * EXP_BYTES + FC1_BYTES,
                             p1, p2, v1b.data(), v2.data(), v3b.data(), sg.data(),
                             0, 0LL, 16, 2, 0.1f, outs.data() + i * EH};
                }
                pool.run_batch(wd.data(), batch, x_had.data());
            }
            auto t0 = std::chrono::steady_clock::now();
            double checksum = 0.0;
            std::vector<CpuExpertDesc> descs(static_cast<size_t>(batch));
            for (int it = 0; it < iters; ++it) {
                for (int i = 0; i < batch; ++i) {
                    const size_t ei = static_cast<size_t>(cursor % N_RING);
                    ++cursor;
                    descs[i] = {arena.data() + ei * EXP_BYTES, arena.data() + ei * EXP_BYTES + FC1_BYTES,
                                p1, p2, v1b.data(), v2.data(), v3b.data(), sg.data(),
                                0, 0LL, 16, 2, 0.1f, outs.data() + i * EH};
                }
                pool.run_batch(descs.data(), batch, x_had.data());
                checksum += outs[0];
            }
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double total_exps = static_cast<double>(iters) * batch;
            const double us_per_batch = (sec / iters) * 1e6;
            const double us_per_exp = (sec / total_exps) * 1e6;
            const double gbs = (total_exps * EXP_BYTES / 1e9) / sec;
            std::printf("  pool nt=%2d batch=%d: %6.1f us/batch | %5.1f us/exp | %5.2f GB/s DDR5 (chk=%.2e)\n",
                        nt, batch, us_per_batch, us_per_exp, gbs, checksum);
        }
        pool.shutdown();
    }
}

} // namespace

int main(int argc, char** argv) {
    test_mul1_lut_exact();
    test_tile_gemv_parity();
    test_full_expert_and_pool_parity();
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") {
            run_ddr5_bench();
        }
    }
    std::printf("ALL test_ernie55_hybrid_cpu PASSED\n");
    return 0;
}
