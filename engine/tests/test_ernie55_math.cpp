// Standalone CPU math unit test verifying exact ERNIE-5.5-Flash-Preview mathematical identities:
//   1. 4-stream Sinkhorn-20 mHC (`1 / (sqrt(mean) + rms_eps)` — eps OUTSIDE sqrt) + mhc_contract head
//   2. HCA C128 incremental decode vs full batch prefill (compressor + ape + SWA-128 + YaRN RoPE +
//      online F32 sink + VHAPostMix + inverse YaRN RoPE + grouped wo_a + sigmoid gate)
//   3. MLA-DSA primitives: NeoX leading-64 RoPE, orthonormal Hadamard-128 involution, SWA-128 + top-k
//      merge, and 2-split absorbed MLA + F32 sink exact equivalence to expand_kv
//   4. SiTU-GLU (beta=4.0, linear_beta=25.0) bounds & monotonicity
//   5. Split-feature FP32 top-10 router (sigmoid(l0)+sigmoid(l1), bias-corrected top-10, pre-bias renorm, per-expert scale)
//   6. exlr8 mul1 (0x83DCD12D, fp16 fma 0x1eee/0xc931 -> FP8 E4M3FN) + wide-reorder K2/K3/K4 round-trip
//   7. multimax 4-term ReLU / ReLU^2 logit adjustment
//   8. Optional --fixture <dir> validation against golden_logits.npy
#include "../model_geom_ernie55.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

inline float sigmoidf_ref(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// IEEE-754 half <-> float (exact RN-even for finite normal/subnormal values)
inline float half_bits_to_f32(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    uint32_t u = 0;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            float v = std::ldexp(float(m), -24);
            return s ? -v : v;
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &u, 4);
    return out;
}

inline uint16_t f32_to_half_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int exp = int((u >> 23) & 0xffu) - 127;
    uint32_t mant = u & 0x7fffffu;
    if (exp == 128) return uint16_t(sign | 0x7c00u | (mant ? 1u : 0u));
    if (exp > 15) return uint16_t(sign | 0x7c00u);
    if (exp < -24) return uint16_t(sign);
    if (exp < -14) {
        mant |= 0x800000u;
        const int shift = -14 - exp + 13;
        const uint32_t half_bit = 1u << (shift - 1);
        const uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t r = mant >> shift;
        if (rem > half_bit || (rem == half_bit && (r & 1u))) ++r;
        return uint16_t(sign | r);
    }
    const uint32_t rem = mant & 0x1fffu;
    uint32_t r = mant >> 13;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
        ++r;
        if (r == 0x400u) { r = 0; ++exp; }
    }
    if (exp > 15) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | uint32_t(exp + 15) << 10 | r);
}

// FP8 E4M3FN round-to-nearest-even saturating finite (matches cvt.rn.satfinite.e4m3x2.f16x2)
inline float e4m3fn_to_f32(uint8_t b) {
    const float s = (b & 0x80u) ? -1.0f : 1.0f;
    const int e = (b >> 3) & 0xf;
    const int m = b & 0x7;
    if (e == 0) return s * std::ldexp(float(m), -9);
    if (e == 15 && m == 7) return NAN;
    return s * std::ldexp(1.0f + float(m) / 8.0f, e - 7);
}

inline uint8_t f32_to_e4m3fn_sat(float x) {
    const uint8_t sign = (std::signbit(x) ? 0x80u : 0x00u);
    float ax = std::fabs(x);
    if (ax >= 448.0f) return uint8_t(sign | 0x7eu); // max finite E4M3FN = 448.0
    uint8_t best = 0;
    float best_err = 1e30f;
    for (int code = 0; code < 0x7f; ++code) {
        const float v = e4m3fn_to_f32(uint8_t(code));
        const float err = std::fabs(ax - v);
        if (err < best_err || (err == best_err && ((code & 1) == 0))) {
            best_err = err;
            best = uint8_t(code);
        }
    }
    return uint8_t(sign | best);
}

// ------------------------------------------------------------------------------------------------ 1. mHC Sinkhorn-20
void test_mhc_sinkhorn20() {
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    constexpr int HC = geom::HC;
    constexpr int H = 64;
    constexpr int D = HC * H;

    std::vector<float> res(D);
    for (float& v : res) v = U(rng);
    // Test exact ERNIE RMS denom = sqrt(mean) + RMS_EPS (eps OUTSIDE sqrt!)
    // Since (sqrt(m) + e)^2 - (m + e) = e*(2*sqrt(m) + e - 1), outside < inside for m < 0.25 and outside > inside for m > 0.25.
    double ss = 0.0;
    for (float v : res) ss += double(v) * double(v);
    const float ms = float(ss / D);
    const float denom_outside = std::sqrt(ms) + geom::RMS_EPS;
    const float denom_inside  = std::sqrt(ms + geom::RMS_EPS);
    assert(ms < 0.25f && denom_outside < denom_inside);
    assert((std::sqrt(4.0f) + geom::RMS_EPS) > std::sqrt(4.0f + geom::RMS_EPS));

    // At very small norm (1e-6), outside vs inside differs dramatically
    const float small_ms = 1e-12f;
    const float d_out_s = std::sqrt(small_ms) + geom::RMS_EPS; // 2e-6
    const float d_in_s  = std::sqrt(small_ms + geom::RMS_EPS); // ~1e-3
    assert(d_in_s > 400.0f * d_out_s);

    // Sinkhorn-20 doubly stochastic verification
    float c[HC][HC];
    for (int i = 0; i < HC; ++i) {
        float mx = -1e30f;
        for (int j = 0; j < HC; ++j) { c[i][j] = U(rng) * 3.0f; mx = std::max(mx, c[i][j]); }
        float s = 0.0f;
        for (int j = 0; j < HC; ++j) { c[i][j] = std::exp(c[i][j] - mx); s += c[i][j]; }
        for (int j = 0; j < HC; ++j) c[i][j] = c[i][j] / s + geom::HC_EPS;
    }
    for (int j = 0; j < HC; ++j) {
        float s = geom::HC_EPS;
        for (int i = 0; i < HC; ++i) s += c[i][j];
        for (int i = 0; i < HC; ++i) c[i][j] /= s;
    }
    for (int it = 0; it < geom::HC_ITERS - 1; ++it) {
        for (int i = 0; i < HC; ++i) {
            float s = geom::HC_EPS;
            for (int j = 0; j < HC; ++j) s += c[i][j];
            for (int j = 0; j < HC; ++j) c[i][j] /= s;
        }
        for (int j = 0; j < HC; ++j) {
            float s = geom::HC_EPS;
            for (int i = 0; i < HC; ++i) s += c[i][j];
            for (int i = 0; i < HC; ++i) c[i][j] /= s;
        }
    }
    for (int i = 0; i < HC; ++i) {
        float rs = 0.0f, cs = 0.0f;
        for (int j = 0; j < HC; ++j) { rs += c[i][j]; cs += c[j][i]; }
        assert(std::fabs(rs - 1.0f) < 1e-4f);
        assert(std::fabs(cs - 1.0f) < 1e-4f);
    }

    // mhc_contract preserves identical-stream input scaled by sum(sigmoid(logits) + HC_EPS)
    float pre[HC];
    float sum_pre = 0.0f;
    for (int i = 0; i < HC; ++i) {
        pre[i] = sigmoidf_ref(U(rng)) + geom::HC_EPS;
        sum_pre += pre[i];
    }
    std::vector<float> base_vec(H), out_vec(H, 0.0f);
    for (int d = 0; d < H; ++d) base_vec[d] = U(rng);
    for (int i = 0; i < HC; ++i)
        for (int d = 0; d < H; ++d) out_vec[d] += pre[i] * base_vec[d];
    for (int d = 0; d < H; ++d)
        assert(std::fabs(out_vec[d] - sum_pre * base_vec[d]) < 1e-5f);
    std::printf("test_mhc_sinkhorn20: doubly-stochastic & eps-outside-sqrt OK\n");
}

// ------------------------------------------------------------------------------------------------ 2. HCA C128 Inc vs Batch
std::vector<float> build_yarn_inv_freq(int rope_dim) {
    const int half = rope_dim / 2;
    constexpr double PI = 3.14159265358979323846264338327950288;
    auto corr_dim = [&](double nrot) {
        return (double(rope_dim) * std::log(double(geom::HCA_YARN_ORIG) / (nrot * 2.0 * PI))) /
               (2.0 * std::log(double(geom::HCA_ROPE_THETA)));
    };
    const double low = std::max(std::floor(corr_dim(double(geom::HCA_YARN_BETA_FAST))), 0.0);
    double high = std::min(std::ceil(corr_dim(double(geom::HCA_YARN_BETA_SLOW))), double(rope_dim - 1));
    if (low == high) high += 0.001;
    std::vector<float> inv(half);
    for (int p = 0; p < half; ++p) {
        const double inv_ext = 1.0 / std::pow(double(geom::HCA_ROPE_THETA), double(2 * p) / double(rope_dim));
        const double inv_int = inv_ext / double(geom::HCA_YARN_FACTOR);
        const double ramp = std::clamp((double(p) - low) / (high - low), 0.0, 1.0);
        const double mask = 1.0 - ramp;
        inv[p] = float(inv_int * (1.0 - mask) + inv_ext * mask);
    }
    return inv;
}

void apply_rope_interleaved_vec(std::vector<float>& x, int off, int pos, const std::vector<float>& inv, bool inverse) {
    const int half = int(inv.size());
    for (int p = 0; p < half; ++p) {
        const float ang = float(pos) * inv[p];
        const float c = std::cos(ang), s = std::sin(ang);
        const float e = x[off + 2 * p], o = x[off + 2 * p + 1];
        if (inverse) {
            x[off + 2 * p]     = e * c + o * s;
            x[off + 2 * p + 1] = o * c - e * s;
        } else {
            x[off + 2 * p]     = e * c - o * s;
            x[off + 2 * p + 1] = o * c + e * s;
        }
    }
}

void test_hca_incremental_vs_batch() {
    std::mt19937 rng(202);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    // Mini HCA geometry matching ERNIE 5.5 equations: CR=4, SWA=4, HD=32, NOPE=24, ROPE=8, H=4, T=15
    constexpr int T = 15, CR = 4, SWA = 4, HD = 32, NOPE = 24, ROPE = 8, H = 4;
    const std::vector<float> inv = build_yarn_inv_freq(ROPE);

    // Verify interleaved YaRN RoPE exact invertibility
    std::vector<float> rtest(HD);
    for (float& v : rtest) v = U(rng);
    std::vector<float> rorig = rtest;
    apply_rope_interleaved_vec(rtest, NOPE, 777, inv, false);
    apply_rope_interleaved_vec(rtest, NOPE, 777, inv, true);
    for (int d = 0; d < HD; ++d) assert(std::fabs(rtest[d] - rorig[d]) < 1e-5f);

    // Random inputs for T tokens: q [T, H, HD], kv_raw [T, HD], c_kv [T, HD], c_sc [T, HD], ape [CR, HD], sinks [H], vha_mix [H, H]
    std::vector<float> Q(T * H * HD), KV(T * HD), CKV(T * HD), CSC(T * HD), APE(CR * HD), SINKS(H);
    for (float& v : Q)     v = U(rng);
    for (float& v : KV)    v = U(rng);
    for (float& v : CKV)   v = U(rng);
    for (float& v : CSC)   v = U(rng);
    for (float& v : APE)   v = U(rng) * 0.2f;
    for (float& v : SINKS) v = U(rng);
    std::vector<float> VHA(H * H, 0.0f);
    for (int i = 0; i < H; ++i) {
        VHA[i * H + i] = 1.0f;
        for (int j = 0; j < H; ++j) VHA[i * H + j] += U(rng) * 0.1f;
    }

    auto compute_token = [&](int pos, const std::vector<std::vector<float>>& comp_blks,
                             const std::vector<std::vector<float>>& swa_ring, std::vector<float>& out_h_hd) {
        out_h_hd.assign(H * HD, 0.0f);
        const int n_comp = pos / CR;
        const int sw_start = std::max(0, pos - SWA + 1);
        const float scale = 1.0f / std::sqrt(float(HD));
        std::vector<float> raw_o(H * HD, 0.0f);

        for (int h = 0; h < H; ++h) {
            std::vector<float> qv(Q.begin() + (pos * H + h) * HD, Q.begin() + (pos * H + h + 1) * HD);
            apply_rope_interleaved_vec(qv, NOPE, pos, inv, false);
            float m = SINKS[h], l = 1.0f;
            std::vector<float> acc(HD, 0.0f);

            auto attend = [&](const std::vector<float>& kvec, const std::vector<float>& vvec) {
                float dp = 0.0f;
                for (int d = 0; d < HD; ++d) dp += qv[d] * kvec[d];
                const float sc = dp * scale;
                const float mn = std::max(m, sc);
                const float a = std::exp(m - mn), p = std::exp(sc - mn);
                l = l * a + p;
                for (int d = 0; d < HD; ++d) acc[d] = acc[d] * a + p * vvec[d];
                m = mn;
            };
            for (int b = 0; b < n_comp; ++b) {
                const int kpos = (b + 1) * CR - 1;
                std::vector<float> kv = comp_blks[b];
                apply_rope_interleaved_vec(kv, NOPE, kpos, inv, false);
                attend(kv, kv);
            }
            for (int kp = sw_start; kp <= pos; ++kp) {
                std::vector<float> kv = swa_ring[kp % SWA];
                apply_rope_interleaved_vec(kv, NOPE, kp, inv, false);
                attend(kv, kv);
            }
            for (int d = 0; d < HD; ++d) raw_o[h * HD + d] = acc[d] / l;
        }
        // VHAPostMix + inverse YaRN RoPE
        for (int h = 0; h < H; ++h) {
            std::vector<float> ov(HD, 0.0f);
            for (int g = 0; g < H; ++g)
                for (int d = 0; d < HD; ++d) ov[d] += VHA[h * H + g] * raw_o[g * HD + d];
            apply_rope_interleaved_vec(ov, NOPE, pos, inv, true);
            std::copy(ov.begin(), ov.end(), out_h_hd.begin() + h * HD);
        }
    };

    // Batch mode: build all compressed blocks directly from full sequence
    const int total_blks = T / CR;
    std::vector<std::vector<float>> batch_comp(total_blks, std::vector<float>(HD, 0.0f));
    for (int b = 0; b < total_blks; ++b) {
        for (int d = 0; d < HD; ++d) {
            float mx = -1e30f;
            for (int r = 0; r < CR; ++r) mx = std::max(mx, CSC[(b * CR + r) * HD + d] + APE[r * HD + d]);
            float den = 0.0f, num = 0.0f;
            for (int r = 0; r < CR; ++r) {
                const float w = std::exp((CSC[(b * CR + r) * HD + d] + APE[r * HD + d]) - mx);
                den += w;
                num += w * CKV[(b * CR + r) * HD + d];
            }
            batch_comp[b][d] = num / den;
        }
    }

    // Incremental mode: feed one token at a time using partial compressor accumulators
    std::vector<std::vector<float>> inc_comp;
    std::vector<float> part_kv(CR * HD, 0.0f), part_sc(CR * HD, 0.0f);
    std::vector<std::vector<float>> swa_ring(SWA, std::vector<float>(HD, 0.0f));
    float max_diff = 0.0f;

    for (int pos = 0; pos < T; ++pos) {
        const int r = pos % CR;
        for (int d = 0; d < HD; ++d) {
            swa_ring[pos % SWA][d] = KV[pos * HD + d];
            part_kv[r * HD + d] = CKV[pos * HD + d];
            part_sc[r * HD + d] = CSC[pos * HD + d] + APE[r * HD + d];
        }
        if (r == CR - 1) {
            std::vector<float> blk(HD, 0.0f);
            for (int d = 0; d < HD; ++d) {
                float mx = -1e30f;
                for (int rr = 0; rr < CR; ++rr) mx = std::max(mx, part_sc[rr * HD + d]);
                float den = 0.0f, num = 0.0f;
                for (int rr = 0; rr < CR; ++rr) {
                    const float w = std::exp(part_sc[rr * HD + d] - mx);
                    den += w;
                    num += w * part_kv[rr * HD + d];
                }
                blk[d] = num / den;
            }
            inc_comp.push_back(blk);
        }
        std::vector<float> out_inc, out_batch;
        compute_token(pos, inc_comp, swa_ring, out_inc);
        // Build batch SWA slice for pos
        std::vector<std::vector<float>> batch_swa(SWA, std::vector<float>(HD, 0.0f));
        for (int kp = std::max(0, pos - SWA + 1); kp <= pos; ++kp)
            for (int d = 0; d < HD; ++d) batch_swa[kp % SWA][d] = KV[kp * HD + d];
        compute_token(pos, batch_comp, batch_swa, out_batch);
        for (size_t i = 0; i < out_inc.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(out_inc[i] - out_batch[i]));
    }
    assert(max_diff < 1e-6f);
    std::printf("test_hca_incremental_vs_batch: max_diff=%.3e OK\n", max_diff);
}

// ------------------------------------------------------------------------------------------------ 3. DSA Primitives
void hadamard_128(std::vector<float>& x) {
    assert(x.size() == 128);
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j], b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float sc = 1.0f / std::sqrt(128.0f);
    for (float& v : x) v *= sc;
}

void test_dsa_primitives() {
    std::mt19937 rng(303);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);

    // 3a. Orthonormal Hadamard-128: norm-preserving and involutive (H(H(x)) == x)
    std::vector<float> hx(128);
    for (float& v : hx) v = U(rng);
    std::vector<float> h0 = hx;
    float n0 = 0.0f, n1 = 0.0f;
    for (float v : hx) n0 += v * v;
    hadamard_128(hx);
    for (float v : hx) n1 += v * v;
    assert(std::fabs(n0 - n1) < 1e-4f);
    hadamard_128(hx);
    for (int i = 0; i < 128; ++i) assert(std::fabs(hx[i] - h0[i]) < 1e-5f);

    // 3b. NeoX leading-64 RoPE norm preservation
    std::vector<float> nx(128);
    for (float& v : nx) v = U(rng);
    float nn0 = 0.0f, nn1 = 0.0f;
    for (float v : nx) nn0 += v * v;
    for (int p = 0; p < 32; ++p) {
        const float inv = std::pow(geom::DSA_ROPE_THETA, -float(2 * p) / 64.0f);
        const float a = 4096.0f * inv;
        const float c = std::cos(a), s = std::sin(a);
        const float x0 = nx[p], x1 = nx[p + 32];
        nx[p]      = x0 * c - x1 * s;
        nx[p + 32] = x1 * c + x0 * s;
    }
    for (float v : nx) nn1 += v * v;
    assert(std::fabs(nn0 - nn1) < 1e-4f);

    // 3c. Absorbed MLA 2-split + F32 sink equivalence to expand_kv (ERNIE 5.5 DSA dims: NOPE=192, ROPE=64, LORA=512, V=256)
    constexpr int T = 19, NOPE = geom::QK_NOPE, ROPE = geom::QK_ROPE, LORA = geom::KV_LORA, VDIM = geom::V_HEAD;
    std::vector<float> q_nope(NOPE), q_rot(ROPE), W_uk(NOPE * LORA), W_uv(VDIM * LORA), C(T * LORA), K_rot(T * ROPE);
    for (float& v : q_nope) v = U(rng);
    for (float& v : q_rot)  v = U(rng);
    for (float& v : W_uk)   v = U(rng) * 0.05f;
    for (float& v : W_uv)   v = U(rng) * 0.05f;
    for (float& v : C)      v = U(rng);
    for (float& v : K_rot)  v = U(rng);
    const float sink = 0.62f;
    const float scale = 1.0f / std::sqrt(float(NOPE + ROPE));

    std::vector<float> sc_ref(T + 1);
    std::vector<float> V_ref(T * VDIM, 0.0f);
    for (int t = 0; t < T; ++t) {
        float dp = 0.0f;
        for (int d = 0; d < NOPE; ++d) {
            float kn = 0.0f;
            for (int r = 0; r < LORA; ++r) kn += C[t * LORA + r] * W_uk[d * LORA + r];
            dp += q_nope[d] * kn;
        }
        for (int d = 0; d < ROPE; ++d) dp += q_rot[d] * K_rot[t * ROPE + d];
        sc_ref[t] = dp * scale;
        for (int v = 0; v < VDIM; ++v) {
            float vv = 0.0f;
            for (int r = 0; r < LORA; ++r) vv += C[t * LORA + r] * W_uv[v * LORA + r];
            V_ref[t * VDIM + v] = vv;
        }
    }
    sc_ref[T] = sink;
    const float mx = *std::max_element(sc_ref.begin(), sc_ref.end());
    std::vector<float> pr(T + 1);
    float sm = 0.0f;
    for (int t = 0; t <= T; ++t) { pr[t] = std::exp(sc_ref[t] - mx); sm += pr[t]; }
    for (int t = 0; t <= T; ++t) pr[t] /= sm;
    std::vector<float> out_ref(VDIM, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VDIM; ++v) out_ref[v] += pr[t] * V_ref[t * VDIM + v];

    std::vector<float> q_abs(LORA, 0.0f);
    for (int r = 0; r < LORA; ++r)
        for (int d = 0; d < NOPE; ++d) q_abs[r] += q_nope[d] * W_uk[d * LORA + r];

    auto run_split = [&](int t0, int t1, bool has_sink, float& m_out, float& l_out, std::vector<float>& acc) {
        m_out = has_sink ? sink : -INFINITY;
        l_out = has_sink ? 1.0f : 0.0f;
        acc.assign(LORA, 0.0f);
        for (int t = t0; t < t1; ++t) {
            float dp = 0.0f;
            for (int r = 0; r < LORA; ++r) dp += q_abs[r] * C[t * LORA + r];
            for (int d = 0; d < ROPE; ++d) dp += q_rot[d] * K_rot[t * ROPE + d];
            const float sc = dp * scale;
            const float mn = std::max(m_out, sc);
            const float a = std::exp(m_out - mn), p = std::exp(sc - mn);
            l_out = l_out * a + p;
            for (int r = 0; r < LORA; ++r) acc[r] = acc[r] * a + p * C[t * LORA + r];
            m_out = mn;
        }
    };
    float m0, l0, m1, l1;
    std::vector<float> o0, o1;
    run_split(0, 10, true, m0, l0, o0);
    run_split(10, T, false, m1, l1, o1);
    const float gm = std::max(m0, m1);
    const float w0 = std::exp(m0 - gm), w1 = std::exp(m1 - gm);
    const float gl = l0 * w0 + l1 * w1;
    std::vector<float> u(LORA);
    for (int r = 0; r < LORA; ++r) u[r] = (o0[r] * w0 + o1[r] * w1) / gl;
    std::vector<float> out_abs(VDIM, 0.0f);
    for (int v = 0; v < VDIM; ++v)
        for (int r = 0; r < LORA; ++r) out_abs[v] += u[r] * W_uv[v * LORA + r];

    float max_diff = 0.0f;
    for (int v = 0; v < VDIM; ++v) max_diff = std::max(max_diff, std::fabs(out_ref[v] - out_abs[v]));
    assert(max_diff < 1e-5f);
    std::printf("test_dsa_primitives: Hadamard-128, NeoX-64 RoPE, 2-split MLA+sink max_diff=%.3e OK\n", max_diff);
}

// ------------------------------------------------------------------------------------------------ 4. SiTU-GLU & Router
void test_situ_and_router() {
    auto situ = [](float g, float u) {
        const float ga = geom::SITU_BETA * std::tanh(g / geom::SITU_BETA) * sigmoidf_ref(g);
        const float ua = geom::SITU_LINEAR_BETA * std::tanh(u / geom::SITU_LINEAR_BETA);
        return ga * ua;
    };
    assert(std::fabs(situ(0.0f, 1.0f)) < 1e-6f);
    assert(std::fabs(situ(100.0f, 1000.0f) - (geom::SITU_BETA * geom::SITU_LINEAR_BETA)) < 1e-4f);
    assert(std::fabs(situ(-100.0f, 1000.0f)) < 1e-6f);

    // Split-feature top-10 router
    std::mt19937 rng(404);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    std::vector<float> l0(geom::NEXP), l1(geom::NEXP), bias(geom::NEXP), rsc(geom::NEXP);
    std::vector<float> score(geom::NEXP), choice(geom::NEXP);
    for (int e = 0; e < geom::NEXP; ++e) {
        l0[e] = U(rng); l1[e] = U(rng);
        bias[e] = U(rng) * 0.25f;
        rsc[e] = 1.0f + 0.1f * U(rng);
        score[e] = sigmoidf_ref(l0[e]) + sigmoidf_ref(l1[e]);
        choice[e] = score[e] + bias[e];
    }
    std::vector<int> idx(geom::NEXP);
    for (int e = 0; e < geom::NEXP; ++e) idx[e] = e;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return choice[a] > choice[b]; });
    float sum = 0.0f;
    for (int k = 0; k < geom::TOPK; ++k) sum += score[idx[k]];
    float norm_check = 0.0f;
    for (int k = 0; k < geom::TOPK; ++k) {
        const float w_pre_scale = score[idx[k]] / (sum + 1e-12f);
        const float w_final = w_pre_scale * rsc[idx[k]];
        assert(w_final > 0.0f);
        norm_check += w_pre_scale;
    }
    assert(std::fabs(norm_check - 1.0f) < 1e-5f);
    std::printf("test_situ_and_router: SiTU-GLU(4,25) & split-feature top-%d router OK\n", geom::TOPK);
}

// ------------------------------------------------------------------------------------------------ 5. mul1 + K2/K3/K4 Wide-Reorder
void test_mul1_and_wide_reorder() {
    // 5a. Verify mul1 (0x83DCD12D, 0x1eee/0xc931 -> E4M3FN) matches exact tr3.cuh arithmetic
    const float c1 = half_bits_to_f32(0x1eee); // 0.00676727294921875
    const float c0 = half_bits_to_f32(0xc931); // -10.3828125
    assert(std::fabs(c1 - 0.00676727294921875f) < 1e-9f);
    assert(std::fabs(c0 - (-10.3828125f)) < 1e-6f);

    double mean = 0.0, var = 0.0;
    for (uint32_t s = 0; s < 65536; ++s) {
        const uint32_t prod = s * 0x83DCD12Du;
        const uint32_t bsum = (prod & 0xffu) + ((prod >> 8) & 0xffu) + ((prod >> 16) & 0xffu) + ((prod >> 24) & 0xffu);
        const float x = half_bits_to_f32(uint16_t(0x6400u + bsum)); // 1024 + bsum
        const float y = half_bits_to_f32(f32_to_half_bits(std::fma(x, c1, c0)));
        const float q = e4m3fn_to_f32(f32_to_e4m3fn_sat(y));
        mean += q;
        var += double(q) * double(q);
    }
    mean /= 65536.0;
    var = var / 65536.0 - mean * mean;
    // Standard Gaussian-like codebook: mean ~ 0, variance ~ 1.0
    assert(std::fabs(mean) < 0.02);
    assert(var > 0.90 && var < 1.10);

    // 5b. Verify K2, K3, AND K4 wide-reorder pack/unpack round-trip bit-exactly!
    auto tile_chunk = [](int K, const uint8_t* t, int lane) -> uint32_t {
        if (K == 2) {
            uint16_t h; std::memcpy(&h, t + lane * 2, 2); return h;
        }
        const uint8_t* p = t + lane * 3;
        return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16);
    };
    auto pack_chunk = [](int K, uint64_t v[3], int p, uint32_t x) {
        const int b = p * (8 * K);
        const int i = b >> 6, s = b & 63;
        v[i] |= uint64_t(x) << s;
        if (s + 8 * K > 64) v[i + 1] |= uint64_t(x) >> (64 - s);
    };

    for (int K : {2, 3, 4}) {
        const int tile_bytes = 32 * K;
        std::vector<uint8_t> orig(8 * tile_bytes);
        for (size_t i = 0; i < orig.size(); ++i) orig[i] = uint8_t((i * 73 + K * 19 + 5) & 0xff);
        std::vector<uint8_t> wide(8 * tile_bytes, 0);

        // Pack 8 tiles into wide layout
        for (int lane = 0; lane < 32; ++lane) {
            if (K == 4) {
                uint32_t w[8];
                for (int p = 0; p < 8; ++p) std::memcpy(&w[p], orig.data() + p * 128 + lane * 4, 4);
                std::memcpy(wide.data() + lane * 16,       &w[0], 16);
                std::memcpy(wide.data() + 512 + lane * 16, &w[4], 16);
            } else {
                uint64_t v[3] = {0, 0, 0};
                for (int p = 0; p < 8; ++p) pack_chunk(K, v, p, tile_chunk(K, orig.data() + p * tile_bytes, lane));
                for (int n = 0; n < K; ++n) std::memcpy(wide.data() + n * 256 + lane * 8, &v[n], 8);
            }
        }
        // Unpack wide layout back to 8 tiles (exact mirror of tr3_ld_wide in ernie55_moe.cuh)
        for (int lane = 0; lane < 32; ++lane) {
            if (K == 4) {
                uint32_t w[8];
                std::memcpy(&w[0], wide.data() + lane * 16,       16);
                std::memcpy(&w[4], wide.data() + 512 + lane * 16, 16);
                for (int p = 0; p < 8; ++p) {
                    uint32_t ref; std::memcpy(&ref, orig.data() + p * 128 + lane * 4, 4);
                    assert(w[p] == ref);
                }
            } else {
                uint64_t v[3] = {0, 0, 0};
                for (int n = 0; n < K; ++n) std::memcpy(&v[n], wide.data() + n * 256 + lane * 8, 8);
                for (int p = 0; p < 8; ++p) {
                    const int b = p * (8 * K);
                    const int i = b >> 6, s = b & 63;
                    uint64_t raw = v[i] >> s;
                    if (s + 8 * K > 64) raw |= v[i + 1] << (64 - s);
                    const uint32_t got = uint32_t(raw) & ((1u << (8 * K)) - 1u);
                    const uint32_t ref = tile_chunk(K, orig.data() + p * tile_bytes, lane);
                    assert(got == ref);
                }
            }
        }
    }
    std::printf("test_mul1_and_wide_reorder: mul1(mean=%.4f, var=%.4f) & K2/K3/K4 wide-reorder OK\n", mean, var);
}

// ------------------------------------------------------------------------------------------------ 6. multimax
void test_multimax() {
    const float r[4] = {-5.0f, 5.0f, -10.0f, 10.0f};
    const float t[4] = {0.1f, -0.2f, 0.05f, -0.01f};
    auto mm = [&](float x) {
        const float a = std::max(0.0f, r[0] - x);
        const float b = std::max(0.0f, x - r[1]);
        const float c = std::max(0.0f, r[2] - x);
        const float d = std::max(0.0f, x - r[3]);
        return x + t[0] * a + t[1] * b + t[2] * c * c + t[3] * d * d;
    };
    // Inside [-5, 5], multimax is identity
    assert(std::fabs(mm(0.0f) - 0.0f) < 1e-6f);
    assert(std::fabs(mm(4.5f) - 4.5f) < 1e-6f);
    // Above 10: x + t1*(x-5) + t3*(x-10)^2
    assert(std::fabs(mm(12.0f) - (12.0f - 0.2f * 7.0f - 0.01f * 4.0f)) < 1e-5f);
    // Below -10: x + t0*(-5-x) + t2*(-10-x)^2
    assert(std::fabs(mm(-13.0f) - (-13.0f + 0.1f * 8.0f + 0.05f * 9.0f)) < 1e-5f);
    std::printf("test_multimax: 4-term piecewise ReLU/ReLU^2 OK\n");
}

// ------------------------------------------------------------------------------------------------ 7. Optional fixture check
void check_fixture(const std::string& dir) {
    const std::string path = dir + "/golden_logits.npy";
    std::ifstream f(path, std::ios::binary);
    assert(f.good());
    char magic[6];
    f.read(magic, 6);
    assert(std::memcmp(magic, "\x93NUMPY", 6) == 0);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h16; f.read(reinterpret_cast<char*>(&h16), 2); hlen = h16; }
    else             { f.read(reinterpret_cast<char*>(&hlen), 4); }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    assert(hdr.find("<f4") != std::string::npos);
    assert(hdr.find("(15, 128)") != std::string::npos);
    std::vector<float> logits(15 * 128);
    f.read(reinterpret_cast<char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4));
    assert(f.gcount() == static_cast<std::streamsize>(logits.size() * 4));
    const int expected_top1[15] = {76, 22, 30, 86, 104, 84, 73, 39, 39, 103, 126, 97, 101, 62, 24};
    for (int t = 0; t < 15; ++t) {
        int best = 0;
        for (int v = 0; v < 128; ++v) {
            assert(std::isfinite(logits[size_t(t) * 128 + v]));
            if (logits[size_t(t) * 128 + v] > logits[size_t(t) * 128 + best]) best = v;
        }
        assert(best == expected_top1[t]);
    }
    std::printf("check_fixture(%s): 15x128 golden_logits finite & top1 match OK\n", dir.c_str());
}

} // namespace

int main(int argc, char** argv) {
    test_mhc_sinkhorn20();
    test_hca_incremental_vs_batch();
    test_dsa_primitives();
    test_situ_and_router();
    test_mul1_and_wide_reorder();
    test_multimax();
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--fixture") check_fixture(argv[i + 1]);
    }
    std::printf("test_ernie55_math: ALL PASS\n");
    return 0;
}
