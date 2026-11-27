// Native Inkling-2-Mini DFlash2 Parallel Block Speculative Drafter
// Supports both pre-quantized Melded Drafter-in-Quant artifacts (`dflash2-melded.safetensors`: SoA Q4_0/Q8_0
// body + Q4_0/Q8_0 selector codebooks, ~278 MB on disk, zero runtime quantization) and stock BF16
// `incoai/Inkling-2-Mini-DFlash2` checkpoints (Apache-2.0, https://huggingface.co/incoai/Inkling-2-Mini-DFlash2).
//
// Target verification remains 100% exact trunk (zero quality or greedy-parity impact).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "sm70_cublas.cuh"
#include "ink2_norm_res.cuh"
#endif

namespace ink2 {
namespace df2 {

constexpr int L = 5;
constexpr int NT = 4;                        // target_layer_ids = [5, 23, 41, 48]
constexpr int HID = 2304;
constexpr int NH = 32;
constexpr int NKV = 8;
constexpr int GQA = NH / NKV;                // 4
constexpr int HD = 128;
constexpr int INTER = 6912;
constexpr int B = 8;                         // frozen bidirectional block size
constexpr int K_MAX = B - 1;                 // 7 max draft tokens
constexpr int WIN = 2048;                    // sliding_window
constexpr int RING = 4096;                   // power-of-two FP16 ring (80 MiB / session)
constexpr int GS = 16;                       // conv_group_size
constexpr int NG = HID / GS;                 // 144
constexpr int CPO = 4 * NG;                  // 576 (2 sides x 2 taps x 144 groups)
constexpr int TOPK = 16;                     // selector_top_k
constexpr int SELR = 256;                    // selector_rank
constexpr int MASK_ID = 200056;              // mask_token_id
constexpr float ROPE_THETA = 10000.0f;
constexpr float LOG_ROPE_THETA = 9.210340371976184f;
constexpr float RMS_EPS = 1e-6f;
constexpr float SOFTCAP = 30.0f;
constexpr int ATTN_SPLIT = 4;

// Head2 2-bit shortlist constants (HID = 2304 -> NCH = 36 chunks of 64 weights)
constexpr int H2_NCH = HID / 64;             // 36
constexpr int H2_K8 = HID / 8;               // 288
constexpr int H2_SL = 64;                    // exact-rescored candidates per row
constexpr int H2_BINS = 2048;
constexpr float H2_RANGE = 32.0f;

// Resolve DFlash2 artifact path from explicit knob override or model directory auto-discovery.
// Returns empty string if explicitly disabled ("0", "off", "none") or no bundled artifact exists.
inline std::string resolve_dflash2_path(const std::string& model_dir, const std::optional<std::string>& knob_val) {
    if (knob_val && !knob_val->empty()) {
        const std::string& v = *knob_val;
        if (v == "0" || v == "off" || v == "OFF" || v == "none" || v == "NONE") return "";
        return v;
    }
    static const char* const kAutoNames[] = {
        "dflash2-melded.safetensors",
        "dflash2.exlr8.safetensors",
    };
    for (const char* name : kAutoNames) {
        const std::string cand = model_dir + "/" + name;
        if (::access(cand.c_str(), R_OK) == 0) return cand;
    }
    return "";
}

// Host IEEE-754 float32 <-> float16 conversion (round-to-nearest-even) for CPU Q4_0 verification tests.
inline uint16_t f32_to_f16_host(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    const uint32_t mant = x & 0x7fffffu;
    if (exp == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = mant | 0x800000u;
        const int shift = 14 - e;
        uint32_t res = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (res & 1u))) ++res;
        return static_cast<uint16_t>(sign | res);
    }
    uint32_t res = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (res & 1u))) ++res;
    return static_cast<uint16_t>(sign | res);
}

inline float f16_to_f32_host(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t x = 0;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            x = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &x, 4);
    return out;
}

// Reference SoA Q4_0 pack/unpack matching k_df2_bf16_to_q4_soa and tools/ink2_pack_dflash2.py bit-for-bit.
inline void q4_pack_block_host(const float src[32], uint8_t dst_qs[16], uint16_t& dst_sc_f16) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(src[i]));
    const float d = (amax > 1e-12f) ? (amax / 7.5f) : 0.0f;
    dst_sc_f16 = f32_to_f16_host(d);
    const float df = f16_to_f32_host(dst_sc_f16);
    const float id = (df > 0.0f) ? (1.0f / df) : 0.0f;
    int q[32];
    for (int i = 0; i < 32; ++i) {
        q[i] = (df > 0.0f) ? std::max(0, std::min(15, static_cast<int>(std::nearbyint(src[i] * id + 8.0f)))) : 8;
    }
    for (int j = 0; j < 16; ++j) {
        dst_qs[j] = static_cast<uint8_t>((q[j] & 15) | ((q[j + 16] & 15) << 4));
    }
}

inline void q4_unpack_block_host(const uint8_t qs[16], uint16_t sc_f16, float dst[32]) {
    const float d = f16_to_f32_host(sc_f16);
    for (int j = 0; j < 16; ++j) {
        const uint8_t b = qs[j];
        dst[j] = static_cast<float>(static_cast<int>(b & 15u) - 8) * d;
        dst[j + 16] = static_cast<float>(static_cast<int>(b >> 4) - 8) * d;
    }
}

// ------------------------------------------------------------------------------------------- Minimal Safetensors Parser
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t off = 0, len = 0;
};

inline bool parse_safetensors_header(const std::string& path, std::unordered_map<std::string, StTensor>& out, size_t& data_base) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    if (!f || hlen == 0 || hlen > (64ull << 20)) return false;
    std::string hdr(static_cast<size_t>(hlen), '\0');
    f.read(&hdr[0], static_cast<std::streamsize>(hlen));
    if (!f) return false;
    data_base = 8 + static_cast<size_t>(hlen);

    // Simple deterministic scanner for top-level entries: "name":{"dtype":"...","shape":[...],"data_offsets":[a,b]}
    size_t pos = 0;
    auto skip_ws = [&]() { while (pos < hdr.size() && (hdr[pos] == ' ' || hdr[pos] == '\n' || hdr[pos] == '\r' || hdr[pos] == '\t')) ++pos; };
    auto read_str = [&]() -> std::string {
        skip_ws();
        if (pos >= hdr.size() || hdr[pos] != '"') return "";
        ++pos;
        std::string s;
        while (pos < hdr.size() && hdr[pos] != '"') {
            if (hdr[pos] == '\\' && pos + 1 < hdr.size()) ++pos;
            s.push_back(hdr[pos++]);
        }
        if (pos < hdr.size() && hdr[pos] == '"') ++pos;
        return s;
    };
    auto skip_val = [&]() {
        int depth = 0;
        bool in_s = false;
        while (pos < hdr.size()) {
            char ch = hdr[pos++];
            if (in_s) { if (ch == '\\' && pos < hdr.size()) ++pos; else if (ch == '"') in_s = false; continue; }
            if (ch == '"') { in_s = true; continue; }
            if (ch == '{' || ch == '[') ++depth;
            else if (ch == '}' || ch == ']') { if (depth == 0) { --pos; break; } --depth; }
            else if (ch == ',' && depth == 0) { --pos; break; }
        }
    };
    skip_ws();
    if (pos >= hdr.size() || hdr[pos] != '{') return false;
    ++pos;
    while (pos < hdr.size()) {
        skip_ws();
        if (pos >= hdr.size() || hdr[pos] == '}') break;
        if (hdr[pos] == ',') { ++pos; continue; }
        std::string key = read_str();
        skip_ws();
        if (pos < hdr.size() && hdr[pos] == ':') ++pos;
        skip_ws();
        if (key == "__metadata__" || pos >= hdr.size() || hdr[pos] != '{') { skip_val(); continue; }
        ++pos;
        StTensor t;
        while (pos < hdr.size()) {
            skip_ws();
            if (hdr[pos] == '}') { ++pos; break; }
            if (hdr[pos] == ',') { ++pos; continue; }
            std::string fk = read_str();
            skip_ws();
            if (pos < hdr.size() && hdr[pos] == ':') ++pos;
            skip_ws();
            if (fk == "dtype") {
                t.dtype = read_str();
            } else if (fk == "shape" && hdr[pos] == '[') {
                ++pos;
                while (pos < hdr.size()) {
                    skip_ws();
                    if (hdr[pos] == ']') { ++pos; break; }
                    if (hdr[pos] == ',') { ++pos; continue; }
                    char* ep = nullptr;
                    t.shape.push_back(std::strtoll(&hdr[pos], &ep, 10));
                    pos = static_cast<size_t>(ep - hdr.data());
                }
            } else if (fk == "data_offsets" && hdr[pos] == '[') {
                ++pos;
                char* ep = nullptr;
                size_t a = std::strtoull(&hdr[pos], &ep, 10); pos = static_cast<size_t>(ep - hdr.data());
                while (pos < hdr.size() && (hdr[pos] == ',' || hdr[pos] == ' ')) ++pos;
                size_t b = std::strtoull(&hdr[pos], &ep, 10); pos = static_cast<size_t>(ep - hdr.data());
                while (pos < hdr.size() && hdr[pos] != ']') ++pos;
                if (pos < hdr.size()) ++pos;
                t.off = a; t.len = b - a;
            } else {
                skip_val();
            }
        }
        out[key] = std::move(t);
    }
    return !out.empty();
}


#ifdef __CUDACC__

struct Q4Mat {
    uint8_t* qs = nullptr;   // [rows, cols/2] Q4_0 (when Q8 off)
    int8_t* q8 = nullptr;    // [rows, cols] Q8_0 (when GR8_INK2_DF2_Q8=1)
    __half* sc = nullptr;    // [rows, cols/32] FP16 block-32 scales
    int rows = 0;
    int cols = 0;
};

struct Q4Codebook {
    const uint8_t* qs = nullptr;           // [vocab, SELR/2] pre-quantized Q4_0
    const int8_t* q8 = nullptr;            // [vocab, SELR] pre-quantized Q8_0
    const __half* sc = nullptr;            // [vocab, SELR/32] FP16 scales
    const __nv_bfloat16* bf16 = nullptr;   // [vocab, SELR] stock BF16 fallback
    int vocab = 0;
};

struct Q8Act {
    int8_t* q = nullptr;     // [max_M * max_K]
    float* sc = nullptr;     // [max_M * (max_K / 32)]
    int* isum = nullptr;     // [max_M * (max_K / 16)] pre-scaled by 8 for exact -8*sum offset
};

struct Cand16 {
    float logit;
    int id;
};

struct DpCal {
    float gamma[K_MAX];
    float beta0[K_MAX];
    float beta1[K_MAX];
    float beta2[K_MAX];
    DpCal() {
        const float dg[K_MAX]  = {0.90f, 0.78f, 0.72f, 0.68f, 0.65f, 0.62f, 0.60f};
        const float db0[K_MAX] = {4.20f, 2.80f, 3.20f, 3.50f, 3.50f, 3.60f, 3.60f};
        const float db1[K_MAX] = {0.50f, 0.52f, 0.48f, 0.46f, 0.45f, 0.44f, 0.43f};
        const float db2[K_MAX] = {-0.20f, -0.35f, -0.30f, -0.35f, -0.30f, -0.30f, -0.25f};
        for (int i = 0; i < K_MAX; ++i) {
            gamma[i] = dg[i]; beta0[i] = db0[i]; beta1[i] = db1[i]; beta2[i] = db2[i];
        }
    }
};

struct LayerWeights {
    Q4Mat q, k, v, o;
    Q4Mat gate, up, down;
    Q4Mat aconv_kp, mconv_kp;
    uint16_t* k_bf16 = nullptr;     // [NKV*HD, HID] for fast cuBLAS prefill commit
    uint16_t* v_bf16 = nullptr;     // [NKV*HD, HID] for fast cuBLAS prefill commit
    float* in_norm = nullptr;       // [HID]
    float* post_norm = nullptr;     // [HID]
    float* q_norm = nullptr;        // [HD]
    float* k_norm = nullptr;        // [HD]
    float* aconv_base = nullptr;    // [2][2][HID]
    float* mconv_base = nullptr;    // [2][2][HID]
};

// ------------------------------------------------------------------------------------------- Utilities
__device__ __forceinline__ float df2_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float df2_silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float df2_bf(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ void df2_quant_warp_q8(float v, size_t blk, int lane,
                                                  int8_t* __restrict__ q_out,
                                                  float* __restrict__ sc_out,
                                                  int* __restrict__ sum_out) {
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = (amax > 1e-12f) ? (amax / 127.0f) : 0.0f;
    const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
    const int qi = (sc > 0.0f) ? max(-127, min(127, __float2int_rn(v * inv))) : 0;

    q_out[blk * 32 + lane] = static_cast<int8_t>(qi);

    int hs = qi;
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) hs += __shfl_xor_sync(0xffffffffu, hs, o);
    if (lane == 0) {
        sc_out[blk] = sc;
        sum_out[blk * 2] = hs << 3;
    } else if (lane == 16) {
        sum_out[blk * 2 + 1] = hs << 3;
    }
}

// Boot quantization: BF16 [rows, cols] -> SoA Q4_0 qs[rows, cols/2] + sc[rows, cols/32].
// Grid: ceil(total_blocks / 8), Block: 256 (1 warp per 32-element block).
__global__ void __launch_bounds__(256) k_df2_bf16_to_q4_soa(
    const uint16_t* __restrict__ src,
    uint8_t* __restrict__ dst_qs,
    __half* __restrict__ dst_sc,
    size_t total_blocks) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blocks) return;

    const float v = df2_bf(src[blk * 32 + lane]);
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // Match ggml Q4_0 symmetric scale: d = max_abs / 7.0 (actually ggml uses max_signed / -8; max_abs / 7.5 is symmetric & tight)
    const float d = (amax > 1e-12f) ? (amax / 7.5f) : 0.0f;
    const __half dh = __float2half_rn(d);
    const float df = __half2float(dh);
    const float id = (df > 0.0f) ? (1.0f / df) : 0.0f;
    const int q = (df > 0.0f) ? max(0, min(15, __float2int_rn(v * id + 8.0f))) : 8;

    const int q_hi = __shfl_down_sync(0xffffffffu, q, 16);
    if (lane < 16) {
        dst_qs[blk * 16 + lane] = static_cast<uint8_t>((q & 15) | ((q_hi & 15) << 4));
    }
    if (lane == 0) dst_sc[blk] = dh;
}

// Boot quantization: BF16 [rows, cols] -> SoA Q8_0 q8[rows, cols] + sc[rows, cols/32].
__global__ void __launch_bounds__(256) k_df2_bf16_to_q8_soa(
    const uint16_t* __restrict__ src,
    int8_t* __restrict__ dst_q8,
    __half* __restrict__ dst_sc,
    size_t total_blocks) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blocks) return;

    const float v = df2_bf(src[blk * 32 + lane]);
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = (amax > 1e-12f) ? (amax / 127.0f) : 0.0f;
    const __half dh = __float2half_rn(d);
    const float df = __half2float(dh);
    const float id = (df > 0.0f) ? (1.0f / df) : 0.0f;
    const int q = (df > 0.0f) ? max(-127, min(127, __float2int_rn(v * id))) : 0;
    dst_q8[blk * 32 + lane] = static_cast<int8_t>(q);
    if (lane == 0) dst_sc[blk] = dh;
}

__global__ void __launch_bounds__(256) k_df2_bf16_to_f32(const uint16_t* __restrict__ src, float* __restrict__ dst, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) dst[i] = df2_bf(src[i]);
}

// Boot dequantization for pre-quantized melded fc/k/v -> BF16 (used only for cuBLAS prefill commit, 0 extra disk bytes).
__global__ void __launch_bounds__(256) k_df2_q4_to_bf16(
    const uint8_t* __restrict__ qs,
    const __half* __restrict__ sc,
    uint16_t* __restrict__ dst_bf16,
    size_t total_blocks) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blocks) return;
    const float d = __half2float(sc[blk]);
    const int j = lane & 15;
    const uint8_t packed = qs[blk * 16 + j];
    const int q = (lane < 16) ? static_cast<int>(packed & 15u) : static_cast<int>(packed >> 4);
    const __nv_bfloat16 b = __float2bfloat16_rn(static_cast<float>(q - 8) * d);
    dst_bf16[blk * 32 + lane] = *reinterpret_cast<const uint16_t*>(&b);
}

__global__ void __launch_bounds__(256) k_df2_q8_to_bf16(
    const int8_t* __restrict__ q8,
    const __half* __restrict__ sc,
    uint16_t* __restrict__ dst_bf16,
    size_t total_blocks) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t blk = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (blk >= total_blocks) return;
    const float d = __half2float(sc[blk]);
    const __nv_bfloat16 b = __float2bfloat16_rn(static_cast<float>(q8[blk * 32 + lane]) * d);
    dst_bf16[blk * 32 + lane] = *reinterpret_cast<const uint16_t*>(&b);
}

// Quantize activation A[M, K] (F32 or BF16) to Q8_1.
template <bool IS_BF16>
__global__ void __launch_bounds__(256) k_df2_quant_q8(const void* __restrict__ A,
                                                      int8_t* __restrict__ q_out,
                                                      float* __restrict__ sc_out,
                                                      int* __restrict__ sum_out,
                                                      int M, int K) {
    const int nb_row = K >> 5;
    const int total = M * nb_row;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 8 + warp;
    if (blk >= total) return;

    const size_t base = static_cast<size_t>(blk) * 32 + lane;
    const float v = IS_BF16 ? df2_bf(static_cast<const uint16_t*>(A)[base]) : static_cast<const float*>(A)[base];
    df2_quant_warp_q8(v, static_cast<size_t>(blk), lane, q_out, sc_out, sum_out);
}

inline void quant_q8_f32(const float* A, Q8Act& qa, int M, int K, cudaStream_t st) {
    const int total = M * (K >> 5);
    k_df2_quant_q8<false><<<(total + 7) / 8, 256, 0, st>>>(A, qa.q, qa.sc, qa.isum, M, K);
}
inline void quant_q8_bf16(const uint16_t* A, Q8Act& qa, int M, int K, cudaStream_t st) {
    const int total = M * (K >> 5);
    k_df2_quant_q8<true><<<(total + 7) / 8, 256, 0, st>>>(A, qa.q, qa.sc, qa.isum, M, K);
}

// ------------------------------------------------------------------------------------------- Fast Q8_1 x Q4_0 __dp4a GEMVs
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_df2_q4_gemv(const int8_t* __restrict__ Aq,
                                                     const float* __restrict__ Asc,
                                                     const int* __restrict__ Asum,
                                                     const uint8_t* __restrict__ Wqs,
                                                     const __half* __restrict__ Wsc,
                                                     float* __restrict__ C,
                                                     int M, int O, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= O) return;

    const int nb = K >> 5;
    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int shift = hi << 2;
    const uint4* qrow = reinterpret_cast<const uint4*>(Wqs + static_cast<size_t>(o) * (K >> 1));
    const __half* srow = Wsc + static_cast<size_t>(o) * nb;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    const size_t a_stride_i4 = static_cast<size_t>(K) >> 4;
    const size_t sum_stride = static_cast<size_t>(nb) << 1;

    for (int b = bi0; b < nb; b += 16) {
        const uint4 qv = __ldcs(qrow + b);
        const float d = __half2float(__ldcs(srow + b));
        const int w0 = static_cast<int>((qv.x >> shift) & 0x0F0F0F0Fu);
        const int w1 = static_cast<int>((qv.y >> shift) & 0x0F0F0F0Fu);
        const int w2 = static_cast<int>((qv.z >> shift) & 0x0F0F0F0Fu);
        const int w3 = static_cast<int>((qv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
                int dot = 0;
                dot = __dp4a(w0, a.x, dot);
                dot = __dp4a(w1, a.y, dot);
                dot = __dp4a(w2, a.z, dot);
                dot = __dp4a(w3, a.w, dot);
                const float val = static_cast<float>(dot - __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx));
                acc[m] = fmaf(val, d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = df2_warp_sum(acc[m]);
            if (lane == 0) C[static_cast<size_t>(m) * O + o] = tot;
        }
    }
}

// Fast Q8_1 x Q8_0 __dp4a GEMV (zero-offset signed int8xint8, ~0.001 MSE vs BF16!)
template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(128) k_df2_q8_gemv(const int8_t* __restrict__ Aq,
                                                     const float* __restrict__ Asc,
                                                     const int8_t* __restrict__ Wq8,
                                                     const __half* __restrict__ Wsc,
                                                     float* __restrict__ C,
                                                     int M, int O, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= O) return;

    const int nb = K >> 5;
    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int4* qrow = reinterpret_cast<const int4*>(Wq8 + static_cast<size_t>(o) * K);
    const __half* srow = Wsc + static_cast<size_t>(o) * nb;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    const size_t a_stride_i4 = static_cast<size_t>(K) >> 4;

    for (int b = bi0; b < nb; b += 16) {
        const int hidx = (b << 1) | hi;
        const int4 wv = __ldcs(qrow + hidx);
        const float d = __half2float(__ldcs(srow + b));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
                int dot = 0;
                dot = __dp4a(wv.x, a.x, dot);
                dot = __dp4a(wv.y, a.y, dot);
                dot = __dp4a(wv.z, a.z, dot);
                dot = __dp4a(wv.w, a.w, dot);
                acc[m] = fmaf(static_cast<float>(dot), d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float tot = df2_warp_sum(acc[m]);
            if (lane == 0) C[static_cast<size_t>(m) * O + o] = tot;
        }
    }
}

inline void launch_q4_gemv(const Q8Act& qa, const Q4Mat& W, float* C, int M, cudaStream_t st) {
    const int g = (W.rows + 3) / 4;
    const int nb = W.cols >> 5;
    int off = 0;
    while (off < M) {
        const int rem = M - off;
        const int8_t* aq = qa.q + static_cast<size_t>(off) * W.cols;
        const float* asc = qa.sc + static_cast<size_t>(off) * nb;
        const int* asum = qa.isum + static_cast<size_t>(off) * (nb << 1);
        float* cd = C + static_cast<size_t>(off) * W.rows;
        if (W.q8) {
            if (rem >= 16) { k_df2_q8_gemv<16, true><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, 16, W.rows, W.cols); off += 16; }
            else if (rem == 8) { k_df2_q8_gemv<8, true><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, 8, W.rows, W.cols); off += 8; }
            else if (rem == 7) { k_df2_q8_gemv<7, true><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, 7, W.rows, W.cols); off += 7; }
            else if (rem == 1) { k_df2_q8_gemv<1, true><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, 1, W.rows, W.cols); off += 1; }
            else if (rem <= 4) { k_df2_q8_gemv<4, false><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, rem, W.rows, W.cols); off += rem; }
            else if (rem <= 8) { k_df2_q8_gemv<8, false><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, rem, W.rows, W.cols); off += rem; }
            else { k_df2_q8_gemv<16, false><<<g, 128, 0, st>>>(aq, asc, W.q8, W.sc, cd, rem, W.rows, W.cols); off += rem; }
            continue;
        }
        if (rem >= 24) {
            k_df2_q4_gemv<24, true><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, 24, W.rows, W.cols);
            off += 24;
        } else if (rem >= 16) {
            k_df2_q4_gemv<16, true><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, 16, W.rows, W.cols);
            off += 16;
        } else if (rem == 8) {
            k_df2_q4_gemv<8, true><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, 8, W.rows, W.cols);
            off += 8;
        } else if (rem == 7) {
            k_df2_q4_gemv<7, true><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, 7, W.rows, W.cols);
            off += 7;
        } else if (rem == 1) {
            k_df2_q4_gemv<1, true><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, 1, W.rows, W.cols);
            off += 1;
        } else if (rem <= 4) {
            k_df2_q4_gemv<4, false><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, rem, W.rows, W.cols);
            off += rem;
        } else if (rem <= 8) {
            k_df2_q4_gemv<8, false><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, rem, W.rows, W.cols);
            off += rem;
        } else {
            k_df2_q4_gemv<16, false><<<g, 128, 0, st>>>(aq, asc, asum, W.qs, W.sc, cd, rem, W.rows, W.cols);
            off += rem;
        }
    }
}

// Fused Q+K+V for M_TILE rows (8 or 16)
template <int M_TILE>
__global__ void __launch_bounds__(128) k_df2_q4_qkv(const int8_t* __restrict__ Aq,
                                                    const float* __restrict__ Asc,
                                                    const int* __restrict__ Asum,
                                                    const uint8_t* __restrict__ Qqs, const __half* __restrict__ Qsc,
                                                    const uint8_t* __restrict__ Kqs, const __half* __restrict__ Ksc,
                                                    const uint8_t* __restrict__ Vqs, const __half* __restrict__ Vsc,
                                                    float* __restrict__ Qout, float* __restrict__ Kout, float* __restrict__ Vout) {
    constexpr int OQ = NH * HD;
    constexpr int OK = NKV * HD;
    constexpr int K = HID;
    constexpr int nb = K >> 5;
    constexpr size_t a_stride_i4 = K >> 4;
    constexpr size_t sum_stride = nb << 1;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int go = blockIdx.x * 4 + warp;
    if (go >= OQ + OK + OK) return;

    const uint8_t* Wqs;
    const __half* Wsc;
    float* C;
    int O, o;
    if (go < OQ) { Wqs = Qqs; Wsc = Qsc; C = Qout; O = OQ; o = go; }
    else if (go < OQ + OK) { Wqs = Kqs; Wsc = Ksc; C = Kout; O = OK; o = go - OQ; }
    else { Wqs = Vqs; Wsc = Vsc; C = Vout; O = OK; o = go - OQ - OK; }

    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int shift = hi << 2;
    const uint4* qrow = reinterpret_cast<const uint4*>(Wqs + static_cast<size_t>(o) * (K >> 1));
    const __half* srow = Wsc + static_cast<size_t>(o) * nb;

    float acc[M_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) acc[m] = 0.0f;

    for (int b = bi0; b < nb; b += 16) {
        const uint4 qv = __ldcs(qrow + b);
        const float d = __half2float(__ldcs(srow + b));
        const int w0 = static_cast<int>((qv.x >> shift) & 0x0F0F0F0Fu);
        const int w1 = static_cast<int>((qv.y >> shift) & 0x0F0F0F0Fu);
        const int w2 = static_cast<int>((qv.z >> shift) & 0x0F0F0F0Fu);
        const int w3 = static_cast<int>((qv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
            int dot = 0;
            dot = __dp4a(w0, a.x, dot);
            dot = __dp4a(w1, a.y, dot);
            dot = __dp4a(w2, a.z, dot);
            dot = __dp4a(w3, a.w, dot);
            const float val = static_cast<float>(dot - __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx));
            acc[m] = fmaf(val, d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const float tot = df2_warp_sum(acc[m]);
        if (lane == 0) C[static_cast<size_t>(m) * O + o] = tot;
    }
}

template <int M_TILE>
__global__ void __launch_bounds__(128) k_df2_q8_qkv(const int8_t* __restrict__ Aq,
                                                    const float* __restrict__ Asc,
                                                    const int8_t* __restrict__ Qq8, const __half* __restrict__ Qsc,
                                                    const int8_t* __restrict__ Kq8, const __half* __restrict__ Ksc,
                                                    const int8_t* __restrict__ Vq8, const __half* __restrict__ Vsc,
                                                    float* __restrict__ Qout, float* __restrict__ Kout, float* __restrict__ Vout) {
    constexpr int OQ = NH * HD, OK = NKV * HD, K = HID, nb = K >> 5;
    constexpr size_t a_stride_i4 = K >> 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int go = blockIdx.x * 4 + warp;
    if (go >= OQ + OK + OK) return;

    const int8_t* Wq8;
    const __half* Wsc;
    float* C;
    int O, o;
    if (go < OQ) { Wq8 = Qq8; Wsc = Qsc; C = Qout; O = OQ; o = go; }
    else if (go < OQ + OK) { Wq8 = Kq8; Wsc = Ksc; C = Kout; O = OK; o = go - OQ; }
    else { Wq8 = Vq8; Wsc = Vsc; C = Vout; O = OK; o = go - OQ - OK; }

    const int bi0 = lane >> 1, hi = lane & 1;
    const int4* qrow = reinterpret_cast<const int4*>(Wq8 + static_cast<size_t>(o) * K);
    const __half* srow = Wsc + static_cast<size_t>(o) * nb;
    float acc[M_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) acc[m] = 0.0f;

    for (int b = bi0; b < nb; b += 16) {
        const int hidx = (b << 1) | hi;
        const int4 wv = __ldcs(qrow + hidx);
        const float d = __half2float(__ldcs(srow + b));
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
            int dot = 0;
            dot = __dp4a(wv.x, a.x, dot);
            dot = __dp4a(wv.y, a.y, dot);
            dot = __dp4a(wv.z, a.z, dot);
            dot = __dp4a(wv.w, a.w, dot);
            acc[m] = fmaf(static_cast<float>(dot), d * __ldg(Asc + static_cast<size_t>(m) * nb + b), acc[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const float tot = df2_warp_sum(acc[m]);
        if (lane == 0) C[static_cast<size_t>(m) * O + o] = tot;
    }
}

inline void launch_q4_qkv(const Q8Act& qa, const LayerWeights& lw, float* qout, float* kout, float* vout, int M, cudaStream_t st) {
    constexpr int qkv_grid = (NH * HD + 2 * NKV * HD + 3) / 4;
    constexpr int nb = HID >> 5;
    int off = 0;
    while (off < M) {
        const int8_t* aq = qa.q + static_cast<size_t>(off) * HID;
        const float* asc = qa.sc + static_cast<size_t>(off) * nb;
        const int* asum = qa.isum + static_cast<size_t>(off) * (nb << 1);
        float* qd = qout + static_cast<size_t>(off) * (NH * HD);
        float* kd = kout + static_cast<size_t>(off) * (NKV * HD);
        float* vd = vout + static_cast<size_t>(off) * (NKV * HD);
        if (lw.q.q8) {
            if (M - off >= 16) { k_df2_q8_qkv<16><<<qkv_grid, 128, 0, st>>>(aq, asc, lw.q.q8, lw.q.sc, lw.k.q8, lw.k.sc, lw.v.q8, lw.v.sc, qd, kd, vd); off += 16; }
            else { k_df2_q8_qkv<8><<<qkv_grid, 128, 0, st>>>(aq, asc, lw.q.q8, lw.q.sc, lw.k.q8, lw.k.sc, lw.v.q8, lw.v.sc, qd, kd, vd); off += 8; }
            continue;
        }
        if (M - off >= 24) {
            k_df2_q4_qkv<24><<<qkv_grid, 128, 0, st>>>(aq, asc, asum, lw.q.qs, lw.q.sc, lw.k.qs, lw.k.sc, lw.v.qs, lw.v.sc, qd, kd, vd);
            off += 24;
        } else if (M - off >= 16) {
            k_df2_q4_qkv<16><<<qkv_grid, 128, 0, st>>>(aq, asc, asum, lw.q.qs, lw.q.sc, lw.k.qs, lw.k.sc, lw.v.qs, lw.v.sc, qd, kd, vd);
            off += 16;
        } else {
            k_df2_q4_qkv<8><<<qkv_grid, 128, 0, st>>>(aq, asc, asum, lw.q.qs, lw.q.sc, lw.k.qs, lw.k.sc, lw.v.qs, lw.v.sc, qd, kd, vd);
            off += 8;
        }
    }
}

// Fused Gate+Up+SwiGLU for M_TILE rows (8 or 16)
template <int M_TILE>
__global__ void __launch_bounds__(128) k_df2_q4_gate_up_swiglu(const int8_t* __restrict__ Aq,
                                                               const float* __restrict__ Asc,
                                                               const int* __restrict__ Asum,
                                                               const uint8_t* __restrict__ Gqs, const __half* __restrict__ Gsc,
                                                               const uint8_t* __restrict__ Uqs, const __half* __restrict__ Usc,
                                                               float* __restrict__ act_out) {
    constexpr int K = HID;
    constexpr int nb = K >> 5;
    constexpr size_t a_stride_i4 = K >> 4;
    constexpr size_t sum_stride = nb << 1;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= INTER) return;

    const int bi0 = lane >> 1;
    const int hi = lane & 1;
    const int shift = hi << 2;
    const uint4* gq = reinterpret_cast<const uint4*>(Gqs + static_cast<size_t>(o) * (K >> 1));
    const __half* gs = Gsc + static_cast<size_t>(o) * nb;
    const uint4* uq = reinterpret_cast<const uint4*>(Uqs + static_cast<size_t>(o) * (K >> 1));
    const __half* us = Usc + static_cast<size_t>(o) * nb;

    float ag[M_TILE], au[M_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }

    for (int b = bi0; b < nb; b += 16) {
        const uint4 qgv = __ldcs(gq + b);
        const float dg = __half2float(__ldcs(gs + b));
        const uint4 quv = __ldcs(uq + b);
        const float du = __half2float(__ldcs(us + b));
        const int wg0 = static_cast<int>((qgv.x >> shift) & 0x0F0F0F0Fu);
        const int wg1 = static_cast<int>((qgv.y >> shift) & 0x0F0F0F0Fu);
        const int wg2 = static_cast<int>((qgv.z >> shift) & 0x0F0F0F0Fu);
        const int wg3 = static_cast<int>((qgv.w >> shift) & 0x0F0F0F0Fu);
        const int wu0 = static_cast<int>((quv.x >> shift) & 0x0F0F0F0Fu);
        const int wu1 = static_cast<int>((quv.y >> shift) & 0x0F0F0F0Fu);
        const int wu2 = static_cast<int>((quv.z >> shift) & 0x0F0F0F0Fu);
        const int wu3 = static_cast<int>((quv.w >> shift) & 0x0F0F0F0Fu);
        const int hidx = (b << 1) | hi;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
            const int s8 = __ldg(Asum + static_cast<size_t>(m) * sum_stride + hidx);
            const float asc = __ldg(Asc + static_cast<size_t>(m) * nb + b);
            int dg_dot = 0, du_dot = 0;
            dg_dot = __dp4a(wg0, a.x, dg_dot); du_dot = __dp4a(wu0, a.x, du_dot);
            dg_dot = __dp4a(wg1, a.y, dg_dot); du_dot = __dp4a(wu1, a.y, du_dot);
            dg_dot = __dp4a(wg2, a.z, dg_dot); du_dot = __dp4a(wu2, a.z, du_dot);
            dg_dot = __dp4a(wg3, a.w, dg_dot); du_dot = __dp4a(wu3, a.w, du_dot);
            ag[m] = fmaf(static_cast<float>(dg_dot - s8), dg * asc, ag[m]);
            au[m] = fmaf(static_cast<float>(du_dot - s8), du * asc, au[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const float tg = df2_warp_sum(ag[m]);
        const float tu = df2_warp_sum(au[m]);
        if (lane == 0) act_out[static_cast<size_t>(m) * INTER + o] = df2_silu(tg) * tu;
    }
}

template <int M_TILE>
__global__ void __launch_bounds__(128) k_df2_q8_gate_up_swiglu(const int8_t* __restrict__ Aq,
                                                               const float* __restrict__ Asc,
                                                               const int8_t* __restrict__ Gq8, const __half* __restrict__ Gsc,
                                                               const int8_t* __restrict__ Uq8, const __half* __restrict__ Usc,
                                                               float* __restrict__ act_out) {
    constexpr int K = HID, nb = K >> 5;
    constexpr size_t a_stride_i4 = K >> 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * 4 + warp;
    if (o >= INTER) return;

    const int bi0 = lane >> 1, hi = lane & 1;
    const int4* gq = reinterpret_cast<const int4*>(Gq8 + static_cast<size_t>(o) * K);
    const __half* gs = Gsc + static_cast<size_t>(o) * nb;
    const int4* uq = reinterpret_cast<const int4*>(Uq8 + static_cast<size_t>(o) * K);
    const __half* us = Usc + static_cast<size_t>(o) * nb;

    float ag[M_TILE], au[M_TILE];
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }

    for (int b = bi0; b < nb; b += 16) {
        const int hidx = (b << 1) | hi;
        const int4 wg = __ldcs(gq + hidx);
        const float dg = __half2float(__ldcs(gs + b));
        const int4 wu = __ldcs(uq + hidx);
        const float du = __half2float(__ldcs(us + b));
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int4 a = __ldg(reinterpret_cast<const int4*>(Aq) + static_cast<size_t>(m) * a_stride_i4 + hidx);
            const float asc = __ldg(Asc + static_cast<size_t>(m) * nb + b);
            int dg_dot = 0, du_dot = 0;
            dg_dot = __dp4a(wg.x, a.x, dg_dot); du_dot = __dp4a(wu.x, a.x, du_dot);
            dg_dot = __dp4a(wg.y, a.y, dg_dot); du_dot = __dp4a(wu.y, a.y, du_dot);
            dg_dot = __dp4a(wg.z, a.z, dg_dot); du_dot = __dp4a(wu.z, a.z, du_dot);
            dg_dot = __dp4a(wg.w, a.w, dg_dot); du_dot = __dp4a(wu.w, a.w, du_dot);
            ag[m] = fmaf(static_cast<float>(dg_dot), dg * asc, ag[m]);
            au[m] = fmaf(static_cast<float>(du_dot), du * asc, au[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const float tg = df2_warp_sum(ag[m]);
        const float tu = df2_warp_sum(au[m]);
        if (lane == 0) act_out[static_cast<size_t>(m) * INTER + o] = df2_silu(tg) * tu;
    }
}

inline void launch_q4_gate_up_swiglu(const Q8Act& qa, const LayerWeights& lw, float* act_out, int M, cudaStream_t st) {
    constexpr int mlp_grid = (INTER + 3) / 4;
    constexpr int nb = HID >> 5;
    int off = 0;
    while (off < M) {
        const int8_t* aq = qa.q + static_cast<size_t>(off) * HID;
        const float* asc = qa.sc + static_cast<size_t>(off) * nb;
        const int* asum = qa.isum + static_cast<size_t>(off) * (nb << 1);
        float* ad = act_out + static_cast<size_t>(off) * INTER;
        if (lw.gate.q8) {
            if (M - off >= 16) { k_df2_q8_gate_up_swiglu<16><<<mlp_grid, 128, 0, st>>>(aq, asc, lw.gate.q8, lw.gate.sc, lw.up.q8, lw.up.sc, ad); off += 16; }
            else { k_df2_q8_gate_up_swiglu<8><<<mlp_grid, 128, 0, st>>>(aq, asc, lw.gate.q8, lw.gate.sc, lw.up.q8, lw.up.sc, ad); off += 8; }
            continue;
        }
        if (M - off >= 24) {
            k_df2_q4_gate_up_swiglu<24><<<mlp_grid, 128, 0, st>>>(aq, asc, asum, lw.gate.qs, lw.gate.sc, lw.up.qs, lw.up.sc, ad);
            off += 24;
        } else if (M - off >= 16) {
            k_df2_q4_gate_up_swiglu<16><<<mlp_grid, 128, 0, st>>>(aq, asc, asum, lw.gate.qs, lw.gate.sc, lw.up.qs, lw.up.sc, ad);
            off += 16;
        } else {
            k_df2_q4_gate_up_swiglu<8><<<mlp_grid, 128, 0, st>>>(aq, asc, asum, lw.gate.qs, lw.gate.sc, lw.up.qs, lw.up.sc, ad);
            off += 8;
        }
    }
}

// ------------------------------------------------------------------------------------------- Layer Kernels
// Gather accepted rows from step_taps [NT][tap_rows][HID] -> ctx_taps [N_acc][NT * HID] BF16.
// Grid: (N_acc * NT * HID + 255) / 256, Block: 256.
__global__ void __launch_bounds__(256) k_df2_gather_taps(
    const uint16_t* __restrict__ step_taps,
    size_t tap_stride,
    const int* __restrict__ acc_idx,   // nullptr -> identity 0..N-1
    uint16_t* __restrict__ out,
    int N) {
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    const size_t total = static_cast<size_t>(N) * NT * HID;
    if (i >= total) return;
    const int r = static_cast<int>(i / (NT * HID));
    const int tc = static_cast<int>(i % (NT * HID));
    const int t = tc / HID;
    const int c = tc % HID;
    const int m = acc_idx ? acc_idx[r] : r;
    out[i] = step_taps[static_cast<size_t>(t) * tap_stride + static_cast<size_t>(m) * HID + c];
}

// Fused RMSNorm -> Q8_1 (optional F32 and/or BF16 output). Grid: M, Block: 256.
template <bool WRITE_F32, bool WRITE_BF16, bool IN_BF16 = false>
__global__ void __launch_bounds__(256) k_df2_rmsnorm_q8(const void* __restrict__ x,
                                                        const float* __restrict__ w,
                                                        float* __restrict__ f32_out,
                                                        uint16_t* __restrict__ bf16_out,
                                                        int8_t* __restrict__ q_out,
                                                        float* __restrict__ sc_out,
                                                        int* __restrict__ sum_out,
                                                        float eps) {
    const int m = blockIdx.x, t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    float ss = 0.0f;
    for (int c = t; c < HID; c += 256) {
        const float v = IN_BF16
            ? df2_bf(static_cast<const uint16_t*>(x)[static_cast<size_t>(m) * HID + c])
            : static_cast<const float*>(x)[static_cast<size_t>(m) * HID + c];
        ss = fmaf(v, v, ss);
    }
    ss = df2_warp_sum(ss);
    __shared__ float sw[8];
    if (lane == 0) sw[warp] = ss;
    __syncthreads();
    if (t < 32) {
        float tot = (t < 8) ? sw[t] : 0.0f;
        tot = df2_warp_sum(tot);
        if (t == 0) sw[0] = rsqrtf(tot / static_cast<float>(HID) + eps);
    }
    __syncthreads();
    const float inv = sw[0];
    constexpr int NB = HID >> 5;   // 72 blocks of 32
    for (int b = warp; b < NB; b += 8) {
        const int c = (b << 5) | lane;
        const float xv = IN_BF16
            ? df2_bf(static_cast<const uint16_t*>(x)[static_cast<size_t>(m) * HID + c])
            : static_cast<const float*>(x)[static_cast<size_t>(m) * HID + c];
        const float val = xv * inv * w[c];
        if constexpr (WRITE_F32) f32_out[static_cast<size_t>(m) * HID + c] = val;
        if constexpr (WRITE_BF16) {
            __nv_bfloat16 bh = __float2bfloat16(val);
            bf16_out[static_cast<size_t>(m) * HID + c] = *reinterpret_cast<uint16_t*>(&bh);
        }
        if (q_out) df2_quant_warp_q8(val, static_cast<size_t>(m) * NB + b, lane, q_out, sc_out, sum_out);
    }
}

// Extract only draft rows (b=1..7) from h[S][B][HID] -> final_norm -> out_bf16[S*7][HID] + Q8_1 for hproj.
// Grid: S * K_MAX, Block: 256.
__global__ void __launch_bounds__(256) k_df2_final_norm_q8(const float* __restrict__ h,
                                                           const float* __restrict__ w,
                                                           uint16_t* __restrict__ bf16_out,
                                                           int8_t* __restrict__ q_out,
                                                           float* __restrict__ sc_out,
                                                           int* __restrict__ sum_out,
                                                           float eps) {
    const int idx = blockIdx.x;   // 0 .. S*K_MAX - 1
    const int s = idx / K_MAX;
    const int k = idx % K_MAX;    // 0..6 -> block row k+1
    const int src_row = s * B + (k + 1);
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const float* xr = h + static_cast<size_t>(src_row) * HID;
    float ss = 0.0f;
    for (int c = t; c < HID; c += 256) {
        const float v = xr[c];
        ss = fmaf(v, v, ss);
    }
    ss = df2_warp_sum(ss);
    __shared__ float sw[8];
    if (lane == 0) sw[warp] = ss;
    __syncthreads();
    if (t < 32) {
        float tot = (t < 8) ? sw[t] : 0.0f;
        tot = df2_warp_sum(tot);
        if (t == 0) sw[0] = rsqrtf(tot / static_cast<float>(HID) + eps);
    }
    __syncthreads();
    const float inv = sw[0];
    constexpr int NB = HID >> 5;
    for (int b = warp; b < NB; b += 8) {
        const int c = (b << 5) | lane;
        const float val = xr[c] * inv * w[c];
        __nv_bfloat16 bh = __float2bfloat16(val);
        bf16_out[static_cast<size_t>(idx) * HID + c] = *reinterpret_cast<uint16_t*>(&bh);
        df2_quant_warp_q8(val, static_cast<size_t>(idx) * NB + b, lane, q_out, sc_out, sum_out);
    }
}

// Multi-session embed: S * B rows. Row b==0 -> golden[s], b>0 -> MASK_ID.
// Optional target embed_norm (RMSNorm(embed[tok]) * embed_norm_w). Grid: S * B, Block: 256.
__global__ void __launch_bounds__(256) k_df2_embed(const uint16_t* __restrict__ emb,
                                                   const uint16_t* __restrict__ emb_norm,
                                                   const int* __restrict__ d_gold,
                                                   float* __restrict__ out,
                                                   int use_emb_norm) {
    const int row = blockIdx.x;
    const int s = row / B;
    const int b = row % B;
    const int tok = (b == 0) ? d_gold[s] : MASK_ID;
    const uint16_t* src = emb + static_cast<size_t>(tok) * HID;
    float* dst = out + static_cast<size_t>(row) * HID;
    const int t = threadIdx.x;
    if (!use_emb_norm || emb_norm == nullptr) {
        for (int c = t; c < HID; c += 256) dst[c] = df2_bf(src[c]);
        return;
    }
    const int warp = t >> 5, lane = t & 31;
    float ss = 0.0f;
    for (int c = t; c < HID; c += 256) {
        const float v = df2_bf(src[c]);
        ss = fmaf(v, v, ss);
    }
    ss = df2_warp_sum(ss);
    __shared__ float sw[8];
    if (lane == 0) sw[warp] = ss;
    __syncthreads();
    if (t < 32) {
        float tot = (t < 8) ? sw[t] : 0.0f;
        tot = df2_warp_sum(tot);
        if (t == 0) sw[0] = rsqrtf(tot / static_cast<float>(HID) + RMS_EPS);
    }
    __syncthreads();
    const float inv = sw[0];
    for (int c = t; c < HID; c += 256) dst[c] = df2_bf(src[c]) * inv * df2_bf(emb_norm[c]);
}

// Multi-session side-0 prepare conv + Q8_1 quantization directly in registers.
// Grid: (HID / 256 = 9, S * B), Block: 256. Note: tap m-1 stays within each session's B-row block!
__global__ void __launch_bounds__(256) k_df2_conv0_q8(const float* __restrict__ x,
                                                      const float* __restrict__ dyn,
                                                      const float* __restrict__ base,
                                                      int8_t* __restrict__ q_out,
                                                      float* __restrict__ sc_out,
                                                      int* __restrict__ sum_out) {
    const int row = blockIdx.y;
    const int m = row % B;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int g = c / GS;
    const float* drow = dyn + static_cast<size_t>(row) * 4 * NG;
    const float k0 = base[c] + drow[g];
    float y = k0 * x[static_cast<size_t>(row) * HID + c];
    if (m > 0) {
        const float k1 = base[HID + c] + drow[NG + g];
        y = fmaf(k1, x[static_cast<size_t>(row - 1) * HID + c], y);
    }
    const size_t blk = static_cast<size_t>(row) * (HID >> 5) + blockIdx.x * 8 + warp;
    df2_quant_warp_q8(y, blk, lane, q_out, sc_out, sum_out);
}

// Multi-session side-1 finish conv + residual add: dst[row] += conv1(x[row], x[row-1]).
// Grid: (HID / 256 = 9, S * B), Block: 256.
__global__ void __launch_bounds__(256) k_df2_conv1_add(const float* __restrict__ x,
                                                       const float* __restrict__ dyn,
                                                       const float* __restrict__ base,
                                                       float* __restrict__ dst) {
    const int row = blockIdx.y;
    const int m = row % B;
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int g = c / GS;
    const float* drow = dyn + (static_cast<size_t>(row) * 4 + 2) * NG;
    const float* brow = base + static_cast<size_t>(2) * HID;
    const float k0 = brow[c] + drow[g];
    float y = k0 * x[static_cast<size_t>(row) * HID + c];
    if (m > 0) {
        const float k1 = brow[HID + c] + drow[NG + g];
        y = fmaf(k1, x[static_cast<size_t>(row - 1) * HID + c], y);
    }
    dst[static_cast<size_t>(row) * HID + c] += y;
}

__device__ __forceinline__ void df2_rope_cs(int pos, int d, float& c, float& s) {
    const float inv = __expf(-static_cast<float>(d & 63) * (2.0f / static_cast<float>(HD) * LOG_ROPE_THETA));
    __sincosf(static_cast<float>(pos) * inv, &s, &c);
}

// Scatter-write context K (norm + RoPE) and V into per-session FP16 ring [slot][L][NKV][RING][HD].
// Grid: N_acc * NKV, Block: 128.
template <bool IN_BF16>
__global__ void __launch_bounds__(128) k_df2_ctx_kv_scatter(
    const void* __restrict__ k_in,
    const void* __restrict__ v_in,
    const float* __restrict__ k_norm_w,
    const int* __restrict__ rw_slot,
    const int* __restrict__ rw_pos,
    const int* __restrict__ acc_idx,   // nullptr -> identity
    __half* __restrict__ kring_base,   // [slots][L][NKV][RING][HD]
    __half* __restrict__ vring_base,
    int layer_idx, float eps) {
    const int idx = blockIdx.x;
    const int r = idx / NKV;
    const int h = idx % NKV;
    const int d = threadIdx.x;
    const int src_m = acc_idx ? acc_idx[r] : r;
    const int slot = rw_slot[src_m];
    const int pos = rw_pos[src_m];

    const size_t ioff = (static_cast<size_t>(r) * NKV + h) * HD + d;
    const float kv = IN_BF16 ? df2_bf(static_cast<const uint16_t*>(k_in)[ioff]) : static_cast<const float*>(k_in)[ioff];
    const float vv = IN_BF16 ? df2_bf(static_cast<const uint16_t*>(v_in)[ioff]) : static_cast<const float*>(v_in)[ioff];

    float ss = df2_warp_sum(kv * kv);
    __shared__ float sw[4], rot[128];
    if ((d & 31) == 0) sw[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / static_cast<float>(HD) + eps);
    const float kn = kv * inv * k_norm_w[d];
    rot[d] = kn;
    __syncthreads();

    float cs, sn;
    df2_rope_cs(pos, d, cs, sn);
    const float o = rot[d ^ 64];
    const float krot = (d < 64) ? (kn * cs - o * sn) : (kn * cs + o * sn);

    const int slot_pos = pos & (RING - 1);
    const size_t layer_stride = static_cast<size_t>(NKV) * RING * HD;
    const size_t slot_stride = static_cast<size_t>(L) * layer_stride;
    const size_t roff = static_cast<size_t>(slot) * slot_stride + static_cast<size_t>(layer_idx) * layer_stride
                      + (static_cast<size_t>(h) * RING + slot_pos) * HD + d;
    kring_base[roff] = __float2half_rn(krot);
    vring_base[roff] = __float2half_rn(vv);
}

// Multi-session Q norm + RoPE. Grid: S * B * NH, Block: 128.
__global__ void __launch_bounds__(128) k_df2_q_norm_rope(const float* __restrict__ q_in,
                                                         const float* __restrict__ q_norm_w,
                                                         float* __restrict__ q_out,
                                                         const int* __restrict__ d_q0,
                                                         float eps) {
    const int idx = blockIdx.x;
    const int row = idx / NH;
    const int h = idx % NH;
    const int s = row / B;
    const int m = row % B;
    const int q0 = d_q0[s];
    const int d = threadIdx.x;
    const float* qr = q_in + (static_cast<size_t>(row) * NH + h) * HD;

    const float qv = qr[d];
    float ss = df2_warp_sum(qv * qv);
    __shared__ float sw[4], rot[128];
    if ((d & 31) == 0) sw[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / static_cast<float>(HD) + eps);
    const float qn = qv * inv * q_norm_w[d];
    rot[d] = qn;
    __syncthreads();

    float cs, sn;
    df2_rope_cs(q0 + m, d, cs, sn);
    const float o = rot[d ^ 64];
    q_out[(static_cast<size_t>(row) * NH + h) * HD + d] = (d < 64) ? (qn * cs - o * sn) : (qn * cs + o * sn);
}

// Multi-session block K norm + RoPE into isolated scratch kblk/vblk [S][NKV][B][HD]. Grid: S * B * NKV, Block: 128.
__global__ void __launch_bounds__(128) k_df2_blk_kv(const float* __restrict__ k_in,
                                                    const float* __restrict__ v_in,
                                                    const float* __restrict__ k_norm_w,
                                                    __half* __restrict__ kblk,
                                                    __half* __restrict__ vblk,
                                                    const int* __restrict__ d_q0,
                                                    float eps) {
    const int idx = blockIdx.x;
    const int row = idx / NKV;
    const int h = idx % NKV;
    const int s = row / B;
    const int m = row % B;
    const int q0 = d_q0[s];
    const int d = threadIdx.x;
    const float* kr = k_in + (static_cast<size_t>(row) * NKV + h) * HD;
    const float* vr = v_in + (static_cast<size_t>(row) * NKV + h) * HD;

    const float kv = kr[d];
    float ss = df2_warp_sum(kv * kv);
    __shared__ float sw[4], rot[128];
    if ((d & 31) == 0) sw[d >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sw[0] + sw[1] + sw[2] + sw[3]) / static_cast<float>(HD) + eps);
    const float kn = kv * inv * k_norm_w[d];
    rot[d] = kn;
    __syncthreads();

    float cs, sn;
    df2_rope_cs(q0 + m, d, cs, sn);
    const float o = rot[d ^ 64];
    const float krot = (d < 64) ? (kn * cs - o * sn) : (kn * cs + o * sn);

    const size_t boff = ((static_cast<size_t>(s) * NKV + h) * B + m) * HD + d;
    kblk[boff] = __float2half_rn(krot);
    vblk[boff] = __float2half_rn(vr[d]);
}

// Multi-session Split-KV bidirectional B=8 attention. Grid: (NH, ATTN_SPLIT, S), Block: 128.
__global__ void __launch_bounds__(128) k_df2_attn_split(const float* __restrict__ qf,
                                                        const __half* __restrict__ kring_base,
                                                        const __half* __restrict__ vring_base,
                                                        const __half* __restrict__ kblk,
                                                        const __half* __restrict__ vblk,
                                                        const int* __restrict__ d_slot,
                                                        const int* __restrict__ d_q0,
                                                        const int* __restrict__ d_lo,
                                                        int layer_idx,
                                                        float* __restrict__ apart) {
    const int h = blockIdx.x;
    const int sp = blockIdx.y;
    const int s = blockIdx.z;
    const int d = threadIdx.x;
    const int slot = d_slot[s];
    const int q0 = d_q0[s];
    const int ctx_lo = d_lo[s];
    const int kh = h / GQA;
    constexpr float scale = 0.08838834764831845f;

    float qv[B];
    #pragma unroll
    for (int m = 0; m < B; ++m) qv[m] = qf[(static_cast<size_t>(s * B + m) * NH + h) * HD + d];

    float rm[B], rl[B], ro[B];
    #pragma unroll
    for (int m = 0; m < B; ++m) { rm[m] = -INFINITY; rl[m] = 0.0f; ro[m] = 0.0f; }

    __shared__ float sw[B][4];
    auto step_key = [&](const __half* kp, const __half* vp, int kpos, bool is_blk) {
        const float kf = __half2float(kp[d]);
        const float vf = __half2float(vp[d]);
        #pragma unroll
        for (int m = 0; m < B; ++m) {
            const float dot = df2_warp_sum(qv[m] * kf);
            if ((d & 31) == 0) sw[m][d >> 5] = dot;
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < B; ++m) {
            if (!is_blk && (q0 + m - kpos) >= WIN) continue;
            const float sc = (sw[m][0] + sw[m][1] + sw[m][2] + sw[m][3]) * scale;
            const float nm = fmaxf(rm[m], sc);
            const float a = (rm[m] == -INFINITY) ? 0.0f : __expf(rm[m] - nm);
            const float b = __expf(sc - nm);
            ro[m] = ro[m] * a + vf * b;
            rl[m] = rl[m] * a + b;
            rm[m] = nm;
        }
        __syncthreads();
    };

    const int lo = max(ctx_lo, q0 - WIN + 1);
    const int nctx = max(0, q0 - lo);
    const int c_per = (nctx + ATTN_SPLIT - 1) / ATTN_SPLIT;
    const int p0 = lo + sp * c_per;
    const int p1 = min(q0, p0 + c_per);

    const size_t layer_stride = static_cast<size_t>(NKV) * RING * HD;
    const size_t base_off = (static_cast<size_t>(slot) * L + layer_idx) * layer_stride + static_cast<size_t>(kh) * RING * HD;
    const __half* kh_ring = kring_base + base_off;
    const __half* vh_ring = vring_base + base_off;

    for (int p = p0; p < p1; ++p) {
        const int r = p & (RING - 1);
        step_key(kh_ring + static_cast<size_t>(r) * HD, vh_ring + static_cast<size_t>(r) * HD, p, false);
    }
    if (sp == 0) {
        const size_t blk_off = (static_cast<size_t>(s) * NKV + kh) * B * HD;
        const __half* kh_blk = kblk + blk_off;
        const __half* vh_blk = vblk + blk_off;
        for (int j = 0; j < B; ++j) {
            step_key(kh_blk + static_cast<size_t>(j) * HD, vh_blk + static_cast<size_t>(j) * HD, q0 + j, true);
        }
    }

    float* dst = apart + ((((static_cast<size_t>(s) * NH + h) * ATTN_SPLIT + sp) * B) * (HD + 2));
    #pragma unroll
    for (int m = 0; m < B; ++m) {
        float* row = dst + m * (HD + 2);
        row[d] = ro[m];
        if (d == 0) { row[HD] = rm[m]; row[HD + 1] = rl[m]; }
    }
}

// Multi-session attention merge. Grid: (B * NH, S), Block: 128.
__global__ void __launch_bounds__(128) k_df2_attn_merge(const float* __restrict__ apart, float* __restrict__ ao) {
    const int idx = blockIdx.x;
    const int s = blockIdx.y;
    const int m = idx / NH;
    const int h = idx % NH;
    const int d = threadIdx.x;

    float gm = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
        const float* row = apart + ((((static_cast<size_t>(s) * NH + h) * ATTN_SPLIT + sp) * B + m) * (HD + 2));
        gm = fmaxf(gm, row[HD]);
    }
    float gl = 0.0f, go = 0.0f;
    for (int sp = 0; sp < ATTN_SPLIT; ++sp) {
        const float* row = apart + ((((static_cast<size_t>(s) * NH + h) * ATTN_SPLIT + sp) * B + m) * (HD + 2));
        const float sm = row[HD];
        if (sm > -INFINITY) {
            const float w = __expf(sm - gm);
            gl += row[HD + 1] * w;
            go += row[d] * w;
        }
    }
    ao[(static_cast<size_t>(s * B + m) * NH + h) * HD + d] = (gl > 0.0f) ? (go / gl) : 0.0f;
}

// ------------------------------------------------------------------------------------------- Fast 2-bit Head2 Screen + Exact BF16 Rescore
__device__ __forceinline__ int h2_swz(int u) { return u ^ ((u >> 3) & 7); }

__global__ void __launch_bounds__(256) k_df2_lm_to_i2(
    const uint16_t* __restrict__ W,
    uint4* __restrict__ codes,
    __half2* __restrict__ scales,
    int vr) {
    const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= static_cast<long long>(vr) * H2_NCH) return;
    const int row = static_cast<int>(id / H2_NCH);
    const int c = static_cast<int>(id % H2_NCH);
    const uint16_t* src = W + static_cast<size_t>(row) * HID + c * 64;

    uint32_t w[4] = {0u, 0u, 0u, 0u};
    __half s16[2];
    #pragma unroll
    for (int gg = 0; gg < 2; ++gg) {
        float v[32];
        float amax = 0.0f;
        const uint4* p = reinterpret_cast<const uint4*>(src + gg * 32);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 qv = p[j];
            const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&qv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __bfloat1622float2(bp[k]);
                v[8 * j + 2 * k] = f.x;
                v[8 * j + 2 * k + 1] = f.y;
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
        s16[gg] = __float2half_rn(__fdiv_rn(amax, 1.5f));
        const float s = __half2float(s16[gg]);
        #pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            const uint32_t q = (s != 0.0f)
                ? static_cast<uint32_t>(min(max(__float2int_rd(__fadd_rn(__fdiv_rn(v[kk], s), 2.0f)), 0), 3))
                : 2u;
            const int e = gg * 32 + kk;
            const int word = e >> 4;
            const int i = e & 15;
            w[word] |= q << (2 * (i >> 1) + 16 * (i & 1));
        }
    }
    codes[id] = make_uint4(w[0], w[1], w[2], w[3]);
    scales[id] = __halves2half2(s16[0], s16[1]);
}

// 4 lanes per vocab row (8 rows/warp, 64 rows/CTA): 36 chunks / 4 lanes = 9 exact iterations, zero wasted lanes!
template <int MAX_M>
__global__ void __launch_bounds__(256) k_df2_head2_screen(
    const uint4* __restrict__ codes,
    const __half2* __restrict__ scales,
    const uint16_t* __restrict__ x,
    float* __restrict__ y,
    int M, int vr) {
    extern __shared__ uint4 xs4_raw[];
    auto xs4 = reinterpret_cast<uint4 (*)[H2_K8]>(xs4_raw);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub_r = lane >> 2;        // 0..7 (8 vocab rows per warp)
    const int sub_l = lane & 3;         // 0..3 (4 lanes per vocab row)

    const int tot_u4 = M * H2_K8;
    for (int i = tid; i < tot_u4; i += 256) {
        const int m = i / H2_K8;
        const int u = i % H2_K8;
        const uint4 xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * HID)[u];
        const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        uint4 hv;
        __half2* hp = reinterpret_cast<__half2*>(&hv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(bp[k]);
            hp[k] = __floats2half2_rn(f.x * 0.25f, f.y * 0.25f);
        }
        xs4[m][h2_swz(u)] = hv;
    }
    __syncthreads();

    const __half2 off = __float2half2_rn(1027.0f);
    for (int v = blockIdx.x * 64 + warp * 8 + sub_r; v < vr; v += gridDim.x * 64) {
        const uint4* crow = codes + static_cast<size_t>(v) * H2_NCH;
        const __half2* srow = scales + static_cast<size_t>(v) * H2_NCH;

        float acc[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const int c = j * 4 + sub_l;
            const uint4 cw = __ldcs(crow + c);
            const __half2 sc = __ldcs(srow + c);
            const uint32_t wd[4] = {cw.x, cw.y, cw.z, cw.w};

            #pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                __half2 p[MAX_M];
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) p[m] = __float2half2_rn(0.0f);

                #pragma unroll
                for (int q = 2 * gg; q < 2 * gg + 2; ++q) {
                    #pragma unroll
                    for (int h8 = 0; h8 < 2; ++h8) {
                        const int u = 8 * c + 2 * q + h8;
                        const int su = h2_swz(u);
                        __half2 wv[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int sh = 2 * (4 * h8 + i);
                            const uint32_t b = ((sh ? (wd[q] >> (sh - 1)) : (wd[q] << 1)) & 0x00060006u) | 0x64006400u;
                            wv[i] = __hsub2(*reinterpret_cast<const __half2*>(&b), off);
                        }
                        #pragma unroll
                        for (int m = 0; m < MAX_M; ++m) {
                            if (m < M) {
                                const uint4 xv = xs4[m][su];
                                const __half2* xp = reinterpret_cast<const __half2*>(&xv);
                                p[m] = __hfma2(wv[0], xp[0], p[m]);
                                p[m] = __hfma2(wv[1], xp[1], p[m]);
                                p[m] = __hfma2(wv[2], xp[2], p[m]);
                                p[m] = __hfma2(wv[3], xp[3], p[m]);
                            }
                        }
                    }
                }
                const float s = 2.0f * (gg ? __high2float(sc) : __low2float(sc));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const float2 f = __half22float2(p[m]);
                        acc[m] = fmaf(s, f.x + f.y, acc[m]);
                    }
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], 2);
                acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], 1);
                if (sub_l == 0) y[static_cast<size_t>(m) * vr + v] = acc[m];
            }
        }
    }
}

// Deterministic top-SL selection from logits y[M, vr] (Grid: M, Block: 1024).
template <int SL>
__global__ void __launch_bounds__(1024) k_df2_top_sl(const float* __restrict__ y, int vr, int* __restrict__ cand) {
    __shared__ unsigned hist[H2_BINS];
    __shared__ float s_mx[32];
    __shared__ int s_cut, s_lt;
    __shared__ int s_n[2][1024];

    const int m = blockIdx.x, t = threadIdx.x;
    const int w = t >> 5, l = t & 31;
    const float* row = y + static_cast<size_t>(m) * vr;
    const int per = (vr + 1023) / 1024;
    const int i0 = min(t * per, vr), i1 = min(i0 + per, vr);

    float mx = -INFINITY;
    for (int i = i0; i < i1; ++i) {
        const float v = row[i];
        if (isfinite(v)) mx = fmaxf(mx, v);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (l == 0) s_mx[w] = mx;
    for (int i = t; i < H2_BINS; i += 1024) hist[i] = 0u;
    __syncthreads();

    mx = s_mx[0];
    #pragma unroll
    for (int i = 1; i < 32; ++i) mx = fmaxf(mx, s_mx[i]);

    auto bin = [&](float v) -> int {
        if (!isfinite(v)) return H2_BINS - 1;
        const float b = (mx - v) * (H2_BINS / H2_RANGE);
        return (b < static_cast<float>(H2_BINS - 1)) ? max(0, static_cast<int>(b)) : (H2_BINS - 1);
    };
    for (int i = i0; i < i1; ++i) atomicAdd(&hist[bin(row[i])], 1u);
    __syncthreads();

    if (w == 0) {
        constexpr int PER = H2_BINS / 32;
        unsigned c = 0;
        for (int i = 0; i < PER; ++i) c += hist[l * PER + i];
        unsigned inc = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned z = __shfl_up_sync(0xffffffffu, inc, o);
            if (l >= o) inc += z;
        }
        const unsigned over = __ballot_sync(0xffffffffu, inc >= static_cast<unsigned>(SL));
        const int Lbin = over ? (__ffs(over) - 1) : 31;
        if (l == Lbin) {
            unsigned cum = inc - c;
            int cut = Lbin * PER + PER - 1;
            for (int i = 0; i < PER; ++i) {
                if (cum + hist[Lbin * PER + i] >= static_cast<unsigned>(SL)) { cut = Lbin * PER + i; break; }
                cum += hist[Lbin * PER + i];
            }
            s_cut = cut;
            s_lt = static_cast<int>(cum);
        }
    }
    __syncthreads();

    const int cut = s_cut, lt = s_lt;
    int nlt = 0, neq = 0;
    for (int i = i0; i < i1; ++i) {
        const int b = bin(row[i]);
        nlt += (b < cut);
        neq += (b == cut);
    }
    s_n[0][t] = nlt;
    s_n[1][t] = neq;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        const int a = (t >= o) ? s_n[0][t - o] : 0;
        const int b = (t >= o) ? s_n[1][t - o] : 0;
        __syncthreads();
        s_n[0][t] += a;
        s_n[1][t] += b;
        __syncthreads();
    }
    int plt = s_n[0][t] - nlt;
    int peq = lt + s_n[1][t] - neq;
    int* out = cand + static_cast<size_t>(m) * SL;
    for (int i = i0; i < i1; ++i) {
        const int b = bin(row[i]);
        if (b < cut) out[plt++] = i;
        else if (b == cut && peq < SL) out[peq++] = i;
    }
}

// Exact BF16 rescore of SL=64 candidates -> sorted top-16 Cand16 per row. Grid: M, Block: 256 (8 warps).
__global__ void __launch_bounds__(256) k_df2_rescore_bf16_top16(
    const uint16_t* __restrict__ W_bf16,
    const uint16_t* __restrict__ x_bf16,
    const int* __restrict__ cand64,
    Cand16* __restrict__ out16) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ uint4 xs[H2_K8];
    const uint4* xsrc = reinterpret_cast<const uint4*>(x_bf16 + static_cast<size_t>(m) * HID);
    for (int i = tid; i < H2_K8; i += 256) xs[i] = xsrc[i];
    __syncthreads();

    __shared__ float sc[H2_SL];
    __shared__ int cid[H2_SL];

    // 8 warps handle 64 candidates (8 candidates per warp)
    for (int j = warp; j < H2_SL; j += 8) {
        const int v = cand64[static_cast<size_t>(m) * H2_SL + j];
        const uint4* wrow = reinterpret_cast<const uint4*>(W_bf16 + static_cast<size_t>(v) * HID);
        float acc = 0.0f;
        for (int u = lane; u < H2_K8; u += 32) {
            const uint4 wv = __ldcs(wrow + u);
            const uint4 xv = xs[u];
            const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 wf = __bfloat1622float2(wp[k]);
                const float2 xf = __bfloat1622float2(xp[k]);
                acc = fmaf(wf.x, xf.x, acc);
                acc = fmaf(wf.y, xf.y, acc);
            }
        }
        const float tot = df2_warp_sum(acc);
        if (lane == 0) {
            sc[j] = tot;
            cid[j] = v;
        }
    }
    __syncthreads();

    // Rank top-16 among 64 candidates in parallel (threads 0..63)
    if (tid < H2_SL) {
        const float li = sc[tid];
        const int ii = cid[tid];
        int r = 0;
        for (int j = 0; j < H2_SL; ++j) {
            if (sc[j] > li || (sc[j] == li && cid[j] < ii)) ++r;
        }
        if (r < TOPK) out16[static_cast<size_t>(m) * TOPK + r] = Cand16{li, ii};
    }
}

__global__ void __launch_bounds__(256) k_df2_sort_top16_from_y(
    const float* __restrict__ y, int vr, const int* __restrict__ cand16, Cand16* __restrict__ out16) {
    const int m = blockIdx.x, t = threadIdx.x;
    __shared__ float sv[TOPK];
    __shared__ int si[TOPK];
    if (t < TOPK) {
        const int idx = cand16[static_cast<size_t>(m) * TOPK + t];
        sv[t] = y[static_cast<size_t>(m) * vr + idx];
        si[t] = idx;
    }
    __syncthreads();
    if (t < TOPK) {
        const float li = sv[t];
        const int ii = si[t];
        int r = 0;
        for (int j = 0; j < TOPK; ++j) {
            if (sv[j] > li || (sv[j] == li && si[j] < ii)) ++r;
        }
        out16[static_cast<size_t>(m) * TOPK + r] = Cand16{li, ii};
    }
}

// ------------------------------------------------------------------------------------------- Multi-session Candidate Selector + Viterbi DP
// Grid: S CTAs (1 CTA per session), Block: 256.
// Implements selector_softcap_mode == "after_correction" with final_logit_softcapping = 30.0.
__global__ void __launch_bounds__(256) k_df2_select_multi(
    const Cand16* __restrict__ cands,        // [S][K_MAX][TOPK]
    const int* __restrict__ d_gold,          // [S]
    const float* __restrict__ hp,            // [S][K_MAX][SELR]
    Q4Codebook pred,                         // [VOCAB][SELR] Q4_0 / Q8_0 / BF16
    Q4Codebook succ,                         // [VOCAB][SELR] Q4_0 / Q8_0 / BF16
    int K, float sel_alpha, int use_dp, DpCal cal,
    int* __restrict__ tok_out) {             // [S][K]
    const int sess = blockIdx.x;
    const int t = threadIdx.x, w = t >> 5, l = t & 31;

    const Cand16* scands = cands + static_cast<size_t>(sess) * K_MAX * TOPK;
    const float* shp = hp + static_cast<size_t>(sess) * K_MAX * SELR;
    int* sout = tok_out + static_cast<size_t>(sess) * K;

    __shared__ float s_l[K_MAX][TOPK];
    __shared__ int s_i[K_MAX][TOPK];
    __shared__ __nv_bfloat16 s_pred[TOPK][SELR];
    __shared__ __nv_bfloat16 s_succ[TOPK][SELR];
    __shared__ float s_hp[SELR];

    if (t < K * TOPK) {
        const int s = t / TOPK, k = t % TOPK;
        s_l[s][k] = scands[s * TOPK + k].logit;
        s_i[s][k] = scands[s * TOPK + k].id;
    }
    __syncthreads();

    auto load_cb1 = [&](const Q4Codebook& cb, int vid, __nv_bfloat16 dst[SELR]) {
        if (vid < 0 || vid >= cb.vocab) {
            dst[t] = __float2bfloat16_rn(0.0f);
            return;
        }
        if (cb.qs != nullptr) {
            const int blk = t >> 5;
            const int rem = t & 31;
            const int j = rem & 15;
            const uint8_t packed = __ldg(cb.qs + static_cast<size_t>(vid) * (SELR >> 1) + blk * 16 + j);
            const float d = __half2float(__ldg(cb.sc + static_cast<size_t>(vid) * (SELR >> 5) + blk));
            const int q = (rem < 16) ? static_cast<int>(packed & 15u) : static_cast<int>(packed >> 4);
            dst[t] = __float2bfloat16_rn(static_cast<float>(q - 8) * d);
        } else if (cb.q8 != nullptr) {
            const int blk = t >> 5;
            const int8_t q = __ldg(cb.q8 + static_cast<size_t>(vid) * SELR + t);
            const float d = __half2float(__ldg(cb.sc + static_cast<size_t>(vid) * (SELR >> 5) + blk));
            dst[t] = __float2bfloat16_rn(static_cast<float>(q) * d);
        } else {
            dst[t] = __ldg(cb.bf16 + static_cast<size_t>(vid) * SELR + t);
        }
    };
    auto load_cb16 = [&](const Q4Codebook& cb, const int* ids, __nv_bfloat16 dst[TOPK][SELR]) {
        if (cb.qs != nullptr) {
            for (int idx = t; idx < TOPK * SELR; idx += 256) {
                const int row = idx >> 8;
                const int col = idx & 255;
                const int vid = ids[row];
                if (vid < 0 || vid >= cb.vocab) { dst[row][col] = __float2bfloat16_rn(0.0f); continue; }
                const int blk = col >> 5;
                const int rem = col & 31;
                const int j = rem & 15;
                const uint8_t packed = __ldg(cb.qs + static_cast<size_t>(vid) * (SELR >> 1) + blk * 16 + j);
                const float d = __half2float(__ldg(cb.sc + static_cast<size_t>(vid) * (SELR >> 5) + blk));
                const int q = (rem < 16) ? static_cast<int>(packed & 15u) : static_cast<int>(packed >> 4);
                dst[row][col] = __float2bfloat16_rn(static_cast<float>(q - 8) * d);
            }
        } else if (cb.q8 != nullptr) {
            for (int idx = t; idx < TOPK * SELR; idx += 256) {
                const int row = idx >> 8;
                const int col = idx & 255;
                const int vid = ids[row];
                if (vid < 0 || vid >= cb.vocab) { dst[row][col] = __float2bfloat16_rn(0.0f); continue; }
                const int blk = col >> 5;
                const int8_t q = __ldg(cb.q8 + static_cast<size_t>(vid) * SELR + col);
                const float d = __half2float(__ldg(cb.sc + static_cast<size_t>(vid) * (SELR >> 5) + blk));
                dst[row][col] = __float2bfloat16_rn(static_cast<float>(q) * d);
            }
        } else {
            for (int idx = t; idx < TOPK * (SELR / 8); idx += 256) {
                const int row = idx >> 5;
                const int vec = idx & 31;
                const int vid = ids[row];
                if (vid < 0 || vid >= cb.vocab) {
                    reinterpret_cast<uint4*>(dst[row])[vec] = make_uint4(0, 0, 0, 0);
                    continue;
                }
                const uint4* src = reinterpret_cast<const uint4*>(cb.bf16 + static_cast<size_t>(vid) * SELR);
                reinterpret_cast<uint4*>(dst[row])[vec] = __ldg(src + vec);
            }
        }
    };
    auto bilinear_dot_smem = [&](int p_slot, int c_slot) -> float {
        float acc = 0.0f;
        const __nv_bfloat16* pr = s_pred[p_slot];
        const __nv_bfloat16* sr = s_succ[c_slot];
        #pragma unroll
        for (int i = l; i < SELR; i += 32) {
            acc = fmaf(__bfloat162float(pr[i]) * s_hp[i], __bfloat162float(sr[i]), acc);
        }
        return df2_warp_sum(acc);
    };
    auto softcap30 = [](float v) -> float {
        return SOFTCAP * tanhf(v * (1.0f / SOFTCAP));
    };

    if (use_dp == 0) {
        __shared__ float sc[TOPK];
        __shared__ int s_prev;
        if (t == 0) s_prev = d_gold[sess];
        __syncthreads();
        for (int s = 0; s < K; ++s) {
            s_hp[t] = shp[static_cast<size_t>(s) * SELR + t];
            load_cb1(pred, s_prev, s_pred[0]);
            load_cb16(succ, s_i[s], s_succ);
            __syncthreads();
            for (int cc = w; cc < TOPK; cc += 8) {
                const float dot = bilinear_dot_smem(0, cc);
                if (l == 0) sc[cc] = softcap30(s_l[s][cc] + sel_alpha * dot);
            }
            __syncthreads();
            if (t == 0) {
                int bk = 0;
                float best = sc[0];
                for (int k = 1; k < TOPK; ++k) {
                    if (sc[k] > best) { best = sc[k]; bk = k; }
                }
                s_prev = s_i[s][bk];
                sout[s] = s_prev;
            }
            __syncthreads();
        }
        return;
    }

    // Calibrated Expected-Reach Viterbi DP over K x 16 lattice
    __shared__ float s_p0[TOPK];
    __shared__ float s_p[K_MAX][TOPK][TOPK];
    __shared__ float s_V[2][TOPK];
    __shared__ int s_pi[K_MAX][TOPK];

    {
        __shared__ float sc0[TOPK];
        s_hp[t] = shp[t];
        load_cb1(pred, d_gold[sess], s_pred[0]);
        load_cb16(succ, s_i[0], s_succ);
        __syncthreads();
        for (int cc = w; cc < TOPK; cc += 8) {
            const float dot = bilinear_dot_smem(0, cc);
            if (l == 0) sc0[cc] = softcap30(s_l[0][cc] + sel_alpha * dot);
        }
        __syncthreads();
        if (t == 0) {
            float m1 = -INFINITY, m2 = -INFINITY;
            for (int k = 0; k < TOPK; ++k) {
                const float v = sc0[k];
                if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v;
            }
            const float g = cal.gamma[0];
            const float z_oov = cal.beta0[0] + cal.beta1[0] * m1 + cal.beta2[0] * fmaxf(0.0f, m1 - m2);
            const float mx = fmaxf(g * m1, z_oov);
            float z = expf(z_oov - mx);
            for (int k = 0; k < TOPK; ++k) z += expf(g * sc0[k] - mx);
            const float inv_z = 1.0f / z;
            for (int k = 0; k < TOPK; ++k) s_p0[k] = expf(g * sc0[k] - mx) * inv_z;
        }
        __syncthreads();
    }
    for (int s = 1; s < K; ++s) {
        s_hp[t] = shp[static_cast<size_t>(s) * SELR + t];
        load_cb16(pred, s_i[s - 1], s_pred);
        load_cb16(succ, s_i[s], s_succ);
        __syncthreads();
        for (int pair = w; pair < TOPK * TOPK; pair += 8) {
            const int p = pair >> 4, c = pair & 15;
            const float dot = bilinear_dot_smem(p, c);
            if (l == 0) s_p[s][p][c] = softcap30(s_l[s][c] + sel_alpha * dot);
        }
        __syncthreads();
        if (t < TOPK) {
            const int p = t;
            float m1 = -INFINITY, m2 = -INFINITY;
            for (int c = 0; c < TOPK; ++c) {
                const float v = s_p[s][p][c];
                if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v;
            }
            const float g = cal.gamma[s];
            const float z_oov = cal.beta0[s] + cal.beta1[s] * m1 + cal.beta2[s] * fmaxf(0.0f, m1 - m2);
            const float mx = fmaxf(g * m1, z_oov);
            float z = expf(z_oov - mx);
            for (int c = 0; c < TOPK; ++c) z += expf(g * s_p[s][p][c] - mx);
            const float inv_z = 1.0f / z;
            for (int c = 0; c < TOPK; ++c) s_p[s][p][c] = expf(g * s_p[s][p][c] - mx) * inv_z;
        }
        __syncthreads();
    }
    if (t < TOPK) s_V[0][t] = 0.0f;
    __syncthreads();
    int cur = 0;
    for (int s = K - 1; s >= 1; --s) {
        const int nxt = cur ^ 1;
        if (t < TOPK) {
            const int p = t;
            float best = -1.0f;
            int bc = 0;
            for (int c = 0; c < TOPK; ++c) {
                const float val = s_p[s][p][c] * (1.0f + s_V[cur][c]);
                if (val > best) { best = val; bc = c; }
            }
            s_V[nxt][p] = best;
            s_pi[s][p] = bc;
        }
        __syncthreads();
        cur = nxt;
    }
    if (t == 0) {
        float best = -1.0f;
        int c = 0;
        for (int k = 0; k < TOPK; ++k) {
            const float val = s_p0[k] * (1.0f + s_V[cur][k]);
            if (val > best) { best = val; c = k; }
        }
        sout[0] = s_i[0][c];
        for (int s = 1; s < K; ++s) {
            c = s_pi[s][c];
            sout[s] = s_i[s][c];
        }
    }
}

// ------------------------------------------------------------------------------------------- Ink2 DFlash2 Drafter Class
struct Ink2DFlash2 {
    bool loaded = false;
    int max_slots = 0;
    int max_commit_rows = 0;
    int valid_vocab = 200058;
    int use_emb_norm = 1;
    int use_head2 = 1;
    int use_q8 = 1;
    int use_dp = 0;
    int use_graph = 1;
    float sel_alpha = 1.0f;
    DpCal dp_cal;

    // Device weights
    Q4Mat fc, hproj;
    uint16_t* fc_bf16 = nullptr;
    float* hidden_norm = nullptr;
    float* final_norm = nullptr;
    Q4Codebook pred_cb;
    Q4Codebook succ_cb;
    LayerWeights ly[L];

    // 2-bit Head2 shortlist weights built from target lm_head[0..valid_vocab-1]
    uint4* h2_codes = nullptr;
    __half2* h2_scales = nullptr;

    // Context KV rings: [max_slots][L][NKV][RING][HD]
    __half* kring = nullptr;
    __half* vring = nullptr;
    std::vector<int> slot_ctx_lo;

    // Commit scratch (supports up to max_commit_rows for fast cuBLAS prefill commit)
    uint16_t* ctx_taps_bf16 = nullptr;   // [max_commit_rows][NT * HID]
    uint16_t* ctx_fc_bf16 = nullptr;     // [max_commit_rows][HID]
    uint16_t* ctx_norm_bf16 = nullptr;   // [max_commit_rows][HID]
    uint16_t* ctx_k_bf16 = nullptr;      // [max_commit_rows][NKV * HD]
    uint16_t* ctx_v_bf16 = nullptr;      // [max_commit_rows][NKV * HD]
    float* ctx_fc_f32 = nullptr;         // [B][HID]
    float* ctx_k_f32 = nullptr;          // [B][NKV * HD]
    float* ctx_v_f32 = nullptr;          // [B][NKV * HD]
    int* d_acc_idx = nullptr;            // [max_commit_rows]
    int* h_acc_idx_pin = nullptr;

    // Multi-session block forward scratch (up to max_slots sessions)
    float* h = nullptr;                  // [max_slots * B][HID]
    float* xn = nullptr;                 // [max_slots * B][HID]
    float* dyn = nullptr;                // [max_slots * B][CPO]
    float* q_buf = nullptr;              // [max_slots * B][NH * HD]
    float* qf = nullptr;                 // [max_slots * B][NH * HD]
    float* k_buf = nullptr;              // [max_slots * B][NKV * HD]
    float* v_buf = nullptr;              // [max_slots * B][NKV * HD]
    __half* kblk = nullptr;              // [max_slots][NKV][B][HD]
    __half* vblk = nullptr;              // [max_slots][NKV][B][HD]
    float* apart = nullptr;              // [max_slots][NH][ATTN_SPLIT][B][HD + 2]
    float* ao = nullptr;                 // [max_slots * B][NH * HD]
    float* o_buf = nullptr;              // [max_slots * B][HID]
    float* act_buf = nullptr;            // [max_slots * B][INTER]
    Q8Act qa;                            // [max_slots * B * max(NT*HID, INTER)]
    uint16_t* out_bf16 = nullptr;        // [max_slots * K_MAX][HID]
    float* hp = nullptr;                 // [max_slots * K_MAX][SELR]
    float* head_logits = nullptr;        // [max_slots * K_MAX][valid_vocab]
    int* h2_cand = nullptr;              // [max_slots * K_MAX][H2_SL]
    Cand16* d_cands = nullptr;           // [max_slots * K_MAX][TOPK]
    int* d_meta_slot = nullptr;
    int* d_meta_q0 = nullptr;
    int* d_meta_lo = nullptr;
    int* d_meta_gold = nullptr;
    int* d_tok_out = nullptr;            // [max_slots * K_MAX]
    int* h_meta_pin = nullptr;           // [4 * max_slots]
    int* h_tok_pin = nullptr;            // [max_slots * K_MAX]

    // CUDA Graph cache per active session count S (1..8) for body+head2+selector
    static constexpr int MAX_GRAPH_S = 8;
    cudaGraphExec_t s_graphs[MAX_GRAPH_S + 1] = {};
    int s_graph_K[MAX_GRAPH_S + 1] = {};
    const uint16_t* s_graph_emb[MAX_GRAPH_S + 1] = {};
    const uint16_t* s_graph_lm[MAX_GRAPH_S + 1] = {};
    bool head2_21_ok = false;

    std::vector<void*> owned_dev;

    ~Ink2DFlash2() { release(); }

    void release() {
        for (int i = 0; i <= MAX_GRAPH_S; ++i) {
            if (s_graphs[i]) { cudaGraphExecDestroy(s_graphs[i]); s_graphs[i] = nullptr; }
        }
        for (void* p : owned_dev) if (p) cudaFree(p);
        owned_dev.clear();
        if (h_acc_idx_pin) { cudaFreeHost(h_acc_idx_pin); h_acc_idx_pin = nullptr; }
        if (h_meta_pin) { cudaFreeHost(h_meta_pin); h_meta_pin = nullptr; }
        if (h_tok_pin) { cudaFreeHost(h_tok_pin); h_tok_pin = nullptr; }
        loaded = false;
    }

    template <typename T>
    T* dalloc(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, (n ? n : 1) * sizeof(T)) != cudaSuccess) {
            fprintf(stderr, "ink2_dflash2: cudaMalloc failed (%zu bytes)\n", n * sizeof(T));
            std::abort();
        }
        owned_dev.push_back(p);
        return static_cast<T*>(p);
    }

    struct InitOpts {
        int use_emb_norm = 1;
        int use_head2 = 1;
        int use_q8 = 0;
        int use_dp = 0;
        int use_graph = 1;
        float sel_alpha = 1.0f;
    };

    bool init(const std::string& st_path, int slots, int max_rows, int valid_v,
              const uint16_t* target_lm_bf16, const InitOpts& opts, cudaStream_t st = 0) {
        max_slots = slots;
        max_commit_rows = std::min(std::max(max_rows, B), 512);
        valid_vocab = valid_v;
        slot_ctx_lo.assign(static_cast<size_t>(slots), 0);

        use_emb_norm = opts.use_emb_norm;
        use_head2 = opts.use_head2;
        use_q8 = opts.use_q8;
        use_dp = opts.use_dp;
        use_graph = opts.use_graph;
        sel_alpha = opts.sel_alpha;

        std::unordered_map<std::string, StTensor> tmap;
        size_t data_base = 0;
        if (!parse_safetensors_header(st_path, tmap, data_base)) {
            fprintf(stderr, "ink2_dflash2: failed parsing safetensors header %s\n", st_path.c_str());
            return false;
        }
        std::ifstream f(st_path, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        const size_t fsize = static_cast<size_t>(f.tellg());
        const size_t payload_bytes = fsize - data_base;
        std::vector<uint8_t> hbuf(payload_bytes);
        f.seekg(static_cast<std::streamoff>(data_base), std::ios::beg);
        f.read(reinterpret_cast<char*>(hbuf.data()), static_cast<std::streamsize>(payload_bytes));
        if (!f) return false;

        // Reusable GPU staging buffer (largest single tensor is ~103 MB)
        size_t max_tb = 0;
        for (const auto& kv : tmap) max_tb = std::max(max_tb, kv.second.len);
        uint16_t* stage = nullptr;
        if (cudaMalloc(&stage, std::max<size_t>(max_tb, 16)) != cudaSuccess) return false;

        const char* body_tag = use_q8 ? "BF16->Q8_0" : "BF16->Q4_0";
        const char* cb_tag = "BF16";
        auto load_q4 = [&](const std::string& name, Q4Mat& m, uint16_t** keep_bf16 = nullptr) -> bool {
            auto it_qs = tmap.find(name + ".qs");
            auto it_q8 = tmap.find(name + ".q8");
            auto it_sc = tmap.find(name + ".sc");
            if (it_qs != tmap.end() && it_sc != tmap.end() && it_qs->second.shape.size() == 2) {
                body_tag = "melded-Q4_0";
                m.rows = static_cast<int>(it_qs->second.shape[0]);
                m.cols = static_cast<int>(it_qs->second.shape[1]) * 2;
                const size_t nblks = static_cast<size_t>(m.rows) * (m.cols >> 5);
                m.qs = dalloc<uint8_t>(nblks * 16);
                m.sc = dalloc<__half>(nblks);
                cudaMemcpy(m.qs, hbuf.data() + it_qs->second.off, it_qs->second.len, cudaMemcpyHostToDevice);
                cudaMemcpy(m.sc, hbuf.data() + it_sc->second.off, it_sc->second.len, cudaMemcpyHostToDevice);
                if (keep_bf16) {
                    *keep_bf16 = dalloc<uint16_t>(static_cast<size_t>(m.rows) * m.cols);
                    k_df2_q4_to_bf16<<<static_cast<unsigned>((nblks + 7) / 8), 256, 0, 0>>>(m.qs, m.sc, *keep_bf16, nblks);
                }
                return true;
            }
            if (it_q8 != tmap.end() && it_sc != tmap.end() && it_q8->second.shape.size() == 2) {
                body_tag = "melded-Q8_0";
                m.rows = static_cast<int>(it_q8->second.shape[0]);
                m.cols = static_cast<int>(it_q8->second.shape[1]);
                const size_t nblks = static_cast<size_t>(m.rows) * (m.cols >> 5);
                m.q8 = dalloc<int8_t>(static_cast<size_t>(m.rows) * m.cols);
                m.sc = dalloc<__half>(nblks);
                cudaMemcpy(m.q8, hbuf.data() + it_q8->second.off, it_q8->second.len, cudaMemcpyHostToDevice);
                cudaMemcpy(m.sc, hbuf.data() + it_sc->second.off, it_sc->second.len, cudaMemcpyHostToDevice);
                if (keep_bf16) {
                    *keep_bf16 = dalloc<uint16_t>(static_cast<size_t>(m.rows) * m.cols);
                    k_df2_q8_to_bf16<<<static_cast<unsigned>((nblks + 7) / 8), 256, 0, 0>>>(m.q8, m.sc, *keep_bf16, nblks);
                }
                return true;
            }
            auto it = tmap.find(name);
            if (it == tmap.end() || it->second.shape.size() != 2) {
                fprintf(stderr, "ink2_dflash2: missing 2D tensor %s\n", name.c_str());
                return false;
            }
            m.rows = static_cast<int>(it->second.shape[0]);
            m.cols = static_cast<int>(it->second.shape[1]);
            const size_t nblks = static_cast<size_t>(m.rows) * (m.cols >> 5);
            m.sc = dalloc<__half>(nblks);
            uint16_t* dst_bf = stage;
            if (keep_bf16) {
                *keep_bf16 = dalloc<uint16_t>(static_cast<size_t>(m.rows) * m.cols);
                dst_bf = *keep_bf16;
            }
            cudaMemcpy(dst_bf, hbuf.data() + it->second.off, it->second.len, cudaMemcpyHostToDevice);
            if (use_q8) {
                m.q8 = dalloc<int8_t>(static_cast<size_t>(m.rows) * m.cols);
                k_df2_bf16_to_q8_soa<<<static_cast<unsigned>((nblks + 7) / 8), 256, 0, 0>>>(dst_bf, m.q8, m.sc, nblks);
            } else {
                m.qs = dalloc<uint8_t>(nblks * 16);
                k_df2_bf16_to_q4_soa<<<static_cast<unsigned>((nblks + 7) / 8), 256, 0, 0>>>(dst_bf, m.qs, m.sc, nblks);
            }
            return true;
        };
        auto load_f32 = [&](const std::string& name, float*& ptr) -> bool {
            auto it = tmap.find(name);
            if (it == tmap.end()) { fprintf(stderr, "ink2_dflash2: missing tensor %s\n", name.c_str()); return false; }
            if (it->second.dtype == "F32") {
                const size_t ne = it->second.len / 4;
                ptr = dalloc<float>(ne);
                cudaMemcpy(ptr, hbuf.data() + it->second.off, it->second.len, cudaMemcpyHostToDevice);
                return true;
            }
            const size_t ne = it->second.len / 2;
            ptr = dalloc<float>(ne);
            cudaMemcpy(stage, hbuf.data() + it->second.off, it->second.len, cudaMemcpyHostToDevice);
            k_df2_bf16_to_f32<<<static_cast<unsigned>((ne + 255) / 256), 256, 0, 0>>>(stage, ptr, ne);
            return true;
        };
        auto load_cb = [&](const std::string& name, Q4Codebook& cb) -> bool {
            auto it_qs = tmap.find(name + ".qs");
            auto it_q8 = tmap.find(name + ".q8");
            auto it_sc = tmap.find(name + ".sc");
            if (it_qs != tmap.end() && it_sc != tmap.end() && it_qs->second.shape.size() == 2) {
                cb_tag = "Q4_0";
                cb.vocab = static_cast<int>(it_qs->second.shape[0]);
                uint8_t* qs = dalloc<uint8_t>(it_qs->second.len);
                __half* sc = dalloc<__half>(it_sc->second.len / 2);
                cudaMemcpy(qs, hbuf.data() + it_qs->second.off, it_qs->second.len, cudaMemcpyHostToDevice);
                cudaMemcpy(sc, hbuf.data() + it_sc->second.off, it_sc->second.len, cudaMemcpyHostToDevice);
                cb.qs = qs;
                cb.sc = sc;
                return true;
            }
            if (it_q8 != tmap.end() && it_sc != tmap.end() && it_q8->second.shape.size() == 2) {
                cb_tag = "Q8_0";
                cb.vocab = static_cast<int>(it_q8->second.shape[0]);
                int8_t* q8 = dalloc<int8_t>(it_q8->second.len);
                __half* sc = dalloc<__half>(it_sc->second.len / 2);
                cudaMemcpy(q8, hbuf.data() + it_q8->second.off, it_q8->second.len, cudaMemcpyHostToDevice);
                cudaMemcpy(sc, hbuf.data() + it_sc->second.off, it_sc->second.len, cudaMemcpyHostToDevice);
                cb.q8 = q8;
                cb.sc = sc;
                return true;
            }
            auto it = tmap.find(name);
            if (it == tmap.end() || it->second.shape.size() != 2) {
                fprintf(stderr, "ink2_dflash2: missing codebook %s\n", name.c_str());
                return false;
            }
            cb.vocab = static_cast<int>(it->second.shape[0]);
            __nv_bfloat16* bf = dalloc<__nv_bfloat16>(it->second.len / 2);
            cudaMemcpy(bf, hbuf.data() + it->second.off, it->second.len, cudaMemcpyHostToDevice);
            cb.bf16 = bf;
            return true;
        };

        bool ok = load_q4("fc.weight", fc, &fc_bf16) &&
                  load_f32("hidden_norm.weight", hidden_norm) &&
                  load_f32("norm.weight", final_norm) &&
                  load_q4("candidate_selector.hidden_projection.weight", hproj) &&
                  load_cb("candidate_selector.predecessor_codebook", pred_cb) &&
                  load_cb("candidate_selector.successor_codebook", succ_cb);
        for (int l = 0; l < L && ok; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            ok = ok &&
                 load_q4(p + "self_attn.q_proj.weight", ly[l].q) &&
                 load_q4(p + "self_attn.k_proj.weight", ly[l].k, &ly[l].k_bf16) &&
                 load_q4(p + "self_attn.v_proj.weight", ly[l].v, &ly[l].v_bf16) &&
                 load_q4(p + "self_attn.o_proj.weight", ly[l].o) &&
                 load_q4(p + "mlp.gate_proj.weight", ly[l].gate) &&
                 load_q4(p + "mlp.up_proj.weight", ly[l].up) &&
                 load_q4(p + "mlp.down_proj.weight", ly[l].down) &&
                 load_q4(p + "attention_conv.kernel_projection.weight", ly[l].aconv_kp) &&
                 load_q4(p + "mlp_conv.kernel_projection.weight", ly[l].mconv_kp) &&
                 load_f32(p + "input_layernorm.weight", ly[l].in_norm) &&
                 load_f32(p + "post_attention_layernorm.weight", ly[l].post_norm) &&
                 load_f32(p + "self_attn.q_norm.weight", ly[l].q_norm) &&
                 load_f32(p + "self_attn.k_norm.weight", ly[l].k_norm) &&
                 load_f32(p + "attention_conv.base_kernel", ly[l].aconv_base) &&
                 load_f32(p + "mlp_conv.base_kernel", ly[l].mconv_base);
        }
        cudaDeviceSynchronize();
        cudaFree(stage);
        if (!ok) return false;

        // Build 2-bit Head2 shortlist table from target's BF16 lm_head[0..valid_vocab-1]
        cudaFuncSetAttribute(k_df2_head2_screen<14>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(14 * H2_K8 * sizeof(uint4)));
        head2_21_ok = (cudaFuncSetAttribute(k_df2_head2_screen<21>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(21 * H2_K8 * sizeof(uint4))) == cudaSuccess);
        if (!head2_21_ok) cudaGetLastError();
        h2_codes = dalloc<uint4>(static_cast<size_t>(valid_vocab) * H2_NCH);
        h2_scales = dalloc<__half2>(static_cast<size_t>(valid_vocab) * H2_NCH);
        const long long nchunks = static_cast<long long>(valid_vocab) * H2_NCH;
        k_df2_lm_to_i2<<<static_cast<unsigned>((nchunks + 255) / 256), 256, 0, 0>>>(target_lm_bf16, h2_codes, h2_scales, valid_vocab);
        cudaDeviceSynchronize();

        // Allocate KV rings and scratch
        const size_t ring_elems = static_cast<size_t>(max_slots) * L * NKV * RING * HD;
        kring = dalloc<__half>(ring_elems);
        vring = dalloc<__half>(ring_elems);
        cudaMemsetAsync(kring, 0, ring_elems * sizeof(__half), st);
        cudaMemsetAsync(vring, 0, ring_elems * sizeof(__half), st);

        ctx_taps_bf16 = dalloc<uint16_t>(static_cast<size_t>(max_commit_rows) * NT * HID);
        ctx_fc_bf16 = dalloc<uint16_t>(static_cast<size_t>(max_commit_rows) * HID);
        ctx_norm_bf16 = dalloc<uint16_t>(static_cast<size_t>(max_commit_rows) * HID);
        ctx_k_bf16 = dalloc<uint16_t>(static_cast<size_t>(max_commit_rows) * NKV * HD);
        ctx_v_bf16 = dalloc<uint16_t>(static_cast<size_t>(max_commit_rows) * NKV * HD);
        ctx_fc_f32 = dalloc<float>(static_cast<size_t>(B) * HID);
        ctx_k_f32 = dalloc<float>(static_cast<size_t>(B) * NKV * HD);
        ctx_v_f32 = dalloc<float>(static_cast<size_t>(B) * NKV * HD);
        d_acc_idx = dalloc<int>(static_cast<size_t>(max_commit_rows));
        cudaHostAlloc(&h_acc_idx_pin, static_cast<size_t>(max_commit_rows) * sizeof(int), cudaHostAllocDefault);

        const size_t mb = static_cast<size_t>(max_slots) * B;
        const size_t mk = static_cast<size_t>(max_slots) * K_MAX;
        const size_t q8_kmax = static_cast<size_t>(NT) * HID;   // 9216 > INTER(6912)
        h = dalloc<float>(mb * HID);
        xn = dalloc<float>(mb * HID);
        dyn = dalloc<float>(mb * CPO);
        q_buf = dalloc<float>(mb * NH * HD);
        qf = dalloc<float>(mb * NH * HD);
        k_buf = dalloc<float>(mb * NKV * HD);
        v_buf = dalloc<float>(mb * NKV * HD);
        kblk = dalloc<__half>(static_cast<size_t>(max_slots) * NKV * B * HD);
        vblk = dalloc<__half>(static_cast<size_t>(max_slots) * NKV * B * HD);
        apart = dalloc<float>(static_cast<size_t>(max_slots) * NH * ATTN_SPLIT * B * (HD + 2));
        ao = dalloc<float>(mb * NH * HD);
        o_buf = dalloc<float>(mb * HID);
        act_buf = dalloc<float>(mb * INTER);
        qa.q = dalloc<int8_t>(mb * q8_kmax);
        qa.sc = dalloc<float>(mb * (q8_kmax >> 5));
        qa.isum = dalloc<int>(mb * (q8_kmax >> 4));
        out_bf16 = dalloc<uint16_t>(mk * HID);
        hp = dalloc<float>(mk * SELR);
        head_logits = dalloc<float>(mk * valid_vocab);
        h2_cand = dalloc<int>(mk * H2_SL);
        d_cands = dalloc<Cand16>(mk * TOPK);
        d_meta_slot = dalloc<int>(static_cast<size_t>(max_slots));
        d_meta_q0 = dalloc<int>(static_cast<size_t>(max_slots));
        d_meta_lo = dalloc<int>(static_cast<size_t>(max_slots));
        d_meta_gold = dalloc<int>(static_cast<size_t>(max_slots));
        d_tok_out = dalloc<int>(mk);
        cudaHostAlloc(&h_meta_pin, static_cast<size_t>(4) * max_slots * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_tok_pin, mk * sizeof(int), cudaHostAllocDefault);
        cudaStreamSynchronize(st);

        loaded = true;
        fprintf(stderr, "# ink2_dflash2: loaded %s (%s body + %s selector + Head2 valid_vocab=%d, slots=%d, emb_norm=%d, head2=%d, dp=%d, graph=%d)\n",
                st_path.c_str(), body_tag, cb_tag, valid_vocab, max_slots, use_emb_norm, use_head2, use_dp, use_graph);
        return true;
    }

    void reset_slot(int slot, cudaStream_t st = 0) {
        if (!loaded || slot < 0 || slot >= max_slots) return;
        slot_ctx_lo[static_cast<size_t>(slot)] = 0;
        const size_t slot_elems = static_cast<size_t>(L) * NKV * RING * HD;
        cudaMemsetAsync(kring + static_cast<size_t>(slot) * slot_elems, 0, slot_elems * sizeof(__half), st);
        cudaMemsetAsync(vring + static_cast<size_t>(slot) * slot_elems, 0, slot_elems * sizeof(__half), st);
    }

    void set_ctx_lo(int slot, int lo) {
        if (!loaded || slot < 0 || slot >= max_slots) return;
        slot_ctx_lo[static_cast<size_t>(slot)] = std::max(0, lo);
    }

    // Ingest accepted committed rows from step_taps [NT][max_rows][HID] into each session's FP16 ring.
    // Tiles large prefill commits in slices of max_commit_rows (<=512) to bound scratch VRAM.
    void commit(const uint16_t* step_taps, size_t tap_stride, const ink2::Rows& rw,
                const std::vector<uint8_t>* acc_host, int M, cublasHandle_t blas, cudaStream_t st) {
        if (!loaded || M <= 0) return;
        auto run_slice = [&](const uint16_t* taps_ptr, const int* slot_ptr, const int* pos_ptr, const int* d_idx, int n_acc) {
            const size_t tot = static_cast<size_t>(n_acc) * NT * HID;
            k_df2_gather_taps<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(taps_ptr, tap_stride, d_idx, ctx_taps_bf16, n_acc);

            if (n_acc <= B) {
                // Fast Q4_0 __dp4a path for small decode commits (~0.15 ms)
                quant_q8_bf16(ctx_taps_bf16, qa, n_acc, NT * HID, st);
                launch_q4_gemv(qa, fc, ctx_fc_f32, n_acc, st);
                k_df2_rmsnorm_q8<false, false, false><<<n_acc, 256, 0, st>>>(ctx_fc_f32, hidden_norm, nullptr, nullptr, qa.q, qa.sc, qa.isum, RMS_EPS);
                for (int l = 0; l < L; ++l) {
                    launch_q4_gemv(qa, ly[l].k, ctx_k_f32, n_acc, st);
                    launch_q4_gemv(qa, ly[l].v, ctx_v_f32, n_acc, st);
                    k_df2_ctx_kv_scatter<false><<<n_acc * NKV, 128, 0, st>>>(
                        ctx_k_f32, ctx_v_f32, ly[l].k_norm, slot_ptr, pos_ptr, d_idx, kring, vring, l, RMS_EPS);
                }
            } else {
                // Fast cuBLAS BF16 path for prefill chunks (M = 9..4096, tiled in slices of <=512), through the SM70
                // bridge: native BF16 on SM80+, FP16 tensor cores on Volta; the ink2 GEMM kernels if cuBLAS refuses
                const float one = 1.0f, zero = 0.0f;
                auto bf16_gemm = [&](const uint16_t* x, const uint16_t* w, int M_, int K_, int N_, void* y) {
                    cublasStatus_t rc = blas ? cublasSetStream(blas, st) : CUBLAS_STATUS_NOT_INITIALIZED;
                    if (rc == CUBLAS_STATUS_SUCCESS)
                        rc = gr8_sm70::cublas_gemm_bf16_tn(blas, N_, M_, K_, &one, w, K_, x, K_,
                                                           &zero, y, CUDA_R_16BF, N_,
                                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (rc != CUBLAS_STATUS_SUCCESS) ink2::gemm(x, w, M_, K_, N_, y, ink2::GEMM_BF16, 0, N_, st);
                };
                bf16_gemm(ctx_taps_bf16, fc_bf16, n_acc, NT * HID, HID, ctx_fc_bf16);
                k_df2_rmsnorm_q8<false, true, true><<<n_acc, 256, 0, st>>>(ctx_fc_bf16, hidden_norm, nullptr, ctx_norm_bf16, nullptr, nullptr, nullptr, RMS_EPS);
                for (int l = 0; l < L; ++l) {
                    bf16_gemm(ctx_norm_bf16, ly[l].k_bf16, n_acc, HID, NKV * HD, ctx_k_bf16);
                    bf16_gemm(ctx_norm_bf16, ly[l].v_bf16, n_acc, HID, NKV * HD, ctx_v_bf16);
                    k_df2_ctx_kv_scatter<true><<<n_acc * NKV, 128, 0, st>>>(
                        ctx_k_bf16, ctx_v_bf16, ly[l].k_norm, slot_ptr, pos_ptr, d_idx, kring, vring, l, RMS_EPS);
                }
            }
        };

        if (acc_host != nullptr) {
            std::vector<int> acc_all;
            acc_all.reserve(static_cast<size_t>(M));
            for (int m = 0; m < M; ++m) {
                if ((*acc_host)[static_cast<size_t>(m)]) acc_all.push_back(m);
            }
            const int total_acc = static_cast<int>(acc_all.size());
            if (total_acc == 0) return;
            if (total_acc <= max_commit_rows) {
                memcpy(h_acc_idx_pin, acc_all.data(), static_cast<size_t>(total_acc) * sizeof(int));
                cudaMemcpyAsync(d_acc_idx, h_acc_idx_pin, static_cast<size_t>(total_acc) * sizeof(int), cudaMemcpyHostToDevice, st);
                run_slice(step_taps, rw.slot, rw.pos, d_acc_idx, total_acc);
            } else {
                for (int off = 0; off < total_acc; off += max_commit_rows) {
                    const int nm = std::min(total_acc - off, max_commit_rows);
                    cudaMemcpyAsync(d_acc_idx, acc_all.data() + off, static_cast<size_t>(nm) * sizeof(int), cudaMemcpyHostToDevice, st);
                    run_slice(step_taps, rw.slot, rw.pos, d_acc_idx, nm);
                    cudaStreamSynchronize(st);
                }
            }
        } else {
            for (int off = 0; off < M; off += max_commit_rows) {
                const int nm = std::min(M - off, max_commit_rows);
                run_slice(step_taps + static_cast<size_t>(off) * HID, rw.slot + off, rw.pos + off, nullptr, nm);
            }
        }
    }

    // Draft K tokens (1..7) for each of S sessions in parallel!
    void draft(const std::vector<int>& slots, const std::vector<int>& q_pos, const std::vector<int>& golden,
               int K, const uint16_t* target_embed, const uint16_t* target_embed_norm,
               const uint16_t* target_lm_bf16, cublasHandle_t blas, std::vector<int>& out, cudaStream_t st) {
        const int S = static_cast<int>(slots.size());
        const int MB = S * B;
        const int MK = S * K_MAX;

        int* pm_slot = h_meta_pin;
        int* pm_q0 = pm_slot + max_slots;
        int* pm_lo = pm_q0 + max_slots;
        int* pm_gold = pm_lo + max_slots;
        for (int s = 0; s < S; ++s) {
            pm_slot[s] = slots[static_cast<size_t>(s)];
            pm_q0[s] = q_pos[static_cast<size_t>(s)] + 1;
            pm_lo[s] = slot_ctx_lo[static_cast<size_t>(slots[static_cast<size_t>(s)])];
            pm_gold[s] = golden[static_cast<size_t>(s)];
        }
        cudaMemcpyAsync(d_meta_slot, pm_slot, static_cast<size_t>(S) * sizeof(int), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_meta_q0, pm_q0, static_cast<size_t>(S) * sizeof(int), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_meta_lo, pm_lo, static_cast<size_t>(S) * sizeof(int), cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_meta_gold, pm_gold, static_cast<size_t>(S) * sizeof(int), cudaMemcpyHostToDevice, st);

        auto launch_all = [&]() {
            k_df2_embed<<<MB, 256, 0, st>>>(target_embed, target_embed_norm, d_meta_gold, h, use_emb_norm);
            for (int l = 0; l < L; ++l) {
                k_df2_rmsnorm_q8<true, false><<<MB, 256, 0, st>>>(h, ly[l].in_norm, xn, nullptr, qa.q, qa.sc, qa.isum, RMS_EPS);
                launch_q4_gemv(qa, ly[l].aconv_kp, dyn, MB, st);
                k_df2_conv0_q8<<<dim3(HID / 256, MB), 256, 0, st>>>(xn, dyn, ly[l].aconv_base, qa.q, qa.sc, qa.isum);
                launch_q4_qkv(qa, ly[l], q_buf, k_buf, v_buf, MB, st);
                k_df2_q_norm_rope<<<MB * NH, 128, 0, st>>>(q_buf, ly[l].q_norm, qf, d_meta_q0, RMS_EPS);
                k_df2_blk_kv<<<MB * NKV, 128, 0, st>>>(k_buf, v_buf, ly[l].k_norm, kblk, vblk, d_meta_q0, RMS_EPS);
                k_df2_attn_split<<<dim3(NH, ATTN_SPLIT, S), 128, 0, st>>>(
                    qf, kring, vring, kblk, vblk, d_meta_slot, d_meta_q0, d_meta_lo, l, apart);
                k_df2_attn_merge<<<dim3(B * NH, S), 128, 0, st>>>(apart, ao);
                quant_q8_f32(ao, qa, MB, NH * HD, st);
                launch_q4_gemv(qa, ly[l].o, o_buf, MB, st);
                k_df2_conv1_add<<<dim3(HID / 256, MB), 256, 0, st>>>(o_buf, dyn, ly[l].aconv_base, h);

                k_df2_rmsnorm_q8<true, false><<<MB, 256, 0, st>>>(h, ly[l].post_norm, xn, nullptr, qa.q, qa.sc, qa.isum, RMS_EPS);
                launch_q4_gemv(qa, ly[l].mconv_kp, dyn, MB, st);
                k_df2_conv0_q8<<<dim3(HID / 256, MB), 256, 0, st>>>(xn, dyn, ly[l].mconv_base, qa.q, qa.sc, qa.isum);
                launch_q4_gate_up_swiglu(qa, ly[l], act_buf, MB, st);
                quant_q8_f32(act_buf, qa, MB, INTER, st);
                launch_q4_gemv(qa, ly[l].down, o_buf, MB, st);
                k_df2_conv1_add<<<dim3(HID / 256, MB), 256, 0, st>>>(o_buf, dyn, ly[l].mconv_base, h);
            }
            // Final norm on draft rows 1..7 of all S sessions + hproj
            k_df2_final_norm_q8<<<MK, 256, 0, st>>>(h, final_norm, out_bf16, qa.q, qa.sc, qa.isum, RMS_EPS);
            launch_q4_gemv(qa, hproj, hp, MK, st);

            // Draft LM head -> sorted top-16 Cand16 per row
            if (use_head2) {
                const int scr_grid = 256;
                int off = 0;
                while (off < MK) {
                    const int rem = MK - off;
                    if (head2_21_ok && rem >= 21) {
                        k_df2_head2_screen<21><<<scr_grid, 256, 21 * H2_K8 * sizeof(uint4), st>>>(
                            h2_codes, h2_scales,
                            out_bf16 + static_cast<size_t>(off) * HID,
                            head_logits + static_cast<size_t>(off) * valid_vocab,
                            21, valid_vocab);
                        off += 21;
                    } else if (rem >= 14) {
                        k_df2_head2_screen<14><<<scr_grid, 256, 14 * H2_K8 * sizeof(uint4), st>>>(
                            h2_codes, h2_scales,
                            out_bf16 + static_cast<size_t>(off) * HID,
                            head_logits + static_cast<size_t>(off) * valid_vocab,
                            14, valid_vocab);
                        off += 14;
                    } else {
                        const int n = std::min(7, rem);
                        k_df2_head2_screen<7><<<scr_grid, 256, 7 * H2_K8 * sizeof(uint4), st>>>(
                            h2_codes, h2_scales,
                            out_bf16 + static_cast<size_t>(off) * HID,
                            head_logits + static_cast<size_t>(off) * valid_vocab,
                            n, valid_vocab);
                        off += n;
                    }
                }
                k_df2_top_sl<H2_SL><<<MK, 1024, 0, st>>>(head_logits, valid_vocab, h2_cand);
                k_df2_rescore_bf16_top16<<<MK, 256, 0, st>>>(target_lm_bf16, out_bf16, h2_cand, d_cands);
            } else {
                const float one = 1.0f, zero = 0.0f;
                cublasStatus_t rc = blas ? cublasSetStream(blas, st) : CUBLAS_STATUS_NOT_INITIALIZED;
                if (rc == CUBLAS_STATUS_SUCCESS)
                    rc = gr8_sm70::cublas_gemm_bf16_tn(blas, valid_vocab, MK, HID, &one,
                                                       target_lm_bf16, HID, out_bf16, HID,
                                                       &zero, head_logits, CUDA_R_32F, valid_vocab,
                                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (rc != CUBLAS_STATUS_SUCCESS)
                    ink2::gemm(out_bf16, target_lm_bf16, MK, HID, valid_vocab, head_logits, ink2::GEMM_F32, 0, valid_vocab, st);
                k_df2_top_sl<TOPK><<<MK, 1024, 0, st>>>(head_logits, valid_vocab, h2_cand);
                k_df2_sort_top16_from_y<<<MK, 256, 0, st>>>(head_logits, valid_vocab, h2_cand, d_cands);
            }

            // Candidate selector walk (all S sessions in parallel!)
            k_df2_select_multi<<<S, 256, 0, st>>>(
                d_cands, d_meta_gold, hp, pred_cb, succ_cb, K, sel_alpha, use_dp, dp_cal, d_tok_out);
        };

        if (use_graph && S >= 1 && S <= MAX_GRAPH_S && use_head2) {
            if (!s_graphs[S] || s_graph_K[S] != K || s_graph_emb[S] != target_embed || s_graph_lm[S] != target_lm_bf16) {
                if (s_graphs[S]) { cudaGraphExecDestroy(s_graphs[S]); s_graphs[S] = nullptr; }
                cudaGraph_t g = nullptr;
                cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed);
                launch_all();
                cudaStreamEndCapture(st, &g);
                cudaGraphInstantiate(&s_graphs[S], g, nullptr, nullptr, 0);
                cudaGraphDestroy(g);
                s_graph_K[S] = K;
                s_graph_emb[S] = target_embed;
                s_graph_lm[S] = target_lm_bf16;
            }
            cudaGraphLaunch(s_graphs[S], st);
        } else {
            launch_all();
        }

        cudaMemcpyAsync(h_tok_pin, d_tok_out, static_cast<size_t>(S) * K * sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);

        out.resize(static_cast<size_t>(S) * K);
        std::memcpy(out.data(), h_tok_pin, static_cast<size_t>(S) * K * sizeof(int));
    }
};

#endif  // __CUDACC__
}  // namespace df2
}  // namespace ink2
