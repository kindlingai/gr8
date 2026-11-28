// Native CUDA Drama 3 (S2-Pro modded_dac) audio codec: streaming/batched decoder + whole-clip encoder.
//
// Architecture (fish-speech @ 0f8a12e8, modded_dac_vq.yaml, exlr8-codec-v1 safetensors):
//   - Sample rate: 44.1 kHz, hop 512 x quantizer factor 4 = 2048 samples/frame (21.533 Hz), zero lookahead.
//   - Decoder (strict IEEE fp32 by default so streaming strides 1,1,2,4 match whole-sequence decode within
//     <= 1 LSB of s16; optional fp16 GEMM fast path via init(..., fp32_decoder=false)):
//       10-codebook projected table sum -> 8-layer causal window-128 transformer (post_module, bf16-rounded RoPE)
//       -> 2x [CausalTransConv1d(k=2,s=2) + ConvNeXtBlock(dwconv k=7, LN eps=1e-6, exact GELU, gamma)]
//       -> CausalConv1d(k=7, 1024->1536)
//       -> 4x DecoderBlock (strides 8,8,4,2; channels 1536->768->384->192->96;
//          each Snake + CausalTransConv1d(k=2s,s=s) + 3 ResidualUnits dilations 1,3,9)
//       -> Snake -> CausalConv1d(k=7, 96->1) -> Tanh -> int16 PCM.
//   - Encoder (all fp32):
//       right-pad to multiple of 2048 -> CausalConv1d(k=7, 1->64)
//       -> 4x EncoderBlock (strides 2,4,8,8; channels 64->128->256->512->1024;
//          each 3 ResidualUnits d=1,3,9 + Snake + CausalConv1d(k=2s,s=s); block 3 adds 4-layer window-512 transformer)
//       -> Snake -> CausalConv1d(k=3, 1024->1024)
//       -> 2x [CausalConv1d(k=2,s=2) + ConvNeXtBlock]
//       -> 8-layer causal window-128 transformer (pre_module)
//       -> 10-stage RVQ (stage 0 size 4096, stages 1..9 size 1024; L2-normalized distance argmin, subtract projected_codes).
//
// CUDA Graph compatibility (Wave 3.3):
//   - `decode_frames()` does zero allocations, zero host copies, and zero host synchronizations when
//     `M <= max_slots` and `stride <= max_decode_stride`. All per-slot state (conv arenas, transconv carry
//     tails, 8-layer KV rings, and device frame positions) is indexed indirectly via `d_slots[M]`, and cuBLAS
//     uses a pre-allocated fixed device workspace.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace g6tts {

namespace codec_detail {

constexpr int CODEC_SAMPLE_RATE = 44100;
constexpr int CODEC_FRAME_SAMPLES = 2048;
constexpr int CODEC_NUM_CODEBOOKS = 10;
constexpr int CODEC_SEM_SIZE = 4096;
constexpr int CODEC_RES_SIZE = 1024;
constexpr int CODEC_PROJ_ROWS = CODEC_SEM_SIZE + 9 * CODEC_RES_SIZE;  // 13312
constexpr int CODEC_LATENT_DIM = 1024;
constexpr int CODEC_CB_DIM = 8;

constexpr int CODEC_N_HEAD = 16;
constexpr int CODEC_HEAD_DIM = 64;
constexpr int CODEC_FFN_DIM = 3072;
constexpr int CODEC_POST_LAYERS = 8;
constexpr int CODEC_PRE_LAYERS = 8;
constexpr int CODEC_ENC_T_LAYERS = 4;
constexpr int CODEC_POST_WINDOW = 128;
constexpr int CODEC_PRE_WINDOW = 128;
constexpr int CODEC_ENC_WINDOW = 512;
constexpr int CODEC_KV_RING = 256;          // power of 2 >= 128 + max_decode_stride
constexpr int CODEC_MAX_ROPE_LEN = 65536;   // > 50 minutes at 21.53 Hz
constexpr float CODEC_ROPE_BASE = 10000.0f;
constexpr float CODEC_RMS_EPS = 1e-5f;
constexpr float CODEC_LN_EPS = 1e-6f;

constexpr int DEC_STRIDES[4] = {8, 8, 4, 2};
constexpr int DEC_IN_CH[4] = {1536, 768, 384, 192};
constexpr int DEC_OUT_CH[4] = {768, 384, 192, 96};
constexpr int ENC_STRIDES[4] = {2, 4, 8, 8};
constexpr int ENC_IN_CH[4] = {64, 128, 256, 512};
constexpr int ENC_OUT_CH[4] = {128, 256, 512, 1024};
constexpr int RES_DILATIONS[3] = {1, 3, 9};

// Per-slot decoder causal conv context slots (stored in fp32 so streaming casts match whole-sequence bit-for-bit).
enum DecCtxId : int {
    CTX_UP0_DW = 0,     // pad=6,  C=1024
    CTX_UP1_DW = 1,     // pad=6,  C=1024
    CTX_DEC_IN = 2,     // pad=6,  C=1024
    CTX_DEC_B0_R0 = 3,  // pad=6,  C=768
    CTX_DEC_B0_R1 = 4,  // pad=18, C=768
    CTX_DEC_B0_R2 = 5,  // pad=54, C=768
    CTX_DEC_B1_R0 = 6,  // pad=6,  C=384
    CTX_DEC_B1_R1 = 7,  // pad=18, C=384
    CTX_DEC_B1_R2 = 8,  // pad=54, C=384
    CTX_DEC_B2_R0 = 9,  // pad=6,  C=192
    CTX_DEC_B2_R1 = 10, // pad=18, C=192
    CTX_DEC_B2_R2 = 11, // pad=54, C=192
    CTX_DEC_B3_R0 = 12, // pad=6,  C=96
    CTX_DEC_B3_R1 = 13, // pad=18, C=96
    CTX_DEC_B3_R2 = 14, // pad=54, C=96
    CTX_DEC_OUT = 15,   // pad=6,  C=96
    CTX_COUNT = 16
};

constexpr int DEC_CTX_PAD[CTX_COUNT] = {6, 6, 6, 6, 18, 54, 6, 18, 54, 6, 18, 54, 6, 18, 54, 6};
constexpr int DEC_CTX_CH[CTX_COUNT] = {1024, 1024, 1024, 768, 768, 768, 384, 384, 384, 192, 192, 192, 96, 96, 96, 96};

inline size_t dec_ctx_offset(int id) {
    size_t off = 0;
    for (int i = 0; i < id; ++i) off += size_t(DEC_CTX_PAD[i]) * size_t(DEC_CTX_CH[i]);
    return off;
}
inline size_t dec_ctx_total_elems() { return dec_ctx_offset(CTX_COUNT); }  // 131,328 floats

inline size_t dec_carry_offset(int block_idx) {
    size_t off = 0;
    for (int b = 0; b < block_idx; ++b) off += size_t(DEC_STRIDES[b]) * size_t(DEC_OUT_CH[b]);
    return off;
}
inline size_t dec_carry_total_elems() { return dec_carry_offset(4); }  // 10,176 floats

inline void cuda_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("tts_codec CUDA error (") + what + "): " + cudaGetErrorString(e));
    }
}
inline void cublas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("tts_codec cuBLAS error (") + what + "): code " + std::to_string(int(s)));
    }
}
#define G6TTS_CODEC_CK(x) ::g6tts::codec_detail::cuda_ck((x), #x)
#define G6TTS_CUBLAS_CK(x) ::g6tts::codec_detail::cublas_ck((x), #x)

// Exact IEEE-754 float32 -> bfloat16 round-to-nearest-even, promoted back to float32.
inline float round_f32_to_bf16_f32(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) return x;  // NaN / Inf
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    u &= 0xFFFF0000u;
    float y = 0.0f;
    std::memcpy(&y, &u, sizeof(y));
    return y;
}

// ------------------------------------------------------------------------- Minimal self-contained safetensors parser

struct StTensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t begin = 0;
    size_t end = 0;
};

struct StHeader {
    size_t data_off = 0;
    size_t file_bytes = 0;
    std::map<std::string, std::string> metadata;
    std::map<std::string, StTensorInfo> tensors;
};

struct MiniJson {
    const std::string& s;
    size_t i = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    void expect(char c) {
        ws();
        if (i >= s.size() || s[i] != c) throw std::runtime_error("tts_codec: malformed safetensors header JSON");
        ++i;
    }
    std::string parse_str() {
        expect('"');
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i >= s.size()) throw std::runtime_error("tts_codec: unterminated JSON escape");
                char c = s[i++];
                if (c == '"' || c == '\\' || c == '/') out.push_back(c);
                else if (c == 'n') out.push_back('\n');
                else if (c == 'r') out.push_back('\r');
                else if (c == 't') out.push_back('\t');
                else if (c == 'u' && i + 4 <= s.size()) {
                    unsigned cp = std::stoul(s.substr(i, 4), nullptr, 16);
                    i += 4;
                    if (cp < 128) out.push_back(static_cast<char>(cp));
                } else {
                    out.push_back(c);
                }
            } else {
                out.push_back(s[i++]);
            }
        }
        expect('"');
        return out;
    }
    int64_t parse_i64() {
        ws();
        size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
        if (i == start) throw std::runtime_error("tts_codec: expected integer in safetensors header");
        return std::stoll(s.substr(start, i - start));
    }
    void skip_value() {
        ws();
        if (i >= s.size()) throw std::runtime_error("tts_codec: unexpected EOF in safetensors JSON");
        char c = s[i];
        if (c == '"') {
            (void)parse_str();
        } else if (c == '{') {
            ++i;
            ws();
            if (i < s.size() && s[i] == '}') { ++i; return; }
            for (;;) {
                (void)parse_str();
                expect(':');
                skip_value();
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                expect('}');
                break;
            }
        } else if (c == '[') {
            ++i;
            ws();
            if (i < s.size() && s[i] == ']') { ++i; return; }
            for (;;) {
                skip_value();
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                expect(']');
                break;
            }
        } else {
            while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != ' ' && s[i] != '\n' &&
                   s[i] != '\r' && s[i] != '\t') {
                ++i;
            }
        }
    }
    std::vector<int64_t> parse_i64_array() {
        expect('[');
        std::vector<int64_t> a;
        ws();
        if (i < s.size() && s[i] == ']') { ++i; return a; }
        for (;;) {
            a.push_back(parse_i64());
            ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            expect(']');
            break;
        }
        return a;
    }
};

inline StHeader parse_safetensors_header(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("tts_codec: cannot open " + path);
    uint64_t hdr_len = 0;
    if (std::fread(&hdr_len, 8, 1, f) != 1 || hdr_len == 0 || hdr_len > (64u << 20)) {
        std::fclose(f);
        throw std::runtime_error("tts_codec: invalid safetensors header length in " + path);
    }
    std::string raw(static_cast<size_t>(hdr_len), '\0');
    if (std::fread(&raw[0], 1, static_cast<size_t>(hdr_len), f) != static_cast<size_t>(hdr_len)) {
        std::fclose(f);
        throw std::runtime_error("tts_codec: truncated safetensors header in " + path);
    }
    std::fseek(f, 0, SEEK_END);
    const long fend = std::ftell(f);
    std::fclose(f);
    if (fend < 0) throw std::runtime_error("tts_codec: ftell failed on " + path);

    StHeader out;
    out.data_off = 8u + static_cast<size_t>(hdr_len);
    out.file_bytes = static_cast<size_t>(fend);

    MiniJson p{raw, 0};
    p.expect('{');
    p.ws();
    if (p.i < raw.size() && raw[p.i] != '}') {
        for (;;) {
            std::string key = p.parse_str();
            p.expect(':');
            if (key == "__metadata__") {
                p.expect('{');
                p.ws();
                if (p.i < raw.size() && raw[p.i] != '}') {
                    for (;;) {
                        std::string mk = p.parse_str();
                        p.expect(':');
                        std::string mv = p.parse_str();
                        out.metadata[mk] = mv;
                        p.ws();
                        if (p.i < raw.size() && raw[p.i] == ',') { ++p.i; continue; }
                        p.expect('}');
                        break;
                    }
                } else {
                    p.expect('}');
                }
            } else {
                p.expect('{');
                StTensorInfo ti;
                p.ws();
                if (p.i < raw.size() && raw[p.i] != '}') {
                    for (;;) {
                        std::string fk = p.parse_str();
                        p.expect(':');
                        if (fk == "dtype") {
                            ti.dtype = p.parse_str();
                        } else if (fk == "shape") {
                            ti.shape = p.parse_i64_array();
                        } else if (fk == "data_offsets") {
                            auto off = p.parse_i64_array();
                            if (off.size() != 2 || off[0] < 0 || off[1] < off[0]) {
                                throw std::runtime_error("tts_codec: bad data_offsets for " + key);
                            }
                            ti.begin = static_cast<size_t>(off[0]);
                            ti.end = static_cast<size_t>(off[1]);
                        } else {
                            p.skip_value();
                        }
                        p.ws();
                        if (p.i < raw.size() && raw[p.i] == ',') { ++p.i; continue; }
                        p.expect('}');
                        break;
                    }
                } else {
                    p.expect('}');
                }
                if (out.data_off + ti.end > out.file_bytes) {
                    throw std::runtime_error("tts_codec: tensor " + key + " exceeds file size");
                }
                out.tensors[key] = std::move(ti);
            }
            p.ws();
            if (p.i < raw.size() && raw[p.i] == ',') { ++p.i; continue; }
            p.expect('}');
            break;
        }
    }
    return out;
}

// ------------------------------------------------------------------------- CUDA Kernels

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float block_sum256(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float out = 0.0f;
    if (threadIdx.x < 32) {
        out = (threadIdx.x < 8) ? red[threadIdx.x] : 0.0f;
        out = warp_sum(out);
        if (threadIdx.x == 0) red[0] = out;
    }
    __syncthreads();
    const float res = red[0];
    __syncthreads();
    return res;
}

// Table-sum latent: d_codes[M, S, 10] -> out[M, S, 1024] (fp32). Grid: (M * S), Block: 256.
__global__ void __launch_bounds__(256) k_table_sum_latent(
    const int32_t* __restrict__ codes,
    const float* __restrict__ projected_codes,
    float* __restrict__ out,
    int MS) {
    const int row = blockIdx.x;
    if (row >= MS) return;
    const int32_t* c = codes + size_t(row) * CODEC_NUM_CODEBOOKS;
    __shared__ int idx_off[CODEC_NUM_CODEBOOKS];
    if (threadIdx.x < CODEC_NUM_CODEBOOKS) {
        const int q = threadIdx.x;
        int v = c[q];
        const int vmax = (q == 0) ? (CODEC_SEM_SIZE - 1) : (CODEC_RES_SIZE - 1);
        if (v < 0) v = 0;
        if (v > vmax) v = vmax;
        const int base = (q == 0) ? 0 : (CODEC_SEM_SIZE + (q - 1) * CODEC_RES_SIZE);
        idx_off[q] = base + v;
    }
    __syncthreads();

    float* y = out + size_t(row) * CODEC_LATENT_DIM;
    for (int d = threadIdx.x; d < CODEC_LATENT_DIM; d += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int q = 0; q < CODEC_NUM_CODEBOOKS; ++q) {
            acc += projected_codes[size_t(idx_off[q]) * CODEC_LATENT_DIM + d];
        }
        y[d] = acc;
    }
}

// RMSNorm fp32 -> fp16 (or fp32 when y_f32 != nullptr). Grid: (N), Block: 256.
__global__ void __launch_bounds__(256) k_rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    __half* __restrict__ y_f16,
    float* __restrict__ y_f32,
    int N, int D, float eps) {
    __shared__ float red[8];
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + size_t(row) * D;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = xr[d];
        ss += v * v;
    }
    const float r = rsqrtf(block_sum256(ss, red) / float(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float o = (xr[d] * r) * w[d];
        if (y_f16) y_f16[size_t(row) * D + d] = __float2half_rn(o);
        if (y_f32) y_f32[size_t(row) * D + d] = o;
    }
}

// LayerNorm (eps=1e-6) fp32 -> fp16 and/or fp32. Grid: (N), Block: 256.
__global__ void __launch_bounds__(256) k_layernorm(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y_f16,
    float* __restrict__ y_f32,
    int N, int D, float eps) {
    __shared__ float red[8];
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + size_t(row) * D;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) sum += xr[d];
    const float mean = block_sum256(sum, red) / float(D);
    __syncthreads();
    float var = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = xr[d] - mean;
        var += diff * diff;
    }
    const float inv = rsqrtf(block_sum256(var, red) / float(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float o = (xr[d] - mean) * inv * w[d] + b[d];
        if (y_f16) y_f16[size_t(row) * D + d] = __float2half_rn(o);
        if (y_f32) y_f32[size_t(row) * D + d] = o;
    }
}

// Interleaved RoPE + KV ring scatter for streaming post_module.
// qkv_f32: [M, S, 3072] (Q|K|V). Writes q_rot_f32[M, S, 1024] and K/V into kv_ring[slot, layer, pos&255, 1024] (fp32).
// Grid: (M * S, 16), Block: 32 (one thread per RoPE pair in head_dim=64).
__global__ void __launch_bounds__(32) k_post_rope_kv_scatter(
    const float* __restrict__ qkv,
    const float2* __restrict__ rope_tbl,
    const int* __restrict__ d_slots,
    const int* __restrict__ d_slot_pos,
    float* __restrict__ q_rot,
    float* __restrict__ k_ring,
    float* __restrict__ v_ring,
    int M, int S, int layer) {
    const int ms = blockIdx.x;
    const int h = blockIdx.y;
    const int j = threadIdx.x;  // 0..31
    if (ms >= M * S || h >= CODEC_N_HEAD || j >= CODEC_HEAD_DIM / 2) return;

    const int m = ms / S;
    const int t = ms % S;
    const int slot = d_slots[m];
    const int pos = d_slot_pos[slot] + t;
    const int pos_c = (pos < CODEC_MAX_ROPE_LEN) ? pos : (CODEC_MAX_ROPE_LEN - 1);
    const float2 cs = rope_tbl[size_t(pos_c) * (CODEC_HEAD_DIM / 2) + size_t(j)];

    const float* row = qkv + size_t(ms) * (3 * CODEC_LATENT_DIM);
    const int hd_off = h * CODEC_HEAD_DIM + 2 * j;

    const float q0 = row[hd_off];
    const float q1 = row[hd_off + 1];
    q_rot[size_t(ms) * CODEC_LATENT_DIM + hd_off] = q0 * cs.x - q1 * cs.y;
    q_rot[size_t(ms) * CODEC_LATENT_DIM + hd_off + 1] = q1 * cs.x + q0 * cs.y;

    const float k0 = row[CODEC_LATENT_DIM + hd_off];
    const float k1 = row[CODEC_LATENT_DIM + hd_off + 1];
    const float kr0 = k0 * cs.x - k1 * cs.y;
    const float kr1 = k1 * cs.x + k0 * cs.y;

    const float v0 = row[2 * CODEC_LATENT_DIM + hd_off];
    const float v1 = row[2 * CODEC_LATENT_DIM + hd_off + 1];

    const size_t ring_idx = ((size_t(slot) * CODEC_POST_LAYERS + size_t(layer)) * CODEC_KV_RING +
                             size_t(pos & (CODEC_KV_RING - 1))) * CODEC_LATENT_DIM + size_t(hd_off);
    k_ring[ring_idx] = kr0;
    k_ring[ring_idx + 1] = kr1;
    v_ring[ring_idx] = v0;
    v_ring[ring_idx + 1] = v1;
}

// Streaming causal window-128 attention reading K/V from per-slot ring buffers.
// Grid: (M * S, 16), Block: 64 (one thread per head_dim=64 element, 2 warps).
__global__ void __launch_bounds__(64) k_post_window_attn_ring(
    const float* __restrict__ q_rot,
    const float* __restrict__ k_ring,
    const float* __restrict__ v_ring,
    const int* __restrict__ d_slots,
    const int* __restrict__ d_slot_pos,
    __half* __restrict__ out_f16,
    float* __restrict__ out_f32,
    int M, int S, int layer) {
    const int ms = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;  // 0..63
    if (ms >= M * S || h >= CODEC_N_HEAD || d >= CODEC_HEAD_DIM) return;

    const int m = ms / S;
    const int t = ms % S;
    const int slot = d_slots[m];
    const int cur_pos = d_slot_pos[slot] + t;
    const int k_start = (cur_pos >= CODEC_POST_WINDOW - 1) ? (cur_pos - CODEC_POST_WINDOW + 1) : 0;
    const int k_len = cur_pos - k_start + 1;

    const float qv = q_rot[size_t(ms) * CODEC_LATENT_DIM + h * CODEC_HEAD_DIM + d];
    const size_t layer_base = (size_t(slot) * CODEC_POST_LAYERS + size_t(layer)) * CODEC_KV_RING * CODEC_LATENT_DIM;
    const int hd_base = h * CODEC_HEAD_DIM + d;

    __shared__ float sm[2];
    __shared__ float scores[CODEC_POST_WINDOW];
    const float scale = 0.125f;  // 1 / sqrt(64)

    float max_s = -1e30f;
    for (int i = 0; i < k_len; ++i) {
        const int kp = k_start + i;
        const size_t koff = layer_base + size_t(kp & (CODEC_KV_RING - 1)) * CODEC_LATENT_DIM + hd_base;
        float prod = qv * k_ring[koff];
        prod = warp_sum(prod);
        if ((d & 31) == 0) sm[d >> 5] = prod;
        __syncthreads();
        if (d == 0) {
            const float sc = (sm[0] + sm[1]) * scale;
            scores[i] = sc;
            if (sc > max_s) max_s = sc;
        }
        __syncthreads();
    }
    if (d == 0) sm[0] = max_s;
    __syncthreads();
    max_s = sm[0];
    __syncthreads();

    float sum_e = 0.0f;
    for (int i = d; i < k_len; i += 64) {
        const float e = expf(scores[i] - max_s);
        scores[i] = e;
        sum_e += e;
    }
    sum_e = warp_sum(sum_e);
    if ((d & 31) == 0) sm[d >> 5] = sum_e;
    __syncthreads();
    const float inv_sum = 1.0f / (sm[0] + sm[1]);

    float acc = 0.0f;
    for (int i = 0; i < k_len; ++i) {
        const int kp = k_start + i;
        const size_t voff = layer_base + size_t(kp & (CODEC_KV_RING - 1)) * CODEC_LATENT_DIM + hd_base;
        acc += (scores[i] * inv_sum) * v_ring[voff];
    }
    const size_t oidx = size_t(ms) * CODEC_LATENT_DIM + hd_base;
    if (out_f16) out_f16[oidx] = __float2half_rn(acc);
    if (out_f32) out_f32[oidx] = acc;
}

// Advance per-slot frame position after post_module. Grid: 1, Block: min(256, M).
__global__ void k_advance_slot_pos(const int* __restrict__ d_slots, int* __restrict__ d_slot_pos, int M, int S) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M) d_slot_pos[d_slots[m]] += S;
}

// Whole-sequence interleaved RoPE for [T, 3072] (Q|K|V) -> Q_rot[T,1024], K_rot[T,1024], V[T,1024] (fp32).
// Grid: (T, 16), Block: 32.
__global__ void __launch_bounds__(32) k_whole_rope_split(
    const float* __restrict__ qkv,
    const float2* __restrict__ rope_tbl,
    float* __restrict__ q_rot,
    float* __restrict__ k_rot,
    float* __restrict__ v_out,
    int T) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int j = threadIdx.x;
    if (t >= T || h >= CODEC_N_HEAD || j >= CODEC_HEAD_DIM / 2) return;

    const int tc = (t < CODEC_MAX_ROPE_LEN) ? t : (CODEC_MAX_ROPE_LEN - 1);
    const float2 cs = rope_tbl[size_t(tc) * (CODEC_HEAD_DIM / 2) + size_t(j)];
    const float* row = qkv + size_t(t) * (3 * CODEC_LATENT_DIM);
    const int hd_off = h * CODEC_HEAD_DIM + 2 * j;
    const size_t dst = size_t(t) * CODEC_LATENT_DIM + hd_off;

    const float q0 = row[hd_off], q1 = row[hd_off + 1];
    q_rot[dst] = q0 * cs.x - q1 * cs.y;
    q_rot[dst + 1] = q1 * cs.x + q0 * cs.y;

    const float k0 = row[CODEC_LATENT_DIM + hd_off], k1 = row[CODEC_LATENT_DIM + hd_off + 1];
    k_rot[dst] = k0 * cs.x - k1 * cs.y;
    k_rot[dst + 1] = k1 * cs.x + k0 * cs.y;

    v_out[dst] = row[2 * CODEC_LATENT_DIM + hd_off];
    v_out[dst + 1] = row[2 * CODEC_LATENT_DIM + hd_off + 1];
}

// Whole-sequence causal window attention (window_size <= 512) using exact 2-pass softmax in fp32,
// matching k_post_window_attn_ring instruction-for-instruction so streaming and whole-sequence post_module agree.
// Grid: (T, 16), Block: 64.
__global__ void __launch_bounds__(64) k_whole_window_attn(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    __half* __restrict__ out_f16,
    float* __restrict__ out_f32,
    int T, int window_size) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (t >= T || h >= CODEC_N_HEAD || d >= CODEC_HEAD_DIM) return;

    const int hd_base = h * CODEC_HEAD_DIM + d;
    const float qv = q[size_t(t) * CODEC_LATENT_DIM + hd_base];
    const int w = (window_size <= CODEC_ENC_WINDOW) ? window_size : CODEC_ENC_WINDOW;
    const int k_start = (t >= w - 1) ? (t - w + 1) : 0;
    const int k_len = t - k_start + 1;

    __shared__ float sm[2];
    __shared__ float scores[CODEC_ENC_WINDOW];
    const float scale = 0.125f;

    float max_s = -1e30f;
    for (int i = 0; i < k_len; ++i) {
        const int kp = k_start + i;
        const size_t koff = size_t(kp) * CODEC_LATENT_DIM + hd_base;
        float prod = qv * k[koff];
        prod = warp_sum(prod);
        if ((d & 31) == 0) sm[d >> 5] = prod;
        __syncthreads();
        if (d == 0) {
            const float sc = (sm[0] + sm[1]) * scale;
            scores[i] = sc;
            if (sc > max_s) max_s = sc;
        }
        __syncthreads();
    }
    if (d == 0) sm[0] = max_s;
    __syncthreads();
    max_s = sm[0];
    __syncthreads();

    float sum_e = 0.0f;
    for (int i = d; i < k_len; i += 64) {
        const float e = expf(scores[i] - max_s);
        scores[i] = e;
        sum_e += e;
    }
    sum_e = warp_sum(sum_e);
    if ((d & 31) == 0) sm[d >> 5] = sum_e;
    __syncthreads();
    const float inv_sum = 1.0f / (sm[0] + sm[1]);

    float acc = 0.0f;
    for (int i = 0; i < k_len; ++i) {
        const int kp = k_start + i;
        const size_t voff = size_t(kp) * CODEC_LATENT_DIM + hd_base;
        acc += (scores[i] * inv_sum) * v[voff];
    }

    const size_t oidx = size_t(t) * CODEC_LATENT_DIM + hd_base;
    if (out_f16) out_f16[oidx] = __float2half_rn(acc);
    if (out_f32) out_f32[oidx] = acc;
}

// SwiGLU activation on concatenated [N, 2*I] (h1 | h3) -> [N, I] (fp16 and/or fp32).
__global__ void __launch_bounds__(256) k_swiglu(
    const float* __restrict__ h13,
    __half* __restrict__ out_f16,
    float* __restrict__ out_f32,
    int N, int I) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(N) * size_t(I);
    if (idx >= total) return;
    const size_t row = idx / size_t(I);
    const size_t col = idx % size_t(I);
    const float a = h13[row * size_t(2 * I) + col];
    const float b = h13[row * size_t(2 * I) + size_t(I) + col];
    const float silu = a / (1.0f + expf(-a));
    const float y = silu * b;
    if (out_f16) out_f16[idx] = __float2half_rn(y);
    if (out_f32) out_f32[idx] = y;
}

// Residual += gamma * delta (LayerScale + residual add in fp32).
__global__ void __launch_bounds__(256) k_layerscale_add(
    float* __restrict__ x,
    const float* __restrict__ delta,
    const float* __restrict__ gamma,
    int N, int D) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(N) * size_t(D);
    if (idx >= total) return;
    x[idx] += gamma[idx % size_t(D)] * delta[idx];
}

// Add channel bias (and optional residual add) in fp32.
__global__ void __launch_bounds__(256) k_add_bias_res(
    float* __restrict__ y,
    const float* __restrict__ bias,
    const float* __restrict__ res,
    int N, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(N) * size_t(C);
    if (idx >= total) return;
    float v = y[idx] + bias[idx % size_t(C)];
    if (res) v += res[idx];
    y[idx] = v;
}

// Bias + exact GELU -> fp16 and/or fp32 (for ConvNeXt pwconv1).
__global__ void __launch_bounds__(256) k_bias_gelu(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    __half* __restrict__ out_f16,
    float* __restrict__ out_f32,
    int N, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(N) * size_t(C);
    if (idx >= total) return;
    const float v = x[idx] + bias[idx % size_t(C)];
    const float g = 0.5f * v * (1.0f + erff(v * 0.7071067811865475244f));
    if (out_f16) out_f16[idx] = __float2half_rn(g);
    if (out_f32) out_f32[idx] = g;
}

// Bias + LayerScale gamma + residual add (for ConvNeXt pwconv2).
__global__ void __launch_bounds__(256) k_convnext_epilogue(
    const float* __restrict__ pw2,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ res,
    float* __restrict__ out,
    int N, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(N) * size_t(C);
    if (idx >= total) return;
    const int c = static_cast<int>(idx % size_t(C));
    out[idx] = res[idx] + gamma[c] * (pw2[idx] + bias[c]);
}

// Snake1d in fp32: y = x + sin^2(alpha * x) / (alpha + 1e-9), optionally emitting fp16 copy too.
__global__ void __launch_bounds__(256) k_snake1d(
    const float* __restrict__ x,
    const float* __restrict__ alpha,
    float* __restrict__ y_f32,
    __half* __restrict__ y_f16,
    int N, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(N) * size_t(C);
    if (idx >= total) return;
    const float a = alpha[idx % size_t(C)];
    const float xv = x[idx];
    const float s = sinf(a * xv);
    const float y = xv + (s * s) / (a + 1e-9f);
    if (y_f32) y_f32[idx] = y;
    if (y_f16) y_f16[idx] = __float2half_rn(y);
}

// Cast fp32 -> fp16.
__global__ void __launch_bounds__(256) k_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half_rn(src[idx]);
}

// Prepend causal conv context [pad, C] from slot arena (or zeros if ctx_base == nullptr) + current [T, C]
// into x_pad[M, T + pad, C] (emitting fp32 and/or fp16).
__global__ void __launch_bounds__(256) k_conv_prepend_ctx(
    const float* __restrict__ x,
    const float* __restrict__ slot_ctx_arena,
    const int* __restrict__ d_slots,
    size_t slot_stride_elems,
    size_t ctx_off,
    float* __restrict__ pad_f32,
    __half* __restrict__ pad_f16,
    int M, int T, int pad, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t Tpad = size_t(T + pad);
    const size_t total = size_t(M) * Tpad * size_t(C);
    if (idx >= total) return;

    const size_t c = idx % size_t(C);
    const size_t rem = idx / size_t(C);
    const size_t t = rem % Tpad;
    const int m = static_cast<int>(rem / Tpad);

    float v = 0.0f;
    if (t < size_t(pad)) {
        if (slot_ctx_arena) {
            const int slot = d_slots[m];
            v = slot_ctx_arena[size_t(slot) * slot_stride_elems + ctx_off + t * size_t(C) + c];
        }
    } else {
        v = x[(size_t(m) * size_t(T) + (t - size_t(pad))) * size_t(C) + c];
    }
    if (pad_f32) pad_f32[idx] = v;
    if (pad_f16) pad_f16[idx] = __float2half_rn(v);
}

// Save last `pad` fp32 frames from pad_f32[M, T + pad, C] (indices T .. T + pad - 1) back to slot arena.
// Works identically for both T >= pad and T < pad with zero data race!
__global__ void __launch_bounds__(256) k_conv_save_ctx(
    const float* __restrict__ pad_f32,
    float* __restrict__ slot_ctx_arena,
    const int* __restrict__ d_slots,
    size_t slot_stride_elems,
    size_t ctx_off,
    int M, int T, int pad, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(M) * size_t(pad) * size_t(C);
    if (idx >= total) return;
    const size_t c = idx % size_t(C);
    const size_t rem = idx / size_t(C);
    const size_t p = rem % size_t(pad);
    const int m = static_cast<int>(rem / size_t(pad));
    const int slot = d_slots[m];
    const size_t src = (size_t(m) * size_t(T + pad) + size_t(T) + p) * size_t(C) + c;
    slot_ctx_arena[size_t(slot) * slot_stride_elems + ctx_off + p * size_t(C) + c] = pad_f32[src];
}

// Save last `pad` frames directly from (old ctx + x) when pad_f32 wasn't materialized (only pad_f16 was).
// Copies from pad_f16? No — better: we read old ctx + x directly into a temp? No: to avoid race when T < pad,
// simply always materialize pad_f32 when streaming (tiny, max 24 MiB) OR use k_conv_save_ctx on pad_f32!
// Since we already have buf_pad_f32 pre-allocated, k_conv_save_ctx from pad_f32 is exact fp32 and race-free!

// Causal depthwise Conv1d k=7, s=1, d=1 in fp32.
// x_pad: [M, T + 6, C], w: [7, C] (tap-major), bias: [C], out: [M, T, C].
__global__ void __launch_bounds__(256) k_dwconv_k7(
    const float* __restrict__ x_pad,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M, int T, int C) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(M) * size_t(T) * size_t(C);
    if (idx >= total) return;
    const int c = static_cast<int>(idx % size_t(C));
    const size_t rem = idx / size_t(C);
    const int t = static_cast<int>(rem % size_t(T));
    const int m = static_cast<int>(rem / size_t(T));

    const float* xp = x_pad + (size_t(m) * size_t(T + 6) + size_t(t)) * size_t(C) + c;
    float acc = bias[c];
    #pragma unroll
    for (int k = 0; k < 7; ++k) {
        acc += xp[size_t(k) * size_t(C)] * w[size_t(k) * size_t(C) + c];
    }
    out[idx] = acc;
}

// CausalTransConv1d overlap-add (k = 2 * stride, carry = stride) + bias in fp32.
// gemm: [M, T_in, 2*s, C_out], carry_arena: [max_slots, carry_total] (or nullptr for whole-seq).
// out:  [M, T_in * s, C_out].
__global__ void __launch_bounds__(256) k_transconv_overlap_add(
    const float* __restrict__ gemm,
    const float* __restrict__ bias,
    const float* __restrict__ slot_carry_arena,
    const int* __restrict__ d_slots,
    size_t carry_stride_elems,
    size_t carry_off,
    float* __restrict__ out,
    int M, int T_in, int s, int C_out) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t T_out = size_t(T_in) * size_t(s);
    const size_t total = size_t(M) * T_out * size_t(C_out);
    if (idx >= total) return;

    const int o = static_cast<int>(idx % size_t(C_out));
    const size_t rem = idx / size_t(C_out);
    const size_t to = rem % T_out;
    const int m = static_cast<int>(rem / T_out);
    const int t = static_cast<int>(to / size_t(s));
    const int r = static_cast<int>(to % size_t(s));

    const size_t k2 = size_t(2 * s);
    const float cur = gemm[((size_t(m) * size_t(T_in) + size_t(t)) * k2 + size_t(r)) * size_t(C_out) + o];
    float prev = 0.0f;
    if (t > 0) {
        prev = gemm[((size_t(m) * size_t(T_in) + size_t(t - 1)) * k2 + size_t(s + r)) * size_t(C_out) + o];
    } else if (slot_carry_arena) {
        const int slot = d_slots[m];
        prev = slot_carry_arena[size_t(slot) * carry_stride_elems + carry_off + size_t(r) * size_t(C_out) + o];
    }
    out[idx] = cur + prev + bias[o];
}

// Save transconv carry tail gemm[m, T_in - 1, s .. 2*s - 1, o] -> slot_carry_arena[slot, r, o].
__global__ void __launch_bounds__(256) k_transconv_save_carry(
    const float* __restrict__ gemm,
    float* __restrict__ slot_carry_arena,
    const int* __restrict__ d_slots,
    size_t carry_stride_elems,
    size_t carry_off,
    int M, int T_in, int s, int C_out) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(M) * size_t(s) * size_t(C_out);
    if (idx >= total) return;

    const int o = static_cast<int>(idx % size_t(C_out));
    const size_t rem = idx / size_t(C_out);
    const int r = static_cast<int>(rem % size_t(s));
    const int m = static_cast<int>(rem / size_t(s));
    const int slot = d_slots[m];
    const size_t k2 = size_t(2 * s);
    const size_t src = ((size_t(m) * size_t(T_in) + size_t(T_in - 1)) * k2 + size_t(s + r)) * size_t(C_out) + o;
    slot_carry_arena[size_t(slot) * carry_stride_elems + carry_off + size_t(r) * size_t(C_out) + o] = gemm[src];
}

// Final decoder conv1d (C_in=96 -> 1, k=7, s=1, d=1) + bias + Tanh -> int16 PCM in fp32.
// x_pad: [M, T + 6, 96], w: [7, 96], bias: [1], out_s16: [M, T].
__global__ void __launch_bounds__(256) k_dec_final_conv_tanh_s16(
    const float* __restrict__ x_pad,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    int16_t* __restrict__ out_s16,
    int M, int T) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(M) * size_t(T);
    if (idx >= total) return;

    const int t = static_cast<int>(idx % size_t(T));
    const int m = static_cast<int>(idx / size_t(T));
    const float* xp = x_pad + (size_t(m) * size_t(T + 6) + size_t(t)) * 96u;
    float acc = bias[0];
    for (int k = 0; k < 7; ++k) {
        const float* xk = xp + size_t(k) * 96u;
        const float* wk = w + size_t(k) * 96u;
        for (int c = 0; c < 96; ++c) {
            acc += xk[c] * wk[c];
        }
    }
    const float y = tanhf(acc);
    const float scaled = fmaxf(-1.0f, fminf(1.0f, y)) * 32767.0f;
    out_s16[idx] = static_cast<int16_t>(lrintf(scaled));
}

// Encoder first conv1d (C_in=1 -> 64, k=7, s=1, d=1, pad_left=6) in fp32.
// x: [T] (fp32), w: [7, 64] (tap-major), bias: [64], out: [T, 64].
__global__ void __launch_bounds__(256) k_enc_first_conv(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int T) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(T) * 64u;
    if (idx >= total) return;
    const int c = static_cast<int>(idx & 63u);
    const int t = static_cast<int>(idx >> 6);
    float acc = bias[c];
    #pragma unroll
    for (int k = 0; k < 7; ++k) {
        const int src_t = t - 6 + k;
        if (src_t >= 0) acc += x[src_t] * w[k * 64 + c];
    }
    out[idx] = acc;
}

// RVQ stage kernel (fp32): given z_e[F, 8] (already projected by in_proj + bias), L2-normalize (eps=1e-12),
// find argmin_k (||e||^2 - 2 e.c + ||c||^2) over V codebook rows (4096 or 1024), write out_codes[q*F + f],
// and subtract projected_codes[proj_offset + best_k, :] from residual[f, :].
// Grid: (F), Block: 256.
__global__ void __launch_bounds__(256) k_rvq_stage_quantize_sub(
    const float* __restrict__ z_e,
    const float* __restrict__ cb_norm,
    const float* __restrict__ cb_norm_sq,
    const float* __restrict__ projected_codes,
    float* __restrict__ residual,
    int32_t* __restrict__ out_codes,
    int F, int q, int V, int proj_offset) {
    const int f = blockIdx.x;
    if (f >= F) return;

    __shared__ float e_norm[CODEC_CB_DIM];
    __shared__ float e_norm_sq;
    __shared__ float best_val[256];
    __shared__ int best_idx[256];

    if (threadIdx.x == 0) {
        const float* ze = z_e + size_t(f) * CODEC_CB_DIM;
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < CODEC_CB_DIM; ++i) ss += ze[i] * ze[i];
        const float denom = fmaxf(sqrtf(ss), 1e-12f);
        const float inv = 1.0f / denom;
        float nsq = 0.0f;
        #pragma unroll
        for (int i = 0; i < CODEC_CB_DIM; ++i) {
            float v = ze[i] * inv;
            e_norm[i] = v;
            nsq += v * v;
        }
        e_norm_sq = nsq;
    }
    __syncthreads();

    float local_best = -1e30f;
    int local_idx = 0;
    const float ensq = e_norm_sq;
    for (int k = threadIdx.x; k < V; k += blockDim.x) {
        const float* ck = cb_norm + size_t(k) * CODEC_CB_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < CODEC_CB_DIM; ++i) dot += e_norm[i] * ck[i];
        const float neg_dist = -(ensq - 2.0f * dot + cb_norm_sq[k]);
        if (neg_dist > local_best || (neg_dist == local_best && k < local_idx)) {
            local_best = neg_dist;
            local_idx = k;
        }
    }
    best_val[threadIdx.x] = local_best;
    best_idx[threadIdx.x] = local_idx;
    __syncthreads();

    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float bv = best_val[threadIdx.x + stride];
            const int bi = best_idx[threadIdx.x + stride];
            if (bv > best_val[threadIdx.x] || (bv == best_val[threadIdx.x] && bi < best_idx[threadIdx.x])) {
                best_val[threadIdx.x] = bv;
                best_idx[threadIdx.x] = bi;
            }
        }
        __syncthreads();
    }

    const int chosen = best_idx[0];
    if (threadIdx.x == 0) {
        out_codes[size_t(q) * size_t(F) + size_t(f)] = chosen;
    }
    const float* p_row = projected_codes + size_t(proj_offset + chosen) * CODEC_LATENT_DIM;
    float* res_row = residual + size_t(f) * CODEC_LATENT_DIM;
    for (int d = threadIdx.x; d < CODEC_LATENT_DIM; d += blockDim.x) {
        res_row[d] -= p_row[d];
    }
}

// ------------------------------------------------------------------------- Weight Structs

struct TransformerLayerW {
    const float* attn_norm = nullptr;      // [1024]
    const void* wqkv = nullptr;            // [3072, 1024] (fp16 for decoder, fp32 for encoder)
    const void* wo = nullptr;              // [1024, 1024]
    const float* attn_gamma = nullptr;     // [1024]
    const float* ffn_norm = nullptr;       // [1024]
    const void* w13 = nullptr;             // [6144, 1024] (w1 concat w3)
    const void* w2 = nullptr;              // [1024, 3072]
    const float* ffn_gamma = nullptr;      // [1024]
};

struct TransformerW {
    std::vector<TransformerLayerW> layers;
    const float* final_norm = nullptr;     // [1024]
};

struct ConvNeXtW {
    const float* dw_w = nullptr;           // [7, 1024] fp32 tap-major
    const float* dw_b = nullptr;           // [1024] fp32
    const float* ln_w = nullptr;           // [1024] fp32
    const float* ln_b = nullptr;           // [1024] fp32
    const void* pw1_w = nullptr;           // [4096, 1024] (fp16 dec / fp32 enc)
    const float* pw1_b = nullptr;          // [4096] fp32
    const void* pw2_w = nullptr;           // [1024, 4096] (fp16 dec / fp32 enc)
    const float* pw2_b = nullptr;          // [1024] fp32
    const float* gamma = nullptr;          // [1024] fp32
};

struct ResUnitW {
    const float* snake0 = nullptr;         // [C] fp32
    const void* conv7_w = nullptr;         // [7, C, C] tap-major (fp16 dec / fp32 enc)
    const float* conv7_b = nullptr;        // [C] fp32
    const float* snake1 = nullptr;         // [C] fp32
    const void* conv1_w = nullptr;         // [C, C] (fp16 dec / fp32 enc)
    const float* conv1_b = nullptr;        // [C] fp32
};

struct DecoderBlockW {
    const float* snake0 = nullptr;         // [in_c] fp32
    const void* trans_w = nullptr;         // [2*s * out_c, in_c] (fp16)
    const float* trans_b = nullptr;        // [out_c] fp32
    ResUnitW res[3];
};

struct EncoderBlockW {
    ResUnitW res[3];
    const float* snake3 = nullptr;         // [in_c] fp32
    const float* conv_w = nullptr;         // [2*s, out_c, in_c] tap-major fp32
    const float* conv_b = nullptr;         // [out_c] fp32
};

struct RvqStageW {
    const float* in_proj_w = nullptr;      // [8, 1024] fp32
    const float* in_proj_b = nullptr;      // [8] fp32
    const float* cb_norm = nullptr;        // [V, 8] fp32
    const float* cb_norm_sq = nullptr;     // [V] fp32
    int vocab_size = 0;
    int proj_offset = 0;
};

struct DeviceArena {
    uint8_t* base = nullptr;
    size_t cap = 0;
    size_t used = 0;

    static size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

    void alloc_bytes(size_t bytes) {
        release();
        cap = align256(bytes);
        if (cap > 0) G6TTS_CODEC_CK(cudaMalloc(&base, cap));
        used = 0;
    }
    void* reserve(size_t bytes) {
        size_t off = align256(used);
        if (off + bytes > cap) throw std::runtime_error("tts_codec: device arena overflow");
        used = off + bytes;
        return base + off;
    }
    void release() {
        if (base) {
            cudaFree(base);
            base = nullptr;
            cap = 0;
            used = 0;
        }
    }
    ~DeviceArena() { release(); }
    DeviceArena() = default;
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& o) noexcept : base(o.base), cap(o.cap), used(o.used) {
        o.base = nullptr; o.cap = 0; o.used = 0;
    }
    DeviceArena& operator=(DeviceArena&& o) noexcept {
        if (this != &o) {
            release();
            base = o.base; cap = o.cap; used = o.used;
            o.base = nullptr; o.cap = 0; o.used = 0;
        }
        return *this;
    }
};

}  // namespace codec_detail

// ------------------------------------------------------------------------- Public TtsCodec API

struct TtsCodec {
    TtsCodec() = default;
    ~TtsCodec() { release(); }
    TtsCodec(const TtsCodec&) = delete;
    TtsCodec& operator=(const TtsCodec&) = delete;
    TtsCodec(TtsCodec&& o) noexcept { *this = std::move(o); }
    TtsCodec& operator=(TtsCodec&& o) noexcept {
        if (this == &o) return *this;
        release();
        loaded_ = o.loaded_; o.loaded_ = false;
        fp32_decoder_ = o.fp32_decoder_;
        max_slots_ = o.max_slots_; o.max_slots_ = 0;
        max_decode_stride_ = o.max_decode_stride_; o.max_decode_stride_ = 0;
        cublas_ = o.cublas_; o.cublas_ = nullptr;
        cublas_ws_ = o.cublas_ws_; o.cublas_ws_ = nullptr;
        cublas_ws_bytes_ = o.cublas_ws_bytes_; o.cublas_ws_bytes_ = 0;
        w_arena_ = std::move(o.w_arena_);
        s_arena_ = std::move(o.s_arena_);
        scratch_arena_ = std::move(o.scratch_arena_);
        projected_codes_ = o.projected_codes_;
        rope_tbl_ = o.rope_tbl_;
        post_mod_ = std::move(o.post_mod_);
        pre_mod_ = std::move(o.pre_mod_);
        enc_t_ = std::move(o.enc_t_);
        for (int i = 0; i < 2; ++i) {
            up_w_[i] = o.up_w_[i]; up_b_[i] = o.up_b_[i]; up_cn_[i] = o.up_cn_[i];
            dn_w_[i] = o.dn_w_[i]; dn_b_[i] = o.dn_b_[i]; dn_cn_[i] = o.dn_cn_[i];
        }
        dec_in_w_ = o.dec_in_w_; dec_in_b_ = o.dec_in_b_;
        for (int i = 0; i < 4; ++i) dec_blk_[i] = o.dec_blk_[i];
        dec_out_snake_ = o.dec_out_snake_;
        dec_out_w_ = o.dec_out_w_; dec_out_b_ = o.dec_out_b_;
        enc_in_w_ = o.enc_in_w_; enc_in_b_ = o.enc_in_b_;
        for (int i = 0; i < 4; ++i) enc_blk_[i] = o.enc_blk_[i];
        enc_out_snake_ = o.enc_out_snake_;
        enc_out_w_ = o.enc_out_w_; enc_out_b_ = o.enc_out_b_;
        for (int i = 0; i < 10; ++i) rvq_[i] = o.rvq_[i];
        d_slot_conv_ctx_ = o.d_slot_conv_ctx_;
        d_slot_carry_ = o.d_slot_carry_;
        d_k_ring_ = o.d_k_ring_;
        d_v_ring_ = o.d_v_ring_;
        d_slot_pos_ = o.d_slot_pos_;
        dec_buf_a_f32_ = o.dec_buf_a_f32_;
        dec_buf_b_f32_ = o.dec_buf_b_f32_;
        dec_buf_gemm_f32_ = o.dec_buf_gemm_f32_;
        dec_buf_pad_f32_ = o.dec_buf_pad_f32_;
        dec_buf_pad_f16_ = o.dec_buf_pad_f16_;
        dec_buf_tmp_f16_ = o.dec_buf_tmp_f16_;
        dec_buf_qrot_f32_ = o.dec_buf_qrot_f32_;
        return *this;
    }

    bool loaded() const { return loaded_; }
    int max_slots() const { return max_slots_; }
    int max_decode_stride() const { return max_decode_stride_; }

    void release() {
        loaded_ = false;
        if (cublas_) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
        if (cublas_ws_) {
            cudaFree(cublas_ws_);
            cublas_ws_ = nullptr;
            cublas_ws_bytes_ = 0;
        }
        scratch_arena_.release();
        s_arena_.release();
        w_arena_.release();
    }

    bool init(const std::string& path, int max_slots, int max_decode_stride = 4, bool fp32_decoder = true) {
        using namespace codec_detail;
        release();
        if (max_slots <= 0 || max_decode_stride <= 0 || max_decode_stride > 128) {
            throw std::runtime_error("tts_codec: invalid max_slots or max_decode_stride");
        }
        max_slots_ = max_slots;
        max_decode_stride_ = max_decode_stride;
        fp32_decoder_ = fp32_decoder;

        // 1. Parse safetensors header & validate format metadata
        const StHeader hdr = parse_safetensors_header(path);
        auto fmt_it = hdr.metadata.find("format");
        if (fmt_it == hdr.metadata.end() || fmt_it->second != "exlr8-codec-v1") {
            throw std::runtime_error("tts_codec: expected metadata format=exlr8-codec-v1 in " + path);
        }

        // 2. Load raw fp32 tensors from file via buffered fread (64K-page safe, no mmap->GPU)
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("tts_codec: cannot reopen " + path);
        struct FileGuard { FILE* fp; ~FileGuard() { if (fp) std::fclose(fp); } } fg{f};

        auto read_f32 = [&](const std::string& name, const std::vector<int64_t>& want_shape) -> std::vector<float> {
            auto it = hdr.tensors.find(name);
            if (it == hdr.tensors.end()) throw std::runtime_error("tts_codec: missing tensor " + name);
            const StTensorInfo& ti = it->second;
            if (ti.dtype != "F32") throw std::runtime_error("tts_codec: expected F32 for " + name + ", got " + ti.dtype);
            if (ti.shape != want_shape) {
                std::ostringstream oss;
                oss << "tts_codec: shape mismatch for " << name << ": got [";
                for (size_t i = 0; i < ti.shape.size(); ++i) oss << (i ? "," : "") << ti.shape[i];
                oss << "], want [";
                for (size_t i = 0; i < want_shape.size(); ++i) oss << (i ? "," : "") << want_shape[i];
                oss << "]";
                throw std::runtime_error(oss.str());
            }
            const size_t nbytes = ti.end - ti.begin;
            std::vector<float> buf(nbytes / sizeof(float));
            if (std::fseek(f, static_cast<long>(hdr.data_off + ti.begin), SEEK_SET) != 0 ||
                std::fread(buf.data(), 1, nbytes, f) != nbytes) {
                throw std::runtime_error("tts_codec: read failed for " + name);
            }
            return buf;
        };

        // Host staging list for single-arena GPU upload (pinned staging chunks)
        struct UploadItem {
            std::vector<uint8_t> bytes;
            void** dst_ptr = nullptr;
            size_t arena_off = 0;
        };
        std::vector<UploadItem> uploads;
        uploads.reserve(512);
        size_t total_w_bytes = 0;

        auto stage_f32 = [&](const std::vector<float>& src, const float** dst) {
            UploadItem u;
            u.bytes.resize(src.size() * sizeof(float));
            std::memcpy(u.bytes.data(), src.data(), u.bytes.size());
            u.dst_ptr = reinterpret_cast<void**>(const_cast<float**>(dst));
            u.arena_off = DeviceArena::align256(total_w_bytes);
            total_w_bytes = u.arena_off + u.bytes.size();
            uploads.push_back(std::move(u));
        };
        auto stage_dec_mat = [&](const std::vector<float>& src, const void** dst) {
            UploadItem u;
            if (fp32_decoder_) {
                u.bytes.resize(src.size() * sizeof(float));
                std::memcpy(u.bytes.data(), src.data(), u.bytes.size());
            } else {
                u.bytes.resize(src.size() * sizeof(__half));
                auto* hp = reinterpret_cast<__half*>(u.bytes.data());
                for (size_t i = 0; i < src.size(); ++i) hp[i] = __float2half_rn(src[i]);
            }
            u.dst_ptr = const_cast<void**>(dst);
            u.arena_off = DeviceArena::align256(total_w_bytes);
            total_w_bytes = u.arena_off + u.bytes.size();
            uploads.push_back(std::move(u));
        };
        auto stage_enc_mat = [&](const std::vector<float>& src, const void** dst) {
            stage_f32(src, reinterpret_cast<const float**>(dst));
        };

        // Weight transpose helpers:
        // Conv1d [C_out, C_in, K] -> tap-major [K, C_out, C_in]
        auto pack_conv1d = [](const std::vector<float>& w, int co, int ci, int K) {
            std::vector<float> out(w.size());
            for (int o = 0; o < co; ++o) {
                for (int i = 0; i < ci; ++i) {
                    for (int k = 0; k < K; ++k) {
                        out[(size_t(k) * size_t(co) + size_t(o)) * size_t(ci) + size_t(i)] =
                            w[(size_t(o) * size_t(ci) + size_t(i)) * size_t(K) + size_t(k)];
                    }
                }
            }
            return out;
        };
        // ConvTranspose1d [C_in, C_out, K] -> [K * C_out, C_in] (for row-major [R, C_in] @ W^T -> [R, K, C_out])
        auto pack_transconv1d = [](const std::vector<float>& w, int ci, int co, int K) {
            std::vector<float> out(w.size());
            for (int i = 0; i < ci; ++i) {
                for (int o = 0; o < co; ++o) {
                    for (int k = 0; k < K; ++k) {
                        out[(size_t(k) * size_t(co) + size_t(o)) * size_t(ci) + size_t(i)] =
                            w[(size_t(i) * size_t(co) + size_t(o)) * size_t(K) + size_t(k)];
                    }
                }
            }
            return out;
        };
        // Depthwise Conv1d [C, 1, K] -> tap-major [K, C]
        auto pack_dwconv = [](const std::vector<float>& w, int C, int K) {
            std::vector<float> out(w.size());
            for (int c = 0; c < C; ++c) {
                for (int k = 0; k < K; ++k) {
                    out[size_t(k) * size_t(C) + size_t(c)] = w[size_t(c) * size_t(K) + size_t(k)];
                }
            }
            return out;
        };

        auto load_transformer = [&](const std::string& pfx, int n_layers, bool is_dec, TransformerW& tw) {
            tw.layers.resize(size_t(n_layers));
            for (int l = 0; l < n_layers; ++l) {
                const std::string lp = pfx + ".layers." + std::to_string(l);
                auto& lw = tw.layers[size_t(l)];
                stage_f32(read_f32(lp + ".attention_norm.weight", {1024}), &lw.attn_norm);
                auto wqkv = read_f32(lp + ".attention.wqkv.weight", {3072, 1024});
                auto wo = read_f32(lp + ".attention.wo.weight", {1024, 1024});
                stage_f32(read_f32(lp + ".attention_layer_scale.gamma", {1024}), &lw.attn_gamma);
                stage_f32(read_f32(lp + ".ffn_norm.weight", {1024}), &lw.ffn_norm);
                auto w1 = read_f32(lp + ".feed_forward.w1.weight", {3072, 1024});
                auto w3 = read_f32(lp + ".feed_forward.w3.weight", {3072, 1024});
                std::vector<float> w13;
                w13.reserve(w1.size() + w3.size());
                w13.insert(w13.end(), w1.begin(), w1.end());
                w13.insert(w13.end(), w3.begin(), w3.end());
                auto w2 = read_f32(lp + ".feed_forward.w2.weight", {1024, 3072});
                stage_f32(read_f32(lp + ".ffn_layer_scale.gamma", {1024}), &lw.ffn_gamma);
                if (is_dec) {
                    stage_dec_mat(wqkv, &lw.wqkv);
                    stage_dec_mat(wo, &lw.wo);
                    stage_dec_mat(w13, &lw.w13);
                    stage_dec_mat(w2, &lw.w2);
                } else {
                    stage_enc_mat(wqkv, &lw.wqkv);
                    stage_enc_mat(wo, &lw.wo);
                    stage_enc_mat(w13, &lw.w13);
                    stage_enc_mat(w2, &lw.w2);
                }
            }
            stage_f32(read_f32(pfx + ".norm.weight", {1024}), &tw.final_norm);
        };

        auto load_convnext = [&](const std::string& pfx, bool is_dec, ConvNeXtW& cw) {
            stage_f32(pack_dwconv(read_f32(pfx + ".dwconv.conv.weight", {1024, 1, 7}), 1024, 7), &cw.dw_w);
            stage_f32(read_f32(pfx + ".dwconv.conv.bias", {1024}), &cw.dw_b);
            stage_f32(read_f32(pfx + ".norm.weight", {1024}), &cw.ln_w);
            stage_f32(read_f32(pfx + ".norm.bias", {1024}), &cw.ln_b);
            auto pw1 = read_f32(pfx + ".pwconv1.weight", {4096, 1024});
            stage_f32(read_f32(pfx + ".pwconv1.bias", {4096}), &cw.pw1_b);
            auto pw2 = read_f32(pfx + ".pwconv2.weight", {1024, 4096});
            stage_f32(read_f32(pfx + ".pwconv2.bias", {1024}), &cw.pw2_b);
            stage_f32(read_f32(pfx + ".gamma", {1024}), &cw.gamma);
            if (is_dec) {
                stage_dec_mat(pw1, &cw.pw1_w);
                stage_dec_mat(pw2, &cw.pw2_w);
            } else {
                stage_enc_mat(pw1, &cw.pw1_w);
                stage_enc_mat(pw2, &cw.pw2_w);
            }
        };

        auto load_resunit = [&](const std::string& pfx, int C, bool is_dec, ResUnitW& rw) {
            stage_f32(read_f32(pfx + ".0.alpha", {1, C, 1}), &rw.snake0);
            auto c7 = pack_conv1d(read_f32(pfx + ".1.conv.weight", {C, C, 7}), C, C, 7);
            stage_f32(read_f32(pfx + ".1.conv.bias", {C}), &rw.conv7_b);
            stage_f32(read_f32(pfx + ".2.alpha", {1, C, 1}), &rw.snake1);
            auto c1 = read_f32(pfx + ".3.conv.weight", {C, C, 1});
            stage_f32(read_f32(pfx + ".3.conv.bias", {C}), &rw.conv1_b);
            if (is_dec) {
                stage_dec_mat(c7, &rw.conv7_w);
                stage_dec_mat(c1, &rw.conv1_w);
            } else {
                stage_enc_mat(c7, &rw.conv7_w);
                stage_enc_mat(c1, &rw.conv1_w);
            }
        };

        // 3. Stage projected_codes and bf16-rounded RoPE table
        stage_f32(read_f32("quantizer.projected_codes", {CODEC_PROJ_ROWS, CODEC_LATENT_DIM}), &projected_codes_);
        {
            std::vector<float> rope_host(size_t(CODEC_MAX_ROPE_LEN) * size_t(CODEC_HEAD_DIM));
            double inv_freq[CODEC_HEAD_DIM / 2];
            for (int j = 0; j < CODEC_HEAD_DIM / 2; ++j) {
                inv_freq[j] = 1.0 / std::pow(double(CODEC_ROPE_BASE), double(2 * j) / double(CODEC_HEAD_DIM));
            }
            for (int pos = 0; pos < CODEC_MAX_ROPE_LEN; ++pos) {
                for (int j = 0; j < CODEC_HEAD_DIM / 2; ++j) {
                    const float ang = static_cast<float>(double(pos) * inv_freq[j]);
                    const float cs = round_f32_to_bf16_f32(std::cos(ang));
                    const float sn = round_f32_to_bf16_f32(std::sin(ang));
                    rope_host[(size_t(pos) * (CODEC_HEAD_DIM / 2) + size_t(j)) * 2 + 0] = cs;
                    rope_host[(size_t(pos) * (CODEC_HEAD_DIM / 2) + size_t(j)) * 2 + 1] = sn;
                }
            }
            UploadItem u;
            u.bytes.resize(rope_host.size() * sizeof(float));
            std::memcpy(u.bytes.data(), rope_host.data(), u.bytes.size());
            u.dst_ptr = reinterpret_cast<void**>(const_cast<float2**>(&rope_tbl_));
            u.arena_off = DeviceArena::align256(total_w_bytes);
            total_w_bytes = u.arena_off + u.bytes.size();
            uploads.push_back(std::move(u));
        }

        // 4. Stage decoder weights
        load_transformer("quantizer.post_module", CODEC_POST_LAYERS, true, post_mod_);
        for (int u = 0; u < 2; ++u) {
            const std::string p = "quantizer.upsample." + std::to_string(u);
            auto tw = pack_transconv1d(read_f32(p + ".0.conv.weight", {1024, 1024, 2}), 1024, 1024, 2);
            stage_dec_mat(tw, &up_w_[u]);
            stage_f32(read_f32(p + ".0.conv.bias", {1024}), &up_b_[u]);
            load_convnext(p + ".1", true, up_cn_[u]);
        }
        stage_dec_mat(pack_conv1d(read_f32("decoder.model.0.conv.weight", {1536, 1024, 7}), 1536, 1024, 7), &dec_in_w_);
        stage_f32(read_f32("decoder.model.0.conv.bias", {1536}), &dec_in_b_);
        for (int b = 0; b < 4; ++b) {
            const std::string p = "decoder.model." + std::to_string(b + 1) + ".block";
            const int ic = DEC_IN_CH[b], oc = DEC_OUT_CH[b], k = 2 * DEC_STRIDES[b];
            stage_f32(read_f32(p + ".0.alpha", {1, ic, 1}), &dec_blk_[b].snake0);
            auto tw = pack_transconv1d(read_f32(p + ".1.conv.weight", {ic, oc, k}), ic, oc, k);
            stage_dec_mat(tw, &dec_blk_[b].trans_w);
            stage_f32(read_f32(p + ".1.conv.bias", {oc}), &dec_blk_[b].trans_b);
            for (int r = 0; r < 3; ++r) {
                load_resunit(p + "." + std::to_string(r + 2) + ".block", oc, true, dec_blk_[b].res[r]);
            }
        }
        stage_f32(read_f32("decoder.model.5.alpha", {1, 96, 1}), &dec_out_snake_);
        stage_f32(pack_conv1d(read_f32("decoder.model.6.conv.weight", {1, 96, 7}), 1, 96, 7), &dec_out_w_);
        stage_f32(read_f32("decoder.model.6.conv.bias", {1}), &dec_out_b_);

        // 5. Stage encoder weights (all fp32)
        stage_f32(pack_conv1d(read_f32("encoder.block.0.conv.weight", {64, 1, 7}), 64, 1, 7), &enc_in_w_);
        stage_f32(read_f32("encoder.block.0.conv.bias", {64}), &enc_in_b_);
        for (int b = 0; b < 4; ++b) {
            const std::string p = "encoder.block." + std::to_string(b + 1) + ".block";
            const int ic = ENC_IN_CH[b], oc = ENC_OUT_CH[b], k = 2 * ENC_STRIDES[b];
            for (int r = 0; r < 3; ++r) {
                load_resunit(p + "." + std::to_string(r) + ".block", ic, false, enc_blk_[b].res[r]);
            }
            stage_f32(read_f32(p + ".3.alpha", {1, ic, 1}), &enc_blk_[b].snake3);
            stage_f32(pack_conv1d(read_f32(p + ".4.conv.weight", {oc, ic, k}), oc, ic, k), &enc_blk_[b].conv_w);
            stage_f32(read_f32(p + ".4.conv.bias", {oc}), &enc_blk_[b].conv_b);
        }
        load_transformer("encoder.block.4.block.5", CODEC_ENC_T_LAYERS, false, enc_t_);
        stage_f32(read_f32("encoder.block.5.alpha", {1, 1024, 1}), &enc_out_snake_);
        stage_f32(pack_conv1d(read_f32("encoder.block.6.conv.weight", {1024, 1024, 3}), 1024, 1024, 3), &enc_out_w_);
        stage_f32(read_f32("encoder.block.6.conv.bias", {1024}), &enc_out_b_);
        for (int u = 0; u < 2; ++u) {
            const std::string p = "quantizer.downsample." + std::to_string(u);
            stage_f32(pack_conv1d(read_f32(p + ".0.conv.weight", {1024, 1024, 2}), 1024, 1024, 2), &dn_w_[u]);
            stage_f32(read_f32(p + ".0.conv.bias", {1024}), &dn_b_[u]);
            load_convnext(p + ".1", false, dn_cn_[u]);
        }
        load_transformer("quantizer.pre_module", CODEC_PRE_LAYERS, false, pre_mod_);

        // 6. Stage 10 RVQ stages (in_proj + pre-L2-normalized codebooks)
        for (int q = 0; q < CODEC_NUM_CODEBOOKS; ++q) {
            const std::string p = (q == 0)
                ? "quantizer.semantic_quantizer.quantizers.0"
                : ("quantizer.quantizer.quantizers." + std::to_string(q - 1));
            const int V = (q == 0) ? CODEC_SEM_SIZE : CODEC_RES_SIZE;
            rvq_[q].vocab_size = V;
            rvq_[q].proj_offset = (q == 0) ? 0 : (CODEC_SEM_SIZE + (q - 1) * CODEC_RES_SIZE);
            stage_f32(read_f32(p + ".in_proj.weight", {CODEC_CB_DIM, CODEC_LATENT_DIM, 1}), &rvq_[q].in_proj_w);
            stage_f32(read_f32(p + ".in_proj.bias", {CODEC_CB_DIM}), &rvq_[q].in_proj_b);
            auto cb = read_f32(p + ".codebook.weight", {V, CODEC_CB_DIM});
            std::vector<float> cbn(cb.size());
            std::vector<float> cbn_sq(static_cast<size_t>(V));
            for (int k = 0; k < V; ++k) {
                double ss = 0.0;
                for (int d = 0; d < CODEC_CB_DIM; ++d) {
                    double v = double(cb[size_t(k) * CODEC_CB_DIM + d]);
                    ss += v * v;
                }
                const double denom = std::max(std::sqrt(ss), 1e-12);
                double nsq = 0.0;
                for (int d = 0; d < CODEC_CB_DIM; ++d) {
                    float nv = static_cast<float>(double(cb[size_t(k) * CODEC_CB_DIM + d]) / denom);
                    cbn[size_t(k) * CODEC_CB_DIM + d] = nv;
                    nsq += double(nv) * double(nv);
                }
                cbn_sq[size_t(k)] = static_cast<float>(nsq);
            }
            stage_f32(cbn, &rvq_[q].cb_norm);
            stage_f32(cbn_sq, &rvq_[q].cb_norm_sq);
        }

        // 7. Upload weights into one contiguous device arena via pinned host staging
        w_arena_.alloc_bytes(total_w_bytes);
        const size_t pin_bytes = 64u << 20;
        void* pin = nullptr;
        G6TTS_CODEC_CK(cudaMallocHost(&pin, pin_bytes));
        struct PinGuard { void* p; ~PinGuard() { if (p) cudaFreeHost(p); } } pg{pin};

        for (auto& u : uploads) {
            *u.dst_ptr = w_arena_.base + u.arena_off;
            size_t done = 0;
            while (done < u.bytes.size()) {
                const size_t chunk = std::min(pin_bytes, u.bytes.size() - done);
                std::memcpy(pin, u.bytes.data() + done, chunk);
                G6TTS_CODEC_CK(cudaMemcpy(w_arena_.base + u.arena_off + done, pin, chunk, cudaMemcpyHostToDevice));
                done += chunk;
            }
        }

        // 8. Allocate persistent per-slot streaming state
        const size_t ctx_bytes = size_t(max_slots_) * dec_ctx_total_elems() * sizeof(float);
        const size_t carry_bytes = size_t(max_slots_) * dec_carry_total_elems() * sizeof(float);
        const size_t ring_bytes = size_t(max_slots_) * CODEC_POST_LAYERS * CODEC_KV_RING * CODEC_LATENT_DIM * sizeof(float);
        const size_t pos_bytes = size_t(max_slots_) * sizeof(int32_t);
        s_arena_.alloc_bytes(DeviceArena::align256(ctx_bytes) + DeviceArena::align256(carry_bytes) +
                             2u * DeviceArena::align256(ring_bytes) + DeviceArena::align256(pos_bytes));
        d_slot_conv_ctx_ = static_cast<float*>(s_arena_.reserve(ctx_bytes));
        d_slot_carry_ = static_cast<float*>(s_arena_.reserve(carry_bytes));
        d_k_ring_ = static_cast<float*>(s_arena_.reserve(ring_bytes));
        d_v_ring_ = static_cast<float*>(s_arena_.reserve(ring_bytes));
        d_slot_pos_ = static_cast<int32_t*>(s_arena_.reserve(pos_bytes));
        G6TTS_CODEC_CK(cudaMemset(s_arena_.base, 0, s_arena_.cap));

        // 9. Pre-allocate streaming decode scratch buffers for (max_slots_, max_decode_stride_)
        const size_t MS = size_t(max_slots_) * size_t(max_decode_stride_);
        const size_t max_act = MS * 196608u;      // 2048 * 96 == 1024 * 192
        const size_t max_gemm = MS * 393216u;     // 2 * 2048 * 96
        const size_t max_pad = size_t(max_slots_) * std::max({
            (4u * size_t(max_decode_stride_) + 6u) * 1024u,
            (32u * size_t(max_decode_stride_) + 54u) * 768u,
            (256u * size_t(max_decode_stride_) + 54u) * 384u,
            (1024u * size_t(max_decode_stride_) + 54u) * 192u,
            (2048u * size_t(max_decode_stride_) + 54u) * 96u,
        });
        const size_t max_tmp = std::max(max_act, MS * 16384u);  // covers ConvNeXt 4*S*4096 and post_module S*6144
        const size_t qrot_elems = MS * CODEC_LATENT_DIM;
        const size_t f16_pad_bytes = fp32_decoder_ ? 0u : DeviceArena::align256(max_pad * sizeof(__half));
        const size_t f16_tmp_bytes = fp32_decoder_ ? 0u : DeviceArena::align256(max_tmp * sizeof(__half));
        const size_t sc_total =
            DeviceArena::align256(max_act * sizeof(float)) * 2u +
            DeviceArena::align256(max_gemm * sizeof(float)) +
            DeviceArena::align256(max_pad * sizeof(float)) +
            f16_pad_bytes +
            f16_tmp_bytes +
            DeviceArena::align256(qrot_elems * sizeof(float));
        scratch_arena_.alloc_bytes(sc_total);
        dec_buf_a_f32_ = static_cast<float*>(scratch_arena_.reserve(max_act * sizeof(float)));
        dec_buf_b_f32_ = static_cast<float*>(scratch_arena_.reserve(max_act * sizeof(float)));
        dec_buf_gemm_f32_ = static_cast<float*>(scratch_arena_.reserve(max_gemm * sizeof(float)));
        dec_buf_pad_f32_ = static_cast<float*>(scratch_arena_.reserve(max_pad * sizeof(float)));
        dec_buf_pad_f16_ = fp32_decoder_ ? nullptr : static_cast<__half*>(scratch_arena_.reserve(max_pad * sizeof(__half)));
        dec_buf_tmp_f16_ = fp32_decoder_ ? nullptr : static_cast<__half*>(scratch_arena_.reserve(max_tmp * sizeof(__half)));
        dec_buf_qrot_f32_ = static_cast<float*>(scratch_arena_.reserve(qrot_elems * sizeof(float)));

        // 10. Create cuBLAS handle with fixed pre-allocated workspace (CUDA-graph-safe!)
        G6TTS_CUBLAS_CK(cublasCreate(&cublas_));
        cublas_ws_bytes_ = 32u << 20;
        G6TTS_CODEC_CK(cudaMalloc(&cublas_ws_, cublas_ws_bytes_));
        G6TTS_CUBLAS_CK(cublasSetWorkspace(cublas_, cublas_ws_, cublas_ws_bytes_));

        loaded_ = true;
        return true;
    }

    void reset_slot(int slot, cudaStream_t s) {
        using namespace codec_detail;
        if (!loaded_ || slot < 0 || slot >= max_slots_) return;
        const size_t ctx_elems = dec_ctx_total_elems();
        const size_t carry_elems = dec_carry_total_elems();
        const size_t ring_elems = size_t(CODEC_POST_LAYERS) * CODEC_KV_RING * CODEC_LATENT_DIM;
        G6TTS_CODEC_CK(cudaMemsetAsync(d_slot_conv_ctx_ + size_t(slot) * ctx_elems, 0, ctx_elems * sizeof(float), s));
        G6TTS_CODEC_CK(cudaMemsetAsync(d_slot_carry_ + size_t(slot) * carry_elems, 0, carry_elems * sizeof(float), s));
        G6TTS_CODEC_CK(cudaMemsetAsync(d_k_ring_ + size_t(slot) * ring_elems, 0, ring_elems * sizeof(float), s));
        G6TTS_CODEC_CK(cudaMemsetAsync(d_v_ring_ + size_t(slot) * ring_elems, 0, ring_elems * sizeof(float), s));
        G6TTS_CODEC_CK(cudaMemsetAsync(d_slot_pos_ + slot, 0, sizeof(int32_t), s));
    }

    // Streaming/batched CUDA-graph-capturable decode:
    //   d_codes_M_stride_10: [M, stride, 10] row-major int32 on device
    //   d_slots:             [M] int32 slot indices on device
    //   d_pcm_M_stride_2048: [M, stride * 2048] row-major int16 on device
    void decode_frames(const int32_t* d_codes_M_stride_10, int M, int stride,
                       const int* d_slots, int16_t* d_pcm_M_stride_2048, cudaStream_t s) {
        using namespace codec_detail;
        if (!loaded_) throw std::runtime_error("tts_codec: decode_frames called before init");
        if (M <= 0 || stride <= 0) return;
        if (M > max_slots_ || stride > max_decode_stride_) {
            throw std::runtime_error("tts_codec: decode_frames exceeds preallocated (max_slots, max_decode_stride)");
        }
        G6TTS_CUBLAS_CK(cublasSetStream(cublas_, s));

        const int MS = M * stride;
        const size_t ctx_stride = dec_ctx_total_elems();
        const size_t carry_stride = dec_carry_total_elems();

        // 1. Table-sum latent -> dec_buf_a_f32_ [M*S, 1024]
        k_table_sum_latent<<<MS, 256, 0, s>>>(d_codes_M_stride_10, projected_codes_, dec_buf_a_f32_, MS);

        // 2. post_module (8-layer causal window-128 transformer, streaming via KV ring)
        for (int l = 0; l < CODEC_POST_LAYERS; ++l) {
            const auto& lw = post_mod_.layers[size_t(l)];
            // Attention norm
            k_rmsnorm<<<MS, 256, 0, s>>>(dec_buf_a_f32_, lw.attn_norm,
                                         fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                                         fp32_decoder_ ? dec_buf_b_f32_ : nullptr,
                                         MS, CODEC_LATENT_DIM, CODEC_RMS_EPS);
            // QKV GEMM -> dec_buf_gemm_f32_ [MS, 3072]
            dec_gemm(dec_buf_tmp_f16_, dec_buf_b_f32_, lw.wqkv, dec_buf_gemm_f32_, MS, 3072, 1024, 0.0f);
            // RoPE + KV ring scatter
            k_post_rope_kv_scatter<<<dim3(MS, CODEC_N_HEAD), 32, 0, s>>>(
                dec_buf_gemm_f32_, rope_tbl_, d_slots, d_slot_pos_,
                dec_buf_qrot_f32_, d_k_ring_, d_v_ring_, M, stride, l);
            // Window-128 causal attention
            k_post_window_attn_ring<<<dim3(MS, CODEC_N_HEAD), 64, 0, s>>>(
                dec_buf_qrot_f32_, d_k_ring_, d_v_ring_, d_slots, d_slot_pos_,
                fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                fp32_decoder_ ? dec_buf_b_f32_ : nullptr,
                M, stride, l);
            // Output projection -> dec_buf_gemm_f32_ [MS, 1024]
            dec_gemm(dec_buf_tmp_f16_, dec_buf_b_f32_, lw.wo, dec_buf_gemm_f32_, MS, 1024, 1024, 0.0f);
            k_layerscale_add<<<(MS * 1024 + 255) / 256, 256, 0, s>>>(
                dec_buf_a_f32_, dec_buf_gemm_f32_, lw.attn_gamma, MS, 1024);

            // FFN norm
            k_rmsnorm<<<MS, 256, 0, s>>>(dec_buf_a_f32_, lw.ffn_norm,
                                         fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                                         fp32_decoder_ ? dec_buf_b_f32_ : nullptr,
                                         MS, CODEC_LATENT_DIM, CODEC_RMS_EPS);
            // w13 GEMM -> dec_buf_gemm_f32_ [MS, 6144]
            dec_gemm(dec_buf_tmp_f16_, dec_buf_b_f32_, lw.w13, dec_buf_gemm_f32_, MS, 6144, 1024, 0.0f);
            k_swiglu<<<(MS * CODEC_FFN_DIM + 255) / 256, 256, 0, s>>>(
                dec_buf_gemm_f32_,
                fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                fp32_decoder_ ? dec_buf_b_f32_ : nullptr,
                MS, CODEC_FFN_DIM);
            // w2 GEMM -> dec_buf_gemm_f32_ [MS, 1024]
            dec_gemm(dec_buf_tmp_f16_, dec_buf_b_f32_, lw.w2, dec_buf_gemm_f32_, MS, 1024, CODEC_FFN_DIM, 0.0f);
            k_layerscale_add<<<(MS * 1024 + 255) / 256, 256, 0, s>>>(
                dec_buf_a_f32_, dec_buf_gemm_f32_, lw.ffn_gamma, MS, 1024);
        }
        // Final post_module RMSNorm (in-place via b -> a)
        k_rmsnorm<<<MS, 256, 0, s>>>(dec_buf_a_f32_, post_mod_.final_norm, nullptr, dec_buf_b_f32_,
                                     MS, CODEC_LATENT_DIM, CODEC_RMS_EPS);
        k_advance_slot_pos<<<(M + 255) / 256, 256, 0, s>>>(d_slots, d_slot_pos_, M, stride);

        // Current activation is in dec_buf_b_f32_ [M, T, 1024], T = stride
        float* cur = dec_buf_b_f32_;
        float* alt = dec_buf_a_f32_;
        int T = stride;

        // 3. quantizer.upsample (2x [CausalTransConv1d(k=2,s=2) + ConvNeXtBlock])
        for (int u = 0; u < 2; ++u) {
            const int MT = M * T;
            if (!fp32_decoder_) k_f32_to_f16<<<(size_t(MT) * 1024 + 255) / 256, 256, 0, s>>>(cur, dec_buf_tmp_f16_, size_t(MT) * 1024);
            // TransConv k=2, s=2 (carry=0): [MT, 1024] @ [2048, 1024]^T -> [MT, 2, 1024] = [M, 2T, 1024]
            dec_gemm(dec_buf_tmp_f16_, cur, up_w_[u], alt, MT, 2048, 1024, 0.0f);
            T *= 2;
            const int MT2 = M * T;
            k_add_bias_res<<<(size_t(MT2) * 1024 + 255) / 256, 256, 0, s>>>(alt, up_b_[u], nullptr, MT2, 1024);
            std::swap(cur, alt);  // cur holds [M, T, 1024] (residual input for ConvNeXt)

            // ConvNeXtBlock: dwconv(k=7, pad=6) -> LN -> pw1 -> GELU -> pw2 -> *gamma + cur
            const int ctx_id = (u == 0) ? CTX_UP0_DW : CTX_UP1_DW;
            const size_t coff = dec_ctx_offset(ctx_id);
            k_conv_prepend_ctx<<<(size_t(M) * size_t(T + 6) * 1024 + 255) / 256, 256, 0, s>>>(
                cur, d_slot_conv_ctx_, d_slots, ctx_stride, coff, dec_buf_pad_f32_, nullptr, M, T, 6, 1024);
            k_conv_save_ctx<<<(size_t(M) * 6u * 1024 + 255) / 256, 256, 0, s>>>(
                dec_buf_pad_f32_, d_slot_conv_ctx_, d_slots, ctx_stride, coff, M, T, 6, 1024);
            k_dwconv_k7<<<(size_t(MT2) * 1024 + 255) / 256, 256, 0, s>>>(
                dec_buf_pad_f32_, up_cn_[u].dw_w, up_cn_[u].dw_b, alt, M, T, 1024);
            k_layernorm<<<MT2, 256, 0, s>>>(alt, up_cn_[u].ln_w, up_cn_[u].ln_b,
                                            fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                                            fp32_decoder_ ? alt : nullptr,
                                            MT2, 1024, CODEC_LN_EPS);
            dec_gemm(dec_buf_tmp_f16_, alt, up_cn_[u].pw1_w, dec_buf_gemm_f32_, MT2, 4096, 1024, 0.0f);
            k_bias_gelu<<<(size_t(MT2) * 4096 + 255) / 256, 256, 0, s>>>(
                dec_buf_gemm_f32_, up_cn_[u].pw1_b,
                fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                fp32_decoder_ ? dec_buf_gemm_f32_ : nullptr,
                MT2, 4096);
            dec_gemm(dec_buf_tmp_f16_, dec_buf_gemm_f32_, up_cn_[u].pw2_w, alt, MT2, 1024, 4096, 0.0f);
            k_convnext_epilogue<<<(size_t(MT2) * 1024 + 255) / 256, 256, 0, s>>>(
                alt, up_cn_[u].pw2_b, up_cn_[u].gamma, cur, cur, MT2, 1024);
        }

        // 4. decoder.model.0: CausalConv1d(1024 -> 1536, k=7, s=1, d=1, pad=6)
        dec_causal_conv_batched(cur, dec_in_w_, dec_in_b_, alt, nullptr, M, T, 1024, 1536, 7, 1,
                                d_slot_conv_ctx_, d_slots, ctx_stride, dec_ctx_offset(CTX_DEC_IN),
                                dec_buf_pad_f32_, dec_buf_pad_f16_, s);
        std::swap(cur, alt);  // cur: [M, 4*stride, 1536]

        // 5. decoder.model.{1..4}
        for (int b = 0; b < 4; ++b) {
            const int ic = DEC_IN_CH[b], oc = DEC_OUT_CH[b], st = DEC_STRIDES[b], k = 2 * st;
            const int MT_in = M * T;
            // Snake0 -> fp16 (or fp32)
            k_snake1d<<<(size_t(MT_in) * size_t(ic) + 255) / 256, 256, 0, s>>>(
                cur, dec_blk_[b].snake0,
                fp32_decoder_ ? alt : nullptr,
                fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                MT_in, ic);
            // TransConv GEMM: [MT_in, ic] @ [k*oc, ic]^T -> dec_buf_gemm_f32_ [M, T, k, oc]
            dec_gemm(dec_buf_tmp_f16_, alt, dec_blk_[b].trans_w, dec_buf_gemm_f32_, MT_in, k * oc, ic, 0.0f);
            const size_t carry_off = dec_carry_offset(b);
            k_transconv_overlap_add<<<(size_t(MT_in) * size_t(st) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                dec_buf_gemm_f32_, dec_blk_[b].trans_b, d_slot_carry_, d_slots, carry_stride, carry_off,
                cur, M, T, st, oc);
            k_transconv_save_carry<<<(size_t(M) * size_t(st) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                dec_buf_gemm_f32_, d_slot_carry_, d_slots, carry_stride, carry_off, M, T, st, oc);
            T *= st;
            const int MT_out = M * T;

            // 3 ResidualUnits (dilations 1, 3, 9)
            for (int r = 0; r < 3; ++r) {
                const int d = RES_DILATIONS[r];
                const int ctx_id = CTX_DEC_B0_R0 + b * 3 + r;
                // Snake0(cur) -> alt
                k_snake1d<<<(size_t(MT_out) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                    cur, dec_blk_[b].res[r].snake0, alt, nullptr, MT_out, oc);
                // CausalConv1d(k=7, d=d) -> dec_buf_gemm_f32_
                dec_causal_conv_batched(alt, dec_blk_[b].res[r].conv7_w, dec_blk_[b].res[r].conv7_b,
                                        dec_buf_gemm_f32_, nullptr, M, T, oc, oc, 7, d,
                                        d_slot_conv_ctx_, d_slots, ctx_stride, dec_ctx_offset(ctx_id),
                                        dec_buf_pad_f32_, dec_buf_pad_f16_, s);
                // Snake1 -> dec_buf_tmp_f16_ (or alt)
                k_snake1d<<<(size_t(MT_out) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                    dec_buf_gemm_f32_, dec_blk_[b].res[r].snake1,
                    fp32_decoder_ ? alt : nullptr,
                    fp32_decoder_ ? nullptr : dec_buf_tmp_f16_,
                    MT_out, oc);
                // Conv1x1 -> dec_buf_gemm_f32_, then + bias + residual(cur) -> cur
                dec_gemm(dec_buf_tmp_f16_, alt, dec_blk_[b].res[r].conv1_w, dec_buf_gemm_f32_, MT_out, oc, oc, 0.0f);
                k_add_bias_res<<<(size_t(MT_out) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                    dec_buf_gemm_f32_, dec_blk_[b].res[r].conv1_b, cur, MT_out, oc);
                G6TTS_CODEC_CK(cudaMemcpyAsync(cur, dec_buf_gemm_f32_, size_t(MT_out) * size_t(oc) * sizeof(float),
                                               cudaMemcpyDeviceToDevice, s));
            }
        }

        // 6. decoder.model.5 (Snake 96) -> model.6 (Conv k=7 96->1 fp32) -> model.7 (Tanh) -> s16
        const int MT_fin = M * T;  // T == stride * 2048
        k_snake1d<<<(size_t(MT_fin) * 96u + 255) / 256, 256, 0, s>>>(cur, dec_out_snake_, alt, nullptr, MT_fin, 96);
        const size_t fin_coff = dec_ctx_offset(CTX_DEC_OUT);
        k_conv_prepend_ctx<<<(size_t(M) * size_t(T + 6) * 96u + 255) / 256, 256, 0, s>>>(
            alt, d_slot_conv_ctx_, d_slots, ctx_stride, fin_coff, dec_buf_pad_f32_, nullptr, M, T, 6, 96);
        k_conv_save_ctx<<<(size_t(M) * 6u * 96u + 255) / 256, 256, 0, s>>>(
            dec_buf_pad_f32_, d_slot_conv_ctx_, d_slots, ctx_stride, fin_coff, M, T, 6, 96);
        k_dec_final_conv_tanh_s16<<<(size_t(MT_fin) + 255) / 256, 256, 0, s>>>(
            dec_buf_pad_f32_, dec_out_w_, dec_out_b_, d_pcm_M_stride_2048, M, T);
    }

    // Whole-sequence single-shot decode (M=1, no streaming chunking, zero left context) for Gate E2 verification.
    // h_codes_10xF: codebook-major [10, F] int32 on host.
    void decode_whole(const int32_t* h_codes_10xF, int F, std::vector<int16_t>& out_pcm, cudaStream_t s) {
        using namespace codec_detail;
        if (!loaded_) throw std::runtime_error("tts_codec: decode_whole called before init");
        out_pcm.clear();
        if (F <= 0) return;
        G6TTS_CUBLAS_CK(cublasSetStream(cublas_, s));

        // Transpose [10, F] -> [F, 10]
        std::vector<int32_t> h_codes_Fx10(size_t(F) * CODEC_NUM_CODEBOOKS);
        for (int f = 0; f < F; ++f) {
            for (int q = 0; q < CODEC_NUM_CODEBOOKS; ++q) {
                h_codes_Fx10[size_t(f) * CODEC_NUM_CODEBOOKS + size_t(q)] = h_codes_10xF[size_t(q) * size_t(F) + size_t(f)];
            }
        }

        // Allocate ephemeral whole-sequence workspace
        const size_t max_act = size_t(F) * 196608u;
        const size_t max_gemm = size_t(F) * 393216u;
        const size_t max_pad = std::max({
            (4u * size_t(F) + 6u) * 1024u,
            (32u * size_t(F) + 54u) * 768u,
            (256u * size_t(F) + 54u) * 384u,
            (1024u * size_t(F) + 54u) * 192u,
            (2048u * size_t(F) + 54u) * 96u,
        });
        const size_t max_tmp = std::max(max_act, size_t(F) * 16384u);
        const size_t qkv_f = size_t(F) * CODEC_LATENT_DIM;
        const size_t codes_bytes = h_codes_Fx10.size() * sizeof(int32_t);
        const size_t pcm_bytes = size_t(F) * CODEC_FRAME_SAMPLES * sizeof(int16_t);

        const size_t f16_pad_bytes = fp32_decoder_ ? 0u : DeviceArena::align256(max_pad * sizeof(__half));
        const size_t f16_tmp_bytes = fp32_decoder_ ? 0u : DeviceArena::align256(max_tmp * sizeof(__half));
        DeviceArena ws;
        ws.alloc_bytes(
            DeviceArena::align256(codes_bytes) +
            DeviceArena::align256(pcm_bytes) +
            DeviceArena::align256(max_act * sizeof(float)) * 2u +
            DeviceArena::align256(max_gemm * sizeof(float)) +
            DeviceArena::align256(max_pad * sizeof(float)) +
            f16_pad_bytes +
            f16_tmp_bytes +
            DeviceArena::align256(qkv_f * sizeof(float)) * 3u);

        auto* d_codes = static_cast<int32_t*>(ws.reserve(codes_bytes));
        auto* d_pcm = static_cast<int16_t*>(ws.reserve(pcm_bytes));
        auto* buf_a = static_cast<float*>(ws.reserve(max_act * sizeof(float)));
        auto* buf_b = static_cast<float*>(ws.reserve(max_act * sizeof(float)));
        auto* buf_gemm = static_cast<float*>(ws.reserve(max_gemm * sizeof(float)));
        auto* buf_pad_f32 = static_cast<float*>(ws.reserve(max_pad * sizeof(float)));
        auto* buf_pad_f16 = fp32_decoder_ ? nullptr : static_cast<__half*>(ws.reserve(max_pad * sizeof(__half)));
        auto* buf_tmp_f16 = fp32_decoder_ ? nullptr : static_cast<__half*>(ws.reserve(max_tmp * sizeof(__half)));
        auto* buf_q = static_cast<float*>(ws.reserve(qkv_f * sizeof(float)));
        auto* buf_k = static_cast<float*>(ws.reserve(qkv_f * sizeof(float)));
        auto* buf_v = static_cast<float*>(ws.reserve(qkv_f * sizeof(float)));

        G6TTS_CODEC_CK(cudaMemcpyAsync(d_codes, h_codes_Fx10.data(), codes_bytes, cudaMemcpyHostToDevice, s));

        // 1. Table-sum latent
        k_table_sum_latent<<<F, 256, 0, s>>>(d_codes, projected_codes_, buf_a, F);

        // 2. post_module whole-sequence (window 128)
        for (int l = 0; l < CODEC_POST_LAYERS; ++l) {
            const auto& lw = post_mod_.layers[size_t(l)];
            k_rmsnorm<<<F, 256, 0, s>>>(buf_a, lw.attn_norm,
                                        fp32_decoder_ ? nullptr : buf_tmp_f16,
                                        fp32_decoder_ ? buf_b : nullptr,
                                        F, CODEC_LATENT_DIM, CODEC_RMS_EPS);
            dec_gemm(buf_tmp_f16, buf_b, lw.wqkv, buf_gemm, F, 3072, 1024, 0.0f);
            k_whole_rope_split<<<dim3(F, CODEC_N_HEAD), 32, 0, s>>>(buf_gemm, rope_tbl_, buf_q, buf_k, buf_v, F);
            k_whole_window_attn<<<dim3(F, CODEC_N_HEAD), 64, 0, s>>>(
                buf_q, buf_k, buf_v,
                fp32_decoder_ ? nullptr : buf_tmp_f16,
                fp32_decoder_ ? buf_b : nullptr,
                F, CODEC_POST_WINDOW);
            dec_gemm(buf_tmp_f16, buf_b, lw.wo, buf_gemm, F, 1024, 1024, 0.0f);
            k_layerscale_add<<<(F * 1024 + 255) / 256, 256, 0, s>>>(buf_a, buf_gemm, lw.attn_gamma, F, 1024);

            k_rmsnorm<<<F, 256, 0, s>>>(buf_a, lw.ffn_norm,
                                        fp32_decoder_ ? nullptr : buf_tmp_f16,
                                        fp32_decoder_ ? buf_b : nullptr,
                                        F, CODEC_LATENT_DIM, CODEC_RMS_EPS);
            dec_gemm(buf_tmp_f16, buf_b, lw.w13, buf_gemm, F, 6144, 1024, 0.0f);
            k_swiglu<<<(F * CODEC_FFN_DIM + 255) / 256, 256, 0, s>>>(
                buf_gemm,
                fp32_decoder_ ? nullptr : buf_tmp_f16,
                fp32_decoder_ ? buf_b : nullptr,
                F, CODEC_FFN_DIM);
            dec_gemm(buf_tmp_f16, buf_b, lw.w2, buf_gemm, F, 1024, CODEC_FFN_DIM, 0.0f);
            k_layerscale_add<<<(F * 1024 + 255) / 256, 256, 0, s>>>(buf_a, buf_gemm, lw.ffn_gamma, F, 1024);
        }
        k_rmsnorm<<<F, 256, 0, s>>>(buf_a, post_mod_.final_norm, nullptr, buf_b, F, CODEC_LATENT_DIM, CODEC_RMS_EPS);

        float* cur = buf_b;
        float* alt = buf_a;
        int T = F;

        // 3. quantizer.upsample
        for (int u = 0; u < 2; ++u) {
            if (!fp32_decoder_) k_f32_to_f16<<<(size_t(T) * 1024 + 255) / 256, 256, 0, s>>>(cur, buf_tmp_f16, size_t(T) * 1024);
            dec_gemm(buf_tmp_f16, cur, up_w_[u], alt, T, 2048, 1024, 0.0f);
            T *= 2;
            k_add_bias_res<<<(size_t(T) * 1024 + 255) / 256, 256, 0, s>>>(alt, up_b_[u], nullptr, T, 1024);
            std::swap(cur, alt);

            k_conv_prepend_ctx<<<(size_t(T + 6) * 1024 + 255) / 256, 256, 0, s>>>(
                cur, nullptr, nullptr, 0, 0, buf_pad_f32, nullptr, 1, T, 6, 1024);
            k_dwconv_k7<<<(size_t(T) * 1024 + 255) / 256, 256, 0, s>>>(
                buf_pad_f32, up_cn_[u].dw_w, up_cn_[u].dw_b, alt, 1, T, 1024);
            k_layernorm<<<T, 256, 0, s>>>(alt, up_cn_[u].ln_w, up_cn_[u].ln_b,
                                          fp32_decoder_ ? nullptr : buf_tmp_f16,
                                          fp32_decoder_ ? alt : nullptr,
                                          T, 1024, CODEC_LN_EPS);
            dec_gemm(buf_tmp_f16, alt, up_cn_[u].pw1_w, buf_gemm, T, 4096, 1024, 0.0f);
            k_bias_gelu<<<(size_t(T) * 4096 + 255) / 256, 256, 0, s>>>(
                buf_gemm, up_cn_[u].pw1_b,
                fp32_decoder_ ? nullptr : buf_tmp_f16,
                fp32_decoder_ ? buf_gemm : nullptr,
                T, 4096);
            dec_gemm(buf_tmp_f16, buf_gemm, up_cn_[u].pw2_w, alt, T, 1024, 4096, 0.0f);
            k_convnext_epilogue<<<(size_t(T) * 1024 + 255) / 256, 256, 0, s>>>(
                alt, up_cn_[u].pw2_b, up_cn_[u].gamma, cur, cur, T, 1024);
        }

        // 4. decoder.model.0
        dec_causal_conv_batched(cur, dec_in_w_, dec_in_b_, alt, nullptr, 1, T, 1024, 1536, 7, 1,
                                nullptr, nullptr, 0, 0, buf_pad_f32, buf_pad_f16, s);
        std::swap(cur, alt);

        // 5. decoder.model.{1..4}
        for (int b = 0; b < 4; ++b) {
            const int ic = DEC_IN_CH[b], oc = DEC_OUT_CH[b], st = DEC_STRIDES[b], k = 2 * st;
            k_snake1d<<<(size_t(T) * size_t(ic) + 255) / 256, 256, 0, s>>>(
                cur, dec_blk_[b].snake0,
                fp32_decoder_ ? alt : nullptr,
                fp32_decoder_ ? nullptr : buf_tmp_f16,
                T, ic);
            dec_gemm(buf_tmp_f16, alt, dec_blk_[b].trans_w, buf_gemm, T, k * oc, ic, 0.0f);
            k_transconv_overlap_add<<<(size_t(T) * size_t(st) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                buf_gemm, dec_blk_[b].trans_b, nullptr, nullptr, 0, 0, cur, 1, T, st, oc);
            T *= st;
            for (int r = 0; r < 3; ++r) {
                const int d = RES_DILATIONS[r];
                k_snake1d<<<(size_t(T) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                    cur, dec_blk_[b].res[r].snake0, alt, nullptr, T, oc);
                dec_causal_conv_batched(alt, dec_blk_[b].res[r].conv7_w, dec_blk_[b].res[r].conv7_b,
                                        buf_gemm, nullptr, 1, T, oc, oc, 7, d,
                                        nullptr, nullptr, 0, 0, buf_pad_f32, buf_pad_f16, s);
                k_snake1d<<<(size_t(T) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                    buf_gemm, dec_blk_[b].res[r].snake1,
                    fp32_decoder_ ? alt : nullptr,
                    fp32_decoder_ ? nullptr : buf_tmp_f16,
                    T, oc);
                dec_gemm(buf_tmp_f16, alt, dec_blk_[b].res[r].conv1_w, buf_gemm, T, oc, oc, 0.0f);
                k_add_bias_res<<<(size_t(T) * size_t(oc) + 255) / 256, 256, 0, s>>>(
                    buf_gemm, dec_blk_[b].res[r].conv1_b, cur, T, oc);
                G6TTS_CODEC_CK(cudaMemcpyAsync(cur, buf_gemm, size_t(T) * size_t(oc) * sizeof(float),
                                               cudaMemcpyDeviceToDevice, s));
            }
        }

        // 6. decoder.model.5..7
        k_snake1d<<<(size_t(T) * 96u + 255) / 256, 256, 0, s>>>(cur, dec_out_snake_, alt, nullptr, T, 96);
        k_conv_prepend_ctx<<<(size_t(T + 6) * 96u + 255) / 256, 256, 0, s>>>(
            alt, nullptr, nullptr, 0, 0, buf_pad_f32, nullptr, 1, T, 6, 96);
        k_dec_final_conv_tanh_s16<<<(size_t(T) + 255) / 256, 256, 0, s>>>(
            buf_pad_f32, dec_out_w_, dec_out_b_, d_pcm, 1, T);

        out_pcm.resize(size_t(F) * CODEC_FRAME_SAMPLES);
        G6TTS_CODEC_CK(cudaMemcpyAsync(out_pcm.data(), d_pcm, pcm_bytes, cudaMemcpyDeviceToHost, s));
        G6TTS_CODEC_CK(cudaStreamSynchronize(s));
    }

    // Streamed decode helper using a stride schedule (e.g. {1,1,2,4}, repeating the last entry) on slot 0.
    void decode_streamed(const int32_t* h_codes_10xF, int F, const std::vector<int>& strides,
                         std::vector<int16_t>& out_pcm, cudaStream_t s) {
        using namespace codec_detail;
        if (!loaded_) throw std::runtime_error("tts_codec: decode_streamed called before init");
        out_pcm.clear();
        if (F <= 0) return;
        std::vector<int> sched = strides.empty() ? std::vector<int>{1, 1, 2, 4} : strides;
        for (int v : sched) {
            if (v <= 0 || v > max_decode_stride_) {
                throw std::runtime_error("tts_codec: stride exceeds max_decode_stride");
            }
        }
        reset_slot(0, s);
        int h_slot = 0;
        int* d_slot = nullptr;
        int32_t* d_chunk_codes = nullptr;
        int16_t* d_chunk_pcm = nullptr;
        G6TTS_CODEC_CK(cudaMalloc(&d_slot, sizeof(int)));
        G6TTS_CODEC_CK(cudaMalloc(&d_chunk_codes, size_t(max_decode_stride_) * CODEC_NUM_CODEBOOKS * sizeof(int32_t)));
        G6TTS_CODEC_CK(cudaMalloc(&d_chunk_pcm, size_t(max_decode_stride_) * CODEC_FRAME_SAMPLES * sizeof(int16_t)));
        struct TmpGuard {
            int* a; int32_t* b; int16_t* c;
            ~TmpGuard() { if (a) cudaFree(a); if (b) cudaFree(b); if (c) cudaFree(c); }
        } tg{d_slot, d_chunk_codes, d_chunk_pcm};

        G6TTS_CODEC_CK(cudaMemcpyAsync(d_slot, &h_slot, sizeof(int), cudaMemcpyHostToDevice, s));
        out_pcm.resize(size_t(F) * CODEC_FRAME_SAMPLES);
        std::vector<int32_t> h_chunk(size_t(max_decode_stride_) * CODEC_NUM_CODEBOOKS);

        int pos = 0;
        size_t si = 0;
        while (pos < F) {
            const int st = std::min(sched[si < sched.size() ? si : (sched.size() - 1)], F - pos);
            ++si;
            for (int t = 0; t < st; ++t) {
                for (int q = 0; q < CODEC_NUM_CODEBOOKS; ++q) {
                    h_chunk[size_t(t) * CODEC_NUM_CODEBOOKS + size_t(q)] =
                        h_codes_10xF[size_t(q) * size_t(F) + size_t(pos + t)];
                }
            }
            G6TTS_CODEC_CK(cudaMemcpyAsync(d_chunk_codes, h_chunk.data(),
                                           size_t(st) * CODEC_NUM_CODEBOOKS * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, s));
            decode_frames(d_chunk_codes, 1, st, d_slot, d_chunk_pcm, s);
            G6TTS_CODEC_CK(cudaMemcpyAsync(out_pcm.data() + size_t(pos) * CODEC_FRAME_SAMPLES,
                                           d_chunk_pcm, size_t(st) * CODEC_FRAME_SAMPLES * sizeof(int16_t),
                                           cudaMemcpyDeviceToHost, s));
            G6TTS_CODEC_CK(cudaStreamSynchronize(s));
            pos += st;
        }
    }

    // Encode mono 44.1 kHz fp32 PCM -> [10, F] codebook-major int32 codes (all fp32).
    // Returns F = ceil(n_samples / 2048).
    int encode_pcm(const float* h_pcm, int n_samples, std::vector<int32_t>& out_codes_10xF, cudaStream_t s) {
        using namespace codec_detail;
        if (!loaded_) throw std::runtime_error("tts_codec: encode_pcm called before init");
        out_codes_10xF.clear();
        if (n_samples <= 0) return 0;
        G6TTS_CUBLAS_CK(cublasSetStream(cublas_, s));

        const int F = (n_samples + CODEC_FRAME_SAMPLES - 1) / CODEC_FRAME_SAMPLES;
        const int T0 = F * CODEC_FRAME_SAMPLES;

        // Peak activation size in encoder: [T0, 64] = F * 131072 == [T0/2, 128] == [T0/8, 256] ...
        const size_t max_act = size_t(T0) * 64u;
        const size_t max_pad = std::max({
            (size_t(T0) + 54u) * 64u,
            (size_t(T0 / 2) + 54u) * 128u,
            (size_t(T0 / 8) + 54u) * 256u,
            (size_t(T0 / 64) + 54u) * 512u,
            (size_t(4 * F) + 6u) * 1024u,
        });
        const size_t max_t_gemm = std::max(size_t(4 * F) * 6144u, size_t(2 * F) * 4096u);
        const size_t max_qkv = size_t(4 * F) * CODEC_LATENT_DIM;
        const size_t pcm_bytes = size_t(T0) * sizeof(float);
        const size_t codes_bytes = size_t(CODEC_NUM_CODEBOOKS) * size_t(F) * sizeof(int32_t);
        const size_t ze_bytes = size_t(F) * CODEC_CB_DIM * sizeof(float);

        DeviceArena ws;
        ws.alloc_bytes(
            DeviceArena::align256(pcm_bytes) +
            DeviceArena::align256(codes_bytes) +
            DeviceArena::align256(ze_bytes) +
            DeviceArena::align256(max_act * sizeof(float)) * 3u +
            DeviceArena::align256(max_pad * sizeof(float)) +
            DeviceArena::align256(max_t_gemm * sizeof(float)) +
            DeviceArena::align256(max_qkv * sizeof(float)) * 3u);

        auto* d_pcm = static_cast<float*>(ws.reserve(pcm_bytes));
        auto* d_codes = static_cast<int32_t*>(ws.reserve(codes_bytes));
        auto* d_ze = static_cast<float*>(ws.reserve(ze_bytes));
        auto* buf_a = static_cast<float*>(ws.reserve(max_act * sizeof(float)));
        auto* buf_b = static_cast<float*>(ws.reserve(max_act * sizeof(float)));
        auto* buf_c = static_cast<float*>(ws.reserve(max_act * sizeof(float)));
        auto* buf_pad = static_cast<float*>(ws.reserve(max_pad * sizeof(float)));
        auto* buf_tgemm = static_cast<float*>(ws.reserve(max_t_gemm * sizeof(float)));
        auto* buf_q = static_cast<float*>(ws.reserve(max_qkv * sizeof(float)));
        auto* buf_k = static_cast<float*>(ws.reserve(max_qkv * sizeof(float)));
        auto* buf_v = static_cast<float*>(ws.reserve(max_qkv * sizeof(float)));

        // Upload PCM + right-pad zeros to T0
        G6TTS_CODEC_CK(cudaMemcpyAsync(d_pcm, h_pcm, size_t(n_samples) * sizeof(float), cudaMemcpyHostToDevice, s));
        if (T0 > n_samples) {
            G6TTS_CODEC_CK(cudaMemsetAsync(d_pcm + n_samples, 0, size_t(T0 - n_samples) * sizeof(float), s));
        }

        // 1. encoder.block.0: CausalConv1d(1 -> 64, k=7, s=1, d=1, pad=6)
        k_enc_first_conv<<<(size_t(T0) * 64u + 255) / 256, 256, 0, s>>>(d_pcm, enc_in_w_, enc_in_b_, buf_a, T0);
        float* cur = buf_a;
        float* alt = buf_b;
        int T = T0;

        // 2. encoder.block.{1..4}
        for (int b = 0; b < 4; ++b) {
            const int ic = ENC_IN_CH[b], oc = ENC_OUT_CH[b], st = ENC_STRIDES[b], k = 2 * st;
            // 3 ResidualUnits (dilations 1, 3, 9)
            for (int r = 0; r < 3; ++r) {
                const int d = RES_DILATIONS[r];
                k_snake1d<<<(size_t(T) * size_t(ic) + 255) / 256, 256, 0, s>>>(
                    cur, enc_blk_[b].res[r].snake0, alt, nullptr, T, ic);
                enc_causal_conv_f32(alt, static_cast<const float*>(enc_blk_[b].res[r].conv7_w),
                                    enc_blk_[b].res[r].conv7_b, buf_c, T, ic, ic, 7, 1, d, buf_pad, s);
                k_snake1d<<<(size_t(T) * size_t(ic) + 255) / 256, 256, 0, s>>>(
                    buf_c, enc_blk_[b].res[r].snake1, alt, nullptr, T, ic);
                f32_gemm(alt, static_cast<const float*>(enc_blk_[b].res[r].conv1_w), buf_c, T, ic, ic, 0.0f);
                k_add_bias_res<<<(size_t(T) * size_t(ic) + 255) / 256, 256, 0, s>>>(
                    buf_c, enc_blk_[b].res[r].conv1_b, cur, T, ic);
                G6TTS_CODEC_CK(cudaMemcpyAsync(cur, buf_c, size_t(T) * size_t(ic) * sizeof(float),
                                               cudaMemcpyDeviceToDevice, s));
            }
            // Snake3 + strided CausalConv1d(k=2*st, s=st, d=1, pad_left=st)
            k_snake1d<<<(size_t(T) * size_t(ic) + 255) / 256, 256, 0, s>>>(cur, enc_blk_[b].snake3, alt, nullptr, T, ic);
            enc_causal_conv_f32(alt, enc_blk_[b].conv_w, enc_blk_[b].conv_b, cur, T, ic, oc, k, st, 1, buf_pad, s);
            T /= st;

            // Block 3 has 4-layer window-512 causal transformer (encoder.block.4.block.5)
            if (b == 3) {
                enc_transformer_f32(cur, alt, buf_tgemm, buf_q, buf_k, buf_v, enc_t_, T, CODEC_ENC_WINDOW, s);
                std::swap(cur, alt);
            }
        }

        // 3. encoder.block.5 (Snake 1024) + encoder.block.6 (CausalConv1d k=3, s=1, d=1, pad=2)
        k_snake1d<<<(size_t(T) * 1024u + 255) / 256, 256, 0, s>>>(cur, enc_out_snake_, alt, nullptr, T, 1024);
        enc_causal_conv_f32(alt, enc_out_w_, enc_out_b_, cur, T, 1024, 1024, 3, 1, 1, buf_pad, s);

        // 4. quantizer.downsample (2x [CausalConv1d(k=2,s=2,pad=0) + ConvNeXtBlock])
        for (int u = 0; u < 2; ++u) {
            enc_causal_conv_f32(cur, dn_w_[u], dn_b_[u], alt, T, 1024, 1024, 2, 2, 1, buf_pad, s);
            T /= 2;
            std::swap(cur, alt);  // cur holds residual input for ConvNeXt

            k_conv_prepend_ctx<<<(size_t(T + 6) * 1024u + 255) / 256, 256, 0, s>>>(
                cur, nullptr, nullptr, 0, 0, buf_pad, nullptr, 1, T, 6, 1024);
            k_dwconv_k7<<<(size_t(T) * 1024u + 255) / 256, 256, 0, s>>>(
                buf_pad, dn_cn_[u].dw_w, dn_cn_[u].dw_b, alt, 1, T, 1024);
            k_layernorm<<<T, 256, 0, s>>>(alt, dn_cn_[u].ln_w, dn_cn_[u].ln_b, nullptr, alt, T, 1024, CODEC_LN_EPS);
            f32_gemm(alt, static_cast<const float*>(dn_cn_[u].pw1_w), buf_tgemm, T, 4096, 1024, 0.0f);
            k_bias_gelu<<<(size_t(T) * 4096u + 255) / 256, 256, 0, s>>>(
                buf_tgemm, dn_cn_[u].pw1_b, nullptr, buf_tgemm, T, 4096);
            f32_gemm(buf_tgemm, static_cast<const float*>(dn_cn_[u].pw2_w), alt, T, 1024, 4096, 0.0f);
            k_convnext_epilogue<<<(size_t(T) * 1024u + 255) / 256, 256, 0, s>>>(
                alt, dn_cn_[u].pw2_b, dn_cn_[u].gamma, cur, cur, T, 1024);
        }

        // 5. quantizer.pre_module (8-layer window-128 causal transformer, T == F)
        enc_transformer_f32(cur, alt, buf_tgemm, buf_q, buf_k, buf_v, pre_mod_, T, CODEC_PRE_WINDOW, s);
        std::swap(cur, alt);  // cur: [F, 1024] residual

        // 6. 10-stage RVQ encode
        for (int q = 0; q < CODEC_NUM_CODEBOOKS; ++q) {
            f32_gemm(cur, rvq_[q].in_proj_w, d_ze, F, CODEC_CB_DIM, CODEC_LATENT_DIM, 0.0f);
            k_add_bias_res<<<(size_t(F) * CODEC_CB_DIM + 255) / 256, 256, 0, s>>>(
                d_ze, rvq_[q].in_proj_b, nullptr, F, CODEC_CB_DIM);
            k_rvq_stage_quantize_sub<<<F, 256, 0, s>>>(
                d_ze, rvq_[q].cb_norm, rvq_[q].cb_norm_sq, projected_codes_,
                cur, d_codes, F, q, rvq_[q].vocab_size, rvq_[q].proj_offset);
        }

        out_codes_10xF.resize(size_t(CODEC_NUM_CODEBOOKS) * size_t(F));
        G6TTS_CODEC_CK(cudaMemcpyAsync(out_codes_10xF.data(), d_codes, codes_bytes, cudaMemcpyDeviceToHost, s));
        G6TTS_CODEC_CK(cudaStreamSynchronize(s));
        return F;
    }

private:
    // Row-major C[R, N] = A[R, K] @ B[N, K]^T + beta * C[R, N] (strict IEEE fp32 by default, or fp16 TensorOp)
    void dec_gemm(const __half* A_f16, const float* A_f32, const void* B_w, float* C_f32,
                  int R, int N, int K, float beta) {
        const float alpha = 1.0f;
        if (fp32_decoder_) {
            G6TTS_CUBLAS_CK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                         N, R, K, &alpha,
                                         B_w, CUDA_R_32F, K,
                                         A_f32, CUDA_R_32F, K,
                                         &beta, C_f32, CUDA_R_32F, N,
                                         CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
        } else {
            G6TTS_CUBLAS_CK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                         N, R, K, &alpha,
                                         B_w, CUDA_R_16F, K,
                                         A_f16, CUDA_R_16F, K,
                                         &beta, C_f32, CUDA_R_32F, N,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

    // Row-major strict IEEE fp32 C[R, N] = A[R, K] @ B[N, K]^T + beta * C[R, N] (TF32 disabled)
    void f32_gemm(const float* A, const float* B, float* C, int R, int N, int K, float beta, int ldb_a = 0) {
        const float alpha = 1.0f;
        const int lda_in = (ldb_a > 0) ? ldb_a : K;
        G6TTS_CUBLAS_CK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, R, K, &alpha,
                                     B, CUDA_R_32F, K,
                                     A, CUDA_R_32F, lda_in,
                                     &beta, C, CUDA_R_32F, N,
                                     CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    }

    // Batched decoder causal conv1d (stride=1, dilation=d, k=7): x[M, T, C_in] -> y[M, T, C_out] (+ optional res)
    void dec_causal_conv_batched(const float* x, const void* w_taps, const float* bias,
                                 float* y, const float* res,
                                 int M, int T, int C_in, int C_out, int K, int dilation,
                                 float* slot_ctx_arena, const int* d_slots, size_t ctx_stride, size_t ctx_off,
                                 float* buf_pad_f32, __half* buf_pad_f16, cudaStream_t s) {
        using namespace codec_detail;
        const int pad = (K - 1) * dilation;
        const size_t pad_elems = size_t(M) * size_t(T + pad) * size_t(C_in);
        k_conv_prepend_ctx<<<(pad_elems + 255) / 256, 256, 0, s>>>(
            x, slot_ctx_arena, d_slots, ctx_stride, ctx_off,
            (slot_ctx_arena || fp32_decoder_) ? buf_pad_f32 : nullptr,
            fp32_decoder_ ? nullptr : buf_pad_f16,
            M, T, pad, C_in);
        if (slot_ctx_arena) {
            k_conv_save_ctx<<<(size_t(M) * size_t(pad) * size_t(C_in) + 255) / 256, 256, 0, s>>>(
                buf_pad_f32, slot_ctx_arena, d_slots, ctx_stride, ctx_off, M, T, pad, C_in);
        }

        const float alpha = 1.0f;
        const long long strideB = static_cast<long long>(T + pad) * static_cast<long long>(C_in);
        const long long strideC = static_cast<long long>(T) * static_cast<long long>(C_out);
        const size_t tap_elems = size_t(C_out) * size_t(C_in);

        for (int k = 0; k < K; ++k) {
            const float beta = (k == 0) ? 0.0f : 1.0f;
            const size_t x_off = size_t(k) * size_t(dilation) * size_t(C_in);
            if (fp32_decoder_) {
                const float* wk = static_cast<const float*>(w_taps) + size_t(k) * tap_elems;
                G6TTS_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    C_out, T, C_in, &alpha,
                    wk, CUDA_R_32F, C_in, 0,
                    buf_pad_f32 + x_off, CUDA_R_32F, C_in, strideB,
                    &beta, y, CUDA_R_32F, C_out, strideC, M,
                    CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
            } else {
                const __half* wk = static_cast<const __half*>(w_taps) + size_t(k) * tap_elems;
                G6TTS_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    C_out, T, C_in, &alpha,
                    wk, CUDA_R_16F, C_in, 0,
                    buf_pad_f16 + x_off, CUDA_R_16F, C_in, strideB,
                    &beta, y, CUDA_R_32F, C_out, strideC, M,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
        k_add_bias_res<<<(size_t(M) * size_t(T) * size_t(C_out) + 255) / 256, 256, 0, s>>>(
            y, bias, res, M * T, C_out);
    }

    // Encoder causal conv1d in fp32 (any K, stride, dilation; M=1, T divisible by stride)
    void enc_causal_conv_f32(const float* x, const float* w_taps, const float* bias, float* y,
                             int T_in, int C_in, int C_out, int K, int stride, int dilation,
                             float* buf_pad, cudaStream_t s) {
        using namespace codec_detail;
        const int pad = (K - 1) * dilation + 1 - stride;
        const int T_out = T_in / stride;
        const float* x_src = x;
        if (pad > 0) {
            k_conv_prepend_ctx<<<(size_t(T_in + pad) * size_t(C_in) + 255) / 256, 256, 0, s>>>(
                x, nullptr, nullptr, 0, 0, buf_pad, nullptr, 1, T_in, pad, C_in);
            x_src = buf_pad;
        }
        const size_t tap_elems = size_t(C_out) * size_t(C_in);
        for (int k = 0; k < K; ++k) {
            const float beta = (k == 0) ? 0.0f : 1.0f;
            const float* wk = w_taps + size_t(k) * tap_elems;
            const float* xk = x_src + size_t(k) * size_t(dilation) * size_t(C_in);
            f32_gemm(xk, wk, y, T_out, C_out, C_in, beta, stride * C_in);
        }
        k_add_bias_res<<<(size_t(T_out) * size_t(C_out) + 255) / 256, 256, 0, s>>>(y, bias, nullptr, T_out, C_out);
    }

    // Encoder WindowLimitedTransformer in fp32 (in: x[T, 1024], out: y[T, 1024], x is modified in-place then normed to y)
    void enc_transformer_f32(float* x, float* y, float* buf_gemm,
                             float* buf_q, float* buf_k, float* buf_v,
                             const codec_detail::TransformerW& tw,
                             int T, int window_size, cudaStream_t s) {
        using namespace codec_detail;
        for (size_t l = 0; l < tw.layers.size(); ++l) {
            const auto& lw = tw.layers[l];
            k_rmsnorm<<<T, 256, 0, s>>>(x, lw.attn_norm, nullptr, y, T, CODEC_LATENT_DIM, CODEC_RMS_EPS);
            f32_gemm(y, static_cast<const float*>(lw.wqkv), buf_gemm, T, 3072, 1024, 0.0f);
            k_whole_rope_split<<<dim3(T, CODEC_N_HEAD), 32, 0, s>>>(buf_gemm, rope_tbl_, buf_q, buf_k, buf_v, T);
            k_whole_window_attn<<<dim3(T, CODEC_N_HEAD), 64, 0, s>>>(buf_q, buf_k, buf_v, nullptr, y, T, window_size);
            f32_gemm(y, static_cast<const float*>(lw.wo), buf_gemm, T, 1024, 1024, 0.0f);
            k_layerscale_add<<<(size_t(T) * 1024u + 255) / 256, 256, 0, s>>>(x, buf_gemm, lw.attn_gamma, T, 1024);

            k_rmsnorm<<<T, 256, 0, s>>>(x, lw.ffn_norm, nullptr, y, T, CODEC_LATENT_DIM, CODEC_RMS_EPS);
            f32_gemm(y, static_cast<const float*>(lw.w13), buf_gemm, T, 6144, 1024, 0.0f);
            k_swiglu<<<(size_t(T) * CODEC_FFN_DIM + 255) / 256, 256, 0, s>>>(buf_gemm, nullptr, y, T, CODEC_FFN_DIM);
            f32_gemm(y, static_cast<const float*>(lw.w2), buf_gemm, T, 1024, CODEC_FFN_DIM, 0.0f);
            k_layerscale_add<<<(size_t(T) * 1024u + 255) / 256, 256, 0, s>>>(x, buf_gemm, lw.ffn_gamma, T, 1024);
        }
        k_rmsnorm<<<T, 256, 0, s>>>(x, tw.final_norm, nullptr, y, T, CODEC_LATENT_DIM, CODEC_RMS_EPS);
    }

    bool loaded_ = false;
    bool fp32_decoder_ = true;
    int max_slots_ = 0;
    int max_decode_stride_ = 0;

    cublasHandle_t cublas_ = nullptr;
    void* cublas_ws_ = nullptr;
    size_t cublas_ws_bytes_ = 0;

    codec_detail::DeviceArena w_arena_;
    codec_detail::DeviceArena s_arena_;
    codec_detail::DeviceArena scratch_arena_;

    // Shared weights
    const float* projected_codes_ = nullptr;
    const float2* rope_tbl_ = nullptr;

    // Decoder weights
    codec_detail::TransformerW post_mod_;
    const void* up_w_[2] = {nullptr, nullptr};
    const float* up_b_[2] = {nullptr, nullptr};
    codec_detail::ConvNeXtW up_cn_[2];
    const void* dec_in_w_ = nullptr;
    const float* dec_in_b_ = nullptr;
    codec_detail::DecoderBlockW dec_blk_[4];
    const float* dec_out_snake_ = nullptr;
    const float* dec_out_w_ = nullptr;
    const float* dec_out_b_ = nullptr;

    // Encoder weights
    const float* enc_in_w_ = nullptr;
    const float* enc_in_b_ = nullptr;
    codec_detail::EncoderBlockW enc_blk_[4];
    codec_detail::TransformerW enc_t_;
    const float* enc_out_snake_ = nullptr;
    const float* enc_out_w_ = nullptr;
    const float* enc_out_b_ = nullptr;
    const float* dn_w_[2] = {nullptr, nullptr};
    const float* dn_b_[2] = {nullptr, nullptr};
    codec_detail::ConvNeXtW dn_cn_[2];
    codec_detail::TransformerW pre_mod_;
    codec_detail::RvqStageW rvq_[codec_detail::CODEC_NUM_CODEBOOKS];

    // Persistent per-slot streaming state
    float* d_slot_conv_ctx_ = nullptr;
    float* d_slot_carry_ = nullptr;
    float* d_k_ring_ = nullptr;
    float* d_v_ring_ = nullptr;
    int32_t* d_slot_pos_ = nullptr;

    // Pre-allocated streaming decode scratch
    float* dec_buf_a_f32_ = nullptr;
    float* dec_buf_b_f32_ = nullptr;
    float* dec_buf_gemm_f32_ = nullptr;
    float* dec_buf_pad_f32_ = nullptr;
    __half* dec_buf_pad_f16_ = nullptr;
    __half* dec_buf_tmp_f16_ = nullptr;
    float* dec_buf_qrot_f32_ = nullptr;
};

// Helper CLI runner for `g6tts_bench --codec-decode` / `--codec-encode` (Wave 1.7 / Gate E2).
//   --codec-decode CODES.bin --out PCM.bin [--codec-strides 1,1,2,4]
//     CODES.bin: raw int32 little-endian [10, F] codebook-major (10*F*4 bytes)
//     PCM.bin:   raw int16 little-endian mono 44.1 kHz (F*2048*2 bytes)
//   --codec-encode PCM.bin --out CODES.bin
//     PCM.bin:   raw float32 little-endian mono 44.1 kHz (N*4 bytes)
//     CODES.bin: raw int32 little-endian [10, F] codebook-major (10*F*4 bytes)
inline int run_codec_bench_cli(const std::string& codec_path,
                               const std::string& decode_in,
                               const std::string& encode_in,
                               const std::string& out_path,
                               const std::string& strides_csv,
                               cudaStream_t s) {
    if (codec_path.empty() || out_path.empty() || (decode_in.empty() == encode_in.empty())) {
        std::fprintf(stderr, "tts_codec: specify exactly one of --codec-decode or --codec-encode plus --out\n");
        return 2;
    }

    std::vector<int> strides;
    int max_stride = 4;
    if (!strides_csv.empty()) {
        std::stringstream ss(strides_csv);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            int v = std::atoi(tok.c_str());
            if (v <= 0) {
                std::fprintf(stderr, "tts_codec: invalid stride '%s'\n", tok.c_str());
                return 2;
            }
            strides.push_back(v);
            if (v > max_stride) max_stride = v;
        }
    }

    TtsCodec codec;
    codec.init(codec_path, 1, max_stride);

    if (!decode_in.empty()) {
        std::ifstream ifs(decode_in, std::ios::binary | std::ios::ate);
        if (!ifs) {
            std::fprintf(stderr, "tts_codec: cannot open %s\n", decode_in.c_str());
            return 1;
        }
        const size_t nbytes = static_cast<size_t>(ifs.tellg());
        if (nbytes == 0 || (nbytes % (codec_detail::CODEC_NUM_CODEBOOKS * sizeof(int32_t))) != 0) {
            std::fprintf(stderr, "tts_codec: %s size (%zu) not a multiple of 40 bytes\n", decode_in.c_str(), nbytes);
            return 1;
        }
        const int F = static_cast<int>(nbytes / (codec_detail::CODEC_NUM_CODEBOOKS * sizeof(int32_t)));
        std::vector<int32_t> codes(nbytes / sizeof(int32_t));
        ifs.seekg(0);
        ifs.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(nbytes));

        std::vector<int16_t> pcm;
        if (!strides.empty()) {
            codec.decode_streamed(codes.data(), F, strides, pcm, s);
        } else {
            codec.decode_whole(codes.data(), F, pcm, s);
        }

        std::ofstream ofs(out_path, std::ios::binary);
        if (!ofs) {
            std::fprintf(stderr, "tts_codec: cannot write %s\n", out_path.c_str());
            return 1;
        }
        ofs.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
        std::fprintf(stderr, "tts_codec: decoded %d frames -> %zu s16 samples (%s) -> %s\n",
                     F, pcm.size(), strides.empty() ? "whole" : ("strides=" + strides_csv).c_str(), out_path.c_str());
        return 0;
    }

    // Encode path
    std::ifstream ifs(encode_in, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::fprintf(stderr, "tts_codec: cannot open %s\n", encode_in.c_str());
        return 1;
    }
    const size_t nbytes = static_cast<size_t>(ifs.tellg());
    if (nbytes == 0 || (nbytes % sizeof(float)) != 0) {
        std::fprintf(stderr, "tts_codec: %s size (%zu) not a multiple of 4 bytes\n", encode_in.c_str(), nbytes);
        return 1;
    }
    const int n_samples = static_cast<int>(nbytes / sizeof(float));
    std::vector<float> pcm(static_cast<size_t>(n_samples));
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(nbytes));

    std::vector<int32_t> codes;
    const int F = codec.encode_pcm(pcm.data(), n_samples, codes, s);

    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        std::fprintf(stderr, "tts_codec: cannot write %s\n", out_path.c_str());
        return 1;
    }
    ofs.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size() * sizeof(int32_t)));
    std::fprintf(stderr, "tts_codec: encoded %d f32 samples -> [10, %d] codes -> %s\n",
                 n_samples, F, out_path.c_str());
    return 0;
}

}  // namespace g6tts
