// DSpark drafter: the small pure rules the device kernels (ds41_dspark.cuh, attn/ds41_attn_dsp.cuh) share with the host test.
// Plain C++ with DSP_HD (= __host__ __device__ under nvcc, empty under g++), so engine/tests/test_ds41_dspark.cpp runs the SAME code
// the kernels run and checks it against the host model in ds41_dspark.h. No includes beyond <cstdint>; no CUDA types.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define DSP_HD __host__ __device__
#else
#define DSP_HD
#endif

namespace ds41
{
namespace dspd
{
constexpr int HID = 5120;
constexpr int BLOCK = 5;                  // rows per draft block (dspark_block_size)
constexpr int WINDOW = 128;               // sliding window keys
constexpr int NOISE_TOKEN = 128799;       // dspark_noise_token_id
constexpr int MARKOV_RANK = 256;          // dspark_markov_rank
constexpr int CONF_IN = HID + MARKOV_RANK;   // 5376, confidence_head.proj width
constexpr int TAPS = 3;                   // main layers 37, 38, 39

// tap value of one element before the BF16 rounding: 0.25 * (((s0 + s1) + s2) + s3), stream order, FP32 (docs/DEEPSEEK-DSPARK.md section 3)
DSP_HD inline float tap_mean_f32(float s0, float s1, float s2, float s3) { return (((s0 + s1) + s2) + s3) * 0.25f; }

// Greedy accept of one session (device twin of ds41::dsp::accept_greedy). The session's verify rows are n = k + 1: vtok[0] = golden,
// vtok[1 .. k] = the drafts; target[i] = argmax of verify row i; P = position of row 0. Draft i sits at row i + 1 and is accepted iff
// it equals target[i]. a = accepted drafts, q = P + a (last accepted position), golden = target[a] (the token at q + 1).
DSP_HD inline void accept_session(const int* vtok, const int* target, int n, int P, int& a, int& q, int& golden)
{
    const int k = n - 1;
    int x = 0;
    while (x < k && vtok[1 + x] == target[x]) ++x;
    a = x;
    q = P + x;
    golden = target[x];
}

// Draft-block row r of a session whose last accepted position is q and whose next token is golden.
DSP_HD inline int block_token(int golden, int r) { return r == 0 ? golden : NOISE_TOKEN; }
DSP_HD inline int block_pos(int q, int r) { return q + 1 + r; }

// Block attention key list of one session (every draft row uses it): WLs window keys, then BLOCK block keys.
DSP_HD inline int window_keys(int q) { return q + 1 < WINDOW ? q + 1 : WINDOW; }
DSP_HD inline int key_count(int q, int blk) { return window_keys(q) + blk; }
// position held by list slot i (0 <= i < key_count)
DSP_HD inline int key_pos(int q, int blk, int i)
{
    (void)blk;
    const int w = window_keys(q);
    return i < w ? q - w + 1 + i : q + 1 + (i - w);
}

// Confidence: official = the raw FP32 dot (model.py:1098-1103). DeepInfer's probability = sigmoid(clamp(dot / T, -60, 60)), T = GR8_DS_CONF_TEMP.
DSP_HD inline float conf_clamped(float dot, float temperature)
{
    const float z = dot / temperature;
    return z < -60.0f ? -60.0f : (z > 60.0f ? 60.0f : z);
}

// Markov step i of session s: the token whose markov embedding biases row i's logits (prev_0 = golden, prev_i = draft_{i-1}).
// dtok is the i-major merge output [BLOCK][nsess] (step i's merge writes dtok[i * nsess + s]).
DSP_HD inline int markov_prev(const int* golden, const int* dtok, int nsess, int s, int i) { return i == 0 ? golden[s] : dtok[(i - 1) * nsess + s]; }
}   // namespace dspd
}   // namespace ds41
