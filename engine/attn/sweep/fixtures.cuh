// Fixture helpers reused from our ds41_comp_fp4_test.cu at 1ed4501c.
#pragma once
#include "../ds41_attn.cuh"
#include "memory_guard.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
using ds41a::bf16;
using ds41a::HD;
using ds41a::CQ_ROW;
using ds41a::CQ_CODES;

#define CK(x)                                                                                                       \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess)                                                                                      \
        {                                                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                \
            exit(e_ == cudaErrorMemoryAllocation ? 77 : 2);                                                     \
        }                                                                                                           \
    } while (0)

static int g_fail = 0;
static std::mt19937_64 rng(12345);

template <class T> static T* dalloc(size_t n)
{
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
        throw std::overflow_error("fixture element count overflow");
    return static_cast<T*>(sweep_memory::allocate(n * sizeof(T)));
}
template <class T> static T* dup(const std::vector<T>& h)
{
    T* p = dalloc<T>(h.size());
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> static std::vector<T> back(const T* d, size_t n)
{
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
static uint16_t f2bf(float f)   // round to nearest even
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
// host decoders, written from the formats (not from the device helpers)
static float h_e4m3(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
static float h_e2m1(uint32_t c)
{
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
// packed row -> 512 bf16 bit patterns (the product is exact in bf16: checked)
static void h_decode(const uint8_t* row, uint16_t* out)
{
    for (int c = 0; c < HD; c++)
    {
        const uint32_t code = (row[c >> 1] >> ((c & 1) * 4)) & 15u;
        const float v = h_e2m1(code) * h_e4m3(row[CQ_CODES + (c >> 4)]);
        const uint16_t b = f2bf(v);
        if (bf2f(b) != v) { fprintf(stderr, "h_decode: product not exact in bf16\n"); exit(2); }
        out[c] = b;
    }
}
static std::vector<float> randf(size_t n, float sd)
{
    std::normal_distribution<float> N(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = N(rng);
    return v;
}
static std::vector<uint16_t> randbf(size_t n, float sd)
{
    std::normal_distribution<float> N(0.0f, sd);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f2bf(N(rng));
    return v;
}

struct Keys
{
    size_t rows = 0, slots = 1;
    bf16* bf = nullptr;      // [slots][rows][512]
    uint8_t* pk = nullptr;   // [slots][rows][288]
};
// random packed rows (every code, scale bytes 0x01 .. 0x47 incl. subnormals, some all-zero groups) and their bf16 values
static Keys make_keys(size_t rows, size_t slots)
{
    Keys k;
    k.rows = rows; k.slots = slots;
    const size_t n = rows * slots;
    std::vector<uint8_t> pk(n * CQ_ROW);
    std::uniform_int_distribution<int> byte(0, 255), scl(1, 0x47), z(0, 99);
    for (size_t r = 0; r < n; r++)
    {
        uint8_t* row = &pk[r * CQ_ROW];
        for (int i = 0; i < CQ_CODES; i++) row[i] = (uint8_t)byte(rng);
        for (int g = 0; g < 32; g++)
        {
            row[CQ_CODES + g] = (uint8_t)scl(rng);
            if (z(rng) == 0) for (int i = 0; i < 8; i++) row[g * 8 + i] = (i & 1) ? 0x88 : 0x00;   // +0 / -0 group
        }
    }
    std::vector<uint16_t> bf(n * HD);
    for (size_t r = 0; r < n; r++) h_decode(&pk[r * CQ_ROW], &bf[r * HD]);
    k.pk = dup(pk);
    k.bf = (bf16*)dup(bf);
    return k;
}
static void free_keys(Keys& k)
{
    sweep_memory::deallocate(k.bf);
    k.bf = nullptr;
    sweep_memory::deallocate(k.pk);
    k.pk = nullptr;
}

struct Case
{
    const char* name;
    int ratio;      // 2: layers 2..19, 1: layers 20..39
    int n;          // rows
    int ns;         // sessions (decode form with dsess when > 1)
    int pos;        // first position (decode form); -1 = prefill from 0
    int floor;      // window floor (bounded replay), 0 = none
    int wl = 0;     // window list length override (decode form only; 0 = the engine's: WIN, or min(n, WIN) for prefill)
};

// Unique synthetic top-k; source-derived shapes, NOT captured model lists.
#include "topk_fixture.h"
static void fill_topk(int32_t* out, int p, int ratio, size_t rows)
{
    if (p < 0 || p > 1048576 || ratio < 1 || ratio > 2)
        throw std::runtime_error("invalid synthetic top-k position/ratio");
    const int vis = (int)std::min<size_t>((p + 1) / ratio, rows);
    sweep_topk::uniform(out, vis, rng);
}

struct Built
{
    ds41a::AttnP p{};
    int *d_pos = nullptr, *d_sess = nullptr, *d_topk = nullptr;
    bf16 *q = nullptr, *kvw = nullptr, *out_a = nullptr, *out_b = nullptr;
    float *sink = nullptr, *split = nullptr;
    size_t out_n = 0;
};
static Built build(const Case& c, int heads, const Keys& K, bool split = true)
{
    Built b;
    ds41a::AttnP& p = b.p;
    const int n = c.n;
    const bool decode = c.pos >= 0;
    b.q = (bf16*)dup(randbf((size_t)n * heads * HD, 0.5f));
    b.sink = dup(randf(heads, 1.0f));
    b.out_n = (size_t)n * heads * HD;
    b.out_a = dalloc<bf16>(b.out_n);
    b.out_b = dalloc<bf16>(b.out_n);
    if (split) b.split = dalloc<float>((size_t)n * heads * 2 * (HD + 2));
    std::vector<int32_t> tk((size_t)n * 512);
    std::vector<int> pos(n), sess(n);
    const int rps = n / c.ns;
    for (int t = 0; t < n; t++)
    {
        const int j = t / rps;
        pos[t] = decode ? c.pos + j * 37 + t % rps : t;   // sessions at nearby positions
        sess[t] = c.ns - 1 - j;                          // slot order reversed: slab strides matter
        fill_topk(&tk[(size_t)t * 512], pos[t], c.ratio, K.rows);
    }
    b.d_topk = dup(tk);
    const int ring_rows = decode ? ds41a::WIN + rps - 1 : 0;
    if (decode)
    {
        b.kvw = (bf16*)dup(randbf((size_t)c.ns * ring_rows * HD, 0.3f));
        b.d_pos = dup(pos);
        if (c.ns > 1) b.d_sess = dup(sess);
    }
    else b.kvw = (bf16*)dup(randbf((size_t)n * HD, 0.3f));
    p.q = b.q; p.kvw = b.kvw; p.topk = b.d_topk; p.sink = b.sink;
    p.ldtk = 512; p.n = n; p.heads = heads; p.start = 0;
    p.dstart = decode ? b.d_pos : nullptr;
    p.decode = decode;
    p.WL = decode ? ds41a::WIN : std::min(n, ds41a::WIN);
    if (c.wl) p.WL = c.wl;
    p.ring_rows = ring_rows;
    p.TK = 512;
    p.scale = 0.04419417382415922f;   // 512^-0.5
    p.floor = c.floor;
    p.dsess = c.ns > 1 ? b.d_sess : nullptr;
    p.split = b.split;
    return b;
}
static void set_rows(ds41a::AttnP& p, const Keys& K, bool packed, bf16* out)
{
    p.ckq = packed;
    p.ck = packed ? (const bf16*)K.pk : K.bf;
    p.ck_ss = packed ? K.rows * CQ_ROW : K.rows * HD;
    p.out = out;
}
static void free_built(Built& b)
{
    for (void* x : {(void*)b.d_pos, (void*)b.d_sess, (void*)b.d_topk, (void*)b.q, (void*)b.kvw, (void*)b.out_a, (void*)b.out_b, (void*)b.sink,
                    (void*)b.split})
        sweep_memory::deallocate(x);
    b = Built{};
}
