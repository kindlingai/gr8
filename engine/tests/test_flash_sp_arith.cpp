// CPU arithmetic / consumer gate for the Flash SP prefill (G6_PF_RS=2), TP4-SP-MHC-RS-AG-BUILD.md "Arithmetic/consumer
// gate". A scalar host reference, not the CUDA kernels: it cannot prove GPU byte equality (that needs the GPU gate
// after a lead GO). What it proves, at the real HID = 4096 for small M and HID = 64 for M = 8191 / 8192:
//  1) owner-row values: the SP reader (MhcRsBlock, flash_prefill.cuh: self rows from the device partial at owner-local
//     index (m - first) * HID + h, peers from the k_put_rs slices) gives, for every owner row, the bit-identical FP32
//     sum of the OFF full-broadcast reader (MhcFabricBlock, flash_step.cuh: global index m * HID + h): loads p = 0..3,
//     sum = 0.f; sum += bf16(v[p]) in rank order. BF16 edge values included (+-0, subnormals, max finite, +-inf, NaN,
//     2^24 cancellation). A tree or reversed order DOES differ on some inputs, so the gate is order-sensitive.
//  2) FP32 AG row placement: each owner's FP32 rows land at global rows pf_r0(M, p) on every rank, all M rows equal.
//  3) FP32 -> FP16: the host round-to-nearest-even used for gates matches _Float16 conversion on every FP16 value,
//     every tie and both neighbours of every tie; the reconstructed P.x row (f2h of the gathered FP32 row) is the same
//     bytes as the owner's own `out` (f2h of its FP32 value); an FP16 or BF16 AG would change the FP32 P.xn rows the
//     consumers read (so the AG must stay 4 B / element).
//  4) final FP32 residual gather (c4): the same pf_put_ag_sp / k_ag_copy FP32 row code as 2) (row_bytes HID * 4),
//     so 2) covers its placement; test_flash_sp_sched.cpp checks the c4 values end to end.
// Build: g++ -std=c++17 -O2 -Wall -ffp-contract=off -I engine/runtime engine/tests/test_flash_sp_arith.cpp -o sp_arith_test
#include "pf_layout.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace g6rt;

namespace
{
using u64 = unsigned long long;
constexpr int NR = 4;
int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)

u64 mix64(u64 x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
float bf16f(unsigned short v) { const unsigned u = (unsigned)v << 16; float f; memcpy(&f, &u, 4); return f; }
unsigned fbits(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
float bitsf(unsigned u) { float f; memcpy(&f, &u, 4); return f; }

const unsigned short EDGE[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80,
                               0x4B80, 0xCB80, 0x3380, 0x7F80, 0xFF80, 0x7FC0, 0x4780, 0xC780};
// BF16 partial of rank r: element e takes an edge value 1 time in 4, else a finite pattern across magnitudes
unsigned short pval(int r, u64 tag, size_t e)
{
    const u64 x = mix64(tag ^ ((u64)r << 56) ^ e);
    if ((x & 3) == 0) return EDGE[(x >> 8) % (sizeof EDGE / sizeof EDGE[0])];
    return (unsigned short)(((x >> 16) & 0x7F7F) | ((x >> 40) & 0x8000));
}
// the OFF reader: rank-ordered FP32 sum at the GLOBAL index (MhcFabricBlock::operator())
float sum_off(const std::vector<std::vector<unsigned short>>& part, size_t k)
{
    float s = 0.f;
    for (int p = 0; p < NR; ++p) s += bf16f(part[p][k]);
    return s;
}

// 1) + 2) + 4) at one (M, HID)
void owner_rows(int M, int HID, u64 tag)
{
    std::vector<std::vector<unsigned short>> part(NR, std::vector<unsigned short>((size_t)M * HID));
    for (int r = 0; r < NR; ++r)
        for (size_t e = 0; e < part[r].size(); ++e) part[r][e] = pval(r, tag, e);
    // k_put_rs on every sender r: row -> slice of its owner, owner-local row index; ss from the shared layout math
    const size_t RBh = (size_t)HID * 2, ss = pf_ss_min_rb(M, NR, RBh);
    std::vector<std::vector<unsigned char>> send(NR, std::vector<unsigned char>(NR * ss, 0xA5));
    for (int r = 0; r < NR; ++r)
        for (int row = 0; row < M; ++row)
        {
            const int p = pf_owner_of(M, row, NR);
            memcpy(&send[r][(size_t)p * ss + (size_t)(row - pf_r0(M, p, NR)) * RBh], &part[r][(size_t)row * HID], RBh);
        }
    std::vector<std::vector<float>> xn(NR, std::vector<float>((size_t)M * HID, bitsf(0x7FBADBADu)));   // P.xn2 / P.xn per rank
    std::vector<float> off((size_t)M * HID);
    for (size_t k = 0; k < off.size(); ++k) off[k] = sum_off(part, k);
    size_t owner_elems = 0;
    for (int me = 0; me < NR; ++me)
    {
        const int first = pf_r0(M, me, NR), rows = pf_rows(M, me, NR);
        const unsigned short* base[NR];   // MhcRsBlock::init: self = device partial at the owner's first row
        for (int p = 0; p < NR; ++p)
            base[p] = p == me ? &part[me][(size_t)first * HID]
                              : reinterpret_cast<const unsigned short*>(&send[p][(size_t)me * ss]);   // recv[c][p] offset 0
        for (int m = first; m < first + rows; ++m)
            for (int h = 0; h < HID; ++h)
            {
                const size_t k = (size_t)(m - first) * HID + h;   // owner-local index
                unsigned short v[NR];
                for (int p = 0; p < NR; ++p) v[p] = base[p][k];
                float s = 0.f;
                for (int p = 0; p < NR; ++p) s += bf16f(v[p]);
                if (fbits(s) != fbits(off[(size_t)m * HID + h]))
                {
                    if (fails < 5) fprintf(stderr, "FAIL owner sum M %d HID %d rank %d row %d h %d: %08x vs OFF %08x\n", M, HID, me, m, h,
                                           fbits(s), fbits(off[(size_t)m * HID + h]));
                    ++fails;
                }
                xn[me][(size_t)m * HID + h] = s;   // the owner keeps its FP32 rows in place
                ++owner_elems;
            }
    }
    CHECK(owner_elems == (size_t)M * HID);   // owners cover every element exactly once
    // AG: owner p's rows from its send offset 0 -> rows pf_r0(M, p) of every other rank (k_ag_copy)
    std::vector<std::vector<float>> gathered = xn;
    for (int me = 0; me < NR; ++me)
        for (int p = 0; p < NR; ++p)
        {
            if (p == me) continue;
            const int f = pf_r0(M, p, NR), n = pf_rows(M, p, NR);
            std::vector<float> slot((size_t)n * HID);
            if (n) memcpy(slot.data(), &xn[p][(size_t)f * HID], (size_t)n * HID * 4);   // pf_put_ag_sp
            if (n) memcpy(&gathered[me][(size_t)f * HID], slot.data(), (size_t)n * HID * 4);
        }
    for (int me = 0; me < NR; ++me) CHECK(!memcmp(gathered[me].data(), off.data(), off.size() * 4));   // all rows, bit-exact
}

// 1b) the gate is order-sensitive: sequential p0..p3 vs a tree / reversed order differ on the 2^24 case
void order_sensitive()
{
    const unsigned short a = 0x4B80, one = 0x3F80, na = 0xCB80;   // 2^24, 1, -2^24
    const unsigned short v[NR] = {a, one, one, na};
    float seq = 0.f;
    for (int p = 0; p < NR; ++p) seq += bf16f(v[p]);
    const float tree = (bf16f(v[0]) + bf16f(v[1])) + (bf16f(v[2]) + bf16f(v[3]));
    float rev = 0.f;
    for (int p = NR - 1; p >= 0; --p) rev += bf16f(v[p]);
    CHECK(seq == 0.f);
    CHECK(tree == 1.f);
    CHECK(fbits(seq) != fbits(tree) || fbits(seq) != fbits(rev));
}

// 3) host RNE FP32 -> FP16 (the __float2half_rn contract) vs _Float16
unsigned short f2h_rne(unsigned x)
{
    const unsigned sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return (unsigned short)(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u | ((ax >> 13) & 0x3FFu) : 0u));
    if (ax >= 0x477FF000u) return (unsigned short)(sign | 0x7C00u);                        // rounds to >= 65520 -> inf
    if (ax < 0x38800000u)                                                                // FP16 subnormal / zero range
    {
        if (ax < 0x33000000u) return (unsigned short)sign;                                // < 2^-25 -> +-0
        const unsigned e = ax >> 23, mant = (ax & 0x7FFFFFu) | 0x800000u;
        const unsigned shift = 126 - e;                                                   // 14..24 -> mant >> (shift)
        unsigned q = mant >> shift;
        const unsigned rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
        return (unsigned short)(sign | q);
    }
    unsigned q = ((ax >> 23) - 112) << 10 | ((ax >> 13) & 0x3FFu);
    const unsigned rem = ax & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1))) ++q;
    return (unsigned short)(sign | q);
}
unsigned short f2h_ref(float f) { const _Float16 h = (_Float16)f; unsigned short u; memcpy(&u, &h, 2); return u; }
void f2h_exhaustive_ties()
{
    size_t n = 0, bad = 0;
    auto one = [&](unsigned x) {
        const float f = bitsf(x);
        const unsigned short a = f2h_rne(x), b = f2h_ref(f);
        const bool nan = std::isnan(f);
        if (nan ? ((a & 0x7C00u) != 0x7C00u || !(a & 0x3FFu)) : a != b)
        {
            if (bad < 5) fprintf(stderr, "FAIL f2h %08x: host %04x vs _Float16 %04x\n", x, a, b);
            ++bad;
        }
        ++n;
    };
    for (unsigned h = 0; h < 0x10000u; ++h)   // every FP16 value, its exact FP32, the tie above it and both neighbours
    {
        const _Float16 hv = [&] { _Float16 t; unsigned short s = (unsigned short)h; memcpy(&t, &s, 2); return t; }();
        const float f = (float)hv;
        if (std::isnan(f) || std::isinf(f)) { one(fbits(f)); continue; }
        const unsigned x = fbits(f);
        one(x);
        const unsigned short hn = (unsigned short)(h + 1);
        _Float16 nv; memcpy(&nv, &hn, 2);
        const float fn = (float)nv;
        if ((h & 0x7FFFu) == 0x7BFFu || std::isnan(fn)) continue;
        const double mid = ((double)f + (double)fn) / 2;     // exact tie between two neighbours (fits FP32)
        const float fm = (float)mid;
        if ((double)fm != mid) continue;
        one(fbits(fm)); one(fbits(fm) - 1); one(fbits(fm) + 1);
    }
    for (u64 i = 0; i < 4000000; ++i) one((unsigned)mix64(i));                                 // random FP32
    for (unsigned x : {0x477FEFFFu, 0x477FF000u, 0xC77FF000u, 0x33000000u, 0x33000001u, 0x7F800000u, 0xFF800000u, 0x7FC00001u})
        one(x);
    printf("f2h: %zu conversions, %zu mismatches vs _Float16\n", n, bad);
    if (bad) ++fails;
}
// the reconstructed P.x row is the owner's own `out` bytes; a narrower AG would change the FP32 rows consumers read
void reconstruct_and_width()
{
    size_t fp16_diff = 0, bf16_diff = 0, n = 0;
    for (u64 i = 0; i < 1000000; ++i)
    {
        const float v = bitsf((unsigned)(mix64(i ^ 0xF00Dull) & 0xBFFFFFFFu));   // finite, sign and magnitude mixed
        if (std::isnan(v) || std::isinf(v)) continue;
        const unsigned short owner_out = f2h_rne(fbits(v));          // the owner's kernel: out = __float2half_rn(v)
        const float gathered = bitsf(fbits(v));                      // FP32 AG: the same 4 bytes
        CHECK(f2h_rne(fbits(gathered)) == owner_out);                // k_f2h_rows on any rank
        _Float16 h16;
        memcpy(&h16, &owner_out, 2);
        const float via_fp16 = (float)h16;
        const float via_bf16 = bitsf(fbits(v) & 0xFFFF0000u);
        fp16_diff += fbits(via_fp16) != fbits(v);
        bf16_diff += fbits(via_bf16) != fbits(v);
        ++n;
    }
    printf("width: %zu values; an FP16 AG would change %zu FP32 P.xn values, a BF16 AG %zu\n", n, fp16_diff, bf16_diff);
    CHECK(fp16_diff > n / 2 && bf16_diff > n / 2);
}
}   // namespace

int main()
{
    order_sensitive();
    for (int M : {1, 2, 3, 4, 5, 63, 64}) owner_rows(M, 4096, 0xA11CE + M);
    for (int M : {8191, 8192}) owner_rows(M, 64, 0xB0B + M);
    f2h_exhaustive_ties();
    reconstruct_and_width();
    if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
    printf("sp_arith_test: OK\n");
    return 0;
}
