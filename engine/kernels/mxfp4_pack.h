// Host packer for the MXFP4 record "mx4g" of mxfp4.cuh. Our own code; no CUDA needed.
//
// Input is the checkpoint's own layout (DeepSeek-V4.1-Flash-1017 routed experts): per matrix
//   packed [R][K / 2] bytes, byte j of a row = k 2j in the low nibble, k 2j + 1 in the high nibble (E2M1: bit 3 sign,
//                      bits 2..0 -> 0, .5, 1, 1.5, 2, 3, 4, 6)       [safetensors I8: w1/w3 [2304, 2560], w2 [5120, 1152]]
//   e8     [R][K / 32] E8M0 bytes, value 2^(b - 127)                  [w1/w3 scale [2304, 160], w2 scale [5120, 72]]
// Because the checkpoint byte order equals the tile byte order, the code bytes are only permuted, never re-nibbled.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace mx4pack
{
constexpr int KG = 64, GR = 544, SOFF = 512;

// One record: rows r0 .. r0 + 15 of the matrix, k k0 .. k0 + 63 (k0 % 64 == 0). K = row length in k.
inline void pack_group(uint8_t* dst, const uint8_t* packed, const uint8_t* e8, int K, int r0, int k0)
{
    const size_t pb = static_cast<size_t>(K) / 2, eb = static_cast<size_t>(K) / 32;
    for (int lane = 0; lane < 32; ++lane)
    {
        const int r = lane >> 2, c = lane & 3;
        for (int t = 0; t < 4; ++t)
        {
            const size_t j = static_cast<size_t>(k0 / 2 + t * 8 + c);       // checkpoint byte of k 2c, 2c + 1 of tile t
            uint8_t* o = dst + lane * 16 + t * 4;
            o[0] = packed[static_cast<size_t>(r0 + r) * pb + j];
            o[1] = packed[static_cast<size_t>(r0 + r + 8) * pb + j];
            o[2] = packed[static_cast<size_t>(r0 + r) * pb + j + 4];
            o[3] = packed[static_cast<size_t>(r0 + r + 8) * pb + j + 4];
        }
    }
    for (int r = 0; r < 8; ++r)
    {
        uint8_t* s = dst + SOFF + r * 4;
        s[0] = e8[static_cast<size_t>(r0 + r) * eb + k0 / 32];
        s[1] = e8[static_cast<size_t>(r0 + r + 8) * eb + k0 / 32];
        s[2] = e8[static_cast<size_t>(r0 + r) * eb + k0 / 32 + 1];
        s[3] = e8[static_cast<size_t>(r0 + r + 8) * eb + k0 / 32 + 1];
    }
}

// fc2 (down) matrix [R][K] -> [R / 16][K / 64][GR]. Bytes = R / 16 * K / 64 * GR.
inline std::vector<uint8_t> pack_matrix(const uint8_t* packed, const uint8_t* e8, int R, int K)
{
    if (R % 16 || K % KG) throw std::runtime_error("mx4pack: R % 16 or K % 64");
    std::vector<uint8_t> out(static_cast<size_t>(R / 16) * (K / KG) * GR);
    for (int ot = 0; ot < R / 16; ++ot)
        for (int g = 0; g < K / KG; ++g) pack_group(out.data() + (static_cast<size_t>(ot) * (K / KG) + g) * GR, packed, e8, K, ot * 16, g * KG);
    return out;
}

// fc1: gate and up [R][K] -> [R / 16][K / 64][gate GR | up GR]
inline std::vector<uint8_t> pack_fc1(const uint8_t* gp, const uint8_t* ge, const uint8_t* up, const uint8_t* ue, int R, int K)
{
    if (R % 16 || K % KG) throw std::runtime_error("mx4pack: R % 16 or K % 64");
    std::vector<uint8_t> out(static_cast<size_t>(R / 16) * (K / KG) * 2 * GR);
    for (int ch = 0; ch < R / 16; ++ch)
        for (int g = 0; g < K / KG; ++g)
        {
            uint8_t* d = out.data() + (static_cast<size_t>(ch) * (K / KG) + g) * 2 * GR;
            pack_group(d, gp, ge, K, ch * 16, g * KG);
            pack_group(d + GR, up, ue, K, ch * 16, g * KG);
        }
    return out;
}

// TP slice of the intermediate dimension: rows [i0, i0 + I) of a [2304][K] gate/up matrix are a plain row range (pass
// packed + i0 * K / 2, e8 + i0 * K / 32); columns [i0, i0 + I) of the [5120][2304] down matrix need the column slice below.
inline void slice_cols(std::vector<uint8_t>& p, std::vector<uint8_t>& e, const uint8_t* packed, const uint8_t* e8, int R, int K, int k0, int Kn)
{
    p.assign(static_cast<size_t>(R) * Kn / 2, 0);
    e.assign(static_cast<size_t>(R) * Kn / 32, 0);
    for (int r = 0; r < R; ++r)
    {
        memcpy(p.data() + static_cast<size_t>(r) * Kn / 2, packed + static_cast<size_t>(r) * K / 2 + k0 / 2, Kn / 2);
        memcpy(e.data() + static_cast<size_t>(r) * Kn / 32, e8 + static_cast<size_t>(r) * K / 32 + k0 / 32, Kn / 32);
    }
}

// ---- host decode, the layout oracle: value of weight (row, k) read BACK from a packed record stream the way the device
// reads it (lane / tile / byte positions), used by tests only.
inline const float* e2m1_table() { static const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; return t; }
inline float e2m1(unsigned code) { return (code & 8 ? -1.f : 1.f) * e2m1_table()[code & 7]; }
inline double e8m0(unsigned b) { return std::ldexp(1.0, static_cast<int>(b) - 127); }

// Emulates what the device computes for one record: out[16][64] = dequantized weights (scale applied), from the 544 B at g.
inline void decode_group(const uint8_t* g, double (&out)[16][64])
{
    for (int lane = 0; lane < 32; ++lane)
    {
        const int r = lane >> 2, c = lane & 3;
        for (int t = 0; t < 4; ++t)
        {
            const uint8_t* w = g + lane * 16 + t * 4;
            const int h = t >> 1;                                       // k32 scale group
            const double sl = e8m0(g[SOFF + r * 4 + 2 * h]), sh = e8m0(g[SOFF + r * 4 + 2 * h + 1]);
            // a0 = byte 0 (row r, k 2c, 2c + 1), a1 = byte 1 (row r + 8), a2 = byte 2 (row r, k + 8), a3 = byte 3 (row r + 8, k + 8)
            const int kb = t * 16 + 2 * c;
            out[r][kb] = e2m1(w[0] & 15) * sl;          out[r][kb + 1] = e2m1(w[0] >> 4) * sl;
            out[r + 8][kb] = e2m1(w[1] & 15) * sh;      out[r + 8][kb + 1] = e2m1(w[1] >> 4) * sh;
            out[r][kb + 8] = e2m1(w[2] & 15) * sl;      out[r][kb + 9] = e2m1(w[2] >> 4) * sl;
            out[r + 8][kb + 8] = e2m1(w[3] & 15) * sh;  out[r + 8][kb + 9] = e2m1(w[3] >> 4) * sh;
        }
    }
}
}   // namespace mx4pack
