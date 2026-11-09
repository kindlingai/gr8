// The rank count, the one compile-time definition the engine shares. Default 6 (TP6); -DG6_NR=4 builds TP4.
// Widths split n units as evenly as possible, the larger chunks first: n / NR + (j < n % NR) (tools/convert/plan.py).
// The NR=6 values equal the TP6 literals they replace, so the default build is SASS-identical to the TP6 tree.
#pragma once
#include "model_geom.h"
#ifndef G6_NR
#define G6_NR 6
#endif
static_assert(G6_NR == 6 || G6_NR == 5 || G6_NR == 3 || (G6_NR > 0 && 64 % G6_NR == 0), "G6_NR: 3, 5, 6 or a divisor of 64");

namespace g6nr
{
constexpr int NR = G6_NR;
constexpr int width(int n, int j) { return n / NR + (j < n % NR ? 1 : 0); }
constexpr int start(int n, int j) { return j * (n / NR) + (j < n % NR ? j : n % NR); }
constexpr int maxw(int n) { return (n + NR - 1) / NR; }
constexpr int HEADS = geom::HEADS, BLOCKS = geom::EXP_BLOCKS, VTILES = geom::VTILES;
constexpr int HMAX = maxw(HEADS);          // attention heads (and shared-expert g32 groups) per rank: 11 / 16
constexpr int HMIN = HEADS / NR;           // 10 / 16
constexpr int EB = maxw(BLOCKS);           // routed expert 128-blocks per rank: 3 / 4
constexpr int DENSE_CH = maxw(geom::DENSE_I / 128) * 128;   // dense MLP channels per rank, max (layers 0..2; 96 blocks of 128): 2048 / 3072 / 2560 at TP5
constexpr int QBW = HMAX * 256;            // q_b rows / attention output / o_proj K per rank: 2816 / 4096
constexpr int SMALLW = QBW > DENSE_CH ? QBW : DENSE_CH;   // widest per-rank non-HID activation (q_a 2048 too): 2816 / 4096
}
