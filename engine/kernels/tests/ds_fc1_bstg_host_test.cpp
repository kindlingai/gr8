// CPU-only exhaustive B-address check for FC1BSTG, with 8 and 32 sessions.
// The staged copy uses the same portable index helpers as the CUDA kernel;
// the direct-load oracle below independently spells out the original layout.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "ds_fc1_bstg.h"

static constexpr int K1 = 5120, KS = 8, NTG = 4, NGP = (K1 / 64) / KS;
static constexpr int STAGE_WORDS = KS * NTG * 4 * 64;

static uint32_t word(int i)
{
    uint32_t v = static_cast<uint32_t>(i) + 0x9e3779b9u;
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}

static size_t direct_word(int tok, int part, int rec, int step, int lane, int reg)
{
    if (tok < 0) tok = 0;
    return static_cast<size_t>(tok >> 3) * K1 * 4 +
           static_cast<size_t>(part * NGP * 4 + rec * 4 + step) * 64 +
           ((tok & 7) * 4 + (lane & 3)) * 2 + reg;
}

static bool check(int sessions)
{
    const int routes = sessions * 5, tiles = (routes + 7) / 8, groups = (tiles + NTG - 1) / NTG;
    std::vector<int> tile_tok(tiles * 8, -1);
    for (int r = 0; r < routes; ++r) tile_tok[r] = (r * 13 + r / 7) % sessions;
    std::vector<uint32_t> xfrag(static_cast<size_t>((sessions + 7) / 8) * K1 * 4);
    for (size_t i = 0; i < xfrag.size(); ++i) xfrag[i] = word(static_cast<int>(i));
    std::vector<uint32_t> stage(2 * STAGE_WORDS, 0xbad0f00du);
    uint64_t checked = 0, digest_direct = 0, digest_staged = 0;
    for (int group = 0; group < groups; ++group)
    {
        const int t0 = group * NTG, ntg = tiles - t0 < NTG ? tiles - t0 : NTG;
        for (int rec = 0; rec < NGP; ++rec)
        {
            uint32_t* dst = stage.data() + (rec & 1) * STAGE_WORDS;
            for (int i = 0; i < KS * NTG * 4 * 16; ++i)
            {
                const dsr::Fc1BstgCopy c = dsr::fc1_bstg_copy<NTG>(i);
                if (c.tile >= ntg) continue;
                const int lane0 = c.pair * 2;
                const int tok = tile_tok[(t0 + c.tile) * 8 + (lane0 >> 2)];
                const int src = dsr::fc1_bstg_src_word<K1, NGP>(tok, lane0, c.part, rec, c.step);
                if (src < 0 || static_cast<size_t>(src + 3) >= xfrag.size()) return false;
                for (int b = 0; b < 4; ++b) dst[i * 4 + b] = xfrag[src + b];
            }
            for (int part = 0; part < KS; ++part)
                for (int t = 0; t < ntg; ++t)
                    for (int step = 0; step < 4; ++step)
                        for (int lane = 0; lane < 32; ++lane)
                            for (int reg = 0; reg < 2; ++reg)
                            {
                                const int tok = tile_tok[(t0 + t) * 8 + (lane >> 2)];
                                const size_t src = direct_word(tok, part, rec, step, lane, reg);
                                const size_t off = (rec & 1) * STAGE_WORDS + dsr::fc1_bstg_dst_word<NTG>(part, t, step, lane) + reg;
                                if (src >= xfrag.size() || off >= stage.size()) return false;
                                const uint32_t a = xfrag[src], b = stage[off];
                                if (a != b) { std::printf("FC1_BSTG_RED sessions=%d group=%d rec=%d part=%d tile=%d step=%d lane=%d reg=%d direct=%08x staged=%08x\n", sessions, group, rec, part, t, step, lane, reg, a, b); return false; }
                                digest_direct = digest_direct * 1315423911u + a;
                                digest_staged = digest_staged * 1315423911u + b;
                                ++checked;
                            }
        }
    }
    if (digest_direct != digest_staged) return false;
    std::printf("FC1_BSTG_CPU_PASS sessions=%d routes=%d groups=%d words=%llu digest=%016llx\n", sessions, routes, groups,
                static_cast<unsigned long long>(checked), static_cast<unsigned long long>(digest_direct));
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const int sessions = std::atoi(argv[1]);
    if (sessions != 8 && sessions != 32) return 2;
    return check(sessions) ? 0 : 1;
}
