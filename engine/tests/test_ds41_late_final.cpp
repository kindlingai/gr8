// F8 #1 GR8_DS_PF_LATE_FINAL (ds41_pf_split.h late_skip / late_final_valid, s2/ds41_prefill.h ds41_chunk_collectives late_skip). Host only.
// 1. The rule: for prompts N and chunks of W rows, every position a later reader needs (MAIN step N-WINDOW..N-2, a later chunk's range B down to
//    its floor, or WINDOW-1 rows before its first row when it is not compacted) is written by a chunk that does NOT skip range B.
// 2. The collectives of a skipping chunk: no layer exchange c >= 2*i_late in sizes (Engram off) or in the order (Engram on); the rest unchanged.
#include "runtime/s2/ds41_prefill.h"
#include "runtime/ds41_pf_split.h"
#include <cstdio>
#include <vector>
static int bad = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); ++bad; } } while (0)
int main()
{
    namespace br = ds41::br;
    const int Wn = geom::WINDOW;
    CHECK(!br::late_final_valid(0, 2048) && !br::late_final_valid(2048, 2048) && br::late_final_valid(128, 2048) && !br::late_final_valid(127, 2048));
    long cases = 0, skips = 0;
    for (int W : {256, 1024, 2048})
        for (int T : {128, 200, 255})
        {
            if (!br::late_final_valid(T, W)) continue;
            for (int N = 2; N <= 4 * W + 3 * Wn + 7; ++N)
            {
                // chunks of the first N-1 rows: W, W, ..., remainder
                std::vector<int> first, rows;
                for (int f = 0; f < N - 1; f += W) { first.push_back(f); rows.push_back(std::min(W, N - 1 - f)); }
                std::vector<char> written(N, 0);   // late-layer positions written by a range B
                std::vector<char> skip(first.size(), 0);
                for (size_t k = 0; k < first.size(); ++k)
                {
                    const int ctx = first[k] + rows[k], after = N - 1 - ctx;
                    skip[k] = br::late_skip(after, T);
                    skips += skip[k];
                    if (!skip[k]) for (int p = ctx - br::late_rows(rows[k], T); p < ctx; ++p) written[p] = 1;
                }
                // readers after each chunk k: later chunks that run range B, then the MAIN step
                for (size_t k = 0; k < first.size(); ++k)
                {
                    if (skip[k]) continue;
                    const int ctx = first[k] + rows[k], late = br::late_rows(rows[k], T), floor = br::win_floor(ctx, rows[k], T);
                    for (int p = ctx - late; p < ctx; ++p)
                        for (int q = std::max(floor, p - Wn + 1); q < ctx - late; ++q)
                            if (!written[q]) { std::printf("W %d T %d N %d chunk %zu row %d reads unwritten %d\n", W, T, N, k, p, q); ++bad; goto next; }
                }
                for (int q = std::max(0, N - Wn); q < N - 1; ++q)
                    if (!written[q]) { std::printf("W %d T %d N %d MAIN reads unwritten %d\n", W, T, N, q); ++bad; goto next; }
                if (!skip.empty()) CHECK(!skip.back());   // the last chunk never skips
                ++cases;
            next:;
            }
        }
    std::printf("late-final rule: %ld prompts, %ld skipped chunks, every read position written\n", cases, skips);
    // collectives
    std::vector<int> ids; for (int l = 0; l < 40; ++l) ids.push_back(l);
    const int i_late = br::late_start(ids), nl = 40;
    CHECK(i_late == 21);
    for (int ecoll : {0, 2})
    {
        const auto a = g6s2::ds41_chunk_collectives(ids, ecoll, 2048, 10240, 12288, i_late, 128, false);
        const auto s = g6s2::ds41_chunk_collectives(ids, ecoll, 2048, 10240, 12288, i_late, 128, true);
        if (!ecoll) { CHECK(s.sizes.size() == size_t(2 * i_late) && s.order.empty()); }
        else
        {
            CHECK(s.sizes.size() == a.sizes.size());
            for (int c = 2 * i_late; c < 2 * nl; ++c) CHECK(s.sizes[c] == 0);
            for (int c : s.order) CHECK(c < 2 * i_late || c > 2 * nl);
            std::vector<int> want; for (int c : a.order) if (c < 2 * i_late || c > 2 * nl) want.push_back(c);
            CHECK(s.order == want && s.order.size() == size_t(2 * i_late + 2));
        }
        for (int c = 0; c < 2 * i_late; ++c) CHECK(s.sizes[c] == a.sizes[c]);
        for (size_t c = 2 * nl + 1; c < a.sizes.size(); ++c) CHECK(s.sizes[c] == a.sizes[c]);
    }
    std::printf("ds41 late final: %s\n", bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
