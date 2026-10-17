// CPU/loopback model of the context-parallel (CP) decode exchange (attn/attn.cu cp_front .. cp_finish) for a BATCHED step
// whose rows belong to DIFFERENT sessions: each row carries its own block table (G6AttnRow: table, context, table_size),
// sessions have different contexts, some below R (every page replicated) and some past R (pages past R/64 on rank
// lp % nr, G6_CP_REMOTE elsewhere: attn/bind.cu cp_table, runtime/s2.cuh s2_solo_cp_pages). nr ranks run one after the
// other; the fabric is plain host buffers with the engine's layouts:
//   cand_all [nr][rows][TOP] (cp_index -> cp_merge), part_all[d] = [nr] blocks of {c [n][Hd][D], m [n*Hd], l [n*Hd]}
//   (cp_group -> cp_finish, block stride rounded up to 4 floats as cp_lse).
// Every kernel is a loop nest over the CUDA grid with the kernel's own index expressions (row, r = start + row, FOR_ROW,
// CP_MINE, the cand / part offsets), so a row that read another row's table, bound or slot would show here.
// Checks per seed:
//   1. batched == solo: every row of session s in the batched step gives the same bits (merged index list, context of
//      every head group) as the step of session s alone (its own rows only, its own fresh physical placement).
//   2. the merged list == the exact global top-TOP of the row (all keys, one rank), and every rank merges the same list.
//   3. context == the replicated (cp_nr = 0) reference within 1e-5 normwise (max abs error / max abs value per
//      row-head; the LSE merge rounds differently).
// Scope: address / candidate-merge / LSE parity of the exchange. Not CUDA numerics, not the KV append (no append here).
//   4. every context value is finite (identical NaNs would pass 1 and 3).
//   5. no rank ever dereferences a G6_CP_REMOTE / out-of-range page (error bits as attn.cu page()).
// Dimensions are scaled down (latent 64, index 16, TOP 128); the algorithm does not depend on them.
// Build/run: g++ -O2 -std=c++17 cp_batch_cpu_test.cpp -o cp_batch_cpu_test && ./cp_batch_cpu_test [--seeds N]
// --break: negative control (for_row uses row 0 for every row); must FAIL every multi-session case.
// Exit: 0 all PASS, 1 any FAIL.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int NR = 4, PS = 64, R = 512, TOP = 128, PART = 8, D = 64, DI = 16, H = 8;   // H heads over NR head groups
constexpr int G6_CP_REMOTE = -2;
int g_pass = 0, g_fail = 0;
bool g_break = false;   // --break: negative control, for_row ignores r (every row reads row 0's table / bounds)

struct Row { const int* table; int context, page_count, table_size, sess; };   // G6AttnRow
struct Attn {                                                                       // the G6Attn fields the CP path reads
    int rows, context, page_size = PS, cp_nr, cp_rank, table_size, page_count, live_context;
    const int* block_table; const int* positions; unsigned error = 0;
    const float* mla;   // [page_count][PS][D] this rank's physical latent pages
    const float* idx;   // [page_count][PS][DI] index keys on the same pages
    const Row* rows_v;  // batched step: the row's own session
};
// attn.cu for_row: the row's own table / bounds; CP keeps page_count (the rank's physical count)
void for_row(Attn& a, int r) {
    if (!a.rows_v) return;
    const Row& x = a.rows_v[g_break ? 0 : r];
    a.block_table = x.table; a.live_context = x.context; a.table_size = x.table_size;
    if (!a.cp_nr) a.page_count = x.page_count;
}
bool cp_mine(const Attn& a, int pos) { return !a.cp_nr || (pos / a.page_size) % a.cp_nr == a.cp_rank; }
int page(Attn& a, int pos) {
    if (pos < 0 || pos >= a.live_context || pos / a.page_size >= a.table_size) { a.error |= 1; return -1; }
    const int p = a.block_table[pos / a.page_size];
    if (p < 0 || p >= a.page_count) { a.error |= 2; return -1; }
    return p;
}
uint32_t ordered_score(float s) {   // select.cuh: CUB float order, -0 == +0, nonfinite -> 0
    if (!std::isfinite(s)) return 0;
    if (s == 0) s = 0;
    uint32_t u; memcpy(&u, &s, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
uint64_t cp_key(float score, int pos) { const uint32_t k = ordered_score(score); return k ? ((uint64_t)k << 32) | (0xffffffffu - (uint32_t)pos) : 0; }

// ---- synthetic model: per session the logical KV (latent, index key) at every position; per row the queries
struct Session { int ctx; std::vector<float> lat, ik; };
struct Step {    // the step's rows (several sessions), activations replicated over ranks, q split by head group
    std::vector<int> sess, pos;
    std::vector<float> q, iq, iw;   // q [M][H][D], iq [M][DI], iw [M]
};
// physical placement of one rank: for every session of the world its logical -> physical table (shuffled pages), pool
struct RankWorld {
    std::vector<std::vector<int>> table;   // per session, table_size entries
    std::vector<float> mla, idx;
    int pages = 0;
};
RankWorld place(const std::vector<Session>& ss, const std::vector<int>& which, int rank, std::mt19937_64& rng) {
    RankWorld w;
    w.table.assign(ss.size(), {});
    std::vector<std::pair<int, int>> owned;   // (session, logical page) this rank stores
    for (int s : which) {
        const int lp_n = (ss[s].ctx + PS - 1) / PS;
        w.table[s].assign(lp_n, -1);
        for (int lp = 0; lp < lp_n; ++lp)
            if (lp < R / PS || lp % NR == rank) owned.push_back({s, lp});
            else w.table[s][lp] = G6_CP_REMOTE;
    }
    std::vector<int> phys(owned.size());
    for (size_t i = 0; i < phys.size(); ++i) phys[i] = (int)i;
    std::shuffle(phys.begin(), phys.end(), rng);   // non-contiguous, interleaved across sessions
    w.pages = (int)owned.size() + 3;               // a few spare pages (garbage, never read)
    w.mla.assign((size_t)w.pages * PS * D, NAN);
    w.idx.assign((size_t)w.pages * PS * DI, NAN);
    for (size_t i = 0; i < owned.size(); ++i) {
        const int s = owned[i].first, lp = owned[i].second, p = phys[i];
        w.table[s][lp] = p;
        for (int o = 0; o < PS; ++o) {
            const int pos = lp * PS + o;
            if (pos >= ss[s].ctx) break;
            memcpy(&w.mla[((size_t)p * PS + o) * D], &ss[s].lat[(size_t)pos * D], D * 4);
            memcpy(&w.idx[((size_t)p * PS + o) * DI], &ss[s].ik[(size_t)pos * DI], DI * 4);
        }
    }
    return w;
}

// ---- the CP path of one batched step on NR ranks (loopback)
struct Out { std::vector<int> indices;   // [M][TOP] (rank 0's; all ranks must agree)
             std::vector<float> ctx;     // [M][H][D] (head group d from rank d)
             bool ranks_agree = true; unsigned error = 0; };
Out run_cp(const std::vector<Session>& ss, const Step& st, const std::vector<RankWorld>& W) {
    const int M = (int)st.pos.size();
    int maxctx = 0;
    for (int s : st.sess) maxctx = std::max(maxctx, ss[s].ctx);
    std::vector<std::vector<Row>> rows(NR, std::vector<Row>(M));
    std::vector<Attn> A(NR);
    for (int r = 0; r < NR; ++r) {
        for (int i = 0; i < M; ++i) {
            const int s = st.sess[i];
            rows[r][i] = Row{W[r].table[s].data(), ss[s].ctx, W[r].pages, (int)W[r].table[s].size(), s};
        }
        Attn& a = A[r];
        a.rows = M; a.context = maxctx; a.cp_nr = NR; a.cp_rank = r; a.page_count = W[r].pages; a.live_context = maxctx;
        a.table_size = 0; a.block_table = nullptr; a.positions = st.pos.data(); a.mla = W[r].mla.data(); a.idx = W[r].idx.data();
        a.rows_v = rows[r].data();
    }
    // cp_index: per rank, index scores over my keys (index_scores: -inf past the row's position or not mine), local top-TOP
    // (selector order = cp_key descending), cp_cand_pack -> cand_all[rank][r][j]
    std::vector<uint64_t> cand((size_t)NR * M * TOP, 0);
    std::vector<std::vector<int>> local_ind(NR, std::vector<int>((size_t)M * TOP, -1));
    for (int rk = 0; rk < NR; ++rk)
        for (int row = 0; row < M; ++row) {
            Attn a = A[rk]; const int r = row; for_row(a, r);
            std::vector<float> scores(a.context, -INFINITY);
            for (int t = 0; t < a.context; ++t) {
                if (t > a.positions[r] || !cp_mine(a, t) || t >= a.live_context) continue;
                const int p = page(a, t); if (p < 0) continue;
                const float* k = a.idx + ((size_t)p * PS + t % PS) * DI;
                float sc = 0; for (int d = 0; d < DI; ++d) sc += std::max(0.f, st.iq[(size_t)r * DI + d] * k[d]);
                scores[t] = sc * st.iw[r];
            }
            std::vector<std::pair<uint64_t, int>> keys;
            for (int t = 0; t < a.context; ++t) if (const uint64_t k = cp_key(scores[t], t)) keys.push_back({k, t});
            std::sort(keys.begin(), keys.end(), [](auto x, auto y) { return x.first > y.first; });
            for (int j = 0; j < TOP && j < (int)keys.size(); ++j) local_ind[rk][(size_t)r * TOP + j] = keys[j].second;
            for (int j = 0; j < TOP; ++j) {   // cp_cand_pack
                const int id = local_ind[rk][(size_t)r * TOP + j];
                cand[((size_t)rk * a.rows + r) * TOP + j] = id >= 0 ? cp_key(scores[id], id) : 0;
            }
            A[rk].error |= a.error;
        }
    // cp_merge on every rank (exact: rank = own index + keys larger in every other list)
    std::vector<std::vector<int>> merged(NR, std::vector<int>((size_t)M * TOP, -1));
    for (int rk = 0; rk < NR; ++rk)
        for (int row = 0; row < M; ++row) {
            const int r = row, nr = A[rk].cp_nr; int* out = merged[rk].data() + (size_t)r * TOP;
            for (int s = 0; s < nr; ++s) {
                const uint64_t* L = cand.data() + ((size_t)s * A[rk].rows + r) * TOP;
                for (int j = 0; j < TOP; ++j) {
                    const uint64_t k = L[j]; if (!k) continue; int rank = j;
                    for (int o = 0; o < nr && rank < TOP; ++o) {
                        if (o == s) continue;
                        const uint64_t* O = cand.data() + ((size_t)o * A[rk].rows + r) * TOP;
                        int lo = 0, hi = TOP; while (lo < hi) { const int mid = (lo + hi) >> 1; if (O[mid] > k) lo = mid + 1; else hi = mid; }
                        rank += lo;
                    }
                    if (rank < TOP) out[rank] = int(0xffffffffu - uint32_t(k));
                }
            }
        }
    Out o; o.indices = merged[0];
    for (int rk = 1; rk < NR; ++rk) if (merged[rk] != merged[0]) o.ranks_agree = false;
    // cp_group: rank s computes, for destination head group d, its keys' unnormalized softmax and context partials
    // (mla_scores with CP_MINE, cp_softmax -> m, l, context_parts PART slices, merge_parts) into part_all[d][s]
    const int HG = H / NR;
    o.ctx.assign((size_t)M * H * D, 0);
    for (int d = 0; d < NR; ++d) {
        const int nh = M * HG; const size_t blk = ((size_t)nh * (D + 2) + 3) & ~(size_t)3;   // cp_lse: c, m, l (D = 512 there)
        std::vector<float> part_all((size_t)NR * blk, 0);
        for (int s = 0; s < NR; ++s) {
            float* part = part_all.data() + (size_t)s * blk;
            for (int row = 0; row < M; ++row)
                for (int h = 0; h < HG; ++h) {
                    Attn a = A[s]; const int r = row; for_row(a, r);
                    const int i = row * HG + h, hh = d * HG + h;
                    const float* q = &st.q[((size_t)r * H + hh) * D];
                    std::vector<float> p(TOP);
                    for (int j = 0; j < TOP; ++j) {
                        const int id = merged[s][(size_t)r * TOP + j];
                        float sum = -INFINITY;
                        if (id >= 0 && id <= a.positions[r] && cp_mine(a, id)) {
                            const int pg = page(a, id);
                            if (pg >= 0) { sum = 0; const float* v = a.mla + ((size_t)pg * PS + id % PS) * D; for (int e = 0; e < D; ++e) sum = fmaf(q[e], v[e], sum); sum *= 1.f / 16; }
                        }
                        p[j] = sum;
                    }
                    float mx = -INFINITY; for (float x : p) mx = std::max(mx, x);
                    float den = 0; for (float& x : p) { x = std::isfinite(mx) ? expf(x - mx) : 0; den += x; }
                    part[(size_t)nh * D + i] = mx; part[(size_t)nh * (D + 1) + i] = den;
                    std::vector<float> c(D, 0);
                    for (int pt = 0; pt < PART; ++pt) {   // context_parts + merge_parts
                        std::vector<float> acc(D, 0);
                        for (int j = pt * (TOP / PART); j < (pt + 1) * (TOP / PART); ++j) {
                            const int id = merged[s][(size_t)r * TOP + j];
                            if (!(id >= 0 && id <= a.positions[r] && p[j] != 0)) continue;
                            const int pg = page(a, id); if (pg < 0) continue;
                            const float* v = a.mla + ((size_t)pg * PS + id % PS) * D;
                            for (int e = 0; e < D; ++e) acc[e] = fmaf(p[j], v[e], acc[e]);
                        }
                        for (int e = 0; e < D; ++e) c[e] += acc[e];
                    }
                    memcpy(part + (size_t)i * D, c.data(), D * 4);
                    A[s].error |= a.error;
                }
        }
        // cp_finish on rank d: LSE merge in shard order 0..nr-1
        for (int i = 0; i < nh; ++i) {
            float Mx = -INFINITY; for (int sh = 0; sh < NR; ++sh) Mx = std::max(Mx, part_all[sh * blk + (size_t)nh * D + i]);
            float w[8], L = 0;
            for (int sh = 0; sh < NR; ++sh) { const float m = part_all[sh * blk + (size_t)nh * D + i]; w[sh] = std::isfinite(m) ? expf(m - Mx) : 0.f; L = fmaf(w[sh], part_all[sh * blk + (size_t)nh * (D + 1) + i], L); }
            if (!(L > 0)) A[d].error |= 16;
            const int row = i / HG, h = i % HG;
            for (int e = 0; e < D; ++e) {
                float c = 0; for (int sh = 0; sh < NR; ++sh) c = fmaf(w[sh], part_all[sh * blk + (size_t)i * D + e], c);
                o.ctx[((size_t)row * H + d * HG + h) * D + e] = L > 0 ? c / L : 0.f;
            }
        }
    }
    for (auto& a : A) o.error |= a.error;
    return o;
}

// ---- replicated reference (cp_nr = 0, one rank, logical KV directly): exact top-TOP, normalized softmax, context
void ref_row(const Session& S, const Step& st, int r, std::vector<int>& ind, std::vector<float>& ctx) {
    const int pos = st.pos[r];
    std::vector<std::pair<uint64_t, int>> keys;
    for (int t = 0; t <= pos; ++t) {
        float sc = 0; for (int d = 0; d < DI; ++d) sc += std::max(0.f, st.iq[(size_t)r * DI + d] * S.ik[(size_t)t * DI + d]);
        if (const uint64_t k = cp_key(sc * st.iw[r], t)) keys.push_back({k, t});
    }
    std::sort(keys.begin(), keys.end(), [](auto x, auto y) { return x.first > y.first; });
    ind.assign(TOP, -1);
    for (int j = 0; j < TOP && j < (int)keys.size(); ++j) ind[j] = keys[j].second;
    ctx.assign((size_t)H * D, 0);
    for (int h = 0; h < H; ++h) {
        const float* q = &st.q[((size_t)r * H + h) * D];
        std::vector<double> p(TOP, 0); double mx = -INFINITY;
        for (int j = 0; j < TOP; ++j) if (ind[j] >= 0) { double s = 0; for (int e = 0; e < D; ++e) s += (double)q[e] * S.lat[(size_t)ind[j] * D + e]; p[j] = s / 16; mx = std::max(mx, p[j]); }
        double den = 0; for (int j = 0; j < TOP; ++j) { p[j] = ind[j] >= 0 ? exp(p[j] - mx) : 0; den += p[j]; }
        for (int j = 0; j < TOP; ++j) if (ind[j] >= 0) for (int e = 0; e < D; ++e) ctx[(size_t)h * D + e] += (float)(p[j] / den * S.lat[(size_t)ind[j] * D + e]);
    }
}

bool run_case(const std::string& name, const std::vector<std::pair<int, int>>& spec, uint64_t seed) {   // (rows, ctx) per session
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0, 1);
    std::vector<Session> ss(spec.size());
    for (size_t s = 0; s < spec.size(); ++s) {
        ss[s].ctx = spec[s].second;
        ss[s].lat.resize((size_t)ss[s].ctx * D); for (auto& x : ss[s].lat) x = nd(rng);
        ss[s].ik.resize((size_t)ss[s].ctx * DI); for (auto& x : ss[s].ik) x = nd(rng);
    }
    Step all;
    for (size_t s = 0; s < spec.size(); ++s)
        for (int i = 0; i < spec[s].first; ++i) { all.sess.push_back((int)s); all.pos.push_back(spec[s].second - spec[s].first + i); }
    const int M = (int)all.pos.size();
    all.q.resize((size_t)M * H * D); for (auto& x : all.q) x = nd(rng);
    all.iq.resize((size_t)M * DI); for (auto& x : all.iq) x = nd(rng);
    all.iw.resize(M); for (auto& x : all.iw) x = 0.5f + std::fabs(nd(rng));
    std::vector<int> every; for (size_t s = 0; s < spec.size(); ++s) every.push_back((int)s);
    std::vector<RankWorld> W; for (int r = 0; r < NR; ++r) W.push_back(place(ss, every, r, rng));
    const Out B = run_cp(ss, all, W);
    std::string bad;
    if (B.error) bad += " batched-error=" + std::to_string(B.error);
    // identical NaNs would pass the bit compare and give normwise 0: every context value must be finite
    if (!std::all_of(B.ctx.begin(), B.ctx.end(), [](float x) { return std::isfinite(x); })) bad += " batched-nonfinite";
    if (!B.ranks_agree) bad += " batched-ranks-disagree";
    double worst = 0;
    int row0 = 0;
    for (size_t s = 0; s < spec.size(); ++s) {
        // solo: session s's rows alone, its own fresh placement (other physical pages), same activations
        Step so; const int n = spec[s].first;
        so.sess.assign(n, (int)s);
        so.pos.assign(all.pos.begin() + row0, all.pos.begin() + row0 + n);
        so.q.assign(all.q.begin() + (size_t)row0 * H * D, all.q.begin() + (size_t)(row0 + n) * H * D);
        so.iq.assign(all.iq.begin() + (size_t)row0 * DI, all.iq.begin() + (size_t)(row0 + n) * DI);
        so.iw.assign(all.iw.begin() + row0, all.iw.begin() + row0 + n);
        std::vector<RankWorld> Ws; for (int r = 0; r < NR; ++r) Ws.push_back(place(ss, {(int)s}, r, rng));
        const Out S = run_cp(ss, so, Ws);
        if (S.error) bad += " solo-error(s" + std::to_string(s) + ")";
        if (!std::all_of(S.ctx.begin(), S.ctx.end(), [](float x) { return std::isfinite(x); })) bad += " solo-nonfinite(s" + std::to_string(s) + ")";
        if (!S.ranks_agree) bad += " solo-ranks-disagree(s" + std::to_string(s) + ")";
        if (!std::equal(S.indices.begin(), S.indices.end(), B.indices.begin() + (size_t)row0 * TOP)) bad += " lists!=solo(s" + std::to_string(s) + ")";
        if (memcmp(S.ctx.data(), B.ctx.data() + (size_t)row0 * H * D, S.ctx.size() * 4)) bad += " ctx!=solo(s" + std::to_string(s) + ")";
        for (int i = 0; i < n; ++i) {
            std::vector<int> ri; std::vector<float> rc;
            ref_row(ss[s], all, row0 + i, ri, rc);
            if (!std::equal(ri.begin(), ri.end(), B.indices.begin() + (size_t)(row0 + i) * TOP)) bad += " lists!=exact(s" + std::to_string(s) + ")";
            for (int h = 0; h < H; ++h) {   // normwise per (row, head): max |got - want| / max |want|
                double num = 0, den = 0;
                for (int e = 0; e < D; ++e) {
                    const double got = B.ctx[((size_t)(row0 + i) * H + h) * D + e], want = rc[(size_t)h * D + e];
                    num = std::max(num, std::fabs(got - want)); den = std::max(den, std::fabs(want));
                }
                if (!std::isfinite(num) || !std::isfinite(den)) num = INFINITY;   // a NaN never passes the bound
                worst = std::max(worst, num / std::max(den, 1e-30));
            }
        }
        row0 += n;
    }
    if (worst > 1e-5) bad += " ctx-vs-replicated rel " + std::to_string(worst);
    char desc[512]; int k = snprintf(desc, sizeof desc, "%s: %zu sessions, %d rows, nr %d, R %d:", name.c_str(), spec.size(), M, NR, R);
    for (auto& x : spec) k += snprintf(desc + k, sizeof desc - k, " %dx%d%s", x.first, x.second, x.second <= R ? "(<R)" : "");
    printf("%s %s | max rel vs replicated %.2e%s\n", bad.empty() ? "PASS" : "FAIL", desc, worst, bad.c_str());
    (bad.empty() ? g_pass : g_fail)++;
    return bad.empty();
}
}   // namespace

int main(int argc, char** argv) {
    int seeds = 8;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--break")) g_break = true;
    }
    run_case("four past-R sessions, one row each", {{1, 3000}, {1, 1500}, {1, 4100}, {1, 900}}, 1);
    run_case("mixed below / past R, verify rows", {{3, 200}, {1, 2600}, {4, 512}, {2, 1300}}, 2);
    run_case("all below R", {{2, 300}, {1, 130}, {1, 500}}, 3);
    run_case("one past R + short sessions", {{1, 5000}, {1, 64}, {2, 100}}, 4);
    run_case("context at page / R edges", {{1, 513}, {1, 576}, {1, 577}, {1, 511}}, 5);
    for (int i = 0; i < seeds; ++i) {
        std::mt19937_64 r(1000 + i);
        const int ns = 2 + (int)(r() % 3);   // 2..4 sessions
        std::vector<std::pair<int, int>> spec;
        int left = 32;
        for (int j = 0; j < ns; ++j) { const int n = 1 + (int)(r() % std::min(8, left - (ns - 1 - j))); left -= n; spec.push_back({n, std::max(n, 64 + (int)(r() % 6000))}); }
        run_case("random seed " + std::to_string(1000 + i), spec, 100 + i);
    }
    printf("cp_batch_cpu_test: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
