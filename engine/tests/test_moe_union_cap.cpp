// CPU unit test & fidelity verifier for GR8_MOE_UNION_CAP / GR8_MOE_UNION_MODE.
// Mirrors the exact algorithm of engine/kernels/moe_union_cap.cuh (including the 256-entry
// bitonic sort, tie-breaking, deterministic ascending-m scoring, L2/L1/none renorm,
// dense M > 8 multi-slot packing, and keep masks) and verifies every RoutePlan invariant
// against an independent reference implementation + stock k_router_plan.
//
// Build & run (CPU only, no GPU):
//   g++ -O2 -std=c++17 -Wall -Wextra -ffp-contract=off -o /tmp/test_moe_union_cap engine/tests/test_moe_union_cap.cpp && /tmp/test_moe_union_cap

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
constexpr int NEXP = 256;
constexpr int TOPK = 8;
constexpr int MMAX = 32;
constexpr int SMAX = 288;
constexpr float ROUTE_SCALE = 2.5f;

struct RoutePlan
{
    int n_slots;
    int pad_[3];
    int slot_expert[SMAX];
    int slot_mask[SMAX];
    float slot_w[SMAX][8];
    int slot_tok[SMAX][8];
    int tok_slot[MMAX][8];
    int tok_col[MMAX][8];
    float tok_w[MMAX][8];
    int tok_expert[MMAX][8];
};

static inline int popc32(uint32_t x) { return __builtin_popcount(x); }
static inline int ctz32(uint32_t x) { return __builtin_ctz(x); }

// Exact CPU mirror of stock glue::k_router_plan
void run_stock_plan(const float* logits, const float* bias, int M, RoutePlan& P, uint32_t keep = 0u)
{
    std::memset(&P, 0, sizeof(P));
    for (int i = 0; i < M * TOPK * 8; ++i)
    {
        (&P.slot_w[0][0])[i] = 0.f;
        (&P.slot_tok[0][0])[i] = -1;
    }
    for (int m = 0; m < MMAX; ++m)
        for (int k = 0; k < TOPK; ++k)
        {
            P.tok_slot[m][k] = -1;
            P.tok_col[m][k] = 0;
            P.tok_w[m][k] = 0.f;
            P.tok_expert[m][k] = -1;
        }

    const bool dense = M > 8;
    const uint32_t on = keep ? keep : 0xffffffffu;
    const uint32_t m_bits = M >= 32 ? 0xffffffffu : ((1u << M) - 1u);
    const uint32_t active_mask = on & m_bits;

    uint32_t emask[NEXP] = {};
    int sel_e[MMAX][TOPK] = {};
    float sel_w[MMAX][TOPK] = {};

    for (int m = 0; m < M; ++m)
    {
        if (!(active_mask >> m & 1u)) continue;
        float s[NEXP], key[NEXP];
        for (int e = 0; e < NEXP; ++e)
        {
            s[e] = 1.f / (1.f + std::exp(-logits[m * NEXP + e]));
            key[e] = s[e] + bias[e];
        }
        bool taken[NEXP] = {};
        for (int k = 0; k < TOPK; ++k)
        {
            float bk = -std::numeric_limits<float>::infinity();
            int be = 0x7fffffff;
            for (int e = 0; e < NEXP; ++e)
            {
                if (!taken[e] && (key[e] > bk || (key[e] == bk && e < be)))
                {
                    bk = key[e];
                    be = e;
                }
            }
            taken[be] = true;
            sel_e[m][k] = be;
            sel_w[m][k] = s[be];
        }
        float sum = 0.f;
        for (int k = 0; k < TOPK; ++k) sum += sel_w[m][k];
        for (int k = 0; k < TOPK; ++k)
        {
            sel_w[m][k] = sel_w[m][k] / sum * ROUTE_SCALE;
            emask[sel_e[m][k]] |= (1u << m);
        }
    }

    int slot_of[NEXP] = {};
    int cur = 0;
    for (int e = 0; e < NEXP; ++e)
    {
        uint32_t em = emask[e];
        int cnt = popc32(em);
        int nsl = dense ? ((cnt + 7) >> 3) : (em != 0u ? 1 : 0);
        if (nsl)
        {
            slot_of[e] = cur;
            if (!dense)
            {
                P.slot_expert[cur] = e;
                P.slot_mask[cur] = static_cast<int>(em);
            }
            else
            {
                for (int j = 0; j < nsl; ++j)
                {
                    P.slot_expert[cur + j] = e;
                    int rem = cnt - 8 * j < 8 ? cnt - 8 * j : 8;
                    P.slot_mask[cur + j] = static_cast<int>((1u << rem) - 1u);
                }
            }
            cur += nsl;
        }
    }
    P.n_slots = cur;

    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < TOPK; ++k)
        {
            if (!(active_mask >> m & 1u)) continue;
            int e = sel_e[m][k];
            int r = popc32(emask[e] & ((1u << m) - 1u));
            int s2 = slot_of[e] + (dense ? r >> 3 : 0);
            int col = dense ? (r & 7) : m;
            P.tok_slot[m][k] = s2;
            P.tok_col[m][k] = col;
            P.tok_w[m][k] = sel_w[m][k];
            P.tok_expert[m][k] = e;
            P.slot_w[s2][col] = sel_w[m][k];
            P.slot_tok[s2][col] = m;
        }
    }
}

// Exact CPU mirror of moe_union_cap::k_router_plan_capped (including the 256-entry bitonic sort!)
void run_capped_plan(const float* logits, const float* bias, int M, RoutePlan& P,
                     int cap, int mode, uint32_t keep = 0u)
{
    std::memset(&P, 0, sizeof(P));
    for (int i = 0; i < M * TOPK * 8; ++i)
    {
        (&P.slot_w[0][0])[i] = 0.f;
        (&P.slot_tok[0][0])[i] = -1;
    }
    for (int m = 0; m < MMAX; ++m)
        for (int k = 0; k < TOPK; ++k)
        {
            P.tok_slot[m][k] = -1;
            P.tok_col[m][k] = 0;
            P.tok_w[m][k] = 0.f;
            P.tok_expert[m][k] = -1;
        }

    const bool dense = M > 8;
    const uint32_t on = keep ? keep : 0xffffffffu;
    const uint32_t m_bits = M >= 32 ? 0xffffffffu : ((1u << M) - 1u);
    const uint32_t active_mask = on & m_bits;

    uint32_t emask[NEXP] = {};
    int sel_e[MMAX][TOPK] = {};
    float sel_w[MMAX][TOPK] = {};
    unsigned char kept_exp[NEXP] = {};

    for (int m = 0; m < M; ++m)
    {
        if (!(active_mask >> m & 1u)) continue;
        float s[NEXP], key[NEXP];
        for (int e = 0; e < NEXP; ++e)
        {
            s[e] = 1.f / (1.f + std::exp(-logits[m * NEXP + e]));
            key[e] = s[e] + bias[e];
        }
        bool taken[NEXP] = {};
        for (int k = 0; k < TOPK; ++k)
        {
            float bk = -std::numeric_limits<float>::infinity();
            int be = 0x7fffffff;
            for (int e = 0; e < NEXP; ++e)
            {
                if (!taken[e] && (key[e] > bk || (key[e] == bk && e < be)))
                {
                    bk = key[e];
                    be = e;
                }
            }
            taken[be] = true;
            sel_e[m][k] = be;
            sel_w[m][k] = s[be];
        }
        float sum = 0.f;
        for (int k = 0; k < TOPK; ++k) sum += sel_w[m][k];
        for (int k = 0; k < TOPK; ++k)
        {
            sel_w[m][k] = sel_w[m][k] / sum * ROUTE_SCALE;
            emask[sel_e[m][k]] |= (1u << m);
        }
    }

    int n_distinct = 0;
    for (int e = 0; e < NEXP; ++e) n_distinct += (emask[e] != 0u);

    if (n_distinct <= cap)
    {
        for (int e = 0; e < NEXP; ++e) kept_exp[e] = (emask[e] != 0u) ? 1 : 0;
    }
    else
    {
        const int policy = mode % 3;
        const int renorm = mode / 3;
        float bs_val[NEXP];
        int bs_id[NEXP];
        for (int t = 0; t < NEXP; ++t)
        {
            float my_score = -std::numeric_limits<float>::infinity();
            uint32_t em0 = emask[t];
            if (em0 != 0u)
            {
                float sc = 0.f;
                bool is_anchor0 = false;
                bool is_top1 = false;
                uint32_t tmp = em0;
                while (tmp)
                {
                    int m = ctz32(tmp);
                    tmp &= tmp - 1u;
                    int rk = -1;
                    for (int k = 0; k < TOPK; ++k)
                        if (sel_e[m][k] == t) { rk = k; break; }
                    if (rk >= 0)
                    {
                        float wm = sel_w[m][rk];
                        if (policy == 1)
                        {
                            float gamma = 1.f / (1.f + 0.25f * static_cast<float>(m));
                            sc += gamma * wm;
                            if (m == 0) is_anchor0 = true;
                            if (rk == 0) is_top1 = true;
                        }
                        else
                        {
                            sc += wm;
                            if (rk == 0) is_top1 = true;
                        }
                    }
                }
                if (policy == 0)
                {
                    if (is_top1) sc += 100.f;
                }
                else if (policy == 1)
                {
                    if (is_anchor0) sc += 1000.f;
                    if (is_top1) sc += 100.f;
                }
                my_score = sc;
            }
            bs_val[t] = my_score;
            bs_id[t] = t;
        }

        // Exact mirror of GPU 256-thread bitonic sort
        for (int k = 2; k <= NEXP; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                for (int t = 0; t < NEXP; ++t)
                {
                    int ixj = t ^ j;
                    if (ixj > t)
                    {
                        bool desc_dir = (t & k) == 0;
                        float va = bs_val[t], vb = bs_val[ixj];
                        int ia = bs_id[t], ib = bs_id[ixj];
                        bool a_better = (va > vb) || (va == vb && ia < ib);
                        bool swap = desc_dir ? !a_better : a_better;
                        if (swap)
                        {
                            bs_val[t] = vb; bs_val[ixj] = va;
                            bs_id[t] = ib; bs_id[ixj] = ia;
                        }
                    }
                }
            }
        }

        // Verify bitonic sort produced exact descending order!
        for (int i = 0; i + 1 < NEXP; ++i)
        {
            bool ok = (bs_val[i] > bs_val[i + 1]) ||
                      (bs_val[i] == bs_val[i + 1] && bs_id[i] < bs_id[i + 1]);
            assert(ok && "Bitonic sort order violation!");
        }

        for (int t = 0; t < cap && t < NEXP; ++t)
            if (bs_val[t] > -std::numeric_limits<float>::infinity())
                kept_exp[bs_id[t]] = 1;

        for (int e = 0; e < NEXP; ++e)
            if (!kept_exp[e]) emask[e] = 0u;

        for (int m = 0; m < M; ++m)
        {
            if (!(active_mask >> m & 1u)) continue;
            int n_kept = 0;
            float sum_w = 0.f, sum_w2 = 0.f, orig_w2 = 0.f;
            for (int k = 0; k < TOPK; ++k)
            {
                float wv = sel_w[m][k];
                orig_w2 += wv * wv;
                if (kept_exp[sel_e[m][k]])
                {
                    ++n_kept;
                    sum_w += wv;
                    sum_w2 += wv * wv;
                }
            }
            if (n_kept > 0 && n_kept < TOPK)
            {
                float scale = 1.f;
                if (renorm == 0) scale = (sum_w2 > 0.f) ? std::sqrt(orig_w2 / sum_w2) : 0.f;
                else if (renorm == 1) scale = (sum_w > 0.f) ? (ROUTE_SCALE / sum_w) : 0.f;
                for (int k = 0; k < TOPK; ++k)
                {
                    if (kept_exp[sel_e[m][k]]) sel_w[m][k] *= scale;
                    else sel_w[m][k] = 0.f;
                }
            }
            else if (n_kept == 0)
            {
                for (int k = 0; k < TOPK; ++k) sel_w[m][k] = 0.f;
            }
        }
    }

    int slot_of[NEXP] = {};
    int cur = 0;
    for (int e = 0; e < NEXP; ++e)
    {
        uint32_t em = emask[e];
        int cnt = popc32(em);
        int nsl = dense ? ((cnt + 7) >> 3) : (em != 0u ? 1 : 0);
        if (nsl)
        {
            slot_of[e] = cur;
            if (!dense)
            {
                P.slot_expert[cur] = e;
                P.slot_mask[cur] = static_cast<int>(em);
            }
            else
            {
                for (int j = 0; j < nsl; ++j)
                {
                    P.slot_expert[cur + j] = e;
                    int rem = cnt - 8 * j < 8 ? cnt - 8 * j : 8;
                    P.slot_mask[cur + j] = static_cast<int>((1u << rem) - 1u);
                }
            }
            cur += nsl;
        }
    }
    P.n_slots = cur;
    if (cur == 0) P.slot_expert[0] = 0;

    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < TOPK; ++k)
        {
            if (!(active_mask >> m & 1u) || !kept_exp[sel_e[m][k]])
            {
                P.tok_slot[m][k] = -1;
                P.tok_col[m][k] = 0;
                P.tok_w[m][k] = 0.f;
                P.tok_expert[m][k] = -1;
                continue;
            }
            int e = sel_e[m][k];
            int r = popc32(emask[e] & ((1u << m) - 1u));
            int s2 = slot_of[e] + (dense ? r >> 3 : 0);
            int col = dense ? (r & 7) : m;
            P.tok_slot[m][k] = s2;
            P.tok_col[m][k] = col;
            P.tok_w[m][k] = sel_w[m][k];
            P.tok_expert[m][k] = e;
            P.slot_w[s2][col] = sel_w[m][k];
            P.slot_tok[s2][col] = m;
        }
    }
}

void verify_invariants(const RoutePlan& P, const RoutePlan& stock, int M, int cap, int mode, uint32_t keep)
{
    const bool dense = M > 8;
    const uint32_t on = keep ? keep : 0xffffffffu;
    const uint32_t m_bits = M >= 32 ? 0xffffffffu : ((1u << M) - 1u);
    const uint32_t active_mask = on & m_bits;
    const int n_active = popc32(active_mask);
    const int policy = mode % 3;
    const int renorm = mode / 3;

    // Count stock distinct experts
    int stock_distinct = 0;
    for (int s = 0; s < stock.n_slots; ++s)
        if (s == 0 || stock.slot_expert[s] != stock.slot_expert[s - 1]) ++stock_distinct;

    // 1. If stock_distinct <= cap, MUST be 0-ULP bit-exact to stock!
    if (stock_distinct <= cap)
    {
        assert(P.n_slots == stock.n_slots);
        for (int s = 0; s < P.n_slots; ++s)
        {
            assert(P.slot_expert[s] == stock.slot_expert[s]);
            assert(P.slot_mask[s] == stock.slot_mask[s]);
            for (int c = 0; c < 8; ++c)
            {
                assert(P.slot_tok[s][c] == stock.slot_tok[s][c]);
                assert(std::memcmp(&P.slot_w[s][c], &stock.slot_w[s][c], sizeof(float)) == 0);
            }
        }
        for (int m = 0; m < MMAX; ++m)
            for (int k = 0; k < TOPK; ++k)
            {
                assert(P.tok_slot[m][k] == stock.tok_slot[m][k]);
                assert(P.tok_col[m][k] == stock.tok_col[m][k]);
                assert(P.tok_expert[m][k] == stock.tok_expert[m][k]);
                assert(std::memcmp(&P.tok_w[m][k], &stock.tok_w[m][k], sizeof(float)) == 0);
            }
        return;
    }

    // 2. Distinct experts <= cap and strictly non-decreasing
    assert(P.n_slots >= 0 && P.n_slots <= SMAX);
    if (P.n_slots == 0) assert(P.slot_expert[0] >= 0 && P.slot_expert[0] < NEXP);
    int distinct = 0;
    for (int s = 0; s < P.n_slots; ++s)
    {
        assert(P.slot_expert[s] >= 0 && P.slot_expert[s] < NEXP);
        if (s > 0)
        {
            if (!dense) assert(P.slot_expert[s] > P.slot_expert[s - 1]);
            else assert(P.slot_expert[s] >= P.slot_expert[s - 1]);
        }
        if (s == 0 || P.slot_expert[s] != P.slot_expert[s - 1]) ++distinct;
    }
    assert(distinct <= cap);
    assert(distinct == std::min(stock_distinct, cap));

    // 3. Cross-check slot_* <-> tok_* consistency & finiteness
    int slot_hits[SMAX][8] = {};
    for (int m = 0; m < MMAX; ++m)
    {
        int n_kept = 0;
        float sum_w = 0.f, sum_w2 = 0.f, orig_w2 = 0.f;
        for (int k = 0; k < TOPK; ++k)
        {
            if (m < M && (active_mask >> m & 1u))
                orig_w2 += stock.tok_w[m][k] * stock.tok_w[m][k];

            int s = P.tok_slot[m][k];
            if (s < 0)
            {
                assert(P.tok_col[m][k] == 0);
                assert(P.tok_w[m][k] == 0.f);
                assert(P.tok_expert[m][k] == -1);
                continue;
            }
            assert(m < M && (active_mask >> m & 1u));
            assert(s >= 0 && s < P.n_slots);
            int c = P.tok_col[m][k];
            assert(c >= 0 && c < 8);
            assert(std::isfinite(P.tok_w[m][k]) && P.tok_w[m][k] > 0.f);
            assert(P.tok_expert[m][k] == P.slot_expert[s]);
            assert(P.tok_expert[m][k] == stock.tok_expert[m][k]);
            assert((P.slot_mask[s] >> c) & 1);
            assert(P.slot_tok[s][c] == m);
            assert(std::memcmp(&P.slot_w[s][c], &P.tok_w[m][k], sizeof(float)) == 0);
            slot_hits[s][c]++;
            n_kept++;
            sum_w += P.tok_w[m][k];
            sum_w2 += P.tok_w[m][k] * P.tok_w[m][k];
        }

        if (m < M && (active_mask >> m & 1u))
        {
            // Policy guarantees:
            if (policy == 0 && cap >= n_active)
            {
                assert(P.tok_slot[m][0] >= 0 && "balanced policy violated top-1 floor!");
            }
            if (policy == 1)
            {
                const bool row0_on = (active_mask & 1u) != 0u;
                if (m == 0 && cap >= TOPK)
                {
                    assert(n_kept == TOPK && "anchor0 policy did not preserve all 8 row-0 experts!");
                    for (int k = 0; k < TOPK; ++k)
                        assert(std::memcmp(&P.tok_w[0][k], &stock.tok_w[0][k], sizeof(float)) == 0);
                }
                int floor_need = row0_on ? (TOPK + (n_active - 1)) : n_active;
                if (cap >= floor_need)
                {
                    assert(P.tok_slot[m][0] >= 0 && "anchor0 policy violated top-1 floor!");
                }
            }
            // Renorm checks:
            if (n_kept == TOPK)
            {
                for (int k = 0; k < TOPK; ++k)
                    assert(std::memcmp(&P.tok_w[m][k], &stock.tok_w[m][k], sizeof(float)) == 0);
            }
            else if (n_kept > 0)
            {
                if (renorm == 0)
                {
                    assert(std::fabs(sum_w2 - orig_w2) <= 1e-4f * orig_w2 + 1e-6f);
                }
                else if (renorm == 1)
                {
                    assert(std::fabs(sum_w - ROUTE_SCALE) <= 1e-4f);
                }
                else if (renorm == 2)
                {
                    for (int k = 0; k < TOPK; ++k)
                        if (P.tok_slot[m][k] >= 0)
                            assert(std::memcmp(&P.tok_w[m][k], &stock.tok_w[m][k], sizeof(float)) == 0);
                }
            }
        }
    }
    for (int s = 0; s < P.n_slots; ++s)
        for (int c = 0; c < 8; ++c)
            if ((P.slot_mask[s] >> c) & 1) assert(slot_hits[s][c] == 1);
            else assert(slot_hits[s][c] == 0 && P.slot_tok[s][c] == -1 && P.slot_w[s][c] == 0.f);
}

}  // namespace

int main()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::normal_distribution<float> bd(0.f, 0.1f);

    const int ms[] = {1, 2, 3, 4, 8, 12, 16, 24, 32};
    const int caps[] = {1, 4, 8, 12, 16, 20, 24, 32, 64, 256};

    int total_checks = 0;
    for (int M : ms)
    {
        for (int trial = 0; trial < 5; ++trial)
        {
            std::vector<float> logits(static_cast<size_t>(M) * NEXP);
            std::vector<float> bias(NEXP);
            for (auto& v : bias) v = bd(rng);
            if (trial == 0)
            {
                // Exact ties everywhere!
                std::fill(logits.begin(), logits.end(), 0.25f);
                std::fill(bias.begin(), bias.end(), 0.f);
            }
            else if (trial == 1)
            {
                // Extreme logits +/- 40
                for (auto& v : logits) v = (rng() & 1) ? 40.f : -40.f;
            }
            else
            {
                // Realistic peaked + correlated across rows (shared prompt/domain bias)
                std::vector<float> base(NEXP);
                for (auto& v : base) v = nd(rng) * 0.6f;
                for (int m = 0; m < M; ++m)
                    for (int e = 0; e < NEXP; ++e)
                        logits[static_cast<size_t>(m) * NEXP + e] = base[e] + nd(rng) * 1.2f;
            }

            const uint32_t keeps[3] = {0u, 1u, (M >= 3 ? (1u | (1u << (M - 1))) : 1u)};
            for (uint32_t keep : keeps)
            {
                RoutePlan stock{};
                run_stock_plan(logits.data(), bias.data(), M, stock, keep);
                for (int cap : caps)
                {
                    for (int mode = 0; mode < 9; ++mode)
                    {
                        RoutePlan P{};
                        run_capped_plan(logits.data(), bias.data(), M, P, cap, mode, keep);
                        verify_invariants(P, stock, M, cap, mode, keep);
                        ++total_checks;
                    }
                }
            }
        }
    }
    std::printf("PASS: %d exhaustive RoutePlan invariant checks passed (0 failures).\n\n", total_checks);

    // Analytical MoE Fidelity Simulation at M=8 (solo DFlash2) and M=24 (C=8 MTP=2)
    // Uses realistic synthetic expert outputs: E_e = mu_shared * sqrt(gamma) + eps_e * sqrt(1 - gamma)
    // with pairwise expert correlation gamma = 0.25, dim D = 256, 200 random batches.
    constexpr int D = 256;
    constexpr int NBATCH = 200;
    constexpr float CORR = 0.25f;
    std::vector<float> mu(D);
    std::vector<std::vector<float>> Evec(NEXP, std::vector<float>(D));
    for (int d = 0; d < D; ++d) mu[d] = nd(rng);
    for (int e = 0; e < NEXP; ++e)
        for (int d = 0; d < D; ++d)
            Evec[e][d] = std::sqrt(CORR) * mu[d] + std::sqrt(1.f - CORR) * nd(rng);

    const char* pol_name[3] = {"balanced", "anchor0 ", "puremass"};
    const char* ren_name[3] = {"L2  ", "L1  ", "none"};

    for (int M : {8, 24})
    {
        std::printf("=== Synthetic MoE Routing Fidelity Sweep (M=%d, %d batches, expert_corr=%.2f) ===\n",
                    M, NBATCH, CORR);
        std::printf("  %-5s %-9s %-5s | %-7s %-8s %-9s %-9s %-9s\n",
                    "B_cap", "Policy", "Renrm", "E_uniq", "MassKept", "SqMass%", "RelL2_All", "RelL2_R0");

        std::vector<int> sweep_caps = (M == 8) ? std::vector<int>{12, 16, 20, 24, 32, 256}
                                               : std::vector<int>{24, 32, 48, 64, 256};
        for (int cap : sweep_caps)
        {
            for (int mode : {0, 1, 3, 4, 6, 7})
            {
                if (cap == 256 && mode != 0) continue;
                double sum_e = 0, sum_mass = 0, sum_sq = 0, sum_err_all = 0, sum_err_r0 = 0;
                int cnt_all = 0;
                std::mt19937 srng(999);
                for (int b = 0; b < NBATCH; ++b)
                {
                    std::vector<float> logits(static_cast<size_t>(M) * NEXP);
                    std::vector<float> bias(NEXP);
                    for (auto& v : bias) v = bd(srng);
                    std::vector<float> shared_l(NEXP);
                    for (auto& v : shared_l) v = nd(srng) * 0.45f;
                    for (int m = 0; m < M; ++m)
                        for (int e = 0; e < NEXP; ++e)
                            logits[static_cast<size_t>(m) * NEXP + e] = shared_l[e] + nd(srng) * 1.35f;

                    RoutePlan st{}, cp{};
                    run_stock_plan(logits.data(), bias.data(), M, st);
                    run_capped_plan(logits.data(), bias.data(), M, cp, cap, mode);

                    int dist = 0;
                    for (int s = 0; s < cp.n_slots; ++s)
                        if (s == 0 || cp.slot_expert[s] != cp.slot_expert[s - 1]) ++dist;
                    sum_e += dist;

                    for (int m = 0; m < M; ++m)
                    {
                        double kept_m = 0, orig_m = 0, kept_sq = 0, orig_sq = 0;
                        std::vector<double> y_st(D, 0.0), y_cp(D, 0.0);
                        for (int k = 0; k < TOPK; ++k)
                        {
                            int es = st.tok_expert[m][k];
                            float ws = st.tok_w[m][k];
                            orig_m += ws;
                            orig_sq += ws * ws;
                            for (int d = 0; d < D; ++d) y_st[d] += ws * Evec[es][d];
                            if (cp.tok_slot[m][k] >= 0)
                            {
                                kept_m += ws;
                                kept_sq += ws * ws;
                                float wc = cp.tok_w[m][k];
                                for (int d = 0; d < D; ++d) y_cp[d] += wc * Evec[es][d];
                            }
                        }
                        double num = 0, den = 0;
                        for (int d = 0; d < D; ++d)
                        {
                            double diff = y_cp[d] - y_st[d];
                            num += diff * diff;
                            den += y_st[d] * y_st[d];
                        }
                        double rel = std::sqrt(num / (den + 1e-12));
                        sum_mass += kept_m / orig_m;
                        sum_sq += kept_sq / orig_sq;
                        sum_err_all += rel;
                        if (m == 0) sum_err_r0 += rel;
                        ++cnt_all;
                    }
                }
                std::printf("  %-5d %-9s %-5s | %6.1f  %7.1f%%  %7.1f%%   %8.4f   %8.4f\n",
                            cap, pol_name[mode % 3], ren_name[mode / 3],
                            sum_e / NBATCH,
                            100.0 * sum_mass / cnt_all,
                            100.0 * sum_sq / cnt_all,
                            sum_err_all / cnt_all,
                            sum_err_r0 / NBATCH);
            }
        }
        std::printf("\n");
    }
    return 0;
}
