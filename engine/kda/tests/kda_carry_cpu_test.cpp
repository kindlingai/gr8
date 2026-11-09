// CPU check of the KDA carry mode (Args::src per row) on the CPU model of k_kda_core (kda_cpu.h: the same phase
// functions as the kernel, run thread by thread):
//   ring: from state X in slot src, a verify step of M = k + 1 rows with row_slot[r] = r; for every accepted count
//         a in 0..k, slot a equals (bit for bit) the state after an M = a + 1 run of rows 0..a from X in the stock mode
//         (src = nullptr, one slot), and the gated outputs of rows 0..a are the same bits.
//   plain: carry mode with row_slot[*] = src equals the stock mode.
//   sessions: S sessions in one step, session q in slots [q (K + 1), (q + 1) (K + 1)) with its first row loading its
//         own source (src[row0] >= 0, the other rows -1): every session's slots and outputs equal its ring case's,
//         so a session neither reads nor writes another's state. One-row sessions (TAIL steps) included.
//   g++ -O2 -ffp-contract=off -std=c++17 -DG6_MODEL_FLASH=1 engine/kda/tests/kda_carry_cpu_test.cpp (scripts/host_tests.sh;
//   no FMA contraction, which the host compiler would do differently in each M path)
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "kda_cpu.h"

using namespace kda;

constexpr int NH = 1, MXR = 8, SLOTS = 4 * (MXR + 1);
constexpr int DL = NH * DK;

static uint16_t host_bf(float f) { uint32_t u; memcpy(&u, &f, 4); return static_cast<uint16_t>(u >> 16); }

struct Buf
{
    std::vector<float> S, C;
    static size_t s_slot() { return (size_t)NH * DK * DV; }
    static size_t c_slot() { return (size_t)NHIST * 3 * NH * DK; }
    explicit Buf(int n) : S(n * s_slot()), C(n * c_slot()) {}
    void put(int slot, const std::vector<float>& s, const std::vector<float>& c)
    {
        std::copy(s.begin(), s.end(), S.begin() + slot * s_slot());
        std::copy(c.begin(), c.end(), C.begin() + slot * c_slot());
    }
    std::vector<float> s_of(int slot) const { return {S.begin() + slot * s_slot(), S.begin() + (slot + 1) * s_slot()}; }
    std::vector<float> c_of(int slot) const { return {C.begin() + slot * c_slot(), C.begin() + (slot + 1) * c_slot()}; }
};

int main()
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    auto vec = [&](size_t n, float sc) { std::vector<float> v(n); for (float& x : v) x = sc * U(rng); return v; };
    auto bvec = [&](size_t n, float sc, float off = 0.f) { std::vector<uint16_t> v(n); for (auto& x : v) x = host_bf(off + sc * U(rng)); return v; };

    PLayout pl; pl.dl = DL;
    const int ld = pl.ld();
    const std::vector<float> P = vec((size_t)MXR * ld, 1.f);
    const std::vector<uint16_t> fb = bvec((size_t)DL * DK, 0.05f), gb = bvec((size_t)DL * DK, 0.05f);
    const std::vector<uint16_t> cw0 = bvec((size_t)DL * CONV, 0.5f), cw1 = bvec((size_t)DL * CONV, 0.5f), cw2 = bvec((size_t)DL * CONV, 0.5f);
    const std::vector<float> A_log = vec(NH, 0.5f), dt_bias = vec(DL, 0.5f);
    const std::vector<uint16_t> onorm = bvec(DK, 0.2f, 1.f);
    const std::vector<float> S0 = vec((size_t)NH * DK * DV, 0.1f), C0 = vec((size_t)NHIST * 3 * NH * DK, 1.f);

    Args base;
    base.nh = NH; base.h0 = 0; base.nhmax = NH;
    base.conv_w[0] = cw0.data(); base.conv_w[1] = cw1.data(); base.conv_w[2] = cw2.data();
    base.fb = fb.data(); base.gb = gb.data(); base.A_log = A_log.data(); base.dt_bias = dt_bias.data(); base.onorm = onorm.data();
    std::vector<float> C((size_t)MXR * 3 * DL), G((size_t)MXR * DL), Z((size_t)MXR * DL), O((size_t)MXR * DL);
    base.C = C.data(); base.G = G.data(); base.Z = Z.data(); base.O = O.data();

    // rows 0..n-1 with the given slots and projections; srcs empty: the stock mode
    auto run = [&](Buf& b, const std::vector<float>& rows, const std::vector<int>& slots, const std::vector<int>& srcs) {
        std::vector<uint16_t> oh((size_t)slots.size() * DL);
        Args a = base; a.S = b.S.data(); a.cst = b.C.data(); a.M = int(slots.size());
        a.P = rows.data(); a.row_slot = slots.data(); a.src = srcs.empty() ? nullptr : srcs.data(); a.OH = oh.data();
        kda_cpu::core(a);
        return oh;
    };
    std::vector<std::vector<float>> rs(MXR + 1), rc(MXR + 1);
    std::vector<std::vector<uint16_t>> ro(MXR + 1);
    for (int n = 1; n <= MXR; ++n)
    {
        Buf b(1); b.put(0, S0, C0);
        ro[n] = run(b, P, std::vector<int>(n, 0), {});
        rs[n] = b.s_of(0); rc[n] = b.c_of(0);
    }
    // the stock mode is causal in the rows (a check of the model, not of the carry)
    int checks = 0, fails = 0;
    auto expect = [&](bool ok, const char* what, int x, int y, int z) {
        ++checks;
        if (!ok) { ++fails; printf("FAIL %s %d %d %d\n", what, x, y, z); }
    };
    for (int n = 2; n <= MXR; ++n)
        expect(!memcmp(ro[n].data(), ro[n - 1].data(), (size_t)(n - 1) * DL * 2) && rs[n] != rs[n - 1], "causal", n, 0, 0);

    for (int k = 1; k < MXR; ++k)
        for (int sv : {k + 1, 0, k / 2 + 1, k})
        {
            std::vector<int> slots(k + 1), srcs(k + 1, -1);
            for (int r = 0; r <= k; ++r) slots[r] = r;
            srcs[0] = sv;
            Buf b(SLOTS); b.put(sv, S0, C0);
            const auto o = run(b, P, slots, srcs);
            for (int a = 0; a <= k; ++a)
                expect(b.s_of(a) == rs[a + 1] && b.c_of(a) == rc[a + 1] &&
                       !memcmp(o.data(), ro[a + 1].data(), (size_t)(a + 1) * DL * 2), "ring", k, sv, a);
        }
    for (int n = 1; n <= MXR; ++n)
        for (int sv : {0, 3})
        {
            std::vector<int> srcs(n, -1);
            srcs[0] = sv;
            Buf b(SLOTS); b.put(sv, S0, C0);
            const auto o = run(b, P, std::vector<int>(n, sv), srcs);
            expect(b.s_of(sv) == rs[n] && b.c_of(sv) == rc[n] && o == ro[n], "plain", n, sv, 0);
        }
    // sessions: rows per session (1 = a TAIL step), each from its own source inside its own slot range
    const std::vector<std::vector<int>> layouts = {{2, 2}, {3, 1}, {1, 3}, {1, 1, 1, 1}, {4, 4}, {2, 3, 3}, {1, 7}, {2, 2, 2, 2}};
    for (size_t li = 0; li < layouts.size(); ++li)
        for (int shift = 0; shift < 3; ++shift)
        {
            const auto& L = layouts[li];
            std::vector<int> slots, srcs;
            std::vector<float> rows;
            Buf b(SLOTS);
            for (size_t q = 0; q < L.size(); ++q)
            {
                const int base_q = int(q) * (MXR + 1), n = L[q], sv = base_q + (n > 1 ? (int(q) + shift) % n : 0);
                b.put(sv, S0, C0);
                for (int r = 0; r < n; ++r)
                {
                    slots.push_back(n > 1 ? base_q + r : base_q);
                    srcs.push_back(r ? -1 : sv);
                }
                rows.insert(rows.end(), P.begin(), P.begin() + (size_t)n * ld);
            }
            const auto o = run(b, rows, slots, srcs);
            int row0 = 0;
            for (size_t q = 0; q < L.size(); ++q)
            {
                const int base_q = int(q) * (MXR + 1), n = L[q];
                for (int a = 0; a < n; ++a)
                    expect(b.s_of(base_q + a) == rs[a + 1] && b.c_of(base_q + a) == rc[a + 1] &&
                           !memcmp(o.data() + (size_t)row0 * DL, ro[a + 1].data(), (size_t)(a + 1) * DL * 2),
                           "sessions", int(li), int(q), a);
                row0 += n;
            }
        }
    printf("kda_carry_cpu_test: %d checks, %d failed -> %s\n", checks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
