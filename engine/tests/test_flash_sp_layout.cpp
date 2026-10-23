// CPU four-rank layout gate for the Flash SP (sequence-parallel, G6_PF_RS=2) prefill — TP4-SP-MHC-RS-AG-BUILD.md
// "CPU four-rank gate". No CUDA, no NIC: this proves static LAYOUT bytes. It packs every step before any consumer
// runs, so it does NOT prove producer/consumer order, parity hazards or AG negatives: test_flash_sp_sched.cpp does
// (real per-rank stream order, FIFO proxies, slot/doorbell hazard shadows, mutation negatives). Not kernel arithmetic (that is
// test_flash_sp_arith.cpp) and not real-fabric bytes (that is engine/comms/tests/sp4_test.cpp, run in the lead's
// slot). Owner rows, slice strides, byte counts and the enqueue plan come from engine/runtime/pf_layout.h — the
// same single source the device kernels use. The byte moves below mirror, line for line:
//   pack_rs  <-> k_put_rs     (engine/runtime/prefill.cuh: owner p = pf_owner_of, dst slice p at (row - pf_r0) * RB)
//   pack_ag  <-> pf_put_ag_sp (engine/runtime/flash_prefill.cuh, k_put_publish_n: owner rows at send offset 0)
//   wire     <-> Fabric proxy (engine/comms/fabric.cpp: Rs slice p -> recv[c][sender] offset 0, u64 tail = seq
//              right after the payload; AG: payload + tail from send offset 0; collectives processed c0 -> c4)
//   rs_slice <-> rs_slice     (engine/runtime/prefill.cuh: local slice at send + me * ss, remote at recv offset 0)
//   wait/consume <-> k_wait_rs / MhcRsBlock / k_wait_ag / k_ag_copy (tails at the exact payload-length offsets,
//              value coll_seq(step, c, 5); AG rows land at global rows pf_r0(M, p))
// Build: g++ -std=c++17 -O2 -Wall -I engine/runtime engine/tests/test_flash_sp_layout.cpp -o sp_layout_test
#include "pf_layout.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <cstdint>
#include <set>
#include <vector>

using namespace g6rt;

namespace
{
constexpr int NR = 4;          // TP4
constexpr int H = 8;           // tiny row width: identical arithmetic, M = 8192 stays cheap
constexpr size_t RB = H * 2;   // BF16 row bytes (RS payload)
constexpr size_t RB4 = H * 4;  // FP32 row bytes (AG payload)
constexpr int MMAX = 8192;     // fabric sized for the real sub-batch maximum
constexpr int NCOLL = sp_ncoll;

int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)
#define CHECK_EQ(a, b) do { auto va_ = (a); auto vb_ = (b); if (!(va_ == vb_)) { fprintf(stderr, "FAIL %s:%d %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b, (long long)va_, (long long)vb_); fails++; } } while (0)

const unsigned POISON_A = 0xA5A5A5A5u, POISON_B = 0x5A5A5A5Au;
using u64 = unsigned long long;   // the engine's u64 (rank.cuh) is not host-safe; same width here

u64 tick_counter = 0;   // global happens-before clock: every wire / read stamps it

// one rank's uniform five-collective fabric: [2 parity][5] send + [2 parity][5][NR] recv, stride bytes each
struct SimFab
{
    int rank = 0;
    size_t stride = 0, ss = 0;
    std::vector<unsigned char> send, recv;
    void init(int rank_, size_t stride_, unsigned poison)
    {
        rank = rank_; stride = stride_;
        ss = (stride / NR) & ~size_t(255);
        const unsigned char b = (unsigned char)poison;
        send.assign(2 * NCOLL * stride, b);
        recv.assign(2 * NCOLL * NR * stride, b);
    }
    unsigned char* send_slot(u64 step, int c) { return send.data() + (((step & 1) * NCOLL + c) * stride); }
    unsigned char* recv_slot(u64 step, int c, int p) { return recv.data() + ((((step & 1) * NCOLL + c) * NR + p) * stride); }
    const unsigned char* rs_slice(u64 step, int c, int p) const   // as prefill.cuh rs_slice(loop = false)
    {
        const size_t par = (size_t)(step & 1) * NCOLL + c;
        return p == rank ? send.data() + par * stride + (size_t)rank * ss : recv.data() + (par * NR + p) * stride;
    }
};

// splitmix64: fast deterministic patterns (a fresh std::mt19937_64 per element was the runtime bottleneck)
inline u64 mix64(u64 x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// distinct finite BF16 payload for (rank, step, c, element); odd low bit keeps every element nonzero
unsigned short pat_rs(int r, u64 step, int c, size_t elem)
{
    return (unsigned short)((mix64(((u64)r << 56) ^ (step << 40) ^ ((u64)c << 32) ^ elem) & 0x7F7F) | 1u);
}
// distinct nonzero FP32 AG payload (finite exponent bits)
unsigned pat_ag(u64 tag, int r, u64 step, int c, size_t elem)
{
    return (unsigned)((mix64(tag ^ ((u64)r << 52) ^ (step << 38) ^ ((u64)c << 30) ^ (elem * 0xD1B54A32D192ED03ull)) & 0x7F7F7F7Fu) | 1u);
}

void put_u64(unsigned char* p, u64 v) { memcpy(p, &v, 8); }
u64 get_u64(const unsigned char* p) { u64 v; memcpy(&v, p, 8); return v; }

// GPU-side pack of one RS collective: src [M][H] BF16 -> slices of the send slot (k_put_rs's mapping)
void pack_rs(SimFab& f, u64 step, int c, const std::vector<unsigned short>& src, int M)
{
    unsigned char* slot = f.send_slot(step, c);
    for (int row = 0; row < M; ++row)
    {
        const int p = pf_owner_of(M, row, NR);
        memcpy(slot + (size_t)p * f.ss + (size_t)(row - pf_r0(M, p, NR)) * RB, &src[(size_t)row * H], RB);
    }
}
// GPU-side pack of one AG collective: this rank's owner rows of src [M][H] FP32 -> send slot offset 0 (k_put_ag_sp)
void pack_ag(SimFab& f, u64 step, int c, const std::vector<unsigned>& src, int M)
{
    const int first = pf_r0(M, f.rank, NR), rows = pf_rows(M, f.rank, NR);
    if (rows) memcpy(f.send_slot(step, c), &src[(size_t)first * H], (size_t)rows * RB4);
}

// the proxy's wire of (step, c) from sender s to every peer (fabric.cpp, uniform Rs form). Guarantees asserted
// here: exactly nb + 8 bytes land at the receiver's slot offset 0 (guards untouched), tail value coll_seq(step, c, 5).
void wire(std::vector<SimFab>& fab, u64 step, int c, int s, const std::vector<uint32_t>& sizes,
          const std::vector<uint32_t>& rs_bytes, unsigned poison)
{
    const u64 seq = g6::coll_seq(step, c, NCOLL);
    const bool rs = (sp_rs_mask >> c) & 1u;
    SimFab& S = fab[s];
    for (int p = 0; p < NR; ++p)
    {
        if (p == s) continue;
        const uint32_t nb = rs ? rs_bytes[p] : sizes[c];
        unsigned char* src = rs ? S.send_slot(step, c) + (size_t)p * S.ss : S.send_slot(step, c);
        put_u64(src + nb, seq);                                   // tail travels with the payload in one write
        unsigned char* dst = fab[p].recv_slot(step, c, s);        // peer's recv slot [c][s], offset 0
        memcpy(dst, src, nb + 8);
        CHECK(nb + 8 <= fab[p].stride);
        const unsigned char want = (unsigned char)poison;
        for (size_t i = nb + 8; i < fab[p].stride; ++i) CHECK_EQ(dst[i], want);   // receiver guard intact
    }
    ++tick_counter;
}

// k_wait_rs replica: every peer's RS tail for MY rows has landed (fail-closed: returns the error encoding on timeout)
int wait_rs(const std::vector<SimFab>& fab, u64 step, int c, int me, int M, bool present)
{
    const u64 seq = g6::coll_seq(step, c, NCOLL);
    for (int p = 0; p < NR; ++p)
    {
        if (p == me) continue;
        const u64 tail = get_u64(fab[me].rs_slice(step, c, p) + (size_t)pf_rows(M, me, NR) * RB);
        if (!present && tail != seq) return 0x61000000u | ((unsigned)c << 8) | (unsigned)p;   // ERR_AR | c | p
        CHECK_EQ(tail, seq);
    }
    ++tick_counter;
    return 0;
}

struct LaneData
{
    // per rank: [2 boundaries] partials [M][H] BF16, normalized FP32 rows, final residual rows
    std::vector<std::vector<std::vector<unsigned short>>> part;      // [r][b][M*H]
    std::vector<std::vector<std::vector<unsigned>>> norm;            // [r][b][M*H]
    std::vector<std::vector<unsigned>> resid;                        // [r][M*H]
};

LaneData gen_data(int M, u64 ls0, int nl)
{
    LaneData d;
    for (int r = 0; r < NR; ++r)
    {
        d.part.emplace_back(2);
        d.norm.emplace_back(2);
        d.resid.emplace_back((size_t)M * H);
        for (int b = 0; b < 2; ++b)
        {
            d.part[r][b].resize((size_t)M * H);
            d.norm[r][b].resize((size_t)M * H);
            for (int i = 0; i < M; ++i)
                for (int h = 0; h < H; ++h)
                {
                    d.part[r][b][(size_t)i * H + h] = pat_rs(r, ls0, b, (size_t)i * H + h);
                    d.norm[r][b][(size_t)i * H + h] = pat_ag(7, r, ls0, b, (size_t)i * H + h);
                }
            if (b == 1 && r == NR - 1) { }   // resid below
        }
        for (int i = 0; i < M; ++i)
            for (int h = 0; h < H; ++h)
                d.resid[r][(size_t)i * H + h] = pat_ag(9, r, ls0, 4, (size_t)i * H + h);
    }
    return d;
}

// one full SP sub-batch on one lane fabric. delay_step: that step's wires of one rank are deferred until the other
// ranks' wait_rs reports the stall (the two-parity exposure), then released. Checks every byte the owners receive.
void run_lane(int M, u64 ls0, int nl, unsigned poison, int delayed_rank, bool* saw_stall)
{
    const size_t ss_min = pf_ss_min_rb(MMAX, NR, RB);
    const size_t slot = pf_slot_bytes_rb(MMAX, true, NR, RB);
    const size_t stride = (slot + 8 + 255) & ~size_t(255);
    CHECK((stride / NR & ~size_t(255)) >= ss_min);          // fabric.cpp FCHECK mirror: ss * nr <= stride, ss >= slices
    CHECK(ss_min * NR <= stride);
    std::vector<SimFab> fab(NR);
    for (int r = 0; r < NR; ++r) fab[r].init(r, stride, poison);
    const size_t ss = fab[0].ss;
    std::vector<uint32_t> rs_bytes(NR), ag_bytes(NR);
    for (int p = 0; p < NR; ++p) { rs_bytes[p] = sp_rs_bytes(M, p, NR, H); ag_bytes[p] = sp_ag_bytes(M, p, NR, H); }
    for (int p = 0; p < NR; ++p) CHECK(rs_bytes[p] % 8 == 0 && rs_bytes[p] + 8 <= ss);   // enqueueStep FCHECK mirror

    const LaneData d = gen_data(M, ls0, nl);

    // enqueue plan (sp_sizes is the single source pf_enqueue_flash_sp uses): four sizes, five with the final c4
    for (int i = 0; i < nl; ++i)
    {
        const std::vector<uint32_t> s = sp_sizes(M, /*me*/ 0, NR, H, i == nl - 1);
        CHECK_EQ(s.size(), (size_t)(i == nl - 1 ? 5 : 4));
        if (i < nl - 1) CHECK(s == std::vector<uint32_t>({0u, ag_bytes[0], 0u, ag_bytes[0]}));
        else CHECK(s.back() == ag_bytes[0]);
    }

    // per-rank read timestamps of (step, c) for the two-parity write-after-read proof
    std::map<std::pair<u64, int>, u64> last_read;

    for (int i = 0; i < nl; ++i)
    {
        const u64 step = ls0 + i;
        const bool last = i == nl - 1;
        const int ncoll_step = last ? 5 : 4;
        // GPU producers of this step: RS packs (c0/c2), AG packs (c1/c3, c4 on the last layer)
        for (int r = 0; r < NR; ++r)
        {
            for (int b = 0; b < 2; ++b)
            {
                pack_rs(fab[r], step, 2 * b, d.part[r][b], M);
                pack_ag(fab[r], step, 2 * b + 1, d.norm[r][b], M);
            }
            if (last) pack_ag(fab[r], step, sp_c_resid_ag, d.resid[r], M);
        }
        // the delayed rank's wires of this step are held back one round
        if (i == 0 && delayed_rank >= 0)
        {
            for (int r = 0; r < NR; ++r)
                if (r != delayed_rank)
                    for (int c = 0; c < ncoll_step; ++c) wire(fab, step, c, r, sp_sizes(M, r, NR, H, last), rs_bytes, poison);
            // the other ranks stall: the delayed rank's RS tails are missing -> k_wait_rs would time out
            CHECK(wait_rs(fab, step, sp_c_attn_rs, 0, M, /*present*/ false) != 0);
            *saw_stall = true;
            for (int c = 0; c < ncoll_step; ++c)
                wire(fab, step, c, delayed_rank, sp_sizes(M, delayed_rank, NR, H, last), rs_bytes, poison);
        }
        else
            for (int r = 0; r < NR; ++r)
                for (int c = 0; c < ncoll_step; ++c)   // proxy order: c0 -> c4, single thread
                    wire(fab, step, c, r, sp_sizes(M, r, NR, H, last), rs_bytes, poison);
        // consumers of both boundaries, every rank
        for (int b = 0; b < 2; ++b)
        {
            const int cr = 2 * b, ca = cr + 1;
            for (int r = 0; r < NR; ++r)
            {
                const int first = pf_r0(M, r, NR), rows = pf_rows(M, r, NR);
                CHECK_EQ(wait_rs(fab, step, cr, r, M, true), 0);
                // owner reads every sender's slice at the owner-local index (MhcRsBlock): compare all bytes
                for (int p = 0; p < NR; ++p)
                {
                    const unsigned char* s = fab[r].rs_slice(step, cr, p);
                    for (int j = 0; j < rows; ++j)
                        CHECK(!memcmp(s + (size_t)j * RB, &d.part[p][b][(size_t)(first + j) * H], RB));
                }
                last_read[{step, cr}] = tick_counter;
                // AG: wait every peer's tail at pf_rows(M, p) * RB4, then the copy lands rows at pf_r0(M, p)
                const u64 aseq = g6::coll_seq(step, ca, NCOLL);
                std::vector<unsigned> gathered((size_t)M * H, POISON_A);
                memcpy(&gathered[(size_t)first * H], &d.norm[r][b][(size_t)first * H], (size_t)rows * RB4);   // owner keeps its rows
                for (int p = 0; p < NR; ++p)
                {
                    if (p == r) continue;
                    const unsigned char* slot = fab[r].recv_slot(step, ca, p);
                    CHECK_EQ(get_u64(slot + (size_t)pf_rows(M, p, NR) * RB4), aseq);
                    memcpy(&gathered[(size_t)pf_r0(M, p, NR) * H], slot, (size_t)pf_rows(M, p, NR) * RB4);   // k_ag_copy
                }
                for (int m2 = 0; m2 < M; ++m2)
                    CHECK(!memcmp(&gathered[(size_t)m2 * H], &d.norm[pf_owner_of(M, m2, NR)][b][(size_t)m2 * H], RB4));
                last_read[{step, ca}] = tick_counter;
                ++tick_counter;
            }
        }
        if (last)   // c4: every rank gathers the final FP32 residual rows
        {
            const u64 rseq = g6::coll_seq(step, sp_c_resid_ag, NCOLL);
            for (int r = 0; r < NR; ++r)
            {
                const int first = pf_r0(M, r, NR), rows = pf_rows(M, r, NR);
                std::vector<unsigned> gathered((size_t)M * H, POISON_A);
                memcpy(&gathered[(size_t)first * H], &d.resid[r][(size_t)first * H], (size_t)rows * RB4);
                for (int p = 0; p < NR; ++p)
                {
                    if (p == r) continue;
                    const unsigned char* slot = fab[r].recv_slot(step, sp_c_resid_ag, p);
                    CHECK_EQ(get_u64(slot + (size_t)pf_rows(M, p, NR) * RB4), rseq);
                    memcpy(&gathered[(size_t)pf_r0(M, p, NR) * H], slot, (size_t)pf_rows(M, p, NR) * RB4);
                }
                for (int m2 = 0; m2 < M; ++m2)
                    CHECK(!memcmp(&gathered[(size_t)m2 * H], &d.resid[pf_owner_of(M, m2, NR)][(size_t)m2 * H], RB4));
                last_read[{step, sp_c_resid_ag}] = tick_counter;
                ++tick_counter;
            }
        }
        // weak ordering sanity only (reads of step - 2 recorded before this step's end tick); the real two-parity
        // hazard gate with early-overwrite detection is test_flash_sp_sched.cpp
        for (int c = 0; c < ncoll_step; ++c)
        {
            const auto it = last_read.find({step - 2, c});
            if (it != last_read.end()) CHECK(it->second < tick_counter);
        }
    }
}

void owners_cover(int M)
{
    std::vector<int> seen(M, -1);
    int prev_end = 0;
    for (int p = 0; p < NR; ++p)
    {
        const int r0 = pf_r0(M, p, NR), r1 = pf_r0(M, p + 1, NR);
        CHECK_EQ(r0, prev_end);              // disjoint and covering, in order
        for (int m = r0; m < r1; ++m) { CHECK(seen[m] == -1); seen[m] = pf_owner_of(M, m, NR); }
        prev_end = r1;
    }
    CHECK_EQ(prev_end, M);
    int zeros = 0;
    for (int p = 0; p < NR; ++p) zeros += pf_rows(M, p, NR) == 0 ? 1 : 0;
    if (M < NR) CHECK_EQ(zeros, NR - M);     // zero-row owners only when M < 4 (M = 1, 2, 3)
    else CHECK_EQ(zeros, 0);
}

void spec_geometry_4096()
{
    // the pinned MP8192 / HID4096 numbers of the build spec (payload math this code shares with the kernels)
    const size_t MiB = 1ull << 20;
    const size_t rb = (size_t)4096 * 2;
    CHECK_EQ(pf_ss_min_rb(8192, NR, rb), 16 * MiB + 256);
    CHECK_EQ(pf_slot_bytes_rb(8192, true, NR, rb), 64 * MiB + 1024);
    const size_t stride_sp = (pf_slot_bytes_rb(8192, true, NR, rb) + 8 + 255) & ~size_t(255);
    CHECK_EQ(stride_sp, 64 * MiB + 1280);
    const size_t stride_off = (pf_slot_bytes_rb(8192, false, NR, rb) + 8 + 255) & ~size_t(255);
    CHECK_EQ(stride_off, 64 * MiB + 256);   // "the current OFF stride"
    CHECK_EQ(2 * 5 * (1 + 4) * stride_sp, (size_t)3355507200ull);   // SP mapped formula (spec table adds region rounding)
    CHECK_EQ(2 * 2 * (1 + 4) * stride_off, (size_t)1342182400ull);  // OFF mapped formula
    CHECK_EQ(2 * 5 * (1 + 4) * stride_sp - 2 * 2 * (1 + 4) * stride_off, (size_t)2013324800ull);
}
}   // namespace

int main()
{
    // 1) owner intervals: disjoint, covering, zero-row owners admitted (M = 1, 2, 3), uneven tails exact
    for (int M : {1, 2, 3, 4, 5, 63, 64, 8191, 8192}) owners_cover(M);
    // 2) the pinned spec geometry at MP8192 / HID4096
    spec_geometry_4096();
    // 3) full byte gate, both lane fabrics independent (distinct poisons and step bases), no delayed rank
    for (int M : {1, 2, 3, 4, 5, 63, 64, 8191, 8192})
    {
        bool stall = false;
        run_lane(M, 101, 3, POISON_A, -1, &stall);   // lane A steps 101..103
        run_lane(M, 5001, 3, POISON_B, -1, &stall);  // lane B steps 5001..5003: separate fabric, same M
        CHECK(!stall);
    }
    // 4) two-parity reuse with one delayed rank: the stall is observed, then everything completes byte-exact
    for (int M : {1, 5, 64, 8192})
    {
        bool stall = false;
        run_lane(M, 7, 4, POISON_A, 3, &stall);
        CHECK(stall);
    }
    if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
    printf("sp_layout_test: OK\n");
    return 0;
}
