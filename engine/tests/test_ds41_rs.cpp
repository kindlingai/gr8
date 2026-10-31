// F10 DS41 RS+AG (GR8_DS_PF_RS=1, runtime/ds41_rs.h): host test, no GPU.
// 1. rules: the owned rows tile [0, M) for every M and NR; every row's publish offset lies inside its owner's slice, no overlap, and
//    the slice + tail fits the slice stride.
// 2. job(): for the DS41 chunk shapes (40 layers, Engram on / off, bounded replay, late skip) the RS set is 0..K-1, c_ag = NC + c sits
//    right after c in the order, the AG sizes are one slice, the order is valid for Fabric::enqueue_step (fabric_order.h order_error).
// 3. bit equality: a CPU model of the whole data path (publish -> proxy RS slices -> reduce -> proxy AG -> gather), with the same index
//    helpers the kernels use, against the all-reduce (sum p = 0..NR-1 in FP32 from 0.f, rn16), random BF16 partials, M % NR != 0.
// 4. no deadlock: NR GPUs and NR proxies stepped round-robin over one RS chunk (Engram on and off); every collective completes.
#include "runtime/ds41_rs.h"
#include "runtime/s2/ds41_prefill.h"
#include "runtime/ds41_pf_split.h"
#include "runtime/ds41_prefill_fabric.h"
#include "fabric_order.h"
#include "fabric.h"            // F10 #8: fab_rs_slice_off / fab_rs_tail_off (the proxy's RS offsets)
#include <cstdio>
#include <cstring>
#include <random>
#include <set>

static int fails = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++fails; fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); } } while (0)
namespace R = ds41::rs;

static float bf(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t rn16(float f) {   // __float2bfloat16_rn for finite values
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static void rules() {
    for (int nr : {2, 3, 4, 6})
        for (int rows = nr; rows <= 2100; rows += (rows < 80 ? 1 : 97)) {
            const int q = R::quarter(rows, nr);
            int sum = 0;
            for (int p = 0; p < nr; ++p) { CHECK(R::owned(rows, nr, p) >= 0 && R::owned(rows, nr, p) <= q); sum += R::owned(rows, nr, p); }
            CHECK(sum == rows && R::owned(rows, nr, 0) == q);
            const size_t rb = 5120 * 2, ss = R::slice_stride(q, rb);
            CHECK(ss % 256 == 0 && (size_t)q * rb + 8 <= ss);
            for (int m = 0; m < rows; ++m) {
                const int o = m / q;
                CHECK(o < nr && m >= o * q && m < o * q + R::owned(rows, nr, o));
                const size_t off = R::pub_off(m, q, ss, rb);
                CHECK(off >= (size_t)o * ss && off + rb <= (size_t)o * ss + (size_t)q * rb);
                if (m) CHECK(off >= R::pub_off(m - 1, q, ss, rb) + rb);
            }
        }
    CHECK(R::slice_stride(512, 5120 * 2) == 5243136);   // W2048 HID 5120: F7-SCOPE item 4
    CHECK(R::on(true, 0, 42, 2048, 48, 4) && R::on(true, 41, 42, 49, 48, 4) && !R::on(true, 42, 42, 2048, 48, 4));
    CHECK(!R::on(true, 0, 42, 48, 48, 4) && !R::on(false, 0, 42, 2048, 48, 4) && !R::on(true, -1, 42, 2048, 48, 4));
}

static std::vector<int> iota_ids(int n) { std::vector<int> v; for (int i = 0; i < n; ++i) v.push_back(i); return v; }

// the chunk shapes g6m4.cu enqueues, with RS on
static void jobs() {
    const std::vector<int> ids = iota_ids(40);
    const int il = ds41::br::late_start(ids);
    CHECK(il == 21);
    for (int ecoll : {0, 2})
        for (bool skip : {false, true})
            for (int rows : {49, 1000, 2047, 2048}) {
                const int nc = 2 * 40 + 1 + ecoll, k = 2 * il;
                const auto cc = g6s2::ds41_chunk_collectives(ids, ecoll, (uint32_t)rows, 5120 * 2, 6144 * 2, il, (uint32_t)std::min(rows, 128), skip);
                CHECK(R::job_error(cc.sizes, nc, k, rows, 4, 5120 * 2) == nullptr);
                const R::Job j = R::job(cc.sizes, cc.order, nc, k, rows, 4, 5120 * 2);
                CHECK((int)j.rs.size() == k);
                for (int c = 0; c < k; ++c) CHECK(j.rs[(size_t)c] == c);
                CHECK(j.sizes.size() == (size_t)(nc + k) && g6::order_error(j.order, j.sizes.size()) == nullptr);
                CHECK(nc + k <= (int)g6::CollMask::kBits);
                const uint32_t qb = (uint32_t)(R::quarter(rows, 4) * 5120 * 2);
                for (int c = 0; c < k; ++c) CHECK(j.sizes[(size_t)(nc + c)] == qb && j.bytes[(size_t)(c % 4)] == qb);
                for (size_t x = 0; x < j.order.size(); ++x) {
                    const int c = j.order[x];
                    if (c < k) CHECK(x + 1 < j.order.size() && j.order[x + 1] == nc + c);
                    if (c >= nc) CHECK(x > 0 && j.order[x - 1] == c - nc);
                }
                // the order without the AG entries is the chunk's own order (or 0..n-1)
                std::vector<int> base;
                for (int c : j.order) if (c < nc) base.push_back(c);
                std::vector<int> want = cc.order;
                if (want.empty()) for (int c = 0; c < (int)cc.sizes.size(); ++c) want.push_back(c);
                CHECK(base == want);
            }
    std::vector<uint32_t> bad(83, 2048u * 10240u);
    CHECK(R::job_error(bad, 83, 42, 3, 4, 10240) != nullptr);              // rows < nr
    bad[5] = 128 * 10240; CHECK(R::job_error(bad, 83, 42, 2048, 4, 10240) != nullptr);   // an RS c of other rows
    CHECK(R::job_error(std::vector<uint32_t>(10, 0), 83, 42, 2048, 4, 10240) != nullptr);  // fewer than k
}

// 3. the data path, NR ranks, one RS collective
// flat (F10 #8 GR8_DS_PF_DIRECT): the producer writes [rows][H] at 0 of the send slot; the proxy sends slice p from fab_rs_slice_off(Q rb, p)
// and its tail word from fab_rs_tail_off (after the NR slices), written into the send slot first (a misplaced tail overwrites a row);
// k_rs_reduce reads its own slice at me Q rb. Not flat: the RS publish layout, tail word at sp + nb (the slice gap).
static void bitequal() {
    std::mt19937 rng(7);
    for (bool flat : {false, true})
    for (int nr : {4, 2, 3})
        for (int rows : {5, 7, 49, 130, 257, 1001}) {
            if (rows < nr) continue;
            const int H = 64;   // columns (the kernels: HID 5120; the index math is per row)
            const size_t rb = (size_t)H * 2;
            const int q = R::quarter(rows, nr);
            const size_t ss = R::slice_stride(q, rb), slot = (size_t)nr * ss, agslot = ss;
            const size_t xs = flat ? (size_t)q * rb : ss;   // the proxy's Rs::ss: the flat slice payload, or the RS slice stride
            CHECK(!flat || (size_t)nr * xs + 8 * (size_t)nr <= slot);
            std::vector<std::vector<uint16_t>> part((size_t)nr, std::vector<uint16_t>((size_t)rows * H));
            for (auto& v : part) for (auto& x : v) {   // finite BF16, mixed scales
                const float f = std::ldexp(std::uniform_real_distribution<float>(-1.f, 1.f)(rng), (int)(rng() % 20) - 10);
                x = rn16(f);
            }
            // reference: the all-reduce consumer value rn16(sum p = 0..nr-1 from 0.f)
            std::vector<uint16_t> ref((size_t)rows * H);
            for (size_t k = 0; k < ref.size(); ++k) { float s = 0.f; for (int p = 0; p < nr; ++p) s += bf(part[(size_t)p][k]); ref[k] = rn16(s); }
            // fabric memory per rank: send slot of c, recv (c, p), send slot of c_ag, recv (c_ag, p); 0xEE = stale bytes
            std::vector<std::vector<uint8_t>> send((size_t)nr, std::vector<uint8_t>(slot, 0xEE)), sag((size_t)nr, std::vector<uint8_t>(agslot, 0xEE));
            std::vector<std::vector<std::vector<uint8_t>>> recv((size_t)nr), rag((size_t)nr);
            for (int r = 0; r < nr; ++r) { recv[(size_t)r].assign((size_t)nr, std::vector<uint8_t>(ss, 0xEE)); rag[(size_t)r].assign((size_t)nr, std::vector<uint8_t>(agslot, 0xEE)); }
            for (int r = 0; r < nr; ++r)   // 1. publish (k_put_publish_rs; flat: the producer's own [rows][H] store)
                for (int m = 0; m < rows; ++m) memcpy(&send[(size_t)r][flat ? (size_t)m * rb : R::pub_off(m, q, ss, rb)], &part[(size_t)r][(size_t)m * H], rb);
            const uint32_t nb = (uint32_t)((size_t)q * rb);
            for (int r = 0; r < nr; ++r)   // 2. proxy RS: the tail words into my send slot, then slice p + its tail -> p's recv (c, r) at 0
                for (int p = 0; p < nr; ++p) if (p != r) {
                    const uint64_t seq = 0x5EC0000000000000ull + (uint64_t)(r * nr + p);
                    const size_t so = g6::fab_rs_slice_off(xs, p), to = g6::fab_rs_tail_off(xs, p, nb, flat, nr);
                    CHECK(to + 8 <= slot);
                    memcpy(&send[(size_t)r][to], &seq, 8);
                    memcpy(recv[(size_t)p][(size_t)r].data(), &send[(size_t)r][so], nb);
                    memcpy(recv[(size_t)p][(size_t)r].data() + nb, &send[(size_t)r][to], 8);
                    uint64_t got; memcpy(&got, recv[(size_t)p][(size_t)r].data() + nb, 8); CHECK(got == seq);
                }
            std::vector<std::vector<uint16_t>> gsum((size_t)nr, std::vector<uint16_t>((size_t)rows * H, 0xDEAD));
            for (int me = 0; me < nr; ++me)   // 3. k_rs_reduce
                for (int l = 0; l < R::owned(rows, nr, me); ++l)
                    for (int h = 0; h < H; ++h) {
                        const size_t off = (size_t)l * rb + (size_t)h * 2;
                        float s = 0.f;
                        for (int p = 0; p < nr; ++p) {
                            uint16_t v;
                            memcpy(&v, p == me ? &send[(size_t)me][(size_t)me * xs + off] : &recv[(size_t)me][(size_t)p][off], 2);
                            s += bf(v);
                        }
                        const uint16_t o = rn16(s);
                        memcpy(&sag[(size_t)me][off], &o, 2);
                        gsum[(size_t)me][(size_t)(me * q + l) * H + h] = o;
                    }
            for (int r = 0; r < nr; ++r)   // 4. proxy AG: my Q rows -> every peer's recv (c_ag, r)
                for (int p = 0; p < nr; ++p) if (p != r) memcpy(rag[(size_t)p][(size_t)r].data(), sag[(size_t)r].data(), (size_t)q * rb);
            for (int me = 0; me < nr; ++me)   // 5. k_rs_gather
                for (int p = 0; p < nr; ++p) if (p != me)
                    for (int l = 0; l < R::owned(rows, nr, p); ++l) memcpy(&gsum[(size_t)me][(size_t)(p * q + l) * H], &rag[(size_t)me][(size_t)p][(size_t)l * rb], rb);
            for (int me = 0; me < nr; ++me) CHECK(gsum[(size_t)me] == ref);
        }
}

// 4. GPUs + proxies over one chunk's job: GPU r walks the chunk's base order (publish c; RS c: wait the peers' slices, ring c_ag,
// wait the peers' AG; else wait the peers' c); proxy r walks job.order (wait doorbell, post to the peers). Round-robin, one action each.
static void nodeadlock() {
    const std::vector<int> ids = iota_ids(40);
    for (int ecoll : {0, 2}) {
        const int nr = 4, rows = 2048, nc = 81 + ecoll, k = 42;
        const auto cc = g6s2::ds41_chunk_collectives(ids, ecoll, (uint32_t)rows, 5120 * 2, 6144 * 2, 21, 128, false);
        const R::Job j = R::job(cc.sizes, cc.order, nc, k, rows, nr, 5120 * 2);
        std::vector<int> prog;   // the GPU's publish order = the chunk's order (no AG entries)
        for (int c : j.order) if (c < nc) prog.push_back(c);
        const int n = nc + k;
        std::vector<std::vector<char>> db((size_t)nr, std::vector<char>((size_t)n, 0));                 // doorbell rung
        std::vector<std::vector<std::vector<char>>> got((size_t)nr, std::vector<std::vector<char>>((size_t)n, std::vector<char>((size_t)nr, 0)));
        std::vector<size_t> gpc((size_t)nr, 0), ppc((size_t)nr, 0);
        std::vector<int> gst((size_t)nr, 0);   // GPU sub-state of the current c: 0 publish, 1 wait c, 2 wait c_ag
        auto have = [&](int r, int c) { for (int p = 0; p < nr; ++p) if (p != r && !got[(size_t)r][(size_t)c][(size_t)p]) return false; return true; };
        bool moved = true;
        int rounds = 0;
        while (moved) {
            moved = false; ++rounds;
            for (int r = 0; r < nr; ++r) {
                if (gpc[(size_t)r] < prog.size()) {
                    const int c = prog[gpc[(size_t)r]];
                    const bool rsc = c < k;
                    int& s = gst[(size_t)r];
                    if (s == 0) { db[(size_t)r][(size_t)c] = 1; s = 1; moved = true; }
                    else if (s == 1 && have(r, c)) { if (rsc) { db[(size_t)r][(size_t)R::c_ag(nc, c)] = 1; s = 2; } else { s = 0; ++gpc[(size_t)r]; } moved = true; }
                    else if (s == 2 && have(r, R::c_ag(nc, c))) { s = 0; ++gpc[(size_t)r]; moved = true; }
                }
                if (ppc[(size_t)r] < j.order.size()) {
                    const int c = j.order[ppc[(size_t)r]];
                    if (db[(size_t)r][(size_t)c]) { for (int p = 0; p < nr; ++p) if (p != r) got[(size_t)p][(size_t)c][(size_t)r] = 1; ++ppc[(size_t)r]; moved = true; }
                }
            }
        }
        for (int r = 0; r < nr; ++r) CHECK(gpc[(size_t)r] == prog.size() && ppc[(size_t)r] == j.order.size());
        printf("no deadlock: Engram %s, %zu GPU collectives, %zu proxy posts per rank, %d rounds\n", ecoll ? "on" : "off", prog.size(), j.order.size(), rounds);
    }
}

// 5. ncoll accounting (F10 run 3 boot RED: build_ds41 checked the fabric against Model::ncoll() alone): for RS off / on x Engram off / on
//    x main / draft, the count g6m4.cu gives the main fabric passes build_ds41's check (fabric_ncoll_error), the old check (ag = 0)
//    refuses every RS-on main fabric, a draft never takes AG collectives, and the fabric slot layout g6m4.cu builds (run1 2 i_late,
//    run2 2 NL + 1, run3 NC under RS) is valid for that count (fab_slot_layout, as Fabric::init checks it) with the AG slots holding Q rows.
static void ncoll() {
    const std::vector<int> ids = iota_ids(40);
    const int il = ds41::br::late_start(ids), draft_ncoll = 11;   // ds41_dspark.h DRAFT_NCOLL = 2 N_NEXTN + B
    for (int rs : {0, 1})
        for (int ecoll : {0, 2})
            for (bool draft : {false, true}) {
                const int model = draft ? draft_ncoll : 2 * 40 + 1 + ecoll;   // Model::ncoll() (rank.cuh)
                const int nc = model, ag = (rs && !draft) ? 2 * il : 0;         // g6m4.cu: NC = 2 NL + 1 + NX; rs_k = 2 i_late on the main fabric only
                const int fab = R::main_ncoll(nc, ag);                           // g6m4.cu mc.ncoll
                CHECK(R::fabric_ncoll_error(fab, model, ag, draft) == nullptr);
                CHECK((R::fabric_ncoll_error(fab, model, 0, draft) == nullptr) == (ag == 0));   // the pre-fix check
                CHECK(R::fabric_ncoll_error(fab + 1, model, ag, draft) != nullptr && R::fabric_ncoll_error(fab - 1, model, ag, draft) != nullptr);
                if (draft) CHECK(R::fabric_ncoll_error(R::main_ncoll(nc, 2 * il), model, 2 * il, true) != nullptr);
                CHECK(fab <= (int)g6::CollMask::kBits);
                if (draft) continue;
                g6::FabricConfig cfg; g6::FabricDev d{};
                const int q = R::quarter(2048, 4);
                cfg.slot_bytes = std::max((size_t)2048 * 5120 * 2, rs ? (size_t)4 * R::slice_stride(q, 5120 * 2) - 8 : (size_t)0);
                cfg.run1 = 2 * il; cfg.run2 = 2 * 40 + 1; cfg.slot_bytes1 = (size_t)128 * 5120 * 2; cfg.slot_bytes2 = ecoll ? (size_t)2048 * 6144 * 2 : 0;
                if (rs) { cfg.run3 = nc; cfg.slot_bytes3 = (size_t)q * 5120 * 2; }
                CHECK(g6::fab_slot_layout(cfg, fab, &d));
                if (rs) for (int c = 0; c < ag; ++c) CHECK(g6::fab_slot_stride(d, R::c_ag(nc, c)) >= (size_t)q * 5120 * 2 + 8);
            }
}

// All-levers boot contract: the F6 fabric sizer and the exact build_ds41
// validator must agree with Engram ON, PF_RS ON and the B12 flag matrix.
// The old fabric count (without the state collective) is a RED mutant.
static void boot_all_levers() {
    const int nc=83, k=42, il=21;
    int valid=0, invalid=0, mutant_red=0;
    for (bool rowshard : {false,true}) for (bool y_mxfp8 : {false,true}) {
        if (y_mxfp8 && !rowshard) { ++invalid; continue; } // g6m4 knob parser rejects this pair
        auto cfg=ds41::pf::fabric_runs(4096,128,il,40,2,48,5120,4,48*2240,true,false,rowshard).cfg;
        cfg.pool=4;
        g6::FabricDev f{}; f.nr=4; f.compact=1;
        CHECK(g6::fab_slot_layout(cfg,cfg.ncoll,&f));
        CHECK(f.ncoll==R::main_ncoll(nc,k,rowshard));
        CHECK(R::fabric_ncoll_error(f.ncoll,nc,k,false,rowshard)==nullptr);
        const bool old_rejected=R::fabric_ncoll_error(R::main_ncoll(nc,k),nc,k,false,rowshard)!=nullptr;
        CHECK(old_rejected==rowshard);
        mutant_red+=old_rejected;
        ++valid;
    }
    CHECK(valid==3 && invalid==1 && mutant_red==2);
    printf("ALL_LEVERS_BOOT_CONTRACT valid=%d invalid_y_without_row=%d old_count_RED=%d\n",valid,invalid,mutant_red);
}

int main() {
    rules();
    jobs();
    ncoll();
    boot_all_levers();
    bitequal();
    nodeadlock();
    printf("test_ds41_rs: %d checks, %d failed -> %s\n", checks, fails, fails ? "FAIL" : "ALL PASS");
    return fails ? 1 : 0;
}
