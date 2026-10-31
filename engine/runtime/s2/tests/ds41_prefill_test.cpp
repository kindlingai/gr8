// Host test of the DS41 S2 prefill admission (s2/ds41_prefill.h and its wiring in sessions.h, dispatch.h, bootstrap.h). No CUDA, no GPU. Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -pthread -DG6_MODEL_DS41=1 -I engine/runtime -I engine/comms engine/runtime/s2/tests/ds41_prefill_test.cpp -o /tmp/ds41_prefill_test
//  a. which prompts prefill: the decode_prompt rule (prefill_min 0 = never, short, prompt logprobs, steering)
//  b. one W-row chunk: its fields, and every refusal (rows 0 / > W / reaches the tail, run too small, sequence 0, slot 8)
//  c. the attention cell of a chunk against a host copy of k_step_in's formula (rank.cuh)
//  d. the chunk's fabric step, Engram off and on: sizes, order, min_stride, order_error
//  e. the real proxy loop (g6::walk_step) against a GPU that rings the REAL non-final plan's collectives: {sizes, order} from the builder
//     pass, the decode-step sizes (c = 80 with no ringer) and the numeric Engram-on order stall where the first-light proxy stalled
//  f. the non-final plan: no STEP_IN / FINISH / head / STEP_OUT, c = 79 still published, check_nonfinal_plan agrees
//  g. Dispatch + Sessions + Driver: one main sequence id per chunk, the next MAIN step has sequence + 1, draft sequence separate, the
//     chunks cover rows 0 .. n - 2, the last token is the first MAIN row; prefill_min 0 and plp / steering requests never prefill
//  h. BootstrapConfig::agreement(): a rank with another prefill_min is refused; the default string is unchanged
#include "s2/driver.h"
#include "s2/sessions.h"
#include "s2/dispatch.h"
#include "s2/bootstrap.h"
#include "s2/ds41_prefill.h"
#include "ds41_dspark_prefill.h"
#include "ds41_pf_split.h"
#include "ds41_rs.h"
#include "fabric_order.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <thread>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
template <class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static std::vector<int> iota(int n) { std::vector<int> v(static_cast<size_t>(n), 0); for (int i = 0; i < n; ++i) v[size_t(i)] = i; return v; }
static std::vector<int> range(int a, int b) { std::vector<int> v; for (int c = a; c < b; ++c) v.push_back(c); return v; }
static Run run_of(uint32_t first_page, uint32_t pages) { Run r; r.first_page = first_page; r.pages = pages; r.table_version = 1; return r; }

// a. the decode_prompt rule
static void check_rule()
{
    CHECK(ds41_decode_prompt(0, 5000, -1, false) && ds41_decode_prompt(0, 1, -1, false) && ds41_decode_prompt(0, 0, -1, false));   // default: every prompt decodes
    CHECK(ds41_decode_prompt(64, 63, -1, false));      // below prefill_min
    CHECK(!ds41_decode_prompt(64, 64, -1, false));     // at prefill_min: prefills
    CHECK(!ds41_decode_prompt(64, 5000, -1, false));
    CHECK(ds41_decode_prompt(64, 5000, 0, false) && ds41_decode_prompt(64, 5000, 20, false));   // prompt logprobs (k = 0 too) stay on the decode path
    CHECK(ds41_decode_prompt(64, 5000, -1, true));     // steering scales MAIN rows
    CHECK(ds41_decode_prompt(64, 5000, 3, true));
}

// b. one chunk
static void check_chunk()
{
    const std::vector<int> prompt = iota(3000);
    const Run run = run_of(7, 48);                     // 48 pages = 3072 rows
    const Ds41PrefillChunk c = ds41_make_chunk(prompt, 1024, 1024, 1024, 5, 9, 0, run);
    CHECK(c.tok == prompt.data() + 1024 && c.tok[0] == 1024 && c.tok[1023] == 2047);
    CHECK(c.first == 1024 && c.rows == 1024 && c.context == 2048 && c.sequence == 5 && c.request == 9 && c.slot == 0);
    CHECK(c.first_page == 7 && c.pages == 48);
    const Ds41PrefillChunk t = ds41_make_chunk(prompt, 2048, 951, 1024, 6, 9, 0, run);   // the last chunk of a 3000-row prompt: rows 2048 .. 2998
    CHECK(t.context == 2999 && t.tok[950] == 2998);
    const Ds41PrefillChunk one = ds41_make_chunk(prompt, 0, 1, 1024, 1, 9, 7, run);      // a 1-row chunk, the last slot
    CHECK(one.rows == 1 && one.context == 1 && one.slot == 7);
    CHECK(throws([&] { ds41_make_chunk(prompt, 0, 0, 1024, 1, 9, 0, run); }));            // rows 0
    CHECK(throws([&] { ds41_make_chunk(prompt, 0, 1025, 1024, 1, 9, 0, run); }));         // rows > W
    CHECK(!throws([&] { ds41_make_chunk(prompt, 0, 1024, 1024, 1, 9, 0, run); }));
    CHECK(throws([&] { ds41_make_chunk(prompt, 2048, 952, 1024, 1, 9, 0, run); }));       // first + rows == n: the last token is a MAIN row
    CHECK(throws([&] { ds41_make_chunk(prompt, 2048, 953, 1024, 1, 9, 0, run); }));       // past the prompt
    CHECK(!throws([&] { ds41_make_chunk(prompt, 1024, 1024, 1024, 1, 9, 0, run_of(0, 32)); }));   // ends exactly at the run's last row (2048)
    CHECK(throws([&] { ds41_make_chunk(prompt, 1025, 1024, 1024, 1, 9, 0, run_of(0, 32)); }));    // one row past the run
    CHECK(throws([&] { ds41_make_chunk(prompt, 0, 1024, 1024, 0, 9, 0, run); }));         // sequence 0 is the fabric's "no step"
    CHECK(throws([&] { ds41_make_chunk(prompt, 0, 1024, 1024, 1, 9, MAX_SESSIONS, run); })); // slot MAX_SESSIONS of MAX_SESSIONS
}

// r2-d4 ring crash: CAND serves SESS=32, so the native passes slots=32 and slot ids reach 31.
static void check_ragged_pack_s32()
{
    std::vector<std::vector<int>> prompts; std::vector<Ds41PrefillChunk> chunks;
    prompts.reserve(9);   // chunks keep prompt.data() pointers
    for (int i = 0; i < 8; ++i) prompts.push_back(iota(39));
    const unsigned slot_ids[8] = {3, 9, 12, 17, 22, 26, 30, 31};
    for (int i = 0; i < 8; ++i)
        chunks.push_back(ds41_make_chunk(prompts[i], 0, 38, 4096, 21, uint64_t(9 + i), slot_ids[i], run_of(0, 1)));
    const Ds41PrefillPack p = ds41_make_pack(chunks, 4096, 32);
    CHECK(p.layout.count == 8 && p.layout.rows == 304 && p.layout.seg[7].slot == 31);
    CHECK(ds41::pack::error(p.layout, 4096, 32) == nullptr);
    CHECK(throws([&] { ds41_make_pack(chunks, 4096, 31); }));                     // slot 31 of 31 slots
    CHECK(throws([&] { auto bad = chunks; bad[1].slot = 3; ds41_make_pack(bad, 4096, 32); }));   // one slot twice
    {   // a ninth member is refused before any layout.seg write (seg holds 8)
        auto nine = chunks; prompts.push_back(iota(39));
        nine.push_back(ds41_make_chunk(prompts.back(), 0, 38, 4096, 21, 17, 5, run_of(0, 1)));
        CHECK(throws([&] { ds41_make_pack(nine, 4096, 32); }));
    }
    ds41::pack::Layout l = p.layout;
    CHECK(ds41::pack::error(l, 4096, 33) != nullptr);                              // more slots than S2 has
    l.count = 9; CHECK(ds41::pack::error(l, 4096, 32) != nullptr);
}

static void check_ragged_pack()
{
    const int lens[8] = {56,58,59,59,59,61,62,65};
    std::vector<std::vector<int>> prompts; prompts.reserve(8);
    std::vector<Ds41PrefillChunk> chunks; chunks.reserve(8);
    for (int i = 0; i < 8; ++i) {
        prompts.push_back(iota(lens[i]));
        chunks.push_back(ds41_make_chunk(prompts.back(), 0, uint32_t(lens[i] - 1), 4096, 17, uint64_t(i + 1),
                                         unsigned(i), run_of(0, 2)));
    }
    const Ds41PrefillPack p = ds41_make_pack(chunks, 4096, 8);
    CHECK(p.sequence == 17 && p.layout.count == 8 && p.layout.rows == 471);
    CHECK(ds41::pack::error(p.layout, 4096, 8) == nullptr);
    // Native ordinals start at 0. The first segment may be request 0, but
    // later segments must still be strictly ascending (including vs 0).
    {
        auto zero = chunks; zero[0].request = 0;
        const auto first_zero = ds41_make_pack(zero, 4096, 8);
        CHECK(first_zero.layout.seg[0].request == 0 && ds41::pack::error(first_zero.layout, 4096, 8) == nullptr);
        CHECK(throws([&] { auto bad = zero; bad[1].request = 0; ds41_make_pack(bad, 4096, 8); }));
        CHECK(throws([&] { auto bad = zero; bad[2].request = 1; ds41_make_pack(bad, 4096, 8); }));
    }
    int end = 0;
    for (int i = 0; i < 8; ++i) {
        const auto& s = p.layout.seg[i];
        CHECK(s.row0 == end && s.rows == lens[i] - 1 && s.slot == i && s.context == s.rows);
        CHECK(p.chunk[i].tok == prompts[i].data());
        end += s.rows;
    }
    CHECK(end == 471);
    int max_context = 0;
    for (int i = 0; i < p.layout.count; ++i) {
        const auto& s = p.layout.seg[i];
        max_context = std::max(max_context, s.context); // idx_set_chunk scratch bound, per-row klen remains in m.pos
        CHECK(s.first == 0 && ds41::br::late_rows(s.rows, 128) == s.rows &&
              ds41::br::win_floor(s.context, s.rows, 128) == 0); // all segment rows run late layers
    }
    CHECK(max_context == 64 && ds41::br::late_rows(p.layout.rows, 128) == 128); // total-row late() would be wrong
    const int nr = 4, q = ds41::rs::quarter(p.layout.rows, nr);
    const size_t rb = 5120 * 2, ss = ds41::rs::slice_stride(q, rb);
    std::set<size_t> offsets;
    for (int i = 0; i < p.layout.count; ++i)
        for (int r = 0; r < p.layout.seg[i].rows; ++r) {
            const int flat = p.layout.seg[i].row0 + r;
            const size_t off = ds41::rs::pub_off(flat, q, ss, rb);
            CHECK(off < size_t(nr) * ss && offsets.insert(off).second);
            CHECK(size_t(flat / q) * size_t(q) * rb + size_t(flat % q) * rb == size_t(flat) * rb); // direct flat slice
        }
    CHECK(offsets.size() == size_t(p.layout.rows));
    CHECK(ds41::pack::attn_error(p.layout, 471, 4096, 8) == nullptr);
    CHECK(ds41::pack::index_error(p.layout, 471, 4096, 64, 8) == nullptr);
    CHECK(ds41::pack::dspark_error(p.layout, 471, 4096, 8, true, 0, 471) == nullptr);
    CHECK(ds41::pack::attn_error(p.layout, 470, 4096, 8) != nullptr);
    CHECK(ds41::pack::index_error(p.layout, 471, 4096, 48, 8) != nullptr); // 55..64 rows exceed old MX scratch
    CHECK(ds41::pack::dspark_error(p.layout, 471, 4096, 8, false, 0, 471) != nullptr);
    CHECK(ds41::pack::dspark_error(p.layout, 471, 4096, 8, true, 1, 471) != nullptr);
    // The two hard edges: eight complete 64-row chunks fill the 512-row pack,
    // while two one-row chunks still have two distinct slot/ring owners.
    {
        std::vector<std::vector<int>> edge_prompts(8, iota(65));
        auto edge = chunks;
        for (int i = 0; i < 8; ++i)
            edge[i] = ds41_make_chunk(edge_prompts[i], 0, 64, 4096, 17, uint64_t(i + 1), unsigned(i), run_of(0, 2));
        const auto full = ds41_make_pack(edge, 4096, 8);
        CHECK(full.layout.count == 8 && full.layout.rows == 512);
        CHECK(ds41::pack::attn_error(full.layout, 512, 4096, 8) == nullptr &&
              ds41::pack::index_error(full.layout, 512, 4096, 64, 8) == nullptr &&
              ds41::pack::dspark_error(full.layout, 512, 4096, 8, true, 0, 512) == nullptr);
        edge.resize(2);
        for (auto& c : edge) { c.rows = 1; c.context = 1; c.prompt_rows = 2; }
        const auto tiny = ds41_make_pack(edge, 4096, 8);
        CHECK(tiny.layout.count == 2 && tiny.layout.rows == 2 && tiny.layout.seg[1].row0 == 1);
    }
    CHECK(throws([&] { ds41_make_pack({chunks[0]}, 4096, 8); }));
    CHECK(throws([&] { ds41_make_pack(chunks, 470, 8); }));
    CHECK(throws([&] { auto bad = chunks; bad[7].slot = bad[0].slot; ds41_make_pack(bad, 4096, 8); }));
    CHECK(throws([&] { auto bad = chunks; bad[7].request = 1; ds41_make_pack(bad, 4096, 8); }));
    CHECK(throws([&] { auto bad = chunks; bad[7].first = 1; ds41_make_pack(bad, 4096, 8); }));
    CHECK(throws([&] { auto bad = chunks; bad[7].rows = 65; ds41_make_pack(bad, 4096, 8); }));
    CHECK(throws([&] { auto bad = chunks; bad[7].rows = 129; ds41_make_pack(bad, 4096, 8); }));
    CHECK(throws([&] { auto bad = chunks; bad[7].sequence = 18; ds41_make_pack(bad, 4096, 8); }));
    auto bad = p.layout; bad.seg[3].row0 += 1; // missing/overlap
    CHECK(ds41::pack::attn_error(bad, 471, 4096, 8) && ds41::pack::index_error(bad, 471, 4096, 64, 8) &&
          ds41::pack::dspark_error(bad, 471, 4096, 8, true, 0, 471));
    bad = p.layout; bad.seg[7].rows += 1; // mismatched total
    CHECK(ds41::pack::attn_error(bad, 471, 4096, 8) && ds41::pack::index_error(bad, 471, 4096, 64, 8) &&
          ds41::pack::dspark_error(bad, 471, 4096, 8, true, 0, 471));
    bad = p.layout; bad.seg[7].slot = 8; // out-of-range slot
    CHECK(ds41::pack::attn_error(bad, 471, 4096, 8) && ds41::pack::index_error(bad, 471, 4096, 64, 8) &&
          ds41::pack::dspark_error(bad, 471, 4096, 8, true, 0, 471));
    bad = p.layout; bad.seg[7].context += 1; // wrong per-segment key bound
    CHECK(ds41::pack::attn_error(bad, 471, 4096, 8) && ds41::pack::index_error(bad, 471, 4096, 64, 8) &&
          ds41::pack::dspark_error(bad, 471, 4096, 8, true, 0, 471));
}

// c. the cell: k_step_in (rank.cuh) writes, per layer, context = sh.context, page_count = page_bound ? page_bound : pages (DEC_S2; DS41 serve
// never runs s2_init, page_bound 0), table_size = pages, reuse_rows = shared[l] ? sh.M : 0, with pages = (context + 63) / 64
struct Cell { int context, page_count, table_size, reuse_rows; };
static Cell k_step_in_cell(int context, int M, bool shared, int page_bound = 0)
{
    const int pages = (context + 63) / 64;
    return {context, page_bound ? page_bound : pages, pages, shared ? M : 0};
}
static void check_cell()
{
    const std::vector<int> prompt = iota(5000);
    const Run run = run_of(0, 79);                     // 79 pages = 5056 rows
    long n = 0;
    for (uint32_t first : {0u, 64u, 1000u, 1024u, 2048u, 4000u})
        for (uint32_t rows : {1u, 31u, 32u, 33u, 64u, 65u, 1024u})
        {
            if (uint64_t(first) + rows >= prompt.size()) continue;
            const Ds41PrefillChunk c = ds41_make_chunk(prompt, first, rows, 1024, 3, 1, 0, run);
            for (bool shared : {false, true})
            {
                const Ds41PrefillCell a = ds41_chunk_cell(c, shared);
                const Cell b = k_step_in_cell(int(first + rows), int(rows), shared);
                CHECK(a.context == b.context && a.page_count == b.page_count && a.table_size == b.table_size && a.reuse_rows == b.reuse_rows);
                CHECK(a.page_count == a.table_size && a.table_size >= 1 && uint32_t(a.table_size) <= run.pages);
                ++n;
            }
        }
    CHECK(n == 82);                                   // 6 x 7 (first, rows) pairs, 41 of which fit the 5000-row prompt, shared and not
    CHECK(ds41_chunk_cell(ds41_make_chunk(prompt, 1024, 1024, 1024, 3, 1, 0, run), true).reuse_rows == 1024);
    CHECK(ds41_chunk_cell(ds41_make_chunk(prompt, 1024, 1024, 1024, 3, 1, 0, run), false).reuse_rows == 0);
    CHECK(ds41_chunk_cell(ds41_make_chunk(prompt, 1024, 1024, 1024, 3, 1, 0, run), true).table_size == 32);   // context 2048 = 32 pages exactly
    CHECK(ds41_chunk_cell(ds41_make_chunk(prompt, 1024, 1025, 1024 + 1, 3, 1, 0, run), true).table_size == 33);   // 2049: one more page
}

// d. the chunk's collectives
static void check_collectives()
{
    const std::vector<int> ids = iota(40);
    const size_t HID = 5120 * 2, EMB = 6144 * 2;
    for (uint32_t rows : {1u, 32u, 33u, 1024u, 2048u})
    {
        const Ds41ChunkColl off = ds41_chunk_collectives(ids, 0, rows, HID, EMB);
        CHECK(off.sizes.size() == 80 && off.order.empty());                      // c = 0..79; the head c = 80 is NOT sized; the plain enqueue_step
        for (uint32_t s : off.sizes) CHECK(s == rows * HID);
        CHECK(off.min_stride() == uint64_t(rows) * HID + 8);
        CHECK(g6::order_error(off.order, off.sizes.size()) == nullptr);
        const Ds41ChunkColl on = ds41_chunk_collectives(ids, 2, rows, HID, EMB);
        CHECK(on.sizes.size() == 83 && on.order.size() == 82);
        for (size_t c = 0; c < 80; ++c) CHECK(on.sizes[c] == rows * HID);
        CHECK(on.sizes[80] == 0 && on.sizes[81] == rows * EMB && on.sizes[82] == rows * EMB);
        CHECK(on.order == ds41::coll_exec_order(ids, 2, false));
        CHECK(g6::order_error(on.order, on.sizes.size()) == nullptr);
        CHECK(on.min_stride() == uint64_t(rows) * EMB + 8);                       // the Engram reduce is the widest: 12,288 B per row
    }
    // the order itself, written out: [0, 1, 81, 2 .. 27, 82, 28 .. 79], not the numeric 0 .. 79, 81, 82
    {
        std::vector<int> want = {0, 1, 81};
        for (int c = 2; c <= 27; ++c) want.push_back(c);
        want.push_back(82);
        for (int c = 28; c <= 79; ++c) want.push_back(c);
        const Ds41ChunkColl on = ds41_chunk_collectives(ids, 2, 1024, HID, EMB);
        CHECK(on.order == want);
        std::vector<int> numeric = range(0, 80); numeric.push_back(81); numeric.push_back(82);
        CHECK(on.order != numeric);
        CHECK(std::find(on.order.begin(), on.order.end(), 80) == on.order.end());
    }
    CHECK(ds41_chunk_collectives(ids, 2, 1024, HID, EMB).min_stride() == 12582920);   // B11: W1024 slot, 12,582,920 B
    CHECK(ds41_chunk_collectives(ids, 2, 2048, HID, EMB).min_stride() == 25165832);   //      W2048
    CHECK(ds41_chunk_collectives(ids, 0, 1024, HID, EMB).min_stride() == 10485768);   // Engram off, W1024
    // the fabric accepts sizes.size() <= ncoll (81 decode / 83 Engram) and sizes[c] + 8 <= stride
    CHECK(ds41_chunk_collectives(ids, 0, 1024, HID, EMB).sizes.size() <= 81 && ds41_chunk_collectives(ids, 2, 1024, HID, EMB).sizes.size() <= 83);
    // refusals
    CHECK(throws([&] { ds41_chunk_collectives({}, 0, 8, HID, EMB); }));
    CHECK(throws([&] { ds41_chunk_collectives(ids, 0, 0, HID, EMB); }));
    CHECK(throws([&] { ds41_chunk_collectives(ids, 1, 8, HID, EMB); }));              // Engram has 0 or 2 collectives
    CHECK(throws([&] { ds41_chunk_collectives(ids, 0, 8, 0, EMB); }));
    CHECK(throws([&] { ds41_chunk_collectives(ids, 0, 8, HID + 4, EMB); }));          // a size must be a multiple of 8
    CHECK(throws([&] { ds41_chunk_collectives(ids, 2, 1u << 20, HID, EMB); }));       // 10.7 GB: past 32 bits
}

// e. the real proxy loop against the device's ring order (comms/tests/fabric_order_test.cpp run_plan, the same threads)
struct Run2 {
    bool proxy_ok = false, device_ok = false;
    unsigned proxy_rc = 0; int proxy_c = -1, device_c = -1;
    std::vector<std::vector<int>> sent;
};
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Run2 walk(int ncoll, size_t nsizes, const std::vector<int>& program, const std::vector<int>& order, int steps, double timeout_s = 0.4)
{
    Run2 r;
    std::vector<std::atomic<unsigned long long>> db(static_cast<size_t>(ncoll)), done(static_cast<size_t>(ncoll));
    for (auto& a : db) a = 0;
    for (auto& a : done) a = 0;
    std::atomic<bool> halt{false};
    r.sent.resize(size_t(steps));
    std::thread dev([&] {
        for (int s = 0; s < steps; s++)
            for (int c : program)
            {
                const unsigned long long seq = g6::coll_seq(s, c, ncoll);
                db[size_t(c)].store(seq, std::memory_order_release);
                const double w0 = now_s();
                while (done[size_t(c)].load(std::memory_order_acquire) != seq)
                {
                    if (halt.load()) { r.device_c = c; return; }
                    if (now_s() - w0 > timeout_s) { r.device_c = c; halt = true; return; }
                    std::this_thread::yield();
                }
            }
        r.device_ok = true;
    });
    std::thread proxy([&] {
        for (int s = 0; s < steps; s++)
        {
            const g6::StepWalk w = g6::walk_step(s, ncoll, nsizes, order, 0,
                [&](int c) { return db[size_t(c)].load(std::memory_order_acquire); }, [&] { return halt.load(); }, now_s, timeout_s,
                [&](int c, unsigned long long seq) { r.sent[size_t(s)].push_back(c); done[size_t(c)].store(seq, std::memory_order_release); return true; });
            if (w.rc != g6::kWalkOk) { r.proxy_rc = w.rc; r.proxy_c = w.c; halt = true; return; }
        }
        r.proxy_ok = true;
    });
    dev.join(); proxy.join();
    return r;
}
// the collectives a non-final chunk plan rings, in plan order: ST_PUBLISH_ATTN i -> c = 2i, ST_PUBLISH_FFN i -> 2i + 1, ST_ENGRAM -> 2nl + 1 + e
static std::vector<int> program_of(const std::vector<ds41::Op>& plan, const std::vector<int>& ids)
{
    const int nl = int(ids.size());
    std::vector<int> v;
    for (const ds41::Op& o : plan)
        if (o.stage == ds41::ST_PUBLISH_ATTN) v.push_back(2 * o.i);
        else if (o.stage == ds41::ST_PUBLISH_FFN) v.push_back(2 * o.i + 1);
        else if (o.stage == ds41::ST_ENGRAM) v.push_back(2 * nl + 1 + (ids[size_t(o.i)] == geom::ENGRAM_L0 ? 0 : 1));
    return v;
}
static void check_walk()
{
    const std::vector<int> ids = iota(40);
    const size_t HID = 5120 * 2, EMB = 6144 * 2;
    const int steps = 3;
    for (const bool engram : {false, true})
    {
        const auto verify = ds41::plan(ids, 0, 40, true, true, engram, true);
        const auto nf = ds41::dsp::pf::nonfinal_plan(verify, true);
        const std::vector<int> program = program_of(nf, ids);
        const Ds41ChunkColl cc = ds41_chunk_collectives(ids, engram ? 2 : 0, 1024, HID, EMB);
        const int ncoll = engram ? 83 : 81;                                   // the decode step's fabric: 2nl + 1 (+ 2 Engram)
        // the builder's order IS the plan's ring order (Engram off: the identity, no order vector)
        CHECK(program.size() == (engram ? 82u : 80u));
        CHECK(cc.order.empty() ? program == range(0, 80) : program == cc.order);
        const Run2 ok = walk(ncoll, cc.sizes.size(), program, cc.order, steps);
        CHECK(ok.proxy_ok && ok.device_ok);
        for (int s = 0; s < steps && ok.proxy_ok; ++s) CHECK(ok.sent[size_t(s)] == program);
        if (!engram)
        {   // the decode-step sizes (81 entries, c = 80 sized) wait for the head's doorbell nobody rings
            const Run2 bad = walk(ncoll, 81, program, {}, steps);
            CHECK(!bad.proxy_ok && bad.proxy_rc == g6::kWalkTimeout && bad.proxy_c == 80);
        }
        else
        {   // the numeric order stalls: the proxy waits for c = 2 while the device waits for c = 81's send
            std::vector<int> numeric = range(0, 80); numeric.push_back(81); numeric.push_back(82);
            const Run2 bad = walk(ncoll, 83, program, numeric, steps);
            CHECK(!bad.proxy_ok && bad.proxy_rc == g6::kWalkTimeout && bad.proxy_c == 2 && bad.device_c == 81);
            const Run2 bad2 = walk(ncoll, 83, program, {}, steps);                  // no order at all, 83 sizes: the same stall
            CHECK(!bad2.proxy_ok && bad2.proxy_c == 2 && bad2.device_c == 81);
            const Run2 bad3 = walk(ncoll, 83, program, ds41::coll_exec_order(ids, 2, true), steps);   // the FULL-step order: waits for c = 80
            CHECK(!bad3.proxy_ok && bad3.proxy_c == 80);
        }
    }
}

// f. the non-final plan
static void check_plan()
{
    const std::vector<int> ids = iota(40);
    for (const bool engram : {false, true})
        for (const bool dsp_rows : {false, true})
        {
            const auto verify = ds41::plan(ids, 0, 40, true, true, engram, true);
            const auto nf = ds41::dsp::pf::nonfinal_plan(verify, dsp_rows);
            using ds41::dsp::pf::count_stage;
            CHECK(ds41::dsp::pf::check_nonfinal_plan(verify, nf, dsp_rows) == "");
            CHECK(count_stage(nf, ds41::ST_STEP_IN) == 0 && count_stage(nf, ds41::ST_FINISH) == 0 && count_stage(nf, ds41::ST_HEAD_LOCAL) == 0 &&
                  count_stage(nf, ds41::ST_HEAD_MERGE) == 0 && count_stage(nf, ds41::ST_STEP_OUT) == 0);
            CHECK(count_stage(verify, ds41::ST_STEP_IN) == 1 && count_stage(verify, ds41::ST_FINISH) == 1);       // the decode plan (the prompt tail) keeps all of them
            CHECK(nf[0].stage == ds41::ST_EMBED);
            CHECK(count_stage(nf, ds41::ST_PUBLISH_ATTN) == 40 && count_stage(nf, ds41::ST_PUBLISH_FFN) == 40);
            CHECK(count_stage(nf, ds41::ST_ENGRAM) == (engram ? 2 : 0));
            int last_ffn = -1;
            for (const ds41::Op& o : nf) if (o.stage == ds41::ST_PUBLISH_FFN) last_ffn = 2 * o.i + 1;
            CHECK(last_ffn == 79);                                                // c = 79 is published with no finish after it: the fabric tolerates it
            auto with = [&](ds41::Stage s, ds41::Unit u) { auto p = nf; ds41::Op o; o.stage = s; o.unit = u; p.insert(p.begin(), o); return p; };
            CHECK(ds41::dsp::pf::check_nonfinal_plan(verify, with(ds41::ST_STEP_IN, ds41::U_STEP_IN), dsp_rows) != "");   // a restored STEP_IN would overwrite the first MX rows
        }
}

// g. Sessions + Dispatch (+ Driver), as service.h wires them
static g6rt::ServeReq serve_req(size_t n, int plp = -1, bool steering = false, int n_gen = 12)
{
    g6rt::ServeReq r;
    for (size_t i = 0; i < n; ++i) r.prompt.push_back(int(i * 7 % 1000));
    r.n_gen = n_gen; r.plp = plp; r.steering.enabled = steering;
    return r;
}
static Config w_cfg(uint32_t W = 1024)
{
    Config c;
    c.epoch = 9; c.mtp = 0; c.max_rows = 24; c.max_sessions = 1; c.prefill_rows = W; c.two_lanes = false;
    return c;
}
static void check_payload()
{
    const Config c = w_cfg();
    const Sessions none(c, "", false, 4), pf(c, "", false, 4, 64);
    for (size_t n : {size_t(1), size_t(63), size_t(64), size_t(3000)})
    {
        CHECK(none.payload(serve_req(n)).decode_prompt);                          // prefill_min 0: every prompt decodes, as before
        CHECK(pf.payload(serve_req(n)).decode_prompt == (n < 64));
        CHECK(pf.payload(serve_req(n)).prompt == serve_req(n).prompt && pf.payload(serve_req(n)).chunk == 4);
    }
    CHECK(pf.payload(serve_req(3000, 5)).decode_prompt && pf.payload(serve_req(3000, 0)).decode_prompt);
    CHECK(pf.payload(serve_req(3000, -1, true)).decode_prompt);
    CHECK(none.payload(serve_req(3000, 5, true)).decode_prompt);
}

struct Fab {                                 // what the hooks saw, in call order: one fabric id per call
    struct Ev { char kind; uint64_t seq; uint32_t first, rows; int pos0; };
    std::vector<Ev> ev;
    std::vector<Ds41PrefillChunk> chunks;
    std::vector<std::vector<int>> chunk_tok;
};
static Dispatch make_dispatch(const Sessions& s, Fab& f, bool seq_hook, uint32_t W)
{
    DispatchHooks h;
    h.bucket = [](bool, int) { return 0; };
    h.mode = [](int, int) {};
    h.launch = [&f](const Launch& l) {
        f.ev.push_back({l.step.kind == s2::Step::MAIN ? 'M' : 'D', l.sequence, 0, uint32_t(l.step.M()), l.step.r[0].pos});
        return std::vector<int>(l.step.r.size(), 1);
    };
    if (seq_hook)
        h.prefill_seq = [&f, W](const Slot& s, const Payload& p, uint32_t first, uint32_t rows, uint64_t seq) {
            const Ds41PrefillChunk c = ds41_make_chunk(p.prompt, first, rows, W, seq, s.req.id, s.key.slot, s.run);   // what ds41_native.cuh does
            f.chunks.push_back(c); f.chunk_tok.emplace_back(c.tok, c.tok + c.rows);
            f.ev.push_back({'P', seq, first, rows, int(first)});
        };
    else
        h.prefill = [&f](const Slot&, const Payload&, uint32_t first, uint32_t rows) { f.ev.push_back({'p', 0, first, rows, int(first)}); };
    return Dispatch(s, 24, std::move(h));
}
static s2::Step main_step(int pos, int rows, s2::Step::Kind kind = s2::Step::MAIN)
{
    s2::Step s; s.kind = kind; s.sess = {0}; s.row0 = {0}; s.rows = {rows}; s.ctx = {pos + rows};
    for (int r = 0; r < rows; ++r) s.r.push_back({0, 5, pos + r, pos + r, 6});
    return s;
}
static void check_dispatch()
{
    const Config c = w_cfg();
    Sessions sessions(c, "", false, 4, 64);
    const g6rt::ServeReq req = serve_req(3000);
    const Request rq = Sessions::request(1, req);
    Grant g{{9, 0, 1}, rq, run_of(0, reserve_pages(c, rq)), 0};
    sessions.bind(g, req);
    Slot slot; slot.key = g.key; slot.req = g.req; slot.run = g.run;
    const Payload p = sessions.payload(req);
    CHECK(!p.decode_prompt);
    {   // seq hook: ids 1, 2 for two chunks; the next MAIN step is 3; a draft step has its own counter; the next chunk is 4, the next MAIN 5
        Fab f;
        Dispatch d = make_dispatch(sessions, f, true, 1024);
        d.begin(0, 0);
        d.prefill(slot, p, 0, 1024); d.prefill(slot, p, 1024, 1024);
        d.run(main_step(2048, 1));
        d.run(main_step(2048, 3, s2::Step::DRAFT));
        d.prefill(slot, p, 2048, 100);
        d.run(main_step(2148, 1));
        CHECK(f.ev.size() == 6);
        const char kinds[] = {'P', 'P', 'M', 'D', 'P', 'M'}; const uint64_t seqs[] = {1, 2, 3, 1, 4, 5};
        for (size_t i = 0; i < f.ev.size() && i < 6; ++i) CHECK(f.ev[i].kind == kinds[i] && f.ev[i].seq == seqs[i]);
        CHECK(f.chunks.size() == 3 && f.chunks[0].sequence == 1 && f.chunks[1].sequence == 2 && f.chunks[2].sequence == 4);
        CHECK(f.chunks[1].first == 1024 && f.chunks[1].rows == 1024 && f.chunks[1].context == 2048 && f.chunks[1].request == 1 && f.chunks[1].slot == 0);
        CHECK(f.chunks[1].first_page == g.run.first_page && f.chunks[1].pages == g.run.pages);
        CHECK(f.chunk_tok[1][0] == req.prompt[1024] && f.chunk_tok[1][1023] == req.prompt[2047] && f.chunk_tok[2][99] == req.prompt[2147]);
        CHECK(throws([&] { d.prefill(slot, p, 2048, 952); }));                    // reaches the prompt tail: Dispatch refuses before any hook
        CHECK(f.ev.size() == 6);
        Slot small = slot; small.run = run_of(0, 16);
        CHECK(throws([&] { d.prefill(small, p, 960, 128); }));                    // outside the run
        CHECK(f.ev.size() == 6);
    }
    {   // no seq hook: the old prefill hook runs, no main id is taken (the default build)
        Fab f;
        Dispatch d = make_dispatch(sessions, f, false, 1024);
        d.begin(0, 0);
        d.prefill(slot, p, 0, 1024);
        d.run(main_step(1024, 1));
        CHECK(f.ev.size() == 2 && f.ev[0].kind == 'p' && f.ev[1].kind == 'M' && f.ev[1].seq == 1);
    }
    {   // native.cuh (GLM / Flash) builds DispatchHooks POSITIONALLY: bucket, mode, launch, prefill, logprob. A member added before those would
        // shift them (nvcc caught prefill_seq placed after prefill); this is the same shape on the host, and prefill_seq stays empty
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"   // the trailing members are meant to be left out
        DispatchHooks h{
            [](bool, int) { return 0; }, [](int, int) {}, [](const Launch&) { return std::vector<int>(); },
            [](const Slot&, const Payload&, uint32_t, uint32_t) {},
            [](int, bool, int) { return g6rt::LpRow{}; }};
#pragma GCC diagnostic pop
        CHECK(!h.prefill_seq && bool(h.prefill) && bool(h.logprob));
    }
    {   // a Dispatch with neither hook is refused
        Fab f;
        DispatchHooks h; h.bucket = [](bool, int) { return 0; }; h.mode = [](int, int) {}; h.launch = [](const Launch&) { return std::vector<int>(); };
        CHECK(throws([&] { Dispatch d(sessions, 24, h); }));
    }
}

// the whole path: admission rule -> Driver quanta -> Dispatch -> the callee's chunk
struct Rank {
    Config c;
    Sessions sessions;
    Fab fab;
    Dispatch dispatch;
    std::map<uint64_t, g6rt::ServeReq> reqs;
    std::map<uint64_t, std::vector<int>> out;
    std::vector<bool> main_decode_work;
    std::unique_ptr<Driver> d;
    uint32_t next_page = 0;
    Rank(const Config& cfg, uint32_t prefill_min, uint32_t W, bool seq_hook = true)
        : c(cfg), sessions(cfg, "", false, 4, prefill_min), dispatch(make_dispatch(sessions, fab, seq_hook, W)) {
        DriverHooks h;
        h.bind = [this](const Grant& g) { sessions.bind(g, reqs.at(g.req.id)); dispatch.begin(g.key.slot, int(g.reuse)); };
        h.retire = [this](const Slot& s) { sessions.retire(s); dispatch.release(s.key.slot); };
        h.prefill = [this](const Slot& s, const Payload& p, uint32_t first, uint32_t rows) { dispatch.prefill(s, p, first, rows); };
        h.run = [this](const s2::Step& st) {
            if (st.kind == s2::Step::MAIN) main_decode_work.push_back(st.scheduler_decode);
            return dispatch.run(st);
        };
        h.emit = [this](Key k, const int* ids, int n, int) {
            const Slot& s = d->state().slots()[k.slot];
            auto& o = out[s.req.id];
            const int cnt = std::min<int>(n, int(s.req.max_tokens - o.size()));
            o.insert(o.end(), ids, ids + cnt);
            return Emission{uint32_t(cnt), o.size() == s.req.max_tokens};
        };
        d = std::make_unique<Driver>(c, h);
    }
    void admit(uint64_t id, const g6rt::ServeReq& r) {
        reqs[id] = r;
        const Request rq = Sessions::request(id, r);
        Commit cm;
        cm.epoch = c.epoch; cm.event = d->state().next_event(); cm.boundary = d->state().boundary(); cm.before_hash = d->state().digest();
        cm.admit = Grant{d->state().next_key(), rq, run_of(next_page, reserve_pages(c, rq)), 0};
        next_page += cm.admit->run.pages;
        const Payload p = sessions.payload(r);
        d->apply(cm, &p);
    }
    void drain() {
        for (long g = 0; d->state().resident(); ++g) {
            const Quantum q = d->step();
            CHECK(q.kind != Kind::Idle);
            if (q.kind == Kind::Idle || g > 100000) { CHECK(!"no progress"); fails += 1000; break; }
        }
    }
};
static void check_path()
{
    const uint32_t W = 1024;
    for (const size_t N : {size_t(3000), size_t(1025), size_t(65), size_t(64)})
    {   // prefill_min 64: n - 1 rows in chunks of at most W, then the last token as the first MAIN row
        Rank r(w_cfg(W), 64, W);
        r.admit(1, serve_req(N));
        r.drain();
        std::vector<uint32_t> want;
        for (uint32_t left = uint32_t(N) - 1; left;) { const uint32_t x = std::min(left, W); want.push_back(x); left -= x; }
        CHECK(r.fab.chunks.size() == want.size());
        uint32_t next = 0; uint64_t seq = 0;
        for (size_t i = 0; i < r.fab.chunks.size() && i < want.size(); ++i) {
            const Ds41PrefillChunk& k = r.fab.chunks[i];
            CHECK(k.first == next && k.rows == want[i] && k.sequence == ++seq && k.context == int(next + want[i]) && k.request == 1);
            CHECK(r.fab.chunk_tok[i] == std::vector<int>(r.reqs[1].prompt.begin() + next, r.reqs[1].prompt.begin() + next + want[i]));
            next += want[i];
        }
        CHECK(next == N - 1);
        // the fabric ids are one counter: chunks 1..k, then every MAIN step k + 1, k + 2, ... (no gap, no repeat), the first MAIN row is the tail
        uint64_t expect = 0; bool first_main = true;
        for (const auto& e : r.fab.ev) {
            if (e.kind == 'D') continue;
            CHECK(e.seq == ++expect);
            if (e.kind == 'M' && first_main) { CHECK(e.pos0 == int(N) - 1 && e.rows == 1 && e.seq == r.fab.chunks.size() + 1); first_main = false; }
        }
        CHECK(!first_main && r.out[1].size() == 12);
        CHECK(!r.main_decode_work.empty() && !r.main_decode_work.front()); // solo PromptTail uses the main cut
        CHECK(std::find(r.main_decode_work.begin(), r.main_decode_work.end(), true) != r.main_decode_work.end());
    }
    {   // the default (prefill_min 0): no chunk, the prompt runs as MAIN steps from row 0, the ids start at 1
        Rank r(w_cfg(W), 0, W);
        r.admit(1, serve_req(3000));
        r.drain();
        CHECK(r.fab.chunks.empty() && !r.fab.ev.empty() && r.fab.ev[0].kind == 'M' && r.fab.ev[0].pos0 == 0 && r.fab.ev[0].seq == 1);
        uint64_t expect = 0;
        for (const auto& e : r.fab.ev) CHECK(e.kind == 'M' && e.seq == ++expect);
        CHECK(!r.main_decode_work.empty() && !r.main_decode_work.front()); // decode_prompt still has a PromptTail quantum
        CHECK(std::find(r.main_decode_work.begin(), r.main_decode_work.end(), true) != r.main_decode_work.end());
    }
    for (const auto& q : {std::make_pair(5, false), std::make_pair(-1, true)})
    {   // prompt logprobs / steering: the decode path even with prefill_min 64
        Rank r(w_cfg(W), 64, W);
        r.admit(1, serve_req(3000, q.first, q.second));
        r.drain();
        CHECK(r.fab.chunks.empty() && r.fab.ev[0].kind == 'M' && r.fab.ev[0].pos0 == 0);
    }
}

// h. a rank with another prefill_min is refused at bootstrap
static void check_agreement()
{
    BootstrapConfig a, b, c;
    a.scheduler = b.scheduler = c.scheduler = w_cfg();
    a.runtime_signature = b.runtime_signature = c.runtime_signature = "sig";
    b.prefill_min = 64; c.prefill_min = 128;
    CHECK(a.agreement().find("pfmin") == std::string::npos);                      // default: the string is what it was
    CHECK(b.agreement().find(" pfmin 64 ctl ") != std::string::npos && c.agreement().find(" pfmin 128 ctl ") != std::string::npos);
    CHECK(a.agreement() != b.agreement() && b.agreement() != c.agreement() && a.agreement() != c.agreement());
    BootstrapConfig b2 = b;
    CHECK(b.agreement() == b2.agreement());
}

static void check_single_lane_config()
{
    Config prefill;
    CHECK(prefill.two_lanes && prefill.prefill_rows == 4096); // shared Config default would refuse the W callback
    ds41_prefill_single_lane(prefill, 1024);
    CHECK(!prefill.two_lanes && prefill.prefill_rows == 1024);
    Config decode_only;
    CHECK(decode_only.two_lanes && decode_only.prefill_rows == 4096); // no pf_min call: unchanged
    CHECK(throws([&] { ds41_prefill_single_lane(prefill, 0); }));
}

int main()
{
    check_single_lane_config();
    check_rule();
    check_chunk();
    check_ragged_pack();
    check_ragged_pack_s32();
    check_cell();
    check_collectives();
    check_walk();
    check_plan();
    check_payload();
    check_dispatch();
    check_path();
    check_agreement();
    if (fails) { std::fprintf(stderr, "ds41_prefill_test: %d FAILED\n", fails); return 1; }
    std::printf("ds41_prefill_test: PASS\n");
    return 0;
}
