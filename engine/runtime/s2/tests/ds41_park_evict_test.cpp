// GR8_DS_PARK eviction vs the grant's own snapshot (ds41_native.cuh bind, Ds41PrefixCache::capture keep). A new prompt that
// matches a prompt-end snapshot PE (claimed by its grant) but not the parked chat's request-end tokens RE, all slots full, so
// the bind evicts that parked slot: the capture of RE must not evict PE, which the bind restores next. Before the fix the
// capture only moved PE to the back of the LRU (touch), so a budget that fits PE or RE but not both evicted PE and the
// bind aborted ("the evicted chat's capture evicted the grant's own snapshot"). the reviewer's fixture (review of 70b7282):
// 1 MiB budget, one attention layer of 256 ring rows; BF16 PE 774144 + RE 876544 B, FP4 PE 521344 + RE 550144 B.
// Host only: the CUDA calls are stubs over host memory with an optional allocation limit.
// Build (one line): g++ -std=c++17 -O1 -g -fsanitize=address,undefined -DG6_MODEL_DS41=1 -I engine/runtime engine/runtime/s2/tests/ds41_park_evict_test.cpp -o /tmp/ds41_park_evict_test <knobs lib>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "prefix_reuse.h"
#include "../engram/engram_ring.h"
#include "../knobs-rs/gr8_knobs.h"

using __nv_bfloat16 = uint16_t;
using cudaStream_t = int;
constexpr int cudaSuccess = 0, cudaMemcpyDeviceToDevice = 0, cudaMemcpyHostToDevice = 1;
static std::unordered_map<void*, size_t> mem;
static size_t live_bytes = 0, alloc_limit = SIZE_MAX;   // alloc_limit: device memory pressure (cudaMalloc fails above it)
static int cudaMalloc(void** p, size_t n) {
    if (n > alloc_limit - live_bytes) return 2;
    *p = std::malloc(n); mem[*p] = n; live_bytes += n; return cudaSuccess;
}
static int cudaFree(void* p) { if (!p) return 0; auto i = mem.find(p); assert(i != mem.end()); live_bytes -= i->second; mem.erase(i); std::free(p); return 0; }
static int cudaGetLastError() { return 0; }
static int cudaStreamSynchronize(int) { return 0; }
static int cudaMemcpyAsync(void* d, const void* s, size_t n, int, int) { std::memcpy(d, s, n); return 0; }
#define G6RT_CK(x) assert((x) == 0)
namespace ds41a { constexpr int HD = 512, CQ_ROW = 288; }
namespace g6rt {
constexpr size_t DS41_CA_ROW = 288, DS41_ICA_ROW = 68;
constexpr int ENG_RING = 64;
struct Cache { uint16_t *ring = nullptr, *comp = nullptr; uint8_t* compq = nullptr; float *kvs = nullptr, *scs = nullptr; int ring_rows = 256; size_t comp_ss = 0; };
struct Ds41AttnLayer { Cache C; };
struct Ds41IdxModel { uint8_t *ca[4]{}, *ica[4]{}; uint16_t* shadow = nullptr; int rows_kv[4]{}, cap = 2048, rcap = 8; };
struct Model { std::vector<Ds41AttnLayer> layers; Ds41IdxModel* idx = nullptr; int* ehist = nullptr; int rank = 0; };
static Ds41AttnLayer* ds41_attn_state(Model& m, int i) { return &m.layers[size_t(i)]; }
[[maybe_unused]] static const Ds41AttnLayer* ds41_attn_state(const Model& m, int i) { return &m.layers[size_t(i)]; }
namespace ds41_idx_detail {
static Ds41IdxModel* find(Model& m, bool) { return m.idx; }
[[maybe_unused]] static const Ds41IdxModel* find(const Model& m, bool) { return m.idx; }
}
}
namespace g6s2 { struct Ds41Dspark { g6rt::Model draft; }; }
#define GR8_DS41_PREFIX_CACHE_STANDALONE 1
#include "s2/ds41_prefix_cache.cuh"
#include "s2/driver.h"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <tuple>

using g6s2::Ds41PrefixCache;
static g6rt::CacheNamespace ns(int n) { g6rt::CacheNamespace s{}; s[0] = n; return s; }
static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)

struct Fixture {
    g6rt::Model m;
    std::vector<uint16_t> ring, comp;
    std::vector<uint8_t> packed;
    int hist[3 * 64]{};
    explicit Fixture(bool fp4) : ring(3 * 256 * 512, 7), comp(3 * 1024 * 512, 7), packed(3 * 1024 * 288, 7) {
        m.layers.resize(1); m.ehist = hist;
        auto& c = m.layers[0].C; c.ring = ring.data(); c.comp_ss = 1024 * size_t(fp4 ? 288 : 512);
        if (fp4) c.compq = packed.data(); else c.comp = comp.data();
    }
};

// the bind's eviction of a parked slot, as ds41_native.cuh runs it (keep = the grant's claimed generation)
static uint64_t evict_parked(Ds41PrefixCache* ck, int q, uint32_t pos, const std::vector<int>& tokens, const g6rt::CacheNamespace& n, uint64_t keep) {
    const uint64_t gen = ck->capture(q, pos, tokens, n, 0, g6s2::Ds41Snap::RequestEnd, 0, false, keep);
    g6s2::require(!keep || ck->find(keep), "DS41 park: the evicted chat's capture evicted the grant's own snapshot");
    return gen;
}

static void tight_budget(bool fp4, bool pressure) {
    Fixture f(fp4);
    Ds41PrefixCache cache(f.m, nullptr, 1);   // 1 MiB
    std::vector<int> t(2048); for (int i = 0; i < 2048; ++i) t[size_t(i)] = i + 1;
    const uint32_t pe_pos = fp4 ? 1800u : 1000u, re_pos = pe_pos + 200;
    const size_t pe_b = fp4 ? 521344 : 774144, re_b = fp4 ? 550144 : 876544;
    CHECK(cache.entry_bytes(pe_pos) == pe_b && cache.entry_bytes(re_pos) == re_b);
    CHECK(pe_b + re_b > cache.budget_bytes() && re_b < cache.budget_bytes());
    const uint64_t pe = cache.capture(0, pe_pos, t, ns(1), 0);
    CHECK(pe != 0);
    std::vector<int> next = t; next[pe_pos + 10]++;   // matches PE, not the parked RE
    const auto hit = cache.match(next, int(re_pos), ns(1));
    CHECK(hit.hit && hit.generation == pe);
    if (pressure) alloc_limit = live_bytes + re_b - 1;   // the budget would fit after evicting PE; device memory does not
    bool threw = false;
    uint64_t gen = 1;
    try { gen = evict_parked(&cache, 0, re_pos, t, ns(1), pe); } catch (const std::runtime_error& e) { threw = true; fprintf(stderr, "abort: %s\n", e.what()); }
    alloc_limit = SIZE_MAX;
    CHECK(!threw);
    CHECK(gen == 0);                       // the optional capture is skipped
    CHECK(cache.find(pe) != nullptr);      // the grant's snapshot survives and restores
    CHECK(cache.evictions() == 0 && cache.used_bytes() == pe_b && cache.used_bytes() <= cache.budget_bytes());
    cache.restore(pe, 0, 0);
    printf("ds41_park_evict_test %s%s: budget %zu PE %zu RE %zu -> capture skipped, PE kept and restored\n", fp4 ? "FP4" : "BF16",
           pressure ? " alloc-pressure" : "", cache.budget_bytes(), pe_b, re_b);
}

// keep only protects one entry: an older entry is still evicted for the capture, and keep = 0 is the knob-off eviction
static void others_evicted(bool fp4) {
    Fixture f(fp4);
    // PE + another entry fit, PE + RE fit, all three do not: BF16 2 MiB, FP4 1 MiB. RE is more than 64 tokens past PE (the
    // largest GR8_DS_CACHE_DEDUP), so its capture is a new entry at every dedup
    Ds41PrefixCache cache(f.m, nullptr, fp4 ? 1 : 2);
    std::vector<int> t(2048); for (int i = 0; i < 2048; ++i) t[size_t(i)] = i + 1;
    std::vector<int> u = t; u[0] = 99999;
    const uint32_t pe_pos = fp4 ? 958u : 1000u, re_pos = fp4 ? 1030u : 1200u;
    CHECK(2 * cache.entry_bytes(pe_pos) <= cache.budget_bytes() && cache.entry_bytes(pe_pos) + cache.entry_bytes(re_pos) <= cache.budget_bytes() &&
          2 * cache.entry_bytes(pe_pos) + cache.entry_bytes(re_pos) > cache.budget_bytes());
    const uint64_t pe = cache.capture(0, pe_pos, t, ns(1), 0);
    const uint64_t other = cache.capture(0, pe_pos, u, ns(1), 0);   // newer than PE
    CHECK(pe && other);
    const uint64_t gen = evict_parked(&cache, 0, re_pos, t, ns(1), pe);
    CHECK(gen != 0 && cache.find(pe) && !cache.find(other) && cache.evictions() == 1);
    // keep = 0: plain LRU (the oldest goes first), as before this change
    const uint64_t gen2 = cache.capture(0, re_pos, u, ns(1), 0);
    CHECK(gen2 != 0 && !cache.find(pe) && cache.find(gen) && cache.evictions() == 2);
    printf("ds41_park_evict_test %s: keep spares only the claimed entry (evictions %llu)\n", fp4 ? "FP4" : "BF16", (unsigned long long)cache.evictions());
}


// The same through admission (the reviewer's regression): Driver + Scheduler with GR8_DS_PARK on and 2 slots, a test native layer
// that plays ds41_native.cuh (prompt-end capture, park at finish, the bind's eviction of a parked slot and the restore of
// the grant's hit). Chats A and B finish and park, so both slots are occupied; chat C's prompt matches A's prompt-end
// snapshot PE but not A's parked tokens (thinking on, no tools); the grant lands on A's parked slot (the oldest), whose
// eviction capture of A's request end RE does not fit beside PE.
static int tok(uint32_t chat, uint32_t i) { return int((chat * 2654435761u ^ i * 40503u) % 5); }
static void admission(bool fp4, bool pressure)
{
    using namespace g6s2;
    Fixture f(fp4);
    Ds41PrefixCache cache(f.m, nullptr, 1), *ck = &cache;
    const uint32_t len = fp4 ? 1901 : 520;   // PE + RE over 1 MiB, each under it
    Config c;
    c.epoch = 1; c.mtp = 1; c.max_rows = 8; c.max_sessions = 2; c.prefill_rows = 64; c.two_lanes = false; c.park = true;
    struct Parked { bool on = false; std::vector<int> tokens; uint32_t pos = 0; };
    std::array<Parked, 2> parked{};
    std::map<uint64_t, std::vector<int>> prompt, out;
    std::map<uint64_t, uint32_t> maxt;
    uint64_t claimed = 0, pe = 0, re_gen = 1;
    bool threw = false;
    std::unique_ptr<Driver> d;
    DriverHooks h;
    h.bind = [&](const Grant& g) {   // ds41_native.cuh h.bind with park, then the restore of the grant's hit
        const int q = int(g.key.slot);
        Parked& k = parked[size_t(q)];
        try {
            if (k.on) re_gen = evict_parked(ck, q, k.pos, k.tokens, ns(1), claimed);
            k = Parked{};
            if (g.reuse) cache.restore(claimed, q, 0);
        } catch (const std::runtime_error& e) { threw = true; fprintf(stderr, "abort: %s\n", e.what()); }
        claimed = 0;
    };
    h.retire = [](const Slot&) {};
    h.prefill = [&](const Slot& sl, const Payload&, uint32_t first, uint32_t rows) {   // prompt end: the PE capture of chat A
        if (sl.req.id == 1 && first + rows == sl.req.prompt_tokens - 1) pe = cache.capture(int(sl.key.slot), first + rows, prompt.at(1), ns(1), 0);
    };
    h.run = [&](const s2::Step& st) { std::vector<int> ids; for (const auto& r : st.r) ids.push_back(tok(r.pos, uint32_t(r.tok))); return ids; };
    h.emit = [&](Key k, const int* ids, int n, int) {
        const Slot& sl = d->state().slots()[k.slot];
        auto& o = out[sl.req.id];
        const int cnt = std::min<int>(n, int(sl.req.max_tokens - o.size()));
        o.insert(o.end(), ids, ids + cnt);
        return Emission{uint32_t(cnt), o.size() == sl.req.max_tokens};
    };
    h.finished = [&](const Slot& sl) {   // the park hook: the slot keeps tokens [0, pos)
        Parked& k = parked[sl.key.slot];
        k.on = true; k.tokens = prompt.at(sl.req.id);
        k.tokens.insert(k.tokens.end(), out.at(sl.req.id).begin(), out.at(sl.req.id).end());
        k.pos = uint32_t(k.tokens.size()) - 1;
    };
    d = std::make_unique<Driver>(c, h);
    auto admit = [&](uint64_t id, std::vector<int> pr, uint32_t mt, uint32_t reuse) {
        const SchedulerView& sc = d->state();
        if (!sc.resident()) d->idle_boundary(sc.boundary() + NOTICE_MIN);
        prompt[id] = pr; maxt[id] = mt;
        const Request rq{id, uint32_t(pr.size()), mt};
        Commit cm; cm.epoch = 1; cm.event = sc.next_event(); cm.boundary = sc.boundary(); cm.before_hash = sc.digest();
        cm.admit = Grant{sc.next_key(), rq, Run{0, reserve_pages(c, rq), 1, id * 77}, reuse};
        Payload p; p.prompt = pr; p.chunk = 4;
        d->apply(cm, &p);
        return cm.admit->key.slot;
    };
    auto drain = [&] { for (int g = 0; d->state().resident() && g < 100000; ++g) d->step(); };
    std::vector<int> a(len), b(len / 2);
    for (uint32_t i = 0; i < len; ++i) a[i] = tok(1, i);
    for (uint32_t i = 0; i < len / 2; ++i) b[i] = tok(2, i);
    admit(1, a, 3, 0); drain();
    admit(2, b, 3, 0); drain();
    CHECK(pe != 0 && parked[0].on && parked[1].on);
    const uint32_t pe_pos = len - 1;
    std::vector<int> cp(a.begin(), a.begin() + pe_pos);   // A's prompt, then a new turn that drops A's reply (thinking)
    for (int i = 0; i < 40; ++i) cp.push_back((tok(3, uint32_t(i)) + 1) % 5);
    const auto hit = cache.match(cp, int(cp.size()) - 1, ns(1));
    CHECK(hit.hit && hit.generation == pe && hit.position == pe_pos);
    CHECK(g6rt::PrefixReuse::common_prefix(parked[0].tokens, cp) < int(parked[0].pos));   // not a resume of A
    CHECK(cache.entry_bytes(pe_pos) + cache.entry_bytes(parked[0].pos) > cache.budget_bytes());
    claimed = pe;
    if (pressure) alloc_limit = live_bytes + cache.entry_bytes(parked[0].pos) - 1;
    const unsigned slot = admit(3, cp, 4, pe_pos);
    alloc_limit = SIZE_MAX;
    CHECK(slot == 0);   // the occupied slot: A's, parked first
    CHECK(!threw && re_gen == 0 && cache.find(pe) && cache.evictions() == 0 && cache.hits() == 1);
    drain();
    CHECK(out.at(3).size() == 4 && d->state().slots()[0].req.id == 3);
    printf("ds41_park_evict_test admission %s%s: chat C on A's parked slot %u, A's capture skipped, PE restored, C served\n",
           fp4 ? "FP4" : "BF16", pressure ? " alloc-pressure" : "", slot);
}


// GR8_DS_PARK with GR8_DS_CACHE_REPLACE (Ds41SnapSlots park / evict_parked / resume): the same request sequence, served once
// as ds41_native.cuh does with PARK=0 (retain captures the request end at the finish) and once with PARK=1 (the finished chat
// parks: a follow-up that extends it resumes the slot, a new chat with no free slot evicts the oldest parked one), frees the
// same snapshots (replaced, siblings) and ends with the same prompt-end entries; the request-end entries differ only by
// capture timing: a resumed chat's request end is never captured (the follow-up continues in the slot). Three chat shapes,
// interleaved on 2 slots: tools (the next prompt keeps the reply), thinking without tools (it drops the reply: only the
// prompt-end snapshot matches, the PE path, not a resume), retry (the same prompt again).
struct Native {
    bool park;
    Fixture f{false};
    Ds41PrefixCache ck{f.m, nullptr, 64};
    g6s2::Ds41SnapSlots snaps{2};
    struct Parked { bool on = false; uint32_t pos = 0; std::vector<int> tokens; uint64_t order = 0; };
    std::array<Parked, 2> parked{};
    uint64_t parks = 0;
    long resumes = 0, evictions = 0;
    std::vector<std::vector<int>> resumed;   // the request ends a resume continued from (never captured with park)
    explicit Native(bool p) : park(p) {}
    // one request: admission (the head's match and take), bind, prompt-end capture, finish (retain or park)
    void serve(const std::vector<int>& prompt, const std::vector<int>& reply) {
        const auto n = ns(1);
        const int limit = int(prompt.size()) - 1;
        const g6s2::Ds41PrefixHit hit = ck.match(prompt, limit, n);
        int q = -1;
        bool resume = false;
        if (park) {
            for (int i = 0; i < 2; ++i) {   // the parked slot the prompt extends, as far as the cache's hit at least
                const Parked& k = parked[size_t(i)];
                if (k.on && k.pos && int(k.pos) <= limit && g6rt::PrefixReuse::common_prefix(k.tokens, prompt) >= int(k.pos) &&
                    (!hit.hit || k.pos >= hit.position) && (q < 0 || k.pos > parked[size_t(q)].pos)) q = i;
            }
            resume = q >= 0;
            if (!resume) {   // a free slot, else the oldest parked one
                for (int i = 0; i < 2 && q < 0; ++i) if (!parked[size_t(i)].on) q = i;
                if (q < 0) q = parked[0].order < parked[1].order ? 0 : 1;
            }
        } else q = 0;
        const uint64_t keep = !resume && hit.hit ? hit.generation : 0;
        if (park) {   // the bind's parked part
            Parked& k = parked[size_t(q)];
            if (resume) { snaps.resume(ck, q, k.pos, k.tokens, n); resumed.push_back(k.tokens); ++resumes; }
            else if (k.on) {
                snaps.evict_parked(ck, q, k.pos, k.tokens, n, 0, keep);
                CHECK(!keep || ck.find(keep));
                ++evictions;
            }
            k = Parked{};
        }
        if (!resume) {
            snaps.reset(q);
            if (keep) snaps.restore(ck, keep, q, 0);
        }
        snaps.capture_prompt_end(ck, q, uint32_t(limit), prompt, n, 0);
        std::vector<int> hist = prompt;
        hist.insert(hist.end(), reply.begin(), reply.end());
        const uint32_t p = uint32_t(hist.size()) - 1;
        if (park) {
            Parked& k = parked[size_t(q)];
            k.on = true; k.pos = p; k.tokens.assign(hist.begin(), hist.begin() + p); k.order = ++parks;
            snaps.park(ck, q, p, k.tokens, n);
        } else {
            snaps.capture_request_end(ck, q, p, hist, n, 0);
            snaps.reset(q);
        }
    }
    void flush() {   // evict every parked chat, oldest first (a later new chat would)
        for (int r = 0; r < 2; ++r) {
            int q = -1;
            for (int i = 0; i < 2; ++i) if (parked[size_t(i)].on && (q < 0 || parked[size_t(i)].order < parked[size_t(q)].order)) q = i;
            if (q < 0) break;
            Parked& k = parked[size_t(q)];
            snaps.evict_parked(ck, q, k.pos, k.tokens, ns(1), 0, 0);
            k = Parked{};
        }
    }
};
static std::vector<std::pair<int, std::vector<int>>> stored(const Ds41PrefixCache& ck, g6s2::Ds41Snap kind) {
    std::vector<std::pair<int, std::vector<int>>> v;
    for (const auto& e : ck.entries()) if (e.kind == kind) v.push_back({int(e.position), e.tokens});
    std::sort(v.begin(), v.end());
    return v;
}
static void combined() {
    using g6s2::Ds41Snap;
    // chat 1 tools, chat 2 thinking without tools, chat 3 retry; 3 turns each in the order 1a 2a 1b 3a 1c 2b 3b 2c 3c (chat 1's
    // follow-ups come while it is parked: resumes; the others evict); replies of 70..73 tokens and follow-ups that add 70,
    // past the largest GR8_DS_CACHE_DEDUP (64): a shorter one folds into the stored entry it extends and adds none
    auto gen = [](int chat, int turn, int n, int salt) { std::vector<int> v; for (int i = 0; i < n; ++i) v.push_back(1 + (chat * 131 + turn * 17 + i * 7 + salt) % 997); return v; };
    std::array<std::vector<int>, 4> prompt, last_reply;
    std::array<int, 4> turns{};
    std::vector<std::pair<std::vector<int>, std::vector<int>>> seq;
    for (int chat : {1, 2, 1, 3, 1, 2, 3, 2, 3}) {
        const int turn = turns[size_t(chat)]++;
        std::vector<int> pr = prompt[size_t(chat)];
        const std::vector<int> reply = gen(chat, turn, 70 + (chat + turn) % 4, 500);
        if (turn == 0) pr = gen(chat, 0, 40 + 10 * chat, 0);
        else if (chat == 1) { pr.insert(pr.end(), last_reply[1].begin(), last_reply[1].end()); auto x = gen(chat, turn, 70, 9); pr.insert(pr.end(), x.begin(), x.end()); }
        else if (chat == 2) { auto x = gen(chat, turn, 70, 9); pr.insert(pr.end(), x.begin(), x.end()); }
        // chat 3: a retry, the same prompt
        prompt[size_t(chat)] = pr; last_reply[size_t(chat)] = reply;
        seq.push_back({pr, reply});
    }
    Native off(false), on(true);
    for (const auto& [pr, re] : seq) { off.serve(pr, re); on.serve(pr, re); }
    on.flush();
    CHECK(on.resumes == 2 && on.evictions >= 2);   // chat 1's turns 2 and 3 resume; the rest evict
    CHECK(on.ck.replaced() == off.ck.replaced() && on.ck.siblings() == off.ck.siblings());
    CHECK(stored(on.ck, Ds41Snap::PromptEnd) == stored(off.ck, Ds41Snap::PromptEnd));
    // request ends: PARK=1's are PARK=0's less the ones a resume continued from
    auto re_on = stored(on.ck, Ds41Snap::RequestEnd), re_off = stored(off.ck, Ds41Snap::RequestEnd);
    for (const auto& t : on.resumed) {
        auto it = std::find(re_off.begin(), re_off.end(), std::make_pair(int(t.size()), t));
        CHECK(it != re_off.end());
        if (it != re_off.end()) re_off.erase(it);
    }
    CHECK(re_on == re_off);
    for (const auto& e : on.ck.entries()) CHECK(!(e.sibling >> 63));   // no placeholder left after the flush
    printf("ds41_park_evict_test combined (GR8_DS_CACHE_REPLACE=%d): %zu requests, %ld resumes, %ld evictions; replaced %llu siblings %llu "
           "(PARK=0 %llu %llu), prompt-end entries %zu (PARK=0 %zu), request-end %zu (PARK=0 %zu, %zu resumed)\n", int(knob::ds_cache_replace()),
           seq.size(), on.resumes, on.evictions, (unsigned long long)on.ck.replaced(), (unsigned long long)on.ck.siblings(),
           (unsigned long long)off.ck.replaced(), (unsigned long long)off.ck.siblings(), stored(on.ck, Ds41Snap::PromptEnd).size(),
           stored(off.ck, Ds41Snap::PromptEnd).size(), re_on.size(), stored(off.ck, Ds41Snap::RequestEnd).size(), on.resumed.size());
}

// (a) the eviction capture of a parked chat is a request-end entry linked to its prompt-end one, as retain's; (c) a thinking
// follow-up (only the prompt-end entry matches) takes the PE path, not a resume, and frees the parked request end as the
// restore of its sibling would
static void lifecycle() {
    using g6s2::Ds41Snap;
    std::vector<int> a(60); for (int i = 0; i < 60; ++i) a[size_t(i)] = 3 + i;
    std::vector<int> reply(70);   // past the largest GR8_DS_CACHE_DEDUP (64): the request end is its own entry
    for (int i = 0; i < 70; ++i) reply[size_t(i)] = 901 + i;
    {   // (a)
        Native on(true);
        on.serve(a, reply);
        std::vector<int> b(50, 77);
        on.serve(b, reply);
        std::vector<int> c(45, 55);
        on.serve(c, reply);   // no free slot: evicts chat a
        const g6s2::Ds41PrefixEntry *pe = nullptr, *re = nullptr;
        for (const auto& e : on.ck.entries()) {
            if (e.position == 59 && e.kind == Ds41Snap::PromptEnd) pe = &e;
            if (e.position == 59 + reply.size() && e.kind == Ds41Snap::RequestEnd) re = &e;
        }
        const bool rp = knob::ds_cache_replace();   // GR8_DS_CACHE_REPLACE=0: no links, as retain's
        CHECK(pe && re && on.evictions == 1 && pe->sibling == (rp ? re->generation : 0) && re->sibling == (rp ? pe->generation : 0));
    }
    {   // (c)
        Native on(true), off(false);
        std::vector<int> think = a;
        for (int i = 0; i < 70; ++i) think.push_back(400 + i);   // drops the reply; 70 new tokens, past the dedup window
        for (Native* n : {&on, &off}) { n->serve(a, reply); n->serve(think, reply); }
        CHECK(on.resumes == 0 && on.ck.hits() == 1 && off.ck.hits() == 1);   // the PE path on both
        on.flush();
        CHECK(on.ck.siblings() == off.ck.siblings() && on.ck.siblings() == (knob::ds_cache_replace() ? 1u : 0u));
        CHECK(stored(on.ck, Ds41Snap::RequestEnd).size() == stored(off.ck, Ds41Snap::RequestEnd).size());
        CHECK(stored(on.ck, Ds41Snap::PromptEnd) == stored(off.ck, Ds41Snap::PromptEnd));
    }
    printf("ds41_park_evict_test lifecycle (GR8_DS_CACHE_REPLACE=%d): eviction capture is RequestEnd (linked to its PE with replace); "
           "a thinking follow-up takes the PE path\n", int(knob::ds_cache_replace()));
}


// The reviewer's review of a2d4ace (GR8_DS_CACHE_REPLACE=1 only; the knob off links nothing):
// P2 #1: chats A and A' with the same prompt share A's prompt-end entry PE. A' parks on q1, A on q0 (PE links to A's park);
// A' is evicted (RE), then A resumes RE or is evicted into that exact entry. A's park must leave no placeholder behind: an
// unrelated chat B parked later on q0 is not freed by a restore of A's PE, and B's eviction captures B's request end.
static std::vector<int> seq_toks(int k) { std::vector<int> t(2048); for (int i = 0; i < 2048; ++i) t[size_t(i)] = i + 1 + k * 10000; return t; }
static bool placeholder(const Ds41PrefixCache& c) { for (const auto& e : c.entries()) if (e.sibling >> 63) return true; return false; }
static void stale_placeholder(bool fp4, bool use_resume) {
    Fixture f(fp4);
    Ds41PrefixCache c(f.m, nullptr, 64);
    g6s2::Ds41SnapSlots s(3);
    const auto a = seq_toks(0), b = seq_toks(1);
    const uint64_t pe = s.capture_prompt_end(c, 0, 100, a, ns(1), 0);
    CHECK(s.capture_prompt_end(c, 1, 100, a, ns(1), 0) == pe);   // A' refreshes A's entry
    s.park(c, 1, 200, a, ns(1));
    s.park(c, 0, 200, a, ns(1));
    CHECK(c.find(pe) && (c.find(pe)->sibling >> 63));            // linked to A's park
    const uint64_t re = s.evict_parked(c, 1, 200, a, ns(1), 0, 0);
    CHECK(re != 0);
    if (use_resume) s.resume(c, 0, 200, a, ns(1));
    else CHECK(s.evict_parked(c, 0, 200, a, ns(1), 0, 0) == re);   // the exact entry
    CHECK(!placeholder(c));                                      // A's park detached on the exact/resume path
    s.reset(0);
    const uint64_t bpe = s.capture_prompt_end(c, 0, 100, b, ns(2), 0);
    s.park(c, 0, 200, b, ns(2));
    const uint64_t sib_before = c.siblings();
    c.restore(pe, 1, 0);                                         // a retry of A
    CHECK(c.siblings() == sib_before);                           // frees nothing of B's
    const uint64_t bre = s.evict_parked(c, 0, 200, b, ns(2), 0, 0);
    CHECK(bre != 0 && c.find(bpe) && c.find(bpe)->sibling == bre && c.find(bre) && c.find(bre)->sibling == bpe);
    CHECK(!placeholder(c));
    printf("ds41_park_evict_test stale placeholder %s %s: A's park detached, B's request end captured (gen %llu) and linked to its PE\n",
           fp4 ? "FP4" : "BF16", use_resume ? "resume" : "exact-eviction", (unsigned long long)bre);
}
// P2 #2: a request restores RE50 (request end), captures its PE150; another request stores the equal RE300 first. At the
// finish PARK=0 frees PE150 (capture's equal-entry path, replace); PARK=1's park must free it too.
static void exact_park_replace(bool fp4) {
    Fixture f(fp4), g(fp4);
    Ds41PrefixCache on(f.m, nullptr, 64), off(g.m, nullptr, 64);
    g6s2::Ds41SnapSlots so(3), sf(3);
    const auto t = seq_toks(0);
    uint64_t pe_on = 0, pe_off = 0;
    for (auto [c, s, pe] : {std::make_tuple(&on, &so, &pe_on), std::make_tuple(&off, &sf, &pe_off)}) {
        const uint64_t old = c->capture(0, 50, t, ns(1), 0, g6s2::Ds41Snap::RequestEnd);
        s->restore(*c, old, 0, 0);
        *pe = s->capture_prompt_end(*c, 0, 150, t, ns(1), 0);
        c->capture(1, 300, t, ns(1), 0, g6s2::Ds41Snap::RequestEnd);   // the other request's equal request end
    }
    so.park(on, 0, 300, t, ns(1));
    sf.capture_request_end(off, 0, 300, t, ns(1), 0);
    sf.reset(0);
    CHECK(off.replaced() == 1 && !off.find(pe_off));
    CHECK(on.replaced() == off.replaced() && !on.find(pe_on));
    so.evict_parked(on, 0, 300, t, ns(1), 0, 0);
    CHECK(!on.find(pe_on) && on.size() == off.size());
    printf("ds41_park_evict_test exact park replace %s: PARK=1 frees its PE as PARK=0 does (replaced %llu = %llu)\n", fp4 ? "FP4" : "BF16",
           (unsigned long long)on.replaced(), (unsigned long long)off.replaced());
}
// the digest gap: a park's placeholder link changes park_digest (the park-on pool digest), digest() (off mode) does not change
static void park_digest_link() {
#ifndef PARK_BEFORE
    Fixture f(false), g(false);
    Ds41PrefixCache a(f.m, nullptr, 64), b(g.m, nullptr, 64);
    const auto t = seq_toks(0);
    const uint64_t pa = a.capture(0, 100, t, ns(1), 0, g6s2::Ds41Snap::PromptEnd), pb = b.capture(0, 100, t, ns(1), 0, g6s2::Ds41Snap::PromptEnd);
    CHECK(pa == pb && a.park_digest() == b.park_digest());
    const uint64_t d0 = a.digest();
    a.park_end(pa, 200, t, ns(1), false);
    CHECK(a.find(pa)->sibling != b.find(pb)->sibling && a.digest() == d0 && a.park_digest() != b.park_digest());
    printf("ds41_park_evict_test park digest: a placeholder link changes park_digest, digest() unchanged\n");
#endif
}

// DS41 image input (the reviewer, review of 4b7ef70): a prefix-cache entry captured after an image (0..3 text tokens before the
// snapshot) restores the Engram ring the live steps left, -1 at the image rows, not the image token. The prompt-end entry
// (capture_prompt_end) and a request-end entry captured by a park eviction (evict_parked), each restored on slot 0.
static void image_ring() {
    constexpr int TI = g6rt::TOK_IMAGE;
    int cases = 0;
    for (int gap = 0; gap <= 3; ++gap) {
        Fixture f(false);
        Ds41PrefixCache cache(f.m, nullptr, 4);
        g6s2::Ds41SnapSlots snaps(1);
        std::vector<int> t;
        for (int i = 0; i < 300; ++i) t.push_back(1000 + i);
        g6rt::ImageIn im; im.pos = 300; im.span = 16;
        for (int i = 0; i < im.span; ++i) t.push_back(TI);
        for (int i = 0; i <= gap; ++i) t.push_back(2000 + i);
        const uint32_t pe_pos = uint32_t(t.size()) - 1;   // the prompt end: the image ends gap tokens before it
        const g6s2::Ds41ImageRows rows = g6s2::ds41_image_rows({im});
        auto live = [&](const std::vector<int>& tok, uint32_t pos, int* want) {   // the live ring: -1 at image rows
            std::fill_n(want, 64, -1);
            for (int q = std::max(0, int(pos) - 64); q < int(pos); ++q)
                want[q & 63] = q >= im.pos && q < im.pos + im.span ? -1 : tok[size_t(q)];
        };
        int want[64];
        const uint64_t pe = snaps.capture_prompt_end(cache, 0, pe_pos, t, ns(1), 0, rows);
        CHECK(pe != 0);
        std::fill_n(f.hist, 64, 12345);
        cache.restore(pe, 0, 0);
        live(t, pe_pos, want);
        CHECK(std::equal(want, want + 64, f.hist));
        CHECK(gap == 3 || f.hist[(pe_pos - 1 - uint32_t(gap)) & 63] == -1);   // the image's last row is in the next n-gram
        ++cases;
        // the request end of the same chat (a park evicted into the cache, in a cache of its own: the dedup would keep the
        // prompt-end entry instead), 0..3 tokens generated after the prompt
        for (uint32_t g = 0; g <= 3; ++g) {
            Fixture f2(false);
            Ds41PrefixCache c2(f2.m, nullptr, 4);
            g6s2::Ds41SnapSlots s2(1);
            std::vector<int> hh = t;
            for (uint32_t i = 0; i < g + 1; ++i) hh.push_back(3000 + int(i));
            const uint32_t re_pos = pe_pos + g;
            const uint64_t re = s2.evict_parked(c2, 0, re_pos, hh, ns(1), 0, 0, rows);
            CHECK(re != 0);
            std::fill_n(f2.hist, 64, 12345);
            if (re) c2.restore(re, 0, 0);
            live(hh, re_pos, want);
            CHECK(std::equal(want, want + 64, f2.hist));
            ++cases;
        }
    }
    printf("ds41_park_evict_test image ring: %d prompt-end and request-end restores equal the live ring (image 0..3 tokens before)\n", cases);
}

int main() {
    for (bool fp4 : {false, true}) {
        tight_budget(fp4, false);
        tight_budget(fp4, true);
        others_evicted(fp4);
        admission(fp4, false);
        admission(fp4, true);
    }
    combined();
    lifecycle();
    image_ring();
    if (knob::ds_cache_replace()) {
        for (bool fp4 : {false, true}) {
            stale_placeholder(fp4, false);
            stale_placeholder(fp4, true);
            exact_park_replace(fp4);
        }
        park_digest_link();
    }
    CHECK(mem.empty());
    if (fails) { fprintf(stderr, "ds41_park_evict_test: %d FAIL\n", fails); return 1; }
    printf("ds41_park_evict_test: ok (%d cases)\n", knob::ds_cache_replace() ? 20 : 13);
    return 0;
}
