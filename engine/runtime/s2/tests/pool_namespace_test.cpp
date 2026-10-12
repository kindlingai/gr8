// Opaque namespaces through PrefixReuse, the C++ adapter and the C ABI. Optional stores exercise physical copies.
#include "s2/pool.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace g6s2;
using g6rt::CacheNamespace;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::vector<int> prompt() { std::vector<int> p(192); for (int i = 0; i < 192; ++i) p[i] = i; return p; }
static g6rt::PrefixReuse finished(const std::vector<int>& p, CacheNamespace ns = {}) {
    g6rt::PrefixReuse st;
    st.history = p; st.valid = int(p.size()) - 1; st.written_end = st.pf_end = int(p.size());
    st.cache_namespace = ns;
    return st;
}

static void prefix() {
    const auto p = prompt(); CacheNamespace ns{}; ns.back() = 3;
    auto st = finished(p, ns); st.noend = true; st.dirty_end = 250;
    CHECK(st.begin(p, 191, true, ns) == 191);
    CHECK(st.begin(p, 191, true) == 0 && std::strcmp(st.why, "namespace") == 0);
    CHECK(st.dirty_end == 0 && st.cache_namespace == g6rt::STOCK_CACHE_NAMESPACE);
    st.resume(64, ns); CHECK(st.cache_namespace == ns && st.main_end == 64);
    st.clear(); CHECK(st.cache_namespace == g6rt::STOCK_CACHE_NAMESPACE);
}

static void ffi() {
    const auto p = prompt(); CacheNamespace ns{}; ns[0] = 1;
    g6kv_pool* pool = nullptr; CHECK(g6kv_new(8, 8, 512, &pool) == 0);
    g6kv_keep(pool, 0, 1, 128);
    g6kv_run r{}; CHECK(g6kv_reserve(pool, 192, 256, &r) == 0);
    const auto before = g6kv_digest(pool);
    const uint8_t short_buffer[1] = {1};
    for (size_t length : {size_t(0), size_t(1), size_t(G6KV_NAMESPACE_BYTES - 1), size_t(G6KV_NAMESPACE_BYTES + 1)}) {
        g6kv_hit h{}; h.skip = -99; uint32_t counted = 77;
        CHECK(g6kv_retain_namespaced(pool, &r, p.data(), p.size(), 191, 192, 192, short_buffer, length) != 0);
        CHECK(g6kv_match_namespaced(pool, p.data(), p.size(), 192, 256, 191, short_buffer, length, &h) != 0);
        CHECK(g6kv_count_namespaced(pool, p.data(), p.size(), 128, short_buffer, length, &counted) != 0);
        CHECK(h.skip == -99 && counted == 77 && g6kv_digest(pool) == before);
    }
    g6kv_hit h{}; h.skip = -99;
    uint32_t counted = 77;
    CHECK(g6kv_retain_namespaced(pool, &r, p.data(), p.size(), 191, 192, 192, nullptr, ns.size()) != 0);
    CHECK(g6kv_match_namespaced(pool, p.data(), p.size(), 192, 256, 191, nullptr, ns.size(), &h) != 0);
    CHECK(g6kv_match_namespaced(pool, p.data(), p.size(), 192, 256, 191, ns.data(), ns.size(), nullptr) != 0);
    CHECK(g6kv_retain_namespaced(pool, nullptr, p.data(), p.size(), 191, 192, 192, ns.data(), ns.size()) != 0);
    CHECK(g6kv_count_namespaced(pool, p.data(), p.size(), 128, nullptr, ns.size(), &counted) != 0);
    CHECK(g6kv_count_namespaced(pool, p.data(), p.size(), 128, ns.data(), ns.size(), nullptr) != 0);
    CHECK(g6kv_match_namespaced(nullptr, p.data(), p.size(), 192, 256, 191, ns.data(), ns.size(), &h) != 0);
    CHECK(counted == 77);
    CHECK(h.skip == -99 && g6kv_digest(pool) == before);
    CHECK(g6kv_retain_namespaced(pool, &r, p.data(), p.size(), 191, 192, 192, ns.data(), ns.size()) == 0);
    ns[0] = 2; // Retention copied the identity; mutating the caller's buffer cannot relabel it.
    CHECK(g6kv_match_namespaced(pool, p.data(), p.size(), 192, 256, 191, ns.data(), ns.size(), &h) == 0 && h.skip == 0);
    ns[0] = 1;
    CHECK(g6kv_match(pool, p.data(), p.size(), 192, 256, 191, &h) == 0 && h.skip == 0);
    CHECK(g6kv_match_namespaced(pool, p.data(), p.size(), 192, 256, 191, ns.data(), ns.size(), &h) == 0 && h.skip == 128);
    g6kv_free(pool);

    // Original ABI callers still use the stock namespace, interoperating with explicit all-zero queries.
    CHECK(g6kv_new(8, 8, 512, &pool) == 0); g6kv_keep(pool, 0, 1, 128);
    CHECK(g6kv_reserve(pool, 192, 256, &r) == 0);
    CHECK(g6kv_retain(pool, &r, p.data(), p.size(), 191, 192, 192) == 0);
    ns = {};
    CHECK(g6kv_match_namespaced(pool, p.data(), p.size(), 192, 256, 191, ns.data(), ns.size(), &h) == 0 && h.skip == 128);
    g6kv_free(pool);
}

static void adapter() {
    const auto p = prompt(); CacheNamespace ns{}; ns.back() = 1;
    RunPool pool(8, 8, 512); g6rt::PrefixReuse rules; rules.noend = true; rules.grid = 128;
    pool.keep(rules);
    const auto r = pool.reserve(192, 256);
    pool.retain(r, finished(p, ns)); // Retain the namespace carried by the finished state by default.
    CHECK(pool.match(p, 192, 256, 191).skip == 0);
    const auto h = pool.match(p, 192, 256, 191, ns);
    CHECK(h.skip == 128 && !h.stored);
    const auto next = pool.take(h, 192, 256); pool.release(next);
}

using Page = std::array<int32_t, 64>;
static void store(const char* spec) {
    const auto p = prompt(); CacheNamespace ns{}; ns[0] = 9;
    RunPool pool(8, 8, 512); g6rt::PrefixReuse rules; rules.noend = true; rules.grid = 128;
    pool.keep(rules); pool.stores(2); pool.add_tier(16, 2, "", "");
    g6kv_tier* tier = nullptr;
    CHECK(g6kv_tier_open(spec, "namespace-test", sizeof(Page), 16, &tier) == 0);
    if (!tier) return;
    std::array<Page, 8> dev{};
    std::vector<uint8_t> buffer(g6kv_tier_block_bytes(tier));
    auto copies = [&]() {
        const auto io = pool.io();
        for (const auto& x : io) {
            CHECK(x.part == 0 && x.tier == 0);
            if (x.kind == 0) {
                std::memcpy(buffer.data(), dev[x.page].data(), sizeof(Page));
                CHECK(g6kv_tier_write(tier, x.slot, buffer.data()) == 0);
            } else if (x.kind == 2) {
                CHECK(g6kv_tier_read(tier, x.slot, buffer.data()) == 0);
                std::memcpy(dev[x.page].data(), buffer.data(), sizeof(Page));
            }
            else CHECK(false);
        }
        CHECK(pool.io_done() == nullptr);
        return io.size();
    };
    auto write = [&](const Run& r, int salt) {
        const auto list = pool.list(r);
        for (size_t i = 0; i < p.size(); ++i) dev[list[i / 64]][i % 64] = p[i] + salt;
    };
    CHECK(pool.count(p, 128, ns) == 0);
    auto r = pool.reserve(192, 256); write(r, 10000); pool.retain(r, finished(p, ns));
    CHECK(copies() == 0 && pool.tiers()[0].free_slots == 16);
    auto all = pool.reserve(512); CHECK(copies() == 0); pool.release(all);
    CHECK(pool.match(p, 192, 256, 191).skip == 0);
    CHECK(pool.match(p, 192, 256, 191, ns).skip == 0);

    // Stock spill/restore still works through the same memory/file/remote backend.
    CHECK(pool.count(p, 128) == 1);
    r = pool.reserve(192, 256); write(r, 0); pool.retain(r, finished(p)); copies();
    all = pool.reserve(512); CHECK(copies() > 0); pool.release(all);
    const auto h = pool.match(p, 192, 256, 191); CHECK(h.stored && h.skip == 128);
    CHECK(pool.match(p, 192, 256, 191, ns).skip == 0);
    r = pool.take(h, 192, 256); CHECK(copies() > 0);
    const auto list = pool.list(r);
    for (int i = 0; i < h.skip; ++i) CHECK(dev[list[i / 64]][i % 64] == p[i]);
    pool.release(r); g6kv_tier_free(tier);
}

int main(int argc, char** argv) {
    prefix(); ffi(); adapter();
    if (argc == 1) store("mem:");
    for (int i = 1; i < argc; ++i) store(argv[i]);
    std::printf("pool_namespace_test: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
