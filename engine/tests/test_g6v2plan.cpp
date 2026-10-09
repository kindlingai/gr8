// The image g6l_v2_plan plans from a base pack, the exlr8 v2 files and the index slices, against the image of the
// v2pack sparse pack for the same rank (tools/convert/v2pack.py): the same manifest entries and, chunk by chunk, the
// same bytes. Hashes chunks instead of holding two images.
//   cargo build --release --manifest-path engine/loader-rs/Cargo.toml
//   g++ -O2 -std=c++20 -pthread -I engine/loader -o test_g6v2plan engine/tests/test_g6v2plan.cpp engine/loader/g6pack.cpp
//       engine/loader-rs/target/release/libg6load.a -ldl -lm      (one command)
//   test_g6v2plan BASE_DIR V2_DIR SLICES_DIR WIDTHS V2PACK_DIR SOURCES_ROOT RANK
#include "g6pack.h"
#include "../loader-rs/g6load.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

struct HashSink
{
    std::mutex mu;
    std::map<uint64_t, std::pair<size_t, size_t>> chunks;   // offset -> (bytes, hash)
    static void* alloc(void*, size_t n) { return aligned_alloc(4096, (n + 4095) / 4096 * 4096); }
    static void release(void*, void* p) { free(p); }
    static int copy(void* ctx, int, uint64_t off, const void* src, size_t n)
    {
        auto& s = *(HashSink*)ctx;
        const size_t h = std::hash<std::string_view>{}(std::string_view((const char*)src, n));
        std::lock_guard<std::mutex> lk(s.mu);
        s.chunks[off] = {n, h};
        return 0;
    }
};

static bool open(g6::Pack& p, const std::string& m)
{
    std::string err;
    if (!p.open_manifest(m, err)) { printf("FAIL %s: %s\n", m.c_str(), err.c_str()); return false; }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 8) { fprintf(stderr, "usage: test_g6v2plan BASE_DIR V2_DIR SLICES_DIR WIDTHS V2PACK_DIR SOURCES_ROOT RANK\n"); return 2; }
    const std::string base = argv[1], v2 = argv[2], slices = argv[3], widths = argv[4], ref = argv[5], root = argv[6];
    const std::string r = argv[7], planned = "/tmp/test_g6v2plan.rank" + r + ".manifest.json";
    char err[1024] = "";
    void* h = g6l_v2_plan((base + "/rank" + r + ".manifest.json").c_str(), (base + "/rank" + r + ".pack").c_str(), v2.c_str(),
                          slices.c_str(), widths.c_str(), planned.c_str(), err, sizeof err);
    if (!h) { printf("FAIL plan: %s\n", err); return 1; }
    g6::Pack a, b;
    if (!open(a, planned) || !open(b, ref + "/rank" + r + ".manifest.json")) return 1;
    int fails = 0;
    if (a.bytes() != b.bytes()) { printf("FAIL image bytes %llu, want %llu\n", (unsigned long long)a.bytes(), (unsigned long long)b.bytes()); ++fails; }
    if (a.entries().size() != b.entries().size()) { printf("FAIL %zu tensors, want %zu\n", a.entries().size(), b.entries().size()); ++fails; }
    for (size_t i = 0; i < std::min(a.entries().size(), b.entries().size()) && fails < 10; ++i)
    {
        const g6::Entry &x = a.entries()[i], &y = b.entries()[i];
        if (x.name != y.name || x.offset != y.offset || x.nbytes != y.nbytes || x.dtype != y.dtype || x.shape != y.shape
            || x.expert_format != y.expert_format)
        {
            printf("FAIL tensor %zu: %s at %llu (%llu bytes), want %s at %llu (%llu bytes)\n", i, x.name.c_str(),
                   (unsigned long long)x.offset, (unsigned long long)x.nbytes, y.name.c_str(), (unsigned long long)y.offset,
                   (unsigned long long)y.nbytes);
            ++fails;
        }
    }
    if (fails) return 1;
    printf("manifest: %zu tensors, %.2f GB, identical\n", a.entries().size(), a.bytes() / 1e9);

    HashSink sa, sb;
    const g6l_sink ka{&sa, &HashSink::alloc, &HashSink::release, &HashSink::copy};
    const g6l_sink kb{&sb, &HashSink::alloc, &HashSink::release, &HashSink::copy};
    g6l_stats st{};
    if (g6l_plan_load(h, &ka, &st, err, sizeof err)) { printf("FAIL planned load: %s\n", err); return 1; }
    printf("planned image: %.2f GB read in %.1f s (%.2f GB/s), %llu ranges\n", st.read_bytes / 1e9, st.seconds,
           st.image_bytes / 1e9 / st.seconds, (unsigned long long)st.ranges);
    g6l_plan_free(h);
    std::vector<g6l_extent> ext;
    for (const g6::Entry& e : b.entries()) ext.push_back({e.offset, e.nbytes});
    if (g6l_load((ref + "/rank" + r + ".pack").c_str(), b.bytes(), ext.data(), ext.size(),
                 (ref + "/rank" + r + ".sources").c_str(), root.c_str(), &kb, &st, err, sizeof err))
    { printf("FAIL v2pack load: %s\n", err); return 1; }
    printf("v2pack image: %.2f GB read in %.1f s\n", st.read_bytes / 1e9, st.seconds);
    for (const auto& [off, c] : sa.chunks)
    {
        auto it = sb.chunks.find(off);
        if (it == sb.chunks.end() || it->second != c)
        {
            const g6::Entry* e = nullptr;
            for (const g6::Entry& x : a.entries()) if (x.offset <= off + c.first && off < x.offset + x.nbytes) { e = &x; break; }
            printf("FAIL chunk at %llu differs (first tensor there: %s)\n", (unsigned long long)off, e ? e->name.c_str() : "-");
            if (++fails >= 10) break;
        }
    }
    if (sa.chunks.size() != sb.chunks.size()) { printf("FAIL %zu chunks, want %zu\n", sa.chunks.size(), sb.chunks.size()); ++fails; }
    printf("%s (%zu chunks)\n", fails ? "test_g6v2plan FAILED" : "test_g6v2plan: identical to the v2pack image", sa.chunks.size());
    return fails ? 1 : 0;
}
