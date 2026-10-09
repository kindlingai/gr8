// The Rust pack loader through its C interface, on the host: load a pack into host memory with a memcpy sink, in both
// modes and at two thread counts, and compare every byte with the pack file.
//   cargo build --release --manifest-path engine/loader-rs/Cargo.toml
//   g++ -O2 -std=c++20 -pthread -I engine/loader -o test_g6load engine/tests/test_g6load.cpp engine/loader/g6pack.cpp
//       engine/loader-rs/target/release/libg6load.a -ldl -lm      (one command)
//   test_g6load DIR RANK...      (DIR/rank<r>.{manifest.json,pack}, e.g. from tools/convert/convert.py)
#include "g6pack.h"
#include "../loader-rs/g6load.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct HostSink
{
    std::vector<uint8_t> img;
    static void* alloc(void*, size_t n) { return aligned_alloc(4096, (n + 4095) / 4096 * 4096); }
    static void release(void*, void* p) { free(p); }
    static int copy(void* ctx, int, uint64_t off, const void* src, size_t n)
    {
        auto& s = *(HostSink*)ctx;
        if (off + n > s.img.size()) return 1;
        memcpy(s.img.data() + off, src, n);
        return 0;
    }
};

static std::vector<uint8_t> slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: test_g6load DIR RANK...\n"); return 2; }
    int fails = 0, runs = 0;
    for (int i = 2; i < argc; ++i)
    {
        const std::string r = argv[i], p = std::string(argv[1]) + "/rank" + r;
        const std::vector<uint8_t> want = slurp(p + ".pack");
        for (const char* mode : {"mmap", "read"})
            for (const char* threads : {"1", "3"})
            {
                setenv("G6_LOAD_MODE", mode, 1);
                setenv("G6_LOAD_THREADS", threads, 1);
                setenv("G6_LOAD_CHUNK_MB", "1", 1);
                g6::Pack pack;
                std::string err;
                if (!pack.open_manifest(p + ".manifest.json", err)) { printf("FAIL %s\n", err.c_str()); return 1; }
                std::vector<g6l_extent> ext;
                for (const g6::Entry& e : pack.entries()) ext.push_back({e.offset, e.nbytes});
                HostSink hs;
                hs.img.assign(pack.bytes(), 0xA5);   // not zero: every padding byte must be written
                const g6l_sink sink{&hs, &HostSink::alloc, &HostSink::release, &HostSink::copy};
                g6l_stats st{};
                char e2[512] = "";
                ++runs;
                if (g6l_load((p + ".pack").c_str(), pack.bytes(), ext.data(), ext.size(), "", "", &sink, &st, e2, sizeof e2))
                {
                    printf("rank %s %s x%s: FAIL %s\n", r.c_str(), mode, threads, e2);
                    ++fails;
                }
                else if (hs.img != want)
                {
                    size_t k = 0;
                    while (k < want.size() && hs.img[k] == want[k]) ++k;
                    printf("rank %s %s x%s: FAIL first difference at byte %zu\n", r.c_str(), mode, threads, k);
                    ++fails;
                }
            }
        printf("rank %s: %zu bytes checked in 4 configurations\n", r.c_str(), want.size());
    }
    printf("%s (%d runs)\n", fails ? "test_g6load FAILED" : "test_g6load: all identical to the pack", runs);
    return fails ? 1 : 0;
}
