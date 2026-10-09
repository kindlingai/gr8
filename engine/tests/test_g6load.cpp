// The Rust pack loader through its C interface, on the host: load a sparse pack + rank<r>.sources (and the full pack
// itself) into host memory with a memcpy sink, in both modes and at several thread counts and chunk sizes, and compare
// every byte with the full pack.
//   cargo build --release --manifest-path engine/loader-rs/Cargo.toml
//   g++ -O2 -std=c++20 -pthread -I engine/loader -o test_g6load engine/tests/test_g6load.cpp engine/loader/g6pack.cpp
//       engine/loader-rs/target/release/libg6load.a -ldl -lm      (one command)
//   test_g6load DIR_SPARSE DIR_FULL RANK...      (packs from tools/convert/test_sources.py)
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
    if (argc < 4) { fprintf(stderr, "usage: test_g6load DIR_SPARSE DIR_FULL RANK...\n"); return 2; }
    int fails = 0, runs = 0;
    for (int i = 3; i < argc; ++i)
    {
        const std::string r = argv[i];
        const std::vector<uint8_t> want = slurp(std::string(argv[2]) + "/rank" + r + ".pack");
        for (const char* dir : {argv[1], argv[2]})   // the sparse pack with sources, then the full pack alone
            for (const char* mode : {"mmap", "read"})
                for (const char* threads : {"1", "3"})
                {
                    const std::string p = std::string(dir) + "/rank" + r;
                    const bool sparse = dir == argv[1];
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
                    const std::string src = sparse ? p + ".sources" : "";
                    ++runs;
                    if (g6l_load((p + ".pack").c_str(), pack.bytes(), ext.data(), ext.size(), src.c_str(), "", &sink, &st,
                                 e2, sizeof e2))
                    {
                        printf("rank %s %s %s x%s: FAIL %s\n", r.c_str(), sparse ? "sparse" : "full", mode, threads, e2);
                        ++fails;
                    }
                    else if (hs.img != want)
                    {
                        size_t k = 0;
                        while (k < want.size() && hs.img[k] == want[k]) ++k;
                        printf("rank %s %s %s x%s: FAIL first difference at byte %zu\n", r.c_str(), sparse ? "sparse" : "full",
                               mode, threads, k);
                        ++fails;
                    }
                }
        printf("rank %s: %zu bytes checked in 8 configurations\n", r.c_str(), want.size());
    }
    printf("%s (%d runs)\n", fails ? "test_g6load FAILED" : "test_g6load: all identical to the full pack", runs);
    return fails ? 1 : 0;
}
