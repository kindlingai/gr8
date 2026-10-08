// g6::Sources (sparse packs, convert.py --v2-refs): load a sparse pack + rank<r>.sources with Pack::load (which checks
// every tensor's sha256 and every padding byte) and compare the image with the full pack, byte for byte. Host only.
//   g++ -O2 -std=c++20 -pthread -I engine/loader -o test_sources engine/tests/test_sources.cpp engine/loader/g6pack.cpp
//   test_sources DIR_SPARSE DIR_FULL RANK...      (packs from tools/convert/test_sources.py)
#include "g6pack.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: test_sources DIR_SPARSE DIR_FULL RANK...\n"); return 2; }
    int fails = 0;
    for (int i = 3; i < argc; ++i)
    {
        const std::string r = argv[i], sp = std::string(argv[1]) + "/rank" + r, fu = std::string(argv[2]) + "/rank" + r;
        g6::Pack pack;
        g6::Sources src;
        g6::LoadOpts lo;
        g6::LoadStats st;
        std::string err;
        lo.direct_io = false;
        lo.chunk = 1 << 16;   // small chunks: halves and ranges cross many chunk boundaries
        if (!pack.open_manifest(sp + ".manifest.json", err) || !src.open(sp + ".sources", "", pack.bytes(), err) ||
            !pack.load(sp + ".pack", lo, st, err, &src))
        {
            printf("rank %s: FAIL %s\n", r.c_str(), err.c_str());
            ++fails;
            continue;
        }
        std::ifstream f(fu + ".pack", std::ios::binary);
        std::vector<char> want((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (want.size() != pack.bytes() || memcmp(want.data(), pack.base(), want.size()))
        {
            printf("rank %s: FAIL image differs from the full pack\n", r.c_str());
            ++fails;
            continue;
        }
        printf("rank %s: OK %llu bytes, %zu source ranges (%llu bytes), sha256-checked, identical to the full pack\n",
               r.c_str(), (unsigned long long)pack.bytes(), src.ranges(), (unsigned long long)src.bytes());
    }
    printf("%s\n", fails ? "test_sources FAILED" : "test_sources: all ranks OK");
    return fails ? 1 : 0;
}
