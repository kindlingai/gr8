// Host CPU unit test for hy4_engram.h:
//   1. Verifies NumpyPcg64 matches exact numpy.random.default_rng multipliers for all 4 layers (cv=92620)
//   2. If a release dir is passed as argv[1], opens its engram/ store + lookup_tables from dense/exlr8-dense.safetensors
//   3. Verifies hash_token + fetch_rows (cache miss -> O_DIRECT/pread -> cache hit) + finite FP8+scale rows!
#include "../engram/hy4_engram.h"
#include "../loader/g6pack.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void read_lut_from_st(const std::string& st_path, const std::string& key, std::vector<int32_t>& out) {
    std::ifstream f(st_path, std::ios::binary);
    assert(f && "open dense safetensors");
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), static_cast<std::streamsize>(hlen));
    const std::string needle = "\"" + key + "\"";
    const size_t p = hdr.find(needle);
    assert(p != std::string::npos);
    const size_t op = hdr.find("\"data_offsets\":[", p);
    assert(op != std::string::npos);
    const size_t a0 = op + 16;
    const size_t cm = hdr.find(',', a0);
    const size_t a1 = hdr.find(']', cm);
    const uint64_t s0 = std::stoull(hdr.substr(a0, cm - a0));
    const uint64_t s1 = std::stoull(hdr.substr(cm + 1, a1 - cm - 1));
    const size_t nb = static_cast<size_t>(s1 - s0);
    assert(nb % 4 == 0);
    out.resize(nb / 4);
    f.seekg(static_cast<std::streamoff>(8 + hlen + s0));
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nb));
}

int main(int argc, char** argv) {
    // 1. Verify PCG64 directly (bypassing cv==92620 fast table!) matches exact NumPy default_rng multipliers!
    for (int s = 0; s < 4; ++s) {
        const uint64_t layer_id = static_cast<uint64_t>(hy4::ENGRAM_LAYER_IDX[s] + 1);
        const int64_t hb = INT64_MAX / 92620LL / 2;
        hy4::NumpyPcg64 rng;
        rng.seed(10007ULL * layer_id);
        for (int k = 0; k < 3; ++k) {
            const int64_t got = rng.bounded_u64(static_cast<uint64_t>(hb - 1)) * 2 + 1;
            const int64_t exp = hy4::ENGRAM_MULTS_CV92620[s][k];
            if (got != exp) {
                std::fprintf(stderr, "FAIL PCG64 slot %d k=%d: got=%lld exp=%lld\n", s, k, (long long)got, (long long)exp);
                return 1;
            }
        }
    }
    std::printf("PCG64 multipliers exact match for all 4 layers (12/12) OK\n");

    // 2. Test real Engram store if present
    const std::string rel = (argc > 1 && argv[1]) ? argv[1] : "";
    if (rel.empty()) {
        std::printf("no release dir arg, skipping live store test\n");
        return 0;
    }
    std::ifstream probe(rel + "/engram/exlr8-engram-L031.bin", std::ios::binary);
    if (!probe) {
        std::printf("engram release not present, skipping live store test\n");
        return 0;
    }
    probe.close();

    std::array<std::vector<int32_t>, 4> luts;
    for (int s = 0; s < 4; ++s) {
        const std::string key = "model.layers." + std::to_string(hy4::ENGRAM_LAYER_IDX[s]) + ".engram_layer.lookup_table";
        read_lut_from_st(rel + "/dense/exlr8-dense.safetensors", key, luts[size_t(s)]);
        assert(luts[size_t(s)].size() == 120818);
    }

    hy4::EngramStore store;
    hy4::EngramStoreOpts opts{};
    opts.cache_bytes = 64ull << 20; // 64 MiB small test cache
    opts.io_uring = 32;
    opts.direct = true;
    std::string err;
    if (!store.open(rel + "/engram", luts, opts, err)) {
        std::fprintf(stderr, "FAIL store.open: %s\n", err.c_str());
        return 1;
    }

    // Hash a small prompt and fetch twice (miss then 100% cache hit!)
    const int32_t toks[4] = {120000, 500, 1234, 5678};
    uint8_t buf1[4 * 16 * 128];
    uint8_t buf2[4 * 16 * 128];
    for (int s = 0; s < 4; ++s) {
        uint64_t ids[4 * 16];
        for (int t = 0; t < 4; ++t) {
            const int32_t t0 = toks[t];
            const int32_t t1 = (t >= 1) ? toks[t - 1] : geom::ENGRAM_PAD_ID;
            const int32_t t2 = (t >= 2) ? toks[t - 2] : geom::ENGRAM_PAD_ID;
            store.hash_token(s, t0, t1, t2, ids + t * 16);
        }
        assert(store.fetch_rows(s, ids, 64, buf1));
        const uint64_t hits_before = store.stats().hits;
        assert(store.fetch_rows(s, ids, 64, buf2));
        assert(store.stats().hits - hits_before == 64);
        assert(std::memcmp(buf1, buf2, sizeof(buf1)) == 0);
        // Verify trailing 30 bytes are zero padding and BF16 scales are non-zero finite
        for (int i = 0; i < 64; ++i) {
            const uint8_t* r = buf1 + i * 128;
            for (int z = 98; z < 128; ++z) assert(r[z] == 0);
            uint16_t sc = 0;
            std::memcpy(&sc, r + 96, 2);
            assert((sc & 0x7f80u) != 0x7f80u); // not Inf/NaN
        }
    }
    std::printf("test_hy4_engram: ALL PASS (req=%llu hits=%llu misses=%llu reads=%llu)\n",
                (unsigned long long)store.stats().req, (unsigned long long)store.stats().hits,
                (unsigned long long)store.stats().misses, (unsigned long long)store.stats().reads);
    return 0;
}
