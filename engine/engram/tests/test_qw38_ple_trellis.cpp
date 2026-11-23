// Host unit tests for Qwen3.8 Layer-1 PLE exlr8 FORMAT v4 Record 9 (Row Trellis) reader, decoder, and PleStore.
// Verifies:
//   1. Official FORMAT.md pmcgx / pmcgx-f16 state vectors (bit-exact FP16 & E4M3-rounded FP16).
//   2. Official FORMAT.md tile SHA-256 vectors for byte j = (37*j + 11) % 256 across all pmcgx widths K0.5..K5
//      and pmcgx-f16 K5.5..K6.
//   3. Row-trellis decode W[r, c] = suh[c] * (H128 v_r)[c] * svh[r] against an independent bit-by-bit reference.
//   4. PleStore::open + fetch_trellis_rows for standalone rowtrellis and exlr8-v4 safetensors in both NVMe+LRU
//      (with in-batch deduplication & LBA sort) and RAM-resident mmap modes, plus invalid header rejection and
//      raw BF16 .bin regression coverage.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qw38_ple.h"
#include "sha256.h"

namespace fs = std::filesystem;

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

// Minimal safetensors writer for tests.
struct TestTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

static void write_safetensors(const std::string& path, const std::vector<TestTensor>& tensors,
                              const std::vector<std::pair<std::string, std::string>>& meta = {}) {
    std::string hdr = "{";
    bool first = true;
    if (!meta.empty()) {
        hdr += "\"__metadata__\":{";
        for (size_t i = 0; i < meta.size(); ++i) {
            if (i) hdr += ",";
            hdr += "\"" + meta[i].first + "\":\"" + meta[i].second + "\"";
        }
        hdr += "}";
        first = false;
    }
    uint64_t off = 0;
    for (const auto& t : tensors) {
        if (!first) hdr += ",";
        first = false;
        hdr += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i) hdr += ",";
            hdr += std::to_string(t.shape[i]);
        }
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) + "]}";
        off += t.bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8 != 0) hdr += ' ';

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.good());
    const uint64_t hlen = hdr.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hdr.data(), std::streamsize(hdr.size()));
    for (const auto& t : tensors) {
        if (!t.bytes.empty()) f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    }
    CHECK(f.good());
}

static std::vector<uint8_t> to_bytes_i64(const int64_t* p, size_t n) {
    std::vector<uint8_t> b(n * 8);
    std::memcpy(b.data(), p, n * 8);
    return b;
}

static std::vector<uint8_t> to_bytes_u16(const uint16_t* p, size_t n) {
    std::vector<uint8_t> b(n * 2);
    std::memcpy(b.data(), p, n * 2);
    return b;
}

// 1. Official FORMAT.md pmcgx / pmcgx-f16 state vectors.
static void test_official_states() {
    struct Vec { uint32_t st; uint16_t f16[2]; uint16_t e4[2]; };
    const Vec vecs[] = {
        {0u,       {0x3f60, 0x3f60}, {0x3f80, 0x3f80}},
        {1u,       {0xb268, 0x34cb}, {0xb280, 0x3500}},
        {74565u,   {0x3e6d, 0xc0b4}, {0x3e80, 0xc080}},
        {524287u,  {0xc120, 0x1e00}, {0xc100, 0x1e00}},
        {524288u,  {0x3ea4, 0x395c}, {0x3e80, 0x3980}},
        {782069u,  {0xbd2e, 0x4070}, {0xbd00, 0x4080}},
        {1048575u, {0xb9b0, 0xac60}, {0xb980, 0xac80}},
    };
    for (const auto& v : vecs) {
        uint16_t h0 = 0, h1 = 0, e0 = 0, e1 = 0;
        qw38::ple_pmcgx_cpu(v.st, false, h0, h1);
        qw38::ple_pmcgx_cpu(v.st, true, e0, e1);
        CHECK(h0 == v.f16[0] && h1 == v.f16[1]);
        CHECK(e0 == v.e4[0] && e1 == v.e4[1]);
    }
}

// 2. Official FORMAT.md tile SHA-256 vectors (with tile_perm applied as in exlr8.py::decode_tiles).
static void test_official_tile_hashes() {
    int perm[256];
    int idx = 0;
    for (int t = 0; t < 32; ++t) {
        const int r0 = 2 * (t % 4), c0 = t / 4;
        const int rc[8][2] = {
            {r0, c0}, {r0 + 1, c0}, {r0 + 8, c0}, {r0 + 9, c0},
            {r0, c0 + 8}, {r0 + 1, c0 + 8}, {r0 + 8, c0 + 8}, {r0 + 9, c0 + 8}
        };
        for (int k = 0; k < 8; ++k) perm[idx++] = 16 * rc[k][0] + rc[k][1];
    }

    struct TV { int k_bits; bool e4; const char* sha16; };
    const TV tvs[] = {
        {1,  true,  "dd6bc537b173f3e8"}, // pmcgx K0.5
        {2,  true,  "f0c40ae8fe7f23d1"}, // pmcgx K1
        {3,  true,  "36a1897a31ccd5d4"}, // pmcgx K1.5
        {4,  true,  "d5b6cc4ed594469f"}, // pmcgx K2
        {5,  true,  "1428ff10cbc4044b"}, // pmcgx K2.5
        {6,  true,  "d4005af0b8591dc1"}, // pmcgx K3
        {7,  true,  "b9898c3e54ae2d7e"}, // pmcgx K3.5
        {8,  true,  "e407108ab19f224c"}, // pmcgx K4
        {9,  true,  "143277248d13e27c"}, // pmcgx K4.5
        {10, true,  "c037dde67a1f5de4"}, // pmcgx K5
        {11, false, "8d06db3bf149c785"}, // pmcgx-f16 K5.5
        {12, false, "b888e32d22699015"}, // pmcgx-f16 K6
    };
    for (const auto& tv : tvs) {
        const size_t n = size_t(16 * tv.k_bits);
        std::vector<uint8_t> raw(n);
        for (size_t j = 0; j < n; ++j) raw[j] = uint8_t((j * 37 + 11) % 256);
        float stream_vals[256];
        qw38::ple_decode_tile_stream_cpu(raw.data(), tv.k_bits, tv.e4, stream_vals);
        float tile_vals[256];
        for (int j = 0; j < 256; ++j) tile_vals[perm[j]] = stream_vals[j];
        ds41::engram::Sha256 sha;
        sha.update(tile_vals, sizeof(tile_vals));
        const std::string h = sha.hex();
        CHECK(h.substr(0, 16) == tv.sha16);
    }
}

// Independent bit-by-bit reference matching FORMAT.md wording directly.
static void decode_row_spec(const uint8_t* row, uint16_t sv_h, const uint16_t* su_h,
                            int nc, int out_cols, int k_bits, bool e4, std::vector<uint16_t>& out_bf16) {
    const int n_tiles = nc / 256;
    const int tile_bytes = 16 * k_bits;
    const int nb = 128 * k_bits;
    const float sv = qw38::ple_fp16_to_f32(sv_h);
    out_bf16.resize(size_t(out_cols));

    // Explicit Sylvester Hadamard matrix H_128 / sqrt(128)
    std::vector<float> H(128 * 128, 1.0f);
    for (int sz = 1; sz < 128; sz <<= 1) {
        for (int r = 0; r < sz; ++r) {
            for (int c = 0; c < sz; ++c) {
                const float v = H[r * 128 + c];
                H[r * 128 + (c + sz)] = v;
                H[(r + sz) * 128 + c] = v;
                H[(r + sz) * 128 + (c + sz)] = -v;
            }
        }
    }
    const float inv_s = 1.0f / std::sqrt(128.0f);

    for (int t = 0; t < n_tiles; ++t) {
        const uint8_t* raw = row + size_t(t) * size_t(tile_bytes);
        // Unpack u16 words, swap each adjacent pair, MSB first
        const int nw = tile_bytes / 2;
        std::vector<uint16_t> words(static_cast<size_t>(nw));
        for (int i = 0; i < nw; ++i) words[size_t(i)] = uint16_t(raw[2 * i] | (uint16_t(raw[2 * i + 1]) << 8));
        std::vector<uint8_t> bits;
        bits.reserve(size_t(nb));
        for (int i = 0; i < nw; ++i) {
            const uint16_t w = words[size_t(i ^ 1)];
            for (int b = 15; b >= 0; --b) bits.push_back(uint8_t((w >> b) & 1));
        }
        float v[256];
        for (int p = 0; p < 128; ++p) {
            const int end = (p + 1) * k_bits;
            uint32_t st = 0;
            for (int i = end - 20; i < end; ++i) {
                const int bi = ((i % nb) + nb) % nb;
                st = (st << 1) | bits[size_t(bi)];
            }
            uint16_t h0 = 0, h1 = 0;
            qw38::ple_pmcgx_cpu(st, e4, h0, h1);
            v[2 * p] = qw38::ple_fp16_to_f32(h0);
            v[2 * p + 1] = qw38::ple_fp16_to_f32(h1);
        }
        for (int blk = 0; blk < 2; ++blk) {
            for (int c = 0; c < 128; ++c) {
                const int col = t * 256 + blk * 128 + c;
                if (col >= out_cols) continue;
                double acc = 0.0;
                for (int k = 0; k < 128; ++k) acc += double(v[blk * 128 + k]) * double(H[k * 128 + c]);
                const float val = float(acc * inv_s) * qw38::ple_fp16_to_f32(su_h[col]) * sv;
                out_bf16[size_t(col)] = qw38::ple_f32_to_bf16(val);
            }
        }
    }
}

static void test_store_roundtrip(const fs::path& dir) {
    struct Case {
        const char* name;
        std::string prefix;
        int64_t k_rec;
        int k_bits;
        int nc;
        int out_cols;
        std::string cb;
        bool e4;
        bool v4_meta;
        bool resident;
    };
    const Case cases[] = {
        {"k4_standalone_nvme.safetensors",  "",     4,  8,  256, 160, "pmcgx",     true,  false, false},
        {"k4p5_standalone_nvme.safetensors","",    -9,  9,  256, 160, "pmcgx",     true,  false, false},
        {"k5_standalone_res.safetensors",   "",     5, 10,  256, 160, "pmcgx",     true,  false, true},
        {"k5_v4_f16_2tile.safetensors",     "ple.", 5, 10,  512, 512, "pmcgx-f16", false, true,  false},
        {"k4p5_v4_res.safetensors",         "ple.",-9,  9,  256, 256, "pmcgx",     true,  true,  true},
    };

    uint64_t rng = 0xCAFEBABE12345678ull;
    auto next_u32 = [&]() {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        return uint32_t((rng * 0x2545F4914F6CDD1Dull) >> 32);
    };

    for (const auto& tc : cases) {
        const int nr = 300; // crosses multiple 4 KiB pages
        const size_t row_bytes = size_t(tc.nc / 256) * size_t(16 * tc.k_bits);
        std::vector<uint8_t> codes(size_t(nr) * row_bytes);
        for (auto& b : codes) b = uint8_t(next_u32() & 0xFF);
        std::vector<uint16_t> suh(static_cast<size_t>(tc.nc));
        std::vector<uint16_t> svh(static_cast<size_t>(nr));
        for (int i = 0; i < tc.nc; ++i)
            suh[size_t(i)] = qw38::ple_f32_to_fp16(((next_u32() & 1) ? 1.0f : -1.0f) * (0.25f + (next_u32() & 255) / 200.0f));
        for (int i = 0; i < nr; ++i)
            svh[size_t(i)] = qw38::ple_f32_to_fp16((int(next_u32() & 511) - 256) / 128.0f);

        const int64_t rec[8] = {9, nr, tc.nc, tc.k_rec, 1, 1, 0, 0};
        std::vector<TestTensor> ts = {
            {tc.prefix + "record", "I64", {8}, to_bytes_i64(rec, 8)},
            {tc.prefix + "rows." + tc.cb, "U8", {nr, int64_t(row_bytes)}, codes},
            {tc.prefix + "suh", "F16", {tc.nc}, to_bytes_u16(suh.data(), suh.size())},
            {tc.prefix + "svh", "F16", {nr}, to_bytes_u16(svh.data(), svh.size())},
        };
        std::vector<std::pair<std::string, std::string>> meta;
        if (tc.v4_meta) {
            meta.push_back({"format", "exlr8-v4"});
            meta.push_back({"codebooks", tc.cb + " V2 L20 0xCBAC1FED 0xBCA14C51"});
        } else {
            meta.push_back({"format", "exlr8-row-trellis"});
        }
        const std::string path = (dir / tc.name).string();
        write_safetensors(path, ts, meta);

        qw38::PleStore store;
        qw38::PleStoreOpts opts;
        opts.cache_bytes = tc.resident ? (1ull << 20) : (16ull << 10); // 1 MiB (>= file) vs 16 KiB (< file)
        opts.ways = 4;
        opts.io_uring = 16;
        opts.direct = false; // tmpfs in CI
        std::string err;
        CHECK(store.open(path, opts, &err));
        CHECK(store.is_row_trellis());
        CHECK(store.is_resident() == tc.resident);
        CHECK(store.total_rows() == uint64_t(nr));
        CHECK(store.row_bytes() == row_bytes);
        CHECK(store.trellis_info().k_bits == tc.k_bits);
        CHECK(store.trellis_info().e4m3 == tc.e4);
        CHECK(store.suh() == suh);

        // Fetch batch with out-of-order IDs, duplicates (dedup!), and one out-of-range ID
        const std::vector<uint64_t> ids = {299, 0, 17, 299, 128, 17, 42, 300, 127, 255};
        std::vector<uint8_t> got_codes(ids.size() * row_bytes, 0xCC);
        std::vector<uint16_t> got_svh(ids.size(), 0xBEEF);
        qw38::PleStats st{};
        CHECK(store.fetch_trellis_rows(ids.data(), ids.size(), got_codes.data(), got_svh.data(), &st));
        if (!tc.resident) CHECK(st.reads == 7); // 7 unique in-range IDs!

        // Fetch again -> 100% cache hits!
        qw38::PleStats st2{};
        CHECK(store.fetch_trellis_rows(ids.data(), ids.size(), got_codes.data(), got_svh.data(), &st2));
        CHECK(st2.reads == 0);

        // Decode fetched rows with ple_rowtrellis_decode_cpu and compare against independent spec reference!
        std::vector<uint16_t> got_bf16(ids.size() * size_t(tc.out_cols));
        qw38::ple_rowtrellis_decode_cpu(got_codes.data(), got_svh.data(), store.suh().data(),
                                        got_bf16.data(), int(ids.size()), tc.nc, tc.out_cols, tc.k_bits, tc.e4);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= uint64_t(nr)) {
                CHECK(got_svh[i] == 0);
                for (size_t b = 0; b < row_bytes; ++b) CHECK(got_codes[i * row_bytes + b] == 0);
                continue;
            }
            CHECK(got_svh[i] == svh[ids[i]]);
            CHECK(std::memcmp(got_codes.data() + i * row_bytes, codes.data() + ids[i] * row_bytes, row_bytes) == 0);
            std::vector<uint16_t> ref_row;
            decode_row_spec(codes.data() + ids[i] * row_bytes, svh[ids[i]], suh.data(),
                            tc.nc, tc.out_cols, tc.k_bits, tc.e4, ref_row);
            for (int c = 0; c < tc.out_cols; ++c) {
                CHECK(got_bf16[i * size_t(tc.out_cols) + size_t(c)] == ref_row[size_t(c)]);
            }
        }
    }
}

static void test_invalid_rejected(const fs::path& dir) {
    const int64_t good_rec[8] = {9, 4, 256, 4, 1, 1, 0, 0};
    const std::vector<uint8_t> codes(4 * 128, 0);
    const std::vector<uint16_t> suh(256, 0x3C00), svh(4, 0x3C00);

    auto must_fail = [&](const std::string& name, const std::vector<TestTensor>& ts,
                         const std::vector<std::pair<std::string, std::string>>& meta) {
        const std::string p = (dir / name).string();
        write_safetensors(p, ts, meta);
        qw38::PleStore s;
        qw38::PleStoreOpts o;
        o.direct = false;
        o.cache_bytes = 0;
        std::string err;
        CHECK(!s.open(p, o, &err));
        CHECK(!err.empty());
    };

    // Wrong record format (1 instead of 9)
    {
        int64_t r[8] = {1, 4, 256, 4, 1, 1, 0, 0};
        must_fail("bad_fmt.safetensors", {
            {"record", "I64", {8}, to_bytes_i64(r, 8)},
            {"rows.pmcgx", "U8", {4, 128}, codes},
            {"suh", "F16", {256}, to_bytes_u16(suh.data(), suh.size())},
            {"svh", "F16", {4}, to_bytes_u16(svh.data(), svh.size())},
        }, {});
    }
    // Unsupported codebook
    must_fail("bad_cb.safetensors", {
        {"record", "I64", {8}, to_bytes_i64(good_rec, 8)},
        {"rows.mcg", "U8", {4, 128}, codes},
        {"suh", "F16", {256}, to_bytes_u16(suh.data(), suh.size())},
        {"svh", "F16", {4}, to_bytes_u16(svh.data(), svh.size())},
    }, {});
    // pmcgx K6 exceeds max K5
    {
        int64_t r[8] = {9, 4, 256, 6, 1, 1, 0, 0};
        std::vector<uint8_t> c6(4 * 192, 0);
        must_fail("bad_k6.safetensors", {
            {"record", "I64", {8}, to_bytes_i64(r, 8)},
            {"rows.pmcgx", "U8", {4, 192}, c6},
            {"suh", "F16", {256}, to_bytes_u16(suh.data(), suh.size())},
            {"svh", "F16", {4}, to_bytes_u16(svh.data(), svh.size())},
        }, {});
    }
    // exlr8-v4 missing codebooks metadata
    must_fail("bad_v4_meta.safetensors", {
        {"record", "I64", {8}, to_bytes_i64(good_rec, 8)},
        {"rows.pmcgx", "U8", {4, 128}, codes},
        {"suh", "F16", {256}, to_bytes_u16(suh.data(), suh.size())},
        {"svh", "F16", {4}, to_bytes_u16(svh.data(), svh.size())},
    }, {{"format", "exlr8-v4"}});
    // nc not multiple of 256
    {
        int64_t r[8] = {9, 4, 160, 4, 1, 1, 0, 0};
        must_fail("bad_nc.safetensors", {
            {"record", "I64", {8}, to_bytes_i64(r, 8)},
            {"rows.pmcgx", "U8", {4, 128}, codes},
            {"suh", "F16", {256}, to_bytes_u16(suh.data(), suh.size())},
            {"svh", "F16", {4}, to_bytes_u16(svh.data(), svh.size())},
        }, {});
    }
}

static void test_raw_bin_and_resolve(const fs::path& dir) {
    fs::path rel = dir / "rel";
    fs::create_directories(rel / "ple");

    // 1. Auto-detect row-trellis when raw .bin is absent
    const std::string k4p5 = (rel / "ple" / "exlr8-ple-L001.k4p5.safetensors").string();
    { std::ofstream f(k4p5); f << "x"; }
    CHECK(qw38::resolve_ple_path(rel.string()) == k4p5);

    // 2. Prefer raw .bin when both exist and no override is given
    const std::string raw = (rel / "ple" / "exlr8-ple-L001.bin").string();
    {
        std::vector<uint8_t> data(32 * qw38::PLE_ROW_BYTES);
        for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t((i * 17 + 3) & 0xFF);
        std::ofstream f(raw, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    }
    CHECK(qw38::resolve_ple_path(rel.string()) == raw);
    // Explicit override picks k4p5 even when raw .bin exists
    CHECK(qw38::resolve_ple_path(rel.string(), "exlr8-ple-L001.k4p5.safetensors") == k4p5);

    // 3. Raw .bin PleStore still fetches exact bytes with dedup
    qw38::PleStore s;
    qw38::PleStoreOpts o;
    o.direct = false;
    o.cache_bytes = 64 << 10;
    o.ways = 4;
    std::string err;
    CHECK(s.open(raw, o, &err));
    CHECK(!s.is_row_trellis());
    CHECK(s.total_rows() == 32);
    const uint64_t ids[4] = {31, 0, 31, 5};
    std::vector<uint8_t> out(4 * qw38::PLE_ROW_BYTES);
    qw38::PleStats st{};
    CHECK(s.fetch_rows(ids, 4, out.data(), &st));
    CHECK(st.reads == 3);
    for (int i = 0; i < 4; ++i) {
        const size_t base = size_t(ids[i]) * qw38::PLE_ROW_BYTES;
        for (size_t b = 0; b < qw38::PLE_ROW_BYTES; ++b) {
            CHECK(out[size_t(i) * qw38::PLE_ROW_BYTES + b] == uint8_t(((base + b) * 17 + 3) & 0xFF));
        }
    }
}

int main(int argc, char** argv) {
    const fs::path tmp = argc > 1 ? fs::path(argv[1]) : (fs::temp_directory_path() / "test_qw38_ple_trellis");
    fs::create_directories(tmp);
    test_official_states();
    test_official_tile_hashes();
    test_store_roundtrip(tmp);
    test_invalid_rejected(tmp);
    test_raw_bin_and_resolve(tmp);
    std::printf("test_qw38_ple_trellis: ok (states, 12 tile hashes, K4/K4.5/K5 roundtrip, NVMe+resident, raw bin)\n");
    return 0;
}
