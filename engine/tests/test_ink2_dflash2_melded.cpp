// Host unit tests for Inkling-2-Mini Melded DFlash2 safetensors header parsing, Q4_0 SoA pack/unpack
// bit-exactness, auto-discovery path precedence, and Inkling S2Cost defaults/overrides.
#include "../kernels/ink2_dflash2.cuh"
#include "../runtime/s2_batch.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static void test_q4_pack_unpack() {
    float src[32];
    for (int i = 0; i < 32; ++i) {
        src[i] = std::sin(static_cast<float>(i) * 0.37f) * 1.5f;
    }
    uint8_t qs[16] = {};
    uint16_t sc = 0;
    ink2::df2::q4_pack_block_host(src, qs, sc);
    float rec[32] = {};
    ink2::df2::q4_unpack_block_host(qs, sc, rec);

    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < 32; ++i) {
        const double d = static_cast<double>(src[i] - rec[i]);
        err2 += d * d;
        ref2 += static_cast<double>(src[i]) * static_cast<double>(src[i]);
    }
    const double nmse = err2 / ref2;
    assert(nmse < 0.02);

    // Zero block must encode to exact 0x88 nibbles and 0 scale
    float zeros[32] = {};
    uint8_t zqs[16] = {};
    uint16_t zsc = 1;
    ink2::df2::q4_pack_block_host(zeros, zqs, zsc);
    assert(zsc == 0);
    for (int j = 0; j < 16; ++j) assert(zqs[j] == 0x88);
    ink2::df2::q4_unpack_block_host(zqs, zsc, rec);
    for (int i = 0; i < 32; ++i) assert(rec[i] == 0.0f);
}

static void test_safetensors_parser_and_discovery() {
    const fs::path tmp = fs::temp_directory_path() / ("ink2_df2_melded_test_" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    // 1. Empty dir -> auto-discovery returns empty
    assert(ink2::df2::resolve_dflash2_path(tmp.string(), std::nullopt).empty());

    // 2. Fallback name dflash2.exlr8.safetensors discovered
    const fs::path p_alt = tmp / "dflash2.exlr8.safetensors";
    { std::ofstream f(p_alt, std::ios::binary); f << "x"; }
    assert(ink2::df2::resolve_dflash2_path(tmp.string(), std::nullopt) == p_alt.string());

    // 3. Preferred name dflash2-melded.safetensors takes precedence over fallback
    const fs::path p_main = tmp / "dflash2-melded.safetensors";
    const std::string hdr =
        R"({"__metadata__":{"format":"gr8-ink2-dflash2-melded-v1","source_license":"apache-2.0"},)"
        R"("fc.weight.qs":{"dtype":"U8","shape":[2304,4608],"data_offsets":[0,10616832]},)"
        R"("fc.weight.sc":{"dtype":"F16","shape":[2304,288],"data_offsets":[10616832,11943936]},)"
        R"("candidate_selector.predecessor_codebook.qs":{"dtype":"U8","shape":[200058,128],"data_offsets":[11943936,37551360]}})";
    {
        std::ofstream f(p_main, std::ios::binary);
        const uint64_t hlen = static_cast<uint64_t>(hdr.size());
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    }
    assert(ink2::df2::resolve_dflash2_path(tmp.string(), std::nullopt) == p_main.string());

    // 4. Explicit "0" or "off" disables auto-discovery
    assert(ink2::df2::resolve_dflash2_path(tmp.string(), std::string("0")).empty());
    assert(ink2::df2::resolve_dflash2_path(tmp.string(), std::string("off")).empty());

    // 5. Explicit path overrides auto-discovery
    assert(ink2::df2::resolve_dflash2_path(tmp.string(), std::string("/custom/df2.safetensors")) == "/custom/df2.safetensors");

    // 6. Verify parse_safetensors_header parses melded Q4_0 entries cleanly
    std::unordered_map<std::string, ink2::df2::StTensor> tmap;
    size_t data_base = 0;
    const bool ok = ink2::df2::parse_safetensors_header(p_main.string(), tmap, data_base);
    assert(ok);
    assert(data_base == 8 + hdr.size());
    assert(tmap.size() == 3);
    assert(tmap["fc.weight.qs"].dtype == "U8");
    assert(tmap["fc.weight.qs"].shape.size() == 2 && tmap["fc.weight.qs"].shape[0] == 2304 && tmap["fc.weight.qs"].shape[1] == 4608);
    assert(tmap["fc.weight.qs"].off == 0 && tmap["fc.weight.qs"].len == 10616832);
    assert(tmap["fc.weight.sc"].dtype == "F16");
    assert(tmap["candidate_selector.predecessor_codebook.qs"].shape[0] == 200058);

    fs::remove_all(tmp);
}

static void test_ink2_s2_cost() {
    s2::S2Cost c;
    assert(std::fabs(c.main0 - 17.0) < 1e-9);
    assert(std::fabs(c.main_row - 2.2) < 1e-9);
    assert(std::fabs(c.draft0 - 2.2) < 1e-9);
    assert(std::fabs(c.draft_sess - 0.35) < 1e-9);
    assert(std::fabs(c.chain0 - 0.0) < 1e-9);
    assert(std::fabs(c.chain_sess - 0.0) < 1e-9);
}

int main() {
    knob::init();
    test_q4_pack_unpack();
    test_safetensors_parser_and_discovery();
    test_ink2_s2_cost();
    std::printf("test_ink2_dflash2_melded: PASS\n");
    return 0;
}
