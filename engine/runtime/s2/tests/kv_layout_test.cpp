// The page block's sizes (kvman-rs src/layout.rs, g6kv_layout_sizes) through the C header, from a page of every binding
// as PageBlocks (s2/kv_tier.cuh) and FlashCheckpoints (s2/flash_checkpoint.cuh) describe theirs with this build's rows
// (attn/kvfmt.h): the offsets and the tail's sizes of a small layout, and the model's block as the stores have kept it
// (GLM 5.3: 79 latent and 22 index caches, 3,502,592 B, 2,289,152 B with G6_KV4; Flash: 12 bindings, 13,824 B a token,
// 4,992 B with G6_KV4). No CUDA.
#include "../../../attn/kvfmt.h"
#include "../../../kvman-rs/g6kvman.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
}

int main() {
    // an indexed CP binding, a plain CP binding, a replicated one with keys
    const std::vector<g6kv_binding> b = {{4096, 512, 1}, {4096, 0, 1}, {1024, 256, 0}};
    g6kv_layout s{};
    std::vector<uint64_t> off(b.size(), 9), rep(b.size(), 9);
    g6kv_layout_sizes(b.data(), b.size(), &s, off.data(), rep.data());
    require(s.block == 9984 && s.slot == 12288 && s.rep == 1280 && s.per_slot == 9, "a block, its slot and its replicated records");
    require(off == std::vector<uint64_t>({0, 4608, 8704}) && rep == std::vector<uint64_t>({0, 0, 0}), "each binding's offsets");
    g6kv_layout none{1, 1, 1, 1};
    g6kv_layout_sizes(nullptr, 0, &none, nullptr, nullptr);
    require(!none.block && !none.slot && !none.rep && !none.per_slot, "no bindings, no offsets wanted");
#if !G6_MODEL_DS41   // DS41's rows are not on pool pages yet (docs/HOST-CORE.md R11)
    std::vector<g6kv_binding> m;
#if G6_MODEL_FLASH
    for (int i = 0; i < 12; ++i) m.push_back({uint32_t(64 * g6kv::ROW), uint32_t(g6kv::IDX_PAGE), 0u});
    const uint64_t block = 64 * (G6_KV4 ? 4992 : G6_FLASH_KV_FP8 ? 7872 : 13824);
#else
    for (int i = 0; i < 79; ++i) m.push_back({uint32_t(64 * g6kv::ROW), i < 22 ? uint32_t(64 * g6kv::IDX_ROW) : 0u, 0u});
    const uint64_t block = G6_KV4 ? 2289152 : 3502592;
#endif
    g6kv_layout_sizes(m.data(), m.size(), &s, nullptr, nullptr);
    require(s.block == block && s.slot == (block + 4095) / 4096 * 4096 && !s.per_slot, "this build's page block");
    printf("kv_layout_test: %s page block %llu B, slot %llu B\n", g6kv::NAME, (unsigned long long)s.block, (unsigned long long)s.slot);
#endif
    puts("kv_layout_test: PASS");
    return 0;
}
