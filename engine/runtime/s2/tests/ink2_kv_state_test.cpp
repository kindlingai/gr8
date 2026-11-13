// Host test of the Inkling-2-Small KvStateDesc (s2/kv_state_desc.h ink2_kv_state): the planes and extents of a rank
// with the model's layer pattern (every sixth layer full attention), and a round trip of a slot's pages and state
// blocks through the host pack / unpack into another slot.
#include "s2/kv_state_desc.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace g6s2;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

namespace {
constexpr uint32_t SLOTS = 3, CAP = 512, MAX_PAGES = CAP / 64, NK = 2 * 128, H = 4608, WINDOW = 1024, LAYERS = 48;

struct Buffers {
    std::vector<std::vector<uint8_t>> mem;
    std::vector<Ink2KvLayer> layers;
    uint8_t* make(size_t n, uint32_t salt) {
        mem.emplace_back(n);
        uint32_t x = salt * 2654435761u + 1;
        for (auto& b : mem.back()) { x = x * 1664525u + 1013904223u; b = uint8_t(x >> 24); }
        return mem.back().data();
    }
    Buffers() {
        const size_t page = size_t(2) * 64 * NK * 2;
        for (uint32_t l = 0; l < LAYERS; ++l) {
            Ink2KvLayer y;
            y.global = l % 6 == 5;
            if (y.global) {
                y.pages = make(size_t(SLOTS) * MAX_PAGES * page, l * 8 + 1);
                y.staging = make(size_t(SLOTS) * page, l * 8 + 2);
            } else y.ring = make(size_t(SLOTS) * 2 * WINDOW * NK * 2, l * 8 + 3);
            y.hist_k = make(size_t(SLOTS) * 3 * NK * 2, l * 8 + 4);
            y.hist_v = make(size_t(SLOTS) * 3 * NK * 2, l * 8 + 5);
            y.hist_a = make(size_t(SLOTS) * 3 * H * 2, l * 8 + 6);
            y.hist_m = make(size_t(SLOTS) * 3 * H * 2, l * 8 + 7);
            layers.push_back(y);
        }
    }
};

bool same_slot(const KvStateDesc& d, uint32_t a, uint32_t b, uint32_t pages) {
    for (const auto& p : d.page_planes)
        if (std::memcmp(p.base + a * p.slot_stride, p.base + b * p.slot_stride, size_t(pages) * p.page_bytes())) return false;
    for (const auto& t : d.tail_extents)
        if (std::memcmp(t.base + a * t.slot_stride, t.base + b * t.slot_stride, size_t(t.bytes))) return false;
    return true;
}
}   // namespace

int main() {
    Buffers b;
    const KvStateDesc d = ink2_kv_state(b.layers, SLOTS, MAX_PAGES, NK, H, WINDOW);
    // 8 full-attention layers: a page row is K and V of one token
    CHECK(d.page_planes.size() == 8);
    for (const auto& p : d.page_planes) CHECK(p.rows_per_page == 64 && p.token_ratio == 1 && p.row_bytes == 2 * NK * 2);
    CHECK(d.page_block_bytes() == size_t(8) * 64 * 2 * NK * 2);
    // 40 rings, 192 convolution histories, 8 staging pages
    CHECK(d.tail_extents.size() == 40 + 192 + 8);
    const size_t tail = size_t(40) * 2 * WINDOW * NK * 2 + size_t(96) * 3 * NK * 2 + size_t(96) * 3 * H * 2 + size_t(8) * 2 * 64 * NK * 2;
    CHECK(d.tail_bytes() == tail);
    CHECK(!d.allow_partial_pages && d.partial_page_bytes(37) == 0 && d.state_payload_bytes(37) == tail);
    CHECK(d.state_blocks() == (tail + d.state_payload_cap_per_block() - 1) / d.state_payload_cap_per_block());

    // slot 0 -> slot 2: the pages below pos and the state blocks
    const uint32_t pos = 389, pages = pos / 64;
    CHECK(!same_slot(d, 0, 2, pages));
    std::vector<uint8_t> block(d.slot_bytes());
    for (uint32_t p = 0; p < pages; ++p) {
        d.pack_page_host(pack_page_handle(0, p, MAX_PAGES), block.data());
        d.unpack_page_host(pack_page_handle(2, p, MAX_PAGES), block.data());
    }
    for (uint32_t sub = 0; sub < d.state_blocks(); ++sub) {
        d.pack_state_block_host(pack_state_handle(0, pos), sub, block.data());
        d.unpack_state_block_host(pack_state_handle(2, pos), sub, block.data());
    }
    CHECK(same_slot(d, 0, 2, pages));
    CHECK(!same_slot(d, 0, 1, pages));

    // a damaged block is refused
    d.pack_state_block_host(pack_state_handle(0, pos), 0, block.data());
    block[sizeof(KvStateHeader32) + 5] ^= 0x40;
    bool refused = false;
    try { d.unpack_state_block_host(pack_state_handle(1, pos), 0, block.data()); } catch (const std::exception&) { refused = true; }
    CHECK(refused);

    // another geometry is another format
    Buffers b2;
    CHECK(ink2_kv_state(b2.layers, SLOTS, MAX_PAGES, NK, H, WINDOW).format_hash() == d.format_hash());
    // TP=4 uneven KV heads (rank 2/3 nk=128 vs rank 0/1 max_nk=256): slot_bytes and state_blocks match rank 0/1
    const KvStateDesc d_small = ink2_kv_state(b2.layers, SLOTS, MAX_PAGES, NK / 2, H, WINDOW, NK);
    CHECK(d_small.slot_bytes() == d.slot_bytes() && d_small.state_blocks() == d.state_blocks());
    b2.layers.pop_back();
    CHECK(ink2_kv_state(b2.layers, SLOTS, MAX_PAGES, NK, H, WINDOW).format_hash() != d.format_hash());
    std::printf("ink2_kv_state_test: PASS (%zu page planes, %zu tail extents, %u state blocks of %zu bytes)\n", d.page_planes.size(),
                d.tail_extents.size(), d.state_blocks(), d.slot_bytes());
    return 0;
}
