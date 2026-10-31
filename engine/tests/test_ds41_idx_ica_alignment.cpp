// Host-only page-layout regression for the pipe-B uint4 code loads.
#include "../attn/ds41_idx_layout.h"
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <initializer_list>

int main() {
    constexpr int cap = 131072, pw = 4096, slots = 8;
    // Stand-in for zalloc -> dmalloc -> cudaMalloc's aligned allocation base.
    alignas(16) unsigned char slab_base[16] = {};
    const auto base = reinterpret_cast<std::uintptr_t>(slab_base);
    assert(base % 16 == 0);
    for (int ratio : {2, 1}) {
        const int nominal = ratio == 2 ? (cap + pw) / 2 + 2 : cap + pw;
        const int rows = ds41i::ica_aligned_rows(nominal);
        const int stride = ds41i::ica_page_bytes(rows);
        assert(rows >= nominal && rows - nominal < 4);
        assert(stride % 16 == 0);
        if (ratio == 2) assert((nominal * ds41i::ICA_ROW_BYTES) % 16 == 8); // pinned v7 regression
        else assert((nominal * ds41i::ICA_ROW_BYTES) % 16 == 0); // ratio-1 unchanged/aligned

        // The device allocation starts at a CUDA-aligned base. The code plane
        // uses 64 bytes per row; every uint4 at group g starts 16 bytes later.
        for (int slot = 0; slot < slots; ++slot)
            for (int tok = 0; tok < rows; ++tok)
                for (int g = 0; g < 4; ++g)
                    assert((base + slot * stride + tok * 64 + g * 16) % 16 == 0);
        std::printf("ICA_ALIGN_PASS ratio=%d nominal=%d rows=%d stride=%d extra_per_slot=%d\n",
                    ratio, nominal, rows, stride, stride - nominal * ds41i::ICA_ROW_BYTES);
    }
}
