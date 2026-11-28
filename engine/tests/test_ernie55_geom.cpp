// CPU host unit test for ERNIE 5.5 Flash geometry constants and layer indexing.
#include "../model_geom_ernie55.h"
#include <cassert>
#include <cstdio>

int main() {
    using namespace ernie55::geom;
    static_assert(VOCAB == 201216);
    static_assert(HID == 4096);
    static_assert(NLAYERS == 43);
    static_assert(N_MOE_LAYERS == 42);
    static_assert(N_DSA_LAYERS == 6);
    static_assert(N_HCA_LAYERS == 37);
    static_assert(HC == 4 && HC_DIM == 16384 && HC_MIX == 24 && HC_ITERS == 20);
    static_assert(N_ROUTED == 512 && TOP_K == 10 && MOE_LATENT == 2048 && MOE_INTER == 2048);
    static_assert(HCA_NOPE + HCA_ROPE == HCA_HD);
    static_assert(HCA_HEADS == HCA_O_GROUPS * HCA_HPG);
    static_assert(DSA_NOPE + DSA_ROPE == DSA_QK_DIM);
    static_assert(DSA_ATTN_TOPK == 2176);
    // Exact trellis byte sizes per expert (gate + up + down):
    // K2 routed = 3 * 2048 * 2048 * 2 / 8 = 3,145,728 B (3.00 MiB)
    static_assert(routed_expert_bytes(2) == 3ull * 1024ull * 1024ull);
    static_assert(routed_expert_bytes(3) == 9ull * 512ull * 1024ull);
    static_assert(routed_expert_bytes(4) == 6ull * 1024ull * 1024ull);
    // Shared K4 = 3 * 4096 * 2048 * 4 / 8 = 12,582,912 B (12.00 MiB)
    static_assert(shared_expert_bytes(4) == 12ull * 1024ull * 1024ull);

    int hca_cnt = 0, dsa_cnt = 0;
    for (int l = 0; l < NLAYERS; ++l) {
        if (is_dsa_layer(l)) {
            assert(compress_ratio(l) == -2);
            assert(dsa_slot_of(l) == dsa_cnt++);
            assert(hca_slot_of(l) == -1);
        } else {
            assert(compress_ratio(l) == 128);
            assert(hca_slot_of(l) == hca_cnt++);
            assert(dsa_slot_of(l) == -1);
        }
    }
    assert(hca_cnt == N_HCA_LAYERS);
    assert(dsa_cnt == N_DSA_LAYERS);
    std::printf("test_ernie55_geom: PASS (layers=%d, hca=%d, dsa=%d, k2_exp=%zu B)\n",
                NLAYERS, hca_cnt, dsa_cnt, routed_expert_bytes(2));
    return 0;
}
