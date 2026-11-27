// Host test of Tencent Hy4-flash geometry (engine/model_geom_hy4.h via model_geom.h, -DG6_MODEL_HY4=1).
// Checks every constant, layer predicate, slot bijection, and TP=1/2 sharding.
#include "model_geom.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if !G6_MODEL_HY4
#error "build with -DG6_MODEL_HY4=1"
#endif

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

int main() {
    using namespace geom;
    CHECK(HID == 3072 && HB == 24 && KSH == 192 && RMS_EPS == 1e-5f);
    CHECK(VOCAB == 120832 && VTILES == 944);
    CHECK(L_MAIN == 42 && NLAYERS == 42 && FIRST_MOE == 1 && DENSE_I == 13824 && DENSE_BLOCKS == 108);
    CHECK(MAX_POS == 1048576 && KV_PAGE == 64);
    CHECK(HC == 4 && HC_D == 12288 && HC_MIX == 8 && HC_EPS == 1e-6f && HC_MAGNITUDE == 2.0f);
    CHECK(HEADS_DSA == 32 && HEADS_SWA == 64 && HEADS == 64 && N_DSA == 12 && N_SWA == 30 && SWA_WINDOW == 512);
    CHECK(QA == 1024 && KV_LORA == 512 && QK_NOPE == 192 && QK_ROPE == 64 && QK_HEAD == 256 && V_HEAD == 256 && KVA == 576);
    CHECK(IDX_HEADS == 16 && IDX_DIM == 128 && IDX_RATIO == 4 && IDX_GROUP_TOPK == 512 && IDX_TOPK == 2048);
    CHECK(NEXP == 512 && TOPK == 16 && EH == 1536 && EH_B == 12 && EXP_I == 1536 && EXP_BLOCKS == 12 && SHARED_I == 1536);
    CHECK(ROUTE_SCALE == 3.998f && SWIGLU_LIMIT == 10.0f);
    CHECK(N_ENGRAM == 4 && ENGRAM_IDS == 16 && ENGRAM_HEAD_DIM == 96 && ENGRAM_DIM == 1536);
    CHECK(ENGRAM_LOOKUP_SIZE == 120818 && ENGRAM_CVOCAB == 92620 && ENGRAM_PAD_ID == 120002 && ENGRAM_ROW_BYTES == 128);
    CHECK(TOK_BOS == 120000 && TOK_PAD == 120002 && TOK_EOS == 120025);

    const std::vector<int> want_dsa = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41};
    int di = 0, si = 0, ei = 0;
    for (int l = 0; l < L_MAIN; ++l) {
        CHECK(is_dense(l) == (l == 0));
        CHECK(is_moe(l) == (l >= 1));
        CHECK(is_dsa(l) != is_swa(l));
        if (is_dsa(l)) {
            CHECK(di < (int)want_dsa.size() && want_dsa[di] == l);
            CHECK(dsa_slot(l) == di && layer_heads(l) == 32 && full_index(l));
            ++di;
        } else {
            CHECK(swa_slot(l) == si && layer_heads(l) == 64 && !full_index(l));
            ++si;
        }
        if (is_engram(l)) {
            CHECK(engram_slot(l) == ei && engram_layer_id(ei) == l + 1);
            ++ei;
        }
    }
    CHECK(di == 12 && si == 30 && ei == 4);

    for (int nr : {1, 2}) {
        CHECK(rank_count_ok(nr));
        int vt = 0;
        for (int r = 0; r < nr; ++r) {
            CHECK(vocab_tile0(nr, r) == vt);
            vt += vocab_tiles(nr, r);
        }
        CHECK(vt == VTILES);
        for (int l = 0; l < L_MAIN; ++l) {
            int htot = 0;
            for (int r = 0; r < nr; ++r) {
                CHECK(h0_layer(l, nr, r) == htot);
                htot += nh_layer(l, nr);
            }
            CHECK(htot == layer_heads(l));
        }
        for (int l = 1; l < L_MAIN; ++l) {
            std::vector<int> cnt(nr, 0);
            for (int e = 0; e < NEXP; ++e) {
                std::vector<int> pe(nr, 0);
                for (int h = 0; h < EXP_BLOCKS; ++h) {
                    int owner = (EXP_BLOCKS * e + h + l) % nr;
                    ++pe[owner];
                    ++cnt[owner];
                }
                for (int r = 0; r < nr; ++r) CHECK(pe[r] == exp_blocks_local(nr));
            }
            for (int r = 0; r < nr; ++r) CHECK(cnt[r] == NEXP * exp_blocks_local(nr));
        }
    }
    std::printf("test_hy4_geom: ok\n");
    return 0;
}
