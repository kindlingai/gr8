// Prints the DeepSeek-V4.1 geometry (engine/model_geom_ds41.h) as "name value" lines for test_ds41_geom.py to compare with
// engine/ref/ds41/config.json. Build: g++ -std=c++17 -DG6_MODEL_DS41=1 -I engine ds41_geom_dump.cpp
#include <cstdio>
#include "model_geom.h"
#define P(n) std::printf("%s %.9g\n", #n, (double)geom::n)
int main()
{
    P(HID); P(VOCAB); P(VTILES); P(KSH); P(HB); P(L_MAIN); P(N_NEXTN); P(MTP_LAYER); P(NLAYERS); P(RMS_EPS); P(DENSE_I);
    P(HEADS); P(HEAD_DIM); P(QK_ROPE); P(QK_NOPE); P(KV_LORA); P(KVA); P(V_HEAD); P(QA); P(Q_OUT); P(O_LORA); P(O_GROUPS);
    P(O_GROUP_IN); P(WINDOW); P(ROPE_THETA); P(COMPRESS_ROPE_THETA); P(MAX_POS); P(YARN_ORIG_POS); P(YARN_FACTOR);
    P(YARN_BETA_FAST); P(YARN_BETA_SLOW); P(IDX_HEADS); P(IDX_DIM); P(IDX_TOPK); P(CAND_SOURCE); P(CAND_TOPK_BLOCKS);
    P(CAND_BLOCK); P(N_PREFILL_LAYER); P(NEXP); P(TOPK); P(EXP_I); P(EXP_BLOCKS); P(SHARED_I); P(ROUTE_SCALE);
    P(SWIGLU_LIMIT); P(FP8_BLOCK); P(HC); P(HC_MIX); P(HC_FLAT); P(HC_ITERS); P(HC_EPS); P(NGRAM_MAX); P(ENGRAM_HEADS);
    P(ENGRAM_HEAD_DIM); P(ENGRAM_VOCAB); P(ENGRAM_CVOCAB); P(ENGRAM_PAD); P(ENGRAM_ROWS_L0); P(ENGRAM_ROWS_L1);
    P(DS_NEXP); P(DS_TOPK); P(DS_BLOCK); P(DS_TARGET0); P(DS_MARKOV); P(DS_NOISE_TOKEN); P(TOK_BOS); P(TOK_EOS);
    P(TOK_PAD); P(TOK_IMAGE);
    for (int l = 0; l < geom::NLAYERS; ++l)
        std::printf("layer %d ratio %d kv_source %d index_source %d engram %d full_index %d\n", l, geom::compress_ratio(l),
                    geom::kv_source(l), geom::index_source(l), (int)geom::is_engram(l), (int)geom::full_index(l));
    return 0;
}
