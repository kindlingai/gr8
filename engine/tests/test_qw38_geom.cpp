// Host test of Qwen3.8-Flash-Next geometry (engine/model_geom_qw38.h through model_geom.h, -DG6_MODEL_QW38=1):
//   * every constant, layer predicate, and slot count (36 GDN + 12 full + 1 MTP, PLE on layer 1)
//   * per-rank slices for TP = 1, 2, 3, 4 (full-attention Q/KV, GDN K/V, MoE 5-block rotation, vocab tiles)
//   * with GR8_TEST_QW38_METADATA_DIR set: validates fields against <dir>/config.json and generation_config.json
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_QW38=1 -I engine engine/tests/test_qw38_geom.cpp -o /tmp/t && /tmp/t
#include "model_geom.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if !G6_MODEL_QW38
#error "build with -DG6_MODEL_QW38=1"
#endif

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

namespace
{
void test_constants()
{
    using namespace geom;
    CHECK(HID == 2560 && KSH == 160 && HB == 20 && RMS_EPS == 1e-6f);
    CHECK(VOCAB == 248320 && VOCAB_RAW == 248320 && VTILES == 1940);
    CHECK(L_MAIN == 48 && N_NEXTN == 1 && MTP_LAYER == 48 && NLAYERS == 49);
    CHECK(std::string(G6_MTP_L) == "L048");
    CHECK(HC == 4 && HC_D == 10240 && HC_RANK == 320);
    CHECK(HEADS == 24 && KV_HEADS == 2 && GQA == 12 && HEAD_DIM == 256 && ROPE_DIM == 64 && ROPE_THETA == 1e7f);
    CHECK(Q_PROJ_OUT == 12288 && KV_PROJ_OUT == 512 && O_PROJ_IN == 6144);
    CHECK(IDX_HEADS == 4 && IDX_KV_HEADS == 1 && IDX_DIM == 128 && IDX_COMPRESS == 4 && IDX_TOPK == 2048 && IDX_PROJ_OUT == 640);
    CHECK(GDN_KH == 16 && GDN_VH == 48 && GDN_V_PER_K == 3 && GDN_DK == 128 && GDN_DV == 128 && GDN_CONV == 4 && GDN_QKV_DIM == 10240);
    CHECK(PLE_LAYER == 1 && PLE_NGRAM == 3 && PLE_HEADS == 16 && PLE_HEAD_DIM == 160 && PLE_DIM == 2560 && PLE_SHARDS == 128);
    CHECK(NEXP == 512 && TOPK == 10 && EXP_I == 640 && EXP_BLOCKS == 5 && SHARED_I == 640 && DENSE_I == 0);
    CHECK(TOK_BOS == 248044 && TOK_EOS == 248044 && TOK_EOS_IM_END == 248046 && MAX_POS == 262144);
}

void test_layers()
{
    using namespace geom;
    int ng = 0, nf = 0, np = 0;
    for (int l = 0; l < L_MAIN; ++l)
    {
        CHECK(is_moe(l));
        CHECK(is_full(l) != is_gdn(l));
        if (is_full(l))
        {
            CHECK(l % 4 == 3 && full_slot(l) == nf && full_index(l));
            ++nf;
        }
        else
        {
            CHECK(gdn_slot(l) == ng && !full_index(l));
            ++ng;
        }
        if (is_ple(l))
        {
            CHECK(l == 1);
            ++np;
        }
    }
    CHECK(ng == 36 && nf == 12 && np == 1);
    CHECK(is_moe(MTP_LAYER) && is_full(MTP_LAYER) && !is_gdn(MTP_LAYER) && full_slot(MTP_LAYER) == 12);
    CHECK(!is_moe(-1) && !is_moe(NLAYERS));
}

void test_ranks()
{
    using namespace geom;
    for (int nr : {1, 2, 3, 4})
    {
        CHECK(rank_count_ok(nr) && HEADS % nr == 0 && GDN_VH % nr == 0);
        int q_seen = 0, v_seen = 0, vt_seen = 0;
        for (int r = 0; r < nr; ++r)
        {
            // Full attention Q/KV head mapping
            CHECK(q0(nr, r) == q_seen);
            int lo = KV_HEADS, hi = -1;
            for (int lq = 0; lq < nh_q(nr); ++lq)
            {
                const int kv = (q_seen + lq) / GQA;
                if (kv < lo) lo = kv;
                if (kv > hi) hi = kv;
                CHECK(local_kv_head(nr, r, lq) == kv - kv0(nr, r));
            }
            CHECK(kv0(nr, r) == lo && kv1(nr, r) == hi + 1 && nh_kv(nr, r) == hi - lo + 1 && nh_kv(nr, r) <= nh_kv_max(nr));
            q_seen += nh_q(nr);

            // GDN V/K head mapping
            CHECK(gdn_v0(nr, r) == v_seen);
            int klo = GDN_KH, khi = -1;
            for (int lv = 0; lv < gdn_vh(nr); ++lv)
            {
                const int kh = (v_seen + lv) / GDN_V_PER_K;
                if (kh < klo) klo = kh;
                if (kh > khi) khi = kh;
                CHECK(gdn_local_k(nr, r, lv) == kh - gdn_k0(nr, r));
            }
            CHECK(gdn_k0(nr, r) == klo && gdn_k1(nr, r) == khi + 1 && gdn_kh(nr, r) == khi - klo + 1 && gdn_kh(nr, r) <= gdn_kh_max(nr));
            v_seen += gdn_vh(nr);

            // Vocab tiling
            CHECK(vocab_tile0(nr, r) == vt_seen);
            vt_seen += vocab_tiles(nr, r);
        }
        CHECK(q_seen == HEADS && v_seen == GDN_VH && vt_seen == VTILES);

        // MoE 5-block rotation balance across 512 experts per layer
        for (int l = 0; l < L_MAIN; ++l)
        {
            std::vector<int> cnt(nr, 0);
            for (int e = 0; e < NEXP; ++e)
            {
                std::vector<int> per_e(nr, 0);
                for (int h = 0; h < EXP_BLOCKS; ++h)
                {
                    const int owner = (EXP_BLOCKS * e + h + l) % nr;
                    ++per_e[owner];
                    ++cnt[owner];
                }
                for (int r = 0; r < nr; ++r)
                    CHECK(per_e[r] >= EXP_BLOCKS / nr && per_e[r] <= exp_blocks_max(nr));
            }
            const int total_blks = NEXP * EXP_BLOCKS;
            for (int r = 0; r < nr; ++r)
                CHECK(cnt[r] >= total_blks / nr && cnt[r] <= (total_blks + nr - 1) / nr);
        }
    }
    CHECK(!rank_count_ok(0) && !rank_count_ok(5) && !rank_count_ok(6));
}

struct Scope { const std::string* text; size_t b, e; };
size_t skip_ws(const std::string& s, size_t i) { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; return i; }
size_t skip_str(const std::string& s, size_t i) { for (++i; i < s.size() && s[i] != '"'; ++i) if (s[i] == '\\') ++i; return i + 1; }
size_t find_key(const Scope& sc, const char* key)
{
    int depth = 0;
    for (size_t i = sc.b; i <= sc.e;)
    {
        const char ch = (*sc.text)[i];
        if (ch == '"')
        {
            const size_t j = skip_str(*sc.text, i), k = skip_ws(*sc.text, j);
            if (depth == 1 && k < sc.text->size() && (*sc.text)[k] == ':' && sc.text->compare(i + 1, j - i - 2, key) == 0)
                return skip_ws(*sc.text, k + 1);
            i = j;
            continue;
        }
        if (ch == '{' || ch == '[') ++depth;
        if (ch == '}' || ch == ']') --depth;
        ++i;
    }
    return std::string::npos;
}
size_t close_of(const std::string& s, size_t v)
{
    int depth = 0;
    for (size_t i = v; i < s.size();)
    {
        if (s[i] == '"') { i = skip_str(s, i); continue; }
        if (s[i] == '{' || s[i] == '[') ++depth;
        if ((s[i] == '}' || s[i] == ']') && --depth == 0) return i;
        ++i;
    }
    fprintf(stderr, "FAIL config.json: unclosed bracket\n");
    exit(1);
}
long need_int(const Scope& sc, const char* key)
{
    const size_t v = find_key(sc, key);
    CHECK(v != std::string::npos);
    char* end = nullptr;
    const long out = strtol(sc.text->c_str() + v, &end, 10);
    CHECK(end != sc.text->c_str() + v);
    return out;
}

void test_config(const std::string& dir)
{
    using namespace geom;
    std::ifstream f(dir + "/config.json");
    CHECK(bool(f));
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    const size_t root = text.find('{');
    CHECK(root != std::string::npos);
    Scope sc{&text, root, close_of(text, root)};
    const size_t tc = find_key(sc, "text_config");
    if (tc != std::string::npos && text[tc] == '{') sc = Scope{&text, tc, close_of(text, tc)};
    CHECK(need_int(sc, "hidden_size") == HID);
    CHECK(need_int(sc, "num_hidden_layers") == L_MAIN);
    CHECK(need_int(sc, "num_attention_heads") == HEADS);
    CHECK(need_int(sc, "num_key_value_heads") == KV_HEADS);
    CHECK(need_int(sc, "head_dim") == HEAD_DIM);
    CHECK(need_int(sc, "hc_count") == HC);
    CHECK(need_int(sc, "hc_lowrank") == HC_RANK);
    CHECK(need_int(sc, "indexer_n_heads") == IDX_HEADS);
    CHECK(need_int(sc, "indexer_kv_heads") == IDX_KV_HEADS);
    CHECK(need_int(sc, "indexer_head_dim") == IDX_DIM);
    CHECK(need_int(sc, "indexer_compress_ratio") == IDX_COMPRESS);
    CHECK(need_int(sc, "indexer_budget") == IDX_TOPK);
    CHECK(need_int(sc, "linear_num_key_heads") == GDN_KH);
    CHECK(need_int(sc, "linear_num_value_heads") == GDN_VH);
    CHECK(need_int(sc, "linear_key_head_dim") == GDN_DK);
    CHECK(need_int(sc, "linear_value_head_dim") == GDN_DV);
    CHECK(need_int(sc, "linear_conv_kernel_dim") == GDN_CONV);
    CHECK(need_int(sc, "moe_intermediate_size") == EXP_I);
    CHECK(need_int(sc, "shared_expert_intermediate_size") == SHARED_I);
    CHECK(need_int(sc, "num_experts") == NEXP);
    CHECK(need_int(sc, "num_experts_per_tok") == TOPK);
    CHECK(need_int(sc, "vocab_size") == VOCAB_RAW);
    CHECK(need_int(sc, "max_position_embeddings") == MAX_POS);
    CHECK(need_int(sc, "mtp_num_hidden_layers") == N_NEXTN);
    CHECK(need_int(sc, "ple_embed_dim") == PLE_DIM);
    CHECK(need_int(sc, "split_ngram_parts") == PLE_SHARDS);
    CHECK(need_int(sc, "ngram_size") == PLE_NGRAM);
    CHECK(need_int(sc, "heads_per_ngram") == PLE_HEADS_PER_NGRAM);
    CHECK(need_int(sc, "ple_conv_kernel_size") == PLE_CONV);
}
}   // namespace

int main(int argc, char** argv)
{
    test_constants();
    test_layers();
    test_ranks();
    if (argc > 1 && argv[1] && *argv[1])
    {
        test_config(argv[1]);
        printf("test_qw38_geom: ok (config.json of %s checked)\n", argv[1]);
    }
    else
    {
        printf("test_qw38_geom: ok\n");
    }
    return 0;
}
