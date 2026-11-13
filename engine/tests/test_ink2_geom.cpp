// Host test of the Inkling-2-Small geometry (engine/model_geom_ink2.h through model_geom.h, -DG6_MODEL_INK2=1):
//   * every constant, the layer lists and their counts
//   * the per-rank head, intermediate and vocabulary slices for 1, 2, 3, 4, 6 and 12 ranks against a count over the
//     heads (the rule of head_ranges and vocab_split in engine/loader-rs/src/ink2.rs)
//   * with GR8_TEST_INK2_METADATA_DIR set: the fields of <dir>/config.json the loader reads
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_INK2=1 -I engine engine/tests/test_ink2_geom.cpp -o /tmp/t <knobs> && /tmp/t
#include "model_geom.h"
#include "../knobs-rs/gr8_knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if !G6_MODEL_INK2
#error "build with -DG6_MODEL_INK2=1"
#endif

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

namespace
{
void test_constants()
{
    using namespace geom;
    CHECK(HID == 4608 && VOCAB_RAW == 201024 && VALID_VOCAB == 200058 && VOCAB == 201216);
    CHECK(VOCAB == 1572 * 128 && VTILES == 1572 && VOCAB % (128 * 6) == 0 && VOCAB % (128 * 4) == 0);
    CHECK(KSH == 288 && HB == 36);
    CHECK(L_MAIN == 48 && N_NEXTN == 5 && MTP_LAYER == 48 && NLAYERS == 53 && ATTN_RES_BLOCK == 6 && ATTN_RES_DEPTH == 9);
    CHECK(RMS_EPS == 1e-6f);
    CHECK(std::string(G6_MTP_L) == "L048");
    CHECK(HEADS == 48 && KV_HEADS == 8 && GQA == 6 && HEAD_DIM == 128 && REL_DIM == 16 && REL_WIN == 1024);
    CHECK(WINDOW == 1024 && LOG_SCALE_BASE == 128000 && MAX_POS == 1048576 && KV_PAGE == 64 && SCONV_K == 4);
    CHECK(NEXP == 384 && TOPK == 8 && EXP_I == 2304 && EXP_BLOCKS == 18 && DENSE_I == 0);
    CHECK(ROUTE_SCALE == 8.0f && LOGIT_SOFTCAP == 30.0f && SWIGLU_GATE_CAP == 4.0f && SWIGLU_UP_CAP == 25.0f);
    CHECK(TOK_BOS == 199998 && TOK_EOT == 199999 && TOK_PAD == 200000 && TOK_EOS == 200006);
    CHECK(TOK_IMAGE == 200054 && TOK_THINK_START == 200008 && TOK_THINK_END == 200010);
    CHECK(TOK_IMAGE < VALID_VOCAB && TOK_EOS < VALID_VOCAB && VALID_VOCAB < VOCAB_RAW);
}

void test_layers()
{
    using namespace geom;
    const int want_global[8] = {5, 11, 17, 23, 29, 35, 41, 47};
    int ng = 0, nl = 0, nm = 0, nb = 0;
    for (int l = 0; l < L_MAIN; ++l)
    {
        CHECK(is_moe(l));
        if (is_global(l))
        {
            CHECK(ng < 8 && want_global[ng] == l && global_slot(l) == ng);
            ++ng;
        }
        else
        {
            CHECK(local_slot(l) == nl);
            ++nl;
        }
        CHECK(is_mxfp8_moe(l) == (l >= 40));
        nm += is_mxfp8_moe(l);
        CHECK(is_block_start(l) == (l % 6 == 0));
        nb += is_block_start(l);
        // a block is 5 local layers then one full-attention layer
        CHECK(is_global(l) == (!is_block_start(l) && is_block_start(l + 1)));
    }
    CHECK(ng == N_GLOBAL && nl == N_LOCAL && nm == N_MXFP8 && ng == 8 && nl == 40 && nm == 8 && nb == 8);
    for (int l = L_MAIN; l < NLAYERS + 2; ++l) CHECK(!is_global(l) && !is_mxfp8_moe(l) && !is_moe(l));
    CHECK(!is_global(-1) && !is_moe(-1));
}

void test_ranks()
{
    using namespace geom;
    for (int nr : {1, 2, 3, 4, 6, 12})
    {
        CHECK(rank_count_ok(nr) && HEADS % nr == 0 && EXP_I % nr == 0);
        CHECK(nh_q(nr) * nr == HEADS && inter_slice(nr) * nr == EXP_I && inter_slice(nr) % 64 == 0);
        const int pad = vocab_padded(nr);
        CHECK(pad >= VOCAB_RAW && pad % (128 * nr) == 0 && pad - VOCAB_RAW < 128 * nr && vocab_slice(nr) * nr == pad && vocab_slice(nr) % 128 == 0);
        CHECK(pad == (nr == 1 ? 201088 : VOCAB));
        int q_seen = 0;
        for (int r = 0; r < nr; ++r)
        {
            CHECK(q0(nr, r) == q_seen && inter0(nr, r) == r * (EXP_I / nr) && vocab0(nr, r) == r * (pad / nr));
            // the K / V heads this rank's query heads read, by counting
            int lo = KV_HEADS, hi = -1;
            for (int lq = 0; lq < nh_q(nr); ++lq)
            {
                const int kv = (q_seen + lq) / GQA;
                lo = kv < lo ? kv : lo;
                hi = kv > hi ? kv : hi;
                CHECK(local_kv_head(nr, r, lq) == kv - kv0(nr, r) && local_kv_head(nr, r, lq) >= 0 && local_kv_head(nr, r, lq) < nh_kv(nr, r));
            }
            CHECK(kv0(nr, r) == lo && kv1(nr, r) == hi + 1 && nh_kv(nr, r) == hi - lo + 1 && kv1(nr, r) <= KV_HEADS);
            q_seen += nh_q(nr);
        }
        CHECK(q_seen == HEADS && kv1(nr, nr - 1) == KV_HEADS);
    }
    for (int nr : {5, 7, 8, 24}) CHECK(!rank_count_ok(nr));
    // one rank: all heads; 4 ranks: 12 query heads over 2 K / V heads; 6 ranks: 8 over 2, the K / V head shared by two ranks
    CHECK(nh_q(1) == 48 && nh_kv(1, 0) == 8 && inter_slice(1) == 2304 && vocab_slice(1) == 201088);
    CHECK(nh_q(4) == 12 && inter_slice(4) == 576 && vocab_slice(4) == 50304);
    for (int r = 0; r < 4; ++r) CHECK(nh_kv(4, r) == 2 && kv0(4, r) == 2 * r);
    CHECK(nh_q(6) == 8 && inter_slice(6) == 384 && vocab_slice(6) == 33536);
    const int kv0_6[6] = {0, 1, 2, 4, 5, 6};
    for (int r = 0; r < 6; ++r) CHECK(nh_kv(6, r) == 2 && kv0(6, r) == kv0_6[r]);
    CHECK(local_kv_head(6, 1, 0) == 0 && local_kv_head(6, 1, 3) == 0 && local_kv_head(6, 1, 4) == 1 && local_kv_head(6, 1, 7) == 1);
    CHECK(local_kv_head(6, 2, 0) == 0 && local_kv_head(6, 2, 1) == 0 && local_kv_head(6, 2, 2) == 1 && local_kv_head(4, 3, 11) == 1);
    // the valid ids end inside the last rank's slice at 4 and 6 ranks
    CHECK(vocab0(6, 5) < VALID_VOCAB && vocab0(4, 3) < VALID_VOCAB);
}

// ---- config.json: the value of a key of one object, by a scan that knows strings and nesting
struct Scope
{
    const std::string* text;
    size_t b, e;   // the object's braces
};

size_t skip_ws(const std::string& s, size_t i)
{
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
    return i;
}
// i at an opening quote -> the index after the closing one
size_t skip_str(const std::string& s, size_t i)
{
    for (++i; i < s.size() && s[i] != '"'; ++i)
        if (s[i] == '\\') ++i;
    return i + 1;
}
// Index of the value of key in the object, or npos.
size_t find_key(const Scope& sc, const char* key)
{
    int depth = 0;
    for (size_t i = sc.b; i <= sc.e;)
    {
        const char ch = (*sc.text)[i];
        if (ch == '"')
        {
            const size_t j = skip_str((*sc.text), i), k = skip_ws((*sc.text), j);
            if (depth == 1 && k < (*sc.text).size() && (*sc.text)[k] == ':' && (*sc.text).compare(i + 1, j - i - 2, key) == 0) return skip_ws((*sc.text), k + 1);
            i = j;
            continue;
        }
        if (ch == '{' || ch == '[') ++depth;
        if (ch == '}' || ch == ']') --depth;
        ++i;
    }
    return std::string::npos;
}
// v at '{' or '[' -> the index of its closing bracket
size_t close_of(const std::string& s, size_t v)
{
    int depth = 0;
    for (size_t i = v; i < s.size();)
    {
        if (s[i] == '"')
        {
            i = skip_str(s, i);
            continue;
        }
        if (s[i] == '{' || s[i] == '[') ++depth;
        if ((s[i] == '}' || s[i] == ']') && --depth == 0) return i;
        ++i;
    }
    fprintf(stderr, "FAIL config.json: an open bracket\n");
    exit(1);
}
bool get_int(const Scope& sc, const char* key, long& out)
{
    const size_t v = find_key(sc, key);
    if (v == std::string::npos) return false;
    char* end = nullptr;
    out = strtol((*sc.text).c_str() + v, &end, 10);
    return end != (*sc.text).c_str() + v;
}
long need_int(const Scope& sc, const char* key)
{
    long v = 0;
    if (!get_int(sc, key, v))
    {
        fprintf(stderr, "FAIL config.json: no %s\n", key);
        exit(1);
    }
    return v;
}

void test_config(const std::string& dir)
{
    using namespace geom;
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f)
    {
        fprintf(stderr, "FAIL cannot read %s\n", path.c_str());
        exit(1);
    }
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
    long inter = 0;
    if (!get_int(sc, "moe_intermediate_size", inter)) inter = need_int(sc, "intermediate_size");
    CHECK(inter == EXP_I);
    CHECK(need_int(sc, "n_routed_experts") == NEXP);
    CHECK(need_int(sc, "num_experts_per_tok") == TOPK);
    CHECK(need_int(sc, "vocab_size") == VOCAB_RAW);
    // the layer list, in either of the two forms the loader reads
    std::vector<int> global;
    const size_t lt = find_key(sc, "layer_types"), ll = find_key(sc, "local_layer_ids");
    if (lt != std::string::npos && text[lt] == '[')
    {
        const size_t end = close_of(text, lt);
        for (size_t i = lt + 1; i < end;)
        {
            if (text[i] != '"')
            {
                ++i;
                continue;
            }
            const size_t j = skip_str(text, i);
            global.push_back(text.compare(i + 1, j - i - 2, "full_attention") == 0);
            i = j;
        }
    }
    else if (ll != std::string::npos && text[ll] == '[')
    {
        global.assign(static_cast<size_t>(L_MAIN), 1);
        const size_t end = close_of(text, ll);
        for (size_t i = ll + 1; i < end;)
        {
            char* e = nullptr;
            const long id = strtol(text.c_str() + i, &e, 10);
            if (e == text.c_str() + i)
            {
                ++i;
                continue;
            }
            CHECK(id >= 0 && id < L_MAIN);
            global[static_cast<size_t>(id)] = 0;
            i = static_cast<size_t>(e - text.c_str());
        }
    }
    else
    {
        fprintf(stderr, "FAIL config.json: neither layer_types nor local_layer_ids\n");
        exit(1);
    }
    CHECK(static_cast<int>(global.size()) == L_MAIN);
    for (int l = 0; l < L_MAIN; ++l) CHECK((global[static_cast<size_t>(l)] != 0) == is_global(l));
    // fields the loader does not need: checked when the file has them
    long v = 0;
    if (get_int(sc, "sliding_window", v)) CHECK(v == WINDOW);
    if (get_int(sc, "max_position_embeddings", v)) CHECK(v == MAX_POS);
    if (get_int(sc, "image_token_id", v)) CHECK(v == TOK_IMAGE);
    if (get_int(sc, "bos_token_id", v)) CHECK(v == TOK_BOS);
    if (get_int(sc, "pad_token_id", v)) CHECK(v == TOK_PAD);
}

// The scanner itself, on a small text of both forms.
void test_scanner()
{
    const std::string a = "{\"model\": \"x{\\\"y\", \"hidden_size\": 1, \"text_config\": {\"hidden_size\": 4608, \"nest\": {\"hidden_size\": 7},"
                          " \"layer_types\": [\"sliding_attention\", \"full_attention\"], \"local_layer_ids\": [0, 2]}}";
    const size_t root = a.find('{');
    Scope sc{&a, root, close_of(a, root)};
    long v = 0;
    CHECK(get_int(sc, "hidden_size", v) && v == 1);
    const size_t tc = find_key(sc, "text_config");
    CHECK(tc != std::string::npos && a[tc] == '{');
    Scope in{&a, tc, close_of(a, tc)};
    CHECK(get_int(in, "hidden_size", v) && v == 4608);
    CHECK(!get_int(in, "model", v) && !get_int(sc, "nest", v));
    const size_t lt = find_key(in, "layer_types");
    CHECK(lt != std::string::npos && a[lt] == '[' && a[close_of(a, lt)] == ']');
}
}   // namespace

int main()
{
    test_constants();
    test_layers();
    test_ranks();
    test_scanner();
    const std::optional<std::string>& dir = knob::test_ink2_metadata_dir();
    if (dir && !dir->empty())
    {
        test_config(*dir);
        printf("test_ink2_geom: ok (config.json of %s checked)\n", dir->c_str());
    }
    else
        printf("test_ink2_geom: ok (GR8_TEST_INK2_METADATA_DIR not set: config.json not read)\n");
    return 0;
}
