// DSpark drafter of DeepSeek-V4.1-Flash-1017: the host-side rules (no CUDA). docs/DEEPSEEK-DSPARK.md is the frozen card (F1);
// engine/tests/test_ds41_dspark.cpp checks every rule below. The device kernels (tap, main-cache, markov, confidence) and
// the stage slots follow the card; the five Stage values and plan(..., dspark) / plan_draft are B9's (ds41_ops.h).
//
// One round (k = GR8_DS_MTP drafts, verify rows = k + 1):
//   verify forward  rows at positions P .. P+k (row 0 = golden), taps at the entry of layers 37/38/39, main-cache update after finish
//   accept          a = longest prefix with draft_i == argmax_i, q = P + a (last accepted position), golden' = argmax_a at q + 1
//   draft block     5 rows, tokens [golden', noise x4], positions q+1 .. q+5, keys = ring q-127 .. q + the block's 5 keys, no causal mask
//   head            lm_head + markov (5 sequential rows) + confidence -> 5 drafts; the next verify uses the first k
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ds41_ops.h"

namespace ds41
{
namespace dsp
{
constexpr int B = geom::DS_BLOCK;                // 5 rows per draft block (the trained geometry)
constexpr int MAX_K = geom::DS_BLOCK;            // drafts per round 1..5
constexpr int RING = 136;                        // DSpark window ring rows per session per layer: >= WINDOW + B = 133, 128 + 8 for blocks up to 8
constexpr int KEYS = geom::WINDOW + geom::DS_BLOCK;   // 133 keys per draft row: 128 main + 5 block
constexpr int TAPS = 3;
constexpr int TAP_ROW = TAPS * geom::HID;        // 15360 = geom::DS_MAIN_PROJ_IN
constexpr int N_MAIN_COLLECTIVES = 2 * geom::L_MAIN;   // 80: the main step uses c = 0..79 for layers, c = 80 for its head merge
constexpr int DRAFT_NCOLL = 2 * geom::N_NEXTN + B;     // 11: the draft fabric (own send/recv/doorbell/stepp): 6 layer exchanges + 5 head merges
constexpr int PREFILL_MS_ROWS = 128;            // ST_DS_MAIN scratch rows (MainS) of a prefill-capable build: <= RING, so one launch never holds two rows of a slot
static_assert(PREFILL_MS_ROWS <= RING && PREFILL_MS_ROWS >= geom::WINDOW, "a 128-row launch covers the draft window and fits the ring");
static_assert(RING >= KEYS && TAP_ROW == geom::DS_MAIN_PROJ_IN && geom::DS_TARGET0 + TAPS == geom::L_MAIN, "taps are the last three main layers");

// ---- knobs ----------------------------------------------------------------------------------------------------------
struct Knobs
{
    int mtp = 0;                    // GR8_DS_MTP: 0 off, 1..5 drafts per round
    int block = B;                  // GR8_DS_BLOCK: 5 only in F1
    double verify_cap = 0;          // GR8_DS_VERIFY_CAP: 0 off, else the confidence threshold in (0, 1]
    bool relaxed = false;           // GR8_DS_RELAXED_ACCEPT: "" / 0 / off = exact (default); else "ratio:R", "topp:P" or both,
    double relax_ratio = 0;         //   comma separated (kernels/relax_accept.h; NOT EXACT). 0 = that rule is not given
    double relax_topp = 0;
    int prefill_keep = 0;           // GR8_DS_PREFILL_KEEP: 0 = every prompt row writes its main key (default), else the last `keep` >= 128 positions only (ds41_dspark_prefill.h)
    std::vector<double> conf_temp = {1.0};   // GR8_DS_CONF_TEMP: one value or one per draft row (DeepInfer default 1.0)
    bool engram = true;             // GR8_DS_ENGRAM (owned by B11): false = every row is "diag"
    std::string err;                // non-empty = refuse to start
    bool on() const { return mtp > 0; }
    bool diag() const { return !engram; }
    double temp_for(int row) const { return conf_temp.size() == 1 ? conf_temp[0] : conf_temp[(size_t)row]; }
};
// get(name) returns the value or nullptr (std::getenv in the engine, a map in the test)
template <class Get> inline Knobs parse_knobs(Get get)
{
    Knobs k;
    auto geti = [&](const char* n, int d, bool& bad) {
        const char* v = get(n);
        if (!v || !*v) return d;
        char* e = nullptr; long x = std::strtol(v, &e, 10);
        if (*e) { bad = true; return d; }
        return (int)x;
    };
    auto fail = [&](const std::string& m) { if (k.err.empty()) k.err = m; };
    bool bad = false;
    k.mtp = geti("GR8_DS_MTP", 0, bad); if (bad) fail("GR8_DS_MTP is not an integer");
    if (k.mtp < 0 || k.mtp > MAX_K) fail("GR8_DS_MTP must be 0..5 (0 = off)");
    bad = false; k.block = geti("GR8_DS_BLOCK", B, bad);
    if (bad || k.block != B) fail("GR8_DS_BLOCK must be 5 (the trained block; F1 supports no other)");
    if (const char* v = get("GR8_DS_RELAXED_ACCEPT"); v && *v && std::strcmp(v, "0") && std::strcmp(v, "off") && std::strcmp(v, "false"))
    {
        const std::string s = v, bad_rx = "GR8_DS_RELAXED_ACCEPT must be 0 / off, or ratio:R (R in (0, 1]) and/or topp:P (P in (0, 1)), comma separated";
        size_t p = 0;
        bool ok = true;
        while (ok && p <= s.size())
        {
            size_t c = s.find(',', p); if (c == std::string::npos) c = s.size();
            const std::string part = s.substr(p, c - p);
            const size_t col = part.find(':');
            const std::string key = col == std::string::npos ? part : part.substr(0, col);
            const std::string val = col == std::string::npos ? "" : part.substr(col + 1);
            char* e = nullptr; const double x = val.empty() ? 0.0 : std::strtod(val.c_str(), &e);
            if (val.empty() || *e) ok = false;
            else if (key == "ratio" && !k.relax_ratio && x > 0.0 && x <= 1.0) k.relax_ratio = x;
            else if (key == "topp" && !k.relax_topp && x > 0.0 && x < 1.0) k.relax_topp = x;
            else ok = false;
            p = c + 1;
        }
        if (!ok) { k.relax_ratio = k.relax_topp = 0; fail(bad_rx); }
        else k.relaxed = true;
    }
    bad = false; k.engram = geti("GR8_DS_ENGRAM", 1, bad) != 0;
    bad = false; k.prefill_keep = geti("GR8_DS_PREFILL_KEEP", 0, bad);
    if (bad || (k.prefill_keep != 0 && k.prefill_keep < geom::WINDOW))
        fail("GR8_DS_PREFILL_KEEP must be 0 (every prompt row) or >= 128 (the draft window: the first round reads the last 128 prompt keys)");
    if (const char* v = get("GR8_DS_VERIFY_CAP"); v && *v)
    {
        char* e = nullptr; k.verify_cap = std::strtod(v, &e);
        if (*e || !(k.verify_cap >= 0.0 && k.verify_cap <= 1.0)) fail("GR8_DS_VERIFY_CAP must be a number in [0, 1] (0 = off)");
    }
    if (const char* v = get("GR8_DS_CONF_TEMP"); v && *v)
    {
        k.conf_temp.clear();
        std::string s = v; size_t p = 0;
        while (p <= s.size())
        {
            size_t c = s.find(',', p); if (c == std::string::npos) c = s.size();
            const std::string part = s.substr(p, c - p);
            char* e = nullptr; const double t = std::strtod(part.c_str(), &e);
            if (part.empty() || *e || !(t > 0.0)) { fail("GR8_DS_CONF_TEMP must be positive numbers, comma separated"); break; }
            k.conf_temp.push_back(t); p = c + 1;
        }
        if (k.err.empty() && k.conf_temp.size() != 1 && (int)k.conf_temp.size() != B) fail("GR8_DS_CONF_TEMP: one value or 5 (one per draft row)");
        if (!k.err.empty() && k.conf_temp.empty()) k.conf_temp = {1.0};
    }
    return k;
}

// ---- taps -----------------------------------------------------------------------------------------------------------
inline bool is_tap_layer(int l) { return l >= geom::DS_TARGET0 && l < geom::DS_TARGET0 + TAPS; }
inline int tap_slot(int l) { return l - geom::DS_TARGET0; }
inline float rn16(float v)   // FP32 -> BF16 -> FP32, round to nearest even (the trunk holds BF16-valued FP32)
{
    uint32_t u; std::memcpy(&u, &v, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return v;
    u += 0x7fffu + ((u >> 16) & 1u); u &= 0xffff0000u;
    std::memcpy(&v, &u, 4); return v;
}
// tap value of one element: BF16( 0.25 * (s0 + s1 + s2 + s3) ), summed in stream order in FP32
inline float tap_mean(const float s[4]) { return rn16(((s[0] + s[1]) + s[2] + s[3]) * 0.25f); }

// ---- ring and key window --------------------------------------------------------------------------------------------
inline int ring_slot(int pos) { return pos % RING; }
// positions the draft rows of a session attend to: q-127 .. q (main) then q+1 .. q+B (block). Negative positions do not exist.
inline std::vector<int> key_positions(int q, int block = B)
{
    std::vector<int> p;
    for (int x = q - (geom::WINDOW - 1); x <= q + block; ++x) if (x >= 0) p.push_back(x);
    return p;
}

// ---- draft block ----------------------------------------------------------------------------------------------------
struct Block { int tok[B]; int pos[B]; };
inline Block make_block(int golden, int q)
{
    Block b;
    for (int r = 0; r < B; ++r) { b.tok[r] = r == 0 ? golden : geom::DS_NOISE_TOKEN; b.pos[r] = q + 1 + r; }
    return b;
}
inline int predicted_pos(int q, int r) { return q + 2 + r; }   // row r of the block predicts this position

// ---- verify and accept (greedy) -------------------------------------------------------------------------------------
struct Verify { std::vector<int> tok, pos; };
// rows of one verify step: golden at P, drafts 0..k-1 at P+1..P+k
inline Verify verify_rows(int golden, const int* drafts, int k, int P)
{
    Verify v;
    v.tok.push_back(golden); v.pos.push_back(P);
    for (int i = 0; i < k; ++i) { v.tok.push_back(drafts[i]); v.pos.push_back(P + 1 + i); }
    return v;
}
struct Accept { int a; int emitted; int q; int golden; };
// target[i] = argmax of verify row i (k + 1 values); draft i sits at row i + 1 and is accepted iff it equals target[i]
inline Accept accept_greedy(const int* drafts, const int* target, int k, int P)
{
    int a = 0;
    while (a < k && drafts[a] == target[a]) ++a;
    return {a, a + 1, P + a, target[a]};
}

// ---- confidence head ------------------------------------------------------------------------------------------------
inline double confidence(double dot, double temperature)
{
    const double z = std::min(60.0, std::max(-60.0, dot / temperature));
    return 1.0 / (1.0 + std::exp(-z));
}

// ---- stages ---------------------------------------------------------------------------------------------------------
// The stage values and the plans live in ds41_ops.h (B9): plan(..., dspark = true) for the verify step, plan_draft({40,41,42}) for
// the draft block. check_plans() states the card's placement rules (DEEPSEEK-DSPARK.md section 3) over those plans; "" = ok.
inline std::string check_plans(const std::vector<Op>& verify, const std::vector<Op>& draft)
{
    auto at = [](const std::vector<Op>& p, Stage s, int l) {
        for (size_t k = 0; k < p.size(); ++k) if (p[k].stage == s && (l < 0 || p[k].l == l)) return (int)k;
        return -1;
    };
    char b[160];
    for (int l = geom::DS_TARGET0; l < geom::DS_TARGET0 + TAPS; ++l)   // tap: after hc_attn(l), before attn_proj(l), once
    {
        int n = 0; for (const Op& o : verify) n += o.stage == ST_DS_TAP && o.l == l;
        const int t = at(verify, ST_DS_TAP, l), h = at(verify, ST_HC_ATTN, l), a = at(verify, ST_ATTN_PROJ, l), c = at(verify, ST_CONSUME_ATTN, l);
        if (n != 1 || t < 0 || h < 0 || !(h < t && t < a && a < c))
        { std::snprintf(b, sizeof b, "ST_DS_TAP of layer %d is not once between hc_attn and attn_proj", l); return b; }
        if (verify[t].unit != U_ATTN) return "ST_DS_TAP is not in U_ATTN";
    }
    int taps = 0; for (const Op& o : verify) taps += o.stage == ST_DS_TAP;
    if (taps != TAPS) return "the verify plan has a tap outside layers 37..39 (or a missing one)";
    const int m = at(verify, ST_DS_MAIN, -1), f = at(verify, ST_FINISH, -1);
    if (m < 0 || f < 0 || m != f + 1) return "ST_DS_MAIN does not directly follow ST_FINISH";
    for (const Stage s : {ST_DS_DRAFT_IN, ST_DS_HEAD_LOCAL, ST_DS_MARKOV}) if (at(verify, s, -1) >= 0) return "a draft-only stage is in the verify plan";
    if (draft.empty() || draft[0].stage != ST_DS_DRAFT_IN) return "the draft plan does not start with ST_DS_DRAFT_IN";
    const size_t n = draft.size();
    if (n < 3 || draft[n - 2].stage != ST_DS_HEAD_LOCAL || draft[n - 1].stage != ST_DS_MARKOV) return "the draft plan does not end with head_local, markov";
    for (const Op& o : draft)
        if (o.stage == ST_DS_TAP || o.stage == ST_DS_MAIN || o.stage == ST_CA_ROW || o.stage == ST_INDEX_TOPK || o.stage == ST_INDEX_BORROW ||
            o.stage == ST_CAND_BLOCKS || o.stage == ST_ENGRAM || o.stage == ST_EMBED || o.stage == ST_STEP_IN)
            return "the draft plan has a stage a window-only DSpark layer must not have";
    int fin = 0; for (const Op& o : draft) fin += o.stage == ST_FINISH;
    if (fin != 1 || draft[n - 3].stage != ST_FINISH) return "the draft plan needs exactly one ST_FINISH, right before the head";
    return "";
}
// collectives of the DRAFT fabric (B9's pick; the main fabric's c = 80 is its own head merge): layer j sub-layer 2j (attn) / 2j + 1 (ffn),
// markov step i's head merge at 2 * N_NEXTN + i (6..10), ticket index = collective index
inline int draft_collective(int j, bool ffn) { return 2 * j + (ffn ? 1 : 0); }
inline int markov_collective(int i) { return 2 * geom::N_NEXTN + i; }
// Wire bytes of each draft collective: ONE rule for the three parties of a collective -- the proxy's enqueue (g6m4.cu), the kernels' publish
// and wait (op_markov's HeadArgs, put_publish / Ds41Block with BLOCK rows) -- and the CPU gate (test_ds41_dspark.cpp). The proxy writes the tail
// word at byte `size` of the slot and the kernels read it at their own `size`: a different size means the tail is never seen (10 s spin timeout).
// c < 6: a layer exchange of the B block rows, BF16 [B][HID]. c >= 6: one head merge of one row (one session) with HEAD_NC candidates.
// HEAD_NC is 0 whatever --top-p-cands says: the draft head is a greedy argmax and sends the 32 B record only. (First light 10-25: the engine
// enqueued head_bytes(1, 256) = 2080 B for c = 6..10 while the kernels waited for the tail at 32 B.)
constexpr int HEAD_NC = 0;
constexpr size_t HEAD_REC_BYTES = 32;     // sizeof(glue::HeadRec) (static_assert in ds41_dspark_ops.cuh)
constexpr size_t HEAD_CAND_BYTES = 8;     // sizeof(glue::HeadCand) (same)
constexpr size_t head_bytes(int rows, int nc) { return size_t(rows) * (HEAD_REC_BYTES + size_t(nc) * HEAD_CAND_BYTES); }
// SESS > 1: a draft step of nsess sessions moves nsess x B rows per layer exchange and one row per session per head merge (HeadArgs::M = nsess).
constexpr size_t coll_bytes(int c, int hid, int nsess = 1)
{
    return c < 2 * geom::N_NEXTN ? size_t(nsess) * size_t(B) * size_t(hid) * 2 : head_bytes(nsess, HEAD_NC);
}
}   // namespace dsp
}   // namespace ds41
