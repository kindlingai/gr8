// Host test of the DSpark rules (engine/runtime/ds41_dspark.h): no CUDA. Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -DG6_MODEL_DS41=1 -I engine engine/tests/test_ds41_dspark.cpp -o /tmp/test_ds41_dspark
// Checks (a) the knobs and what they refuse, (b) the tap value and its BF16 rounding, (c) the ring: 136 rows are enough for the 133
// keys of a draft row and 128 are not, (d) the key window at the start of a sequence, (e) the draft block layout, (f) greedy accept
// and the verify rows, (g) the confidence head, (h) the step plans against the DSpark placement rules (dsp::check_plans), (i) the collective numbers,
// (j) the relaxed-accept rule (kernels/relax_accept.h) on a 4-rank split record against the full-vocabulary reference.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "runtime/ds41_dspark.h"
#include "runtime/ds41_dspark_dev.h"   // the same functions the device kernels call (DSP_HD), checked against the host model above
#include "kernels/relax_accept.h"     // the relaxed-accept rule k_head_relax calls (RELAX_HD)
#include <algorithm>
#include <random>

using namespace ds41;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

static dsp::Knobs knobs(std::map<std::string, std::string> m)
{
    return dsp::parse_knobs([&](const char* n) -> const char* { auto it = m.find(n); return it == m.end() ? nullptr : it->second.c_str(); });
}

int main()
{
    // (a) knobs: defaults are off, 1..5 is valid, everything else is refused with a message
    {
        const auto d = knobs({});
        EXPECT(d.err.empty() && d.mtp == 0 && !d.on() && d.block == 5 && d.verify_cap == 0 && d.relaxed == 0 && d.engram && !d.diag());
        EXPECT(d.conf_temp.size() == 1 && d.conf_temp[0] == 1.0 && d.temp_for(3) == 1.0);   // DeepInfer default temperature 1.0
        for (int k = 1; k <= 5; ++k) EXPECT(knobs({{"GR8_DS_MTP", std::to_string(k)}}).err.empty() && knobs({{"GR8_DS_MTP", std::to_string(k)}}).on());
        EXPECT(!knobs({{"GR8_DS_MTP", "6"}}).err.empty() && !knobs({{"GR8_DS_MTP", "-1"}}).err.empty() && !knobs({{"GR8_DS_MTP", "x"}}).err.empty());
        // relaxed acceptance: off by default and for 0 / off; a bad spec is refused, not ignored
        for (const char* v : {"0", "off", "false", ""}) { const auto r = knobs({{"GR8_DS_RELAXED_ACCEPT", v}}); EXPECT(r.err.empty() && !r.relaxed && r.relax_ratio == 0 && r.relax_topp == 0); }
        for (const char* v : {"1", "ratio", "ratio:", "ratio:0", "ratio:1.5", "topp:1", "topp:0", "topp:0.9x", "top:0.9", "ratio:0.5,ratio:0.3", "ratio:0.5,", "on"})
        { const auto r = knobs({{"GR8_DS_RELAXED_ACCEPT", v}}); EXPECT(!r.err.empty() && !r.relaxed); }
        { const auto r = knobs({{"GR8_DS_RELAXED_ACCEPT", "ratio:0.5"}}); EXPECT(r.err.empty() && r.relaxed && r.relax_ratio == 0.5 && r.relax_topp == 0); }
        { const auto r = knobs({{"GR8_DS_RELAXED_ACCEPT", "topp:0.9,ratio:0.1"}}); EXPECT(r.err.empty() && r.relaxed && r.relax_ratio == 0.1 && r.relax_topp == 0.9); }
        { const auto r = knobs({{"GR8_DS_RELAXED_ACCEPT", "ratio:1"}}); EXPECT(r.err.empty() && r.relaxed && r.relax_ratio == 1.0); }
        EXPECT(!knobs({{"GR8_DS_BLOCK", "4"}}).err.empty() && knobs({{"GR8_DS_BLOCK", "5"}}).err.empty());   // D1: block is always 5
        EXPECT(!knobs({{"GR8_DS_VERIFY_CAP", "1.5"}}).err.empty() && !knobs({{"GR8_DS_VERIFY_CAP", "-0.1"}}).err.empty());
        EXPECT(knobs({{"GR8_DS_VERIFY_CAP", "0.1"}}).verify_cap == 0.1);
        const auto t5 = knobs({{"GR8_DS_CONF_TEMP", "0.8,0.9,1,1.1,1.2"}});
        EXPECT(t5.err.empty() && t5.conf_temp.size() == 5 && t5.temp_for(0) == 0.8 && t5.temp_for(4) == 1.2);
        EXPECT(!knobs({{"GR8_DS_CONF_TEMP", "0.8,0.9"}}).err.empty() && !knobs({{"GR8_DS_CONF_TEMP", "0"}}).err.empty() && !knobs({{"GR8_DS_CONF_TEMP", "a"}}).err.empty());
        const auto g = knobs({{"GR8_DS_ENGRAM", "0"}, {"GR8_DS_MTP", "5"}});
        EXPECT(g.err.empty() && g.diag());   // Engram off = every row is diag
    }
    // (b) tap: the mean of 4 BF16-valued streams, rounded to BF16, summed in stream order
    {
        const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT(dsp::tap_mean(a) == 2.5f);
        const float b[4] = {dsp::rn16(0.1f), dsp::rn16(0.2f), dsp::rn16(0.3f), dsp::rn16(0.4f)};
        const float want = dsp::rn16((((b[0] + b[1]) + b[2]) + b[3]) * 0.25f);
        EXPECT(dsp::tap_mean(b) == want && dsp::rn16(want) == want);
        EXPECT(dsp::rn16(1.00390625f) == 1.0f && dsp::rn16(1.01171875f) == 1.015625f);   // ties to even: 1+2^-8 -> 1, 1+3*2^-8 -> 1+2^-6
        EXPECT(dsp::is_tap_layer(37) && dsp::is_tap_layer(39) && !dsp::is_tap_layer(36) && !dsp::is_tap_layer(40));
        EXPECT(dsp::tap_slot(37) == 0 && dsp::tap_slot(39) == 2 && dsp::TAP_ROW == 15360);
    }
    // (c) ring: a draft row reads positions q-127 .. q+5 (133); with 136 slots they are distinct, with 128 the key q+1 lands on q-127's slot
    {
        for (int q : {127, 128, 300, 1000, 5081})
        {
            const auto keys = dsp::key_positions(q);
            EXPECT((int)keys.size() == dsp::KEYS && keys.front() == q - 127 && keys.back() == q + 5);
            std::set<int> s136, s128;
            for (int p : keys) { s136.insert(dsp::ring_slot(p)); s128.insert(p % 128); }
            EXPECT((int)s136.size() == dsp::KEYS);           // all distinct: no block key evicts a main key
            EXPECT((int)s128.size() < dsp::KEYS);            // 128 slots collide: why the ring is 136
        }
        EXPECT(dsp::RING >= dsp::KEYS && dsp::RING == 136);
        // a verify chunk of 6 rows writes slots P .. P+5 before acceptance; the next draft needs q-127 .. q with q >= P
        for (int P : {200, 777})
            for (int a = 0; a <= 5; ++a)
            {
                const int q = P + a; std::set<int> written;
                for (int i = 0; i <= 5; ++i) written.insert(dsp::ring_slot(P + i));
                for (int p = q - 127; p <= q; ++p) if (p >= 0) { if (p < P || p > P + 5) EXPECT(written.count(dsp::ring_slot(p)) == 0); }
            }
    }
    // (d) window at the start of a sequence: negative positions do not exist
    {
        const auto k0 = dsp::key_positions(0);
        EXPECT(k0.size() == 1 + 5 && k0[0] == 0 && k0.back() == 5);
        const auto k10 = dsp::key_positions(10);
        EXPECT(k10.size() == 11 + 5 && k10[0] == 0);
        EXPECT(dsp::key_positions(127).size() == 128 + 5 && dsp::key_positions(127)[0] == 0);
    }
    // (e) draft block: row 0 = golden, rows 1..4 = noise token, positions q+1 .. q+5; row r predicts q+2+r
    {
        const auto b = dsp::make_block(4242, 130);
        EXPECT(b.tok[0] == 4242 && b.tok[1] == 128799 && b.tok[4] == 128799);
        for (int r = 0; r < 5; ++r) EXPECT(b.pos[r] == 131 + r && dsp::predicted_pos(130, r) == 132 + r);
    }
    // (f) verify rows and greedy accept
    {
        const int drafts[5] = {11, 12, 13, 14, 15};
        const auto v = dsp::verify_rows(99, drafts, 3, 200);
        EXPECT(v.tok == (std::vector<int>{99, 11, 12, 13}) && v.pos == (std::vector<int>{200, 201, 202, 203}));
        const int allok[4] = {11, 12, 13, 77};   // target argmax per verify row: drafts all match, row 3 gives the bonus token 77
        const auto a3 = dsp::accept_greedy(drafts, allok, 3, 200);
        EXPECT(a3.a == 3 && a3.emitted == 4 && a3.q == 203 && a3.golden == 77);
        const int mid[4] = {11, 55, 13, 77};     // draft 1 differs: accept 1, emitted [11, 55], golden 55 at q + 1
        const auto a1 = dsp::accept_greedy(drafts, mid, 3, 200);
        EXPECT(a1.a == 1 && a1.emitted == 2 && a1.q == 201 && a1.golden == 55);
        const int none[4] = {5, 12, 13, 77};
        const auto a0 = dsp::accept_greedy(drafts, none, 3, 200);
        EXPECT(a0.a == 0 && a0.emitted == 1 && a0.q == 200 && a0.golden == 5);
        const int five[6] = {11, 12, 13, 14, 15, 9};
        const auto a5 = dsp::accept_greedy(drafts, five, 5, 10);
        EXPECT(a5.a == 5 && a5.emitted == 6 && a5.q == 15 && a5.golden == 9);
        // MTP=0 is the plain loop: no drafts, accept always emits the one argmax token
        const int one[1] = {31};
        const auto a_0 = dsp::accept_greedy(drafts, one, 0, 7);
        EXPECT(a_0.a == 0 && a_0.emitted == 1 && a_0.q == 7 && a_0.golden == 31);
    }
    // (g) confidence head: sigmoid of the clamped logit over the temperature
    {
        EXPECT(std::fabs(dsp::confidence(0.0, 1.0) - 0.5) < 1e-12);
        EXPECT(std::fabs(dsp::confidence(2.0, 1.0) - 1.0 / (1.0 + std::exp(-2.0))) < 1e-12);
        EXPECT(std::fabs(dsp::confidence(2.0, 0.5) - 1.0 / (1.0 + std::exp(-4.0))) < 1e-12);   // temperature divides the logit
        EXPECT(dsp::confidence(1e6, 1.0) == 1.0 / (1.0 + std::exp(-60.0)) && dsp::confidence(-1e6, 1.0) == 1.0 / (1.0 + std::exp(60.0)));   // clamp +-60
    }
    // (h) the step plans against the placement rules: whole model (40 layers), and the subset 0,2,20,39 has no tap except layer 39
    {
        std::vector<int> ids(40); for (int i = 0; i < 40; ++i) ids[i] = i;
        const std::vector<int> dids = {40, 41, 42};
        const auto verify = plan(ids, 0, 40, true, true, false, true), draft = plan_draft(dids);
        const std::string e = dsp::check_plans(verify, draft);
        if (!e.empty()) std::printf("check_plans: %s\n", e.c_str());
        EXPECT(e.empty());
        int taps = 0; for (const Op& o : verify) taps += o.stage == ST_DS_TAP;
        EXPECT(taps == 3);
        // dspark off: no DSpark stage anywhere (the plain plan)
        for (const Op& o : plan(ids, 0, 40, true, true)) EXPECT(o.stage != ST_DS_TAP && o.stage != ST_DS_MAIN);
        // the draft plan covers exactly the three DSpark layers once each
        for (int l : {40, 41, 42}) { int n = 0; for (const Op& o : draft) n += o.stage == ST_ATTN_CORE && o.l == l; EXPECT(n == 1); }
        // a broken plan is caught: move the tap after attn_proj
        auto bad = verify;
        for (size_t k = 0; k + 1 < bad.size(); ++k) if (bad[k].stage == ST_DS_TAP) { std::swap(bad[k], bad[k + 1]); break; }
        EXPECT(!dsp::check_plans(bad, draft).empty());
        auto bad2 = draft; bad2.pop_back();
        EXPECT(!dsp::check_plans(verify, bad2).empty());
    }
    // (i) collectives: the main fabric uses 0..79 for layers and 80 for its head merge; the draft block has its own fabric of 11
    {
        EXPECT(dsp::N_MAIN_COLLECTIVES == 80 && dsp::DRAFT_NCOLL == 11);
        EXPECT(dsp::draft_collective(0, false) == 0 && dsp::draft_collective(0, true) == 1 && dsp::draft_collective(2, true) == 5);
        std::set<int> used;
        for (int j = 0; j < 3; ++j) for (bool f : {false, true}) used.insert(dsp::draft_collective(j, f));
        for (int i = 0; i < dsp::B; ++i) used.insert(dsp::markov_collective(i));
        EXPECT(dsp::markov_collective(0) == 6 && dsp::markov_collective(4) == 10);
        EXPECT((int)used.size() == dsp::DRAFT_NCOLL && *used.begin() == 0 && *used.rbegin() == dsp::DRAFT_NCOLL - 1);   // 11 distinct, dense
    }
    // (i2) sizes: for every collective of the draft fabric, the bytes the proxy enqueues = the bytes its producer publishes = the bytes its consumer
    //      waits for, and every collective has exactly one producer and one consumer in the draft plan. An earlier draft plan died on c = 6..10: the engine
    //      enqueued head_bytes(1, 256) = 2080 B, the head merge waited for its tail at 32 B (10 s spin timeout). Roles come from plan_draft, the c of an
    //      op from ds41_step.cuh (publish_attn 2i, publish_ffn 2i+1, consume_attn 2i, consume_ffn 2i+1, finish 2i+1) and ds41_dspark_ops.cuh (markov
    //      step i -> 6 + i), the bytes of the kernels from put_publish (M * (HID * 2 / 16) uint4), Ds41Block (rows * HID * 2) and head_wait_bytes of
    //      op_markov's HeadArgs (nc = HEAD_NC, no tk). The CUDA ops cannot run here: a GPU gate of the draft fabric (m4sim TP4) is still missing.
    {
        const auto draft = plan_draft({40, 41, 42});
        auto walk = [&](size_t (*enqueue)(int c)) -> std::string {
            const size_t layer_pub = size_t(dsp::B) * (geom::HID * 2 / 16) * 16, layer_wait = size_t(dsp::B) * geom::HID * 2;
            const size_t head_pub = dsp::head_bytes(1, dsp::HEAD_NC), head_wait = dsp::head_bytes(1, dsp::HEAD_NC);
            std::map<int, std::vector<size_t>> pub, wait;
            for (const Op& o : draft)
                switch (o.stage)
                {
                case ST_PUBLISH_ATTN: pub[dsp::draft_collective(o.i, false)].push_back(layer_pub); break;
                case ST_PUBLISH_FFN: pub[dsp::draft_collective(o.i, true)].push_back(layer_pub); break;
                case ST_CONSUME_ATTN: wait[dsp::draft_collective(o.i, false)].push_back(layer_wait); break;
                case ST_CONSUME_FFN: case ST_FINISH: wait[dsp::draft_collective(o.i, true)].push_back(layer_wait); break;
                case ST_DS_MARKOV:
                    for (int i = 0; i < dsp::B; ++i) { pub[dsp::markov_collective(i)].push_back(head_pub); wait[dsp::markov_collective(i)].push_back(head_wait); }
                    break;
                default: break;
                }
            for (int c = 0; c < dsp::DRAFT_NCOLL; ++c)
            {
                if (pub[c].size() != 1 || wait[c].size() != 1) return "collective " + std::to_string(c) + " has " + std::to_string(pub[c].size()) + " publishes and " + std::to_string(wait[c].size()) + " waits";
                if (pub[c][0] != enqueue(c) || wait[c][0] != enqueue(c))
                    return "collective " + std::to_string(c) + ": enqueue " + std::to_string(enqueue(c)) + " B, publish " + std::to_string(pub[c][0]) + " B, wait " + std::to_string(wait[c][0]) + " B";
            }
            for (const auto& kv : pub) if (kv.first < 0 || kv.first >= dsp::DRAFT_NCOLL) return "a publish on collective " + std::to_string(kv.first) + " outside the draft fabric";
            for (const auto& kv : wait) if (kv.first < 0 || kv.first >= dsp::DRAFT_NCOLL) return "a wait on collective " + std::to_string(kv.first) + " outside the draft fabric";
            return "";
        };
        const std::string e = walk([](int c) { return dsp::coll_bytes(c, geom::HID); });
        if (!e.empty()) std::printf("draft collective sizes: %s\n", e.c_str());
        EXPECT(e.empty());
        EXPECT(dsp::coll_bytes(0, geom::HID) == 51200 && dsp::coll_bytes(5, geom::HID) == 51200 && dsp::coll_bytes(6, geom::HID) == 32 && dsp::coll_bytes(10, geom::HID) == 32);
        // SESS > 1: a draft step of n sessions moves n x 5 rows per layer exchange and n head records per merge (8 sessions: 40 rows <= G6_MX 48)
        EXPECT(dsp::coll_bytes(0, geom::HID, 1) == 51200 && dsp::coll_bytes(5, geom::HID, 8) == 8 * 51200 && dsp::coll_bytes(6, geom::HID, 8) == 8 * 32 &&
               dsp::coll_bytes(10, geom::HID, 2) == 64);
        // teeth: a wrong enqueue rule (head_bytes(1, nc) with --top-p-cands 256) is caught, and so is a head size that follows nc
        const std::string old_rule = walk([](int c) { return c < 2 * geom::N_NEXTN ? size_t(dsp::B) * geom::HID * 2 : dsp::head_bytes(1, 256); });
        EXPECT(old_rule == "collective 6: enqueue 2080 B, publish 32 B, wait 32 B");
        EXPECT(dsp::head_bytes(1, 256) == 2080 && dsp::head_bytes(1, 0) == 32);
    }
    // (j) the shared device/host rules (ds41_dspark_dev.h) agree with the host model
    {
        namespace dd = ds41::dspd;
        // tap: same value as dsp::tap_mean after the BF16 rounding, over many random streams
        unsigned rng = 12345u;
        auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return ((int)(rng >> 8) - (1 << 23)) / (float)(1 << 18); };
        for (int t = 0; t < 20000; ++t)
        {
            const float s4[4] = {rnd(), rnd(), rnd(), rnd()};
            EXPECT(dsp::rn16(dd::tap_mean_f32(s4[0], s4[1], s4[2], s4[3])) == dsp::tap_mean(s4));
        }
        // accept: every equality pattern of k = 1..5 drafts against the targets
        for (int k = 1; k <= 5; ++k)
            for (int mask = 0; mask < (1 << k); ++mask)
            {
                int drafts[5], target[6], vtok[6];
                for (int i = 0; i <= k; ++i) target[i] = 1000 + i;
                vtok[0] = 77;
                for (int i = 0; i < k; ++i) { drafts[i] = ((mask >> i) & 1) ? target[i] : 5000 + i; vtok[1 + i] = drafts[i]; }
                const int P = 300 + k;
                const dsp::Accept ref = dsp::accept_greedy(drafts, target, k, P);
                int a, q, g;
                dd::accept_session(vtok, target, k + 1, P, a, q, g);
                EXPECT(a == ref.a && q == ref.q && g == ref.golden);
            }
        // block rows and the key list, for q at and around the window edge
        for (int q = 0; q < 400; ++q)
        {
            const dsp::Block b = dsp::make_block(4242, q);
            for (int r = 0; r < dd::BLOCK; ++r) EXPECT(dd::block_token(4242, r) == b.tok[r] && dd::block_pos(q, r) == b.pos[r]);
            const std::vector<int> ref = dsp::key_positions(q);
            EXPECT(dd::key_count(q, dd::BLOCK) == (int)ref.size());
            for (int i = 0; i < (int)ref.size(); ++i) EXPECT(dd::key_pos(q, dd::BLOCK, i) == ref[i]);
        }
        // confidence: sigmoid(clamped dot / T) is the host's probability
        for (double dot : {-100.0, -3.5, 0.0, 0.7, 12.0, 400.0})
            for (double T : {0.5, 1.0, 2.0})
            {
                const double z = dd::conf_clamped((float)dot, (float)T);
                EXPECT(std::fabs(1.0 / (1.0 + std::exp(-z)) - dsp::confidence(dot, T)) < 1e-6);
            }
        // markov: prev_0 = golden, prev_i = draft_{i-1}, merge output i-major [5][nsess]
        {
            const int nsess = 3; int golden[3] = {10, 11, 12}; int dtok[5 * 3];
            for (int i = 0; i < 5; ++i) for (int s = 0; s < nsess; ++s) dtok[i * nsess + s] = 100 * i + s;
            for (int s = 0; s < nsess; ++s)
            {
                EXPECT(dd::markov_prev(golden, dtok, nsess, s, 0) == golden[s]);
                for (int i = 1; i < 5; ++i) EXPECT(dd::markov_prev(golden, dtok, nsess, s, i) == 100 * (i - 1) + s);
            }
        }
        static_assert(dd::BLOCK == dsp::B && dd::WINDOW == geom::WINDOW && dd::NOISE_TOKEN == geom::DS_NOISE_TOKEN && dd::MARKOV_RANK == geom::DS_MARKOV &&
                      dd::CONF_IN == geom::DS_CONF_IN && dd::HID == geom::HID && dd::TAPS == dsp::TAPS, "dev constants match the geometry");
    }
    // (j) relaxed accept (relax_accept.h, the decision k_head_relax makes): the 4-rank split record (lmax / lsum merged in rank
    // order, nc highest logits per rank as candidates, largest unsent logit = cut) gives SGLang's full-vocabulary decision
    // (relaxed_accept.py relaxed_predict, in double) whenever the draft's logit is >= cut; below the cut the top-p rule refuses.
    // Draws within 1e-4 of a rule boundary are skipped (float vs double). ratio:1 accepts only an argmax tie.
    {
        std::mt19937 rng(1234);
        const int V = 4096, NR = 4, NC = 64, SL = V / NR;
        struct Cfg { double ratio, topp; } cfgs[] = {{0.1, 0.9}, {0.5, 0}, {0, 0.9}, {0.3, 0.5}, {1.0, 0}};
        int checked = 0, accepted = 0, refused_cut = 0;
        for (const Cfg& c : cfgs)
        {
            relax::Rule r;
            r.on = 1; r.log_ratio = c.ratio > 0 ? (float)std::log(c.ratio) : -INFINITY; r.topp = (float)c.topp;
            for (int it = 0; it < 3000; ++it)
            {
                std::vector<float> lg(V);
                const double sc = 1.0 + 5.0 * std::uniform_real_distribution<double>(0, 1)(rng);   // peaked .. flat rows
                std::normal_distribution<float> nd(0.f, (float)sc);
                for (float& x : lg) x = nd(rng);
                const int d = std::uniform_int_distribution<int>(0, 7)(rng) < 6 ? (int)(std::max_element(lg.begin(), lg.end()) - lg.begin()) : std::uniform_int_distribution<int>(0, V - 1)(rng);
                std::vector<int> ord(V); for (int i = 0; i < V; ++i) ord[i] = i;
                std::sort(ord.begin(), ord.end(), [&](int x, int y) { return lg[x] > lg[y]; });
                const int pick = std::uniform_int_distribution<int>(0, 5)(rng);
                const int dd = it % 2 ? ord[pick] : d;   // half the draws: one of the 6 best tokens
                // reference, double, full vocabulary
                double M = -1e300; for (float x : lg) M = std::max(M, (double)x);
                double Z = 0, A = 0; for (float x : lg) { Z += std::exp(x - M); if (x > lg[dd]) A += std::exp(x - M); }
                const double gap = lg[dd] - M, pa = A / Z;
                if ((c.ratio > 0 && std::fabs(gap - std::log(c.ratio)) < 1e-4) || (c.topp > 0 && std::fabs(pa - c.topp) < 1e-4)) continue;
                const bool ref = (c.ratio <= 0 || gap >= std::log(c.ratio)) && (c.topp <= 0 || pa < c.topp);
                // split record: per rank lmax, lsum, candidates, unsent max; merged in rank order (float, as the kernels)
                float mx = -INFINITY, sm = 0.f, cut = -INFINITY;
                std::vector<float> cand;
                for (int p = 0; p < NR; ++p)
                {
                    std::vector<float> sl(lg.begin() + p * SL, lg.begin() + (p + 1) * SL);
                    float lmx = -INFINITY; for (float x : sl) lmx = std::max(lmx, x);
                    float ls = 0.f; for (float x : sl) ls += std::exp(x - lmx);
                    std::sort(sl.begin(), sl.end(), [](float x, float y) { return x > y; });
                    cand.insert(cand.end(), sl.begin(), sl.begin() + NC);
                    cut = std::max(cut, sl[NC]);
                    const float nmx = std::max(mx, lmx);
                    sm = (mx == -INFINITY ? 0.f : sm * std::exp(mx - nmx)) + ls * std::exp(lmx - nmx);
                    mx = nmx;
                }
                const float ld = lg[dd];
                const bool complete = c.topp > 0 && ld >= cut;
                float above = 0.f;
                if (complete) for (float x : cand) if (x > ld) above += std::exp(x - mx);
                const bool got = relax::ok(r, ld, mx, above, sm, complete);
                if (c.topp > 0 && !complete) { EXPECT(!got); ++refused_cut; continue; }
                EXPECT(got == ref);
                ++checked; accepted += got;
            }
        }
        EXPECT(checked > 10000 && accepted > 1000 && accepted < checked);
        // edges: -inf / NaN draft logit never accepted; ratio:1 = argmax tie only; no rule given = always accept a finite logit
        relax::Rule r1; r1.on = 1; r1.log_ratio = 0.f;
        EXPECT(relax::ok(r1, 3.f, 3.f, 0.f, 1.f, false) && !relax::ok(r1, 2.999f, 3.f, 0.f, 1.f, false));
        EXPECT(!relax::ok(r1, -INFINITY, 3.f, 0.f, 1.f, true) && !relax::ok(r1, NAN, 3.f, 0.f, 1.f, true));
        std::printf("relaxed accept: %d decisions equal to the reference (%d accepted), %d below the cut refused\n", checked, accepted, refused_cut);
    }
    std::printf("test_ds41_dspark: all checks passed\n");
    return 0;
}
