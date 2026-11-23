// Host unit test of the Inkling-2-Mini looped forward pass in engine/ref/ink2/ink2_cpu_ref.h:
//   - 8 physical layers (2 prelude + 4 core x 3 passes + 2 coda = 16 virtual layers), NVFP4 + MXFP8 experts,
//     3-layer MTP drafter with virtual-layer target_layer_ids [1, 7, 13, 16].
//   - Verifies chunk-split bit-exactness (12 tokens vs 5+4+3 vs token-by-token), rollback exactness,
//     cross-pass message sensitivity (perturbing msg_queries or controllers changes pass 1/2 output and logits),
//     MTP draft parity, and optional --dump DIR mode for external PyTorch/HF parity check.
#include "../ref/ink2/ink2_cpu_ref.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace ink2ref;

struct Rng
{
    uint64_t s;
    uint32_t next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 40);
    }
    float uni() { return static_cast<float>(next()) / 8388608.0f - 1.0f; }
    std::vector<float> bf(size_t n, float scale, float bias = 0.0f)
    {
        std::vector<float> v(n);
        for (float& x : v) x = rbf(bias + scale * uni());
        return v;
    }
    std::vector<float> f32(size_t n, float scale)
    {
        std::vector<float> v(n);
        for (float& x : v) x = scale * uni();
        return v;
    }
};

struct LoopMini
{
    Config c;
    Weights w;
};

LoopMini make_loop_mini(int tp = 1)
{
    LoopMini m;
    Config& c = m.c;
    c.hidden = 128;
    c.n_layers = 8;
    c.block = 2;
    c.n_heads = 24;
    c.n_kv_heads = 8;
    c.head_dim = 16;
    c.rel_dim = 16;
    c.rel_win = 8;
    c.sliding_window = 8;
    c.log_scale_base = 6;
    c.inter = 256;
    c.n_exp = 8;
    c.top_k = 4;
    c.route_scale = 4.0f;
    c.vocab = 256;
    c.valid_vocab = 240;
    c.tp = tp;
    c.global = {0, 1, 0, 1, 0, 1, 0, 1};
    c.mxfp8 = {0, 0, 0, 0, 0, 0, 1, 1};
    c.loop_start = 2;
    c.loop_end = 6;
    c.loop_passes = 3;
    c.loop_gate_rank = 16;
    // Exact Inkling-2-Mini topology scaled down (1 prelude block + 2 core blocks x 3 passes + 1 coda block = 16 virtual layers):
    //   v=0..1   (prelude l=0..1)
    //   v=2..5   (core pass 0 l=2..5)
    //   v=6..9   (core pass 1 l=2..5; prev_states has [h_4] + prev_h [h_6] = 2 rows!)
    //   v=10..13 (core pass 2 l=2..5)
    //   v=14..15 (coda l=6..7)
    c.mtp_layers = 3;
    c.mtp_inter = 256;
    c.mtp_window = 5;
    c.mtp_block = 8;
    c.mtp_drafts = 4;
    c.markov_rank = 16;
    c.mtp_global = {0, 1, 0};
    c.mtp_target_layers = {1, 7, 13, 16};

    Rng r{0x20261122ULL};
    Weights& w = m.w;
    const int H = c.hidden, nh = c.n_heads, nkv = c.n_kv_heads, hd = c.head_dim, rd = c.rel_dim, rw = c.rel_win;
    const int I = c.inter, NE = c.n_exp, V = c.vocab, R = c.loop_gate_rank;
    const float ih = 1.0f / std::sqrt(static_cast<float>(H));
    const size_t nq = static_cast<size_t>(nh) * hd, nk = static_cast<size_t>(nkv) * hd;

    w.embed = r.bf(static_cast<size_t>(V) * H, 0.8f);
    w.embed_norm = r.bf(H, 0.2f, 1.0f);
    w.final_norm = r.bf(H, 0.2f, 1.0f);
    w.output_query = r.f32(H, 0.5f);
    w.lm_head = r.bf(static_cast<size_t>(V) * H, 0.25f);
    w.msg_queries = r.f32(static_cast<size_t>(c.n_loop_layers()) * 4 * H, 0.5f);

    w.layers.resize(static_cast<size_t>(c.n_layers));
    for (int l = 0; l < c.n_layers; ++l)
    {
        Layer& y = w.layers[static_cast<size_t>(l)];
        const bool g = c.global[static_cast<size_t>(l)] != 0;
        if (l) y.attn_query = r.f32(static_cast<size_t>(g ? 3 : 1) * H, 0.5f);
        y.mlp_query = r.f32(H, 0.5f);
        if (g)
        {
            y.q_norm = r.bf(H, 0.2f, 1.0f);
            y.k_norm = r.bf(H, 0.2f, 1.0f);
            y.v_norm = r.bf(H, 0.2f, 1.0f);
            y.k_static = r.bf(4 * nk, 0.3f);
            y.v_static = r.bf(4 * nk, 0.3f);
            y.k_wproj = r.bf(static_cast<size_t>(nkv) * 4 * H, ih);
            y.v_wproj = r.bf(static_cast<size_t>(nkv) * 4 * H, ih);
        }
        else
        {
            y.attn_norm = r.bf(H, 0.2f, 1.0f);
            y.k_sconv = r.bf(nk * 4, 0.3f);
            y.v_sconv = r.bf(nk * 4, 0.3f);
        }
        y.wq = r.bf(nq * H, ih);
        y.wr = r.bf(static_cast<size_t>(nh) * rd * H, ih);
        y.gate = r.bf(nq * H, ih);
        y.sink = r.bf(static_cast<size_t>(nh) * H, ih);
        y.wk = r.bf(nk * H, ih);
        y.wv = r.bf(nk * H, ih);
        y.wo = r.bf(static_cast<size_t>(H) * nq, 1.0f / std::sqrt(static_cast<float>(nq)));
        y.q_clip = r.bf(nq, 0.2f, 1.0f);
        y.rel_clip = r.bf(static_cast<size_t>(nh) * rd, 0.2f, 1.0f);
        y.rel_proj = r.bf(static_cast<size_t>(rd) * rw, 0.2f);
        y.attn_sconv = r.bf(static_cast<size_t>(H) * 4, 0.3f);
        y.mlp_sconv = r.bf(static_cast<size_t>(H) * 4, 0.3f);
        y.mlp_norm = r.bf(H, 0.2f, 1.0f);
        y.router_w = r.bf(static_cast<size_t>(NE) * H, ih);
        y.router_bias = r.f32(NE, 0.1f);
        if (l >= c.loop_start && l < c.loop_end)
        {
            for (Layer::MsgGate* mg : {&y.q_gate, &y.k_gate, &y.v_gate, &y.mlp_gate})
            {
                mg->down = r.bf(static_cast<size_t>(R) * 2 * H, ih * 0.5f);
                mg->forget = r.bf(static_cast<size_t>(H) * R, 1.0f / std::sqrt(static_cast<float>(R)));
                mg->gain = r.bf(static_cast<size_t>(H) * R, 1.0f / std::sqrt(static_cast<float>(R)));
            }
        }
        for (int e = 0; e < NE; ++e)
        {
            const std::vector<float> wg = r.bf(static_cast<size_t>(I) * H, ih);
            const std::vector<float> wu = r.bf(static_cast<size_t>(I) * H, ih);
            const std::vector<float> wd = r.bf(static_cast<size_t>(H) * I, 1.0f / std::sqrt(static_cast<float>(I)));
            if (c.mxfp8[static_cast<size_t>(l)])
            {
                std::vector<uint8_t> gc(wg.size()), gs(wg.size() / 32), uc(wu.size()), us(wu.size() / 32), dc(wd.size()), ds(wd.size() / 32);
                mx8::quantize(wg.data(), wg.size() / 32, gc.data(), gs.data());
                mx8::quantize(wu.data(), wu.size() / 32, uc.data(), us.data());
                mx8::quantize(wd.data(), wd.size() / 32, dc.data(), ds.data());
                auto f1 = mx8::pack_fc1(gc.data(), gs.data(), uc.data(), us.data(), I, H);
                auto f2 = mx8::pack_fc2(dc.data(), ds.data(), H, I, 0, I);
                y.fc1.insert(y.fc1.end(), f1.begin(), f1.end());
                y.fc2.insert(y.fc2.end(), f2.begin(), f2.end());
            }
            else
            {
                std::vector<float> g_u(wg.size() + wu.size());
                memcpy(g_u.data(), wg.data(), wg.size() * 4);
                memcpy(g_u.data() + wg.size(), wu.data(), wu.size() * 4);
                std::vector<uint8_t> c13(g_u.size() / 2), s13(g_u.size() / 16), c2(wd.size() / 2), s2(wd.size() / 16);
                const float o1 = nv4pack::quantize_row(g_u.data(), g_u.size(), nv4pack::ScaleRule::FourOverSixMae, c13.data(), s13.data());
                const float o2 = nv4pack::quantize_row(wd.data(), wd.size(), nv4pack::ScaleRule::FourOverSixMae, c2.data(), s2.data());
                const size_t n1 = wg.size();
                auto f1 = nv4pack::pack_fc1(c13.data(), s13.data(), c13.data() + n1 / 2, s13.data() + n1 / 16, I, H);
                auto f2 = nv4pack::pack_fc2(c2.data(), s2.data(), H, I, 0, I);
                y.fc1.insert(y.fc1.end(), f1.begin(), f1.end());
                y.fc2.insert(y.fc2.end(), f2.begin(), f2.end());
                y.s1.push_back(o1);
                y.s2.push_back(o2);
            }
        }
    }

    // MTP drafter
    MtpWeights& mtp = w.mtp;
    const int ML = c.mtp_layers, MI = c.mtp_inter, MR = c.markov_rank, NT = static_cast<int>(c.mtp_target_layers.size());
    mtp.fuse_weight = r.bf(static_cast<size_t>(ML + 1) * NT, 0.5f);
    mtp.mask_embed = r.bf(H, 0.5f);
    mtp.norm = r.bf(H, 0.2f, 1.0f);
    mtp.head_dflare_norm = r.bf(H, 0.2f, 1.0f);
    mtp.markov_w1 = r.bf(static_cast<size_t>(V) * MR, 0.2f);
    mtp.rnn_w = r.bf(static_cast<size_t>(3) * MR * (2 * MR + 2 * H), ih);
    mtp.rnn_b = r.bf(static_cast<size_t>(3) * MR, 0.1f);
    mtp.hidden_out_w = r.bf(static_cast<size_t>(H) * MR, 0.2f);
    mtp.hidden_out_b = r.bf(H, 0.1f);
    mtp.layers.resize(static_cast<size_t>(ML));
    for (int l = 0; l < ML; ++l)
    {
        MtpLayer& ml = mtp.layers[static_cast<size_t>(l)];
        ml.dflare_norm = r.bf(H, 0.2f, 1.0f);
        ml.input_norm = r.bf(H, 0.2f, 1.0f);
        ml.post_norm = r.bf(H, 0.2f, 1.0f);
        ml.q_norm = r.bf(hd, 0.2f, 1.0f);
        ml.k_norm = r.bf(hd, 0.2f, 1.0f);
        ml.k_norm_ctx = r.bf(hd, 0.2f, 1.0f);
        ml.wq = r.bf(nq * H, ih);
        ml.wk = r.bf(nk * H, ih);
        ml.wv = r.bf(nk * H, ih);
        ml.k_ctx = r.bf(nk * H, ih);
        ml.v_ctx = r.bf(nk * H, ih);
        ml.wo = r.bf(static_cast<size_t>(H) * nq, 1.0f / std::sqrt(static_cast<float>(nq)));
        ml.gate = r.bf(static_cast<size_t>(MI) * H, ih);
        ml.up = r.bf(static_cast<size_t>(MI) * H, ih);
        ml.down = r.bf(static_cast<size_t>(H) * MI, 1.0f / std::sqrt(static_cast<float>(MI)));
    }
    return m;
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return false;
    return memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void require(bool cond, const char* msg)
{
    if (!cond)
    {
        fprintf(stderr, "FAIL: %s\n", msg);
        exit(1);
    }
}

void write_f32(const std::string& path, const std::vector<float>& v)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
}
}   // namespace

int main(int argc, char** argv)
{
    const LoopMini m = make_loop_mini(1);
    const std::vector<int> toks = {3, 17, 42, 99, 5, 128, 7, 200, 11, 64, 19, 88};
    const int N = static_cast<int>(toks.size());

    // 1. Whole-sequence forward
    Model whole(m.c, m.w);
    const std::vector<float> lg_whole = whole.forward(toks.data(), N);
    require(static_cast<int>(lg_whole.size()) == N * m.c.vocab, "logit count");
    for (int i = 0; i < N; ++i)
    {
        for (int v = 0; v < m.c.valid_vocab; ++v) require(std::isfinite(lg_whole[static_cast<size_t>(i) * m.c.vocab + v]), "finite valid logit");
        for (int v = m.c.valid_vocab; v < m.c.vocab; ++v) require(std::isinf(lg_whole[static_cast<size_t>(i) * m.c.vocab + v]) && lg_whole[static_cast<size_t>(i) * m.c.vocab + v] < 0, "-inf padded logit");
    }

    // 2. Chunk-split bit-exactness: 5 + 4 + 3 vs whole
    Model chunked(m.c, m.w);
    std::vector<float> lg_chunked;
    const int chunks[] = {5, 4, 3};
    int off = 0;
    for (int k : chunks)
    {
        auto part = chunked.forward(&toks[static_cast<size_t>(off)], k);
        lg_chunked.insert(lg_chunked.end(), part.begin(), part.end());
        off += k;
    }
    require(bits_equal(lg_whole, lg_chunked), "5+4+3 chunked logits bit-exact vs whole");

    // 3. Token-by-token bit-exactness vs whole
    Model single(m.c, m.w);
    std::vector<float> lg_single;
    for (int i = 0; i < N; ++i)
    {
        auto part = single.forward(&toks[static_cast<size_t>(i)], 1);
        lg_single.insert(lg_single.end(), part.begin(), part.end());
    }
    require(bits_equal(lg_whole, lg_single), "token-by-token logits bit-exact vs whole");

    // 4. Rollback exactness across all 10 virtual states + MTP context
    chunked.rollback(5);
    require(chunked.pos() == 5, "rollback pos");
    auto replay = chunked.forward(&toks[5], N - 5);
    require(memcmp(replay.data(), lg_whole.data() + static_cast<size_t>(5) * m.c.vocab, replay.size() * sizeof(float)) == 0,
            "rollback + replay bit-exact");

    // 5. Cross-pass message sensitivity: zeroing msg_queries or controllers must change logits
    {
        Weights w2 = m.w;
        std::fill(w2.msg_queries.begin(), w2.msg_queries.end(), 0.0f);
        Model m2(m.c, w2);
        auto lg2 = m2.forward(toks.data(), N);
        require(!bits_equal(lg_whole, lg2), "msg_queries must affect logits");
    }
    {
        Weights w3 = m.w;
        std::fill(w3.layers[2].q_gate.gain.begin(), w3.layers[2].q_gate.gain.end(), 0.0f);
        Model m3(m.c, w3);
        auto lg3 = m3.forward(toks.data(), N);
        require(!bits_equal(lg_whole, lg3), "core layer controller must affect logits");
    }

    // 6. MTP drafter with virtual-layer taps [1, 5, 9, 10]
    std::vector<float> d_whole_lg, d_chunk_lg;
    auto d_whole = whole.mtp_draft(42, 4, &d_whole_lg);
    auto d_chunk = chunked.mtp_draft(42, 4, &d_chunk_lg);
    require(d_whole == d_chunk && bits_equal(d_whole_lg, d_chunk_lg), "MTP drafts bit-exact after rollback/chunking");

    // Optional dump for external PyTorch modeling_inkling2_loop.py parity check
    if (argc == 3 && std::string(argv[1]) == "--dump")
    {
        const std::string d = argv[2];
        write_f32(d + "/logits.bin", lg_whole);
        write_f32(d + "/embed.bin", m.w.embed);
        write_f32(d + "/embed_norm.bin", m.w.embed_norm);
        write_f32(d + "/final_norm.bin", m.w.final_norm);
        write_f32(d + "/output_query.bin", m.w.output_query);
        write_f32(d + "/lm_head.bin", m.w.lm_head);
        write_f32(d + "/msg_queries.bin", m.w.msg_queries);
        for (int l = 0; l < m.c.n_layers; ++l)
        {
            const Layer& y = m.w.layers[static_cast<size_t>(l)];
            const std::string p = d + "/L" + std::to_string(l) + "_";
            if (l) write_f32(p + "attn_query.bin", y.attn_query);
            write_f32(p + "mlp_query.bin", y.mlp_query);
            if (m.c.global[static_cast<size_t>(l)])
            {
                write_f32(p + "q_norm.bin", y.q_norm);
                write_f32(p + "k_norm.bin", y.k_norm);
                write_f32(p + "v_norm.bin", y.v_norm);
                write_f32(p + "k_static.bin", y.k_static);
                write_f32(p + "v_static.bin", y.v_static);
                write_f32(p + "k_wproj.bin", y.k_wproj);
                write_f32(p + "v_wproj.bin", y.v_wproj);
            }
            else
            {
                write_f32(p + "attn_norm.bin", y.attn_norm);
                write_f32(p + "k_sconv.bin", y.k_sconv);
                write_f32(p + "v_sconv.bin", y.v_sconv);
            }
            write_f32(p + "wq.bin", y.wq);
            write_f32(p + "wr.bin", y.wr);
            write_f32(p + "gate.bin", y.gate);
            write_f32(p + "sink.bin", y.sink);
            write_f32(p + "wk.bin", y.wk);
            write_f32(p + "wv.bin", y.wv);
            write_f32(p + "wo.bin", y.wo);
            write_f32(p + "q_clip.bin", y.q_clip);
            write_f32(p + "rel_clip.bin", y.rel_clip);
            write_f32(p + "rel_proj.bin", y.rel_proj);
            write_f32(p + "attn_sconv.bin", y.attn_sconv);
            write_f32(p + "mlp_sconv.bin", y.mlp_sconv);
            write_f32(p + "mlp_norm.bin", y.mlp_norm);
            write_f32(p + "router_w.bin", y.router_w);
            write_f32(p + "router_bias.bin", y.router_bias);
            if (l >= m.c.loop_start && l < m.c.loop_end)
            {
                const char* names[] = {"q", "k", "v", "mlp"};
                const Layer::MsgGate* gs[] = {&y.q_gate, &y.k_gate, &y.v_gate, &y.mlp_gate};
                for (int gi = 0; gi < 4; ++gi)
                {
                    write_f32(p + "ctrl_" + names[gi] + "_down.bin", gs[gi]->down);
                    write_f32(p + "ctrl_" + names[gi] + "_forget.bin", gs[gi]->forget);
                    write_f32(p + "ctrl_" + names[gi] + "_gain.bin", gs[gi]->gain);
                }
            }
            // Dump dequantized expert weights for PyTorch eager parity check
            std::vector<float> eg, eu, ed;
            const int H = m.c.hidden, I = m.c.inter, NE = m.c.n_exp;
            for (int e = 0; e < NE; ++e)
            {
                std::vector<float> dg(static_cast<size_t>(I) * H), du(static_cast<size_t>(I) * H), dd(static_cast<size_t>(H) * I);
                if (m.c.mxfp8[static_cast<size_t>(l)])
                {
                    const int n1 = H / 32, n2 = I / 32;
                    const size_t per1 = static_cast<size_t>(I / 16) * n1 * 2 * mx8::GR, per2 = static_cast<size_t>(H / 16) * n2 * mx8::GR;
                    float t[16][32];
                    for (int ch = 0; ch < I / 16; ++ch)
                        for (int g = 0; g < n1; ++g)
                            for (int half = 0; half < 2; ++half)
                            {
                                mx8::decode_group(&y.fc1[e * per1 + ((static_cast<size_t>(ch) * n1 + g) * 2 + half) * mx8::GR], t);
                                auto& dst = half ? du : dg;
                                for (int rr = 0; rr < 16; ++rr) memcpy(&dst[static_cast<size_t>(ch * 16 + rr) * H + g * 32], t[rr], sizeof t[rr]);
                            }
                    for (int ot = 0; ot < H / 16; ++ot)
                        for (int g = 0; g < n2; ++g)
                        {
                            mx8::decode_group(&y.fc2[e * per2 + (static_cast<size_t>(ot) * n2 + g) * mx8::GR], t);
                            for (int rr = 0; rr < 16; ++rr) memcpy(&dd[static_cast<size_t>(ot * 16 + rr) * I + g * 32], t[rr], sizeof t[rr]);
                        }
                }
                else
                {
                    const int n1 = H / 64, n2 = I / 64;
                    const size_t per1 = static_cast<size_t>(I / 16) * n1 * 2 * nv4pack::GR, per2 = static_cast<size_t>(H / 16) * n2 * nv4pack::GR;
                    float t[16][64];
                    for (int ch = 0; ch < I / 16; ++ch)
                        for (int g = 0; g < n1; ++g)
                            for (int half = 0; half < 2; ++half)
                            {
                                nv4pack::decode_group(&y.fc1[e * per1 + ((static_cast<size_t>(ch) * n1 + g) * 2 + half) * nv4pack::GR], y.s1[static_cast<size_t>(e)], t);
                                auto& dst = half ? du : dg;
                                for (int rr = 0; rr < 16; ++rr) memcpy(&dst[static_cast<size_t>(ch * 16 + rr) * H + g * 64], t[rr], sizeof t[rr]);
                            }
                    for (int ot = 0; ot < H / 16; ++ot)
                        for (int g = 0; g < n2; ++g)
                        {
                            nv4pack::decode_group(&y.fc2[e * per2 + (static_cast<size_t>(ot) * n2 + g) * nv4pack::GR], y.s2[static_cast<size_t>(e)], t);
                            for (int rr = 0; rr < 16; ++rr) memcpy(&dd[static_cast<size_t>(ot * 16 + rr) * I + g * 64], t[rr], sizeof t[rr]);
                        }
                }
                eg.insert(eg.end(), dg.begin(), dg.end());
                eu.insert(eu.end(), du.begin(), du.end());
                ed.insert(ed.end(), dd.begin(), dd.end());
            }
            write_f32(p + "exp_gate.bin", eg);
            write_f32(p + "exp_up.bin", eu);
            write_f32(p + "exp_down.bin", ed);
        }
    }
    printf("test_ink2_loop_ref: ok (8 phys -> 16 virtual layers, chunk/single/rollback/MTP bit-exact)\n");
    return 0;
}
