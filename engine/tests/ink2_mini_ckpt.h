// A miniature Inkling 2 Small checkpoint, in memory (ink2ref::Config + ink2ref::Weights) and as a checkpoint
// directory the loader plans: config.json (text_config + mtp_config), hf_quant_config.json, model.safetensors (6
// layers, local and global attention, NVFP4 experts on layers 0 .. 3 and MXFP8 experts on layers 4 and 5) and
// mtp.safetensors (3 drafter layers: local, global, local; the rnn_hidden Markov head). Host code only.
//
// The trunk is the model of engine/tests/ink2_layers_gpu_test.cu (same generator, same order of draws), without an
// image token. The intermediate sizes split over 1, 4 and 6 ranks.
//
// build_mini(mm, true) adds a vision tower (2 layers of 256 wide, 4 heads of 64, 20-pixel patches, a 4 x 4 position
// table, 2 x 2 patches per image token, a 384-wide projector) and the image token IMG: config.json then has
// image_token_id and vision_config, and model.safetensors the model.visual.* tensors. The tower's weights come from
// their own generator, so the trunk and the drafter are the same with and without it.
#pragma once
#include "../ref/ink2/ink2_cpu_ref.h"
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace ink2mini
{

using ink2ref::Config;
using ink2ref::Layer;
using ink2ref::MtpLayer;
using ink2ref::MtpWeights;
using ink2ref::VisionLayer;
using ink2ref::VisionWeights;
using ink2ref::Weights;

constexpr int H = 256, NL = 6, NH = 48, NKV = 8, HD = 16, RD = 16, RELW = 8, WIN = 8, BASE = 6, INTER = 768, NE = 16, TK = 4, VOCAB = 768, VALID = 750;
constexpr int ML = 3, MI = 768, MR = 16, MW = 5, MB = 8, MK = 7;
constexpr int VD = 256, VL = 2, VH = 4, VI = 512, VP = 20, VM = 2, VG = 4, VPJ = 384, IMG = 760;   // the vision tower; IMG = the image token

struct Rng   // the generator of engine/tests/test_ink2_cpu_ref.cpp
{
    uint64_t s;
    uint32_t next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 40);
    }
    float uni() { return static_cast<float>(next()) / 8388608.0f - 1.0f; }   // [-1, 1)
    std::vector<float> bf(size_t n, float scale, float bias = 0.0f)
    {
        std::vector<float> v(n);
        for (float& x : v) x = ink2ref::rbf(bias + scale * uni());
        return v;
    }
    std::vector<float> f32(size_t n, float scale)
    {
        std::vector<float> v(n);
        for (float& x : v) x = scale * uni();
        return v;
    }
};

struct Mini
{
    Config c;
    Weights w;
    struct Codes { std::vector<uint8_t> c13, s13, c2, s2; };
    std::vector<std::vector<Codes>> codes;   // [layer][expert]: the experts' quantized matrices before packing
};

inline Config mini_config(int tp, bool with_vision = false)
{
    Config c;
    c.hidden = H;
    c.n_layers = NL;
    c.block = 2;
    c.n_heads = NH;
    c.n_kv_heads = NKV;
    c.head_dim = HD;
    c.rel_dim = RD;
    c.rel_win = RELW;
    c.sliding_window = WIN;
    c.log_scale_base = BASE;
    c.inter = INTER;
    c.n_exp = NE;
    c.top_k = TK;
    c.route_scale = 4.0f;
    c.vocab = VOCAB;
    c.valid_vocab = VALID;
    c.image_token_id = -1;
    c.tp = tp;
    c.global = {0, 0, 1, 0, 0, 1};
    c.mxfp8 = {0, 0, 0, 0, 1, 1};
    c.mtp_layers = ML;
    c.mtp_inter = MI;
    c.mtp_window = MW;
    c.mtp_block = MB;
    c.mtp_drafts = MK;
    c.markov_rank = MR;
    c.mtp_global = {0, 1, 0};
    c.mtp_target_layers = {1, 3, 5, 6};
    if (with_vision)
    {
        c.has_vision = true;
        c.image_token_id = IMG;
        c.vis_hidden = VD;
        c.vis_layers = VL;
        c.vis_heads = VH;
        c.vis_mlp = VI;
        c.vis_patch = VP;
        c.vis_merge = VM;
        c.vis_grid = VG;
        c.vis_proj = VPJ;
    }
    return c;
}

// fc1 / fc2 records of every expert of layer l for rank r of T: intermediate rows / columns r * INTER / T ..
inline void pack_experts(const Mini& mm, int l, int T, int r, std::vector<uint8_t>& fc1, std::vector<uint8_t>& fc2)
{
    const int is = INTER / T;
    const size_t n = static_cast<size_t>(INTER) * H, o = static_cast<size_t>(r) * is * H;
    fc1.clear();
    fc2.clear();
    for (int e = 0; e < NE; ++e)
    {
        const Mini::Codes& k = mm.codes[l][e];
        std::vector<uint8_t> f1, f2;
        if (mm.c.mxfp8[l])
        {
            f1 = ink2ref::mx8::pack_fc1(k.c13.data() + o, k.s13.data() + o / 32, k.c13.data() + n + o, k.s13.data() + (n + o) / 32, is, H);
            f2 = ink2ref::mx8::pack_fc2(k.c2.data(), k.s2.data(), H, INTER, r * is, is);
        }
        else
        {
            f1 = nv4pack::pack_fc1(k.c13.data() + o / 2, k.s13.data() + o / 16, k.c13.data() + (n + o) / 2, k.s13.data() + (n + o) / 16, is, H);
            f2 = nv4pack::pack_fc2(k.c2.data(), k.s2.data(), H, INTER, r * is, is);
        }
        fc1.insert(fc1.end(), f1.begin(), f1.end());
        fc2.insert(fc2.end(), f2.begin(), f2.end());
    }
}

inline void build_mini(Mini& mm, bool with_vision = false)
{
    mm.c = mini_config(1, with_vision);
    const Config& c = mm.c;
    Rng r{0x1ce2b00c};
    Weights& w = mm.w;
    const float ih = 1.0f / std::sqrt(static_cast<float>(H));
    const size_t nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD;
    w.embed = r.bf(static_cast<size_t>(VOCAB) * H, 1.0f);
    w.embed_norm = r.bf(H, 0.2f, 1.0f);
    w.final_norm = r.bf(H, 0.2f, 1.0f);
    w.output_query = r.f32(H, 1.0f);
    w.lm_head = r.bf(static_cast<size_t>(VOCAB) * H, 0.3f);
    w.layers.resize(NL);
    mm.codes.resize(NL);
    for (int l = 0; l < NL; ++l)
    {
        Layer& y = w.layers[l];
        const bool g = c.global[l] != 0;
        if (l) y.attn_query = r.f32(static_cast<size_t>(g ? 3 : 1) * H, 1.0f);
        y.mlp_query = r.f32(H, 1.0f);
        if (g)
        {
            y.q_norm = r.bf(H, 0.2f, 1.0f);
            y.k_norm = r.bf(H, 0.2f, 1.0f);
            y.v_norm = r.bf(H, 0.2f, 1.0f);
            y.k_static = r.bf(4 * nk, 0.5f);
            y.v_static = r.bf(4 * nk, 0.5f);
            y.k_wproj = r.bf(static_cast<size_t>(NKV) * 4 * H, ih);
            y.v_wproj = r.bf(static_cast<size_t>(NKV) * 4 * H, ih);
        }
        else
        {
            y.attn_norm = r.bf(H, 0.2f, 1.0f);
            y.k_sconv = r.bf(nk * 4, 0.5f);
            y.v_sconv = r.bf(nk * 4, 0.5f);
        }
        y.wq = r.bf(nq * H, 4.0f * ih);
        y.gate = r.bf(nq * H, 2.0f * ih);
        y.wr = r.bf(static_cast<size_t>(NH) * RD * H, 4.0f * ih);
        y.sink = r.bf(static_cast<size_t>(NH) * H, 2.0f * ih);
        y.wk = r.bf(nk * H, 4.0f * ih);
        y.wv = r.bf(nk * H, 2.0f * ih);
        y.wo = r.bf(H * nq, 0.1f);
        y.q_clip = r.bf(nq, 0.5f, 1.0f);
        y.rel_clip = r.bf(static_cast<size_t>(NH) * RD, 0.5f, 1.0f);
        y.rel_proj = r.bf(static_cast<size_t>(RD) * RELW, 0.3f);
        y.attn_sconv = r.bf(static_cast<size_t>(H) * 4, 0.5f);
        y.mlp_sconv = r.bf(static_cast<size_t>(H) * 4, 0.5f);
        y.mlp_norm = r.bf(H, 0.2f, 1.0f);
        y.router_w = r.bf(static_cast<size_t>(NE) * H, 4.0f * ih);
        y.router_bias = r.f32(NE, 0.1f);
        // the layer's 4 experts, in a fixed pick order: a score is in (0, 1), so steps of 2 leave no near tie
        for (int k = 0; k < TK; ++k) y.router_bias[(3 * l + 5 * k) % NE] += 2.0f * static_cast<float>(TK - k);
        const size_t n = static_cast<size_t>(INTER) * H;
        mm.codes[l].resize(NE);
        for (int e = 0; e < NE; ++e)
        {
            const std::vector<float> w13 = r.f32(2 * n, 0.15f), w2 = r.f32(n, 0.05f);
            Mini::Codes& k = mm.codes[l][e];
            if (c.mxfp8[l])
            {
                k.c13.resize(2 * n); k.s13.resize(2 * n / 32); k.c2.resize(n); k.s2.resize(n / 32);
                ink2ref::mx8::quantize(w13.data(), 2 * n / 32, k.c13.data(), k.s13.data());
                ink2ref::mx8::quantize(w2.data(), n / 32, k.c2.data(), k.s2.data());
            }
            else
            {
                k.c13.resize(n); k.s13.resize(2 * n / 16); k.c2.resize(n / 2); k.s2.resize(n / 16);
                y.s1.push_back(nv4pack::quantize(w13.data(), 2 * INTER, H, nv4pack::ScaleRule::Static6, nullptr, k.c13.data(), k.s13.data()));
                y.s2.push_back(nv4pack::quantize(w2.data(), H, INTER, nv4pack::ScaleRule::Static6, nullptr, k.c2.data(), k.s2.data()));
            }
        }
        pack_experts(mm, l, 1, 0, y.fc1, y.fc2);
    }

    // the drafter
    Rng q{0x6d7470};
    const size_t nt = c.mtp_target_layers.size();
    MtpWeights& y = w.mtp;
    y.fuse_weight = q.bf((ML + 1) * nt, 1.0f);
    y.mask_embed = q.bf(H, 1.0f);
    y.norm = q.bf(H, 0.2f, 1.0f);
    y.layers.resize(ML);
    for (MtpLayer& l : y.layers)
    {
        l.dflare_norm = q.bf(H, 0.2f, 1.0f);
        l.k_ctx = q.bf(nk * H, 4.0f * ih);
        l.v_ctx = q.bf(nk * H, 2.0f * ih);
        l.k_norm_ctx = q.bf(HD, 0.2f, 1.0f);
        l.input_norm = q.bf(H, 0.2f, 1.0f);
        l.post_norm = q.bf(H, 0.2f, 1.0f);
        l.wq = q.bf(nq * H, 4.0f * ih);
        l.wk = q.bf(nk * H, 4.0f * ih);
        l.wv = q.bf(nk * H, 2.0f * ih);
        l.q_norm = q.bf(HD, 0.2f, 1.0f);
        l.k_norm = q.bf(HD, 0.2f, 1.0f);
        l.wo = q.bf(H * nq, 0.1f);
        l.gate = q.bf(static_cast<size_t>(MI) * H, 2.0f * ih);
        l.up = q.bf(static_cast<size_t>(MI) * H, 2.0f * ih);
        l.down = q.bf(static_cast<size_t>(H) * MI, 0.1f);
    }
    y.head_dflare_norm = q.bf(H, 0.2f, 1.0f);
    y.markov_w1 = q.bf(static_cast<size_t>(VOCAB) * MR, 1.0f);
    y.rnn_w = q.bf(static_cast<size_t>(3) * MR * (2 * MR + 2 * H), 2.0f * ih);
    y.rnn_b = q.bf(3 * MR, 0.5f);
    y.hidden_out_w = q.bf(static_cast<size_t>(H) * MR, 0.5f);
    y.hidden_out_b = q.bf(H, 0.5f);

    // the vision tower
    if (!with_vision) return;
    Rng v{0x76697369};
    VisionWeights& x = w.vis;
    const size_t vd = VD, pk = static_cast<size_t>(VP) * VP * 3, md = static_cast<size_t>(VM) * VM * VD;
    const float iv = 1.0f / std::sqrt(static_cast<float>(VD));
    x.patch_embedding = v.bf(vd * pk, 2.0f / std::sqrt(static_cast<float>(pk)));
    x.class_embedding = v.bf(vd, 0.5f);
    x.cls_position_embedding = v.bf(vd, 0.5f);
    x.position_embedding = v.bf(static_cast<size_t>(VG) * VG * vd, 0.5f);
    x.ln_pre_w = v.bf(vd, 0.2f, 1.0f);
    x.ln_pre_b = v.bf(vd, 0.1f);
    x.layers.resize(VL);
    for (VisionLayer& l : x.layers)
    {
        l.ln1_w = v.bf(vd, 0.2f, 1.0f);
        l.ln1_b = v.bf(vd, 0.1f);
        l.ln2_w = v.bf(vd, 0.2f, 1.0f);
        l.ln2_b = v.bf(vd, 0.1f);
        l.q_w = v.bf(vd * vd, 4.0f * iv);
        l.q_b = v.bf(vd, 0.1f);
        l.k_w = v.bf(vd * vd, 4.0f * iv);
        l.k_b = v.bf(vd, 0.1f);
        l.v_w = v.bf(vd * vd, 2.0f * iv);
        l.v_b = v.bf(vd, 0.1f);
        l.o_w = v.bf(vd * vd, 2.0f * iv);
        l.o_b = v.bf(vd, 0.1f);
        l.ls1 = v.bf(vd, 0.2f, 0.5f);
        l.ls2 = v.bf(vd, 0.2f, 0.5f);
        l.fc_w = v.bf(static_cast<size_t>(VI) * vd, 2.0f * iv);
        l.fc_b = v.bf(VI, 0.1f);
        l.proj_w = v.bf(vd * VI, 2.0f / std::sqrt(static_cast<float>(VI)));
        l.proj_b = v.bf(vd, 0.1f);
    }
    x.proj_in = v.bf(static_cast<size_t>(VPJ) * md, 2.0f / std::sqrt(static_cast<float>(md)));
    x.proj_out = v.bf(static_cast<size_t>(H) * VPJ, 2.0f / std::sqrt(static_cast<float>(VPJ)));
}

// ------------------------------------------------------------------------------------------- the checkpoint files
struct StItem
{
    std::string name, dtype;
    std::vector<size_t> shape;
    std::vector<uint8_t> bytes;
};

inline std::vector<uint8_t> bf16_bytes(const std::vector<float>& v)
{
    std::vector<uint8_t> b(v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i)
    {
        const uint16_t h = ink2ref::f32_to_bf16(v[i]);
        b[2 * i] = static_cast<uint8_t>(h & 0xff);
        b[2 * i + 1] = static_cast<uint8_t>(h >> 8);
    }
    return b;
}

inline std::vector<uint8_t> f32_bytes(const std::vector<float>& v)
{
    std::vector<uint8_t> b(v.size() * 4);
    if (!v.empty()) memcpy(b.data(), v.data(), b.size());
    return b;
}

inline void write_file(const std::string& path, const void* p, size_t n, const char* mode = "wb")
{
    FILE* f = fopen(path.c_str(), mode);
    if (!f) throw std::runtime_error(path + ": " + strerror(errno));
    const bool ok = fwrite(p, 1, n, f) == n;
    if (fclose(f) != 0 || !ok) throw std::runtime_error(path + ": write failed");
}

// One safetensors file: u64 header length, the JSON header padded with spaces to a multiple of 8, the data.
inline void st_write(const std::string& path, const std::vector<StItem>& items)
{
    std::string head = "{\"__metadata__\": {\"format\": \"pt\"}";
    size_t off = 0;
    for (const StItem& it : items)
    {
        size_t want = it.dtype == "BF16" ? 2 : it.dtype == "F32" ? 4 : 1;
        std::string shape;
        for (size_t d : it.shape)
        {
            want *= d;
            shape += (shape.empty() ? "" : ", ") + std::to_string(d);
        }
        if (want != it.bytes.size()) throw std::runtime_error(it.name + ": shape and byte count differ");
        head += ", \"" + it.name + "\": {\"dtype\": \"" + it.dtype + "\", \"shape\": [" + shape + "], \"data_offsets\": [" + std::to_string(off) + ", " +
                std::to_string(off + it.bytes.size()) + "]}";
        off += it.bytes.size();
    }
    head += "}";
    while ((8 + head.size()) % 8) head += ' ';
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error(path + ": " + strerror(errno));
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(static_cast<uint64_t>(head.size()) >> (8 * i));
    bool ok = fwrite(len, 1, 8, f) == 8 && fwrite(head.data(), 1, head.size(), f) == head.size();
    for (const StItem& it : items) ok = ok && (it.bytes.empty() || fwrite(it.bytes.data(), 1, it.bytes.size(), f) == it.bytes.size());
    if (fclose(f) != 0 || !ok) throw std::runtime_error(path + ": write failed");
}

// [E][2 INTER][row] with the row pairs (gate i, up i) from one expert's gate block then up block
inline void interleave(const uint8_t* blocks, size_t row, std::vector<uint8_t>& out)
{
    for (int i = 0; i < INTER; ++i)
    {
        out.insert(out.end(), blocks + static_cast<size_t>(i) * row, blocks + static_cast<size_t>(i + 1) * row);
        out.insert(out.end(), blocks + static_cast<size_t>(INTER + i) * row, blocks + static_cast<size_t>(INTER + i + 1) * row);
    }
}

inline std::string int_list(const std::vector<int>& v)
{
    std::string s;
    for (int x : v) s += (s.empty() ? "" : ", ") + std::to_string(x);
    return s;
}

// The model.visual.* tensors of the checkpoint, in name order; none without the tower.
inline std::vector<StItem> vision_items(const Mini& mm)
{
    std::vector<StItem> it;
    if (!mm.c.has_vision) return it;
    const VisionWeights& x = mm.w.vis;
    const size_t vd = VD, pk = static_cast<size_t>(VP) * VP * 3, md = static_cast<size_t>(VM) * VM * VD;
    const auto bf = [&](const std::string& n, std::vector<size_t> shape, const std::vector<float>& v) {
        it.push_back(StItem{"model.visual." + n, "BF16", std::move(shape), bf16_bytes(v)});
    };
    const std::string e = "backbone.embeddings.", t = "backbone.transformer.";
    bf(e + "class_embedding", {vd}, x.class_embedding);
    bf(e + "cls_position_embedding", {vd}, x.cls_position_embedding);
    bf(e + "patch_embedding.weight", {vd, pk}, x.patch_embedding);
    bf(e + "position_embedding.weight", {static_cast<size_t>(VG) * VG, vd}, x.position_embedding);
    for (int l = 0; l < VL; ++l)
    {
        const VisionLayer& y = x.layers[l];
        const std::string p = t + "layers." + std::to_string(l) + ".";
        bf(p + "ln_1.bias", {vd}, y.ln1_b);
        bf(p + "ln_1.weight", {vd}, y.ln1_w);
        bf(p + "ln_2.bias", {vd}, y.ln2_b);
        bf(p + "ln_2.weight", {vd}, y.ln2_w);
        bf(p + "ls_1.gamma", {vd}, y.ls1);
        bf(p + "ls_2.gamma", {vd}, y.ls2);
        bf(p + "mlp.c_fc.bias", {VI}, y.fc_b);
        bf(p + "mlp.c_fc.weight", {VI, vd}, y.fc_w);
        bf(p + "mlp.c_proj.bias", {vd}, y.proj_b);
        bf(p + "mlp.c_proj.weight", {vd, VI}, y.proj_w);
        bf(p + "self_attn.k_proj.bias", {vd}, y.k_b);
        bf(p + "self_attn.k_proj.weight", {vd, vd}, y.k_w);
        bf(p + "self_attn.out_proj.bias", {vd}, y.o_b);
        bf(p + "self_attn.out_proj.weight", {vd, vd}, y.o_w);
        bf(p + "self_attn.q_proj.bias", {vd}, y.q_b);
        bf(p + "self_attn.q_proj.weight", {vd, vd}, y.q_w);
        bf(p + "self_attn.v_proj.bias", {vd}, y.v_b);
        bf(p + "self_attn.v_proj.weight", {vd, vd}, y.v_w);
    }
    bf(t + "ln_pre.bias", {vd}, x.ln_pre_b);
    bf(t + "ln_pre.weight", {vd}, x.ln_pre_w);
    bf("proj_in.weight", {VPJ, md}, x.proj_in);
    bf("proj_out.weight", {H, VPJ}, x.proj_out);
    return it;
}

// Writes the checkpoint directory `dir` (made when absent). Throws std::runtime_error on an I/O error.
inline void write_ckpt(const Mini& mm, const std::string& dir)
{
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) throw std::runtime_error(dir + ": " + strerror(errno));
    const Config& c = mm.c;
    const Weights& w = mm.w;
    const size_t h = H, nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD;
    const auto bf = [](const std::string& n, std::vector<size_t> shape, const std::vector<float>& v) { return StItem{n, "BF16", std::move(shape), bf16_bytes(v)}; };
    const auto f32 = [](const std::string& n, std::vector<size_t> shape, const std::vector<float>& v) { return StItem{n, "F32", std::move(shape), f32_bytes(v)}; };

    // config.json
    {
        std::vector<int> local, mlocal;
        for (int l = 0; l < NL; ++l)
            if (!c.global[l]) local.push_back(l);
        for (int l = 0; l < ML; ++l)
            if (!c.mtp_global[l]) mlocal.push_back(l);
        const std::string s =
            "{\"model_type\": \"inkling2\", \"text_config\": {\"hidden_size\": " + std::to_string(H) + ", \"num_attention_heads\": " + std::to_string(NH) +
            ", \"num_key_value_heads\": " + std::to_string(NKV) + ", \"head_dim\": " + std::to_string(HD) + ", \"n_routed_experts\": " + std::to_string(NE) +
            ", \"num_experts_per_tok\": " + std::to_string(TK) + ", \"intermediate_size\": " + std::to_string(INTER) + ", \"vocab_size\": " +
            std::to_string(VOCAB) + ", \"num_hidden_layers\": " + std::to_string(NL) + ", \"sliding_window\": " + std::to_string(WIN) +
            ", \"log_scaling_n_floor\": " + std::to_string(BASE) + ", \"n_layers_per_attn_res_block\": " + std::to_string(c.block) +
            ", \"route_scale\": 4.0, \"unpadded_vocab_size\": " + std::to_string(VALID) + ", \"local_layer_ids\": [" + int_list(local) + "]}, " +
            "\"mtp_config\": {\"num_hidden_layers\": " + std::to_string(ML) + ", \"intermediate_size\": " + std::to_string(MI) +
            ", \"sliding_window_size\": " + std::to_string(MW) + ", \"block_size\": " + std::to_string(MB) + ", \"num_speculative_tokens\": " +
            std::to_string(MK) + ", \"markov_rank\": " + std::to_string(MR) + ", \"rope_theta\": 1000000.0, \"target_layer_ids\": [" +
            int_list(c.mtp_target_layers) + "], \"local_layer_ids\": [" + int_list(mlocal) + "]}" +
            (c.has_vision ? ", \"image_token_id\": " + std::to_string(c.image_token_id) + ", \"vision_config\": {\"hidden_size\": " +
                                std::to_string(c.vis_hidden) + ", \"num_hidden_layers\": " + std::to_string(c.vis_layers) +
                                ", \"num_attention_heads\": " + std::to_string(c.vis_heads) + ", \"mlp_ratio\": " +
                                std::to_string(c.vis_mlp / c.vis_hidden) + ".0, \"native_patch_size\": " + std::to_string(c.vis_patch) +
                                ", \"adapter_merge_size\": " + std::to_string(c.vis_merge) + ", \"posemb_grid_size\": " + std::to_string(c.vis_grid) +
                                ", \"projection_hidden_dim\": " + std::to_string(c.vis_proj) + ", \"projector_bias\": false, \"layer_norm_eps\": 1e-05}"
                          : std::string()) +
            "}\n";
        write_file(dir + "/config.json", s.data(), s.size(), "w");
    }
    // hf_quant_config.json
    {
        std::string nv, mx;
        for (int l = 0; l < NL; ++l)
            for (const char* t : {"w13_weight", "w2_weight"})
            {
                std::string& s = c.mxfp8[l] ? mx : nv;
                s += (s.empty() ? "" : ", ") + std::string("\"model.llm.layers.") + std::to_string(l) + ".mlp.experts." + t + "\"";
            }
        const std::string s = "{\"model\": {\"parameter_groups\": [{\"compute_dtype\": \"bfloat16\", \"tensors\": [\"model.llm.embed.weight\"]}, "
                              "{\"compute_dtype\": \"nvfp4\", \"required_activation_quantization\": {\"scale_selection_rule\": \"fouroversix_mae\"}, \"tensors\": [" +
                              nv + "]}, {\"compute_dtype\": \"mxfp8\", \"tensors\": [" + mx + "]}]}, \"quant_method\": \"inkling2\"}\n";
        write_file(dir + "/hf_quant_config.json", s.data(), s.size(), "w");
    }
    // model.safetensors
    {
        std::vector<StItem> it;
        const std::string g = "model.llm.";
        it.push_back(bf(g + "embed.weight", {VOCAB, h}, w.embed));
        it.push_back(bf(g + "embed_norm.weight", {h}, w.embed_norm));
        it.push_back(bf(g + "norm.weight", {h}, w.final_norm));
        it.push_back(f32(g + "attn_res.output_query", {h}, w.output_query));
        it.push_back(bf(g + "unembed.weight", {VOCAB, h}, w.lm_head));
        for (int l = 0; l < NL; ++l)
        {
            const Layer& y = w.layers[l];
            const bool gl = c.global[l] != 0;
            const std::string p = g + "layers." + std::to_string(l) + ".";
            if (l) it.push_back(f32(g + "attn_res.attn_queries." + std::to_string(l - 1), gl ? std::vector<size_t>{3, h} : std::vector<size_t>{h}, y.attn_query));
            it.push_back(f32(g + "attn_res.mlp_queries." + std::to_string(l), {h}, y.mlp_query));
            if (gl)
            {
                it.push_back(bf(p + "q_input_norm.weight", {h}, y.q_norm));
                it.push_back(bf(p + "k_input_norm.weight", {h}, y.k_norm));
                it.push_back(bf(p + "v_input_norm.weight", {h}, y.v_norm));
                it.push_back(bf(p + "attn.k_sconv.static_w", {4, nk}, y.k_static));
                it.push_back(bf(p + "attn.v_sconv.static_w", {4, nk}, y.v_static));
                it.push_back(bf(p + "attn.k_sconv.weight_proj", {static_cast<size_t>(NKV) * 4, h}, y.k_wproj));
                it.push_back(bf(p + "attn.v_sconv.weight_proj", {static_cast<size_t>(NKV) * 4, h}, y.v_wproj));
            }
            else
            {
                it.push_back(bf(p + "attn_norm.weight", {h}, y.attn_norm));
                it.push_back(bf(p + "attn.k_sconv.weight", {nk, 1, 4}, y.k_sconv));
                it.push_back(bf(p + "attn.v_sconv.weight", {nk, 1, 4}, y.v_sconv));
            }
            it.push_back(bf(p + "attn.wq_du.weight", {nq, h}, y.wq));
            it.push_back(bf(p + "attn.gate_attn.weight", {nq, h}, y.gate));
            it.push_back(bf(p + "attn.wr_du.weight", {static_cast<size_t>(NH) * RD, h}, y.wr));
            it.push_back(bf(p + "attn.attn_sink_proj.weight", {NH, h}, y.sink));
            it.push_back(bf(p + "attn.wk_dv.weight", {nk, h}, y.wk));
            it.push_back(bf(p + "attn.wv_dv.weight", {nk, h}, y.wv));
            it.push_back(bf(p + "attn.wo_ud.weight", {h, nq}, y.wo));
            it.push_back(bf(p + "attn.q_l2_clip.scale", {NH, HD}, y.q_clip));
            it.push_back(bf(p + "attn.rel_l2_clip.scale", {NH, RD}, y.rel_clip));
            it.push_back(bf(p + "attn.rel_logits_proj.proj", {RD, RELW}, y.rel_proj));
            it.push_back(bf(p + "attn_sconv.weight", {h, 1, 4}, y.attn_sconv));
            it.push_back(bf(p + "mlp_sconv.weight", {h, 1, 4}, y.mlp_sconv));
            it.push_back(bf(p + "mlp_norm.weight", {h}, y.mlp_norm));
            it.push_back(bf(p + "mlp.gate.weight", {NE, h}, y.router_w));
            it.push_back(f32(p + "mlp.gate.bias", {NE}, y.router_bias));
            // the routed experts
            const bool mx = c.mxfp8[l] != 0;
            const size_t c13 = mx ? h : h / 2, s13 = mx ? h / 32 : h / 16, c2 = mx ? INTER : INTER / 2, s2 = mx ? INTER / 32 : INTER / 16;
            std::vector<uint8_t> a, b, d, e;
            for (int x = 0; x < NE; ++x)
            {
                const Mini::Codes& k = mm.codes[l][x];
                interleave(k.c13.data(), c13, a);
                interleave(k.s13.data(), s13, b);
                d.insert(d.end(), k.c2.begin(), k.c2.end());
                e.insert(e.end(), k.s2.begin(), k.s2.end());
            }
            const std::string x = p + "mlp.experts.";
            it.push_back(StItem{x + "w13_weight", mx ? "F8_E4M3" : "U8", {NE, 2 * static_cast<size_t>(INTER), c13}, std::move(a)});
            it.push_back(StItem{x + "w13_weight_scale", mx ? "U8" : "F8_E4M3", {NE, 2 * static_cast<size_t>(INTER), s13}, std::move(b)});
            it.push_back(StItem{x + "w2_weight", mx ? "F8_E4M3" : "U8", {NE, h, c2}, std::move(d)});
            it.push_back(StItem{x + "w2_weight_scale", mx ? "U8" : "F8_E4M3", {NE, h, s2}, std::move(e)});
            if (!mx)
            {
                it.push_back(f32(x + "w13_weight_scale_2", {NE}, y.s1));
                it.push_back(f32(x + "w2_weight_scale_2", {NE}, y.s2));
            }
        }
        for (StItem& t : vision_items(mm)) it.push_back(std::move(t));
        st_write(dir + "/model.safetensors", it);
    }
    // mtp.safetensors
    {
        std::vector<StItem> it;
        const MtpWeights& y = w.mtp;
        const size_t nt = c.mtp_target_layers.size(), R = MR;
        it.push_back(bf("fuse_weight", {static_cast<size_t>(ML) + 1, nt}, y.fuse_weight));
        it.push_back(bf("mask_embed", {h}, y.mask_embed));
        it.push_back(bf("norm.weight", {h}, y.norm));
        it.push_back(bf("markov_head.dflare_norm.weight", {h}, y.head_dflare_norm));
        it.push_back(bf("markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * h}, y.rnn_w));
        it.push_back(bf("markov_head.rnn_proj.bias", {3 * R}, y.rnn_b));
        it.push_back(bf("markov_head.hidden_out.weight", {h, R}, y.hidden_out_w));
        it.push_back(bf("markov_head.hidden_out.bias", {h}, y.hidden_out_b));
        it.push_back(bf("markov_head.markov_w1.weight", {VOCAB, R}, y.markov_w1));
        for (int m = 0; m < ML; ++m)
        {
            const MtpLayer& l = y.layers[m];
            const std::string p = "layers." + std::to_string(m) + ".";
            it.push_back(bf(p + "dflare_norm.weight", {h}, l.dflare_norm));
            it.push_back(bf(p + "input_layernorm.weight", {h}, l.input_norm));
            it.push_back(bf(p + "post_attention_layernorm.weight", {h}, l.post_norm));
            it.push_back(bf(p + "self_attn.q_norm.weight", {HD}, l.q_norm));
            it.push_back(bf(p + "self_attn.k_norm.weight", {HD}, l.k_norm));
            it.push_back(bf(p + "self_attn.k_norm_ctx.weight", {HD}, l.k_norm_ctx));
            it.push_back(bf(p + "self_attn.q_proj.weight", {nq, h}, l.wq));
            it.push_back(bf(p + "self_attn.k_proj.weight", {nk, h}, l.wk));
            it.push_back(bf(p + "self_attn.k_proj_ctx.weight", {nk, h}, l.k_ctx));
            it.push_back(bf(p + "self_attn.v_proj.weight", {nk, h}, l.wv));
            it.push_back(bf(p + "self_attn.v_proj_ctx.weight", {nk, h}, l.v_ctx));
            it.push_back(bf(p + "self_attn.o_proj.weight", {h, nq}, l.wo));
            it.push_back(bf(p + "mlp.gate_proj.weight", {MI, h}, l.gate));
            it.push_back(bf(p + "mlp.up_proj.weight", {MI, h}, l.up));
            it.push_back(bf(p + "mlp.down_proj.weight", {h, MI}, l.down));
        }
        st_write(dir + "/mtp.safetensors", it);
    }
}

}   // namespace ink2mini
