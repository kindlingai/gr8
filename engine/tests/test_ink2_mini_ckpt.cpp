// The miniature checkpoint of ink2_mini_ckpt.h against the loader: write the directory, plan it for every rank of 1,
// 4 and 6, and compare tensors of the planned image with the in-memory model (the expert records with the packers of
// the reference, head / intermediate slices of trunk and drafter matrices, replicated tensors). Then the same with
// the vision tower: every model.visual.* tensor is in every rank's image under V., whole, and the trunk is unchanged.
// Host only.
//
//   g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off engine/tests/test_ink2_mini_ckpt.cpp engine/loader/g6pack.cpp
//       engine/loader-rs/target/release/libg6load.a -lpthread -ldl -lm -o /tmp/test_ink2_mini_ckpt
#include "ink2_mini_ckpt.h"
#include "../loader/g6pack.h"
#include "../loader-rs/g6load.h"
#include <cstdlib>
#include <unistd.h>

using namespace ink2mini;

static int fails = 0;

static void check(bool ok, const std::string& what)
{
    if (ok) return;
    ++fails;
    if (fails <= 20) fprintf(stderr, "FAIL %s\n", what.c_str());
}

struct Image
{
    void* plan = nullptr;
    g6::Pack pack;
    std::string tag;
    std::vector<uint8_t> get(const std::string& name)
    {
        const g6::Entry* e = pack.find(name);
        if (!e)
        {
            check(false, tag + ": no " + name);
            return {};
        }
        std::vector<uint8_t> b(e->nbytes);
        char err[1024] = "";
        if (e->nbytes && g6l_plan_read(plan, e->offset, e->nbytes, b.data(), err, sizeof err)) check(false, tag + ": read of " + name + ": " + err);
        return b;
    }
    void same(const std::string& name, const std::vector<uint8_t>& want)
    {
        const std::vector<uint8_t> got = get(name);
        check(got == want, tag + ": " + name + " (" + std::to_string(got.size()) + " bytes, expected " + std::to_string(want.size()) + ")");
    }
};

// rows r0 .. r0 + nr and columns c0 .. c0 + nc of the [rows][cols] matrix m, as BF16 bytes
static std::vector<uint8_t> cut(const std::vector<float>& m, size_t cols, size_t r0, size_t nr, size_t c0, size_t nc)
{
    std::vector<float> v;
    v.reserve(nr * nc);
    for (size_t r = r0; r < r0 + nr; ++r)
        for (size_t c = c0; c < c0 + nc; ++c) v.push_back(m[r * cols + c]);
    return bf16_bytes(v);
}

// The checkpoint with or without the vision tower, planned for every rank of 1, 4 and 6. Returns the images checked.
static int run(bool with_vision)
{
    Mini mm;
    build_mini(mm, with_vision);
    char dir[] = "/tmp/ink2-mini-ckpt.XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); exit(1); }
    const std::string d = dir;
    write_ckpt(mm, d);
    const Weights& w = mm.w;
    const size_t nq = static_cast<size_t>(NH) * HD, nk = static_cast<size_t>(NKV) * HD;
    int images = 0;
    for (int T : {1, 4, 6})
        for (int r = 0; r < T; ++r)
        {
            Image im;
            im.tag = "tp " + std::to_string(T) + " rank " + std::to_string(r);
            const std::string man = d + "/rank.manifest.json";
            char err[1024] = "";
            im.plan = g6l_ink2_plan(d.c_str(), static_cast<uint32_t>(T), static_cast<uint32_t>(r), man.c_str(), err, sizeof err);
            if (!im.plan)
            {
                fprintf(stderr, "FAIL %s: plan: %s\n", im.tag.c_str(), err);
                exit(1);
            }
            std::string e;
            if (!im.pack.open_manifest(man, e))
            {
                fprintf(stderr, "FAIL %s: manifest: %s\n", im.tag.c_str(), e.c_str());
                exit(1);
            }
            // the rank's query heads and the KV heads they read
            const size_t nh_q = NH / T, q0 = static_cast<size_t>(r) * nh_q, gqa = NH / NKV, kv0 = q0 / gqa, nh_kv = (q0 + nh_q - 1) / gqa + 1 - kv0;
            const size_t is = MI / T, i0 = static_cast<size_t>(r) * is;

            im.same("G.embed", bf16_bytes(w.embed));
            im.same("G.embed_norm", bf16_bytes(w.embed_norm));
            im.same("G.final_norm", bf16_bytes(w.final_norm));
            im.same("G.attn_res.output_query", f32_bytes(w.output_query));
            check(im.pack.find("G.lm_head") != nullptr, im.tag + ": no G.lm_head");
            for (int l = 0; l < NL; ++l)
            {
                const Layer& y = w.layers[l];
                char p[16];
                snprintf(p, sizeof p, "L%03d.", l);
                const std::string P = p;
                std::vector<uint8_t> fc1, fc2;
                pack_experts(mm, l, T, r, fc1, fc2);
                im.same(P + "x.fc1", fc1);
                im.same(P + "x.fc2", fc2);
                if (!mm.c.mxfp8[l])
                {
                    im.same(P + "x.s1", f32_bytes(y.s1));
                    im.same(P + "x.s2", f32_bytes(y.s2));
                }
                im.same(P + "wq", cut(y.wq, H, q0 * HD, nh_q * HD, 0, H));
                im.same(P + "wr", cut(y.wr, H, q0 * RD, nh_q * RD, 0, H));
                im.same(P + "gate_attn", cut(y.gate, H, q0 * HD, nh_q * HD, 0, H));
                im.same(P + "attn_sink_proj", cut(y.sink, H, q0, nh_q, 0, H));
                im.same(P + "wk", cut(y.wk, H, kv0 * HD, nh_kv * HD, 0, H));
                im.same(P + "wv", cut(y.wv, H, kv0 * HD, nh_kv * HD, 0, H));
                im.same(P + "wo", cut(y.wo, nq, 0, H, q0 * HD, nh_q * HD));
                im.same(P + "q_l2_clip", cut(y.q_clip, HD, q0, nh_q, 0, HD));
                im.same(P + "rel_l2_clip", cut(y.rel_clip, RD, q0, nh_q, 0, RD));
                im.same(P + "rel_logits_proj", bf16_bytes(y.rel_proj));
                if (mm.c.global[l])
                {
                    im.same(P + "k_sconv.static_w", cut(y.k_static, nk, 0, 4, kv0 * HD, nh_kv * HD));
                    im.same(P + "v_sconv.weight_proj", cut(y.v_wproj, H, kv0 * 4, nh_kv * 4, 0, H));
                    im.same(P + "q_input_norm", bf16_bytes(y.q_norm));
                }
                else
                {
                    im.same(P + "k_sconv", cut(y.k_sconv, 4, kv0 * HD, nh_kv * HD, 0, 4));
                    im.same(P + "attn_norm", bf16_bytes(y.attn_norm));
                }
                if (l) im.same(P + "attn_res.attn_query", f32_bytes(y.attn_query));
                im.same(P + "attn_res.mlp_query", f32_bytes(y.mlp_query));
                im.same(P + "attn_sconv", bf16_bytes(y.attn_sconv));
                im.same(P + "mlp_sconv", bf16_bytes(y.mlp_sconv));
                im.same(P + "mlp_norm", bf16_bytes(y.mlp_norm));
                im.same(P + "router.w", bf16_bytes(y.router_w));
                im.same(P + "router.bias", f32_bytes(y.router_bias));
            }
            const MtpWeights& y = w.mtp;
            im.same("MTP.fuse_weight", bf16_bytes(y.fuse_weight));
            im.same("MTP.mask_embed", bf16_bytes(y.mask_embed));
            im.same("MTP.norm.weight", bf16_bytes(y.norm));
            im.same("MTP.markov_head.dflare_norm.weight", bf16_bytes(y.head_dflare_norm));
            im.same("MTP.markov_head.rnn_proj.weight", bf16_bytes(y.rnn_w));
            im.same("MTP.markov_head.rnn_proj.bias", bf16_bytes(y.rnn_b));
            im.same("MTP.markov_head.hidden_out.weight", bf16_bytes(y.hidden_out_w));
            im.same("MTP.markov_head.hidden_out.bias", bf16_bytes(y.hidden_out_b));
            im.same("MTP.markov_head.markov_w1", bf16_bytes(y.markov_w1));
            for (int m = 0; m < ML; ++m)
            {
                const MtpLayer& l = y.layers[m];
                char p[16];
                snprintf(p, sizeof p, "MTP.L%02d.", m);
                const std::string P = p;
                im.same(P + "dflare_norm.weight", bf16_bytes(l.dflare_norm));
                im.same(P + "input_layernorm.weight", bf16_bytes(l.input_norm));
                im.same(P + "post_attention_layernorm.weight", bf16_bytes(l.post_norm));
                im.same(P + "self_attn.q_norm.weight", bf16_bytes(l.q_norm));
                im.same(P + "self_attn.k_norm.weight", bf16_bytes(l.k_norm));
                im.same(P + "self_attn.k_norm_ctx.weight", bf16_bytes(l.k_norm_ctx));
                im.same(P + "self_attn.q_proj.weight", cut(l.wq, H, q0 * HD, nh_q * HD, 0, H));
                im.same(P + "self_attn.k_proj.weight", cut(l.wk, H, kv0 * HD, nh_kv * HD, 0, H));
                im.same(P + "self_attn.k_proj_ctx.weight", cut(l.k_ctx, H, kv0 * HD, nh_kv * HD, 0, H));
                im.same(P + "self_attn.v_proj.weight", cut(l.wv, H, kv0 * HD, nh_kv * HD, 0, H));
                im.same(P + "self_attn.v_proj_ctx.weight", cut(l.v_ctx, H, kv0 * HD, nh_kv * HD, 0, H));
                im.same(P + "self_attn.o_proj.weight", cut(l.wo, nq, 0, H, q0 * HD, nh_q * HD));
                im.same(P + "mlp.gate_proj.weight", cut(l.gate, H, i0, is, 0, H));
                im.same(P + "mlp.up_proj.weight", cut(l.up, H, i0, is, 0, H));
                im.same(P + "mlp.down_proj.weight", cut(l.down, MI, 0, H, i0, is));
            }
            // the vision tower: every tensor whole on every rank, and none without the tower
            const std::vector<StItem> vis = vision_items(mm);
            check(vis.size() == (with_vision ? static_cast<size_t>(8 + 18 * VL) : 0), im.tag + ": vision tensor count");
            for (const StItem& t : vis)
            {
                const std::string name = "V." + t.name.substr(strlen("model.visual."));
                im.same(name, t.bytes);
                const g6::Entry* en = im.pack.find(name);
                check(en && en->dtype == g6::DType::BF16 && en->shape.size() == t.shape.size(), im.tag + ": " + name + " dtype or rank");
            }
            check((im.pack.find("V.proj_out.weight") != nullptr) == with_vision, im.tag + ": V.proj_out.weight presence");
            g6l_plan_free(im.plan);
            unlink(man.c_str());
            ++images;
        }
    for (const char* f : {"config.json", "hf_quant_config.json", "model.safetensors", "mtp.safetensors"}) unlink((d + "/" + f).c_str());
    rmdir(dir);
    return images;
}

int main()
{
    // the tower does not change a trunk or drafter draw
    {
        Mini a, b;
        build_mini(a);
        build_mini(b, true);
        check(a.w.embed == b.w.embed && a.w.layers[NL - 1].wo == b.w.layers[NL - 1].wo && a.w.mtp.hidden_out_b == b.w.mtp.hidden_out_b, "trunk draws with vision");
        check(a.w.vis.layers.empty() && a.c.image_token_id == -1 && b.c.image_token_id == IMG && b.w.vis.layers.size() == VL, "vision config");
    }
    const int plain = run(false), vision = run(true);
    if (fails)
    {
        fprintf(stderr, "test_ink2_mini_ckpt: %d FAILED\n", fails);
        return 1;
    }
    printf("test_ink2_mini_ckpt: ok (%d rank images of 1, 4 and 6 ranks, and %d with the vision tower)\n", plain, vision);
    return 0;
}
