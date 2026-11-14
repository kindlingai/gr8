// Inkling 2 Small image input on the device, against the CPU reference: a miniature checkpoint directory with a vision
// tower (ink2_mini_ckpt.h build_mini(mm, true): 2 tower layers of 256 wide, 4 heads of 64, a 384-wide projector, the
// image token 760) is written to /tmp, planned by the loader and run by ink2rt::Engine. The tower's kernels are those
// of the full model (ink2_vision.cuh takes its sizes from config.json).
//
//   1. The tower alone (Engine::encode_vision) against ink2ref::Model::encode_image on four images: 2 x 2 patches (one
//      token), 4 x 6 (not square), 4 x 4 with a content of 70 x 65 pixels (the rest is padding, so the tower reads -1
//      there) and 6 x 4. For each: the largest difference is at most 2e-2 * max(1, the largest reference value), and
//      the cosine of the two outputs is above 0.9999. The device rounds the attention probabilities to BF16 and the
//      reference keeps them in fp32, so the two are not equal bit for bit.
//   2. The staging of a prefill launch (Engine::stage_vision), a 12-token image at prompt position 5: one launch of 12
//      rows; then launches of 6 + 6 rows; then launches of 8 + 8 rows from position 3, whose rows outside the image are
//      zeros. Every image row is the same bits each time, and the tower ran once.
//   3. The whole model, a prompt of 26 tokens with a 6-token image at position 7, on 1, 4 and 6 simulated ranks:
//      * A single prefill launch of all 26 rows and chunked launches of 8 + 8 + 8 + 2 rows (so the image straddles the
//        first two launches) must give the same 750 valid logits and greedy token at every prompt row, bit for bit.
//      * Against ink2ref::Model::forward_with_images (and teacher-forced ink2ref::Model::forward over 12 generated
//        tokens) at the same Config::tp: the rows before the image hold both the argmax rule (a reference gap of at
//        most max(|top|, 1) / 32 counts as a near tie) and row cosine >= 0.75; from the image on, the tower's BF16
//        attention probabilities differ from the reference's fp32 ones by one BF16 step, which the 4-bit and 8-bit
//        dynamic activation quantizers of the 6-layer MoE trunk amplify (see engine/tests/ink2_layers_gpu_test.cu), so
//        both the 19 post-image prompt rows (p >= im0, checked separately from pre-image and decode rows) and the 11
//        teacher-forced decode steps (checked separately) are held to valid-logit cosine >= 0.85 and relative L2 <= 0.55,
//        with every row's cosine >= 0.75.
//      * On each rank count, the engine with the 3-layer MTP drafter: plain steps and 3 speculative drafts a round must
//        give the exact tokens of the run without the drafter.
//
// Build (from engine/runtime, after scripts/build.sh has built the Rust libraries and bin/obj/g6pack.o):
//   nvcc -O3 -std=c++17 -Xcompiler -ffp-contract=off -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_INK2=1 -DDEC_S2 \
//     -DG6_RS_LOADER -I . -I ../kernels -I ../attn -I ../ref/ink2 -o ../../bin/ink2_vision_test ../tests/ink2_vision_test.cu \
//     ../../bin/obj/g6pack.o ../loader-rs/target/release/libg6load.a ../knobs-rs/target/release/libgr8knobs.a \
//     ../target/release/libg6host.a -lcublas -lpthread -ldl -lm
// Run: bin/ink2_vision_test   (one GPU; prints PASS and returns 0)
#include "../runtime/ink2_step.cuh"
#include "ink2_mini_ckpt.h"
#include <filesystem>
#include <memory>

using namespace ink2mini;
using ink2rt::Engine;
using ink2rt::Options;
using ink2rt::RowOut;
using ink2rt::RowSample;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { fprintf(stderr, "FATAL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } \
    } while (0)

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    if (ok) return;
    ++fails;
    if (fails <= 40) fprintf(stderr, "FAIL %s\n", what.c_str());
}

constexpr int SLOTS = 1, CAP = 192, CHUNK = 8, GEN = 12, PX = ink2v::PATCH;

static std::unique_ptr<Engine> engine(const std::string& dir, int T, int mtp)
{
    Options o;
    o.nr = T;
    o.rank0 = 0;
    o.nloc = T;
    o.slots = SLOTS;
    o.mtp = mtp;
    o.max_rows = 32;
    o.cap = CAP;
    o.nc = VALID;
    o.vision = true;
    std::unique_ptr<Engine> e(new Engine);
    e->init(dir, o);
    return e;
}

// The pixels of a vit_h x vit_w patch grid, as the front end sends them.
static std::shared_ptr<const std::vector<uint8_t>> pixels(Rng& r, int vit_h, int vit_w)
{
    auto v = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(vit_h) * PX * vit_w * PX * 3);
    for (uint8_t& x : *v) x = static_cast<uint8_t>(r.next() & 255u);
    return v;
}

static g6rt::ImageIn image(Rng& r, int pos, int vit_h, int vit_w)
{
    g6rt::ImageIn im;
    im.pos = pos;
    im.vit_h = vit_h;
    im.vit_w = vit_w;
    im.span = ink2v::span_of(vit_h, vit_w);
    im.rgb = pixels(r, vit_h, vit_w);
    return im;
}

struct DevRows
{
    uint16_t* p = nullptr;
    explicit DevRows(size_t cells) { CK(cudaMalloc(&p, cells * sizeof(uint16_t))); }
    ~DevRows() { cudaFree(p); }
    DevRows(const DevRows&) = delete;
    DevRows& operator=(const DevRows&) = delete;
};

static std::vector<uint16_t> pull(const uint16_t* dev, size_t cells)
{
    std::vector<uint16_t> v(cells);
    if (cells) CK(cudaMemcpy(v.data(), dev, cells * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    return v;
}

static std::string list(const std::vector<int>& v)
{
    std::string s;
    for (int x : v) s += " " + std::to_string(x);
    return s;
}

// ---------------------------------------------------------------------------------------------- check 1
static void check_tower(const Mini& mm, Engine& e)
{
    const ink2ref::Model ref(mm.c, mm.w);
    Rng r{0x76697331};
    struct Case { int vit_h, vit_w, content_h, content_w; };
    for (const Case& k : {Case{2, 2, -1, -1}, Case{4, 6, -1, -1}, Case{4, 4, 70, 65}, Case{6, 4, -1, -1}})
    {
        const std::string tag = "tower " + std::to_string(k.vit_h) + "x" + std::to_string(k.vit_w) + (k.content_h < 0 ? "" : " padded");
        const auto rgb = pixels(r, k.vit_h, k.vit_w);
        const int tokens = ink2v::span_of(k.vit_h, k.vit_w);
        const size_t cells = static_cast<size_t>(tokens) * H;
        DevRows d(cells);
        e.encode_vision(rgb->data(), k.vit_h, k.vit_w, d.p, k.content_h, k.content_w);
        const std::vector<uint16_t> got = pull(d.p, cells);
        const std::vector<float> want = ref.encode_image(rgb->data(), k.vit_h, k.vit_w, k.content_h, k.content_w);
        check(want.size() == cells, tag + ": the reference's rows");
        if (want.size() != cells) continue;
        double dot = 0, na = 0, nb = 0;
        float worst = 0, top = 0;
        bool finite = true;
        for (size_t i = 0; i < cells; ++i)
        {
            const float a = ink2ref::bf16_to_f32(got[i]), b = want[i];
            finite &= std::isfinite(a);
            worst = std::max(worst, std::fabs(a - b));
            top = std::max(top, std::fabs(b));
            dot += static_cast<double>(a) * b;
            na += static_cast<double>(a) * a;
            nb += static_cast<double>(b) * b;
        }
        const double cosine = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
        const float tol = 2e-2f * std::max(1.0f, top);
        printf("  %s: %d tokens, largest difference %.5f (limit %.5f, largest value %.4f), cosine %.6f\n", tag.c_str(), tokens, worst, tol, top, cosine);
        check(finite, tag + ": a value is not finite");
        check(worst <= tol, tag + ": largest difference " + std::to_string(worst));
        check(cosine > 0.9999, tag + ": cosine " + std::to_string(cosine));
    }
}

// ---------------------------------------------------------------------------------------------- check 2
static void check_stage(Engine& e)
{
    Rng r{0x76697332};
    const int POS = 5, N = 12;
    const std::vector<g6rt::ImageIn> images{image(r, POS, 6, 8)};
    check(images[0].span == N, "stage: a 6 x 8 grid is 12 tokens");
    const size_t row = static_cast<size_t>(H);
    const long calls0 = e.vision_encode_calls();
    // rows [first, first + rows) of the prompt, as the launch's embed would read them
    const auto launch = [&](int first, int rows) {
        const uint16_t* p = e.stage_vision(images, first, rows, 1);
        check(p != nullptr, "stage: a launch over the image got no rows");
        return p ? pull(p, static_cast<size_t>(rows) * row) : std::vector<uint16_t>(static_cast<size_t>(rows) * row, 0);
    };
    const std::vector<uint16_t> whole = launch(POS, N);
    bool some = false;
    for (uint16_t x : whole) some |= (x & 0x7fffu) != 0;
    check(some, "stage: the image's rows are all zeros");

    std::vector<uint16_t> halves = launch(POS, 6);
    const std::vector<uint16_t> second = launch(POS + 6, 6);
    halves.insert(halves.end(), second.begin(), second.end());
    check(halves == whole, "stage: 6 + 6 rows differ from 12 rows");

    // launches that start before the image and end past it: prompt positions 3 .. 10 and 11 .. 18
    const std::vector<uint16_t> a = launch(POS - 2, 8), b = launch(POS + 6, 8);
    const std::vector<uint16_t> zeros(2 * row, 0);
    check(std::equal(zeros.begin(), zeros.end(), a.begin()), "stage: the rows before the image are not zeros");
    check(std::equal(a.begin() + 2 * row, a.end(), whole.begin()), "stage: image rows 0 .. 5 of a launch that starts before the image");
    check(std::equal(b.begin(), b.begin() + 6 * row, whole.begin() + 6 * row), "stage: image rows 6 .. 11 of a launch that ends past the image");
    check(std::equal(zeros.begin(), zeros.end(), b.begin() + 6 * row), "stage: the rows past the image are not zeros");
    check(e.stage_vision(images, POS + N, 4, 1) == nullptr, "stage: a launch past the image got rows");
    check(e.stage_vision(images, 0, POS, 1) == nullptr, "stage: a launch before the image got rows");
    const long calls = e.vision_encode_calls() - calls0;
    check(calls == 1, "stage: the tower ran " + std::to_string(calls) + " times for one image");
    printf("  stage: a 12-token image in launches of 12, 6 + 6 and 8 + 8 rows: the same rows, the tower ran %ld time(s)\n", calls);
}

// ---------------------------------------------------------------------------------------------- check 3
constexpr double E2E_SEQ_COS = 0.85, E2E_SEQ_REL = 0.55, E2E_ROW_COS = 0.75;

struct Sess
{
    int pos = 0, next = 0;
    std::vector<int> prompt_toks;       // [P]: the greedy token of every prompt row
    std::vector<float> prompt_logits;   // [P][VALID]
    std::vector<int> said;              // [GEN]: said[0] = prompt_toks[P - 1] pushed in prefill; said[g + 1] predicted from said[g] in plain_step
    std::vector<float> gen_logits;      // [GEN - 1][VALID]: gen_logits[g] = logits of plain_step(said[g]) predicting said[g + 1]
};

static RowSample greedy_row(int pos) { return RowSample{0.f, 0.f, 0u, static_cast<uint32_t>(pos + 1)}; }

static void unpack_valid_logits(const RowOut& r, float* dst, const std::string& what)
{
    check(r.top.size() == static_cast<size_t>(VALID), what + ": candidate count " + std::to_string(r.top.size()));
    std::vector<uint8_t> seen(static_cast<size_t>(VALID), 0);
    bool ok = true;
    for (const ink2rt::Cand& c : r.top)
    {
        if (c.id < 0 || c.id >= VALID || seen[static_cast<size_t>(c.id)] || !std::isfinite(c.logit) || std::fabs(c.logit) > 30.0f)
        {
            ok = false;
            continue;
        }
        seen[static_cast<size_t>(c.id)] = 1;
        dst[c.id] = c.logit;
    }
    check(ok, what + ": invalid candidate id or logit");
}

static Sess prefill(Engine& e, const std::vector<int>& prompt, const std::vector<g6rt::ImageIn>& images, int chunk = CHUNK)
{
    Sess s;
    e.reset_slot(0);
    std::vector<RowOut> out;
    const int P = static_cast<int>(prompt.size());
    s.prompt_logits.assign(static_cast<size_t>(P) * VALID, 0.0f);
    for (int p = 0; p < P; p += chunk)
    {
        const int n = std::min(chunk, P - p);
        ink2::RowsHost rh;
        rh.add(0, p, n);
        std::vector<RowSample> sm;
        for (int i = 0; i < n; ++i) sm.push_back(greedy_row(p + i));
        e.forward(rh, &prompt[static_cast<size_t>(p)], sm.data(), out, e.stage_vision(images, p, n, 7));
        e.commit();
        for (int i = 0; i < n; ++i)
        {
            s.prompt_toks.push_back(out[static_cast<size_t>(i)].tok);
            unpack_valid_logits(out[static_cast<size_t>(i)], &s.prompt_logits[static_cast<size_t>(p + i) * VALID], "prefill row " + std::to_string(p + i));
        }
        s.next = out[static_cast<size_t>(n) - 1].tok;
    }
    s.pos = P;
    s.said.push_back(s.next);
    return s;
}

static void plain_step(Engine& e, Sess& s)
{
    ink2::RowsHost rh;
    rh.add(0, s.pos, 1);
    const RowSample sm = greedy_row(s.pos);
    std::vector<RowOut> out;
    e.forward(rh, &s.next, &sm, out);
    e.commit();
    const size_t at = s.gen_logits.size();
    s.gen_logits.resize(at + static_cast<size_t>(VALID), 0.0f);
    unpack_valid_logits(out[0], &s.gen_logits[at], "decode pos " + std::to_string(s.pos));
    s.next = out[0].tok;
    s.said.push_back(s.next);
    s.pos += 1;
}

// One speculative round with K drafts: draft, verify in one step, commit the accepted run.
static void spec_round(Engine& e, Sess& s, int K)
{
    const int n = K + 1;
    std::vector<int> d, tk;
    e.mtp_draft({0}, {s.pos - 1}, {s.next}, {RowSample{0.f, 0.f, 0u, 0u}}, K, d);
    ink2::RowsHost rh;
    rh.add(0, s.pos, n);
    tk.push_back(s.next);
    std::vector<RowSample> sm;
    for (int j = 0; j < K; ++j) tk.push_back(d[static_cast<size_t>(j)]);
    for (int j = 0; j < n; ++j) sm.push_back(greedy_row(s.pos + j));
    std::vector<RowOut> out;
    e.forward(rh, tk.data(), sm.data(), out);
    std::vector<uint8_t> acc(static_cast<size_t>(n), 0);
    int a = 0;
    while (a < K && d[static_cast<size_t>(a)] == out[static_cast<size_t>(a)].tok) ++a;
    for (int j = 0; j <= a; ++j)
    {
        acc[static_cast<size_t>(j)] = 1;
        s.said.push_back(out[static_cast<size_t>(j)].tok);
    }
    s.next = out[static_cast<size_t>(a)].tok;
    s.pos += a + 1;
    e.commit(&acc);
}

static int ref_ties = 0;
// The device's token against a row of reference logits (rows before the image, where the tower's BF16 rounding has not entered).
static void judge(const float* row, int dv, const std::string& what)
{
    int best = 0;
    for (int t = 1; t < VALID; ++t)
        if (row[t] > row[best]) best = t;
    if (dv == best) return;
    const float gap = dv >= 0 && dv < VALID ? row[best] - row[dv] : 1e30f, tol = std::max(std::fabs(row[best]), 1.0f) / 32.0f;
    if (gap <= tol)
    {
        ++ref_ties;
        printf("  %s: near tie, device %d (%.5f) reference %d (%.5f)\n", what.c_str(), dv, row[dv], best, row[best]);
    }
    else check(false, what + ": device " + std::to_string(dv) + ", reference " + std::to_string(best) + " (gap " + std::to_string(gap) + ")");
}

struct RowAcc
{
    double dot = 0, na = 0, nb = 0, nd = 0, min_row_cos = 2.0;
    size_t rows = 0;
    bool mask_ok = true;
    void add_row(const float* dv, const float* rf, const std::string& what)
    {
        ++rows;
        double rd = 0, ra = 0, rb = 0;
        for (int v = 0; v < VALID; ++v)
        {
            const double a = dv[v], b = rf[v];
            rd += a * b;
            ra += a * a;
            rb += b * b;
            nd += (a - b) * (a - b);
        }
        for (int v = VALID; v < VOCAB; ++v) mask_ok = mask_ok && std::isinf(rf[v]) && rf[v] < 0.0f;
        dot += rd;
        na += ra;
        nb += rb;
        if (!(ra > 0 && rb > 0))
        {
            check(false, what + ": degenerate all-zero valid logits");
            return;
        }
        const double rc = rd / std::sqrt(ra * rb);
        min_row_cos = std::min(min_row_cos, rc);
        check(rc >= E2E_ROW_COS, what + ": row cosine " + std::to_string(rc));
    }
    double cos() const { return na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0; }
    double rel() const { return nb > 0 ? std::sqrt(nd / nb) : 1.0; }
};

static void check_model(const Mini& mm, Engine& plain, Engine& spec, const std::vector<int>& prompt, const std::vector<g6rt::ImageIn>& images)
{
    const int T = plain.o.nr, P = static_cast<int>(prompt.size()), im0 = images[0].pos;
    const std::string tag = std::to_string(T) + " rank(s)";

    // one-launch prefill (26 rows) vs straddling 8-row chunks (8 + 8 + 8 + 2): bit for bit on all 26 x 750 valid logits
    const Sess whole = prefill(plain, prompt, images, P);
    Sess s = prefill(plain, prompt, images, CHUNK);
    check(whole.prompt_toks == s.prompt_toks && whole.prompt_logits.size() == s.prompt_logits.size() &&
              memcmp(whole.prompt_logits.data(), s.prompt_logits.data(), whole.prompt_logits.size() * sizeof(float)) == 0,
          tag + ": 8-row straddling chunks differ from the single-launch prefill");
    while (static_cast<int>(s.said.size()) < GEN) plain_step(plain, s);

    Config c = mm.c;
    c.tp = T;
    ink2ref::Model ref(c, mm.w);
    std::vector<ink2ref::ImageSpan> spans;
    for (const g6rt::ImageIn& im : images)
    {
        ink2ref::ImageSpan sp;
        sp.pos = im.pos;
        sp.vit_h = im.vit_h;
        sp.vit_w = im.vit_w;
        sp.rgb = im.rgb->data();
        spans.push_back(sp);
    }
    const std::vector<float> lg = ref.forward_with_images(prompt.data(), P, spans);
    RowAcc pre_pf, post_pf, gen_dec;
    for (int p = 0; p < P; ++p)
    {
        const std::string rw = tag + ": prompt row " + std::to_string(p);
        if (p < im0)
        {
            judge(&lg[static_cast<size_t>(p) * VOCAB], s.prompt_toks[static_cast<size_t>(p)], rw);
            pre_pf.add_row(&s.prompt_logits[static_cast<size_t>(p) * VALID], &lg[static_cast<size_t>(p) * VOCAB], rw);
        }
        else post_pf.add_row(&s.prompt_logits[static_cast<size_t>(p) * VALID], &lg[static_cast<size_t>(p) * VOCAB], rw);
    }
    check(s.gen_logits.size() == static_cast<size_t>(GEN - 1) * VALID, tag + ": gen_logits size");
    for (int g = 0; g + 1 < GEN; ++g)
    {
        // prefill pushed said[0] = prompt_toks[P - 1]; plain_step(g) fed said[g] at pos P + g and wrote gen_logits[g] predicting said[g + 1]
        const std::vector<float> row = ref.forward(&s.said[static_cast<size_t>(g)], 1);
        gen_dec.add_row(&s.gen_logits[static_cast<size_t>(g) * VALID], row.data(), tag + ": generated token " + std::to_string(g + 1));
    }
    check(pre_pf.rows > 0 && post_pf.rows > 0 && gen_dec.rows > 0 && pre_pf.mask_ok && post_pf.mask_ok && gen_dec.mask_ok,
          tag + ": reference valid_vocab mask");
    check(post_pf.cos() >= E2E_SEQ_COS && post_pf.rel() <= E2E_SEQ_REL,
          tag + ": post-image prefill cosine " + std::to_string(post_pf.cos()) + ", rel " + std::to_string(post_pf.rel()));
    check(gen_dec.cos() >= E2E_SEQ_COS && gen_dec.rel() <= E2E_SEQ_REL,
          tag + ": decode cosine " + std::to_string(gen_dec.cos()) + ", rel " + std::to_string(gen_dec.rel()));
    printf("  %s: single launch == 8-row straddling chunks (exact bits); post-image prefill cos %.6f (rel %.5f), decode cos %.6f (rel %.5f), min row %.6f; tokens:%s\n",
           tag.c_str(), post_pf.cos(), post_pf.rel(), gen_dec.cos(), gen_dec.rel(),
           std::min({pre_pf.min_row_cos, post_pf.min_row_cos, gen_dec.min_row_cos}), list(s.said).c_str());

    // the engine with the drafter: plain steps, then 3 drafts a round
    Sess a = prefill(spec, prompt, images, CHUNK);
    while (static_cast<int>(a.said.size()) < GEN) plain_step(spec, a);
    check(a.prompt_toks == s.prompt_toks && a.said == s.said, tag + ": plain steps on the drafter's engine:" + list(a.said) + " expected" + list(s.said));
    Sess b = prefill(spec, prompt, images, CHUNK);
    while (static_cast<int>(b.said.size()) < GEN) spec_round(spec, b, 3);
    b.said.resize(static_cast<size_t>(GEN));
    check(b.said == s.said, tag + ": 3 drafts a round:" + list(b.said) + " expected" + list(s.said));
    printf("  %s: 3 drafts a round give the same %d tokens\n", tag.c_str(), GEN);
}

int main()
{
    knob::init();
    Mini mm;
    build_mini(mm, true);
    char tmpl[] = "/tmp/ink2-vision-test.XXXXXX";
    if (!mkdtemp(tmpl)) { perror("mkdtemp"); return 2; }
    const std::string tmp = tmpl, dir = tmp + "/ckpt";
    write_ckpt(mm, dir);

    // the prompt of check 3: 7 text tokens, a 4 x 6 patch image (6 tokens), 13 text tokens
    Rng r{0x76697333};
    std::vector<int> prompt;
    for (int i = 0; i < 26; ++i) prompt.push_back(static_cast<int>(r.next() % VALID));
    const std::vector<g6rt::ImageIn> images{image(r, 7, 4, 6)};
    for (int i = 0; i < images[0].span; ++i) prompt[static_cast<size_t>(images[0].pos + i)] = IMG;

    std::unique_ptr<Engine> e0 = engine(dir, 1, 0);
    const std::string vid0 = e0->vision_id();
    check(e0->vision_ready() && vid0.size() == 16, "the engine of 1 rank(s) runs no vision tower");
    if (!fails)
    {
        printf("1. the tower against the CPU reference (id %s)\n", vid0.c_str());
        check_tower(mm, *e0);
        printf("2. the staging of a prefill launch\n");
        check_stage(*e0);
    }
    e0.reset();

    if (!fails)
    {
        printf("3. the model with an image against the CPU reference, and with the drafter\n");
        for (int T : {1, 4, 6})
        {
            std::unique_ptr<Engine> plain = engine(dir, T, 0), spec = engine(dir, T, 3);
            check(plain->vision_ready() && spec->vision_ready(), "the engine of " + std::to_string(T) + " rank(s) runs no vision tower");
            check(plain->vision_id() == vid0 && spec->vision_id() == vid0, "the tower's id differs between the rank counts");
            if (!fails) check_model(mm, *plain, *spec, prompt, images);
        }
        printf("  %d near ties on pre-image rows against the reference\n", ref_ties);
    }
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(tmpl), ec);
    if (fails)
    {
        printf("ink2_vision_test: %d FAILED\n", fails);
        return 1;
    }
    printf("ink2_vision_test: PASS\n");
    return 0;
}
