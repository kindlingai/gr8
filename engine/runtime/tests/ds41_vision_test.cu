// The DS41 vision tower + aligner (ds41_vision.cuh) against the official code's fp32 run (engine/ref/ds41/vision_golden.py).
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I engine/runtime -o bin/ds41_vision_test \
//        engine/runtime/tests/ds41_vision_test.cu -lcublas
//   bin/ds41_vision_test RELEASE/dense/exlr8-extra-vision.safetensors engine/ref/ds41/fixtures/vision
//
// One GPU, about 1.6 GB. For each fixture image (three patch grids: 39x39, 20x79, 72x23):
//  - the span's delimiter rows are the learned image_start / image_newline / image_end bit for bit, in the official order;
//  - the aligner rows match the golden rows in full (8 sampled rows) and through 64 fixed projections of every row:
//    relative Frobenius error <= TOL_SCALE x the official bf16 run's own error against fp32 (the json's bf16_vs_f32);
//  - a second encode of the same pixels gives the same bits (every TP rank runs the tower and must agree).
// Then it times the largest grids (1 x 1021 and 31 x 31 tokens) for the prefill-cost estimate. Exit 0 = pass.
#include "../ds41_vision.cuh"

#include <chrono>
#include <fstream>
#include <sstream>

using namespace ds41v;

static constexpr double TOL_SCALE = 1.5;

static std::string slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

static std::vector<uint8_t> read_ppm(const std::string& p, int& w, int& h)
{
    const std::string s = slurp(p);
    std::istringstream in(s);
    std::string magic;
    int mx = 0;
    in >> magic >> w >> h >> mx;
    if (magic != "P6" || mx != 255) throw std::runtime_error(p + ": not a binary 8-bit PPM");
    const size_t at = (size_t)in.tellg() + 1, n = (size_t)w * h * 3;
    if (s.size() < at + n) throw std::runtime_error(p + ": short");
    return std::vector<uint8_t>(s.begin() + at, s.begin() + at + n);
}

static std::vector<float> read_f32(const std::string& p, size_t n)
{
    const std::string s = slurp(p);
    if (s.size() != n * 4) throw std::runtime_error(p + ": expected " + std::to_string(n * 4) + " bytes, got " + std::to_string(s.size()));
    std::vector<float> v(n);
    std::memcpy(v.data(), s.data(), n * 4);
    return v;
}

// vision_golden.py proj_matrix: [dim][nproj] of +-1/sqrt(dim), xorshift32 from 0x9E3779B9, one bit per entry
static std::vector<float> proj_matrix(int dim, int nproj)
{
    uint32_t s = 0x9E3779B9u;
    std::vector<float> m((size_t)dim * nproj);
    const float v = (float)(1.0 / std::sqrt((double)dim));
    for (size_t i = 0; i < m.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        m[i] = (s & 1) ? v : -v;
    }
    return m;
}

static double rel_fro(const std::vector<double>& a, const std::vector<double>& ref)
{
    double e = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { e += (a[i] - ref[i]) * (a[i] - ref[i]); r += ref[i] * ref[i]; }
    return std::sqrt(e / r);
}

static std::vector<uint16_t> d2h(const bf16* p, size_t n)
{
    std::vector<uint16_t> v(n);
    DS41V_CUDA(cudaMemcpy(v.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return v;
}
static float bf(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; std::memcpy(&f, &x, 4); return f; }

int main(int argc, char** argv)
{
    if (argc != 3) { std::fprintf(stderr, "usage: %s VISION.safetensors FIXTURE_DIR\n", argv[0]); return 2; }
    const std::string wpath = argv[1], dir = argv[2];
    int fails = 0;
    try {
        Tower t;
        t.load(wpath);
        t.ensure_workspace();
        std::printf("weights %.3f GB on the device, workspace %.3f GB\n", t.w.bytes / 1e9, t.workspace_bytes() / 1e9);
        cudaStream_t st;
        DS41V_CUDA(cudaStreamCreate(&st));
        bf16* span = nullptr;
        DS41V_CUDA(cudaMalloc(&span, (size_t)MAX_TOKENS * OD * 2));
        const auto R = proj_matrix(OD, 64);
        const auto st_row = d2h(t.w.img_start, OD), nl_row = d2h(t.w.img_nl, OD), en_row = d2h(t.w.img_end, OD);
        for (const char* name : {"square_300x300", "wide_1100x280", "tall_320x1000"}) {
            const g6v::Json j = g6v::Json::parse(slurp(dir + "/" + name + ".json"));
            const int vh = (int)j.at("n_vit_h").i64(), vw = (int)j.at("n_vit_w").i64(), lh = (int)j.at("n_llm_h").i64(), lw = (int)j.at("n_llm_w").i64();
            const int nproj = (int)j.at("nproj").i64();
            const double tol = TOL_SCALE * j.at("bf16_vs_f32").at("rel_fro").num();
            int pw = 0, ph = 0;
            const auto px = read_ppm(dir + "/" + name + ".ppm", pw, ph);
            const Grid g = grid_of(vh, vw);
            bool ok = pw == vw * PATCH && ph == vh * PATCH && g.ok && g.llm_h == lh && g.llm_w == lw && g.span() == (int)j.at("n_span_tokens").i64() && nproj == 64;
            if (!ok) { std::printf("%s: FAIL geometry\n", name); ++fails; continue; }
            const auto t0 = std::chrono::steady_clock::now();
            t.encode(px.data(), vh, vw, span, st);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const int L = lh * lw, S = g.span();
            const auto sp = d2h(span, (size_t)S * OD);
            const auto al = d2h(t.aligner_rows(), (size_t)L * OD);
            // delimiters and the placement of the aligner rows
            int bad_rows = 0;
            for (int s = 0, k = 0; s < S; ++s) {
                const int ty = g.type(s);
                const uint16_t* want = ty == 0 ? st_row.data() : ty == 2 ? nl_row.data() : ty == 3 ? en_row.data() : al.data() + (size_t)(k++) * OD;
                if (std::memcmp(sp.data() + (size_t)s * OD, want, OD * 2)) ++bad_rows;
            }
            // sampled rows in full
            std::vector<int> rows;
            for (const auto& r : j.at("rows").a) rows.push_back((int)r.i64());
            const auto ref = read_f32(dir + "/" + name + ".ref.bin", rows.size() * OD);
            std::vector<double> a, b;
            for (size_t i = 0; i < rows.size(); ++i)
                for (int h = 0; h < OD; ++h) { a.push_back(bf(al[(size_t)rows[i] * OD + h])); b.push_back(ref[i * OD + h]); }
            const double e_rows = rel_fro(a, b);
            // every row through the projections
            const auto pref = read_f32(dir + "/" + name + ".proj.bin", (size_t)L * nproj);
            std::vector<double> pa((size_t)L * nproj, 0.0), pb(pref.begin(), pref.end());
            for (int r = 0; r < L; ++r)
                for (int h = 0; h < OD; ++h) {
                    const double v = bf(al[(size_t)r * OD + h]);
                    for (int c = 0; c < nproj; ++c) pa[(size_t)r * nproj + c] += v * R[(size_t)h * nproj + c];
                }
            const double e_proj = rel_fro(pa, pb);
            // the same bits again
            t.encode(px.data(), vh, vw, span, st);
            const bool same = d2h(span, (size_t)S * OD) == sp;
            ok = bad_rows == 0 && e_rows <= tol && e_proj <= tol && same;
            std::printf("%s: grid %dx%d -> %dx%d, span %d: rows rel %.4f, proj rel %.4f (tol %.4f = %.1f x official bf16 %.4f), "
                        "span rows misplaced %d, repeat %s, %.1f ms (first run, includes cuBLAS warmup on the first image): %s\n",
                        name, vh, vw, lh, lw, S, e_rows, e_proj, tol, TOL_SCALE, tol / TOL_SCALE, bad_rows, same ? "identical" : "DIFFERS", ms, ok ? "PASS" : "FAIL");
            fails += !ok;
        }
        // the cost of the largest images (timing only: gray pixels)
        for (const auto hw : {std::pair<int, int>{3, 3063}, std::pair<int, int>{93, 93}}) {
            const Grid g = grid_of(hw.first, hw.second);
            std::vector<uint8_t> px((size_t)g.patches() * PATCH * PATCH * 3, 127);
            t.encode(px.data(), hw.first, hw.second, span, st);
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 3; ++k) t.encode(px.data(), hw.first, hw.second, span, st);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3;
            std::printf("timing: grid %dx%d (%d patches, %d span tokens): %.1f ms per image\n", hw.first, hw.second, g.patches(), g.span(), ms);
        }
        cudaFree(span);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf(fails ? "ds41_vision_test: %d FAILED\n" : "ds41_vision_test: all passed\n", fails);
    return fails ? 1 : 0;
}
