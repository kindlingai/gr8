// The GLM-5.3-Flash vision tower (flash_vision.cuh) against transformers' fp32 run (engine/ref/flash/vision_golden.py).
//
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -I engine/runtime -o bin/flash_vision_test \
//        engine/runtime/tests/flash_vision_test.cu -lcublas
//   bin/flash_vision_test RELEASE/dense/vision-bf16.safetensors engine/ref/flash/fixtures/vision
//
// Needs a GPU and about 2 GB. For each fixture image (a red disc, a bar chart, a photo-like scene; 12x12, 16x20 and
// 14x18 patches):
//  - the output matches the golden data through 64 fixed projections of every row: relative Frobenius error <=
//    TOL_SCALE x the reference bf16 run's own error against fp32, and the median row error (of the projected rows) <=
//    TOL_SCALE x the bf16 run's. The 8 sampled rows in full are reported and left unjudged: the tower's
//    massive-activation patches move with the rounding, so a few rows differ between any two bf16 runs
//    (docs/FLASH-VISION.md), and a sample of 8 can hold one of them;
//  - a second encode of the same pixels gives the same bits (every TP rank runs the tower and must agree).
// Then it times grids of 1024, 4096 and 8000 tokens. Exit 0 = pass.
#include "../flash_vision.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace glm5v;

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

template <class A, class B>
static double rel_fro(const A& a, const B& ref, size_t off = 0, size_t n = 0)
{
    if (!n) n = ref.size();
    double e = 0, r = 0;
    for (size_t i = off; i < off + n; ++i) { e += ((double)a[i] - ref[i]) * ((double)a[i] - ref[i]); r += (double)ref[i] * ref[i]; }
    return std::sqrt(e / r);
}

// the median over rows of each row's relative error (rows of `w` values)
template <class A>
static double median_row(const A& a, const std::vector<float>& ref, int rows, int w)
{
    std::vector<double> e;
    for (int r = 0; r < rows; ++r) e.push_back(rel_fro(a, ref, (size_t)r * w, (size_t)w));
    std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
    return e[e.size() / 2];
}

static std::vector<uint16_t> d2h(const bf16* p, size_t n)
{
    std::vector<uint16_t> v(n);
    GLM5V_CUDA(cudaMemcpy(v.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return v;
}
static float bf(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; std::memcpy(&f, &x, 4); return f; }

int main(int argc, char** argv)
{
    if (argc != 3) { std::fprintf(stderr, "usage: %s vision-bf16.safetensors FIXTURE_DIR\n", argv[0]); return 2; }
    const std::string wpath = argv[1], dir = argv[2];
    int fails = 0;
    try {
        size_t f0 = 0, tot = 0;
        GLM5V_CUDA(cudaMemGetInfo(&f0, &tot));
        Tower t;
        t.load(wpath);
        t.ensure_workspace();
        size_t f1 = 0;
        GLM5V_CUDA(cudaMemGetInfo(&f1, &tot));
        std::printf("weights %.1f MiB, workspace %.1f MiB, %.1f MiB less free on the device\n", t.w.bytes / 1048576.0,
                    t.workspace_bytes() / 1048576.0, (double)(f0 - f1) / 1048576.0);
        cudaStream_t st;
        GLM5V_CUDA(cudaStreamCreate(&st));
        bf16* out = nullptr;
        GLM5V_CUDA(cudaMalloc(&out, (size_t)MAX_TOKENS * OD * 2));
        const auto R = proj_matrix(OD, 64);
        for (const char* name : {"red_disc_150x150", "bar_chart_280x200", "photo_scene_250x180"}) {
            const g6v::Json j = g6v::Json::parse(slurp(dir + "/" + name + ".json"));
            const int vh = (int)j.at("vit_h").i64(), vw = (int)j.at("vit_w").i64(), T = (int)j.at("tokens").i64();
            const int nproj = (int)j.at("nproj").i64();
            const double tol = TOL_SCALE * j.at("bf16_vs_f32").at("rel_fro").num();
            int pw = 0, ph = 0;
            const auto px = read_ppm(dir + "/" + name + ".ppm", pw, ph);
            bool ok = pw == vw * PATCH && ph == vh * PATCH && span_of(vh, vw) == T && nproj == 64 && (int)j.at("dim").i64() == OD;
            if (!ok) { std::printf("%s: FAIL geometry\n", name); ++fails; continue; }
            const auto t0 = std::chrono::steady_clock::now();
            t.encode(px.data(), vh, vw, out, st);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto y = d2h(out, (size_t)T * OD);
            // sampled rows in full
            std::vector<int> rows;
            for (const auto& r : j.at("rows").a) rows.push_back((int)r.i64());
            const auto ref = read_f32(dir + "/" + name + ".ref.bin", rows.size() * OD);
            std::vector<double> a;
            for (size_t i = 0; i < rows.size(); ++i)
                for (int h = 0; h < OD; ++h) a.push_back(bf(y[(size_t)rows[i] * OD + h]));
            const double e_rows = rel_fro(a, ref);
            // every row through the projections, against the fp32 run and the bf16 run
            const auto pref = read_f32(dir + "/" + name + ".proj.bin", (size_t)T * nproj);
            const auto pbf = read_f32(dir + "/" + name + ".proj_bf16.bin", (size_t)T * nproj);
            std::vector<double> pa((size_t)T * nproj, 0.0);
            for (int r = 0; r < T; ++r)
                for (int h = 0; h < OD; ++h) {
                    const double v = bf(y[(size_t)r * OD + h]);
                    for (int c = 0; c < nproj; ++c) pa[(size_t)r * nproj + c] += v * R[(size_t)h * nproj + c];
                }
            const double e_proj = rel_fro(pa, pref), e_bf = rel_fro(pa, pbf), bf_proj = rel_fro(pbf, pref);
            const double med = median_row(pa, pref, T, nproj), med_bf = median_row(pbf, pref, T, nproj);
            // the same bits again
            t.encode(px.data(), vh, vw, out, st);
            const bool same = d2h(out, (size_t)T * OD) == y;
            ok = e_proj <= tol && med <= TOL_SCALE * med_bf && same;
            std::printf("%s: grid %dx%d, %d tokens: vs fp32 rows rel %.4f, proj rel %.4f (tol %.4f = %.1f x bf16 run %.4f; "
                        "its proj rel %.4f), median row %.4f (bf16 run %.4f), vs bf16 run proj rel %.4f, repeat %s, "
                        "%.1f ms (first run): %s\n",
                        name, vh, vw, T, e_rows, e_proj, tol, TOL_SCALE, tol / TOL_SCALE, bf_proj, med, med_bf, e_bf,
                        same ? "identical" : "DIFFERS", ms, ok ? "PASS" : "FAIL");
            fails += !ok;
        }
        // the cost of larger images (timing only: grey pixels)
        for (const auto hw : {std::pair<int, int>{64, 64}, std::pair<int, int>{128, 128}, std::pair<int, int>{160, 200}}) {
            std::vector<uint8_t> px((size_t)hw.first * hw.second * PATCH * PATCH * 3, 127);
            t.encode(px.data(), hw.first, hw.second, out, st);
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 3; ++k) t.encode(px.data(), hw.first, hw.second, out, st);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3;
            std::printf("timing: grid %dx%d (%d patches, %d tokens): %.1f ms per image\n", hw.first, hw.second,
                        hw.first * hw.second, span_of(hw.first, hw.second), ms);
        }
        cudaFree(out);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf(fails ? "flash_vision_test: %d FAILED\n" : "flash_vision_test: all passed\n", fails);
    return fails ? 1 : 0;
}
