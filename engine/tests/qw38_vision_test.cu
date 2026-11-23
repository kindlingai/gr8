// Standalone test for Qwen3.8-Flash-Next vision tower (qw38_vision.cuh) and 3D MRoPE coordinates.
//
// Usage:
//   1. Self-test (no weights needed): verifies qw38_vision_plan.h grid math and qw38_build_mrope / qw38_mrope_delta.
//      ./qw38_vision_test
//   2. Real-weights golden check (requires dense/vision-bf16.safetensors and optional reference .bin files):
//      ./qw38_vision_test --weights /path/to/dense/vision-bf16.safetensors \
//                         [--rgb in.rgb --vit-h 16 --vit-w 20 --ref ref_f32.bin]
//
// Build:
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -DG6_MODEL_QW38=1 \
//     -I engine/runtime -I engine/kernels -I engine/attn \
//     -o bin/qw38_vision_test engine/tests/qw38_vision_test.cu -lcublas
#include "../runtime/qw38_vision.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fails = 0;
void check(bool ok, const std::string& msg) {
    if (ok) return;
    ++fails;
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

// Mirror of qw38rt::qw38_mrope_delta / qw38_build_mrope (kept self-contained so test builds without loader-rs).
int mrope_delta(const std::vector<g6rt::ImageIn>& images) {
    int delta = 0;
    for (const auto& im : images) {
        const int lh = im.vit_h / qw38v::MERGE;
        const int lw = im.vit_w / qw38v::MERGE;
        delta += std::max(lh, lw) - im.span;
    }
    return delta;
}

void build_mrope(const std::vector<g6rt::ImageIn>& images, int first, int n, int3* out_thw) {
    int img_idx = 0;
    const int n_img = static_cast<int>(images.size());
    int delta = 0;
    while (img_idx < n_img && images[size_t(img_idx)].pos + images[size_t(img_idx)].span <= first) {
        const auto& im = images[size_t(img_idx)];
        delta += std::max(im.vit_h / qw38v::MERGE, im.vit_w / qw38v::MERGE) - im.span;
        ++img_idx;
    }
    for (int i = 0; i < n; ++i) {
        const int p = first + i;
        while (img_idx < n_img && images[size_t(img_idx)].pos + images[size_t(img_idx)].span <= p) {
            const auto& im = images[size_t(img_idx)];
            delta += std::max(im.vit_h / qw38v::MERGE, im.vit_w / qw38v::MERGE) - im.span;
            ++img_idx;
        }
        if (img_idx < n_img && p >= images[size_t(img_idx)].pos) {
            const auto& im = images[size_t(img_idx)];
            const int lw = im.vit_w / qw38v::MERGE;
            const int rel = p - im.pos;
            const int base = im.pos + delta;
            out_thw[i] = make_int3(base, base + rel / lw, base + rel % lw);
        } else {
            const int cp = p + delta;
            out_thw[i] = make_int3(cp, cp, cp);
        }
    }
}

void test_plan_and_mrope() {
    check(qw38v::span_of(2, 2) == 1, "2x2 == 1 token");
    check(qw38v::span_of(16, 16) == 64, "16x16 == 64 tokens");
    check(qw38v::span_of(0, 16) == 0, "0x16 refused");
    check(qw38v::span_of(17, 16) == 0, "odd vit_h refused");
    check(qw38v::span_of(32, 32) == 256, "32x32 == 256 tokens");
    check(qw38v::span_of(256, 256) == 16384, "256x256 == 16384 tokens (max)");
    check(qw38v::span_of(258, 256) == 0, "> 16384 tokens refused");

    // 10 text tokens, image 4x6 (2x3 = 6 tokens) at pos 10..15, 5 text tokens at 16..20,
    // second image 4x4 (2x2 = 4 tokens) at pos 21..24, 2 text tokens at 25..26.
    std::vector<g6rt::ImageIn> imgs(2);
    imgs[0].pos = 10; imgs[0].vit_h = 4; imgs[0].vit_w = 6; imgs[0].span = 6;
    imgs[1].pos = 21; imgs[1].vit_h = 4; imgs[1].vit_w = 4; imgs[1].span = 4;
    check(mrope_delta(imgs) == (3 - 6) + (2 - 4), "mrope_delta == -5");

    std::vector<int3> whole(27);
    build_mrope(imgs, 0, 27, whole.data());
    auto eq3 = [](int3 a, int3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
    check(eq3(whole[0], make_int3(0, 0, 0)), "pos 0");
    check(eq3(whole[9], make_int3(9, 9, 9)), "pos 9");
    check(eq3(whole[10], make_int3(10, 10, 10)), "img0 (0,0)");
    check(eq3(whole[12], make_int3(10, 10, 12)), "img0 (0,2)");
    check(eq3(whole[13], make_int3(10, 11, 10)), "img0 (1,0)");
    check(eq3(whole[15], make_int3(10, 11, 12)), "img0 (1,2)");
    check(eq3(whole[16], make_int3(13, 13, 13)), "text after img0");
    check(eq3(whole[20], make_int3(17, 17, 17)), "text before img1");
    check(eq3(whole[21], make_int3(18, 18, 18)), "img1 (0,0)");
    check(eq3(whole[24], make_int3(18, 19, 19)), "img1 (1,1)");
    check(eq3(whole[25], make_int3(20, 20, 20)), "text after img1 == 25 + (-5)");
    check(eq3(whole[26], make_int3(21, 21, 21)), "text pos 26 == 26 + (-5)");

    // Chunked build must match whole build bit-for-bit across mid-image boundaries.
    for (int chunk : {3, 5, 7}) {
        std::vector<int3> chunked(27);
        for (int p = 0; p < 27; p += chunk) {
            const int n = std::min(chunk, 27 - p);
            build_mrope(imgs, p, n, chunked.data() + p);
        }
        for (int p = 0; p < 27; ++p)
            check(eq3(whole[size_t(p)], chunked[size_t(p)]), "chunked mrope mismatch at pos " + std::to_string(p));
    }
    std::printf("  plan + 3D MRoPE checks: OK\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string weights, rgb_path, ref_path;
    int vit_h = 16, vit_w = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--weights") weights = val();
        else if (a == "--rgb") rgb_path = val();
        else if (a == "--ref") ref_path = val();
        else if (a == "--vit-h") vit_h = std::atoi(val().c_str());
        else if (a == "--vit-w") vit_w = std::atoi(val().c_str());
    }

    test_plan_and_mrope();

    if (!weights.empty()) {
        qw38rt::Qw38Vision vis;
        vis.log = true;
        vis.load(weights, 256);
        check(vis.ready(), "vision loaded");
        std::printf("  loaded %s (id=%s, %.1f MiB)\n", weights.c_str(), vis.id().c_str(),
                    vis.device_bytes() / (1024.0 * 1024.0));

        const int tokens = qw38v::span_of(vit_h, vit_w);
        check(tokens > 0, "valid vit_h x vit_w");
        const size_t rgb_bytes = size_t(vit_h) * vit_w * qw38v::PATCH * qw38v::PATCH * 3;
        auto rgb = std::make_shared<std::vector<uint8_t>>(rgb_bytes);
        if (!rgb_path.empty()) {
            std::ifstream f(rgb_path, std::ios::binary);
            f.read(reinterpret_cast<char*>(rgb->data()), static_cast<std::streamsize>(rgb_bytes));
            check(size_t(f.gcount()) == rgb_bytes, "read exact rgb bytes");
        } else {
            for (size_t i = 0; i < rgb_bytes; ++i) (*rgb)[i] = static_cast<uint8_t>((i * 37 + 13) & 255u);
        }

        const size_t cells = size_t(tokens) * qw38rt::vis::VOUT;
        __nv_bfloat16* d_out = nullptr;
        QW38V_CUDA(cudaMalloc(&d_out, cells * sizeof(__nv_bfloat16)));
        vis.encode(rgb->data(), vit_h, vit_w, d_out, 0);
        std::vector<__nv_bfloat16> got_bf(cells);
        QW38V_CUDA(cudaMemcpy(got_bf.data(), d_out, cells * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        cudaFree(d_out);

        bool all_fin = true;
        double norm2 = 0.0;
        for (size_t i = 0; i < cells; ++i) {
            const float v = __bfloat162float(got_bf[i]);
            all_fin &= std::isfinite(v);
            norm2 += double(v) * v;
        }
        check(all_fin && norm2 > 0.0, "encoded embeddings finite & non-zero");
        std::printf("  encode %dx%d (%d tokens): norm=%.4f\n", vit_h, vit_w, tokens, std::sqrt(norm2));

        // Verify staging cache hit + straddling chunks
        g6rt::ImageIn im;
        im.pos = 5; im.vit_h = vit_h; im.vit_w = vit_w; im.span = tokens; im.rgb = rgb;
        const long c0 = vis.encode_calls;
        const int n1 = vis.stage({im}, 0, 32, 0, 0, 1);
        const int n2 = vis.stage({im}, 32, 64, 0, 0, 1);
        check(n1 == 27 && n2 == std::min(64, tokens - 27), "straddling stage counts");
        check(vis.encode_calls == c0 + 1, "stage cached image across chunks (1 encode call)");

        if (!ref_path.empty()) {
            std::vector<float> ref(cells);
            std::ifstream rf(ref_path, std::ios::binary);
            rf.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(cells * sizeof(float)));
            check(size_t(rf.gcount()) == cells * sizeof(float), "read exact ref float32 bytes");
            double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0, ref_max = 0.0;
            for (size_t i = 0; i < cells; ++i) {
                const double a = __bfloat162float(got_bf[i]);
                const double b = ref[i];
                dot += a * b; na += a * a; nb += b * b;
                max_abs = std::max(max_abs, std::fabs(a - b));
                ref_max = std::max(ref_max, std::fabs(b));
            }
            const double cos = dot / (std::sqrt(na * nb) + 1e-12);
            const double rel_l2 = std::sqrt(std::max(0.0, na + nb - 2.0 * dot)) / (std::sqrt(nb) + 1e-12);
            std::printf("  golden vs transformers: cosine=%.6f, rel_l2=%.5f, max_abs=%.5f (ref_max=%.4f)\n",
                        cos, rel_l2, max_abs, ref_max);
            check(cos > 0.999, "cosine > 0.999 vs transformers");
        }
    }

    if (fails) {
        std::printf("qw38_vision_test: %d FAILED\n", fails);
        return 1;
    }
    std::printf("qw38_vision_test: PASS\n");
    return 0;
}
