// Standalone CUDA parity & latency test for Hy4Vision (HYV4FlashVisionTransformer).
// Usage: hy4_vision_test <vision-bf16.safetensors> <rgb.bin> <vit_h> <vit_w> [golden_f32.bin] [dump_f32.bin]
#define G6_MODEL_HY4 1
#include "../runtime/hy4_vision.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <vision.safetensors> <rgb.bin> <vit_h> <vit_w> [golden_f32.bin] [dump_f32.bin]\n", argv[0]);
        return 2;
    }
    const std::string vp = argv[1];
    const std::string rgb_path = argv[2];
    const int vit_h = std::atoi(argv[3]);
    const int vit_w = std::atoi(argv[4]);
    const std::string ref_path = (argc >= 6 && argv[5][0]) ? argv[5] : "";
    const std::string dump_path = (argc >= 7 && argv[6][0]) ? argv[6] : "";

    const int span = hy4v::span_of(vit_h, vit_w);
    if (span <= 0) {
        std::fprintf(stderr, "invalid grid %dx%d\n", vit_h, vit_w);
        return 2;
    }
    const size_t rgb_bytes = size_t(vit_h) * hy4v::PATCH * size_t(vit_w) * hy4v::PATCH * 3;
    std::vector<uint8_t> rgb(rgb_bytes);
    {
        std::ifstream f(rgb_path, std::ios::binary);
        if (!f || !f.read(reinterpret_cast<char*>(rgb.data()), std::streamsize(rgb_bytes))) {
            std::fprintf(stderr, "cannot read %zu RGB bytes from %s\n", rgb_bytes, rgb_path.c_str());
            return 2;
        }
    }

    hy4rt::Hy4Vision vis;
    vis.load(vp, 0);
    std::printf("# loaded %s (%.1f MiB, id=%s)\n", vp.c_str(), vis.device_bytes() / (1024.0 * 1024.0), vis.id().c_str());

    __nv_bfloat16* d_out = nullptr;
    const size_t elems = size_t(span) * hy4rt::vis::VOUT;
    cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16));
    cudaStream_t st = nullptr;
    cudaStreamCreate(&st);

    // Warmup + timed run
    vis.encode(rgb.data(), vit_h, vit_w, d_out, st);
    const auto t0 = std::chrono::steady_clock::now();
    vis.encode(rgb.data(), vit_h, vit_w, d_out, st);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::vector<__nv_bfloat16> h_out(elems);
    cudaMemcpy(h_out.data(), d_out, elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> out_f32(elems);
    int non_finite = 0;
    for (size_t i = 0; i < elems; ++i) {
        out_f32[i] = __bfloat162float(h_out[i]);
        if (!std::isfinite(out_f32[i])) ++non_finite;
    }
    std::printf("# encode %dx%d (%d patches -> %d tokens x %d): %.2f ms, non_finite=%d\n",
                vit_h, vit_w, vit_h * vit_w, span, hy4rt::vis::VOUT, ms, non_finite);
    if (non_finite) return 1;

    if (!dump_path.empty()) {
        std::ofstream f(dump_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(out_f32.data()), std::streamsize(elems * sizeof(float)));
    }
    if (!ref_path.empty()) {
        std::vector<float> ref(elems);
        std::ifstream f(ref_path, std::ios::binary);
        if (!f || !f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(elems * sizeof(float)))) {
            std::fprintf(stderr, "cannot read golden %s\n", ref_path.c_str());
            return 2;
        }
        double dot = 0, na = 0, nb = 0, sq = 0;
        float max_abs = 0;
        for (size_t i = 0; i < elems; ++i) {
            const double a = out_f32[i], b = ref[i];
            dot += a * b; na += a * a; nb += b * b;
            const double d = a - b;
            sq += d * d;
            if (std::fabs(float(d)) > max_abs) max_abs = std::fabs(float(d));
        }
        const double cos = dot / (std::sqrt(na * nb) + 1e-30);
        const double rel_l2 = std::sqrt(sq / (nb + 1e-30));
        std::printf("PARITY cos=%.6f rel_l2=%.6f max_abs=%.6f\n", cos, rel_l2, max_abs);
        if (cos < 0.999) {
            std::fprintf(stderr, "FAIL: vision parity cos=%.6f < 0.999\n", cos);
            return 1;
        }
    }
    cudaFree(d_out);
    cudaStreamDestroy(st);
    return 0;
}
