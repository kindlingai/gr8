// GPU gate for engine/mhc/ds_mhc.cuh against engine/mhc/ds_ref.py case files (and the host oracle for the BF16-fn arm).
//   python3 engine/mhc/ds_ref.py gen /tmp/ds_case.bin 4 5120 8 2000 1e-20 1
//   ds_mhc_gpu_test /tmp/ds_case.bin [timing_reps]
// Build: nvcc -O2 -std=c++17 -gencode=arch=compute_121,code=sm_121 -DG6_MODEL_DS41=1 -I engine -o ds_mhc_gpu_test ds_mhc_gpu_test.cu
// Needs a GPU; the CPU twin ds_mhc_cpu_test needs none.
#include "../../mhc/ds_mhc.cuh"
#include "../../mhc/ds_cpu.h"
#include "ds_mhc_case.h"
#include <cstdio>
#include <cstring>
using namespace dscase;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
template <class T> static T* up(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
static std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = __float2bfloat16_rn(v[i]);
    return o;
}
static std::vector<float> down(const void* d, size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }
static std::vector<float> down_bf16(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> t(n); CK(cudaMemcpy(t.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> v(n); for (size_t i = 0; i < n; ++i) v[i] = __bfloat162float(t[i]); return v;
}
struct Expected { std::vector<std::vector<float>> y, post, comb, pre; std::vector<float> T, h; };

// One chain on the device. FN = float or bf16 fn. The expected tensors come from the caller.
template <class WT>
static bool run_chain(const Case& c, const std::vector<std::vector<WT>>& fn_host, const std::vector<std::vector<float>>& fn_f, const Expected& ex, const char* tag, int reps, bool fused) {
    const int M = c.M, H = c.H;
#ifdef DS_MHC_TRUNK_BF16
#error "ds_mhc_gpu_test is the FP32-trunk oracle test: it mallocs float streams (the BF16 trunk is gated by ds_mhc_mtest)"
#endif
    ds_mhc::Buffers b{};
    CK(cudaMalloc(&b.streams, (size_t)M * 4 * H * 4)); CK(cudaMalloc(&b.mix, (size_t)M * 24 * 4));
    CK(cudaMalloc(&b.pre, (size_t)M * 16)); CK(cudaMalloc(&b.post, (size_t)M * 16)); CK(cudaMalloc(&b.comb, (size_t)M * 64));
    CK(cudaMalloc(&b.done, (size_t)M * 4)); CK(cudaMemset(b.done, 0, (size_t)M * 4));
    CK(cudaMalloc(&b.part, (size_t)M * ds_mhc::NV * (H / ds_mhc::PW) * 4));
    int* tok = up(c.tok);
    __nv_bfloat16* embed = nullptr;
    { auto e = to_bf16(c.embed); embed = up(e); }
    __nv_bfloat16 *y, *hout;
    CK(cudaMalloc(&y, (size_t)M * H * 2)); CK(cudaMalloc(&hout, (size_t)M * H * 2));
    const size_t frag_words = (size_t)((M + 7) / 8) * (H / 16) * 64;
    unsigned* fr; CK(cudaMalloc(&fr, frag_words * 4)); CK(cudaMemset(fr, 0xff, frag_words * 4));
    __nv_bfloat16* final_norm; { auto n = to_bf16(c.final_norm); final_norm = up(n); }
    std::vector<ds_mhc::Weights<WT>> W(c.NS);
    std::vector<float*> blocks(c.NS);
    for (int s = 0; s < c.NS; ++s) {
        auto n = to_bf16(c.L[s].norm);
        W[s] = {up(fn_host[s]), up(c.L[s].base), up(c.L[s].scale), up(n)};
        blocks[s] = up(c.L[s].block);
    }
    bool ok = true;
    printf("[%s%s] M=%d H=%d sublayers=%d\n", tag, fused ? " fused consume+prepare" : "", M, H, c.NS);
    // one-step gate. A BF16 trunk makes any two implementations drift apart over many sub-layers (one rounding flip stays in
    // the trunk and the mismatch count grows), so each stage is checked from the device's own state: the host oracle starts
    // from the trunk/pre the device holds before the stage and must produce what the device produced. The chain against the
    // torch reference is printed as info only (sub-layer 0 is gated: it has no history).
    const size_t NT = (size_t)M * 4 * H;
    auto gate_state = [&](const char* nm, const Err& e, double mism, double rel) { return gate_bf16(nm, e, mism, rel); };
    ds_mhc::embed(b, tok, embed, M, H, 0);
    std::vector<float> Tp, Th, prh, poh, cbh, yh((size_t)M * H), po, cb, pr;
    for (int s = 0; s < c.NS; ++s) {
        char nm[64];
        Tp = down(b.streams, NT); prh = down(b.pre, (size_t)M * 4);
        const bool fuse_now = fused && s > 0;      // device trunk still holds block s-1 unconsumed
        if (fuse_now) ds_mhc::consume_cpu(Tp.data(), c.L[s - 1].block.data(), po.data(), cb.data(), M, H);   // po/cb: prepare(s-1) outputs
        poh.assign((size_t)M * 4, 0); cbh.assign((size_t)M * 16, 0);
        ds_mhc::prepare_cpu(Tp.data(), prh.data(), fn_f[s].data(), c.L[s].base.data(), c.L[s].scale.data(), c.L[s].norm.data(), M, H, c.eps, poh.data(), cbh.data(), yh.data());
        if (fuse_now) ds_mhc::consume_prepare<WT>(b, flash_mhc::Fp32Block{blocks[s - 1]}, W[s], M, H, c.eps, y, 0);
        else ds_mhc::prepare<WT>(b, W[s], M, H, c.eps, y, 0);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        snprintf(nm, sizeof nm, "s%d y", s); auto yy = down_bf16(y, (size_t)M * H);
        ok &= gate_state(nm, cmp(yy.data(), yh.data(), yy.size()), 5e-4, 6e-3);
        if (fuse_now) { auto Td = down(b.streams, NT); snprintf(nm, sizeof nm, "s%d trunk", s); ok &= gate_state(nm, cmp(Td.data(), Tp.data(), NT), 5e-4, 6e-3); }
        po = down(b.post, (size_t)M * 4); cb = down(b.comb, (size_t)M * 16); pr = down(b.pre, (size_t)M * 4);
        snprintf(nm, sizeof nm, "s%d post", s); ok &= gate_f32(nm, cmp(po.data(), poh.data(), po.size()), 5e-5);
        snprintf(nm, sizeof nm, "s%d comb", s); ok &= gate_f32(nm, cmp(cb.data(), cbh.data(), cb.size()), 5e-5);
        snprintf(nm, sizeof nm, "s%d pre_next", s); ok &= gate_f32(nm, cmp(pr.data(), prh.data(), pr.size()), 5e-5);
        if (s == 0) {   // no history yet: must agree with the torch reference (fp32 summation order may flip a BF16 ulp)
            snprintf(nm, sizeof nm, "s0 y vs torch"); ok &= gate_bf16(nm, cmp(yy.data(), ex.y[0].data(), yy.size()), 5e-4, 6e-3);
            snprintf(nm, sizeof nm, "s0 post vs torch"); ok &= gate_f32(nm, cmp(po.data(), ex.post[0].data(), po.size()), 5e-5);
            snprintf(nm, sizeof nm, "s0 comb vs torch"); ok &= gate_f32(nm, cmp(cb.data(), ex.comb[0].data(), cb.size()), 5e-5);
        }
        if (!fused && s + 1 < c.NS) {
            Th = Tp;
            ds_mhc::consume_cpu(Th.data(), c.L[s].block.data(), po.data(), cb.data(), M, H);
            ds_mhc::consume(b, flash_mhc::Fp32Block{blocks[s]}, M, H, 0);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            auto Td = down(b.streams, NT);
            snprintf(nm, sizeof nm, "s%d consume T", s); ok &= gate_state(nm, cmp(Td.data(), Th.data(), NT), 5e-4, 6e-3);
        }
    }
    // finish: last consume + collapse with the carried pre + final norm, from the device's own state.
    Tp = down(b.streams, NT); prh = down(b.pre, (size_t)M * 4);
    po = down(b.post, (size_t)M * 4); cb = down(b.comb, (size_t)M * 16);
    Th = Tp; ds_mhc::consume_cpu(Th.data(), c.L[c.NS - 1].block.data(), po.data(), cb.data(), M, H);
    std::vector<float> hexp((size_t)M * H), xpexp((size_t)M * H);
    ds_mhc::final_cpu(Th.data(), prh.data(), c.final_norm.data(), M, H, c.eps, hexp.data(), xpexp.data());
    __nv_bfloat16* xpre; CK(cudaMalloc(&xpre, (size_t)M * H * 2));
    ds_mhc::finish(b, flash_mhc::Fp32Block{blocks[c.NS - 1]}, final_norm, M, H, c.eps, true, hout, 0, nullptr, fr, xpre);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    auto T = down(b.streams, NT), hh = down_bf16(hout, (size_t)M * H), xp = down_bf16(xpre, (size_t)M * H);
    ok &= gate_state("finish xpre", cmp(xp.data(), xpexp.data(), xp.size()), 5e-4, 6e-3);
    ok &= gate_state("finish trunk", cmp(T.data(), Th.data(), T.size()), 5e-4, 6e-3);
    ok &= gate_state("finish h", cmp(hh.data(), hexp.data(), hh.size()), 5e-4, 6e-3);
    { auto e1 = cmp(T.data(), ex.T.data(), T.size()), e2 = cmp(hh.data(), ex.h.data(), hh.size());
      printf("  info: whole chain vs reference (not gated; BF16 flips accumulate): trunk rel_l2 %.2e mismatch %.2f%%, h rel_l2 %.2e mismatch %.2f%%\n", e1.rel_l2, 100*e1.mismatch, e2.rel_l2, 100*e2.mismatch); }
    // Fragment table: word (m,h pair) must hold the BF16 pair of out[m][h], out[m][h+1] at the Xfrag index.
    std::vector<unsigned> frh(frag_words); CK(cudaMemcpy(frh.data(), fr, frag_words * 4, cudaMemcpyDeviceToHost));
    size_t frag_bad = 0;
    for (int m = 0; m < M; ++m) for (int h = 0; h < H; h += 2) {
        const size_t dst = (size_t)(m / 8) * (H / 16) * 64 + ((h / 16) * 32 + (m % 8) * 4 + (h % 8) / 2) * 2 + (h / 8) % 2;
        __nv_bfloat16 a = __float2bfloat16_rn(hh[(size_t)m * H + h]), bb = __float2bfloat16_rn(hh[(size_t)m * H + h + 1]);
        const unsigned want = (unsigned)__bfloat16_as_ushort(a) | ((unsigned)__bfloat16_as_ushort(bb) << 16);
        if (frh[dst] != want) ++frag_bad;
    }
    printf("  %-18s %zu bad of %d words  %s\n", "head fragments", frag_bad, M * H / 2, frag_bad ? "FAIL" : "ok");
    ok &= frag_bad == 0;
    if (reps > 0) {   // launch-only timing of each kernel on the same buffers (values drift; timing only)
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        auto timeit = [&](const char* what, auto launch) {
            for (int i = 0; i < 3; ++i) launch();
            CK(cudaEventRecord(e0));
            for (int i = 0; i < reps; ++i) launch();
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("  time %-10s M=%d: %.2f us/launch\n", what, M, 1000.0 * ms / reps);
        };
        timeit("embed", [&] { ds_mhc::embed(b, tok, embed, M, H, 0); });
        timeit("prepare", [&] { ds_mhc::prepare<WT>(b, W[0], M, H, c.eps, y, 0); });
        timeit("consume", [&] { ds_mhc::consume(b, flash_mhc::Fp32Block{blocks[0]}, M, H, 0); });
        timeit("cons+prep", [&] { ds_mhc::consume_prepare<WT>(b, flash_mhc::Fp32Block{blocks[0]}, W[1 % c.NS], M, H, c.eps, y, 0); });
        timeit("finish", [&] { ds_mhc::finish(b, flash_mhc::Fp32Block{blocks[0]}, final_norm, M, H, c.eps, false, hout, 0, nullptr, fr); });
    }
    return ok;
}
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    Case c;
    if (!load(argv[1], c)) return 2;
    const int reps = argc > 2 ? atoi(argv[2]) : 0;
    const int M = c.M, H = c.H;
    Expected ex;
    for (int s = 0; s < c.NS; ++s) { ex.y.push_back(c.E[s].y); ex.post.push_back(c.E[s].post); ex.comb.push_back(c.E[s].comb); ex.pre.push_back(c.E[s].pre_next); }
    ex.T = c.streams_last; ex.h = c.final_h;
    std::vector<std::vector<float>> fnf(c.NS);
    for (int s = 0; s < c.NS; ++s) fnf[s] = c.L[s].fn;
    bool ok = run_chain<float>(c, fnf, fnf, ex, "F32 fn vs torch reference", reps, false);
    ok &= run_chain<float>(c, fnf, fnf, ex, "F32 fn vs torch reference", reps, true);
    // BF16-fn arm: the fn rounded to BF16, expected from the host oracle (same rounded weights).
    std::vector<std::vector<__nv_bfloat16>> fnb(c.NS);
    Case cb = c;
    for (int s = 0; s < c.NS; ++s) {
        fnb[s] = to_bf16(c.L[s].fn);
        for (size_t i = 0; i < fnb[s].size(); ++i) cb.L[s].fn[i] = __bfloat162float(fnb[s][i]);
    }
    Expected eb; eb.T.assign((size_t)M * 4 * H, 0); eb.h.assign((size_t)M * H, 0);
    std::vector<float> T((size_t)M * 4 * H), pre(M * 4), post(M * 4), comb(M * 16), y((size_t)M * H);
    ds_mhc::embed_cpu(cb.tok.data(), cb.embed.data(), M, H, T.data(), pre.data());
    for (int s = 0; s < c.NS; ++s) {
        ds_mhc::prepare_cpu(T.data(), pre.data(), cb.L[s].fn.data(), cb.L[s].base.data(), cb.L[s].scale.data(), cb.L[s].norm.data(), M, H, c.eps, post.data(), comb.data(), y.data());
        eb.y.push_back(y); eb.post.push_back(post); eb.comb.push_back(comb); eb.pre.push_back(pre);
        ds_mhc::consume_cpu(T.data(), cb.L[s].block.data(), post.data(), comb.data(), M, H);
    }
    // device finish consumes the last block itself: the host trunk after the last prepare is the pre-consume trunk.
    // Recompute the expected end state: consume_cpu above already applied the last block, so T is the final trunk.
    eb.T = T;
    std::vector<float> hh((size_t)M * H);
    ds_mhc::final_cpu(T.data(), pre.data(), c.final_norm.data(), M, H, c.eps, hh.data());
    eb.h = hh;
    std::vector<std::vector<float>> fnbf(c.NS);
    for (int s = 0; s < c.NS; ++s) fnbf[s] = cb.L[s].fn;
    ok &= run_chain<__nv_bfloat16>(c, fnb, fnbf, eb, "BF16 fn vs host oracle", 0, false);
    ok &= run_chain<__nv_bfloat16>(c, fnb, fnbf, eb, "BF16 fn vs host oracle", 0, true);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
