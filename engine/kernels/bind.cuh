// g6 binder: one rank's pack tensors of one layer -> ready kernel arguments + launch helpers (M3-PLAN §4).
// Decode path, M <= 8 rows (M <= G6_MX under DEC_S2, g6mx.h). Everything a kernel reads per thread (control words, plan, scratch, fragment tables)
// must be DEVICE memory; only the two partial outputs may point into a fabric send slot (see put_partial below).
//
//   bind::init();                                        once per process (shared-memory attribute of K1 fc1)
//   bind::Scratch sc = bind::scratch_at(block);          block = bind::scratch_bytes() of device memory, zeroed once
//   bind::LayerTensors T = bind::layer_tensors(l, find); find("L003.q_a.gptq") -> bind::Tensor (p = nullptr: absent)
//   bind::widen_fc1(T, st);                              optional, once per layer: K3 fc1 into the wide order, in place
//   bind::widen_fc2(T, st);                              optional, once per layer: K3 fc2 likewise
//   bind::scan_widths(T);                                once per layer: whether it needs the 2-bit kernels (#65)
//   bind::Layer L;  const char* err = bind::make_layer(T, M, sc, io, L);      err == nullptr: ready
//
// Step order (each call is one kernel launch on the stream; `->` = where the result lands):
//   launch(L.xf_x, st)                      io.x -> fragment tables (fp16; BF16 too on indexer layers)
//   launch(L.q_a, st)                       -> io.q_a   fp16 [8][2048]        (then glue::k_rmsnorm_rows -> io.q_a_n)
//   launch(L.kv_a, st)                      -> io.kv_a  fp16 [8][576]
//   launch(L.idx_k, st), launch(L.idx_w, st)   indexer layers -> io.idx_k fp16 [8][128], io.idx_w fp16 [8][32]
//   launch(L.xf_qa, st)                     io.q_a_n -> fragment table
//   launch(L.q_b, st)                       -> io.q_b   fp16 [8][H*256]       (H = L.heads local heads)
//   launch(L.idx_q, st)                     indexer layers -> io.idx_q fp16 [8][4096]
//   ... attention ...
//   launch(L.xf_attn, st), launch(L.o_proj, st)     io.attn [M][H*256] -> io.attn_part BF16 [8][6144]
//   ... all-reduce 1, mlp_norm ...
//   launch(L.xf_x2, st)                     io.x2 -> fragment tables (fp16; BF16 too on MoE layers)
//   launch(L.router, st)                    MoE -> io.logits fp32 [8][256]    (then glue::k_router_plan -> io.plan)
//   launch_k1(L.k1, st)                     fc1, fc2, finalize (dense layer: fc1, fc2) -> io.mlp_part BF16 [8][6144]
// A launch of a member with on == 0 (tensor not in this layer) does nothing. Output rows >= M are not written.
// Tensors read: q_a.gptq, o_proj.gptq (int4 g32 records), kv_a / q_b / idx.wq_b .mx.gv (MXFP8 records), router.w /
// idx.wk / idx.weights_proj .gv (BF16 records), shared|dense .gptq.fc1/.fc2, tr3.fc1/fc2/v1/v2/v3/table (+ tr3.kmix
// in exlr8 v2 packs).
// Draft layer instead of the .gptq records: q_a.mx.gv, o_proj.mx.gv, shared.gate.mx.gv, shared.up.mx.gv, shared.down.mx.gv.
// BF16-weight GEMVs (router, idx.wk, idx.weights_proj) take the activations rounded to BF16 (the bf16 mma takes
// both operands as bf16): their reference is bf16(x) @ W^T, fp32 accumulate.
//
// MTP draft layer (L078, not GPTQ): same calls, same step order, same Io. A layer without q_a.gptq / o_proj.gptq /
// shared.gptq.fc1 binds the MXFP8 W8A16 records instead: q_a.mx.gv, o_proj.mx.gv (the kv_a / q_b kernel, fp16
// activations) and shared.gate.mx.gv + shared.up.mx.gv + shared.down.mx.gv (the shared expert inside the K1 grids,
// as on the other MoE layers). Routed experts are TR3 as everywhere. Layer::w8 tells which layers took this path.
//
// Head + MTP projection (once per rank, same scratch block):
//   bind::HeadTensors HT = bind::head_tensors(find);     "G.lm_head.mx.gv", "L078.eh.mx.gv"
//   bind::Head H;  const char* err = bind::make_head(HT, rank, M, sc, hio, H);
//   launch(H.xf_h, st), launch(H.lm_head, st)   hio.h -> hio.logits fp32 [8][H.vn], this rank's vocab slice
//                                               (then glue::k_head_local: bind::head_args(H, a) fills logits/ldl/vlo/vn)
//   launch(H.xf_eh, st), launch(H.eh, st)       hio.eh_in [8][12288] -> hio.eh fp16 [8][6144]   (MTP eh_proj, full on each rank)
#pragma once
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include "k1.cuh"
#include "densev.cuh"
#include "dtr.cuh"
#if DEC_I4PAIR
#include <mutex>
#include <unordered_map>
#endif

namespace bind
{

// routed-expert format of a tr3.* tensor: the manifest entry's "expert_format" (absent = TR3, every pack before #19)
enum { FMT_TR3 = 0, FMT_EXLR8_E4M3 = 1 };

struct Tensor
{
    const void* p = nullptr;          // device pointer (nullptr: the pack has no such tensor)
    long long shape[4] = {0, 0, 0, 0};    // manifest shape
    int nd = 0;
    int fmt = FMT_TR3;                // routed-expert format (tr3.* tensors)
};

enum
{
    T_Q_A, T_KV_A, T_Q_B, T_O_PROJ, T_IDX_WQ_B, T_IDX_WK, T_IDX_WPROJ, T_ROUTER, T_S_FC1, T_S_FC2, T_D_FC1, T_D_FC2,
    T_TR3_FC1, T_TR3_FC2, T_TR3_V1, T_TR3_V2, T_TR3_V3, T_TR3_TAB,
    T_Q_A_MX, T_O_PROJ_MX, T_S_GATE_MX, T_S_UP_MX, T_S_DOWN_MX,         // MXFP8 twins, read only when the int4 record is absent
    T_TR3_KMIX,                                                         // exlr8 v2 packs only (#65): fc1 width per (block, matrix)
    T_TR3_SG, T_TR3_V1B, T_TR3_V3B,                                     // exlr8 v2 layers with the trellis shared expert: hidden-
                                                                        // side vectors as signs + block scales (no tr3.v1 / v3)
    T_Q_B_I4,                                                           // exlr8 int4 g32 q_b (layers where int4 won the encode)
    NT
};
static const char* const NAMES[NT] = {
    "q_a.gptq", "kv_a.mx.gv", "q_b.mx.gv", "o_proj.gptq", "idx.wq_b.mx.gv", "idx.wk.gv", "idx.weights_proj.gv", "router.w.gv",
    "shared.gptq.fc1", "shared.gptq.fc2", "dense.gptq.fc1", "dense.gptq.fc2",
    "tr3.fc1", "tr3.fc2", "tr3.v1", "tr3.v2", "tr3.v3", "tr3.table",
    "q_a.mx.gv", "o_proj.mx.gv", "shared.gate.mx.gv", "shared.up.mx.gv", "shared.down.mx.gv", "tr3.kmix", "tr3.sg", "tr3.v1b",
    "tr3.v3b", "q_b.gptq"};

struct LayerTensors
{
    Tensor t[NT];
    int fc1_wide = 0;                 // tr3.fc1 is in the wide order (widen_fc1 ran)
    int fc2_wide = 0;                 // tr3.fc2 is in the wide order (widen_fc2 ran)
    int e4m3 = 0;                     // routed experts are exlr8-e4m3 (tr3.table's format): K1 decodes with E4M3 rounding
    int k2 = 1;                       // some routed fc1 / fc2 unit has width 2 (exlr8 v2, #65): the K2 instantiations.
                                      // 1 until scan_widths ran (correct for any pack); scan_widths sets the real value
    int k2_pct = 0;                   // share of the rank's routed trellis bytes at width 2, percent (scan_widths)
    int sh_trellis = 0;               // tr3.table has row 256: the shared expert as trellis (exlr8 v2, #65); scan_widths
                                      // sets it, a caller may clear it to run the s_* records instead (the A/B)
    long long sh_tr[6] = {};          // tr3.table row 256 (scan_widths)
    long long sh_kmix = 0;            // its tr3.kmix entry
    int sh_uniform = 0;               // every fc1 unit of row 256 has the table width (its tr3.kmix entry is 0)
    struct ShFp8                      // the shared expert's decoded tiles as MXFP8 GEMM operands (make_shared_fp8)
    {
        uint8_t *codes[3] = {}, *scales[3] = {};    // gate, up [I][6144]; down [6144][I]; E8M0 scales all 127
        int I = 0;                    // owned intermediate channels (row 256's blocks x 128)
    } shfp8;
#if DEC_I4PAIR
    int i4pair_ks[3] = {0, 0, 0};     // q_a.gptq, o_proj.gptq, q_b.gptq repacked into pair records for this k-part count (i4pair ran)
#endif
    dtr::LayerMats mats;              // dense trellis matrices (exlr8 release form, dtr::layer_mats); w = nullptr: absent
    const char* mats_err = nullptr;   // dtr::layer_mats refused the layer's trellis tensors (make_layer reports it)
};

// find(const char* name) -> Tensor, name = "L003.q_a.gptq" ...
template <class F> inline LayerTensors layer_tensors(int layer, F find)
{
    LayerTensors T;
    char n[64];
    for (int i = 0; i < NT; ++i)
    {
        snprintf(n, sizeof n, "L%03d.%s", layer, NAMES[i]);
        T.t[i] = find(n);
    }
    T.e4m3 = T.t[T_TR3_TAB].fmt == FMT_EXLR8_E4M3;
    T.mats_err = dtr::layer_mats(layer, [&](const char* nm) { return find(nm); }, T.mats);
    return T;
}

// Optional, once per layer after its tensors are on the device and before make_layer: reorder the K3 records of
// tr3.fc1 in place into the wide order (tr3_wide.cuh; fc1 then reads 256 B per load instruction, RESULTS-M1 item 1c).
// The tensor memory must be writable device memory. Same results bit for bit. Wait for the stream before using T.
inline void widen_fc1(LayerTensors& T, cudaStream_t st)
{
    const Tensor &tab = T.t[T_TR3_TAB], &fc1 = T.t[T_TR3_FC1];
    if (!tab.p || !fc1.p || T.fc1_wide) return;
    launch_tr3_widen(const_cast<void*>(fc1.p), static_cast<const long long*>(tab.p), static_cast<const long long*>(T.t[T_TR3_KMIX].p),
                     static_cast<int>(tab.shape[0]), g6nr::EB, st);
    T.fc1_wide = 1;
}
// The same for tr3.fc2 (k_tr3_widen2: the 8 k-tiles of an n-tile in three 64-bit loads). Same rules, same results bit for bit.
inline void widen_fc2(LayerTensors& T, cudaStream_t st)
{
    const Tensor &tab = T.t[T_TR3_TAB], &fc2 = T.t[T_TR3_FC2];
    if (!tab.p || !fc2.p || T.fc2_wide) return;
    launch_tr3_widen2(const_cast<void*>(fc2.p), static_cast<const long long*>(tab.p), static_cast<int>(tab.shape[0]), g6nr::EB, st);
    T.fc2_wide = 1;
}

// Host twin of tr3_unit for tr3.table row 256 (LayerTensors::sh_tr, sh_kmix): byte offset of its fc1 unit u in tr3.fc1
inline void sh_unit_off(const LayerTensors& T, int u, size_t& off)
{
    constexpr size_t MB = static_cast<size_t>(KSH) * 8 * 32;
    off = static_cast<size_t>(T.sh_tr[0]);
    for (int i = 0; i < u; ++i) off += static_cast<size_t>(T.sh_kmix ? (T.sh_kmix >> (4 * i)) & 15 : T.sh_tr[4]) * MB;
}

// Once per layer after its tensors are on the device: reads tr3.table (and tr3.kmix) back, checks every width is 2, 3
// or 4, and sets T.k2 (whether the layer needs the kernels with the 2-bit body). Synchronous. nullptr = ok.
// Every routed expert has at least two halves on every rank at TP3-7, so an expert with no blocks means a partial test
// pack (tools/convert/v2.py --routed): refused unless allow_empty (test programs).
inline const char* scan_widths(LayerTensors& T, bool allow_empty = false)
{
    const Tensor &tab = T.t[T_TR3_TAB], &km = T.t[T_TR3_KMIX];
    T.k2 = 0;
    T.sh_trellis = 0;
    if (!tab.p) return nullptr;
    if (tab.nd != 2 || tab.shape[1] != 6 || (tab.shape[0] != 256 && tab.shape[0] != 257)) return "tr3.table is not [256 or 257][6]";
    if (tab.shape[0] == 257 && !T.e4m3) return "a trellis shared expert (tr3.table row 256) needs an exlr8-e4m3 layer";
    T.sh_trellis = tab.shape[0] == 257;
    const size_t n = static_cast<size_t>(tab.shape[0]);
    long long* h = new long long[n * 7];
    long long* k = h + n * 6;
    cudaError_t e = cudaMemcpy(h, tab.p, n * 6 * 8, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess && km.p) e = km.shape[0] == tab.shape[0] ? cudaMemcpy(k, km.p, n * 8, cudaMemcpyDeviceToHost) : cudaErrorInvalidValue;
    if (e != cudaSuccess) { delete[] h; return "scan_widths: cannot read tr3.table / tr3.kmix"; }
    const char* err = nullptr;
    double all = 0, two = 0;
    for (size_t i = 0; i < n && !err; ++i)
    {
        const long long* tr = h + i * 6;
        const long long kmi = km.p ? k[i] : 0;
        if (tr[3] <= 0)
        {
            if (!allow_empty && i < 256) err = "a routed expert has no blocks on this rank (a partial test pack)";
            continue;
        }
        if (tr[3] > g6nr::EB) err = "an expert has more blocks than this build's EB";
        auto chk = [&](long long w) {
            if (w < 2 || w > 4) err = "routed expert width outside 2..4";
            all += w;
            if (w == 2) { T.k2 = 1; two += w; }
        };
        for (long long b = 0; b < tr[3]; ++b) chk(tr[4]);                            // down blocks
        for (int u = 0; u < 2 * tr[3]; ++u) chk(kmi ? (kmi >> (4 * u)) & 15 : tr[4]);   // gate / up units
    }
    T.k2_pct = all > 0 ? static_cast<int>(100 * two / all) : 0;
    if (T.sh_trellis)
    {
        for (int i = 0; i < 6; ++i) T.sh_tr[i] = h[256 * 6 + i];
        T.sh_kmix = km.p ? k[256] : 0;
        T.sh_uniform = T.sh_kmix == 0;
    }
    delete[] h;
    return err;
}

// The trellis shared expert's tiles decoded once into E4M3 bytes in the MXFP8 GEMM layout, codes[n][k] with k the
// input side, both sides still rotated (the prefill binder applies the rotations around the GEMMs). The codebook values
// are E4M3 already, so the copy is exact. Reads tr3.fc1 / fc2 in pack order: run it before widen_fc1 / widen_fc2.
// grid ceil(I * 6144 * 3 / 256), block 256: one thread per weight.
__global__ void __launch_bounds__(256) k_shared_fp8(const uint8_t* fc1, const uint8_t* fc2, long long o1, long long o2, int bits, int I,
                                                    uint8_t* cg, uint8_t* cu, uint8_t* cd)
{
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x, per = static_cast<long long>(I) * HID;
    if (i >= 3 * per) return;
    const int mat = static_cast<int>(i / per);
    const long long w = i % per;
    const size_t TB = 32 * bits, MBY = static_cast<size_t>(KSH) * 8 * TB;
    const int t = static_cast<int>(w & 255);
    const long long tile = w >> 8;
    const uint8_t* p;
    int k, n;
    const int lane = t >> 3, j = t & 7;
    const int kk = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1), nn = (lane >> 2) + 8 * (j >> 2);
    if (mat < 2)          // gate / up: block b, tile (k16, c) at fc1 + o1 + (2b + mat) MBY + (8 k16 + c) TB
    {
        const int b = static_cast<int>(tile / (KSH * 8)), r = static_cast<int>(tile % (KSH * 8)), k16 = r >> 3, c = r & 7;
        p = fc1 + o1 + (2 * b + mat) * MBY + static_cast<size_t>(r) * TB;
        k = k16 * 16 + kk;
        n = b * 128 + c * 16 + nn;
    }
    else                  // down: block b, tile (n-tile nt, k-tile kt) at fc2 + o2 + b MBY + (8 nt + kt) TB
    {
        const int b = static_cast<int>(tile / (KSH * 8)), r = static_cast<int>(tile % (KSH * 8)), nt = r >> 3, kt = r & 7;
        p = fc2 + o2 + b * MBY + static_cast<size_t>(r) * TB;
        k = b * 128 + kt * 16 + kk;
        n = nt * 16 + nn;
    }
    // state of weight t: the 16 stream bits ending at bit (t + 1) bits, MSB first over LE u32 words, tail-biting
    const int nb = 256 * bits;
    u32 st = 0;
    for (int q = 0; q < 16; ++q)
    {
        const int bit = (((t + 1) * bits - 16 + q) % nb + nb) % nb;
        const u32 word = *reinterpret_cast<const u32*>(p + (bit >> 5) * 4);
        st = (st << 1) | ((word >> (31 - (bit & 31))) & 1u);
    }
    const u32 h2 = tr3_cb2<true>(st, 0u);                         // low half = weight t, E4M3-rounded
    unsigned short e;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e) : "r"(h2));
    const uint8_t q8 = static_cast<uint8_t>(e & 0xff);
    if (mat == 0) cg[static_cast<size_t>(n) * HID + k] = q8;
    else if (mat == 1) cu[static_cast<size_t>(n) * HID + k] = q8;
    else cd[static_cast<size_t>(n) * I + k] = q8;
}

// After scan_widths, before widen_fc1 / widen_fc2: the shared expert's FP8 copy for the prefill GEMMs, when row 256 has
// one width for all its fc1 units (otherwise prefill runs it as a trellis pass). Allocates T.shfp8; synchronous.
// Device memory: 3 x I x 6144 bytes + scales, 9.7 MB per layer at TP4.
inline const char* make_shared_fp8(LayerTensors& T, cudaStream_t st)
{
    if (!T.sh_trellis || !T.sh_uniform) return nullptr;
    if (T.fc1_wide || T.fc2_wide) return "make_shared_fp8 after widen: run it before widen_fc1 / widen_fc2";
    LayerTensors::ShFp8& F = T.shfp8;
    F.I = static_cast<int>(T.sh_tr[3]) * 128;
    const size_t nc = static_cast<size_t>(F.I) * HID;
    for (int m = 0; m < 3; ++m)
    {
        if (cudaMalloc(&F.codes[m], nc) != cudaSuccess || cudaMalloc(&F.scales[m], nc / 32) != cudaSuccess)
            return "make_shared_fp8: cudaMalloc failed";
        cudaMemsetAsync(F.scales[m], 127, nc / 32, st);
    }
    k_shared_fp8<<<static_cast<unsigned>((3 * nc + 255) / 256), 256, 0, st>>>(
        static_cast<const uint8_t*>(T.t[T_TR3_FC1].p), static_cast<const uint8_t*>(T.t[T_TR3_FC2].p), T.sh_tr[0], T.sh_tr[1],
        static_cast<int>(T.sh_tr[4]), F.I, F.codes[0], F.codes[1], F.codes[2]);
    return cudaStreamSynchronize(st) == cudaSuccess ? nullptr : "make_shared_fp8: decode failed";
}

// exlr8-e4m3 layers share one hidden-side sign vector per layer (input of gate and up, output of down). Each block row of
// v1 and v3 must carry block row 0's signs. One CTA per row after row 0, counted into *bad.
__global__ void __launch_bounds__(256) k_shared_signs(const uint16_t* v1, const uint16_t* v3, unsigned* bad)
{
    const size_t row = blockIdx.x + 1;
    bool diff = false;
    for (int i = threadIdx.x; i < HID; i += 256)
    {
        diff |= ((v1[row * 2 * HID + i] ^ v1[i]) & 0x8000) != 0;
        diff |= ((v1[row * 2 * HID + HID + i] ^ v1[HID + i]) & 0x8000) != 0;
        diff |= ((v3[row * HID + i] ^ v3[i]) & 0x8000) != 0;
    }
    if (__syncthreads_or(diff) && threadIdx.x == 0) atomicAdd(bad, 1u);
}

// Optional, once per exlr8-e4m3 layer after its tensors are on the device: refuses a layer whose block rows disagree on
// the hidden-side signs (a mislabelled or edited pack). Synchronizes the stream.
inline const char* check_shared_signs(const LayerTensors& T, cudaStream_t st)
{
    // (a block-scale layer has no tr3.v1 / v3: its signs are one vector, tr3.sg, by construction)
    const Tensor &v1 = T.t[T_TR3_V1], &v3 = T.t[T_TR3_V3];
    if (!T.e4m3 || !v1.p || !v3.p || v1.shape[0] < 2) return nullptr;
    unsigned *d = nullptr, *h = nullptr;
    if (cudaMalloc(&d, sizeof(unsigned)) != cudaSuccess) return "check_shared_signs: cudaMalloc failed";
    if (cudaMallocHost(&h, sizeof(unsigned)) != cudaSuccess) { cudaFree(d); return "check_shared_signs: cudaMallocHost failed"; }
    cudaMemsetAsync(d, 0, sizeof(unsigned), st);
    k_shared_signs<<<static_cast<unsigned>(v1.shape[0] - 1), 256, 0, st>>>(static_cast<const uint16_t*>(v1.p),
                                                                        static_cast<const uint16_t*>(v3.p), d);
    cudaMemcpyAsync(h, d, sizeof(unsigned), cudaMemcpyDeviceToHost, st);
    const cudaError_t e = cudaStreamSynchronize(st);
    const unsigned bad = *h;
    cudaFree(d);
    cudaFreeHost(h);
    if (e != cudaSuccess) return "check_shared_signs: CUDA error";
    return bad ? "exlr8-e4m3 layer whose tr3.v1 / tr3.v3 block rows do not share one sign vector" : nullptr;
}

#if DEC_I4PAIR
// DEC_I4PAIR: once per layer after its tensors are on the device and before make_layer: repack the int4 q_a / o_proj
// records in place into pair records (densev.cuh k_i4pair) for the k-part count make_layer uses (8, halved like make_gv).
// Same rules as widen_fc1 (writable device memory, wait for the stream). Same results bit for bit.
// The repack is in place, so it runs once per device tensor per process: ranks that share a device tensor (m4sim keeps one
// copy of a tensor that is identical on all ranks) must not permute it again; they record the k-part count it was packed for.
inline void i4pair(LayerTensors& T, cudaStream_t st)
{
    static std::mutex mu;
    static std::unordered_map<const void*, int> packed;   // device pointer -> ks it was repacked for
    std::lock_guard<std::mutex> lk(mu);
    const int id[3] = {T_Q_A, T_O_PROJ, T_Q_B_I4};
    for (int j = 0; j < 3; ++j)
    {
        const Tensor& t = T.t[id[j]];
        if (!t.p || T.i4pair_ks[j] || t.nd != 3 || t.shape[2] != 288) continue;
        const int ng = static_cast<int>(t.shape[1]);
        int ks = 8;
        while (ks > 1 && ng % ks) ks >>= 1;
        const auto it = packed.find(t.p);
        if (it == packed.end())
        {
            launch_i4pair(const_cast<void*>(t.p), static_cast<int>(t.shape[0]), ng, ks, st);
            packed[t.p] = ks;
            T.i4pair_ks[j] = ks;
        }
        else T.i4pair_ks[j] = it->second;   // already packed (shared tensor); make_layer checks it against its ks
    }
}
#endif

// ------------------------------------------------------------------------------------------ scratch (device memory)
struct Scratch
{
    int* ctl;                         // [3] K1 plan cells (stay 0: one plan per call) + [1] zero
    int* fctl;                        // [FCTL_N] counters of the fused K1 launch (zero between steps)
    u32* xf_x;                        // fragment tables [K/16][32][2]: attn_norm x, fp16
    u32* xb_x;                        //   same rows rounded to BF16 (idx.wk, idx.weights_proj)
    u32* xf_qa;                       //   q_a_norm output, K = 2048
    u32* xf_attn;                     //   attention output, K = local heads * 256
    u32* xf_x2;                       //   mlp_norm x, fp16 (shared / dense MLP)
    u32* xb_x2;                       //   mlp_norm x rounded to BF16 (router)
    float* gpart;                     // K1Args::gpart
    __half *z, *act, *ypart;          // K1Args::z, act, ypart
    uint16_t* part[2];                // [8][6144] BF16 staging for the two fabric partials (put_partial)
    u32* xf_head;                     // fragment table of the lm_head input, K = 6144
    u32* xf_eh;                       //   of the MTP eh input, K = 12288
    dtr::Scratch dt;                  // dense trellis GEMVs: fp32 partials and block counters (dtr.cuh)
    u32* dxd;                         //   the dense MLP down's input fragment tables [ceil(rows/8)][DENSE_CH/16][32][2]
    int rows = G6_MX;                 // row limit the block was laid out for (scratch_layout): sizes below follow it
    int smax = SMAX;                  // K1 slot capacity: SMAX at <= 8 rows, min(SMAX, 8 rows + ceil(rows / 8)) above
    int ysh = SMAX * 8;               // ypart rows ahead of the shared expert's: SMAX * 8 at <= 8 rows, 0 above
};

// row limit -> scratch geometry. rows <= 8 is the layout of the 8-row build, byte for byte.
// Above 8 rows: 8 routed slots per row, plus one trellis shared-expert slot per 8 rows (exlr8 v2 layers, k1.cuh k1_nsh).
constexpr int SMAX8 = SMAX < 72 ? SMAX : 72;      // routed slots at <= 8 rows (SMAX itself in the default build)
inline int scratch_smax(int rows)
{
    const int n = 8 * rows + g6mx::tiles(rows);
    return rows <= 8 ? SMAX8 : (n < SMAX ? n : SMAX);
}
inline int scratch_ysh(int rows) { return rows <= 8 ? SMAX8 * 8 : 0; }

// dense trellis partials (floats) for up to `rows` rows: the largest launch of make_layer's matrices at the
// dtr::split_for splits
inline size_t dtr_part_floats(int rows)
{
    struct S { int K, N, side, epi, rot, nm; };
    const S sh[4] = {{2048, g6nr::QBW, dtr::SIDE_OUT, dtr::EPI_OUT, 1, 1}, {g6nr::QBW, HID, dtr::SIDE_IN, dtr::EPI_OUT, 1, 1},
                     {HID, g6nr::DENSE_CH, dtr::SIDE_OUT, dtr::EPI_GLU, 1, 2}, {g6nr::DENSE_CH, HID, dtr::SIDE_IN, dtr::EPI_OUT, 0, 1}};
    size_t n = 0;
    for (int M = 1; M <= rows; ++M)
        for (const S& x : sh)
        {
            int ks, sub;
            dtr::split_for(x.K, x.side, x.epi, x.rot, M, ks, sub);
            const size_t f = dtr::part_floats(x.N, x.nm, ks, M);
            if (f > n) n = f;
        }
    return n;
}
constexpr int DTR_CNT = 256;          // dense trellis block counters: q_b 0, o_proj 64, MLP gate + up 128, down 192

inline size_t scratch_layout(Scratch* s, char* base, int rows = G6_MX)
{
    size_t o = 0;
    auto take = [&](size_t bytes) { char* p = base ? base + o : nullptr; o += (bytes + 255) / 256 * 256; return p; };
    if (rows < 1) rows = 1;
    if (rows > G6_MX) rows = G6_MX;
    const int NT = g6mx::tiles(rows), smax = scratch_smax(rows), ysh = scratch_ysh(rows);
    const size_t xf = static_cast<size_t>(KSH) * 64 * 4 * NT;     // NT row tiles of 8, each a [K/16][32][2] table
    Scratch t;
    t.ctl = reinterpret_cast<int*>(take(4 * sizeof(int)));
    t.fctl = reinterpret_cast<int*>(take(FCTL_N * sizeof(int)));
    t.xf_x = reinterpret_cast<u32*>(take(xf));
    t.xb_x = reinterpret_cast<u32*>(take(xf));
    t.xf_qa = reinterpret_cast<u32*>(take(xf));
    t.xf_attn = reinterpret_cast<u32*>(take(xf));
    t.xf_x2 = reinterpret_cast<u32*>(take(xf));
    t.xb_x2 = reinterpret_cast<u32*>(take(xf));
    t.gpart = reinterpret_cast<float*>(take(static_cast<size_t>(smax) * g6nr::EB * 2 * GPMAX * 8 * 128 * sizeof(float)));
    t.z = reinterpret_cast<__half*>(take(static_cast<size_t>(smax) * g6nr::EB * 8 * HID * 2));
    t.act = reinterpret_cast<__half*>(take((static_cast<size_t>(smax) * 8 * ACTW + static_cast<size_t>(8 * NT) * SACTW) * 2));
    t.ypart = reinterpret_cast<__half*>(take((static_cast<size_t>(ysh) + 8 * NT) * HID * 2));
    t.part[0] = reinterpret_cast<uint16_t*>(take(static_cast<size_t>(rows < 8 ? 8 : rows) * HID * 2));
    t.part[1] = reinterpret_cast<uint16_t*>(take(static_cast<size_t>(rows < 8 ? 8 : rows) * HID * 2));
    t.xf_head = reinterpret_cast<u32*>(take(xf));
    t.xf_eh = reinterpret_cast<u32*>(take(2 * xf));
    t.dt.part_n = dtr_part_floats(rows);
    t.dt.part = reinterpret_cast<float*>(take(t.dt.part_n * sizeof(float)));
    t.dt.cnt_n = DTR_CNT;
    t.dt.cnt = reinterpret_cast<int*>(take(DTR_CNT * sizeof(int)));
    t.dxd = reinterpret_cast<u32*>(take(static_cast<size_t>(NT) * (g6nr::DENSE_CH / 16) * 64 * 4));
    t.rows = rows; t.smax = smax; t.ysh = ysh;
    if (s) *s = t;
    return o;
}
// One block per rank (58.3 MB at TP6). Simulated ranks that run one after another on one stream may share a block.
// Zero it once after allocation (the control cells must start at 0).
inline size_t scratch_bytes(int rows = G6_MX) { return scratch_layout(nullptr, nullptr, rows); }
inline Scratch scratch_at(void* block, int rows = G6_MX)
{
    Scratch s;
    scratch_layout(&s, static_cast<char*>(block), rows);
    return s;
}

// ------------------------------------------------------------------------------------------ fragment tables
// src fp16 rows [M][ld] (16-byte aligned, ld % 8 == 0) -> mma B fragment table [K/16][32][2] (layout of
// make_xfrag, engine/bench/host.h): fp16 bits into `f16`, the same values rounded to BF16 into `bf16` (either may be
// null). M > 8: ceil(M / 8) tables of 8 rows each, K / 16 * 64 u32 apart (tile t = rows 8 t .. 8 t + 7). Rows >= M are written as zero. One thread per (k16 step, row): two 16-byte loads, two 16-byte stores per table.
// Index formula (to fold the table write into another kernel): for k16 step ks, row m (0..7), c = 0..3, r = 0..1
//   table[((ks * 32 + m * 4 + c) * 2 + r)] = the two halfs x[m][16 * ks + 8 * r + 2 * c], x[..+ 1] (low half first).
// Lane = m * 4 + c (32 lanes per k16 step); the lanes of rows >= M hold zero.
struct Xfrag
{
    int on = 0;
    const __half* src = nullptr;
    int ld = 0, K = 0, M = 0;
    u32* f16 = nullptr;
    u32* bf16 = nullptr;
};

__device__ __forceinline__ u32 h2_to_bf2(u32 w)
{
    const float2 f = __half22float2(as_h2(w));
    return pack_bf2(f.x, f.y);
}

__global__ void __launch_bounds__(256) k_xfrag(Xfrag a)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int ks = i >> 3, row = i & 7;
    if (ks >= a.K / 16) return;
    const int tile = blockIdx.y, grow = tile * 8 + row;     // grid.y = row tile: rows 8 tile .. 8 tile + 7
    uint4 lo = make_uint4(0u, 0u, 0u, 0u), hi = lo;
    if (grow < a.M)
    {
        const uint4* s = reinterpret_cast<const uint4*>(a.src + static_cast<size_t>(grow) * a.ld + ks * 16);
        lo = s[0]; hi = s[1];
    }
    // lane row*4 + c: reg 0 = halfs 2c, 2c+1 (word c of lo), reg 1 = halfs 8+2c, 9+2c (word c of hi); tile t's table follows tile t - 1's
    const size_t o = static_cast<size_t>(tile) * (a.K / 16) * 64 + (static_cast<size_t>(ks) * 32 + row * 4) * 2;
    if (a.f16)
    {
        uint4* d = reinterpret_cast<uint4*>(a.f16 + o);
        d[0] = make_uint4(lo.x, hi.x, lo.y, hi.y);
        d[1] = make_uint4(lo.z, hi.z, lo.w, hi.w);
    }
    if (a.bf16)
    {
        uint4* d = reinterpret_cast<uint4*>(a.bf16 + o);
        d[0] = make_uint4(h2_to_bf2(lo.x), h2_to_bf2(hi.x), h2_to_bf2(lo.y), h2_to_bf2(hi.y));
        d[1] = make_uint4(h2_to_bf2(lo.z), h2_to_bf2(hi.z), h2_to_bf2(lo.w), h2_to_bf2(hi.w));
    }
}

inline void launch(const Xfrag& a, cudaStream_t st)
{
    if (a.on) k_xfrag<<<dim3((a.K / 16 * 8 + 255) / 256, g6mx::tiles(a.M)), 256, 0, st>>>(a);
}

// ------------------------------------------------------------------------------------------ K2 + dense GEMVs
enum { INT4 = 0, MXFP8 = 1, BF16W = 2 };

struct Gv
{
    int on = 0;
    int codec = 0;
    const uint8_t* w = nullptr;       // records [O/16][NG][GR]
    const u32* xfrag = nullptr;       // [K/16][32][2], BF16 fragments for codec BF16W
    int K = 0, O = 0, tpc = 1, ks = 1, M = 0;
    void* y = nullptr;                // [8][ld] of odt
    int ld = 0, odt = gv::O_F16;
#ifdef DEC_SLOT
    dslot::Slot slot;                 // send != nullptr: BF16 output into the send slot instead of y (gv_gemv_slot)
#endif
};

// largest tiles-per-CTA <= tpc whose CTA (tpc x ks warps) fits the kernel's register budget; the output tiles of a CTA are
// independent, so fewer per CTA (more CTAs) gives the same bits. NT = 1 kernels always fit (the M <= 8 launch is today's).
template <class K> inline int gv_fit_tpc(K* kern, int tpc, int ks)
{
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(kern)) != cudaSuccess) return tpc;
    const int t = fa.maxThreadsPerBlock / (ks * 32);
    return t < 1 ? 1 : (t < tpc ? t : tpc);
}
inline void launch(const Gv& g, cudaStream_t st)
{
    if (!g.on) return;
    const int ot = g.O / 16;
    g6mx::nt_dispatch(g6mx::tiles(g.M), [&](auto N) {
        constexpr int NTv = decltype(N)::value;     // row tiles of 8: M <= 8 launches the NT = 1 kernel
        auto go = [&](auto* kern, auto&&... tail) {
            const int tpc = NTv == 1 ? g.tpc : gv_fit_tpc(kern, g.tpc, g.ks);
            kern<<<(ot + tpc - 1) / tpc, tpc * g.ks * 32, 0, st>>>(g.w, g.xfrag, g.K, ot, tpc, g.ks, g.M, tail...);
        };
        const bool b512 = g.ks <= 16;      // NT > 1: the 512-thread kernels (more registers) when one tile's ks warps fit
#ifdef DEC_SLOT
        if (g.slot.send)
        {
            if (g.codec == INT4) { if constexpr (NTv == 1) go(gv_gemv_slot<Int4v, 1>, g.ld, g.slot); else if (b512) go(gv_gemv_slot_lb512<Int4v, NTv>, g.ld, g.slot); else go(gv_gemv_slot_lb<Int4v, NTv>, g.ld, g.slot); }
            else { if constexpr (NTv == 1) go(gv_gemv_slot<Mxfp8, 1>, g.ld, g.slot); else if (b512) go(gv_gemv_slot_lb512<Mxfp8, NTv>, g.ld, g.slot); else go(gv_gemv_slot_lb<Mxfp8, NTv>, g.ld, g.slot); }
            return;
        }
#endif
        if (g.codec == INT4) { if constexpr (NTv == 1) go(gv_gemv<Int4v, 1>, g.y, g.ld, g.odt); else if (b512) go(gv_gemv_lb512<Int4v, NTv>, g.y, g.ld, g.odt); else go(gv_gemv_lb<Int4v, NTv>, g.y, g.ld, g.odt); }
        else if (g.codec == MXFP8) { if constexpr (NTv == 1) go(gv_gemv<Mxfp8, 1>, g.y, g.ld, g.odt); else if (b512) go(gv_gemv_lb512<Mxfp8, NTv>, g.y, g.ld, g.odt); else go(gv_gemv_lb<Mxfp8, NTv>, g.y, g.ld, g.odt); }
        else { if constexpr (NTv == 1) go(gv_gemv<Bf16w, 1>, g.y, g.ld, g.odt); else if (b512) go(gv_gemv_lb512<Bf16w, NTv>, g.y, g.ld, g.odt); else go(gv_gemv_lb<Bf16w, NTv>, g.y, g.ld, g.odt); }
    });
}

// the GEMVs gs as one gv_multi launch; out.n = 0 (launch them one by one) above 8 rows, where the 1024-thread bound's
// 64 registers slow the multi-tile kernels (gr8 #86), or when one does not fit a 1024-thread CTA or writes a send slot
inline void make_gv_multi(std::initializer_list<const Gv*> gs, int M, GvMulti& out)
{
    out = GvMulti();
    if (M > 8) return;
    for (const Gv* g : gs)
    {
        if (!g->on) continue;
#ifdef DEC_SLOT
        if (g->slot.send) { out = GvMulti(); return; }
#endif
        if (g->tpc * g->ks > 32 || out.n == GvMulti::MAXP) { out = GvMulti(); return; }
        const int ot = g->O / 16;
        out.p[out.n++] = GvPart{g->w, g->xfrag, g->y, g->codec, g->K, ot, g->tpc, g->ks, g->ld, g->odt, out.ctas};
        out.ctas += (ot + g->tpc - 1) / g->tpc;
    }
    out.M = M;
}
static_assert(INT4 == 0 && MXFP8 == 1 && BF16W == 2, "gv_multi dispatches on these codec numbers");
inline void launch(const GvMulti& a, cudaStream_t st) { gv_multi<<<a.ctas, 1024, 0, st>>>(a); }

// tpc (output tiles per CTA) and ks (k parts per tile, one warp each): best split per shape on the GB10
// (RESULTS-M1 items 2, 2b; time is flat in M). ks is halved until it divides the tensor's group count.
inline const char* make_gv(const Tensor& t, int codec, int tpc, int ks, int M, const u32* xfrag, void* y, int odt, Gv& g)
{
    g = Gv();
    if (!t.p) return nullptr;
    const int GR = codec == INT4 ? 288 : codec == MXFP8 ? 528 : 512, gk = codec == BF16W ? 16 : 32;
    if (t.nd != 3 || t.shape[2] != GR) return "GEMV tensor is not a [O/16][NG][GR] record of its codec";
    const int ng = static_cast<int>(t.shape[1]);
    while (ks > 1 && ng % ks) ks >>= 1;
    while (tpc * ks > 32) tpc >>= 1;
    g.on = 1; g.codec = codec;
    g.w = static_cast<const uint8_t*>(t.p);
    g.xfrag = xfrag;
    g.K = ng * gk; g.O = static_cast<int>(t.shape[0]) * 16;
    g.tpc = tpc; g.ks = ks; g.M = M;
    g.y = y; g.ld = g.O; g.odt = odt;
    return nullptr;
}

// ------------------------------------------------------------------------------------------ K1
struct K1
{
    int on = 0, dense = 0;
    K1Args a;
    int g1 = 0, b1 = 0, g2 = 0, b2 = 0, g3 = 0, b3 = 0;
    size_t sm1 = 0, sm2 = 0;
    int fuse = 0, gf = 0;             // fuse (1 | 2): one persistent fc1+fc2 launch (k1_tr3_fused, gf CTAs of 512) instead of fc1, fc2
    size_t smf = 0;
    int e4 = 0;                       // exlr8-e4m3 layer: the E4 instantiations of fc1, fc2, fused (chosen once, in make_k1)
    int k2 = 0;                       // layer with 2-bit units (LayerTensors::k2): the K2 instantiations
    int sh = 0;                       // trellis shared expert (LayerTensors::sh_trellis): the SH instantiations (E4 only)
};

// the (E4, K2, SH) instantiations of a K1 kernel template; SH implies E4 (make_k1)
#define K1_PICK(kern, k, ...) \
    do { if ((k).sh) { if ((k).k2) kern<true, true, true>__VA_ARGS__; else kern<true, false, true>__VA_ARGS__; } \
         else if ((k).e4) { if ((k).k2) kern<true, true>__VA_ARGS__; else kern<true, false>__VA_ARGS__; } \
         else { if ((k).k2) kern<false, true>__VA_ARGS__; else kern<false, false>__VA_ARGS__; } } while (0)
template <bool E4, bool K2, bool SH> inline const void* k1_fn_fc1() { return reinterpret_cast<const void*>(k1_tr3_fc1<E4, K2, SH>); }
template <bool E4, bool K2, bool SH> inline const void* k1_fn_fc2() { return reinterpret_cast<const void*>(k1_tr3_fc2<E4, K2, SH>); }
template <bool E4, bool K2, bool SH> inline const void* k1_fn_fused()
{
    if constexpr (SH) return reinterpret_cast<const void*>(k1_tr3_fused_sh<E4, K2>);
    else return reinterpret_cast<const void*>(k1_tr3_fused<E4, K2, false>);
}

inline void launch_k1_fc1(const K1& k, cudaStream_t st)
{
    if (!k.on) return;
    K1_PICK(k1_tr3_fc1, k, <<<k.g1, k.b1, k.sm1, st>>>(k.a));
}
inline void launch_k1_fc2(const K1& k, cudaStream_t st)
{
    if (!k.on) return;
    K1_PICK(k1_tr3_fc2, k, <<<k.g2, k.b2, k.sm2, st>>>(k.a));
}
inline void launch_k1_fused(const K1& k, cudaStream_t st)
{
    if (!k.on) return;
    if (k.sh)
    {
        if (k.k2) k1_tr3_fused_sh<true, true><<<k.gf, 512, k.smf, st>>>(k.a);
        else k1_tr3_fused_sh<true, false><<<k.gf, 512, k.smf, st>>>(k.a);
    }
    else K1_PICK(k1_tr3_fused, k, <<<k.gf, 512, k.smf, st>>>(k.a));
}
inline void launch_k1_fin(const K1& k, cudaStream_t st)
{
    if (!k.on || k.dense) return;
    if (k.sh) k1_tr3_fin<true><<<k.g3, k.b3, 0, st>>>(k.a);
    else k1_tr3_fin<<<k.g3, k.b3, 0, st>>>(k.a);
}
inline void launch_k1(const K1& k, cudaStream_t st)
{
    if (k.fuse) launch_k1_fused(k, st);
    else
    {
        launch_k1_fc1(k, st);
        launch_k1_fc2(k, st);
    }
    launch_k1_fin(k, st);
}
// persistent grid of the fused launch: min(items, resident CTAs of the device at its smem size)
inline int k1_fused_grid(int items, size_t smem, int e4 = 0, int k2 = 0, int sh = 0)
{
    int dev = 0, nsm = 0, nb = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 0;
    // the attribute is per function (process-wide): set the maximum as init() does, never a layer's own smem, or a later
    // make_layer with a smaller split lowers it below an earlier layer's launch size (invalid argument at launch)
    if (smem > static_cast<size_t>(KSH) * 256) return 0;
    const void* fn = sh ? (k2 ? k1_fn_fused<true, true, true>() : k1_fn_fused<true, false, true>())
                   : e4 ? (k2 ? k1_fn_fused<true, true, false>() : k1_fn_fused<true, false, false>())
                        : (k2 ? k1_fn_fused<false, true, false>() : k1_fn_fused<false, false, false>());
    if (cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, KSH * 256) != cudaSuccess) return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, 512, smem) != cudaSuccess || nb < 1) return 0;
    const int g = nsm * nb;
    return items < g ? items : g;
}

// fc1 k-splits per CTA grid, fc1 warp groups, fc2 n-splits, shared-expert fc1 k parts and fc2 out splits.
// Best of the GB10 tune at M = 1, 3, 5 (engine/results/item1-tune.jsonl, the buckets of run_m1.split_for);
// M = 2 uses the M = 3 split, M = 4 and M >= 6 the M = 5 split (not tuned separately).
#ifndef TR3_W2SPLIT
#define TR3_W2SPLIT 0
#endif
struct K1Split { int KS1, WG, NS2, s_ks, s_ns2; };
// W1b (TR3_XCOMP builds): NS2 = 2 at 2-3 rows only for layers with 2-bit routed units (k2); every other layer, and
// every layer of a default build, keeps the release split (NS2 = 8). TP3-specific tuning (window1 microbench, K2).
inline K1Split k1_split(int M, int k2 = 0)
{
    K1Split s = M == 1 ? K1Split{4, 1, 2, 8, 24} : M <= 3 ? K1Split{1, 2, TR3_XCOMP && k2 ? 2 : 8, 8, 16} : K1Split{4, 2, 2, 8, 24};
#if TR3_W2SPLIT
    // Window2 winner (single-GB10 harness, K2 layers): 256-thread fc1 CTAs with the same k ranges (KS1 x2, WG 1) at M > 1.
    // Baked, no env (harness K1_KW). NS2 = 2 at M 2..3 is the W1b rule above (k2 layers). Non-k2 layers keep release.
    if (k2 && M > 1 && s.WG == 2) { s.KS1 *= 2; s.WG = 1; }
#endif
    return s;
}
constexpr K1Split K1_DENSE_SPLIT = {4, 2, 1, 8, 24};   // dense layers 0..2: only WG, s_ks, s_ns2 matter. Not tuned.

// The route plan of the call (glue::RoutePlan or any struct with the same members), device memory.
struct PlanPtrs
{
    const int* n_slots = nullptr;
    const int* slot_expert = nullptr;
    const int* slot_mask = nullptr;
    const float* slot_w = nullptr;
    const int* slot_tok = nullptr;
    const int* tok_slot = nullptr;
    const int* tok_col = nullptr;
    const float* tok_w = nullptr;
};
template <class P> inline PlanPtrs plan_ptrs(const P* p)
{
    PlanPtrs q;
    q.n_slots = &p->n_slots; q.slot_expert = p->slot_expert; q.slot_mask = p->slot_mask;
    q.slot_w = &p->slot_w[0][0]; q.slot_tok = &p->slot_tok[0][0]; q.tok_slot = &p->tok_slot[0][0]; q.tok_col = &p->tok_col[0][0]; q.tok_w = &p->tok_w[0][0];
    return q;
}

// x2 = mlp_norm output fp16 [8][6144], xf_x2 = its fp16 fragment table, out = [8][6144] (BF16 bits if out_bf16).
inline const char* make_k1(const LayerTensors& T, int M, const Scratch& sc, const __half* x2, const u32* xf_x2,
                           const PlanPtrs& plan, void* out, int out_bf16, K1& k, Tensor* sh_out, int fuse = 0, int split_m = 0)
{
    k = K1();
    const bool moe = T.t[T_TR3_TAB].p != nullptr;
    const bool sh = moe && T.sh_trellis;                                 // shared expert = trellis row 256 (#65)
    const bool mx = moe && !sh && !T.t[T_S_FC1].p && T.t[T_S_GATE_MX].p; // shared expert in MXFP8 records (draft layer)
    static const Tensor none;
    const Tensor& f1 = sh ? none : T.t[mx ? T_S_GATE_MX : moe ? T_S_FC1 : T_D_FC1];
    const Tensor& f2 = sh ? none : T.t[mx ? T_S_DOWN_MX : moe ? T_S_FC2 : T_D_FC2];
    const Tensor& fu = T.t[T_S_UP_MX];
    if (sh) { sh_out[0] = sh_out[1] = sh_out[2] = none; }
    else if (!f1.p || !f2.p || (mx && !fu.p))
        return mx ? "MoE layer with shared.gate.mx.gv but without shared.up.mx.gv / shared.down.mx.gv"
             : moe ? "MoE layer with neither shared.gptq.fc1/fc2 nor shared.gate/up/down.mx.gv" : "layer has neither tr3.table nor dense.gptq.fc1/fc2";
    if (!sh) { sh_out[0] = f1; sh_out[1] = fu; sh_out[2] = f2; }
    const int s_nch = sh ? 0 : static_cast<int>(f1.shape[0]);
    if (sh) {}
    else if (mx)
    {
        if (f1.nd != 3 || f1.shape[1] != 192 || f1.shape[2] != 528 || fu.nd != 3 || fu.shape[0] != s_nch || fu.shape[1] != 192 ||
            fu.shape[2] != 528 || f2.nd != 3 || f2.shape[0] != KSH || f2.shape[1] * 2 != s_nch || f2.shape[2] != 528 || s_nch * 16 > SACTW)
            return "shared MXFP8 records are not gate, up [I/16][192][528] + down [384][I/32][528] with I <= SACTW";
    }
    else if (f1.nd != 3 || f1.shape[1] != 192 || f1.shape[2] != 576 || f2.nd != 3 || f2.shape[0] != KSH || f2.shape[1] * 2 != s_nch ||
             f2.shape[2] != 288 || s_nch * 16 > SACTW)
        return "shared / dense MLP records are not [I/16][192][576] + [384][I/32][288] with I <= SACTW";
    if (M < 1 || M > G6_MX) return "M must be 1..G6_MX (8, or 32 under DEC_S2)";
    // split_m (S2 design 2.1): the split of a split_m-row step (a session's row class) at any M
    K1Split s = moe ? k1_split(split_m ? split_m : M, T.k2) : K1_DENSE_SPLIT;
    // exlr8 v2 layers (2-bit units or the trellis shared expert, #65) choose fused or separate fc1 / fc2 launches from a
    // table timed on GB10 (TP4 rank 0, MoE step, G6_FUSE=1 only). Separate wins at 2 rows for every layer (3-7.5%), and
    // at 3+ rows when 2-bit units are at least half the bytes (5-11%). Fused wins at 1 row (1-3%) and for mostly 3/4-bit
    // layers at 5-8 rows (3-5%). TR3 and v1 layers keep G6_FUSE as given (that choice is #8's).
    static const bool policy = !getenv("G6_K1_FUSE_POLICY") || atoi(getenv("G6_K1_FUSE_POLICY"));   // 0: A/B runs only
    const int rows = split_m ? split_m : M;
    if (policy && fuse == 1 && moe && (T.k2 || T.sh_trellis) && (rows == 2 || (rows >= 3 && T.k2 && T.k2_pct >= 50))) fuse = 0;
    // 16+ rows in the step (S2 batches; split_m keeps rows at the session's class): separate launches, 2-3.5% faster at
    // 24-32 rows (row_budget_test --time-v2, GB10 2200 MHz: M=24 1551 vs 1577 us, M=32 1752 vs 1812 us)
    if (policy && fuse == 1 && moe && T.sh_trellis && M >= 16) fuse = 0;
    if (fuse) { s.KS1 = s.KS1 * s.WG / 2; s.WG = 2; }     // 512-thread CTAs; same k ranges, same partial order (k1.cuh)
    // fuse = 2: where fc1 sums 2 k partials (M = 2, 3) split it in 4 (KS1 = 2). Half the fc1 shared memory per CTA;
    // NOT the bytes of fuse = 0 / 1 there: the fp32 fc1 sums are grouped differently (RESULTS-M1 item 1f).
    if (fuse == 2 && s.KS1 == 1) s.KS1 = 2;
    const int w1 = 8 * s.WG;

    K1Args& a = k.a;
    memset(&a, 0, sizeof a);
    a.x = x2; a.xfrag = xf_x2;
    a.ctl = sc.ctl; a.n_plans = 1; a.M = M;
    a.B = g6nr::EB; a.KS1 = s.KS1; a.WG = s.WG; a.NS2 = s.NS2;
    a.s_fc1 = static_cast<const uint8_t*>(f1.p); a.s_fc2 = static_cast<const uint8_t*>(f2.p);
    a.s_up = mx ? static_cast<const uint8_t*>(fu.p) : nullptr; a.s_mx = mx;
    a.s_nch = s_nch; a.s_ks = s.s_ks; a.s_cw = w1 / s.s_ks; a.s_ns2 = s.s_ns2;
    a.gpart = sc.gpart; a.z = sc.z; a.act = sc.act; a.ypart = sc.ypart; a.fctl = sc.fctl;
    a.smax = sc.smax; a.ysh = sc.ysh;
    if (M > sc.rows) return "M exceeds the row limit the scratch block was laid out for";
    a.out = static_cast<__half*>(out); a.out_bf16 = out_bf16;
    if (moe)
    {
        const Tensor &tab = T.t[T_TR3_TAB], &v1 = T.t[T_TR3_V1], &v2 = T.t[T_TR3_V2], &v3 = T.t[T_TR3_V3];
        const Tensor &sg = T.t[T_TR3_SG], &v1b = T.t[T_TR3_V1B], &v3b = T.t[T_TR3_V3B];
        const bool bsf = v1b.p != nullptr;           // block-scale hidden-side vectors (exlr8 v2 with the trellis shared expert)
        if (!T.t[T_TR3_FC1].p || !T.t[T_TR3_FC2].p || !v2.p || (bsf ? !sg.p || !v3b.p : !v1.p || !v3.p))
            return "MoE layer without tr3.fc1/fc2/v2 and v1/v3 (or sg/v1b/v3b)";
        const long long rows = v2.shape[0];
        if (tab.nd != 2 || tab.shape[1] != 6 || v2.nd != 3 || v2.shape[1] != 3 || v2.shape[2] != 128 ||
            (bsf ? sg.nd != 2 || sg.shape[0] != 2 || sg.shape[1] != HID || v1b.nd != 3 || v1b.shape[0] != rows || v1b.shape[1] != 2 ||
                       v1b.shape[2] != HID / 128 || v3b.nd != 2 || v3b.shape[0] != rows || v3b.shape[1] != HID / 128
                 : v1.nd != 3 || v1.shape[1] != 2 || v1.shape[2] != HID || v3.nd != 2 || v3.shape[1] != HID || v1.shape[0] != rows ||
                       v3.shape[0] != rows))
            return "tr3 table / v1 / v2 / v3 (or sg / v1b / v3b) shapes do not match the pack layout";
        for (int i = T_TR3_FC1; i <= T_TR3_V3; ++i)
            if (T.t[i].p && T.t[i].fmt != tab.fmt) return "tr3.* tensors of one layer carry different expert formats";
        if (bsf != sh) return "block-scale hidden-side vectors (tr3.sg / v1b / v3b) go with the trellis shared expert, and only with it";
        const Tensor& km = T.t[T_TR3_KMIX];
        if (km.p && (km.nd != 1 || km.shape[0] != tab.shape[0])) return "tr3.kmix is not one entry per tr3.table row";
        k.e4 = T.e4m3;
        k.k2 = T.k2;
        k.sh = sh;
        if (sh && !k.e4) return "trellis shared expert in a layer that is not exlr8-e4m3";
        if (sh && T.t[T_TR3_TAB].shape[0] != 257) return "trellis shared expert without tr3.table row 256";
        a.sh_e = 256;
        if (!plan.n_slots || !plan.slot_expert || !plan.slot_mask || !plan.slot_w || !plan.slot_tok || !plan.tok_slot || !plan.tok_col || !plan.tok_w) return "MoE layer needs the route plan";
        a.n_slots = M * 8;
        // routed slots, then the trellis shared expert's (one per 8 rows): scratch_smax leaves room for both
        if (a.n_slots + (sh ? g6mx::tiles(M) : 0) > sc.smax) return "the scratch block has too few K1 slots for M rows";
        a.n_slots_dev = plan.n_slots;
        a.slot_expert = plan.slot_expert; a.slot_mask = plan.slot_mask; a.slot_w = plan.slot_w; a.slot_tok = plan.slot_tok;
        a.tok_slot = plan.tok_slot; a.tok_col = plan.tok_col; a.tok_w = plan.tok_w;
        a.t_tab = static_cast<const long long*>(tab.p);
        a.t_kmix = static_cast<const long long*>(km.p);
        a.t_sg = static_cast<const __half*>(sg.p); a.t_v1b = static_cast<const __half*>(v1b.p); a.t_v3b = static_cast<const __half*>(v3b.p);
        a.t_fc1 = static_cast<const uint8_t*>(T.t[T_TR3_FC1].p); a.t_fc2 = static_cast<const uint8_t*>(T.t[T_TR3_FC2].p);
        a.t_wide = T.fc1_wide; a.t_wide2 = T.fc2_wide;
        a.t_v1 = static_cast<const __half*>(v1.p); a.t_v2 = static_cast<const __half*>(v2.p); a.t_v3 = static_cast<const __half*>(v3.p);
    }
    else
    {
        a.n_slots = 0;
        a.n_slots_dev = sc.ctl + 3;       // a zero; never read (no routed CTAs)
        a.s_direct = 1;
        k.dense = 1;
    }
    const int sh1 = sh ? 0 : (s_nch + a.s_cw - 1) / a.s_cw;
    const int nsh = sh ? g6mx::tiles(M) : 0;        // trellis shared-expert slots (k1.cuh k1_nsh)
    k.g1 = (a.n_slots + nsh) * a.B * 2 * a.KS1 + sh1; k.b1 = 256 * s.WG;
    // TR3_XCOMP (W1b): compact x table only for separate fc1 launches at M <= 3 (the step M, not split_m: an S2
    // step of 8 rows in the 2-3 row class has KS1 = 1 and 8 * (384 * 16 + 8) * 2 = 98432 B > the KSH * 256 = 98304 B
    // attribute -> invalid argument at launch, the bin-w1 crash). The fused kernels keep the fragment table.
    // M <= 3 is one row tile, and column c of a slot is token c there (glue.cuh RoutePlan), so the table's rows are columns.
    a.xc = (TR3_XCOMP && !fuse && M <= 3) ? 1 : 0;
    k.sm1 = a.xc ? static_cast<size_t>(M) * (KSH / a.KS1 * 16 + 8) * 2 : static_cast<size_t>(KSH / a.KS1) * 256;
    if (k.sm1 < static_cast<size_t>(w1) * 1024 * g6mx::tiles(M)) k.sm1 = static_cast<size_t>(w1) * 1024 * g6mx::tiles(M);   // fc1 k-part sums, one 1 KB slab per warp and row tile
    k.g2 = (a.n_slots + nsh) * a.B * a.NS2 + (sh ? 0 : a.s_ns2); k.b2 = 512;
    // shared fc2: B tables of s_ntp row tiles in smem per pass over the weight stream (1 at M <= 8: sm2 as before)
    const int s_tab = (s_nch < 8 ? 8 : s_nch) * 256, ntot = g6mx::tiles(M);
    const int fit = KSH * 256 / s_tab;
    a.s_ntp = ntot < fit ? ntot : (fit < 1 ? 1 : fit);
    k.sm2 = static_cast<size_t>(s_tab) * a.s_ntp;
    k.g3 = M * 48; k.b3 = sh ? 160 : 128;            // SH: a fifth finalize warp sums the trellis shared expert
    if (fuse)
    {
        k.fuse = fuse;
        k.smf = k.sm1 > k.sm2 ? k.sm1 : k.sm2;
        k.gf = k1_fused_grid(k.g1 + k.g2, k.smf, k.e4, k.k2, k.sh);
        if (!k.gf) return "fused K1 launch: no resident CTA at its shared memory size";
    }
    k.on = 1;
    return nullptr;
}

// ------------------------------------------------------------------------------------------ fabric partial
// Device staging [M][6144] BF16 -> fabric send slot (pinned host memory) with 16-byte stores from one CTA:
// the producers (K1 finalize, dense fc2, o_proj) store 2 or 8 bytes per thread from thousands of threads, which is
// the access pattern measured as slow on pinned memory. Point io.attn_part / io.mlp_part at
// Scratch::part[0] / part[1] and copy with this; or point them at the send slot and skip it. grid 1 (M <= 8) or 16 CTAs, block 128.
struct Put { const uint4* src; uint4* dst; int n16; };      // n16 = M * 6144 * 2 / 16
__global__ void __launch_bounds__(128) k_put_partial(Put a)
{
    for (int i = blockIdx.x * 128 + threadIdx.x; i < a.n16; i += gridDim.x * 128) a.dst[i] = a.src[i];
}
inline void put_partial(const uint16_t* staging, void* send_slot, int M, cudaStream_t st)
{
    const Put a = {reinterpret_cast<const uint4*>(staging), static_cast<uint4*>(send_slot), M * (HID * 2 / 16)};
    k_put_partial<<<M <= 8 ? 1 : 16, 128, 0, st>>>(a);
}

// ------------------------------------------------------------------------------------------ one layer of one rank
struct Io                             // step buffers of one rank; device memory (attn_part / mlp_part: see put_partial)
{
    const __half* x = nullptr;        // in  [8][6144] attn_norm output
    __half* q_a = nullptr;            // out [8][2048]
    const __half* q_a_n = nullptr;    // in  [8][2048] q_a_norm output
    __half* kv_a = nullptr;           // out [8][576]
    __half* q_b = nullptr;            // out [8][H*256]
    __half* idx_q = nullptr;          // out [8][4096]   idx.wq_b   (indexer layers)
    __half* idx_k = nullptr;          // out [8][128]    idx.wk
    __half* idx_w = nullptr;          // out [8][32]     idx.weights_proj
    const __half* attn = nullptr;     // in  [M][H*256] attention output, contiguous rows
    void* attn_part = nullptr;        // out [8][6144] BF16   o_proj partial
    const __half* x2 = nullptr;       // in  [8][6144] mlp_norm output
    float* logits = nullptr;          // out [8][256] fp32 router logits (MoE layers)
    PlanPtrs plan;                    // in  route plan written by glue::k_router_plan (MoE layers)
    void* mlp_part = nullptr;         // out [8][6144] BF16   MLP partial
};

struct Layer
{
    int M = 0, moe = 0, indexer = 0, heads = 0;
    int w8 = 0;                       // bit 0: q_a, bit 1: o_proj, bit 2: shared expert run from MXFP8 records (draft layer: 7)
    Xfrag xf_x, xf_qa, xf_attn, xf_x2;
    Gv q_a, kv_a, q_b, idx_q, idx_k, idx_w, o_proj, router;
    GvMulti x_in;                     // q_a, kv_a, idx_k, idx_w as one launch (n = 0: one launch each)
    K1 k1;
    Tensor sh[3];                     // the K1's shared / dense MLP records (fc1 or gate, up, fc2 or down), for warm_mlp
    const void* shw = nullptr;        // trellis shared expert (#65): its fc1 tiles (tr3.table row 256) and their bytes,
    size_t shw_bytes = 0;             // for warm_mlp in place of sh[]
    // dense trellis matrices (dtr.cuh). make_layer binds them where the layer has them, and leaves q_b, o_proj and the
    // dense MLP's K1 off there. launch_q_b / launch_o_proj / launch_mlp run whichever is on.
    dtr::Gemv q_b_t, o_proj_t, glu_t, down_t;
};

inline void launch_q_b(const Layer& L, cudaStream_t st) { launch(L.q_b, st); dtr::launch(L.q_b_t, st); }
inline void launch_o_proj(const Layer& L, cudaStream_t st) { launch(L.o_proj, st); dtr::launch(L.o_proj_t, st); }
inline void launch_mlp(const Layer& L, cudaStream_t st)
{
    launch_k1(L.k1, st);
    dtr::launch(L.glu_t, st);
    dtr::launch(L.down_t, st);
}
#ifdef DEC_SLOT
// the send slots of the layer's two partials: attention (o_proj) and MLP (K1 finalize / dense fc2 / trellis down)
inline void set_slots(Layer& L, const dslot::Slot& attn, const dslot::Slot& mlp)
{
    L.o_proj.slot = attn; L.o_proj_t.a.slot = attn;
    L.k1.a.slot = mlp; L.down_t.a.slot = mlp;
}
#endif

// ---------------------------------------------------------------------------------------------- L2 warm-up
// launch_warm (common.cuh) on a binding's weights: loads only, results dropped, no output changes. Launched on the
// step's stream just before a k_wait_peers, the read runs while the rank waits for its peers (DRAM idle), and the
// kernel after the wait finds the bytes in the GB10 L2 (24 MB; re-reads run about 385 GB/s against about 190 from
// DRAM, RESULTS-M1 item 1b). Sizing: a warm-up that is longer than the wait it hides in runs serially and costs
// its DRAM time, so each set is kept near one wait (about 30 us = 6 MB per the window 3 profile). On a side stream
// (event fork before the wait, join after the add+norm) the serial risk is gone and the sets may grow.
// Untested on a GB10: measure per shape with `m1 gv ... warm=0|1|2` (slot_win3.sh section 3) before wiring.
inline size_t rec_bytes(const Tensor& t)
{
    if (!t.p) return 0;
    size_t b = 1;
    for (int i = 0; i < t.nd; ++i) b *= static_cast<size_t>(t.shape[i]);
    return b;
}
inline size_t gv_bytes(const Gv& g)
{
    const int GR = g.codec == INT4 ? 288 : g.codec == MXFP8 ? 528 : 512, gk = g.codec == BF16W ? 16 : 32;
    return g.on ? static_cast<size_t>(g.O / 16) * (g.K / gk) * GR : 0;
}
inline void warm(const Gv& g, cudaStream_t st) { if (g.on) launch_warm(g.w, gv_bytes(g), st); }
// before the wait for the attention partials (collective 2i): this layer's router (3.15 MB) and shared / dense fc1
// (2.4 MB int4; gate + up 3.0 MB MXFP8). The routed experts are unknown until the router ran, so they cannot be warmed.
inline void warm_mlp(const Layer& L, cudaStream_t st)
{
    warm(L.router, st);
    launch_warm(L.sh[0].p, rec_bytes(L.sh[0]), st);
    launch_warm(L.sh[1].p, rec_bytes(L.sh[1]), st);
    launch_warm(L.shw, L.shw_bytes, st);
}
// before the wait for the MLP partials (collective 2i+1): the NEXT layer's q_a (7.1 MB). kv_a and q_b (9.6 MB more)
// only on a side stream or after the GB10 numbers say the wait is long enough.
inline void warm_proj(const Layer& next, cudaStream_t st, int more = 0)
{
    warm(next.q_a, st);
    if (more) { warm(next.kv_a, st); warm(next.q_b, st); }
}

// fuse = 1: the routed and shared fc1 + fc2 as one persistent launch (K1::fuse, k1_tr3_fused); same bytes out.
// fuse = 2: the same launch with the 4-partial fc1 split at M = 2, 3 (other bytes there, see make_k1).
// split_m: as make_k1 (0 = the split of M rows)
inline const char* make_layer(const LayerTensors& T, int M, const Scratch& sc, const Io& io, Layer& L, int fuse = 0, int split_m = 0)
{
    L = Layer();
    if (M < 1 || M > G6_MX) return "M must be 1..G6_MX (8, or 32 under DEC_S2)";
    L.M = M;
    L.moe = T.t[T_TR3_TAB].p != nullptr;
    L.indexer = T.t[T_IDX_WQ_B].p != nullptr;
    const char* e = nullptr;
    // G6_DTR (default 1): the dense trellis matrices of the exlr8 release form (dtr.cuh) where the layer has them;
    // 0 binds the int4 / MXFP8 records instead (a layer must then have them). The int4 record wins over MXFP8 when a
    // layer has both. MXFP8 q_a / o_proj splits: five splits each swept on the GB10 at M = 1, 3
    // (measured on one GB10); all within 3% (best time) and no split wins at both M, so these stay.
    static const bool dtr_on = !getenv("G6_DTR") || atoi(getenv("G6_DTR"));
    if (dtr_on && T.mats_err) return T.mats_err;
    const dtr::LayerMats& D = T.mats;
    const bool tq = dtr_on && D.q_b.w, to = dtr_on && D.o_proj.w, tm = dtr_on && D.gate.w;
    const bool qmx = !T.t[T_Q_A].p && T.t[T_Q_A_MX].p, omx = !T.t[T_O_PROJ].p && T.t[T_O_PROJ_MX].p;
    const bool qi4 = !tq && T.t[T_Q_B_I4].p;
    //                 tensor             codec  tpc ks      fragments   output       dtype
    if (qmx) e = make_gv(T.t[T_Q_A_MX], MXFP8, 2,  16, M, sc.xf_x,    io.q_a,       gv::O_F16,  L.q_a);
    else     e = make_gv(T.t[T_Q_A],    INT4,  4,  8,  M, sc.xf_x,    io.q_a,       gv::O_F16,  L.q_a);
    if (e) return e;
    if ((e = make_gv(T.t[T_KV_A],      MXFP8, 2,  16, M, sc.xf_x,    io.kv_a,      gv::O_F16,  L.kv_a))) return e;
    if (qi4) e = make_gv(T.t[T_Q_B_I4], INT4,  2,  8,  M, sc.xf_qa,   io.q_b,       gv::O_F16,  L.q_b);
    else if (!tq) e = make_gv(T.t[T_Q_B], MXFP8, 1, 16, M, sc.xf_qa,  io.q_b,       gv::O_F16,  L.q_b);
    if (e) return e;
    if ((e = make_gv(T.t[T_IDX_WQ_B],  MXFP8, 1,  32, M, sc.xf_qa,   io.idx_q,     gv::O_F16,  L.idx_q))) return e;
    if ((e = make_gv(T.t[T_IDX_WK],    BF16W, 1,  32, M, sc.xb_x,    io.idx_k,     gv::O_F16,  L.idx_k))) return e;
    if ((e = make_gv(T.t[T_IDX_WPROJ], BF16W, 1,  32, M, sc.xb_x,    io.idx_w,     gv::O_F16,  L.idx_w))) return e;
    if (to) {}
    else if (omx) e = make_gv(T.t[T_O_PROJ_MX], MXFP8, 2, 8, M, sc.xf_attn, io.attn_part, gv::O_BF16, L.o_proj);
    else     e = make_gv(T.t[T_O_PROJ],    INT4,  2, 8, M, sc.xf_attn, io.attn_part, gv::O_BF16, L.o_proj);
    if (e) return e;
    if ((e = make_gv(T.t[T_ROUTER],    BF16W, 2,  16, M, sc.xb_x2,   io.logits,    gv::O_F32,  L.router))) return e;
    if (!L.q_a.on || !L.kv_a.on || !(L.q_b.on || tq) || !(L.o_proj.on || to))
        return "layer without q_a.gptq (or q_a.mx.gv) / kv_a.mx.gv / q_b (trellis, q_b.gptq or q_b.mx.gv) / o_proj (trellis, o_proj.gptq or o_proj.mx.gv)";
    if (L.moe != L.router.on) return "router.w.gv and tr3.table must come together";
#if DEC_I4PAIR
    if ((!qmx && L.q_a.ks != T.i4pair_ks[0]) || (!omx && !to && L.o_proj.ks != T.i4pair_ks[1]) || (qi4 && L.q_b.ks != T.i4pair_ks[2]))
        return "DEC_I4PAIR: int4 q_a / o_proj / q_b not repacked (bind::i4pair) for the launch's k-part count";
#endif
    const int qbK = tq ? D.q_b.K : L.q_b.K, qbO = tq ? D.q_b.N : L.q_b.O, opK = to ? D.o_proj.K : L.o_proj.K, opO = to ? D.o_proj.N : L.o_proj.O;
    if (L.q_a.K != HID || L.kv_a.K != HID || qbK != L.q_a.O || opO != HID || opK != qbO || qbO % 256)
        return "attention projection shapes do not chain (q_a -> q_b, q_b width = o_proj K)";
    L.heads = qbO / 256;
    if (tq && (e = dtr::make_out(D.q_b, M, io.q_a_n, qbK, nullptr, io.q_b, qbO, dtr::O_F16, sc.dt, 0, 0, 1, L.q_b_t))) return e;
    if (to && (e = dtr::make_out(D.o_proj, M, io.attn, opK, nullptr, io.attn_part, HID, dtr::O_BF16, sc.dt, 64, 0, 1, L.o_proj_t)))
        return e;
    static const bool multi = !getenv("G6_GV_MULTI") || atoi(getenv("G6_GV_MULTI"));
    if (multi) make_gv_multi({&L.q_a, &L.kv_a, &L.idx_k, &L.idx_w}, M, L.x_in);

    L.xf_x.on = 1;    L.xf_x.src = io.x;        L.xf_x.ld = HID;          L.xf_x.K = HID;          L.xf_x.f16 = sc.xf_x;
    L.xf_x.bf16 = L.indexer ? sc.xb_x : nullptr;
    // the fragment tables of q_a_norm (q_b, idx_q) and of the attention output (o_proj): a trellis matrix rotates its
    // input itself
    L.xf_qa.on = L.q_b.on || L.idx_q.on; L.xf_qa.src = io.q_a_n; L.xf_qa.ld = L.q_a.O; L.xf_qa.K = L.q_a.O; L.xf_qa.f16 = sc.xf_qa;
    L.xf_attn.on = L.o_proj.on; L.xf_attn.src = io.attn; L.xf_attn.ld = opK; L.xf_attn.K = opK; L.xf_attn.f16 = sc.xf_attn;
    L.xf_x2.on = 1;   L.xf_x2.src = io.x2;      L.xf_x2.ld = HID;         L.xf_x2.K = HID;         L.xf_x2.f16 = sc.xf_x2;
    L.xf_x2.bf16 = L.moe ? sc.xb_x2 : nullptr;
    L.xf_x.M = L.xf_qa.M = L.xf_attn.M = L.xf_x2.M = M;

    if (tm)
    {
        if (L.moe) return "dense trellis MLP tensors in a MoE layer";
        if (D.gate.N != g6nr::DENSE_CH) return "dense trellis MLP width is not DENSE_CH";
        if ((e = dtr::make_glu(D.gate, D.up, D.down, M, io.x2, HID, sc.dxd, sc.dt, 128, 0, 1, L.glu_t))) return e;
        if ((e = dtr::make_out(D.down, M, nullptr, 0, sc.dxd, io.mlp_part, HID, dtr::O_BF16, sc.dt, 192, 0, 1, L.down_t))) return e;
        L.xf_x2.on = 0;
    }
    else if ((e = make_k1(T, M, sc, io.x2, sc.xf_x2, io.plan, io.mlp_part, 1, L.k1, L.sh, fuse, split_m))) return e;
    L.w8 = (qmx ? 1 : 0) | (omx ? 2 : 0) | (L.k1.a.s_mx ? 4 : 0);
    if (L.k1.sh)                      // row 256's fc1 bytes: 2 matrices per block at each unit's width
    {
        size_t o0 = 0, o1 = 0;
        sh_unit_off(T, 0, o0);
        sh_unit_off(T, 2 * static_cast<int>(T.sh_tr[3]), o1);
        L.shw = static_cast<const uint8_t*>(T.t[T_TR3_FC1].p) + o0;
        L.shw_bytes = o1 - o0;
    }
    return nullptr;
}

// ------------------------------------------------------------------------------------------ head + MTP projection
struct HeadTensors
{
    Tensor lm_head;                   // G.lm_head.mx.gv   [vn/16][192][528]  this rank's vocab rows
    Tensor eh;                        // L078.eh.mx.gv     [384][384][528]    MTP eh_proj, O = 6144, K = 12288 (p = nullptr: no MTP)
};
template <class F> inline HeadTensors head_tensors(F find)
{
    HeadTensors T;
    T.lm_head = find("G.lm_head.mx.gv");
    T.eh = find("L078.eh.mx.gv");
    return T;
}

// Vocab slice of rank r (SPEC §9): 154880 rows = 1210 tiles of 128 in rank order, not rotated, the larger slices first
// (g6nr.h): TP6 [202,202,202,202,201,201] tiles, TP4 [303,303,302,302]. Returns false if r is not 0..NR-1.
inline bool vocab_slice(int rank, int& vlo, int& vn)
{
    if (rank < 0 || rank >= g6nr::NR) return false;
    vlo = g6nr::start(g6nr::VTILES, rank) * 128;
    vn = g6nr::width(g6nr::VTILES, rank) * 128;
    return true;
}

struct HeadIo                         // device memory
{
    const __half* h = nullptr;        // in  [8][6144]  lm_head input (final_norm output; MTP: shared_head_norm output)
    float* logits = nullptr;          // out [8][vn] fp32, vn = this rank's slice width (at most 25856 at TP6, 38784 at TP4)
    const __half* eh_in = nullptr;    // in  [8][12288] eh_proj input row (the two normed halves, concatenated by the caller)
    __half* eh = nullptr;             // out [8][6144]
};

struct Head
{
    int M = 0, vlo = 0, vn = 0;       // first global vocab id and width of this rank's slice
    Xfrag xf_h, xf_eh;
    Gv lm_head, eh;                   // lm_head: y = logits [8][vn] fp32, ld = vn.  eh.on == 0 without L078.eh
};

// A second lm_head input (MTP draft logits) is a second Head from make_head with its own HeadIo::h / logits; both may
// use the same scratch (one fragment table per input, written right before its GEMV on the same stream).
inline const char* make_head(const HeadTensors& T, int rank, int M, const Scratch& sc, const HeadIo& io, Head& H)
{
    H = Head();
    if (M < 1 || M > G6_MX) return "M must be 1..G6_MX (8, or 32 under DEC_S2)";
    if (!vocab_slice(rank, H.vlo, H.vn)) return "rank must be 0..NR-1";
    H.M = M;
    const char* e = nullptr;
    // best split on the GB10 (engine/results/item2-gv.jsonl): lm_head_rank tpc 1 ks 32, mtp_eh tpc 2 ks 16
    if ((e = make_gv(T.lm_head, MXFP8, 1, 32, M, sc.xf_head, io.logits, gv::O_F32, H.lm_head))) return e;
    if ((e = make_gv(T.eh,      MXFP8, 2, 16, M, sc.xf_eh,   io.eh,     gv::O_F16, H.eh))) return e;
    if (!H.lm_head.on) return "no G.lm_head.mx.gv";
    if (H.lm_head.K != HID || H.lm_head.O != H.vn) return "G.lm_head.mx.gv is not this rank's [vn/16][192][528] vocab slice";
    if (H.eh.on && (H.eh.K != 2 * HID || H.eh.O != HID)) return "L078.eh.mx.gv is not [384][384][528]";
    H.xf_h.on = 1;  H.xf_h.src = io.h;      H.xf_h.ld = HID;      H.xf_h.K = HID;      H.xf_h.f16 = sc.xf_head;  H.xf_h.M = M;
    H.xf_eh.on = H.eh.on;
    H.xf_eh.src = io.eh_in; H.xf_eh.ld = 2 * HID; H.xf_eh.K = 2 * HID; H.xf_eh.f16 = sc.xf_eh; H.xf_eh.M = M;
    return nullptr;
}

// the four slice members of glue::HeadArgs (or any struct with the same members)
template <class A> inline void head_args(const Head& H, A& a)
{
    a.logits = static_cast<const float*>(H.lm_head.y);
    a.ldl = H.lm_head.ld; a.vlo = H.vlo; a.vn = H.vn;
}

// once per process, before the first launch or graph capture
inline cudaError_t init()
{
    // fc2 too: above 8 rows the shared fc2's s_ntp B tables exceed 48 KB
    const void* fn[18] = {k1_fn_fc1<false, false, false>(), k1_fn_fused<false, false, false>(), k1_fn_fc2<false, false, false>(),
                          k1_fn_fc1<true, false, false>(), k1_fn_fused<true, false, false>(), k1_fn_fc2<true, false, false>(),
                          k1_fn_fc1<false, true, false>(), k1_fn_fused<false, true, false>(), k1_fn_fc2<false, true, false>(),
                          k1_fn_fc1<true, true, false>(), k1_fn_fused<true, true, false>(), k1_fn_fc2<true, true, false>(),
                          k1_fn_fc1<true, false, true>(), k1_fn_fused<true, false, true>(), k1_fn_fc2<true, false, true>(),
                          k1_fn_fc1<true, true, true>(), k1_fn_fused<true, true, true>(), k1_fn_fc2<true, true, true>()};
    for (const void* f : fn)
    {
        const cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, KSH * 256);
        if (e != cudaSuccess) return e;
    }
    return dtr::init();
}

}   // namespace bind
