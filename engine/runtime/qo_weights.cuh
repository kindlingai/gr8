// G6_PF_OWN=3 full-head prefill weights (QO-INTEGRATION.md section 4): q_b and o_proj MXFP8 for all 64 heads, UK / UV
// BF16 for all 64 heads, logical head order. Identical on every rank.
//   qo_from_pack:  the names L%03d.qo.{q_b.mx.codes, q_b.mx.scales, o_proj.mx.codes, o_proj.mx.scales, kv_b.uk, kv_b.uv}
//   qo_from_ranks: assembled on the device from the NR ranks' head slices (m4sim holds all the packs). Rank r holds
//                  head group g = (r - l % NR) mod NR of layer l, heads start(64, g) .. + width(64, g) (g6nr.h,
//                  attn/bind.cu; TP6 edges 0, 11, 22, 33, 44, 54, 64).
//   G6_PF_OWN=4 (TP4 builds only, below): the pair-owner weights of docs/DESIGN-QO-PAIR.md, 32 heads per head pair.
#pragma once
#include "rank.cuh"
#include "../attn/bind.h"
#include "../attn/prefill-qo-bind.h"
#include "qp_wire.h"
#include <functional>

namespace g6rt
{

struct QoW
{
    bind::Tensor qb_c, qb_s, o_c, o_s;   // [16384][2048], [16384][64], [6144][16384], [6144][512]
    G6AttnTensor uk{}, uv{};             // [64][192][512], [64][256][512] BF16
};
inline G6AttnTensor qo_attn_tensor(const TensorRef& r)
{
    G6AttnTensor t{};
    t.data = r.p; t.ndim = r.nd;
    for (int i = 0; i < 3 && i < r.nd; ++i) t.shape[i] = r.shape[i];
    return t;
}
inline bool qo_check(const QoW& w)
{
    auto is = [](const bind::Tensor& t, long long a, long long b) { return t.p && t.nd == 2 && t.shape[0] == a && t.shape[1] == b; };
    auto is3 = [](const G6AttnTensor& t, long long a, long long b, long long c) { return t.data && t.ndim == 3 && t.shape[0] == a && t.shape[1] == b && t.shape[2] == c; };
    return is(w.qb_c, 16384, 2048) && is(w.qb_s, 16384, 64) && is(w.o_c, HID, 16384) && is(w.o_s, HID, 512) &&
           is3(w.uk, 64, 192, 512) && is3(w.uv, 64, 256, 512);
}
inline bool qo_from_pack(const Find& find, int l, QoW& w)
{
    char n[80];
    auto get = [&](const char* s) { snprintf(n, sizeof n, "L%03d.qo.%s", l, s); return find(n); };
    w.qb_c = as_bind(get("q_b.mx.codes")); w.qb_s = as_bind(get("q_b.mx.scales"));
    w.o_c = as_bind(get("o_proj.mx.codes")); w.o_s = as_bind(get("o_proj.mx.scales"));
    w.uk = qo_attn_tensor(get("kv_b.uk")); w.uv = qo_attn_tensor(get("kv_b.uv"));
    return qo_check(w);
}
// finds[r] = rank r's tensors. Device allocations are never freed (load-time, like the packs).
inline QoW qo_from_ranks(const std::vector<Find>& finds, int l)
{
    if (finds.size() != (size_t)g6nr::NR) die("qo_from_ranks needs every rank's pack");
    char tn[32];
    snprintf(tn, sizeof tn, "L%03d.q_b.trellis", l);
    if (finds[0](tn).p) die("G6_PF_OWN=3 with dense trellis q_b / o_proj needs --qo-pack (the full-head weights are MXFP8)");
    QoW w;
    auto alloc2 = [](bind::Tensor& t, long long a, long long b, size_t esz) {
        t.p = dmalloc<char>((size_t)a * b * esz); t.nd = 2; t.shape[0] = a; t.shape[1] = b;
    };
    auto alloc3 = [](G6AttnTensor& t, long long a, long long b, long long c) {
        t.data = dmalloc<char>((size_t)a * b * c * 2); t.ndim = 3; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    };
    alloc2(w.qb_c, 16384, 2048, 1); alloc2(w.qb_s, 16384, 64, 1);
    alloc2(w.o_c, HID, 16384, 1); alloc2(w.o_s, HID, 512, 1);
    alloc3(w.uk, 64, 192, 512); alloc3(w.uv, 64, 256, 512);
    char n[80];
    for (int g = 0; g < g6nr::NR; ++g)
    {
        const int r = (g + l % g6nr::NR) % g6nr::NR, h0 = g6nr::start(g6nr::HEADS, g), h = g6nr::width(g6nr::HEADS, g);
        auto get = [&](const char* s) {
            snprintf(n, sizeof n, "L%03d.%s", l, s);
            TensorRef t = finds[r](n);
            if (!t.p) die(std::string("qo_from_ranks: rank ") + std::to_string(r) + " has no " + n);
            return t;
        };
        const TensorRef qc = get("q_b.mx.codes"), qs = get("q_b.mx.scales"), oc = get("o_proj.mx.codes"), os = get("o_proj.mx.scales");
        const TensorRef uk = get("kv_b.uk"), uv = get("kv_b.uv");
        if (qc.shape[0] != h * 256 || qc.shape[1] != 2048 || qs.shape[1] != 64 || oc.shape[0] != HID || oc.shape[1] != h * 256 ||
            os.shape[1] != h * 8 || uk.shape[0] != h || uv.shape[0] != h)
            die("qo_from_ranks: layer " + std::to_string(l) + " rank " + std::to_string(r) + " slice shapes are not head group " + std::to_string(g));
        char* qbc = (char*)w.qb_c.p; char* qbs = (char*)w.qb_s.p; char* oc_ = (char*)w.o_c.p; char* os_ = (char*)w.o_s.p;
        G6RT_CK(cudaMemcpy(qbc + (size_t)h0 * 256 * 2048, qc.p, (size_t)h * 256 * 2048, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy(qbs + (size_t)h0 * 256 * 64, qs.p, (size_t)h * 256 * 64, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy2D(oc_ + (size_t)h0 * 256, 16384, oc.p, (size_t)h * 256, (size_t)h * 256, HID, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy2D(os_ + (size_t)h0 * 8, 512, os.p, (size_t)h * 8, (size_t)h * 8, HID, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy((char*)w.uk.data + (size_t)h0 * 192 * 512 * 2, uk.p, (size_t)h * 192 * 512 * 2, cudaMemcpyDeviceToDevice));
        G6RT_CK(cudaMemcpy((char*)w.uv.data + (size_t)h0 * 256 * 512 * 2, uv.p, (size_t)h * 256 * 512 * 2, cudaMemcpyDeviceToDevice));
    }
    if (!qo_check(w)) die("qo_from_ranks: assembled shapes wrong");
    return w;
}

// ---- G6_PF_OWN=4 pair-owner weights (docs/DESIGN-QO-PAIR.md section 1, TP4 only). Rank r's head pair {r & ~1, r | 1}
// holds 32 heads of layer l, the two groups g(m) = (m - l) mod 4 in logical order (lower group first): q_b [8192][2048]
// (+ [8192][64]), o_proj [6144][8192] (+ [6144][256]), UK / UV [32][192][512] / [32][256][512] BF16. Byte-identical on
// both pair members.
//   qp_from_pack:  L%03d.qp.{q_b.mx.codes, q_b.mx.scales, o_proj.mx.codes, o_proj.mx.scales} of tools/convert/pairpack.py's
//                  qp.rank<r>.pack, UK / UV assembled on the device from the rank's own L%03d.kv_b.{uk,uv} and the pack's
//                  L%03d.qp.kv_b.{uk,uv} (the partner's 16 heads)
//   qp_from_ranks: everything assembled on the device from the two pair members' head slices (m4sim holds all packs)
//   QpRoll:        G6_PF_QP_ROLL=S, S slots that roll through the layers during a prefill (g6m4: own half from the rank
//                  pack, partner half over RDMA, qp_wire.h; m4sim: both halves from the packs)
// The BF16 UK / UV are only read by g6_attn_qo_create / g6_attn_qo_load (they keep FP16 copies): qp_free_kvb releases
// the resident ones after create.
struct QpW;
struct QpRoll;
#if G6_NR == 4
struct QpW
{
    bind::Tensor qb_c, qb_s, o_c, o_s;
    G6AttnTensor uk{}, uv{};
    // dense trellis q_b / o_proj (kernels/dtr.cuh): qb_c / o_c hold the FP8 tile copies, and these the pair's vectors in
    // the same half order: q_b svh [8192], o_proj suh [8192] f16. nullptr: MXFP8 weights.
    __half *qb_svh = nullptr, *o_suh = nullptr;
};
constexpr int QP_HEADS = 32;
inline int qp_group(int m, int l) { return ((m - l) % 4 + 4) % 4; }
// the pair's members in logical head order: {lower group's member, upper group's member}
inline std::pair<int, int> qp_order(int r, int l)
{
    const int a = r & ~1, b = r | 1;
    return qp_group(a, l) < qp_group(b, l) ? std::make_pair(a, b) : std::make_pair(b, a);
}
inline bool qp_check(const QpW& w)
{
    auto is = [](const bind::Tensor& t, long long a, long long b) { return t.p && t.nd == 2 && t.shape[0] == a && t.shape[1] == b; };
    auto is3 = [](const G6AttnTensor& t, long long a, long long b, long long c) { return t.data && t.ndim == 3 && t.shape[0] == a && t.shape[1] == b && t.shape[2] == c; };
    return is(w.qb_c, 8192, 2048) && is(w.qb_s, 8192, 64) && is(w.o_c, HID, 8192) && is(w.o_s, HID, 256) &&
           is3(w.uk, QP_HEADS, 192, 512) && is3(w.uv, QP_HEADS, 256, 512);
}
inline void qp_alloc2(bind::Tensor& t, long long a, long long b)
{
    t.p = dmalloc<char>((size_t)a * b); t.nd = 2; t.shape[0] = a; t.shape[1] = b;
}
inline void qp_alloc_kvb(QpW& w)
{
    auto alloc3 = [](G6AttnTensor& t, long long a, long long b, long long c) {
        t.data = dmalloc<char>((size_t)a * b * c * 2); t.ndim = 3; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    };
    alloc3(w.uk, QP_HEADS, 192, 512); alloc3(w.uv, QP_HEADS, 256, 512);
}
inline void qp_alloc_vec(QpW& w)
{
    if (!w.qb_svh) { w.qb_svh = dmalloc<__half>(8192); w.o_suh = dmalloc<__half>(8192); }
}
// half k (0 = lower group, 1 = upper) of the pair's UK / UV <- a 16-head slice [16][d][512] BF16
inline void qp_put_kvb(QpW& w, int k, const TensorRef& uk, const TensorRef& uv, const std::string& what)
{
    if (uk.nd != 3 || uk.shape[0] != 16 || uk.shape[1] != 192 || uk.shape[2] != 512 || uv.nd != 3 || uv.shape[0] != 16 ||
        uv.shape[1] != 256 || uv.shape[2] != 512)
        die("qp: " + what + " kv_b.uk / kv_b.uv are not [16][192|256][512]");
    G6RT_CK(cudaMemcpy((char*)w.uk.data + (size_t)k * 16 * 192 * 512 * 2, uk.p, (size_t)16 * 192 * 512 * 2, cudaMemcpyDeviceToDevice));
    G6RT_CK(cudaMemcpy((char*)w.uv.data + (size_t)k * 16 * 256 * 512 * 2, uv.p, (size_t)16 * 256 * 512 * 2, cudaMemcpyDeviceToDevice));
}
inline void qp_free_kvb(QpW& w)
{
    G6RT_CK(cudaFree(const_cast<void*>(w.uk.data))); G6RT_CK(cudaFree(const_cast<void*>(w.uv.data)));
    w.uk.data = w.uv.data = nullptr;
}
// qfind: rank r's qp pack; find: rank r's own pack (its kv_b)
inline bool qp_from_pack(const Find& qfind, const Find& find, int r, int l, QpW& w)
{
    char n[80];
    auto get = [&](const Find& f, const char* s) { snprintf(n, sizeof n, "L%03d.%s", l, s); return f(n); };
    w.qb_c = as_bind(get(qfind, "qp.q_b.mx.codes")); w.qb_s = as_bind(get(qfind, "qp.q_b.mx.scales"));
    w.o_c = as_bind(get(qfind, "qp.o_proj.mx.codes")); w.o_s = as_bind(get(qfind, "qp.o_proj.mx.scales"));
    if (get(find, "q_b.trellis").p) die("--qp-pack (resident pair pack) has no dense trellis form: use G6_PF_QP_ROLL");
    const TensorRef puk = get(qfind, "qp.kv_b.uk"), puv = get(qfind, "qp.kv_b.uv"), ouk = get(find, "kv_b.uk"), ouv = get(find, "kv_b.uv");
    if (!puk.p || !puv.p || !ouk.p || !ouv.p) return false;
    qp_alloc_kvb(w);
    const bool own_lo = qp_order(r, l).first == r;
    qp_put_kvb(w, own_lo ? 0 : 1, ouk, ouv, "layer " + std::to_string(l) + " own");
    qp_put_kvb(w, own_lo ? 1 : 0, puk, puv, "layer " + std::to_string(l) + " partner (qp pack)");
    return qp_check(w);
}
// the 16-head slice of pair member m: q_b / o_proj codes and scales, UK / UV (shape-checked); qv / ov: a dense trellis
// q_b's svh slice and o_proj's suh slice [4096] (p = nullptr: MXFP8 weights)
struct QpHalf { TensorRef qc, qs, oc, os, uk, uv, qv, ov; };
inline QpHalf qp_half(const Find& f, int m, int l, const char* who)
{
    char n[80];
    auto get = [&](const char* s) {
        snprintf(n, sizeof n, "L%03d.%s", l, s);
        TensorRef t = f(n);
        if (!t.p) die(std::string(who) + ": rank " + std::to_string(m) + " has no " + n);
        return t;
    };
    QpHalf h{get("q_b.mx.codes"), get("q_b.mx.scales"), get("o_proj.mx.codes"), get("o_proj.mx.scales"), get("kv_b.uk"), get("kv_b.uv"), {}, {}};
    snprintf(n, sizeof n, "L%03d.q_b.trellis", l);
    const bool tq = f(n).p != nullptr;
    snprintf(n, sizeof n, "L%03d.o_proj.trellis", l);
    if (tq != (f(n).p != nullptr)) die(std::string(who) + ": layer " + std::to_string(l) + " has one of q_b / o_proj in trellis form, not both");
    if (tq)
    {
        h.qv = get("q_b.svh");
        h.ov = get("o_proj.suh");
        if (h.qv.nd != 1 || h.qv.shape[0] != 4096 || h.ov.nd != 1 || h.ov.shape[0] != 4096)
            die(std::string(who) + ": layer " + std::to_string(l) + " rank " + std::to_string(m) + " trellis q_b svh / o_proj suh are not [4096]");
    }
    if (h.qc.shape[0] != 16 * 256 || h.qc.shape[1] != 2048 || h.qs.shape[1] != 64 || h.oc.shape[0] != HID || h.oc.shape[1] != 16 * 256 ||
        h.os.shape[1] != 16 * 8 || h.uk.nd != 3 || h.uk.shape[0] != 16 || h.uk.shape[1] != 192 || h.uv.nd != 3 ||
        h.uv.shape[0] != 16 || h.uv.shape[1] != 256)
        die(std::string(who) + ": layer " + std::to_string(l) + " rank " + std::to_string(m) + " slice shapes are not 16 heads");
    return h;
}
// slot half k <- the six tensors at qc .. uv (device or pinned), on stream b. The same bytes and layout as pairpack.py.
// qv / ov: the dense trellis vectors of the half (nullptr: MXFP8 weights). A rolling slot holds layers of both forms, and
// the reader picks the vectors by the layer's form (qp.cuh).
inline void qp_put_half(QpW& w, int k, const void* qc, const void* qs, const void* oc, const void* os, const void* uk, const void* uv,
                        cudaStream_t b, const void* qv = nullptr, const void* ov = nullptr)
{
    if (qv && !w.qb_svh) die("qp: a dense trellis half for a slot without vector buffers");
    if (qv)
    {
        G6RT_CK(cudaMemcpyAsync(w.qb_svh + (size_t)k * 4096, qv, (size_t)4096 * 2, cudaMemcpyDefault, b));
        G6RT_CK(cudaMemcpyAsync(w.o_suh + (size_t)k * 4096, ov, (size_t)4096 * 2, cudaMemcpyDefault, b));
    }
    G6RT_CK(cudaMemcpyAsync((char*)w.qb_c.p + (size_t)k * 4096 * 2048, qc, (size_t)4096 * 2048, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync((char*)w.qb_s.p + (size_t)k * 4096 * 64, qs, (size_t)4096 * 64, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpy2DAsync((char*)w.o_c.p + (size_t)k * 4096, 8192, oc, 4096, 4096, HID, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpy2DAsync((char*)w.o_s.p + (size_t)k * 128, 256, os, 128, 128, HID, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync((char*)w.uk.data + (size_t)k * 16 * 192 * 512 * 2, uk, (size_t)16 * 192 * 512 * 2, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync((char*)w.uv.data + (size_t)k * 16 * 256 * 512 * 2, uv, (size_t)16 * 256 * 512 * 2, cudaMemcpyDefault, b));
}
inline void qp_alloc(QpW& w)
{
    qp_alloc2(w.qb_c, 8192, 2048); qp_alloc2(w.qb_s, 8192, 64); qp_alloc2(w.o_c, HID, 8192); qp_alloc2(w.o_s, HID, 256);
    qp_alloc_kvb(w);
}
// finds[m] = rank m's tensors; rank r's pair weights of layer l. Device allocations of q_b / o_proj are never freed.
inline QpW qp_from_ranks(const std::vector<Find>& finds, int r, int l)
{
    if (finds.size() != 4) die("qp_from_ranks: pair-owner is TP4 only (all four packs)");
    QpW w;
    qp_alloc(w);
    const std::pair<int, int> ord = qp_order(r, l);
    for (int k = 0; k < 2; ++k)
    {
        const int m = k ? ord.second : ord.first;
        const QpHalf h = qp_half(finds[m], m, l, "qp_from_ranks");
        if (h.qv.p) qp_alloc_vec(w);
        qp_put_half(w, k, h.qc.p, h.qs.p, h.oc.p, h.os.p, h.uk.p, h.uv.p, nullptr, h.qv.p, h.ov.p);
    }
    G6RT_CK(cudaDeviceSynchronize());
    if (!qp_check(w)) die("qp_from_ranks: assembled shapes wrong");
    return w;
}

// ---- rolling pair pack (G6_PF_QP_ROLL=S): S slots of pair weights in place of one resident QpW per layer. Slot s holds
// model layer position i with i % S == s: the same bytes qp_from_ranks assembles (q_b / o_proj halves in qp_order, BF16
// UK / UV staged in the slot, then reconverted into the slot's QO state by g6_attn_qo_load = create's kernels), so OWN=4
// output is byte-identical to the resident pack. A build runs on the stage stream ss (G6_PF_QP_ROLL_SER=1: on the lane
// stream that triggers it, fully exposed) after both lanes' `used` event of that slot; the lanes wait `ready`.
// Builds are triggered by the first lane to reach layer i (qp_roll_acquire): layer i if the slot does not hold it, then
// prefetch i + 1 .. i + S - 1. That reuses a slot whose layer is <= i - 1, which both lanes have already ENQUEUED (every
// driver enqueues lane B's attention block of layer i - 1 before lane A's of layer i), so `used` is recorded before the wait.
// Sources: m4sim: the pair members' own head slices in device memory (it holds all four packs). g6m4 (QpWire set): only
// the own half has device sources (this rank's pack); the partner half arrives by RDMA (qp_wire.h protocol) into the wire
// fabric's bulk ring and is copied from there into the slot.
struct QpSrc { QpHalf h[2]; bool have[2] = {false, false}; int partner_half = -1; int layer = -1; bool trellis = false; };
// the wire (g6m4): the bulk Fabric's kernel view, its job queue, and the build counter n (qp_wire.h)
struct QpWire
{
    g6::BulkDev bd{};
    std::function<void(g6::Fabric::BulkJob)> enqueue;   // g6::Fabric::enqueue_bulk of the wire fabric
    int E = 1, peer = -1;
    int ctx = 0;                                         // P.qp_ctx (QO table), set by build_prefill: part of the ready tag
    int S = 0;
    long n = 0;                                          // builds issued so far (equal on both pair members)
};
struct QpRoll
{
    int S = 0;
    bool ser = false;
    std::vector<QpSrc> src;                  // per model layer position (unset outside the mask)
    std::vector<QpW> slot;                   // q_b / o_proj MXFP8 + BF16 UK / UV staging, device, owned
    std::vector<G6AttnQo*> qa;               // per slot: QO state (FP16 UK / UV), created by build_prefill
    std::vector<int> have;                   // per slot: layer position loaded or in flight, -1 none
    cudaStream_t ss = nullptr;
    std::vector<cudaEvent_t> ready, used[2];
    static constexpr int NT = 512;           // timing ring: event pairs per build, summed lazily (sync only on wrap)
    std::vector<cudaEvent_t> t0, t1;
    long builds = 0, timed = 0;
    double ms = 0, bytes_own = 0, bytes_partner = 0;
    QpWire* wire = nullptr;                  // g6m4: the partner half by RDMA (nullptr = m4sim device copies)
};
inline void qp_roll_ev(cudaEvent_t& e, bool timing)
{
    G6RT_CK(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming));
}
// rank r's roll over `layers` (positions with qpl[k]): S slots; qpw[k] = slot k % S (pointer copies: the binding is
// static, the slot content rolls). wire (g6m4): finds = {this rank's own pack}; else finds = all four packs (m4sim).
inline QpRoll* qp_roll_new(int S, bool ser, const std::vector<Find>& finds, int r, const std::vector<int>& layers,
                           const std::vector<char>& qpl, std::vector<QpW>& qpw, QpWire* wire = nullptr)
{
    if (finds.size() != (wire ? 1u : 4u)) die("G6_PF_QP_ROLL: pair-owner is TP4 only (all four packs, or the own pack + the wire)");
    if (S < 2 || S > 4) die("G6_PF_QP_ROLL: 2..4 slots");
    if (wire && (wire->E < 1 || wire->E > qw::max_entries() || wire->peer != qw::partner(r) || !wire->enqueue || !wire->bd.bulk))
        die("G6_PF_QP_ROLL: wire not set up (E 1.." + std::to_string(qw::max_entries()) + ", peer r ^ 1, bulk fabric)");
    auto* R = new QpRoll;
    R->S = S; R->ser = ser; R->wire = wire;
    if (wire) wire->S = S;
    R->src.resize(layers.size());
    for (size_t k = 0; k < layers.size(); ++k)
    {
        if (!qpl[k]) continue;
        const int l = layers[k];
        const std::pair<int, int> ord = qp_order(r, l);
        QpSrc& q = R->src[k];
        q.layer = l;
        for (int h = 0; h < 2; ++h)
        {
            const int m = h ? ord.second : ord.first;
            if (m != r) q.partner_half = h;
            if (wire && m != r) continue;            // g6m4: the partner's half has no local source
            q.h[h] = qp_half(finds[wire ? 0 : m], m, l, "G6_PF_QP_ROLL");
            q.have[h] = true;
            q.trellis = q.h[h].qv.p != nullptr;      // the wire tag makes the partner agree (qp_wire_build)
        }
        if (q.partner_half < 0) die("G6_PF_QP_ROLL: rank " + std::to_string(r) + " is not in its own pair");
    }
    R->slot.resize(S);
    bool trellis = false;
    for (const QpSrc& q : R->src) trellis |= q.trellis;
    for (QpW& w : R->slot)
    {
        qp_alloc(w);
        if (trellis) qp_alloc_vec(w);
        G6RT_CK(cudaMemset(const_cast<void*>(w.uk.data), 0, (size_t)QP_HEADS * 192 * 512 * 2));   // defined bytes for create
        G6RT_CK(cudaMemset(const_cast<void*>(w.uv.data), 0, (size_t)QP_HEADS * 256 * 512 * 2));
    }
    for (size_t k = 0; k < layers.size(); ++k) if (qpl[k]) qpw[k] = R->slot[k % S];
    R->have.assign(S, -1);
    G6RT_CK(cudaStreamCreateWithFlags(&R->ss, cudaStreamNonBlocking));
    R->ready.resize(S); R->used[0].resize(S); R->used[1].resize(S);
    for (int s = 0; s < S; ++s) { qp_roll_ev(R->ready[s], false); qp_roll_ev(R->used[0][s], false); qp_roll_ev(R->used[1][s], false); }
    R->t0.resize(QpRoll::NT); R->t1.resize(QpRoll::NT);
    for (int k = 0; k < QpRoll::NT; ++k) { qp_roll_ev(R->t0[k], true); qp_roll_ev(R->t1[k], true); }
    return R;
}
inline void qp_roll_sum(QpRoll& R, long upto)       // add the elapsed ms of timed builds [timed, upto) (host sync on each)
{
    for (; R.timed < upto; ++R.timed)
    {
        const int k = (int)(R.timed % QpRoll::NT);
        float t = 0;
        G6RT_CK(cudaEventSynchronize(R.t1[k]));
        G6RT_CK(cudaEventElapsedTime(&t, R.t0[k], R.t1[k]));
        R.ms += t;
    }
}
// grid 1, block 32: GPU -> proxy doorbell after the stream's earlier copies into the pinned bulk area.
// aux (non-null): *aux = av first (the proxy reads aux_out[post_flag] after the doorbell: the ready tag)
__global__ void __launch_bounds__(32) k_qp_bell(unsigned long long* db, unsigned long long v, unsigned long long* aux, unsigned long long av)
{
    if (threadIdx.x == 0)
    {
        if (aux) *reinterpret_cast<volatile unsigned long long*>(aux) = av;
        __threadfence_system();
        glue::st_release_sys(reinterpret_cast<u64*>(db), v);
    }
}
// grid 1, block 32: wait flags[f][peer] >= want (one peer); timeout or an error already set -> error word, return.
// tag != 0: then the flag's aux (written before val, bulk_proxy.h flag_writes) must equal tag, else tcode
__global__ void __launch_bounds__(32) k_qp_wait(g6::BulkDev bd, int f, int peer, unsigned long long want, unsigned code,
                                                unsigned long long tag, unsigned tcode)
{
    if (threadIdx.x != 0 || want == 0) return;
    const u64* fl = reinterpret_cast<const u64*>(bd.flags + ((size_t)f * bd.nr + peer) * 2);
    const long long t0 = clock64();
    unsigned it = 0;
    while (glue::ld_acquire_sys(fl) < want)
        if ((++it & 1023u) == 0 && (*reinterpret_cast<volatile unsigned*>(bd.error) != 0 || clock64() - t0 > G6_SPIN_TIMEOUT_CYC))
        {
            atomicCAS(bd.error, 0u, code);
            return;
        }
    if (tag && glue::ld_acquire_sys(fl + 1) != tag) atomicCAS(bd.error, 0u, tcode);
}
// one build n on stream b: (1) wait p's consumed of entry e (build n - E); (2) my half (pack) -> my position of entry e,
// doorbell ready e = n + 1, proxy jobs TX(n) and CN(n); (3) my half -> slot (pack); (4) wait p's ready e >= n + 1;
// (5) partner position -> slot (2D copies for o_proj); (6) doorbell consumed e = n + 1. The caller then runs qo_load.
inline void qp_wire_build(QpRoll& R, QpWire& x, const QpSrc& q, QpW& w, cudaStream_t b)
{
    const long n = x.n++;
    const int e = qw::entry(n, x.E), ph = q.partner_half, oh = 1 - ph;
    const unsigned pe = (unsigned)(e << 4) | (unsigned)x.peer;
    const unsigned long long tg = qw::tag(q.layer, x.S, x.E, x.ctx, q.trellis);
    const QpHalf& o = q.h[oh];
    k_qp_wait<<<1, 32, 0, b>>>(x.bd, qw::f_consumed(e), x.peer, qw::consumed_min(n, x.E), qw::ERR_WAIT_CONSUMED | pe, 0ull, 0u);
    G6RT_CK(cudaGetLastError());
    char* tx = x.bd.bulk + qw::pos_off(e, oh);
    G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_QC, o.qc.p, qw::QC, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_QS, o.qs.p, qw::QS, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_OC, o.oc.p, qw::OC, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_OS, o.os.p, qw::OS, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_UK, o.uk.p, qw::UK, cudaMemcpyDefault, b));
    G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_UV, o.uv.p, qw::UV, cudaMemcpyDefault, b));
    if (q.trellis)
    {
        G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_QV, o.qv.p, qw::QV, cudaMemcpyDefault, b));
        G6RT_CK(cudaMemcpyAsync(tx + qw::OFF_OV, o.ov.p, qw::OV, cudaMemcpyDefault, b));
    }
    k_qp_bell<<<1, 32, 0, b>>>(x.bd.doorbell + qw::f_ready(e), qw::val(n), x.bd.aux_out + qw::f_ready(e), tg);
    G6RT_CK(cudaGetLastError());
    {
        g6::Fabric::BulkJob j;                 // TX(n)
        j.db = qw::f_ready(e); j.seq = qw::val(n);
        j.wait_flag = qw::f_consumed(e); j.wait_min = qw::consumed_min(n, x.E);
        j.ranges.push_back({(uint64_t)qw::pos_off(e, oh), (uint32_t)qw::HALF});
        j.post_flag = qw::f_ready(e); j.post_val = qw::val(n); j.peer = x.peer;
        x.enqueue(std::move(j));
        g6::Fabric::BulkJob c;                 // CN(n)
        c.db = qw::f_consumed(e); c.seq = qw::val(n);
        c.post_flag = qw::f_consumed(e); c.post_val = qw::val(n); c.peer = x.peer;
        x.enqueue(std::move(c));
    }
    qp_put_half(w, oh, o.qc.p, o.qs.p, o.oc.p, o.os.p, o.uk.p, o.uv.p, b, o.qv.p, o.ov.p);
    R.bytes_own += (double)qw::HALF;
    k_qp_wait<<<1, 32, 0, b>>>(x.bd, qw::f_ready(e), x.peer, qw::val(n), qw::ERR_WAIT_READY | pe, tg, qw::ERR_TAG | pe);
    G6RT_CK(cudaGetLastError());
    const char* rx = x.bd.bulk + qw::pos_off(e, ph);
    qp_put_half(w, ph, rx + qw::OFF_QC, rx + qw::OFF_QS, rx + qw::OFF_OC, rx + qw::OFF_OS, rx + qw::OFF_UK, rx + qw::OFF_UV, b,
                q.trellis ? rx + qw::OFF_QV : nullptr, q.trellis ? rx + qw::OFF_OV : nullptr);
    R.bytes_partner += (double)qw::HALF;
    k_qp_bell<<<1, 32, 0, b>>>(x.bd.doorbell + qw::f_consumed(e), qw::val(n), nullptr, 0ull);
    G6RT_CK(cudaGetLastError());
}
// layer position i -> slot i % S on the stage stream (or `st` when ser): waits both lanes' last use of the slot
inline void qp_roll_build(QpRoll& R, int i, cudaStream_t st)
{
    const int s = i % R.S;
    cudaStream_t b = R.ser ? st : R.ss;
    const QpSrc& q = R.src[i];
    QpW& w = R.slot[s];
    G6RT_CK(cudaStreamWaitEvent(b, R.used[0][s], 0));
    G6RT_CK(cudaStreamWaitEvent(b, R.used[1][s], 0));
    if (R.builds - R.timed >= QpRoll::NT) qp_roll_sum(R, R.builds - QpRoll::NT + 1);
    const int k = (int)(R.builds % QpRoll::NT);
    G6RT_CK(cudaEventRecord(R.t0[k], b));
    if (R.wire) qp_wire_build(R, *R.wire, q, w, b);   // own half from the pack, partner half over the wire fabric
    else
        for (int h = 0; h < 2; ++h)                    // m4sim: qp_from_ranks' copies, async on b
        {
            const QpHalf& x = q.h[h];
            qp_put_half(w, h, x.qc.p, x.qs.p, x.oc.p, x.os.p, x.uk.p, x.uv.p, b, x.qv.p, x.ov.p);
            (h == q.partner_half ? R.bytes_partner : R.bytes_own) += (double)qw::HALF;
        }
    if (int e = g6_attn_qo_load(R.qa[s], &w.uk, &w.uv, b)) die("G6_PF_QP_ROLL: g6_attn_qo_load failed (" + std::to_string(e) + ")");
    G6RT_CK(cudaEventRecord(R.t1[k], b));
    G6RT_CK(cudaEventRecord(R.ready[s], b));
    R.have[s] = i;
    ++R.builds;
}
// lane at layer position i, before anything reads the pair weights: slot holds i (build if not), prefetch, wait ready
inline void qp_roll_acquire(QpRoll& R, const std::vector<char>& qpl, int nl, int i, cudaStream_t st)
{
    const int s = i % R.S;
    if (R.have[s] != i) qp_roll_build(R, i, st);
    for (int d = 1; d < R.S; ++d)
    {
        const int j = i + d;
        if (j < nl && qpl[j] && R.have[j % R.S] != j) qp_roll_build(R, j, st);
    }
    G6RT_CK(cudaStreamWaitEvent(st, R.ready[s], 0));
}
// lane `lane` (0 = A, 1 = B) is done reading layer position i's slot (after its o_proj)
inline void qp_roll_release(QpRoll& R, int lane, int i, cudaStream_t st)
{
    G6RT_CK(cudaEventRecord(R.used[lane & 1][i % R.S], st));
}
inline void qp_roll_print(QpRoll& R, int rank)
{
    G6RT_CK(cudaStreamSynchronize(R.ss));
    qp_roll_sum(R, R.builds);
    const double nb = R.builds ? (double)R.builds : 1.0;
    fprintf(stderr, "# qproll rank %d: S %d %s, builds %ld, build ms %.3f (%.3f per build), per build own %.2f MB partner %.2f MB\n",
            rank, R.S, R.ser ? "serial (lane stream)" : "stage stream", R.builds, R.ms, R.ms / nb, R.bytes_own / nb / 1e6,
            R.bytes_partner / nb / 1e6);
}
#endif   // G6_NR == 4

}   // namespace g6rt
