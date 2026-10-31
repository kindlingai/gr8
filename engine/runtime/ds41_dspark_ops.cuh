// DSpark op-table adapters for the DS41 decode step: ST_DS_TAP, ST_DS_MAIN (main Model), ST_DS_DRAFT_IN, ST_DS_HEAD_LOCAL, ST_DS_MARKOV
// (draft Model) and the block-mode attention for the draft layers 40..42, plus build() which fills Model::dsp. Included inside namespace g6rt
// by ds41_step.cuh (the '// DSpark' line of ds41_table(): `ds41dsp_ops::fill(t)`); the kernels are ds41_dspark.cuh at global scope (rank.cuh,
// next to ds_mhc.cuh). docs/DEEPSEEK-DSPARK.md describes the design. Namespace is ds41dsp_ops, not ds41dsp: a g6rt::ds41dsp would hide ::ds41dsp.
//
// Two Models run one round: the main Model (verify step, rows = nsess * (k + 1)) and the draft Model (ids 40..42, rows = nsess * 5).
// Both carry a Model::dsp filled by build(). Contract per stage (ds41_ops.h Fn = int(void* model, int i, int rows, void* stream), 0 = ok):
//   ST_DS_TAP        main  m.resid4 (FP32 [rows][4][5120]) -> dsp.tap[:, l - 37, :] BF16 mean, at the entry of main layers 37 / 38 / 39
//   ST_DS_MAIN       main  dsp.tap -> main_proj -> main_norm -> 3 x wkv -> kv_norm -> rope(m.pos) -> the 3 key rings, every verify row
//   ST_DS_DRAFT_IN   draft launch_step_in (the draft's StepIn: fabric step, zero SampleRows) then dsp.vtok/vtgt/vpos -> accept -> q, golden,
//                    acc and the block's m.tok / m.pos; embed on the draft rows
//   ST_ATTN_PROJ/CORE/OUT, draft layers (l >= 40): block-mode attention over the attention's per-layer ring (ds41_attn.cuh state); l < 40 -> the attention ops
//   ST_DS_HEAD_LOCAL draft lm_head GEMV over the shard for all rows into m.hlog (no argmax)
//   ST_DS_MARKOV     draft for i = 0..4: markov add -> local argmax -> head merge (collective 6 + i) -> dtok; then confidence + draft ids
// SESS > 1 (Model::ds_slots > 1): the key rings of the draft layers are one slab per S2 slot (the attention's per-slot ring); a row's slab is Model::sess[row]
// (k_kv_prep_pos / k_attn_blk dsess). The sessions of a step are Model::ds_ns (uniform rows per session); Dsp::nsess is the allocation (S2 slots).
//   ST_FINISH       main: the step driver's op; draft (dsp.draft_model): ds_mhc::finish with store = false and xpre = dsp.x (the confidence head
//                   reads dsp.x, so a missing xpre would read uninitialised memory with no error). op_finish does this.
// Wiring (4 lines in the step driver):
//   rank.cuh   inside `#if G6_MODEL_DS41`, global scope after ds_mhc.cuh:  #include "ds41_dspark.h"  and  #include "ds41_dspark.cuh"
//   rank.cuh   Model, in the DS41 field block:                                ds41dsp::Dsp dsp;
//   ds41_step.cuh  after the ds41_moe_ops.cuh include:                       #include "ds41_dspark_ops.cuh"
//   ds41_step.cuh  ds41_table(), at the '// DSpark' line, after the attention / MoE fills:  ds41dsp_ops::fill(t);   (it wraps ST_ATTN_*, ST_FINISH)
// The driver then calls ds41dsp_ops::build(main, draft, nsess, k) once after build_ds41 of both Models, and runs the verify plan
// (plan(..., dspark = true)) on the main Model and plan_draft({40,41,42}) on the draft Model (own fabric, ncoll 11, tickets sized 11).
#pragma once
#if G6_MODEL_DS41
// needs ds41_dspark.h (ds41::dsp::markov_collective) and ds41_dspark.cuh at global scope: rank.cuh includes both next to ds41_ops.h / ds_mhc.cuh

namespace ds41dsp_ops
{
using ::ds41dsp::Dsp;
namespace dd = ::ds41::dspd;
constexpr int BLOCK = dd::BLOCK;                       // 5 draft rows per session
static_assert(sizeof(glue::HeadRec) == ::ds41::dsp::HEAD_REC_BYTES && sizeof(glue::HeadCand) == ::ds41::dsp::HEAD_CAND_BYTES,
              "ds41::dsp::coll_bytes assumes the head merge record and candidate sizes of glue.cuh");
constexpr int HD = ds41a::HD;                          // 512

inline ::ds41dsp::Ring ring_of(const Ds41AttnLayer& a)
{
    ::ds41dsp::Ring r;
    r.base = a.C.ring;
    r.ring_rows = a.C.ring_rows;
    r.sess_stride = (size_t)a.C.ring_rows * HD;   // the attention ring: one slab of ring_rows x 512 per S2 slot (ds41_attn_bind)
    return r;
}
// the draft step's rings: SESS > 1 reads each row's slot from the draft Model's m.sess (its StepIn)
inline ::ds41dsp::Ring ring_of(const Model& m, const Ds41AttnLayer& a)
{
    ::ds41dsp::Ring r = ring_of(a);
    r.sess = m.ds_slots > 1 ? m.sess : nullptr;
    return r;
}

// ---- main Model
// Prefill (docs/DEEPSEEK-DSPARK.md section 9): a non-final chunk of W rows runs the same two stages. dsp.tap holds dsp.tap_rows rows (0 = MX: the
// decode step), dsp.main_lo / main_hi pick the rows [lo, hi) that are tapped and entered into the key rings (hi = -1: all). Row r of the tap is
// chunk row r, so m.pos[r] is its position; ds_main_rows runs at most `cap` rows per launch (cap = min(MX, RING): MainS holds MX rows and a launch
// of more than RING rows would put two positions of one ring slot in the same launch). A prefill-capable build (tap_rows > MX) carves MainS for
// PREFILL_MS_ROWS = 128 rows (about 5 MB), so cap = 128: 8 launches for a 1024-row chunk, one for the last-128 window of GR8_DS_PREFILL_KEEP = 128.
inline int tap_cap(const Dsp& D) { return D.tap_rows ? D.tap_rows : MX; }
inline int ms_cap(const Dsp& D) { return std::min(D.ms_rows ? D.ms_rows : MX, (int)::ds41::dsp::RING); }   // rows per ST_DS_MAIN launch: the scratch rows, and the ring
inline int op_tap(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    const int slot = m.layers[i].l - geom::DS_TARGET0;
    if (slot < 0 || slot >= dd::TAPS || !m.dsp.tap) return ds41::NOTSUP;
    if (rows < 1 || rows > tap_cap(m.dsp)) return 1;
    const int lo = m.dsp.main_lo, hi = m.dsp.main_hi < 0 ? rows : std::min(m.dsp.main_hi, rows);
    if (lo < 0 || lo > hi) return 1;
    if (lo == hi) return 0;                                  // the whole chunk is older than the kept window: nothing to tap
    if (lo == 0 && hi == rows) ::ds41dsp::ds_tap(m.resid4, m.dsp.tap, slot, rows, (cudaStream_t)stream);   // the decode call
    else ::ds41dsp::ds_tap(m.resid4 + (size_t)lo * 4 * HID, m.dsp.tap + (size_t)lo * dd::TAPS * HID, slot, hi - lo, (cudaStream_t)stream);
    return 0;
}
inline int op_main(void* model, int, int rows, void* stream)
{
    Model& m = *(Model*)model;
    const Dsp& D = m.dsp;
    if (!D.tap || !D.mw.main_proj.w || !D.ms.tq) return ds41::NOTSUP;
    // Every main step of a served request runs this stage, prompt chunks (1..chunk rows) included: the draft layers' key rings need every row.
    // One session in the step (m.ds_ns = 1): any 1..MX rows are that session's (a prefill chunk: 1..tap_rows). Several: uniform rows per session
    // (the verify shape k + 1). SESS > 1: every row's key goes to its slot's ring slab (m.sess); one S2 slot: dsess = nullptr (session = row / rows_per_sess).
    const int S = m.ds_ns;
    const int* dsess = m.ds_slots > 1 ? m.sess : nullptr;
    if (m.ds41_pack) {
        if (ds41::pack::dspark_error(*m.ds41_pack, rows, tap_cap(D), m.ds_slots, dsess != nullptr,
                                     D.main_lo, D.main_hi)) return 1;
        // The prefill launcher tiles at <=128 rows and reads each row's slot from dsess.
        // A segment is <=64 rows, so no tile can overwrite a slot's ring position twice.
        ::ds41dsp::ds_main_rows(D.mw, D.ms, D.tap, rows, m.pos, ms_cap(D), (cudaStream_t)stream, dsess);
        return 0;
    }
    if (rows < 1 || rows > tap_cap(D) || S < 1 || rows % S || (S > 1 && rows > MX)) return 1;
    if (S == 1 && (rows > MX || D.main_lo != 0 || D.main_hi >= 0))
    {
        const int lo = D.main_lo, hi = D.main_hi < 0 ? rows : std::min(D.main_hi, rows);
        if (lo < 0 || lo > hi) return 1;
        if (lo < hi) ::ds41dsp::ds_main_rows(D.mw, D.ms, D.tap + (size_t)lo * dd::TAPS * HID, hi - lo, m.pos + lo, ms_cap(D), (cudaStream_t)stream,
                                             dsess ? dsess + lo : nullptr);
        return 0;
    }
    ::ds41dsp::ds_main(D.mw, D.ms, D.tap, rows, rows / S, m.pos, (cudaStream_t)stream, false, dsess);   // the decode call (one session: rows / 1)
    return 0;
}

// ---- draft Model
inline int op_draft_in(void* model, int, int rows, void* stream)
{
    Model& m = *(Model*)model;
    const Dsp& D = m.dsp;
    cudaStream_t st = (cudaStream_t)stream;
    if (!D.vtok || !D.q) return ds41::NOTSUP;
    const int S = m.ds_ns;   // sessions of this draft step (SESS > 1: the runner sets it; the draft StepIn carries each row's slot -> m.sess)
    if (S < 1 || S > D.nsess || rows != S * BLOCK || rows > MX) return 1;
    launch_step_in(m, st);   // *m.stepp (the draft fabric's step), the cells, the SampleRows: the draft's own host StepIn (zero rows = greedy)
    ::ds41dsp::ds_draft_in(D.host_accept ? nullptr : D.vtok, D.vtgt, D.vpos, D.k + 1, S, D.q, D.golden, D.acc, m.tok, m.pos, st);   // overwrites m.tok / m.pos
    ds_mhc::embed(ds41_buffers(m), m.tok, m.emb, rows, HID, st);
    return 0;
}
inline int op_attn_proj(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    if (m.layers[i].l < geom::L_MAIN) return ds41_attn_op_proj(model, i, rows, stream);
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (rows != m.ds_ns * BLOCK || m.ds_ns > m.dsp.nsess || rows > MX) return 1;
    ::ds41dsp::attn_proj_blk(a->W, ring_of(m, *a), a->S, m.hc_y, rows, BLOCK, m.pos, a->heads, (cudaStream_t)stream);
    return 0;
}
inline int op_attn_core(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    if (m.layers[i].l < geom::L_MAIN) return ds41_attn_op_core(model, i, rows, stream);
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (rows != m.ds_ns * BLOCK || m.ds_ns > m.dsp.nsess || rows > MX) return 1;
    ::ds41dsp::attn_core_blk(a->W, ring_of(m, *a), a->S, rows, BLOCK, m.dsp.q, a->heads, (cudaStream_t)stream);
    return 0;
}
inline int op_attn_out(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    if (m.layers[i].l < geom::L_MAIN) return ds41_attn_op_out(model, i, rows, stream);
    Ds41AttnLayer* a = ds41_attn_state(m, i);
    if (!a) return ds41::NOTSUP;
    if (rows != m.ds_ns * BLOCK || rows > MX) return 1;
    ::ds41dsp::attn_out_blk(a->W, a->S, rows, m.pos, a->heads, (ds41a::bf16*)m.sc.part[0], (cudaStream_t)stream);
    return 0;
}
// ST_FINISH: the main Model keeps the step driver's op; the draft Model's last layer also writes the pre-norm head input (xpre = dsp.x, read by the
// confidence head) and does not store the trunk (the draft trunk is discarded). Same call as ds41_op_finish otherwise.
inline int op_finish(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    if (!m.dsp.draft_model) return ds41_op_finish(model, i, rows, stream);
    cudaStream_t st = (cudaStream_t)stream;
    if (!m.dsp.x) return ds41::NOTSUP;
    dmark(DP_OTHER, st);
    const int hcodec = head_for(m, rows).lm_head.codec;
    const bool f16_in = hcodec == bind::MXFP8 || hcodec == bind::FP8R;
    ds_mhc::finish(ds41_buffers(m), Ds41Block{m.f, m.stepp, 2 * i + 1, (size_t)rows * HID * 2}, m.final_norm, rows, HID, geom::RMS_EPS,
                   /*store=*/false, m.hc_y, st, f16_in ? m.sc.xf_head : nullptr, f16_in ? nullptr : m.sc.xf_head, m.dsp.x);
    dmark(DP_NORM_M, st);
    return 0;
}
// the lm_head GEMV of all draft rows (ds_mhc::finish already wrote the head's fragment table); no local argmax: the markov stage adds its
// logits first, then picks the id row by row
inline int op_head_local(void* model, int, int rows, void* stream)
{
    Model& m = *(Model*)model;
    cudaStream_t st = (cudaStream_t)stream;
    if (rows != m.ds_ns * BLOCK || rows > MX) return 1;
    const bind::Head& H = head_for(m, rows);
    bind::launch(H.lm_head, st);
    dmark(DP_HEAD_L, st);
    return 0;
}
inline int op_markov(void* model, int, int rows, void* stream)
{
    Model& m = *(Model*)model;
    const Dsp& D = m.dsp;
    cudaStream_t st = (cudaStream_t)stream;
    const int S = m.ds_ns;
    if (!D.membed || !D.mhead || !D.conf_proj || !D.x) return ds41::NOTSUP;
    if (rows != S * BLOCK || S < 1 || S > D.nsess || S > ::ds41dsp::MARKOV_MAXS) return 1;
    for (int i = 0; i < BLOCK; ++i)
    {
        const int c = ds41::dsp::markov_collective(i);
        ::ds41dsp::markov_add(D.membed, D.mhead, D.golden, D.dtok, i, S, m.hlog, (long)m.vn, m.vn, st);
        // the draft rows of step i are rows s * 5 + i of hlog: base hlog + i * vn, row stride 5 * vn, one row per session
        glue::HeadArgs a{m.f, m.stepp, c, S, m.hlog + (size_t)i * m.vn, 5 * m.vn, m.vlo, m.vn, m.srow, m.tickets + c, D.dtok + (size_t)i * S, m.lp,
                         nullptr, nullptr, ::ds41::dsp::HEAD_NC, nullptr, nullptr};   // nc: the bytes the proxy enqueues (coll_bytes), not m.nc
        glue::k_head_local<<<S, 1024, 0, st>>>(a);
        glue::k_head_merge<<<1, 32, 0, st>>>(a);
    }
    ::ds41dsp::conf_head(D.x, D.membed, D.conf_proj, D.golden, D.dtok, S, D.conf, D.draft, st);
    dmark(DP_HEAD_M, st);
    return 0;
}

// ---- the table slots
inline void fill(ds41::Table& t)
{
    t.fn[ds41::ST_DS_TAP] = &op_tap;
    t.fn[ds41::ST_DS_MAIN] = &op_main;
    t.fn[ds41::ST_DS_DRAFT_IN] = &op_draft_in;
    t.fn[ds41::ST_DS_HEAD_LOCAL] = &op_head_local;
    t.fn[ds41::ST_DS_MARKOV] = &op_markov;
    t.fn[ds41::ST_FINISH] = &op_finish;
    // The attention slots, wrapped: a main layer (l < 40) goes to the attention op as is, a DSpark layer (40..42) to the block-mode kernels
    t.fn[ds41::ST_ATTN_PROJ] = &op_attn_proj;
    t.fn[ds41::ST_ATTN_CORE] = &op_attn_core;
    t.fn[ds41::ST_ATTN_OUT] = &op_attn_out;
}

// ---- build: after build_ds41 of both Models (their attention layer state exists). Fills main.dsp and draft.dsp with the same values.
// Tensors (docs/DEEPSEEK-WEIGHTS.md): L040.main_proj / main_norm, L04j.kv / kv_norm (the attention's bound ones), L042.markov_head.{embed,head},
// L042.confidence_head.proj, L042.norm (the draft head's final norm). markov_head.head is stored full [VOCAB][256]: the rank's shard starts at row vlo.
// tap_rows: capacity of the tap buffer in rows (0 = MX, the decode step; a prefill chunk of W rows needs W: 31.5 MB at W = 1024).
inline Dsp build(Model& main, Model& draft, int nsess, int k, bool host_accept = false, int tap_rows = 0)
{
    // nsess = the S2 slots (Model::ds_slots of both Models): the arrays below hold that many sessions; a step runs 1..nsess of them (Model::ds_ns)
    if (const char* e = ds41slots::dsp_nsess_error(nsess, main.ds_slots, draft.ds_slots, k, MX, ::ds41dsp::MARKOV_MAXS, BLOCK))
        die(std::string(e) + ": nsess " + std::to_string(nsess) + ", slots main " + std::to_string(main.ds_slots) + " draft " + std::to_string(draft.ds_slots) +
            ", k " + std::to_string(k) + ", MX " + std::to_string(MX) + " (" + std::to_string(std::max(nsess * (k + 1), nsess * BLOCK)) + " rows a step)");
    // the head merge waits for head_bytes_tk(1, HEAD_NC, false) (op_markov passes no tk); the proxy sends coll_bytes: the same number or no boot
    if (glue::head_bytes_tk(1, ::ds41::dsp::HEAD_NC, false) != ::ds41::dsp::coll_bytes(::ds41::dsp::markov_collective(0), HID))
        die("DSpark: the draft head merge wait bytes differ from ds41::dsp::coll_bytes");
    // the rows of a round (nsess x (k + 1) verify, nsess x 5 draft; the multi-tile MoE takes M 9..MX): checked above by dsp_nsess_error
    if (draft.nl != geom::N_NEXTN) die("DSpark: the draft Model must hold layers 40..42");
    for (int j = 0; j < geom::N_NEXTN; ++j)
        if (draft.layers[j].l != geom::L_MAIN + j || !ds41_attn_state(draft, j)) die("DSpark: draft layer " + std::to_string(j) + " is not bound");
    if (main.vn != draft.vn || main.vlo != draft.vlo) die("DSpark: main and draft Models must share the vocab shard");
    Dsp D;
    D.nsess = nsess; D.k = k;
    D.tap_rows = tap_rows > MX ? tap_rows : 0;
    D.tap = dmalloc<__nv_bfloat16>((size_t)(D.tap_rows ? D.tap_rows : MX) * dd::TAPS * HID);
    D.mw.main_proj = ds41_fp8(main, geom::L_MAIN, "main_proj", HID, (long long)dd::TAPS * HID);
    D.mw.main_norm = ds41_vec(main, geom::L_MAIN, "main_norm", HID);
    // The draft head norm is the DSpark stage's own mtp.2.norm (official DSparkBlock.forward_head in the reference model.py: head(self.norm(x))), not
    // the main G.final_norm that build_ds41 bound on both Models. op_finish reads m.final_norm at call time, so rebinding the draft Model here is enough.
    draft.final_norm = ds41_vec(main, geom::NLAYERS - 1, "norm", HID);
    for (int j = 0; j < geom::N_NEXTN; ++j)
    {
        const Ds41AttnLayer& a = *ds41_attn_state(draft, j);
        D.mw.wkv[j] = a.W.wkv;
        D.mw.kv_norm[j] = a.W.kv_norm;
        D.mw.ring[j] = ring_of(a);
        D.mw.inv = a.W.inv;                              // plain rope theta 10000: the same table for all three (compress ratio 0)
    }
    D.ms_rows = D.tap_rows ? std::max(MX, (int)::ds41::dsp::PREFILL_MS_ROWS) : 0;   // decode-only build: MX rows
    const int ms_n = D.ms_rows ? D.ms_rows : MX;
    D.ms.carve(dmalloc<uint8_t>(::ds41dsp::MainS::bytes(ms_n)), ms_n);
    if (::ds41dsp::dsp_gv_enabled())
    {
        // GR8_DS_DSP_GV=1: all 19 projection launches of a one-session step must take the GEMV (draft layers at 5 rows, ds_main at k + 1); a matrix
        // without gv records or unbound hooks stops the boot here instead of falling back to the GEMM. Steps of several sessions (more than 8 rows)
        // take the GEMM (dsp_mm).
        int n = 0;
        std::string bad;
        auto chk = [&](const ds41a::Fp8Mat& w, int rows, const std::string& nm) {
            if (::ds41dsp::dsp_gv_ok(w, rows)) ++n;
            else bad += " " + nm;
        };
        chk(D.mw.main_proj, k + 1, "main_proj");
        for (int j = 0; j < geom::N_NEXTN; ++j)
        {
            const ds41a::Weights& w = ds41_attn_state(draft, j)->W;
            const std::string L = "L" + std::to_string(geom::L_MAIN + j) + ".";
            chk(D.mw.wkv[j], k + 1, L + "main_wkv");
            chk(w.wq_a, BLOCK, L + "q_a"); chk(w.wq_b, BLOCK, L + "q_b"); chk(w.wkv, BLOCK, L + "kv");
            chk(w.wo_a, BLOCK, L + "o_a"); chk(w.wo_b, BLOCK, L + "o_b");
        }
        std::fprintf(stderr, "ds41 dspark: GR8_DS_DSP_GV=1: %d of 19 projection launches per step take the GEMV (verify %d rows, draft %d rows a session)\n", n, k + 1,
                     BLOCK);
        if (n != 19) die("DSpark: GR8_DS_DSP_GV=1 but these projections cannot take the GEMV (missing .mx.gv records or hooks):" + bad);
    }
    D.vtok = main.tok; D.vtgt = main.tok_out; D.vpos = main.pos;
    D.q = dmalloc<int>(nsess); D.golden = dmalloc<int>(nsess); D.acc = dmalloc<int>(nsess);
    D.crow0 = dmalloc<int>(nsess); D.cslot = dmalloc<int>(nsess);
    D.x = dmalloc<__nv_bfloat16>((size_t)MX * HID);
    auto bf16_mat = [&](const char* nm, long long rows, long long cols) {
        const std::string p = lname(geom::NLAYERS - 1, nm);
        const TensorRef t = main.ds41_find(p);
        if (!t.p || !t.bf16 || t.nd != 2 || t.shape[0] != rows || t.shape[1] != cols) die("DSpark bind: bad tensor " + p);
        return (const __nv_bfloat16*)t.p;
    };
    D.membed = bf16_mat("markov_head.embed", geom::VOCAB, dd::MARKOV_RANK);
    D.mhead = bf16_mat("markov_head.head", geom::VOCAB, dd::MARKOV_RANK) + (size_t)main.vlo * dd::MARKOV_RANK;
    D.conf_proj = bf16_mat("confidence_head.proj", 1, dd::CONF_IN);
    D.dtok = dmalloc<int>((size_t)BLOCK * nsess);
    D.draft = dmalloc<int>((size_t)nsess * BLOCK);
    D.conf = dmalloc<float>((size_t)nsess * BLOCK);
    main.dsp = D;
    draft.dsp = D;
    draft.dsp.draft_model = true;   // op_finish: xpre = dsp.x, store = false on the draft Model only
    draft.dsp.host_accept = host_accept;   // the served round runner writes q / golden / acc (ds41_dspark_round.cuh)
    return D;
}
}   // namespace ds41dsp_ops
#endif
