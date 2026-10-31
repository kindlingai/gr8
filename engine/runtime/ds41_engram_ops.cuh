// The DS41 step's Engram wiring (ST_ENGRAM, layers 1 and 14). Included by ds41_step.cuh inside g6rt, before the op table.
// Contract: docs/DEEPSEEK-ENGRAM.md. The engine is ds41::engram::Engram (engine/engram/engram_dev.cuh); this file adds what needs the
// step driver: the token window of the step's rows, the reduce collective (EngramHooks::reduce) and the op.
// Everything here is enqueue-only on the step stream (capturable) and is reached only when GR8_DS_ENGRAM=1 (Model::engram set).
#pragma once
#if G6_MODEL_DS41

// Token window. win[r][k] = the token at position pos[r]-k (k = 0..3), or -1 before the start of the sequence. The rows of one step are
// consecutive positions of one sequence (the offline loop; the basic serve: one session, no prefix reuse, the host builds every row; a DSpark
// verify round has the same shape), so a ring of tokens by absolute position holds what is needed: this step's rows are written first,
// then every row reads its three predecessors, which are either rows of this step or rows written by an earlier step. A verify step that is
// rolled back rewrites the same positions before it reads them (a rejected draft's entry sits above the accepted position and the next step,
// which starts there, overwrites it; the positional cache of the official reference engram.py does the same).
// Assumes each session starts at position 0 (no prefix reuse): the ring needs no reset because a row at position p reads
// p-1..p-3 only after this request wrote them, and a position below 0 gives -1. Several sessions per step (SESS > 1): one ring per S2 slot,
// row r uses the ring of its slot sess[r] (sess = nullptr: slot 0 for every row, the one-session form).
constexpr int ENG_RING = ds41::engram::kRing;   // MX = 32 rows + 3 earlier positions = 35; a power of two (engram_ring.h)
static_assert(MX + 3 <= ENG_RING, "the Engram token ring must hold the rows of a step and the 3 positions before them");
__global__ void __launch_bounds__(MX) k_engram_win(const int* tok, const int* pos, int rows, int* ring0, int32_t* win, const int* sess)
{
    const int r = threadIdx.x;
    int* ring = ring0 + (sess && r < rows ? sess[r] * ENG_RING : 0);
    if (r < rows) ds41::engram::ring_put(ring, pos[r], tok[r]);
    __syncthreads();
    if (r < rows)
        for (int k = 0; k < 4; ++k) win[r * 4 + k] = ds41::engram::ring_get(ring, pos[r] - k);
}

// Collective c of the sum: every rank publishes its emb [rows][6144] BF16 (k_put_publish: the slot copy + the doorbell), then this kernel waits
// for the peers and sums their slots in rank order as FP32 into emb (in place). Rows another rank owns are exact zeros here, so the sum
// is exact. Grid (rows, 6), block 256: 4 columns per thread.
__global__ void __launch_bounds__(256) k_engram_sum(g6::FabricDev f, const u64* stepp, int c, int rows, __nv_bfloat16* emb)
{
    constexpr int D = ds41::engram::kEmbDim;
    static_assert(D == 6 * 256 * 4, "k_engram_sum covers one row with 6 CTAs of 256 threads x 4 columns");
    const u64 step = *stepp;
    glue::wait_peers(f, step, c, g6::coll_seq(step, c, f.ncoll), (size_t)rows * D * 2, glue::ERR_AR);
    const size_t o = (size_t)blockIdx.x * D + (size_t)(blockIdx.y * 256 + threadIdx.x) * 4;
    uint2 v[g6nr::NR];
#pragma unroll
    for (int p = 0; p < g6nr::NR; ++p) v[p] = __ldcv(reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(glue::peer_slot(f, step, c, p)) + o));
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p = 0; p < g6nr::NR; ++p) {
        s[0] += __uint_as_float(v[p].x << 16); s[1] += __uint_as_float(v[p].x & 0xffff0000u);
        s[2] += __uint_as_float(v[p].y << 16); s[3] += __uint_as_float(v[p].y & 0xffff0000u);
    }
    const __nv_bfloat162 a = __floats2bfloat162_rn(s[0], s[1]), b = __floats2bfloat162_rn(s[2], s[3]);
    uint2 out;
    out.x = *reinterpret_cast<const unsigned*>(&a); out.y = *reinterpret_cast<const unsigned*>(&b);
    *reinterpret_cast<uint2*>(emb + o) = out;
}

// EngramHooks::reduce. Collective c = 2*nl+1+e of the fabric (ncoll = 2*nl+3, payload rows*6144*2 B): slots of MX rows in the decode build,
// of the W rows of GR8_DS_PREFILL_W in a prefill build (the boot sizes the slots: stride >= W*12288 + 8). A chunk of more than 32 rows publishes with
// kEngramPutCta CTAs (k_put_publish is grid-stride; the doorbell comes from the last CTA to arrive, whatever the grid), up to 4096 rows.
constexpr int kEngramPutCta = 64;   // CTAs of the W-row publish copy: 12.6 MB at W = 1024 (decode: 1 or 16 CTAs for 32 rows or fewer)
inline int ds41_engram_reduce(void* model, int e, int rows, void* stream, __nv_bfloat16* emb)
{
    Model& m = *(Model*)model;
    if (e < 0 || e >= ds41::engram::kNE || rows < 1 || m.ecoll != ds41::engram::kNE) return (int)cudaErrorInvalidValue;
    const int c = 2 * m.nl + 1 + e;
    if ((size_t)rows * ds41::engram::kEmbDim * 2 + 8 > g6::fab_slot_stride(m.f, c)) return (int)cudaErrorInvalidValue;   // the slot holds the payload and the doorbell word
    cudaStream_t st = (cudaStream_t)stream;
    k_put_publish<<<rows > MX ? kEngramPutCta : rows <= 8 ? 1 : 16, 128, 0, st>>>(reinterpret_cast<const uint4*>(emb), m.f, m.stepp, c,
                                                       rows * (ds41::engram::kEmbDim * 2 / 16), m.tickets + c);
    k_engram_sum<<<dim3(rows, 6), 256, 0, st>>>(m.f, m.stepp, c, rows, emb);
    return (int)cudaGetLastError();
}

// The step's Engram prologue, after k_step_in and before the embed: the token window, then the hash + NVMe prefetch of both layers.
// M <= MX: the decode kernel (put, then get, in one block). M > MX (a W-row prefill chunk): get kernel, then put kernel (Engram::window_wide).
// slot: the S2 slot of a W-row chunk (one session; its ring); the decode kernel reads each row's slot from Model::sess when ds_slots > 1.
inline void ds41_engram_begin(Model& m, int M, cudaStream_t st, int slot = 0)
{
    if (slot < 0 || slot >= m.ds_slots) die("DS41 Engram: slot " + std::to_string(slot) + " outside the S2 slots");
    if (M <= MX) k_engram_win<<<1, MX, 0, st>>>(m.tok, m.pos, M, m.ehist, m.win, m.ds_slots > 1 ? m.sess : nullptr);
    else if (const int x = m.engram->window_wide(m.tok, m.pos, M, m.ehist + (size_t)slot * ENG_RING, m.win, st)) die("DS41 Engram window_wide: code " + std::to_string(x));
    G6RT_CK(cudaGetLastError());
    if (const int x = m.engram->begin_step(M, m.win, st)) die("DS41 Engram begin_step: code " + std::to_string(x));
}
// Ragged W pack: build each slot's own consecutive token window before hashing the
// concatenated [rows][4] windows once.
inline void ds41_engram_begin_pack(Model& m, const ds41::pack::Layout& p, cudaStream_t st)
{
    if (const char* e = ds41::pack::error(p, m.hc_rows, m.ds_slots)) die(std::string("DS41 Engram pack: ") + e);
    for (int i = 0; i < p.count; ++i) {
        const auto& s = p.seg[i];
        const int rc = m.engram->window_wide(m.tok + s.row0, m.pos + s.row0, s.rows,
                                           m.ehist + size_t(s.slot) * ENG_RING, m.win + size_t(s.row0) * 4, st);
        if (rc) die("DS41 Engram ragged window_wide: code " + std::to_string(rc));
    }
    if (const int rc = m.engram->begin_step(p.rows, m.win, st)) die("DS41 Engram ragged begin_step: code " + std::to_string(rc));
}

// ST_ENGRAM, in U_CONS_F of the layer before each Engram layer: the trunk m.resid4 ([rows][4][HID], BF16-valued; ds_mhc::trunk_t = float, or __nv_bfloat16 under -DDS_MHC_TRUNK_BF16) is edited in place;
// op() is overloaded on the trunk type, so the pointer type picks the k_gate that matches the storage (no cast here).
inline int ds41_op_engram(void* model, int i, int rows, void* stream)
{
    Model& m = *(Model*)model;
    if (!m.engram) return ds41::NOTSUP;
    return m.engram->op(model, m.layers[i].l == geom::ENGRAM_L0 ? 0 : 1, rows, m.resid4, (cudaStream_t)stream);
}

// Serve (s2/ds41_native.cuh): the Engram engine fails closed after a step whose stream is synchronized (rank.cuh step_line does the same for the
// offline loop). Nonzero = a failed NVMe read, a window token outside the vocabulary or rows of a W-row chunk that are not consecutive positions; it ORs into the serve step's error word, which stops the service.
inline unsigned ds41_engram_error(const Model& m)
{
    if (!m.engram) return 0;
    const int es = m.engram->status();
    if (!es) return 0;
    fprintf(stderr, "g6m4: DS41 Engram: %s\n", ds41::engram::Engram::status_text(es).c_str());
    return 0x80000000u;
}

// ---- boot
// GR8_DS_ENGRAM and its knobs. Off (default) reads nothing else, so a stray knob cannot change an Engram-off run.
inline ds41::engram::Knobs ds41_engram_knobs()
{
    ds41::engram::Knobs k;
    std::string err;
    if (!knob::ds_engram()) return k;
    if (!ds41::engram::read_knobs(&k, &err)) die(err);
    if (k.prefix.empty()) die("GR8_DS_ENGRAM=1 needs GR8_DS_ENGRAM_PREFIX (PREFIX.engram.json, written at boot, and PREFIX.engram.cmap)");
    return k;
}
// Extra collectives of the decode fabric (g6m4.cu, before the Model exists): 0 off, else kNE = 2 after the head slot. Both Engram layers must be served.
inline int ds41_engram_colls(const std::vector<int>& ids)
{
    if (!ds41_engram_knobs().on) return 0;
    for (int l : ids) if (l == geom::ENGRAM_L0)
        for (int l2 : ids) if (l2 == geom::ENGRAM_L1) return ds41::engram::kNE;
    die("DS41 build: GR8_DS_ENGRAM=1 needs layers " + std::to_string(geom::ENGRAM_L0) + " and " + std::to_string(geom::ENGRAM_L1) + " in --layers");
    return 0;
}
#endif
