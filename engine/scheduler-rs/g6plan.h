// The C ABI of the fabric plan and its sizing rules in Rust (engine/scheduler-rs src/fabric_plan.rs and
// src/plan_sizes.rs, docs/HOST-CORE.md R6), part of libg6host.a. Apart from g6sched.h so the sizing headers that
// kernels and tests include (pf_slots.h, ds41_pf_split.h, cplayout.h, ...) pull in nothing else. g6sched.h includes
// it, and its layout digest covers these structs. Texts are null (unset) or C strings; a char* result is freed with
// g6plan_str_free.
#pragma once
#include <stddef.h>
#include <stdint.h>

extern "C" {

// ds41_prefill_fabric.h FabricRuns: the FabricConfig fields it sets
struct g6plan_runs {
    int32_t run1, run2, run3, ncoll;
    size_t slot_bytes, slot_bytes1, slot_bytes2, slot_bytes3;
    int32_t rs_k, quarter;
    size_t slice_stride;
};
// cplayout.h CpColls's HEADS, KV_RANK, QK_DIM and TOPK
struct g6plan_cp_dims { int32_t heads, kv_rank, qk_dim, topk; };
#ifdef __cplusplus
// ds41_lanes.h, m4sim and the DS41 unit tests reach these two without fabric_plan()'s layout digest check
static_assert(sizeof(g6plan_runs) == 64 && offsetof(g6plan_runs, slot_bytes) == 16 && offsetof(g6plan_runs, rs_k) == 48 &&
              offsetof(g6plan_runs, slice_stride) == 56, "g6plan_runs layout (plan_sizes.rs Runs)");
static_assert(sizeof(g6plan_cp_dims) == 16 && offsetof(g6plan_cp_dims, topk) == 12, "g6plan_cp_dims layout (plan_sizes.rs CpDims)");
#endif

size_t g6plan_pf_ss_min(int32_t mmax, int32_t nr, int32_t hid);
size_t g6plan_pf_slot_bytes(int32_t mmax, bool rs, int32_t nr, int32_t hid);
size_t g6plan_pf_sp_slot_bytes(int32_t mp, int32_t nr, int32_t hid);
void g6plan_ds41_runs(int32_t pf_rows, int32_t tail, int32_t i_late, int32_t nl, int32_t nx, int32_t mx, int32_t hid,
                      int32_t nr, size_t head_payload, bool rs_on, bool c8_pack, bool rowshard, int32_t pack_rows,
                      g6plan_runs* out);
int32_t g6plan_ds41_late_rows(int32_t rows, int32_t tail);
int32_t g6plan_ds41_late_start(const int32_t* ids, size_t n, int32_t late_layer);
int32_t g6plan_ds41_parse_tail(const char* v, int32_t tail_rows, int32_t window);
char* g6plan_ds41_tail_error(const char* v, int32_t window, int32_t late_layer);
void g6plan_str_free(char* s);
int32_t g6plan_ds41_lanes(const char* v);
const char* g6plan_ds41_lanes_error(void);
int32_t g6plan_ds41_prefill_rowcap(const char* v, int32_t decode_rows);
size_t g6plan_ds41_prefill_payload(int32_t rows, int32_t hidden, int32_t engram_dim);
bool g6plan_ds41_rowshard_ok(int32_t width, int32_t tail, int32_t nr, bool rs, bool fuse);
size_t g6plan_ds41_state_row_bytes(int32_t hid, size_t trunk_bytes);
size_t g6plan_ds41_state_bytes(int32_t late, int32_t hid, size_t trunk_bytes);
int32_t g6plan_ds41_main_ncoll(int32_t nc, int32_t k, bool rowshard);
int32_t g6plan_ds41_rs_quarter(int32_t rows, int32_t nr);
size_t g6plan_ds41_rs_slice_stride(int32_t q, size_t row_bytes);
int32_t g6plan_ds41_tail_max(void);
size_t g6plan_qw_bulk_bytes(int32_t e, size_t half);
int32_t g6plan_qw_bulk_flags(int32_t e);
int32_t g6plan_qw_partner(int32_t r);
// CpColls::make into the caller's buffers: rows = {q, cand, part} row bytes; q and cand hold nl, part nl (nr - 1), cap
// and dst_k nl (nr + 1). Returns the collectives.
size_t g6plan_cp_colls(const bool* full, size_t nl, int32_t nr, int32_t mmax, g6plan_cp_dims d, uint32_t* rows, int32_t* q,
                       int32_t* cand, int32_t* part, uint32_t* cap, uint8_t* dst_k);

// fabric_plan.h: a g6::FabricConfig without ips, ips2 and ring_devs
struct g6plan_cfg {
    int32_t rank;
    size_t stripe_min, slot_bytes;
    int32_t run1, run2;
    size_t slot_bytes1, slot_bytes2;
    int32_t run3;
    size_t slot_bytes3;
    int32_t pool;
    bool sge2, compact;
    int32_t ncoll, port, proxy_cpu;
    const uint32_t* vcap;
    size_t n_vcap;
    const uint8_t* vdst_k;
    size_t n_vdst_k;
    size_t bulk_bytes;
    int32_t bulk_flags, xtrace;
    bool ring;
    int32_t relay_cpu;
};
// one planned fabric (kind = FabKind); use_pf_ips2: rail 1 is --pf-ips2, not the main config's ips2
struct g6plan_entry {
    uint32_t kind;
    bool late, use_pf_ips2;
    int32_t port_off;
    g6plan_cfg cfg;
};
// fabric_plan.h PlanKnobs; pf_stripe and cp are -1 unset, 0 or 1
struct g6plan_knobs {
    bool xtrace_pf;
    int32_t pf_stripe;
    int64_t pf_stripe_min;
    bool pf_rs, fpf_attn_p1, fpf_dsa_p1, fpf_p1, has_qa_split_cpu;
    int64_t qa_split_cpu;
    int32_t cp;
    int64_t cp_r, cp_step_rows;
    bool cp_s3, has_cp_mmax;
    int64_t cp_mmax;
    bool has_cp_r2;
    int64_t cp_r2;
    const char *ds_prefill_w, *ds_pf_tail, *ds_pf_pool, *ds_pf_lanes;
    bool ds_pf_l20_tail, ds_pf_rs, ds_mhc_row_shard, ds_mhc_fuse, ds_mhc_y_mxfp8, ds_pf_direct, ds_subdump, ds_stripe, ds_xtrace;
    int64_t ds_stripe_min;
};
// the build (model = G6ModelId) and the constants of its headers the plan reads
struct g6plan_build {
    uint32_t model;
    int32_t nr, mx, hid;
    int32_t ds_window, ds_tail_rows, ds_cand_source, ds_late_start, ds_pack_rows, ds_draft_ncoll;
    int32_t coll_bits;
    g6plan_cp_dims cp;
    size_t qw_half;
    int32_t route_max_nr;
};
// the plan's inputs: the main config as the parse left it, the options, the knobs and the values main computes for it
struct g6plan_in {
    g6plan_build b;
    g6plan_cfg cfg;
    size_t n_ips, n_ips2, n_pf_ips2;
    int32_t nc, nx, nl, mp, mtp, lanes, drafts, chunk, s2sess, rt_win, route_split_mode, c8_pack_env, ds_draft_cap, qp_roll_s, qp_wire_e;
    bool s2mode, dfl, dsk_on;
    const int32_t* layers;
    size_t n_layers;
    const int32_t* row_counts;
    size_t n_row_counts;
    g6plan_knobs knobs;
    bool pf_sp;
    int32_t pool_owner_mode, dfl_ncoll;
    size_t head_payload, trunk_bytes;
    const bool* full_index;
    size_t n_full_index;
    bool cp_build, gather, draft_gather;
    size_t pool_payload, route_payload, dsp_coll_max;
};
struct g6plan_cp {
    int32_t nr, nl, mmax;
    uint32_t q_row, cand_row, part_row;
    const int32_t* q;
    size_t n_q;
    const int32_t* cand;
    size_t n_cand;
    const int32_t* part;
    size_t n_part;
    const uint32_t* cap;
    const uint8_t* dst_k;
    size_t ncoll;
};
// fabric_plan.h FabricPlan; after a refusal n is 0, and the log and scalars hold what was planned before it
struct g6plan_plan {
    const char* refusal;
    const char* const* log;
    size_t n_log;
    int32_t n;
    g6plan_entry entries[16];
    size_t dec_slot_bytes;
    int32_t pf_rows, tail, i_late;
    bool l20_tail;
    int32_t rs_k, quarter;
    size_t slice_stride;
    bool rowshard, y_mxfp8, pf_direct;
    int32_t pf_lanes_fab;
    bool two, pool_owner, route_split, cp_on;
    int32_t cp_r, cp_r2, cp_s3, cp_mmax, cp_step, cp_step_r;
    g6plan_cp cp;
};
struct g6plan_result;
g6plan_result* g6plan_run(const g6plan_in* in);
const g6plan_plan* g6plan_view(const g6plan_result* r);   // lives until g6plan_free
void g6plan_free(g6plan_result* r);
}
