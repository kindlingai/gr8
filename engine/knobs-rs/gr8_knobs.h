#pragma once
// Written by `cargo run --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen` from the knobs/ YAML files:
// change the table, not this file. docs/ENV.md describes every knob.
//
// knob::name() returns the value of GR8_NAME: its default when unset, or an optional for a knob without one. The value
// is read once per process. A value the table refuses stops the process at its first read. knob::init() at the start
// of main prints every problem of the environment and returns how many are errors. The C API is in libg6kvman.a.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

extern "C" {
int gr8_knobs_init(uint64_t table_hash);
const char* gr8_knobs_signature(void);
const char* gr8_knobs_tests_set(void);
int gr8_knob_flag(uint32_t id, uint32_t n, int* out);
int gr8_knob_int(uint32_t id, uint32_t n, int64_t* out);
int gr8_knob_float(uint32_t id, uint32_t n, double* out);
const char* gr8_knob_text(uint32_t id, uint32_t n);
ptrdiff_t gr8_knob_ints(uint32_t id, uint32_t n, const int64_t** out);
ptrdiff_t gr8_knob_floats(uint32_t id, uint32_t n, const double** out);
}

namespace knob
{

/// Prints the environment's problems (bad values, unknown GR8_ names, old G6_ names) to stderr and returns the number
/// of errors. The engine stops when it is not 0.
inline int init() { return gr8_knobs_init(0x00132b9fa0b8335cull); }

/// The set agree-scope knobs that differ from their defaults, as " env NAME=value ...", or "".
inline std::string signature() { return gr8_knobs_signature(); }

/// The test knobs that are set, as GR8_TEST_ names joined by spaces, or "". A serving engine refuses to start with any.
inline std::string tests_set() { return gr8_knobs_tests_set(); }

namespace detail
{
inline std::optional<bool> flag(uint32_t id, uint32_t n)
{
    int v;
    if (!gr8_knob_flag(id, n, &v)) return std::nullopt;
    return v != 0;
}
inline std::optional<int64_t> int_(uint32_t id, uint32_t n)
{
    int64_t v;
    if (!gr8_knob_int(id, n, &v)) return std::nullopt;
    return v;
}
inline std::optional<double> float_(uint32_t id, uint32_t n)
{
    double v;
    if (!gr8_knob_float(id, n, &v)) return std::nullopt;
    return v;
}
inline std::optional<std::string> text(uint32_t id, uint32_t n)
{
    const char* v = gr8_knob_text(id, n);
    if (!v) return std::nullopt;
    return std::string(v);
}
inline std::optional<std::vector<int64_t>> ints(uint32_t id, uint32_t n)
{
    const int64_t* v;
    const ptrdiff_t len = gr8_knob_ints(id, n, &v);
    if (len < 0) return std::nullopt;
    return std::vector<int64_t>(v, v + len);
}
inline std::optional<std::vector<double>> floats(uint32_t id, uint32_t n)
{
    const double* v;
    const ptrdiff_t len = gr8_knob_floats(id, n, &v);
    if (len < 0) return std::nullopt;
    return std::vector<double>(v, v + len);
}
} // namespace detail

/// GR8_ATTN_CTX2
inline int64_t attn_ctx2()
{
    static const int64_t v = *detail::int_(0, 0);
    return v;
}

/// GR8_ATTN_CTX3
inline int64_t attn_ctx3()
{
    static const int64_t v = *detail::int_(1, 0);
    return v;
}

/// GR8_ATTN_CTX3_HG
inline int64_t attn_ctx3_hg()
{
    static const int64_t v = *detail::int_(2, 0);
    return v;
}

/// GR8_ATTN_CTX3_SP
inline int64_t attn_ctx3_sp()
{
    static const int64_t v = *detail::int_(3, 0);
    return v;
}

/// GR8_ATTN_CTX_TILE
inline const std::optional<std::string>& attn_ctx_tile()
{
    static const std::optional<std::string> v = detail::text(4, 0);
    return v;
}

/// GR8_ATTN_D256
inline bool attn_d256()
{
    static const bool v = *detail::flag(5, 0);
    return v;
}

/// GR8_ATTN_FLIP
inline bool attn_flip()
{
    static const bool v = *detail::flag(6, 0);
    return v;
}

/// GR8_ATTN_FLIP_MIN
inline int64_t attn_flip_min()
{
    static const int64_t v = *detail::int_(7, 0);
    return v;
}

/// GR8_ATTN_FRONT
inline bool attn_front()
{
    static const bool v = *detail::flag(8, 0);
    return v;
}

/// GR8_ATTN_HEADS
inline bool attn_heads()
{
    static const bool v = *detail::flag(9, 0);
    return v;
}

/// GR8_ATTN_P11
inline bool attn_p11()
{
    static const bool v = *detail::flag(10, 0);
    return v;
}

/// GR8_ATTN_POSIDX
inline bool attn_posidx()
{
    static const bool v = *detail::flag(11, 0);
    return v;
}

/// GR8_ATTN_QT
inline bool attn_qt()
{
    static const bool v = *detail::flag(12, 0);
    return v;
}

/// GR8_ATTN_SC2
inline int64_t attn_sc2()
{
    static const int64_t v = *detail::int_(13, 0);
    return v;
}

/// GR8_ATTN_SC2_PART
inline bool attn_sc2_part()
{
    static const bool v = *detail::flag(14, 0);
    return v;
}

/// GR8_ATTN_SC2_TR
inline int64_t attn_sc2_tr()
{
    static const int64_t v = *detail::int_(15, 0);
    return v;
}

/// GR8_ATTN_SEL2
inline int64_t attn_sel2()
{
    static const int64_t v = *detail::int_(16, 0);
    return v;
}

/// GR8_ATTN_TILE_L
inline int64_t attn_tile_l()
{
    static const int64_t v = *detail::int_(17, 0);
    return v;
}

/// GR8_AVP1
inline bool avp1()
{
    static const bool v = *detail::flag(18, 0);
    return v;
}

/// GR8_CACHE
inline const std::optional<std::string>& cache()
{
    static const std::optional<std::string> v = detail::text(19, 0);
    return v;
}

/// GR8_CLUSTER
inline const std::optional<std::string>& cluster()
{
    static const std::optional<std::string> v = detail::text(20, 0);
    return v;
}

/// GR8_CONTAINER
inline std::optional<bool> container()
{
    static const std::optional<bool> v = detail::flag(21, 0);
    return v;
}

/// GR8_CP
inline std::optional<bool> cp()
{
    static const std::optional<bool> v = detail::flag(22, 0);
    return v;
}

/// GR8_CP_MMAX
inline std::optional<int64_t> cp_mmax()
{
    static const std::optional<int64_t> v = detail::int_(23, 0);
    return v;
}

/// GR8_CP_ONECHAIN
inline bool cp_onechain()
{
    static const bool v = *detail::flag(24, 0);
    return v;
}

/// GR8_CP_PF_HASH2
inline bool cp_pf_hash2()
{
    static const bool v = *detail::flag(25, 0);
    return v;
}

/// GR8_CP_PF_VERIFY
inline bool cp_pf_verify()
{
    static const bool v = *detail::flag(26, 0);
    return v;
}

/// GR8_CP_R
inline int64_t cp_r()
{
    static const int64_t v = *detail::int_(27, 0);
    return v;
}

/// GR8_CP_R2
inline std::optional<int64_t> cp_r2()
{
    static const std::optional<int64_t> v = detail::int_(28, 0);
    return v;
}

/// GR8_CP_R2_BIG
inline int64_t cp_r2_big()
{
    static const int64_t v = *detail::int_(29, 0);
    return v;
}

/// GR8_CP_S3
inline bool cp_s3()
{
    static const bool v = *detail::flag(30, 0);
    return v;
}

/// GR8_CP_STEP_ROWS
inline int64_t cp_step_rows()
{
    static const int64_t v = *detail::int_(31, 0);
    return v;
}

/// GR8_CP_STEP_SEG
inline bool cp_step_seg()
{
    static const bool v = *detail::flag(32, 0);
    return v;
}

/// GR8_DEC_BK
inline int64_t dec_bk()
{
    static const int64_t v = *detail::int_(33, 0);
    return v;
}

/// GR8_DEC_EVPROF
inline int64_t dec_evprof()
{
    static const int64_t v = *detail::int_(34, 0);
    return v;
}

/// GR8_DEC_PROF
inline bool dec_prof()
{
    static const bool v = *detail::flag(35, 0);
    return v;
}

/// GR8_DEC_SHARED_SCRATCH
inline bool dec_shared_scratch()
{
    static const bool v = *detail::flag(36, 0);
    return v;
}

/// GR8_DEC_SPLIT
inline int64_t dec_split()
{
    static const int64_t v = *detail::int_(37, 0);
    return v;
}

/// GR8_DFL_BODY_I4
inline bool dfl_body_i4()
{
    static const bool v = *detail::flag(38, 0);
    return v;
}

/// GR8_DFL_DUMP
inline const std::optional<std::string>& dfl_dump()
{
    static const std::optional<std::string> v = detail::text(39, 0);
    return v;
}

/// GR8_DFL_HEAD_I4
inline bool dfl_head_i4()
{
    static const bool v = *detail::flag(40, 0);
    return v;
}

/// GR8_DFL_PFCTX
inline bool dfl_pfctx()
{
    static const bool v = *detail::flag(41, 0);
    return v;
}

/// GR8_DFL_REUSE
inline std::optional<int64_t> dfl_reuse()
{
    static const std::optional<int64_t> v = detail::int_(42, 0);
    return v;
}

/// GR8_DFL_REUSE_MIN
inline int64_t dfl_reuse_min()
{
    static const int64_t v = *detail::int_(43, 0);
    return v;
}

/// GR8_DFL_SEL_ALPHA
inline double dfl_sel_alpha()
{
    static const double v = *detail::float_(44, 0);
    return v;
}

/// GR8_DRAFT_HEAD2
inline int64_t draft_head2()
{
    static const int64_t v = *detail::int_(45, 0);
    return v;
}

/// GR8_DRAFT_HEAD_I4
inline bool draft_head_i4()
{
    static const bool v = *detail::flag(46, 0);
    return v;
}

/// GR8_DRAFT_HEAD_KS
inline int64_t draft_head_ks()
{
    static const int64_t v = *detail::int_(47, 0);
    return v;
}

/// GR8_DRAFT_HEAD_TPC
inline int64_t draft_head_tpc()
{
    static const int64_t v = *detail::int_(48, 0);
    return v;
}

/// GR8_DRAFT_IN_DUMP
inline const std::optional<std::string>& draft_in_dump()
{
    static const std::optional<std::string> v = detail::text(49, 0);
    return v;
}

/// GR8_DRAFT_LOGIT_DUMP
inline const std::optional<std::string>& draft_logit_dump()
{
    static const std::optional<std::string> v = detail::text(50, 0);
    return v;
}

/// GR8_DRAFT_PAR
inline std::optional<int64_t> draft_par()
{
    static const std::optional<int64_t> v = detail::int_(51, 0);
    return v;
}

/// GR8_DS41_BORROW_TOPK
inline const std::optional<std::string>& ds41_borrow_topk()
{
    static const std::optional<std::string> v = detail::text(52, 0);
    return v;
}

/// GR8_DS41_IDX_DENSE_MAXK
inline int64_t ds41_idx_dense_maxk()
{
    static const int64_t v = *detail::int_(53, 0);
    return v;
}

/// GR8_DS41_IDX_LOGITS
inline const std::string& ds41_idx_logits()
{
    static const std::string v = *detail::text(54, 0);
    return v;
}

/// GR8_DS41_IDX_MAXK
inline int64_t ds41_idx_maxk()
{
    static const int64_t v = *detail::int_(55, 0);
    return v;
}

/// GR8_DS41_IDX_PIPE_B
inline bool ds41_idx_pipe_b()
{
    static const bool v = *detail::flag(56, 0);
    return v;
}

/// GR8_DS41_IDX_PIPE_QR
inline bool ds41_idx_pipe_qr()
{
    static const bool v = *detail::flag(57, 0);
    return v;
}

/// GR8_DS41_IDX_PW
inline std::optional<int64_t> ds41_idx_pw()
{
    static const std::optional<int64_t> v = detail::int_(58, 0);
    return v;
}

/// GR8_DS41_IDX_PWK
inline int64_t ds41_idx_pwk()
{
    static const int64_t v = *detail::int_(59, 0);
    return v;
}

/// GR8_DS41_IDX_Q_TRIG
inline bool ds41_idx_q_trig()
{
    static const bool v = *detail::flag(60, 0);
    return v;
}

/// GR8_DS41_IDX_REUSE
inline bool ds41_idx_reuse()
{
    static const bool v = *detail::flag(61, 0);
    return v;
}

/// GR8_DS41_IDX_TOPK
inline const std::string& ds41_idx_topk()
{
    static const std::string v = *detail::text(62, 0);
    return v;
}

/// GR8_DS41_IDX_WQB_TC
inline int64_t ds41_idx_wqb_tc()
{
    static const int64_t v = *detail::int_(63, 0);
    return v;
}

/// GR8_DS_ATTN_CPA
inline bool ds_attn_cpa()
{
    static const bool v = *detail::flag(64, 0);
    return v;
}

/// GR8_DS_ATTN_GV
inline bool ds_attn_gv()
{
    static const bool v = *detail::flag(65, 0);
    return v;
}

/// GR8_DS_ATTN_GV_KS
inline int64_t ds_attn_gv_ks()
{
    static const int64_t v = *detail::int_(66, 0);
    return v;
}

/// GR8_DS_ATTN_GV_TPC
inline int64_t ds_attn_gv_tpc()
{
    static const int64_t v = *detail::int_(67, 0);
    return v;
}

/// GR8_DS_ATTN_PW
inline int64_t ds_attn_pw()
{
    static const int64_t v = *detail::int_(68, 0);
    return v;
}

/// GR8_DS_ATTN_SPLIT
inline bool ds_attn_split()
{
    static const bool v = *detail::flag(69, 0);
    return v;
}

/// GR8_DS_ATTN_TC
inline bool ds_attn_tc()
{
    static const bool v = *detail::flag(70, 0);
    return v;
}

/// GR8_DS_ATTN_TC_MF
inline int64_t ds_attn_tc_mf()
{
    static const int64_t v = *detail::int_(71, 0);
    return v;
}

/// GR8_DS_BLOCK
inline const std::optional<std::string>& ds_block()
{
    static const std::optional<std::string> v = detail::text(72, 0);
    return v;
}

/// GR8_DS_C2_TRACE
inline bool ds_c2_trace()
{
    static const bool v = *detail::flag(73, 0);
    return v;
}

/// GR8_DS_C8PACK
inline bool ds_c8pack()
{
    static const bool v = *detail::flag(74, 0);
    return v;
}

/// GR8_DS_CACHE_MIB
inline std::optional<int64_t> ds_cache_mib()
{
    static const std::optional<int64_t> v = detail::int_(75, 0);
    return v;
}

/// GR8_DS_COMP_FP8
inline bool ds_comp_fp8()
{
    static const bool v = *detail::flag(76, 0);
    return v;
}

/// GR8_DS_DEC_STAGE
inline bool ds_dec_stage()
{
    static const bool v = *detail::flag(77, 0);
    return v;
}

/// GR8_DS_DSP_GV
inline bool ds_dsp_gv()
{
    static const bool v = *detail::flag(78, 0);
    return v;
}

/// GR8_DS_DSP_TC
inline bool ds_dsp_tc()
{
    static const bool v = *detail::flag(79, 0);
    return v;
}

/// GR8_DS_ENGRAM
inline bool ds_engram()
{
    static const bool v = *detail::flag(80, 0);
    return v;
}

/// GR8_DS_ENGRAM_CACHE_MIB
inline int64_t ds_engram_cache_mib()
{
    static const int64_t v = *detail::int_(81, 0);
    return v;
}

/// GR8_DS_ENGRAM_IO_THREADS
inline int64_t ds_engram_io_threads()
{
    static const int64_t v = *detail::int_(82, 0);
    return v;
}

/// GR8_DS_ENGRAM_LOOKAHEAD
inline bool ds_engram_lookahead()
{
    static const bool v = *detail::flag(83, 0);
    return v;
}

/// GR8_DS_ENGRAM_MTP
inline bool ds_engram_mtp()
{
    static const bool v = *detail::flag(84, 0);
    return v;
}

/// GR8_DS_ENGRAM_PREFIX
inline const std::optional<std::string>& ds_engram_prefix()
{
    static const std::optional<std::string> v = detail::text(85, 0);
    return v;
}

/// GR8_DS_ENGRAM_VERBOSE
inline bool ds_engram_verbose()
{
    static const bool v = *detail::flag(86, 0);
    return v;
}

/// GR8_DS_ENGRAM_WAYS
inline int64_t ds_engram_ways()
{
    static const int64_t v = *detail::int_(87, 0);
    return v;
}

/// GR8_DS_FEED_DIR
inline const std::optional<std::string>& ds_feed_dir()
{
    static const std::optional<std::string> v = detail::text(88, 0);
    return v;
}

/// GR8_DS_GATE_RO
inline bool ds_gate_ro()
{
    static const bool v = *detail::flag(89, 0);
    return v;
}

/// GR8_DS_MANIFEST
inline const std::optional<std::string>& ds_manifest()
{
    static const std::optional<std::string> v = detail::text(90, 0);
    return v;
}

/// GR8_DS_MHC_DECOMP
inline bool ds_mhc_decomp()
{
    static const bool v = *detail::flag(91, 0);
    return v;
}

/// GR8_DS_MHC_FUSE
inline bool ds_mhc_fuse()
{
    static const bool v = *detail::flag(92, 0);
    return v;
}

/// GR8_DS_MHC_ROW_SHARD
inline bool ds_mhc_row_shard()
{
    static const bool v = *detail::flag(93, 0);
    return v;
}

/// GR8_DS_MHC_Y_MXFP8
inline bool ds_mhc_y_mxfp8()
{
    static const bool v = *detail::flag(94, 0);
    return v;
}

/// GR8_DS_MOE_ELOG
inline bool ds_moe_elog()
{
    static const bool v = *detail::flag(95, 0);
    return v;
}

/// GR8_DS_MOE_FC1BSTG
inline bool ds_moe_fc1bstg()
{
    static const bool v = *detail::flag(96, 0);
    return v;
}

/// GR8_DS_MOE_FC2STG
inline bool ds_moe_fc2stg()
{
    static const bool v = *detail::flag(97, 0);
    return v;
}

/// GR8_DS_MOE_L2HINT
inline int64_t ds_moe_l2hint()
{
    static const int64_t v = *detail::int_(98, 0);
    return v;
}

/// GR8_DS_MOE_PDL
inline bool ds_moe_pdl()
{
    static const bool v = *detail::flag(99, 0);
    return v;
}

/// GR8_DS_MOE_PW_BOUND
inline bool ds_moe_pw_bound()
{
    static const bool v = *detail::flag(100, 0);
    return v;
}

/// GR8_DS_MOE_SH_1PASS
inline bool ds_moe_sh_1pass()
{
    static const bool v = *detail::flag(101, 0);
    return v;
}

/// GR8_DS_MOE_TOPK
inline int64_t ds_moe_topk()
{
    static const int64_t v = *detail::int_(102, 0);
    return v;
}

/// GR8_DS_MOE_TOPK_DECODE
inline std::optional<int64_t> ds_moe_topk_decode()
{
    static const std::optional<int64_t> v = detail::int_(103, 0);
    return v;
}

/// GR8_DS_MTP
inline const std::optional<std::string>& ds_mtp()
{
    static const std::optional<std::string> v = detail::text(104, 0);
    return v;
}

/// GR8_DS_PF_DIRECT
inline bool ds_pf_direct()
{
    static const bool v = *detail::flag(105, 0);
    return v;
}

/// GR8_DS_PF_GATHER
inline bool ds_pf_gather()
{
    static const bool v = *detail::flag(106, 0);
    return v;
}

/// GR8_DS_PF_LANES
inline const std::optional<std::string>& ds_pf_lanes()
{
    static const std::optional<std::string> v = detail::text(107, 0);
    return v;
}

/// GR8_DS_PF_LATE_FINAL
inline bool ds_pf_late_final()
{
    static const bool v = *detail::flag(108, 0);
    return v;
}

/// GR8_DS_PF_POOL
inline const std::optional<std::string>& ds_pf_pool()
{
    static const std::optional<std::string> v = detail::text(109, 0);
    return v;
}

/// GR8_DS_PF_RS
inline bool ds_pf_rs()
{
    static const bool v = *detail::flag(110, 0);
    return v;
}

/// GR8_DS_PF_TAIL
inline const std::optional<std::string>& ds_pf_tail()
{
    static const std::optional<std::string> v = detail::text(111, 0);
    return v;
}

/// GR8_DS_PREFILL_KEEP
inline const std::optional<std::string>& ds_prefill_keep()
{
    static const std::optional<std::string> v = detail::text(112, 0);
    return v;
}

/// GR8_DS_PREFILL_MIN
inline int64_t ds_prefill_min()
{
    static const int64_t v = *detail::int_(113, 0);
    return v;
}

/// GR8_DS_PREFILL_W
inline const std::optional<std::string>& ds_prefill_w()
{
    static const std::optional<std::string> v = detail::text(114, 0);
    return v;
}

/// GR8_DS_RELAXED_ACCEPT
inline const std::optional<std::string>& ds_relaxed_accept()
{
    static const std::optional<std::string> v = detail::text(115, 0);
    return v;
}

/// GR8_DS_RO_L2LA
inline int64_t ds_ro_l2la()
{
    static const int64_t v = *detail::int_(116, 0);
    return v;
}

/// GR8_DS_RO_ORDER
inline bool ds_ro_order()
{
    static const bool v = *detail::flag(117, 0);
    return v;
}

/// GR8_DS_SH_TC
inline bool ds_sh_tc()
{
    static const bool v = *detail::flag(118, 0);
    return v;
}

/// GR8_DS_SPAN
inline int64_t ds_span()
{
    static const int64_t v = *detail::int_(119, 0);
    return v;
}

/// GR8_DS_SRC_FAST
inline int64_t ds_src_fast()
{
    static const int64_t v = *detail::int_(120, 0);
    return v;
}

/// GR8_DS_STRIPE
inline bool ds_stripe()
{
    static const bool v = *detail::flag(121, 0);
    return v;
}

/// GR8_DS_STRIPE_MIN
inline int64_t ds_stripe_min()
{
    static const int64_t v = *detail::int_(122, 0);
    return v;
}

/// GR8_DS_SUBDUMP
inline const std::optional<std::string>& ds_subdump()
{
    static const std::optional<std::string> v = detail::text(123, 0);
    return v;
}

/// GR8_DS_WAIT_K
inline int64_t ds_wait_k()
{
    static const int64_t v = *detail::int_(124, 0);
    return v;
}

/// GR8_DS_WOB_PERSIST
inline bool ds_wob_persist()
{
    static const bool v = *detail::flag(125, 0);
    return v;
}

/// GR8_DS_XTRACE
inline bool ds_xtrace()
{
    static const bool v = *detail::flag(126, 0);
    return v;
}

/// GR8_DTR
inline bool dtr()
{
    static const bool v = *detail::flag(127, 0);
    return v;
}

/// GR8_DTR_NBFAST
inline bool dtr_nbfast()
{
    static const bool v = *detail::flag(128, 0);
    return v;
}

/// GR8_DTR_NBFAST_OUT
inline bool dtr_nbfast_out()
{
    static const bool v = *detail::flag(129, 0);
    return v;
}

/// GR8_DTR_PF_COPY
inline bool dtr_pf_copy()
{
    static const bool v = *detail::flag(130, 0);
    return v;
}

/// GR8_DTR_PF_KDA_COPY
inline bool dtr_pf_kda_copy()
{
    static const bool v = *detail::flag(131, 0);
    return v;
}

/// GR8_DTR_PF_SHARE_SUH
inline bool dtr_pf_share_suh()
{
    static const bool v = *detail::flag(132, 0);
    return v;
}

/// GR8_ENV
inline const std::optional<std::string>& env()
{
    static const std::optional<std::string> v = detail::text(133, 0);
    return v;
}

/// GR8_FABRIC_CONNECT_TRIES
inline int64_t fabric_connect_tries()
{
    static const int64_t v = *detail::int_(134, 0);
    return v;
}

/// GR8_FETCH_WORKERS
inline std::optional<int64_t> fetch_workers()
{
    static const std::optional<int64_t> v = detail::int_(135, 0);
    return v;
}

/// GR8_FPF_ATTN_P1
inline bool fpf_attn_p1()
{
    static const bool v = *detail::flag(136, 0);
    return v;
}

/// GR8_FPF_DSA_P1
inline bool fpf_dsa_p1()
{
    static const bool v = *detail::flag(137, 0);
    return v;
}

/// GR8_FPF_P1
inline bool fpf_p1()
{
    static const bool v = *detail::flag(138, 0);
    return v;
}

/// GR8_FUSE
inline int64_t fuse()
{
    static const int64_t v = *detail::int_(139, 0);
    return v;
}

/// GR8_GEMM
inline int64_t gemm()
{
    static const int64_t v = *detail::int_(140, 0);
    return v;
}

/// GR8_GLM53F_CACHE
inline const std::string& glm53f_cache()
{
    static const std::string v = *detail::text(141, 0);
    return v;
}

/// GR8_GLM53F_DEC_MHC_STAGE
inline bool glm53f_dec_mhc_stage()
{
    static const bool v = *detail::flag(142, 0);
    return v;
}

/// GR8_GLM53F_DRAFT_CACHE_DUMP
inline const std::optional<std::string>& glm53f_draft_cache_dump()
{
    static const std::optional<std::string> v = detail::text(143, 0);
    return v;
}

/// GR8_GLM53F_DRAFT_KV
inline bool glm53f_draft_kv()
{
    static const bool v = *detail::flag(144, 0);
    return v;
}

/// GR8_GLM53F_DSA_IDXQ_TPC
inline int64_t glm53f_dsa_idxq_tpc()
{
    static const int64_t v = *detail::int_(145, 0);
    return v;
}

/// GR8_GLM53F_DSA_OPROJ_ROT
inline std::optional<bool> glm53f_dsa_oproj_rot()
{
    static const std::optional<bool> v = detail::flag(146, 0);
    return v;
}

/// GR8_GLM53F_DSA_OPROJ_SPLIT
inline const std::optional<std::vector<int64_t>>& glm53f_dsa_oproj_split()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(147, 0);
    return v;
}

/// GR8_GLM53F_DSA_QB_FUSE
inline int64_t glm53f_dsa_qb_fuse()
{
    static const int64_t v = *detail::int_(148, 0);
    return v;
}

/// GR8_GLM53F_DSA_QB_ROT
inline bool glm53f_dsa_qb_rot()
{
    static const bool v = *detail::flag(149, 0);
    return v;
}

/// GR8_GLM53F_DSA_QB_SPLIT
inline const std::optional<std::vector<int64_t>>& glm53f_dsa_qb_split()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(150, 0);
    return v;
}

/// GR8_GLM53F_DSA_XIN_ORDER
inline bool glm53f_dsa_xin_order()
{
    static const bool v = *detail::flag(151, 0);
    return v;
}

/// GR8_GLM53F_FPF_DENSE_P1
inline bool glm53f_fpf_dense_p1()
{
    static const bool v = *detail::flag(152, 0);
    return v;
}

/// GR8_GLM53F_FPF_KDA_P1
inline bool glm53f_fpf_kda_p1()
{
    static const bool v = *detail::flag(153, 0);
    return v;
}

/// GR8_GLM53F_KDA_MX
inline bool glm53f_kda_mx()
{
    static const bool v = *detail::flag(154, 0);
    return v;
}

/// GR8_GLM53F_KDA_OPROJ_SPLIT
inline const std::optional<std::vector<int64_t>>& glm53f_kda_oproj_split()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(155, 0);
    return v;
}

/// GR8_GLM53F_KDA_PF_IN1
inline bool glm53f_kda_pf_in1()
{
    static const bool v = *detail::flag(156, 0);
    return v;
}

/// GR8_GLM53F_KDA_PF_P1
inline bool glm53f_kda_pf_p1()
{
    static const bool v = *detail::flag(157, 0);
    return v;
}

/// GR8_GLM53F_KDA_PF_ST1
inline bool glm53f_kda_pf_st1()
{
    static const bool v = *detail::flag(158, 0);
    return v;
}

/// GR8_GLM53F_KDA_PROJ_KS
inline int64_t glm53f_kda_proj_ks()
{
    static const int64_t v = *detail::int_(159, 0);
    return v;
}

/// GR8_GLM53F_KDA_PROJ_MODE
inline int64_t glm53f_kda_proj_mode()
{
    static const int64_t v = *detail::int_(160, 0);
    return v;
}

/// GR8_GLM53F_KDA_SCAN
inline int64_t glm53f_kda_scan()
{
    static const int64_t v = *detail::int_(161, 0);
    return v;
}

/// GR8_GLM53F_KDA_STATE_DUMP
inline const std::optional<std::string>& glm53f_kda_state_dump()
{
    static const std::optional<std::string> v = detail::text(162, 0);
    return v;
}

/// GR8_GLM53F_KDA_VS
inline int64_t glm53f_kda_vs()
{
    static const int64_t v = *detail::int_(163, 0);
    return v;
}

/// GR8_GLM53F_KV_CHUNKS
inline bool glm53f_kv_chunks()
{
    static const bool v = *detail::flag(164, 0);
    return v;
}

/// GR8_GLM53F_KV_CRASH_AT
inline const std::optional<std::string>& glm53f_kv_crash_at()
{
    static const std::optional<std::string> v = detail::text(165, 0);
    return v;
}

/// GR8_GLM53F_KV_IDLE_CHUNKS
inline int64_t glm53f_kv_idle_chunks()
{
    static const int64_t v = *detail::int_(166, 0);
    return v;
}

/// GR8_GLM53F_KV_LEDGER
inline const std::optional<std::string>& glm53f_kv_ledger()
{
    static const std::optional<std::string> v = detail::text(167, 0);
    return v;
}

/// GR8_GLM53F_KV_LEND_MS
inline int64_t glm53f_kv_lend_ms()
{
    static const int64_t v = *detail::int_(168, 0);
    return v;
}

/// GR8_GLM53F_MHC_FUSE
inline int64_t glm53f_mhc_fuse()
{
    static const int64_t v = *detail::int_(169, 0);
    return v;
}

/// GR8_GLM53F_MHC_FUSE_M1
inline std::optional<int64_t> glm53f_mhc_fuse_m1()
{
    static const std::optional<int64_t> v = detail::int_(170, 0);
    return v;
}

/// GR8_GLM53F_MHC_FUSE_M4
inline std::optional<int64_t> glm53f_mhc_fuse_m4()
{
    static const std::optional<int64_t> v = detail::int_(171, 0);
    return v;
}

/// GR8_GLM53F_PF_AMARK
inline bool glm53f_pf_amark()
{
    static const bool v = *detail::flag(172, 0);
    return v;
}

/// GR8_GLM53F_PF_BF16_EXACT
inline bool glm53f_pf_bf16_exact()
{
    static const bool v = *detail::flag(173, 0);
    return v;
}

/// GR8_GLM53F_PF_DRAFT_FULL
inline int64_t glm53f_pf_draft_full()
{
    static const int64_t v = *detail::int_(174, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_F16GEMM
inline bool glm53f_pf_dsa_f16gemm()
{
    static const bool v = *detail::flag(175, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_FULL_GV
inline bool glm53f_pf_dsa_full_gv()
{
    static const bool v = *detail::flag(176, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_LT
inline bool glm53f_pf_dsa_lt()
{
    static const bool v = *detail::flag(177, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_PROJ_P1
inline bool glm53f_pf_dsa_proj_p1()
{
    static const bool v = *detail::flag(178, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_GZF
inline bool glm53f_pf_kda_gzf()
{
    static const bool v = *detail::flag(179, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_TAIL
inline int64_t glm53f_pf_kda_tail()
{
    static const int64_t v = *detail::int_(180, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_TAIL_CHECK
inline bool glm53f_pf_kda_tail_check()
{
    static const bool v = *detail::flag(181, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_WS
inline bool glm53f_pf_kda_ws()
{
    static const bool v = *detail::flag(182, 0);
    return v;
}

/// GR8_GLM53F_PF_KP_TC
inline int64_t glm53f_pf_kp_tc()
{
    static const int64_t v = *detail::int_(183, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_FUSE
inline int64_t glm53f_pf_mhc_fuse()
{
    static const int64_t v = *detail::int_(184, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_PK
inline bool glm53f_pf_mhc_pk()
{
    static const bool v = *detail::flag(185, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_RR
inline bool glm53f_pf_mhc_rr()
{
    static const bool v = *detail::flag(186, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_STAGE
inline bool glm53f_pf_mhc_stage()
{
    static const bool v = *detail::flag(187, 0);
    return v;
}

/// GR8_GLM53F_PF_MLA_PROJ2
inline bool glm53f_pf_mla_proj2()
{
    static const bool v = *detail::flag(188, 0);
    return v;
}

/// GR8_GLM53F_PF_NOWIDE
inline bool glm53f_pf_nowide()
{
    static const bool v = *detail::flag(189, 0);
    return v;
}

/// GR8_GLM53F_PF_POOL_CAPTURE
inline const std::optional<std::string>& glm53f_pf_pool_capture()
{
    static const std::optional<std::string> v = detail::text(190, 0);
    return v;
}

/// GR8_GLM53F_PF_POOL_OWNER
inline std::optional<int64_t> glm53f_pf_pool_owner()
{
    static const std::optional<int64_t> v = detail::int_(191, 0);
    return v;
}

/// GR8_GLM53F_PF_PREWARM
inline std::optional<bool> glm53f_pf_prewarm()
{
    static const std::optional<bool> v = detail::flag(192, 0);
    return v;
}

/// GR8_GLM53F_PF_PUT_CTA
inline int64_t glm53f_pf_put_cta()
{
    static const int64_t v = *detail::int_(193, 0);
    return v;
}

/// GR8_GLM53F_PF_ROUTER1
inline bool glm53f_pf_router1()
{
    static const bool v = *detail::flag(194, 0);
    return v;
}

/// GR8_GLM53F_PF_SHFP8
inline std::optional<bool> glm53f_pf_shfp8()
{
    static const std::optional<bool> v = detail::flag(195, 0);
    return v;
}

/// GR8_GLM53F_PF_SP
inline bool glm53f_pf_sp()
{
    static const bool v = *detail::flag(196, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA
inline bool glm53f_pf_sparse_mla()
{
    static const bool v = *detail::flag(197, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA_512
inline bool glm53f_pf_sparse_mla_512()
{
    static const bool v = *detail::flag(198, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA_H16C
inline bool glm53f_pf_sparse_mla_h16c()
{
    static const bool v = *detail::flag(199, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA_V2
inline std::optional<bool> glm53f_pf_sparse_mla_v2()
{
    static const std::optional<bool> v = detail::flag(200, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_AGD
inline bool glm53f_pf_sp_agd()
{
    static const bool v = *detail::flag(201, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_AGK
inline bool glm53f_pf_sp_agk()
{
    static const bool v = *detail::flag(202, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_AGKA
inline bool glm53f_pf_sp_agka()
{
    static const bool v = *detail::flag(203, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_ATTD
inline bool glm53f_pf_sp_attd()
{
    static const bool v = *detail::flag(204, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_FIND
inline bool glm53f_pf_sp_find()
{
    static const bool v = *detail::flag(205, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_PUT_CTA
inline int64_t glm53f_pf_sp_put_cta()
{
    static const int64_t v = *detail::int_(206, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_X16
inline bool glm53f_pf_sp_x16()
{
    static const bool v = *detail::flag(207, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_XFRAG
inline bool glm53f_pf_sp_xfrag()
{
    static const bool v = *detail::flag(208, 0);
    return v;
}

/// GR8_GLM53F_PF_STAGE_ASYNC
inline std::optional<bool> glm53f_pf_stage_async()
{
    static const std::optional<bool> v = detail::flag(209, 0);
    return v;
}

/// GR8_GLM53F_PF_XFRAG
inline bool glm53f_pf_xfrag()
{
    static const bool v = *detail::flag(210, 0);
    return v;
}

/// GR8_GLM53F_PF_XTRACE
inline bool glm53f_pf_xtrace()
{
    static const bool v = *detail::flag(211, 0);
    return v;
}

/// GR8_GLM53F_RESET_ALL
inline bool glm53f_reset_all()
{
    static const bool v = *detail::flag(212, 0);
    return v;
}

/// GR8_GLM53F_RESET_CHECK
inline bool glm53f_reset_check()
{
    static const bool v = *detail::flag(213, 0);
    return v;
}

/// GR8_GLM53F_ROUTER_MF
inline bool glm53f_router_mf()
{
    static const bool v = *detail::flag(214, 0);
    return v;
}

/// GR8_GLM53F_ROUTER_NW
inline int64_t glm53f_router_nw()
{
    static const int64_t v = *detail::int_(215, 0);
    return v;
}

/// GR8_GLM53F_S2_COST
inline bool glm53f_s2_cost()
{
    static const bool v = *detail::flag(216, 0);
    return v;
}

/// GR8_GLM53F_STATE_BUDGET_MIB
inline int64_t glm53f_state_budget_mib()
{
    static const int64_t v = *detail::int_(217, 0);
    return v;
}

/// GR8_GV_MULTI
inline bool gv_multi()
{
    static const bool v = *detail::flag(218, 0);
    return v;
}

/// GR8_HF_ENDPOINT
inline const std::optional<std::string>& hf_endpoint()
{
    static const std::optional<std::string> v = detail::text(219, 0);
    return v;
}

/// GR8_HOME
inline const std::optional<std::string>& home()
{
    static const std::optional<std::string> v = detail::text(220, 0);
    return v;
}

/// GR8_IDXQ_SPLIT
inline std::optional<bool> idxq_split()
{
    static const std::optional<bool> v = detail::flag(221, 0);
    return v;
}

/// GR8_IDX_GRP
inline int64_t idx_grp()
{
    static const int64_t v = *detail::int_(222, 0);
    return v;
}

/// GR8_IDX_VEC
inline bool idx_vec()
{
    static const bool v = *detail::flag(223, 0);
    return v;
}

/// GR8_IMAGE
inline const std::optional<std::string>& image()
{
    static const std::optional<std::string> v = detail::text(224, 0);
    return v;
}

/// GR8_IMG_VMM
inline int64_t img_vmm()
{
    static const int64_t v = *detail::int_(225, 0);
    return v;
}

/// GR8_K1_FUSE_POLICY
inline bool k1_fuse_policy()
{
    static const bool v = *detail::flag(226, 0);
    return v;
}

/// GR8_K1_SPLIT
inline const std::optional<std::string>& k1_split()
{
    static const std::optional<std::string> v = detail::text(227, 0);
    return v;
}

/// GR8_K1_SPLIT_M1
inline const std::optional<std::vector<int64_t>>& k1_split_m1()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(228, 0);
    return v;
}

/// GR8_K1_SPLIT_M2
inline const std::optional<std::vector<int64_t>>& k1_split_m2()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(229, 0);
    return v;
}

/// GR8_K1_SPLIT_M4
inline const std::optional<std::vector<int64_t>>& k1_split_m4()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(230, 0);
    return v;
}

/// GR8_K1_TLBW
inline int64_t k1_tlbw()
{
    static const int64_t v = *detail::int_(231, 0);
    return v;
}

/// GR8_KNOBS_LIB
inline const std::optional<std::string>& knobs_lib()
{
    static const std::optional<std::string> v = detail::text(232, 0);
    return v;
}

/// GR8_KVB
inline const std::string& kvb()
{
    static const std::string v = *detail::text(233, 0);
    return v;
}

/// GR8_KVMAN_TRANSPORT
inline const std::string& kvman_transport()
{
    static const std::string v = *detail::text(234, 0);
    return v;
}

/// GR8_KV_DRAIN_DELAY
inline const std::optional<std::string>& kv_drain_delay()
{
    static const std::optional<std::string> v = detail::text(235, 0);
    return v;
}

/// GR8_KV_SHARE_WHEN
inline const std::string& kv_share_when()
{
    static const std::string v = *detail::text(236, 0);
    return v;
}

/// GR8_L2PF
inline int64_t l2pf()
{
    static const int64_t v = *detail::int_(237, 0);
    return v;
}

/// GR8_L2PF_GBPS
inline double l2pf_gbps()
{
    static const double v = *detail::float_(238, 0);
    return v;
}

/// GR8_L2PF_MB
inline double l2pf_mb()
{
    static const double v = *detail::float_(239, 0);
    return v;
}

/// GR8_L2PF_OPMB
inline double l2pf_opmb()
{
    static const double v = *detail::float_(240, 0);
    return v;
}

/// GR8_L2PF_T
inline int64_t l2pf_t()
{
    static const int64_t v = *detail::int_(241, 0);
    return v;
}

/// GR8_L2PF_US
inline const std::vector<double>& l2pf_us()
{
    static const std::vector<double> v = *detail::floats(242, 0);
    return v;
}

/// GR8_L2PF_WAVE
inline int64_t l2pf_wave()
{
    static const int64_t v = *detail::int_(243, 0);
    return v;
}

/// GR8_LOADER
inline const std::optional<std::string>& loader()
{
    static const std::optional<std::string> v = detail::text(244, 0);
    return v;
}

/// GR8_LOAD_CHUNK_MB
inline int64_t load_chunk_mb()
{
    static const int64_t v = *detail::int_(245, 0);
    return v;
}

/// GR8_LOAD_MODE
inline const std::string& load_mode()
{
    static const std::string v = *detail::text(246, 0);
    return v;
}

/// GR8_LOAD_THREADS
inline int64_t load_threads()
{
    static const int64_t v = *detail::int_(247, 0);
    return v;
}

/// GR8_LOGIT_DUMP
inline const std::optional<std::string>& logit_dump()
{
    static const std::optional<std::string> v = detail::text(248, 0);
    return v;
}

/// GR8_LOGIT_DUMP_FROM
inline int64_t logit_dump_from()
{
    static const int64_t v = *detail::int_(249, 0);
    return v;
}

/// GR8_MEM_MIN
inline double mem_min()
{
    static const double v = *detail::float_(250, 0);
    return v;
}

/// GR8_MHC_BF16
inline int64_t mhc_bf16()
{
    static const int64_t v = *detail::int_(251, 0);
    return v;
}

/// GR8_MODEL
inline const std::optional<std::string>& model()
{
    static const std::optional<std::string> v = detail::text(252, 0);
    return v;
}

/// GR8_MOE_TOPK
inline int64_t moe_topk()
{
    static const int64_t v = *detail::int_(253, 0);
    return v;
}

/// GR8_MTP_KSTOP
inline std::optional<int64_t> mtp_kstop()
{
    static const std::optional<int64_t> v = detail::int_(254, 0);
    return v;
}

/// GR8_MTP_KSTOPB
inline bool mtp_kstopb()
{
    static const bool v = *detail::flag(255, 0);
    return v;
}

/// GR8_MTP_KSTOPB_EXPLORE
inline int64_t mtp_kstopb_explore()
{
    static const int64_t v = *detail::int_(256, 0);
    return v;
}

/// GR8_MTP_KSTOPB_KMIN
inline int64_t mtp_kstopb_kmin()
{
    static const int64_t v = *detail::int_(257, 0);
    return v;
}

/// GR8_MTP_KSTOPB_TAU
inline const std::vector<double>& mtp_kstopb_tau()
{
    static const std::vector<double> v = *detail::floats(258, 0);
    return v;
}

/// GR8_MTP_KSTOPB_TAU_S<n>
inline std::optional<std::vector<double>> mtp_kstopb_tau_s(uint32_t n) { return detail::floats(259, n); }

/// GR8_MTP_KSTOPB_WIDTHS
inline const std::optional<std::vector<int64_t>>& mtp_kstopb_widths()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(260, 0);
    return v;
}

/// GR8_MTP_KSTOP_KMAX
inline std::optional<int64_t> mtp_kstop_kmax()
{
    static const std::optional<int64_t> v = detail::int_(261, 0);
    return v;
}

/// GR8_MTP_KSTOP_SEED3
inline double mtp_kstop_seed3()
{
    static const double v = *detail::float_(262, 0);
    return v;
}

/// GR8_MTP_KSTOP_TAU
inline const std::vector<double>& mtp_kstop_tau()
{
    static const std::vector<double> v = *detail::floats(263, 0);
    return v;
}

/// GR8_MTP_L2PF
inline int64_t mtp_l2pf()
{
    static const int64_t v = *detail::int_(264, 0);
    return v;
}

/// GR8_MTP_L2PF_US
inline double mtp_l2pf_us()
{
    static const double v = *detail::float_(265, 0);
    return v;
}

/// GR8_MTP_LASTROW
inline bool mtp_lastrow()
{
    static const bool v = *detail::flag(266, 0);
    return v;
}

/// GR8_MTP_LPLOG
inline int64_t mtp_lplog()
{
    static const int64_t v = *detail::int_(267, 0);
    return v;
}

/// GR8_MTP_SPLIT
inline std::optional<bool> mtp_split()
{
    static const std::optional<bool> v = detail::flag(268, 0);
    return v;
}

/// GR8_NGRAM
inline const std::optional<std::string>& ngram()
{
    static const std::optional<std::string> v = detail::text(269, 0);
    return v;
}

/// GR8_NGRAM_KMAX
inline std::optional<int64_t> ngram_kmax()
{
    static const std::optional<int64_t> v = detail::int_(270, 0);
    return v;
}

/// GR8_NGRAM_KMIN
inline int64_t ngram_kmin()
{
    static const int64_t v = *detail::int_(271, 0);
    return v;
}

/// GR8_NGRAM_LOG
inline bool ngram_log()
{
    static const bool v = *detail::flag(272, 0);
    return v;
}

/// GR8_NGRAM_MARGIN
inline double ngram_margin()
{
    static const double v = *detail::float_(273, 0);
    return v;
}

/// GR8_NOWIDE2
inline bool nowide2()
{
    static const bool v = *detail::flag(274, 0);
    return v;
}

/// GR8_NR
inline const std::optional<std::string>& nr()
{
    static const std::optional<std::string> v = detail::text(275, 0);
    return v;
}

/// GR8_NVCC_CACHE
inline bool nvcc_cache()
{
    static const bool v = *detail::flag(276, 0);
    return v;
}

/// GR8_NVCC_CACHE_DIR
inline const std::optional<std::string>& nvcc_cache_dir()
{
    static const std::optional<std::string> v = detail::text(277, 0);
    return v;
}

/// GR8_NVCC_CACHE_LOG
inline const std::optional<std::string>& nvcc_cache_log()
{
    static const std::optional<std::string> v = detail::text(278, 0);
    return v;
}

/// GR8_PF_ADUMP
inline const std::optional<std::string>& pf_adump()
{
    static const std::optional<std::string> v = detail::text(279, 0);
    return v;
}

/// GR8_PF_AOP
inline std::optional<bool> pf_aop()
{
    static const std::optional<bool> v = detail::flag(280, 0);
    return v;
}

/// GR8_PF_AOP_MMIN
inline int64_t pf_aop_mmin()
{
    static const int64_t v = *detail::int_(281, 0);
    return v;
}

/// GR8_PF_ATTN_ROWS
inline std::optional<int64_t> pf_attn_rows()
{
    static const std::optional<int64_t> v = detail::int_(282, 0);
    return v;
}

/// GR8_PF_BLAS_ROWS
inline std::optional<int64_t> pf_blas_rows()
{
    static const std::optional<int64_t> v = detail::int_(283, 0);
    return v;
}

/// GR8_PF_DENSE
inline int64_t pf_dense()
{
    static const int64_t v = *detail::int_(284, 0);
    return v;
}

/// GR8_PF_DRAFT_SHARE
inline std::optional<int64_t> pf_draft_share()
{
    static const std::optional<int64_t> v = detail::int_(285, 0);
    return v;
}

/// GR8_PF_DTR8
inline bool pf_dtr8()
{
    static const bool v = *detail::flag(286, 0);
    return v;
}

/// GR8_PF_EVENTS
inline bool pf_events()
{
    static const bool v = *detail::flag(287, 0);
    return v;
}

/// GR8_PF_F12
inline std::optional<bool> pf_f12()
{
    static const std::optional<bool> v = detail::flag(288, 0);
    return v;
}

/// GR8_PF_FC2ONE
inline bool pf_fc2one()
{
    static const bool v = *detail::flag(289, 0);
    return v;
}

/// GR8_PF_FC2V2
inline bool pf_fc2v2()
{
    static const bool v = *detail::flag(290, 0);
    return v;
}

/// GR8_PF_GEMMF32
inline bool pf_gemmf32()
{
    static const bool v = *detail::flag(291, 0);
    return v;
}

/// GR8_PF_GEMMG
inline int64_t pf_gemmg()
{
    static const int64_t v = *detail::int_(292, 0);
    return v;
}

/// GR8_PF_HASH
inline bool pf_hash()
{
    static const bool v = *detail::flag(293, 0);
    return v;
}

/// GR8_PF_HASH_GPU
inline bool pf_hash_gpu()
{
    static const bool v = *detail::flag(294, 0);
    return v;
}

/// GR8_PF_HEAD
inline bool pf_head()
{
    static const bool v = *detail::flag(295, 0);
    return v;
}

/// GR8_PF_HX
inline bool pf_hx()
{
    static const bool v = *detail::flag(296, 0);
    return v;
}

/// GR8_PF_HXF
inline bool pf_hxf()
{
    static const bool v = *detail::flag(297, 0);
    return v;
}

/// GR8_PF_IDDUMP
inline const std::optional<std::string>& pf_iddump()
{
    static const std::optional<std::string> v = detail::text(298, 0);
    return v;
}

/// GR8_PF_ISGRT_RG
inline int64_t pf_isgrt_rg()
{
    static const int64_t v = *detail::int_(299, 0);
    return v;
}

/// GR8_PF_MOE8
inline bool pf_moe8()
{
    static const bool v = *detail::flag(300, 0);
    return v;
}

/// GR8_PF_OWN
inline int64_t pf_own()
{
    static const int64_t v = *detail::int_(301, 0);
    return v;
}

/// GR8_PF_OWNPERM
inline bool pf_ownperm()
{
    static const bool v = *detail::flag(302, 0);
    return v;
}

/// GR8_PF_PDUMP
inline const std::optional<std::string>& pf_pdump()
{
    static const std::optional<std::string> v = detail::text(303, 0);
    return v;
}

/// GR8_PF_PDUMP_EVERY
inline int64_t pf_pdump_every()
{
    static const int64_t v = *detail::int_(304, 0);
    return v;
}

/// GR8_PF_PDUMP_LAYERS
inline const std::vector<int64_t>& pf_pdump_layers()
{
    static const std::vector<int64_t> v = *detail::ints(305, 0);
    return v;
}

/// GR8_PF_PROF
inline bool pf_prof()
{
    static const bool v = *detail::flag(306, 0);
    return v;
}

/// GR8_PF_Q8
inline bool pf_q8()
{
    static const bool v = *detail::flag(307, 0);
    return v;
}

/// GR8_PF_QO
inline const std::string& pf_qo()
{
    static const std::string v = *detail::text(308, 0);
    return v;
}

/// GR8_PF_QP_CTX
inline std::optional<int64_t> pf_qp_ctx()
{
    static const std::optional<int64_t> v = detail::int_(309, 0);
    return v;
}

/// GR8_PF_QP_EARLY
inline int64_t pf_qp_early()
{
    static const int64_t v = *detail::int_(310, 0);
    return v;
}

/// GR8_PF_QP_LAYERS
inline const std::optional<std::string>& pf_qp_layers()
{
    static const std::optional<std::string> v = detail::text(311, 0);
    return v;
}

/// GR8_PF_QP_MIN
inline int64_t pf_qp_min()
{
    static const int64_t v = *detail::int_(312, 0);
    return v;
}

/// GR8_PF_QP_ROLL
inline int64_t pf_qp_roll()
{
    static const int64_t v = *detail::int_(313, 0);
    return v;
}

/// GR8_PF_QP_ROLL_SER
inline bool pf_qp_roll_ser()
{
    static const bool v = *detail::flag(314, 0);
    return v;
}

/// GR8_PF_QP_WARM
inline bool pf_qp_warm()
{
    static const bool v = *detail::flag(315, 0);
    return v;
}

/// GR8_PF_QP_WIRE_E
inline std::optional<int64_t> pf_qp_wire_e()
{
    static const std::optional<int64_t> v = detail::int_(316, 0);
    return v;
}

/// GR8_PF_RCDUMP
inline const std::optional<std::string>& pf_rcdump()
{
    static const std::optional<std::string> v = detail::text(317, 0);
    return v;
}

/// GR8_PF_ROUTER_SPLIT
inline bool pf_router_split()
{
    static const bool v = *detail::flag(318, 0);
    return v;
}

/// GR8_PF_RS
inline bool pf_rs()
{
    static const bool v = *detail::flag(319, 0);
    return v;
}

/// GR8_PF_RSDIRECT
inline bool pf_rsdirect()
{
    static const bool v = *detail::flag(320, 0);
    return v;
}

/// GR8_PF_RSHASH
inline bool pf_rshash()
{
    static const bool v = *detail::flag(321, 0);
    return v;
}

/// GR8_PF_SELX_CTX
inline int64_t pf_selx_ctx()
{
    static const int64_t v = *detail::int_(322, 0);
    return v;
}

/// GR8_PF_SHR
inline bool pf_shr()
{
    static const bool v = *detail::flag(323, 0);
    return v;
}

/// GR8_PF_STAGGER
inline int64_t pf_stagger()
{
    static const int64_t v = *detail::int_(324, 0);
    return v;
}

/// GR8_PF_STREAMS
inline int64_t pf_streams()
{
    static const int64_t v = *detail::int_(325, 0);
    return v;
}

/// GR8_PF_STRIPE
inline std::optional<bool> pf_stripe()
{
    static const std::optional<bool> v = detail::flag(326, 0);
    return v;
}

/// GR8_PF_STRIPE_MIN
inline int64_t pf_stripe_min()
{
    static const int64_t v = *detail::int_(327, 0);
    return v;
}

/// GR8_PF_TILE
inline int64_t pf_tile()
{
    static const int64_t v = *detail::int_(328, 0);
    return v;
}

/// GR8_PF_TR_BM
inline int64_t pf_tr_bm()
{
    static const int64_t v = *detail::int_(329, 0);
    return v;
}

/// GR8_PF_WM
inline bool pf_wm()
{
    static const bool v = *detail::flag(330, 0);
    return v;
}

/// GR8_PF_WM2
inline bool pf_wm2()
{
    static const bool v = *detail::flag(331, 0);
    return v;
}

/// GR8_PF_WM_MIN
inline int64_t pf_wm_min()
{
    static const int64_t v = *detail::int_(332, 0);
    return v;
}

/// GR8_POOL_LOG
inline bool pool_log()
{
    static const bool v = *detail::flag(333, 0);
    return v;
}

/// GR8_PROGRESS_ISOLATE
inline bool progress_isolate()
{
    static const bool v = *detail::flag(334, 0);
    return v;
}

/// GR8_PSEUDO_TOKENIZER
inline const std::optional<std::string>& pseudo_tokenizer()
{
    static const std::optional<std::string> v = detail::text(335, 0);
    return v;
}

/// GR8_QA_SPLIT
inline std::optional<int64_t> qa_split()
{
    static const std::optional<int64_t> v = detail::int_(336, 0);
    return v;
}

/// GR8_QA_SPLIT_CPU
inline std::optional<int64_t> qa_split_cpu()
{
    static const std::optional<int64_t> v = detail::int_(337, 0);
    return v;
}

/// GR8_QA_SPLIT_TPC
inline int64_t qa_split_tpc()
{
    static const int64_t v = *detail::int_(338, 0);
    return v;
}

/// GR8_QO_CACHE
inline const std::optional<std::string>& qo_cache()
{
    static const std::optional<std::string> v = detail::text(339, 0);
    return v;
}

/// GR8_QO_DUMP
inline bool qo_dump()
{
    static const bool v = *detail::flag(340, 0);
    return v;
}

/// GR8_QO_NANINJ
inline bool qo_naninj()
{
    static const bool v = *detail::flag(341, 0);
    return v;
}

/// GR8_QO_ROPECHK
inline bool qo_ropechk()
{
    static const bool v = *detail::flag(342, 0);
    return v;
}

/// GR8_REFUSAL_LINGER
inline std::optional<double> refusal_linger()
{
    static const std::optional<double> v = detail::float_(343, 0);
    return v;
}

/// GR8_RESID_DUMP
inline const std::optional<std::string>& resid_dump()
{
    static const std::optional<std::string> v = detail::text(344, 0);
    return v;
}

/// GR8_REUSE_ALIGN
inline std::optional<bool> reuse_align()
{
    static const std::optional<bool> v = detail::flag(345, 0);
    return v;
}

/// GR8_REUSE_HDUMP
inline const std::optional<std::string>& reuse_hdump()
{
    static const std::optional<std::string> v = detail::text(346, 0);
    return v;
}

/// GR8_REUSE_HDUMP_SEQ
inline const std::optional<std::vector<int64_t>>& reuse_hdump_seq()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(347, 0);
    return v;
}

/// GR8_REUSE_NOEND
inline std::optional<bool> reuse_noend()
{
    static const std::optional<bool> v = detail::flag(348, 0);
    return v;
}

/// GR8_REUSE_POISON
inline bool reuse_poison()
{
    static const bool v = *detail::flag(349, 0);
    return v;
}

/// GR8_REUSE_STRICT
inline bool reuse_strict()
{
    static const bool v = *detail::flag(350, 0);
    return v;
}

/// GR8_REUSE_ZSCRATCH
inline bool reuse_zscratch()
{
    static const bool v = *detail::flag(351, 0);
    return v;
}

/// GR8_ROUTER_TPC
inline int64_t router_tpc()
{
    static const int64_t v = *detail::int_(352, 0);
    return v;
}

/// GR8_ROUTE_LOG
inline const std::optional<std::string>& route_log()
{
    static const std::optional<std::string> v = detail::text(353, 0);
    return v;
}

/// GR8_ROUTE_OVL
inline int64_t route_ovl()
{
    static const int64_t v = *detail::int_(354, 0);
    return v;
}

/// GR8_RT_CHECK
inline bool rt_check()
{
    static const bool v = *detail::flag(355, 0);
    return v;
}

/// GR8_RT_FMT
inline const std::string& rt_fmt()
{
    static const std::string v = *detail::text(356, 0);
    return v;
}

/// GR8_RT_L
inline std::optional<int64_t> rt_l()
{
    static const std::optional<int64_t> v = detail::int_(357, 0);
    return v;
}

/// GR8_RT_LOCAL
inline int64_t rt_local()
{
    static const int64_t v = *detail::int_(358, 0);
    return v;
}

/// GR8_RT_PF_SLICE
inline bool rt_pf_slice()
{
    static const bool v = *detail::flag(359, 0);
    return v;
}

/// GR8_RT_PLACE
inline std::optional<int64_t> rt_place()
{
    static const std::optional<int64_t> v = detail::int_(360, 0);
    return v;
}

/// GR8_RT_POOL
inline std::optional<int64_t> rt_pool()
{
    static const std::optional<int64_t> v = detail::int_(361, 0);
    return v;
}

/// GR8_RT_SELSTAT
inline bool rt_selstat()
{
    static const bool v = *detail::flag(362, 0);
    return v;
}

/// GR8_RT_SIZE
inline std::optional<int64_t> rt_size()
{
    static const std::optional<int64_t> v = detail::int_(363, 0);
    return v;
}

/// GR8_RT_STORE
inline const std::optional<std::string>& rt_store()
{
    static const std::optional<std::string> v = detail::text(364, 0);
    return v;
}

/// GR8_S2_ADAPT
inline bool s2_adapt()
{
    static const bool v = *detail::flag(365, 0);
    return v;
}

/// GR8_S2_ADMIT_HOLD
inline bool s2_admit_hold()
{
    static const bool v = *detail::flag(366, 0);
    return v;
}

/// GR8_S2_BELOW
inline std::optional<int64_t> s2_below()
{
    static const std::optional<int64_t> v = detail::int_(367, 0);
    return v;
}

/// GR8_S2_BIND_DELAY
inline const std::optional<std::string>& s2_bind_delay()
{
    static const std::optional<std::string> v = detail::text(368, 0);
    return v;
}

/// GR8_S2_DECODE_PF
inline int64_t s2_decode_pf()
{
    static const int64_t v = *detail::int_(369, 0);
    return v;
}

/// GR8_S2_DECODE_ROUNDS
inline int64_t s2_decode_rounds()
{
    static const int64_t v = *detail::int_(370, 0);
    return v;
}

/// GR8_S2_DSEL
inline bool s2_dsel()
{
    static const bool v = *detail::flag(371, 0);
    return v;
}

/// GR8_S2_DSEL_DWELL
inline int64_t s2_dsel_dwell()
{
    static const int64_t v = *detail::int_(372, 0);
    return v;
}

/// GR8_S2_DSEL_MARGIN
inline double s2_dsel_margin()
{
    static const double v = *detail::float_(373, 0);
    return v;
}

/// GR8_S2_DSEL_MODEL
inline bool s2_dsel_model()
{
    static const bool v = *detail::flag(374, 0);
    return v;
}

/// GR8_S2_DSEL_PROBE
inline int64_t s2_dsel_probe()
{
    static const int64_t v = *detail::int_(375, 0);
    return v;
}

/// GR8_S2_DSEL_TABLE
inline const std::optional<std::string>& s2_dsel_table()
{
    static const std::optional<std::string> v = detail::text(376, 0);
    return v;
}

/// GR8_S2_EVENTLOG
inline bool s2_eventlog()
{
    static const bool v = *detail::flag(377, 0);
    return v;
}

/// GR8_S2_GROW
inline bool s2_grow()
{
    static const bool v = *detail::flag(378, 0);
    return v;
}

/// GR8_S2_NOTICE_MAX
inline int64_t s2_notice_max()
{
    static const int64_t v = *detail::int_(379, 0);
    return v;
}

/// GR8_S2_ROUNDLOG
inline bool s2_roundlog()
{
    static const bool v = *detail::flag(380, 0);
    return v;
}

/// GR8_S2_SHARE_MS
inline int64_t s2_share_ms()
{
    static const int64_t v = *detail::int_(381, 0);
    return v;
}

/// GR8_S2_TAIL_FIRST
inline bool s2_tail_first()
{
    static const bool v = *detail::flag(382, 0);
    return v;
}

/// GR8_S2_TIMELINE
inline int64_t s2_timeline()
{
    static const int64_t v = *detail::int_(383, 0);
    return v;
}

/// GR8_S2_TIMEOUT
inline int64_t s2_timeout()
{
    static const int64_t v = *detail::int_(384, 0);
    return v;
}

/// GR8_S2_TRACE
inline const std::optional<std::string>& s2_trace()
{
    static const std::optional<std::string> v = detail::text(385, 0);
    return v;
}

/// GR8_SPEC_DRAFT
inline const std::vector<double>& spec_draft()
{
    static const std::vector<double> v = *detail::floats(386, 0);
    return v;
}

/// GR8_SPEC_TABLE
inline const std::optional<std::vector<double>>& spec_table()
{
    static const std::optional<std::vector<double>> v = detail::floats(387, 0);
    return v;
}

/// GR8_STEP_TMR
inline bool step_tmr()
{
    static const bool v = *detail::flag(388, 0);
    return v;
}

/// GR8_TESTS
inline const std::string& tests()
{
    static const std::string v = *detail::text(389, 0);
    return v;
}

/// GR8_TEST_CP_PF_BREAK
inline const std::optional<std::string>& test_cp_pf_break()
{
    static const std::optional<std::string> v = detail::text(390, 0);
    return v;
}

/// GR8_TEST_CP_PF_NET
inline bool test_cp_pf_net()
{
    static const bool v = *detail::flag(391, 0);
    return v;
}

/// GR8_TEST_DS41_KERNELS_DIR
inline const std::optional<std::string>& test_ds41_kernels_dir()
{
    static const std::optional<std::string> v = detail::text(392, 0);
    return v;
}

/// GR8_TEST_DS41_M4_PFW
inline int64_t test_ds41_m4_pfw()
{
    static const int64_t v = *detail::int_(393, 0);
    return v;
}

/// GR8_TEST_DS41_MGATE_DIAG
inline bool test_ds41_mgate_diag()
{
    static const bool v = *detail::flag(394, 0);
    return v;
}

/// GR8_TEST_DS41_TOKENIZER_DIR
inline const std::optional<std::string>& test_ds41_tokenizer_dir()
{
    static const std::optional<std::string> v = detail::text(395, 0);
    return v;
}

/// GR8_TEST_FUZZ_ROUNDS
inline int64_t test_fuzz_rounds()
{
    static const int64_t v = *detail::int_(396, 0);
    return v;
}

/// GR8_TEST_IDX_DUMP
inline const std::optional<std::string>& test_idx_dump()
{
    static const std::optional<std::string> v = detail::text(397, 0);
    return v;
}

/// GR8_TEST_KVMAN_LEDGER
inline const std::string& test_kvman_ledger()
{
    static const std::string v = *detail::text(398, 0);
    return v;
}

/// GR8_TEST_KVSIM_ENTRIES
inline bool test_kvsim_entries()
{
    static const bool v = *detail::flag(399, 0);
    return v;
}

/// GR8_TEST_KV_HASH
inline bool test_kv_hash()
{
    static const bool v = *detail::flag(400, 0);
    return v;
}

/// GR8_TEST_KV_RANKS_CMP
inline bool test_kv_ranks_cmp()
{
    static const bool v = *detail::flag(401, 0);
    return v;
}

/// GR8_TEST_KV_TIER_CHECK
inline const std::optional<std::string>& test_kv_tier_check()
{
    static const std::optional<std::string> v = detail::text(402, 0);
    return v;
}

/// GR8_TEST_KV_TIER_CHECK_DROP
inline std::optional<int64_t> test_kv_tier_check_drop()
{
    static const std::optional<int64_t> v = detail::int_(403, 0);
    return v;
}

/// GR8_TEST_LEDGER_CRASH_AT
inline const std::optional<std::string>& test_ledger_crash_at()
{
    static const std::optional<std::string> v = detail::text(404, 0);
    return v;
}

/// GR8_TEST_MTP_LASTROW_CHECK
inline bool test_mtp_lastrow_check()
{
    static const bool v = *detail::flag(405, 0);
    return v;
}

/// GR8_TEST_PERTURB
inline double test_perturb()
{
    static const double v = *detail::float_(406, 0);
    return v;
}

/// GR8_TEST_PF_DRAFTDUMP
inline const std::optional<std::string>& test_pf_draftdump()
{
    static const std::optional<std::string> v = detail::text(407, 0);
    return v;
}

/// GR8_TEST_PF_HASH_POKE
inline const std::optional<std::vector<int64_t>>& test_pf_hash_poke()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(408, 0);
    return v;
}

/// GR8_TEST_PF_REPLAY_VERBOSE
inline bool test_pf_replay_verbose()
{
    static const bool v = *detail::flag(409, 0);
    return v;
}

/// GR8_TEST_PF_RESDUMP
inline const std::optional<std::string>& test_pf_resdump()
{
    static const std::optional<std::string> v = detail::text(410, 0);
    return v;
}

/// GR8_TEST_PF_TIMES
inline bool test_pf_times()
{
    static const bool v = *detail::flag(411, 0);
    return v;
}

/// GR8_TEST_ROW_LINES
inline bool test_row_lines()
{
    static const bool v = *detail::flag(412, 0);
    return v;
}

/// GR8_TEST_RT_ADOPT_SLOT
inline bool test_rt_adopt_slot()
{
    static const bool v = *detail::flag(413, 0);
    return v;
}

/// GR8_TEST_RT_POISON
inline bool test_rt_poison()
{
    static const bool v = *detail::flag(414, 0);
    return v;
}

/// GR8_TEST_S2_ATTN_PERSESSION
inline bool test_s2_attn_persession()
{
    static const bool v = *detail::flag(415, 0);
    return v;
}

/// GR8_TEST_S2_CLS
inline int64_t test_s2_cls()
{
    static const int64_t v = *detail::int_(416, 0);
    return v;
}

/// GR8_TEST_S2_GEN
inline const std::optional<std::vector<int64_t>>& test_s2_gen()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(417, 0);
    return v;
}

/// GR8_TEST_S2_GROW_POOL
inline std::optional<int64_t> test_s2_grow_pool()
{
    static const std::optional<int64_t> v = detail::int_(418, 0);
    return v;
}

/// GR8_TEST_S2_INSERT
inline const std::optional<std::string>& test_s2_insert()
{
    static const std::optional<std::string> v = detail::text(419, 0);
    return v;
}

/// GR8_TEST_S2_PAGES
inline const std::optional<std::vector<int64_t>>& test_s2_pages()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(420, 0);
    return v;
}

/// GR8_TEST_S2_PFBASE
inline const std::optional<std::vector<int64_t>>& test_s2_pfbase()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(421, 0);
    return v;
}

/// GR8_TEST_S2_PFNOFIX
inline bool test_s2_pfnofix()
{
    static const bool v = *detail::flag(422, 0);
    return v;
}

/// GR8_TEST_S2_PFPOISON
inline bool test_s2_pfpoison()
{
    static const bool v = *detail::flag(423, 0);
    return v;
}

/// GR8_TEST_S2_POOL
inline std::optional<int64_t> test_s2_pool()
{
    static const std::optional<int64_t> v = detail::int_(424, 0);
    return v;
}

/// GR8_TEST_S2_RELOC
inline bool test_s2_reloc()
{
    static const bool v = *detail::flag(425, 0);
    return v;
}

/// GR8_TEST_S2_RELOC_PAGES
inline std::optional<int64_t> test_s2_reloc_pages()
{
    static const std::optional<int64_t> v = detail::int_(426, 0);
    return v;
}

/// GR8_TEST_S2_ROWHASH
inline bool test_s2_rowhash()
{
    static const bool v = *detail::flag(427, 0);
    return v;
}

/// GR8_TEST_S2_RUNS
inline const std::optional<std::string>& test_s2_runs()
{
    static const std::optional<std::string> v = detail::text(428, 0);
    return v;
}

/// GR8_TEST_S2_SHRINK_AT
inline std::optional<int64_t> test_s2_shrink_at()
{
    static const std::optional<int64_t> v = detail::int_(429, 0);
    return v;
}

/// GR8_TEST_S2_SOLO1
inline bool test_s2_solo1()
{
    static const bool v = *detail::flag(430, 0);
    return v;
}

/// GR8_TEST_S2_XLIST
inline const std::optional<std::vector<int64_t>>& test_s2_xlist()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(431, 0);
    return v;
}

/// GR8_TEST_TOKENIZER_DIR
inline const std::optional<std::string>& test_tokenizer_dir()
{
    static const std::optional<std::string> v = detail::text(432, 0);
    return v;
}

/// GR8_TEST_ZERO_DELAY_MS
inline int64_t test_zero_delay_ms()
{
    static const int64_t v = *detail::int_(433, 0);
    return v;
}

/// TMPDIR
inline const std::string& tmpdir()
{
    static const std::string v = *detail::text(434, 0);
    return v;
}

/// GR8_TR3_W2ALL
inline bool tr3_w2all()
{
    static const bool v = *detail::flag(435, 0);
    return v;
}

/// GR8_V2_SHARED
inline const std::optional<std::string>& v2_shared()
{
    static const std::optional<std::string> v = detail::text(436, 0);
    return v;
}

/// GR8_WARM
inline int64_t warm()
{
    static const int64_t v = *detail::int_(437, 0);
    return v;
}

/// GR8_WEIGHT_CHECK
inline bool weight_check()
{
    static const bool v = *detail::flag(438, 0);
    return v;
}

/// GR8_WEIGHT_CHECK_FLIP
inline const std::optional<std::string>& weight_check_flip()
{
    static const std::optional<std::string> v = detail::text(439, 0);
    return v;
}

/// GR8_WIDTHS
inline const std::optional<std::string>& widths()
{
    static const std::optional<std::string> v = detail::text(440, 0);
    return v;
}

/// GR8_WORK_DIR
inline const std::optional<std::string>& work_dir()
{
    static const std::optional<std::string> v = detail::text(441, 0);
    return v;
}

/// GR8_XGRAMMAR
inline const std::optional<std::string>& xgrammar()
{
    static const std::optional<std::string> v = detail::text(442, 0);
    return v;
}

/// GR8_XGRAMMAR_LIB
inline const std::optional<std::string>& xgrammar_lib()
{
    static const std::optional<std::string> v = detail::text(443, 0);
    return v;
}

/// GR8_XTRACE
inline bool xtrace()
{
    static const bool v = *detail::flag(444, 0);
    return v;
}

/// GR8_XTRACE_OUT
inline const std::optional<std::string>& xtrace_out()
{
    static const std::optional<std::string> v = detail::text(445, 0);
    return v;
}

/// GR8_XTRACE_PF
inline bool xtrace_pf()
{
    static const bool v = *detail::flag(446, 0);
    return v;
}

/// GR8_XTRACE_QUIT
inline const std::optional<std::string>& xtrace_quit()
{
    static const std::optional<std::string> v = detail::text(447, 0);
    return v;
}

} // namespace knob
