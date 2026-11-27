#pragma once
// Written by `cargo run --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen` from the knobs/ YAML files:
// change the table, not this file. docs/ENV.md describes every knob.
//
// knob::name() returns the value of GR8_NAME: its default when unset, or an optional for a knob without one. The value
// is read once per process. A value the table refuses stops the process at its first read. knob::init() at the start
// of main prints every problem of the environment and returns how many are errors. The C API is in libgr8knobs.a.
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

/// The hash of the table this header was written from.
constexpr uint64_t TABLE_HASH = 0xadfd45ecdaa602cbull;

/// Prints the environment's problems (bad values, unknown GR8_ names, old G6_ names) to stderr and returns the number
/// of errors. The engine stops when it is not 0.
inline int init() { return gr8_knobs_init(TABLE_HASH); }

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

/// GR8_BUSY_GUARD
inline bool busy_guard()
{
    static const bool v = *detail::flag(19, 0);
    return v;
}

/// GR8_BUSY_IDLE_SECS
inline int64_t busy_idle_secs()
{
    static const int64_t v = *detail::int_(20, 0);
    return v;
}

/// GR8_BUSY_QUEUE_MAX
inline int64_t busy_queue_max()
{
    static const int64_t v = *detail::int_(21, 0);
    return v;
}

/// GR8_BUSY_RETRY_AFTER
inline int64_t busy_retry_after()
{
    static const int64_t v = *detail::int_(22, 0);
    return v;
}

/// GR8_CACHE
inline const std::optional<std::string>& cache()
{
    static const std::optional<std::string> v = detail::text(23, 0);
    return v;
}

/// GR8_CAP_TOTAL
inline std::optional<int64_t> cap_total()
{
    static const std::optional<int64_t> v = detail::int_(24, 0);
    return v;
}

/// GR8_CLUSTER
inline const std::optional<std::string>& cluster()
{
    static const std::optional<std::string> v = detail::text(25, 0);
    return v;
}

/// GR8_CONTAINER
inline std::optional<bool> container()
{
    static const std::optional<bool> v = detail::flag(26, 0);
    return v;
}

/// GR8_CP
inline std::optional<bool> cp()
{
    static const std::optional<bool> v = detail::flag(27, 0);
    return v;
}

/// GR8_CP_MMAX
inline std::optional<int64_t> cp_mmax()
{
    static const std::optional<int64_t> v = detail::int_(28, 0);
    return v;
}

/// GR8_CP_ONECHAIN
inline bool cp_onechain()
{
    static const bool v = *detail::flag(29, 0);
    return v;
}

/// GR8_CP_PF_HASH2
inline bool cp_pf_hash2()
{
    static const bool v = *detail::flag(30, 0);
    return v;
}

/// GR8_CP_PF_VERIFY
inline bool cp_pf_verify()
{
    static const bool v = *detail::flag(31, 0);
    return v;
}

/// GR8_CP_R
inline int64_t cp_r()
{
    static const int64_t v = *detail::int_(32, 0);
    return v;
}

/// GR8_CP_R2
inline std::optional<int64_t> cp_r2()
{
    static const std::optional<int64_t> v = detail::int_(33, 0);
    return v;
}

/// GR8_CP_R2_BIG
inline int64_t cp_r2_big()
{
    static const int64_t v = *detail::int_(34, 0);
    return v;
}

/// GR8_CP_S3
inline bool cp_s3()
{
    static const bool v = *detail::flag(35, 0);
    return v;
}

/// GR8_CP_STEP_ROWS
inline int64_t cp_step_rows()
{
    static const int64_t v = *detail::int_(36, 0);
    return v;
}

/// GR8_CP_STEP_SEG
inline bool cp_step_seg()
{
    static const bool v = *detail::flag(37, 0);
    return v;
}

/// GR8_DEC_BK
inline int64_t dec_bk()
{
    static const int64_t v = *detail::int_(38, 0);
    return v;
}

/// GR8_DEC_BK_MAX
inline int64_t dec_bk_max()
{
    static const int64_t v = *detail::int_(39, 0);
    return v;
}

/// GR8_DEC_DENSE_OPT
inline bool dec_dense_opt()
{
    static const bool v = *detail::flag(40, 0);
    return v;
}

/// GR8_DEC_EVPROF
inline int64_t dec_evprof()
{
    static const int64_t v = *detail::int_(41, 0);
    return v;
}

/// GR8_DEC_PROF
inline bool dec_prof()
{
    static const bool v = *detail::flag(42, 0);
    return v;
}

/// GR8_DEC_SHARED_SCRATCH
inline bool dec_shared_scratch()
{
    static const bool v = *detail::flag(43, 0);
    return v;
}

/// GR8_DEC_SPLIT
inline int64_t dec_split()
{
    static const int64_t v = *detail::int_(44, 0);
    return v;
}

/// GR8_DFL_BODY_I4
inline bool dfl_body_i4()
{
    static const bool v = *detail::flag(45, 0);
    return v;
}

/// GR8_DFL_DEDUP
inline bool dfl_dedup()
{
    static const bool v = *detail::flag(46, 0);
    return v;
}

/// GR8_DFL_DUMP
inline const std::optional<std::string>& dfl_dump()
{
    static const std::optional<std::string> v = detail::text(47, 0);
    return v;
}

/// GR8_DFL_HEAD_I4
inline bool dfl_head_i4()
{
    static const bool v = *detail::flag(48, 0);
    return v;
}

/// GR8_DFL_PFCTX
inline bool dfl_pfctx()
{
    static const bool v = *detail::flag(49, 0);
    return v;
}

/// GR8_DFL_REUSE
inline std::optional<int64_t> dfl_reuse()
{
    static const std::optional<int64_t> v = detail::int_(50, 0);
    return v;
}

/// GR8_DFL_REUSE_MIN
inline int64_t dfl_reuse_min()
{
    static const int64_t v = *detail::int_(51, 0);
    return v;
}

/// GR8_DFL_SEL_ALPHA
inline double dfl_sel_alpha()
{
    static const double v = *detail::float_(52, 0);
    return v;
}

/// GR8_DFL_VERIFY_CAP
inline double dfl_verify_cap()
{
    static const double v = *detail::float_(53, 0);
    return v;
}

/// GR8_DISPRAM
inline bool dispram()
{
    static const bool v = *detail::flag(54, 0);
    return v;
}

/// DISPRAM_DISABLE
inline bool dispram_disable()
{
    static const bool v = *detail::flag(55, 0);
    return v;
}

/// DISPRAM_KEY_FILE
inline const std::optional<std::string>& dispram_key_file()
{
    static const std::optional<std::string> v = detail::text(56, 0);
    return v;
}

/// GR8_DISPRAM_MIN_BYTES
inline int64_t dispram_min_bytes()
{
    static const int64_t v = *detail::int_(57, 0);
    return v;
}

/// DISPRAM_SOCKET
inline const std::string& dispram_socket()
{
    static const std::string v = *detail::text(58, 0);
    return v;
}

/// GR8_DRAFT_HEAD2
inline int64_t draft_head2()
{
    static const int64_t v = *detail::int_(59, 0);
    return v;
}

/// GR8_DRAFT_HEAD_I4
inline bool draft_head_i4()
{
    static const bool v = *detail::flag(60, 0);
    return v;
}

/// GR8_DRAFT_HEAD_KS
inline int64_t draft_head_ks()
{
    static const int64_t v = *detail::int_(61, 0);
    return v;
}

/// GR8_DRAFT_HEAD_TPC
inline int64_t draft_head_tpc()
{
    static const int64_t v = *detail::int_(62, 0);
    return v;
}

/// GR8_DRAFT_IN_DUMP
inline const std::optional<std::string>& draft_in_dump()
{
    static const std::optional<std::string> v = detail::text(63, 0);
    return v;
}

/// GR8_DRAFT_LOGIT_DUMP
inline const std::optional<std::string>& draft_logit_dump()
{
    static const std::optional<std::string> v = detail::text(64, 0);
    return v;
}

/// GR8_DRAFT_PAR
inline std::optional<int64_t> draft_par()
{
    static const std::optional<int64_t> v = detail::int_(65, 0);
    return v;
}

/// GR8_DS41_ADMIT_KEEP_SCRATCH
inline bool ds41_admit_keep_scratch()
{
    static const bool v = *detail::flag(66, 0);
    return v;
}

/// GR8_DS41_BORROW_TOPK
inline const std::optional<std::string>& ds41_borrow_topk()
{
    static const std::optional<std::string> v = detail::text(67, 0);
    return v;
}

/// GR8_DS41_GV_QUAD
inline bool ds41_gv_quad()
{
    static const bool v = *detail::flag(68, 0);
    return v;
}

/// GR8_DS41_HEAD_MX
inline bool ds41_head_mx()
{
    static const bool v = *detail::flag(69, 0);
    return v;
}

/// GR8_DS41_IDX_DENSE_MAXK
inline int64_t ds41_idx_dense_maxk()
{
    static const int64_t v = *detail::int_(70, 0);
    return v;
}

/// GR8_DS41_IDX_LOGITS
inline const std::string& ds41_idx_logits()
{
    static const std::string v = *detail::text(71, 0);
    return v;
}

/// GR8_DS41_IDX_MAXK
inline int64_t ds41_idx_maxk()
{
    static const int64_t v = *detail::int_(72, 0);
    return v;
}

/// GR8_DS41_IDX_PIPE_B
inline bool ds41_idx_pipe_b()
{
    static const bool v = *detail::flag(73, 0);
    return v;
}

/// GR8_DS41_IDX_PIPE_QR
inline bool ds41_idx_pipe_qr()
{
    static const bool v = *detail::flag(74, 0);
    return v;
}

/// GR8_DS41_IDX_PW
inline std::optional<int64_t> ds41_idx_pw()
{
    static const std::optional<int64_t> v = detail::int_(75, 0);
    return v;
}

/// GR8_DS41_IDX_PWK
inline int64_t ds41_idx_pwk()
{
    static const int64_t v = *detail::int_(76, 0);
    return v;
}

/// GR8_DS41_IDX_Q_TRIG
inline bool ds41_idx_q_trig()
{
    static const bool v = *detail::flag(77, 0);
    return v;
}

/// GR8_DS41_IDX_REUSE
inline bool ds41_idx_reuse()
{
    static const bool v = *detail::flag(78, 0);
    return v;
}

/// GR8_DS41_IDX_TOPK
inline const std::string& ds41_idx_topk()
{
    static const std::string v = *detail::text(79, 0);
    return v;
}

/// GR8_DS41_IDX_WQB_TC
inline int64_t ds41_idx_wqb_tc()
{
    static const int64_t v = *detail::int_(80, 0);
    return v;
}

/// GR8_DS41_PUB16
inline bool ds41_pub16()
{
    static const bool v = *detail::flag(81, 0);
    return v;
}

/// GR8_DS41_VISION
inline bool ds41_vision()
{
    static const bool v = *detail::flag(82, 0);
    return v;
}

/// GR8_DS_ATTN_CPA
inline bool ds_attn_cpa()
{
    static const bool v = *detail::flag(83, 0);
    return v;
}

/// GR8_DS_ATTN_GV
inline bool ds_attn_gv()
{
    static const bool v = *detail::flag(84, 0);
    return v;
}

/// GR8_DS_ATTN_GV_KS
inline int64_t ds_attn_gv_ks()
{
    static const int64_t v = *detail::int_(85, 0);
    return v;
}

/// GR8_DS_ATTN_GV_TPC
inline int64_t ds_attn_gv_tpc()
{
    static const int64_t v = *detail::int_(86, 0);
    return v;
}

/// GR8_DS_ATTN_PW
inline int64_t ds_attn_pw()
{
    static const int64_t v = *detail::int_(87, 0);
    return v;
}

/// GR8_DS_ATTN_SPLIT
inline bool ds_attn_split()
{
    static const bool v = *detail::flag(88, 0);
    return v;
}

/// GR8_DS_ATTN_TC
inline bool ds_attn_tc()
{
    static const bool v = *detail::flag(89, 0);
    return v;
}

/// GR8_DS_ATTN_TC_MF
inline int64_t ds_attn_tc_mf()
{
    static const int64_t v = *detail::int_(90, 0);
    return v;
}

/// GR8_DS_ATTN_TC_TILE
inline int64_t ds_attn_tc_tile()
{
    static const int64_t v = *detail::int_(91, 0);
    return v;
}

/// GR8_DS_BLOCK
inline const std::optional<std::string>& ds_block()
{
    static const std::optional<std::string> v = detail::text(92, 0);
    return v;
}

/// GR8_DS_C2_TRACE
inline bool ds_c2_trace()
{
    static const bool v = *detail::flag(93, 0);
    return v;
}

/// GR8_DS_C8PACK
inline bool ds_c8pack()
{
    static const bool v = *detail::flag(94, 0);
    return v;
}

/// GR8_DS_CACHE_DEDUP
inline int64_t ds_cache_dedup()
{
    static const int64_t v = *detail::int_(95, 0);
    return v;
}

/// GR8_DS_CACHE_DEFER_REPLACE
inline bool ds_cache_defer_replace()
{
    static const bool v = *detail::flag(96, 0);
    return v;
}

/// GR8_DS_CACHE_MIB
inline std::optional<int64_t> ds_cache_mib()
{
    static const std::optional<int64_t> v = detail::int_(97, 0);
    return v;
}

/// GR8_DS_CACHE_REPLACE
inline bool ds_cache_replace()
{
    static const bool v = *detail::flag(98, 0);
    return v;
}

/// GR8_DS_COMP_FP4
inline bool ds_comp_fp4()
{
    static const bool v = *detail::flag(99, 0);
    return v;
}

/// GR8_DS_COMP_FP8
inline bool ds_comp_fp8()
{
    static const bool v = *detail::flag(100, 0);
    return v;
}

/// GR8_DS_CONF_TEMP
inline const std::optional<std::string>& ds_conf_temp()
{
    static const std::optional<std::string> v = detail::text(101, 0);
    return v;
}

/// GR8_DS_DEC_STAGE
inline bool ds_dec_stage()
{
    static const bool v = *detail::flag(102, 0);
    return v;
}

/// GR8_DS_DRAFT_LOG
inline bool ds_draft_log()
{
    static const bool v = *detail::flag(103, 0);
    return v;
}

/// GR8_DS_DSP_GV
inline bool ds_dsp_gv()
{
    static const bool v = *detail::flag(104, 0);
    return v;
}

/// GR8_DS_DSP_TC
inline bool ds_dsp_tc()
{
    static const bool v = *detail::flag(105, 0);
    return v;
}

/// GR8_DS_ENGRAM
inline bool ds_engram()
{
    static const bool v = *detail::flag(106, 0);
    return v;
}

/// GR8_DS_ENGRAM_CACHE_MIB
inline int64_t ds_engram_cache_mib()
{
    static const int64_t v = *detail::int_(107, 0);
    return v;
}

/// GR8_DS_ENGRAM_COL_SHARD
inline bool ds_engram_col_shard()
{
    static const bool v = *detail::flag(108, 0);
    return v;
}

/// GR8_DS_ENGRAM_IO_THREADS
inline int64_t ds_engram_io_threads()
{
    static const int64_t v = *detail::int_(109, 0);
    return v;
}

/// GR8_DS_ENGRAM_IO_URING
inline int64_t ds_engram_io_uring()
{
    static const int64_t v = *detail::int_(110, 0);
    return v;
}

/// GR8_DS_ENGRAM_KEEPALIVE_MS
inline int64_t ds_engram_keepalive_ms()
{
    static const int64_t v = *detail::int_(111, 0);
    return v;
}

/// GR8_DS_ENGRAM_LOOKAHEAD
inline bool ds_engram_lookahead()
{
    static const bool v = *detail::flag(112, 0);
    return v;
}

/// GR8_DS_ENGRAM_MTP
inline bool ds_engram_mtp()
{
    static const bool v = *detail::flag(113, 0);
    return v;
}

/// GR8_DS_ENGRAM_PREFIX
inline const std::optional<std::string>& ds_engram_prefix()
{
    static const std::optional<std::string> v = detail::text(114, 0);
    return v;
}

/// GR8_DS_ENGRAM_VERBOSE
inline bool ds_engram_verbose()
{
    static const bool v = *detail::flag(115, 0);
    return v;
}

/// GR8_DS_ENGRAM_WAYS
inline int64_t ds_engram_ways()
{
    static const int64_t v = *detail::int_(116, 0);
    return v;
}

/// GR8_DS_FEED_DIR
inline const std::optional<std::string>& ds_feed_dir()
{
    static const std::optional<std::string> v = detail::text(117, 0);
    return v;
}

/// GR8_DS_GATE_RO
inline bool ds_gate_ro()
{
    static const bool v = *detail::flag(118, 0);
    return v;
}

/// GR8_DS_MANIFEST
inline const std::optional<std::string>& ds_manifest()
{
    static const std::optional<std::string> v = detail::text(119, 0);
    return v;
}

/// GR8_DS_MHC_DECOMP
inline bool ds_mhc_decomp()
{
    static const bool v = *detail::flag(120, 0);
    return v;
}

/// GR8_DS_MHC_FUSE
inline bool ds_mhc_fuse()
{
    static const bool v = *detail::flag(121, 0);
    return v;
}

/// GR8_DS_MHC_ROW_SHARD
inline bool ds_mhc_row_shard()
{
    static const bool v = *detail::flag(122, 0);
    return v;
}

/// GR8_DS_MHC_Y_MXFP8
inline bool ds_mhc_y_mxfp8()
{
    static const bool v = *detail::flag(123, 0);
    return v;
}

/// GR8_DS_MOE_ELOG
inline bool ds_moe_elog()
{
    static const bool v = *detail::flag(124, 0);
    return v;
}

/// GR8_DS_MOE_FC1BSTG
inline bool ds_moe_fc1bstg()
{
    static const bool v = *detail::flag(125, 0);
    return v;
}

/// GR8_DS_MOE_FC2STG
inline bool ds_moe_fc2stg()
{
    static const bool v = *detail::flag(126, 0);
    return v;
}

/// GR8_DS_MOE_L2HINT
inline int64_t ds_moe_l2hint()
{
    static const int64_t v = *detail::int_(127, 0);
    return v;
}

/// GR8_DS_MOE_PDL
inline bool ds_moe_pdl()
{
    static const bool v = *detail::flag(128, 0);
    return v;
}

/// GR8_DS_MOE_PW_BOUND
inline bool ds_moe_pw_bound()
{
    static const bool v = *detail::flag(129, 0);
    return v;
}

/// GR8_DS_MOE_SH_1PASS
inline bool ds_moe_sh_1pass()
{
    static const bool v = *detail::flag(130, 0);
    return v;
}

/// GR8_DS_MOE_TC
inline bool ds_moe_tc()
{
    static const bool v = *detail::flag(131, 0);
    return v;
}

/// GR8_DS_MOE_TC_OS2
inline int64_t ds_moe_tc_os2()
{
    static const int64_t v = *detail::int_(132, 0);
    return v;
}

/// GR8_DS_MOE_TOPK
inline int64_t ds_moe_topk()
{
    static const int64_t v = *detail::int_(133, 0);
    return v;
}

/// GR8_DS_MOE_TOPK_DECODE
inline std::optional<int64_t> ds_moe_topk_decode()
{
    static const std::optional<int64_t> v = detail::int_(134, 0);
    return v;
}

/// GR8_DS_MTP
inline const std::optional<std::string>& ds_mtp()
{
    static const std::optional<std::string> v = detail::text(135, 0);
    return v;
}

/// GR8_DS_PARK
inline bool ds_park()
{
    static const bool v = *detail::flag(136, 0);
    return v;
}

/// GR8_DS_PF_DIRECT
inline bool ds_pf_direct()
{
    static const bool v = *detail::flag(137, 0);
    return v;
}

/// GR8_DS_PF_GATHER
inline bool ds_pf_gather()
{
    static const bool v = *detail::flag(138, 0);
    return v;
}

/// GR8_DS_PF_L20_TAIL
inline bool ds_pf_l20_tail()
{
    static const bool v = *detail::flag(139, 0);
    return v;
}

/// GR8_DS_PF_LANES
inline const std::optional<std::string>& ds_pf_lanes()
{
    static const std::optional<std::string> v = detail::text(140, 0);
    return v;
}

/// GR8_DS_PF_LATE_FINAL
inline bool ds_pf_late_final()
{
    static const bool v = *detail::flag(141, 0);
    return v;
}

/// GR8_DS_PF_POOL
inline const std::optional<std::string>& ds_pf_pool()
{
    static const std::optional<std::string> v = detail::text(142, 0);
    return v;
}

/// GR8_DS_PF_RS
inline bool ds_pf_rs()
{
    static const bool v = *detail::flag(143, 0);
    return v;
}

/// GR8_DS_PF_TAIL
inline const std::optional<std::string>& ds_pf_tail()
{
    static const std::optional<std::string> v = detail::text(144, 0);
    return v;
}

/// GR8_DS_PF_V2
inline bool ds_pf_v2()
{
    static const bool v = *detail::flag(145, 0);
    return v;
}

/// GR8_DS_PF_V2_ATTN_PIPE
inline bool ds_pf_v2_attn_pipe()
{
    static const bool v = *detail::flag(146, 0);
    return v;
}

/// GR8_DS_PF_V2_BF16_TC
inline const std::vector<int64_t>& ds_pf_v2_bf16_tc()
{
    static const std::vector<int64_t> v = *detail::ints(147, 0);
    return v;
}

/// GR8_DS_PF_V2_FC2
inline bool ds_pf_v2_fc2()
{
    static const bool v = *detail::flag(148, 0);
    return v;
}

/// GR8_DS_PF_V2_GATE_RO
inline bool ds_pf_v2_gate_ro()
{
    static const bool v = *detail::flag(149, 0);
    return v;
}

/// GR8_DS_PF_V2_L20_TAIL
inline const std::vector<int64_t>& ds_pf_v2_l20_tail()
{
    static const std::vector<int64_t> v = *detail::ints(150, 0);
    return v;
}

/// GR8_DS_PF_V2_PACK_BLOCKS
inline const std::vector<int64_t>& ds_pf_v2_pack_blocks()
{
    static const std::vector<int64_t> v = *detail::ints(151, 0);
    return v;
}

/// GR8_DS_PF_V2_RS_BLOCKS
inline const std::vector<int64_t>& ds_pf_v2_rs_blocks()
{
    static const std::vector<int64_t> v = *detail::ints(152, 0);
    return v;
}

/// GR8_DS_PF_V2_SEND_FFN_OPT
inline const std::vector<int64_t>& ds_pf_v2_send_ffn_opt()
{
    static const std::vector<int64_t> v = *detail::ints(153, 0);
    return v;
}

/// GR8_DS_PF_V2_STRIPE_MIN
inline int64_t ds_pf_v2_stripe_min()
{
    static const int64_t v = *detail::int_(154, 0);
    return v;
}

/// GR8_DS_PF_V2_TIMING
inline bool ds_pf_v2_timing()
{
    static const bool v = *detail::flag(155, 0);
    return v;
}

/// GR8_DS_PF_V2_W16K
inline bool ds_pf_v2_w16k()
{
    static const bool v = *detail::flag(156, 0);
    return v;
}

/// GR8_DS_PF_V2_XTRACE
inline bool ds_pf_v2_xtrace()
{
    static const bool v = *detail::flag(157, 0);
    return v;
}

/// GR8_DS_PREFILL_KEEP
inline const std::optional<std::string>& ds_prefill_keep()
{
    static const std::optional<std::string> v = detail::text(158, 0);
    return v;
}

/// GR8_DS_PREFILL_MIN
inline int64_t ds_prefill_min()
{
    static const int64_t v = *detail::int_(159, 0);
    return v;
}

/// GR8_DS_PREFILL_W
inline const std::optional<std::string>& ds_prefill_w()
{
    static const std::optional<std::string> v = detail::text(160, 0);
    return v;
}

/// GR8_DS_RELAXED_ACCEPT
inline const std::optional<std::string>& ds_relaxed_accept()
{
    static const std::optional<std::string> v = detail::text(161, 0);
    return v;
}

/// GR8_DS_RO_L2LA
inline int64_t ds_ro_l2la()
{
    static const int64_t v = *detail::int_(162, 0);
    return v;
}

/// GR8_DS_RO_ORDER
inline bool ds_ro_order()
{
    static const bool v = *detail::flag(163, 0);
    return v;
}

/// GR8_DS_SESS_DRAFTS
inline bool ds_sess_drafts()
{
    static const bool v = *detail::flag(164, 0);
    return v;
}

/// GR8_DS_SH_TC
inline bool ds_sh_tc()
{
    static const bool v = *detail::flag(165, 0);
    return v;
}

/// GR8_DS_SPAN
inline int64_t ds_span()
{
    static const int64_t v = *detail::int_(166, 0);
    return v;
}

/// GR8_DS_SRC_FAST
inline int64_t ds_src_fast()
{
    static const int64_t v = *detail::int_(167, 0);
    return v;
}

/// GR8_DS_STRIPE
inline bool ds_stripe()
{
    static const bool v = *detail::flag(168, 0);
    return v;
}

/// GR8_DS_STRIPE_MIN
inline int64_t ds_stripe_min()
{
    static const int64_t v = *detail::int_(169, 0);
    return v;
}

/// GR8_DS_SUBDUMP
inline const std::optional<std::string>& ds_subdump()
{
    static const std::optional<std::string> v = detail::text(170, 0);
    return v;
}

/// GR8_DS_TRUNK_BF16
inline bool ds_trunk_bf16()
{
    static const bool v = *detail::flag(171, 0);
    return v;
}

/// GR8_DS_VERIFY_CAP
inline const std::optional<std::string>& ds_verify_cap()
{
    static const std::optional<std::string> v = detail::text(172, 0);
    return v;
}

/// GR8_DS_WAIT_K
inline int64_t ds_wait_k()
{
    static const int64_t v = *detail::int_(173, 0);
    return v;
}

/// GR8_DS_WOB_PERSIST
inline bool ds_wob_persist()
{
    static const bool v = *detail::flag(174, 0);
    return v;
}

/// GR8_DS_XTRACE
inline bool ds_xtrace()
{
    static const bool v = *detail::flag(175, 0);
    return v;
}

/// GR8_DTR
inline bool dtr()
{
    static const bool v = *detail::flag(176, 0);
    return v;
}

/// GR8_DTR_EXACT_OPT
inline bool dtr_exact_opt()
{
    static const bool v = *detail::flag(177, 0);
    return v;
}

/// GR8_DTR_NBFAST
inline bool dtr_nbfast()
{
    static const bool v = *detail::flag(178, 0);
    return v;
}

/// GR8_DTR_NBFAST_OUT
inline bool dtr_nbfast_out()
{
    static const bool v = *detail::flag(179, 0);
    return v;
}

/// GR8_DTR_PF_COPY
inline bool dtr_pf_copy()
{
    static const bool v = *detail::flag(180, 0);
    return v;
}

/// GR8_DTR_PF_KDA_COPY
inline bool dtr_pf_kda_copy()
{
    static const bool v = *detail::flag(181, 0);
    return v;
}

/// GR8_DTR_PF_SHARE_SUH
inline bool dtr_pf_share_suh()
{
    static const bool v = *detail::flag(182, 0);
    return v;
}

/// GR8_ENV
inline const std::optional<std::string>& env()
{
    static const std::optional<std::string> v = detail::text(183, 0);
    return v;
}

/// GR8_FABRIC_CONNECT_TRIES
inline int64_t fabric_connect_tries()
{
    static const int64_t v = *detail::int_(184, 0);
    return v;
}

/// GR8_FABRIC_DUMP
inline bool fabric_dump()
{
    static const bool v = *detail::flag(185, 0);
    return v;
}

/// GR8_FABRIC_POOL
inline int64_t fabric_pool()
{
    static const int64_t v = *detail::int_(186, 0);
    return v;
}

/// GR8_FETCH_WORKERS
inline std::optional<int64_t> fetch_workers()
{
    static const std::optional<int64_t> v = detail::int_(187, 0);
    return v;
}

/// GR8_FPF_ATTN_P1
inline bool fpf_attn_p1()
{
    static const bool v = *detail::flag(188, 0);
    return v;
}

/// GR8_FPF_DSA_P1
inline bool fpf_dsa_p1()
{
    static const bool v = *detail::flag(189, 0);
    return v;
}

/// GR8_FPF_P1
inline bool fpf_p1()
{
    static const bool v = *detail::flag(190, 0);
    return v;
}

/// GR8_FUSE
inline int64_t fuse()
{
    static const int64_t v = *detail::int_(191, 0);
    return v;
}

/// GR8_GEMM
inline int64_t gemm()
{
    static const int64_t v = *detail::int_(192, 0);
    return v;
}

/// GR8_GEN_CACHE
inline const std::optional<std::string>& gen_cache()
{
    static const std::optional<std::string> v = detail::text(193, 0);
    return v;
}

/// GR8_GLM53F_CACHE
inline const std::string& glm53f_cache()
{
    static const std::string v = *detail::text(194, 0);
    return v;
}

/// GR8_GLM53F_DEC_MHC_STAGE
inline bool glm53f_dec_mhc_stage()
{
    static const bool v = *detail::flag(195, 0);
    return v;
}

/// GR8_GLM53F_DRAFT_CACHE_DUMP
inline const std::optional<std::string>& glm53f_draft_cache_dump()
{
    static const std::optional<std::string> v = detail::text(196, 0);
    return v;
}

/// GR8_GLM53F_DRAFT_KV
inline bool glm53f_draft_kv()
{
    static const bool v = *detail::flag(197, 0);
    return v;
}

/// GR8_GLM53F_DSA_IDXQ_TPC
inline int64_t glm53f_dsa_idxq_tpc()
{
    static const int64_t v = *detail::int_(198, 0);
    return v;
}

/// GR8_GLM53F_DSA_OPROJ_ROT
inline std::optional<bool> glm53f_dsa_oproj_rot()
{
    static const std::optional<bool> v = detail::flag(199, 0);
    return v;
}

/// GR8_GLM53F_DSA_OPROJ_SPLIT
inline const std::optional<std::vector<int64_t>>& glm53f_dsa_oproj_split()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(200, 0);
    return v;
}

/// GR8_GLM53F_DSA_QB_FUSE
inline int64_t glm53f_dsa_qb_fuse()
{
    static const int64_t v = *detail::int_(201, 0);
    return v;
}

/// GR8_GLM53F_DSA_QB_ROT
inline bool glm53f_dsa_qb_rot()
{
    static const bool v = *detail::flag(202, 0);
    return v;
}

/// GR8_GLM53F_DSA_QB_SPLIT
inline const std::optional<std::vector<int64_t>>& glm53f_dsa_qb_split()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(203, 0);
    return v;
}

/// GR8_GLM53F_DSA_XIN_ORDER
inline bool glm53f_dsa_xin_order()
{
    static const bool v = *detail::flag(204, 0);
    return v;
}

/// GR8_GLM53F_FPF_DENSE_P1
inline bool glm53f_fpf_dense_p1()
{
    static const bool v = *detail::flag(205, 0);
    return v;
}

/// GR8_GLM53F_FPF_KDA_P1
inline bool glm53f_fpf_kda_p1()
{
    static const bool v = *detail::flag(206, 0);
    return v;
}

/// GR8_GLM53F_KDA_MX
inline bool glm53f_kda_mx()
{
    static const bool v = *detail::flag(207, 0);
    return v;
}

/// GR8_GLM53F_KDA_OPROJ_SPLIT
inline const std::optional<std::vector<int64_t>>& glm53f_kda_oproj_split()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(208, 0);
    return v;
}

/// GR8_GLM53F_KDA_PF_IN1
inline bool glm53f_kda_pf_in1()
{
    static const bool v = *detail::flag(209, 0);
    return v;
}

/// GR8_GLM53F_KDA_PF_P1
inline bool glm53f_kda_pf_p1()
{
    static const bool v = *detail::flag(210, 0);
    return v;
}

/// GR8_GLM53F_KDA_PF_ST1
inline bool glm53f_kda_pf_st1()
{
    static const bool v = *detail::flag(211, 0);
    return v;
}

/// GR8_GLM53F_KDA_PROJ_KS
inline int64_t glm53f_kda_proj_ks()
{
    static const int64_t v = *detail::int_(212, 0);
    return v;
}

/// GR8_GLM53F_KDA_PROJ_MODE
inline int64_t glm53f_kda_proj_mode()
{
    static const int64_t v = *detail::int_(213, 0);
    return v;
}

/// GR8_GLM53F_KDA_SCAN
inline int64_t glm53f_kda_scan()
{
    static const int64_t v = *detail::int_(214, 0);
    return v;
}

/// GR8_GLM53F_KDA_STATE_DUMP
inline const std::optional<std::string>& glm53f_kda_state_dump()
{
    static const std::optional<std::string> v = detail::text(215, 0);
    return v;
}

/// GR8_GLM53F_KDA_VS
inline int64_t glm53f_kda_vs()
{
    static const int64_t v = *detail::int_(216, 0);
    return v;
}

/// GR8_GLM53F_KV_CHUNKS
inline bool glm53f_kv_chunks()
{
    static const bool v = *detail::flag(217, 0);
    return v;
}

/// GR8_GLM53F_KV_CRASH_AT
inline const std::optional<std::string>& glm53f_kv_crash_at()
{
    static const std::optional<std::string> v = detail::text(218, 0);
    return v;
}

/// GR8_GLM53F_KV_IDLE_CHUNKS
inline int64_t glm53f_kv_idle_chunks()
{
    static const int64_t v = *detail::int_(219, 0);
    return v;
}

/// GR8_GLM53F_KV_LEDGER
inline const std::optional<std::string>& glm53f_kv_ledger()
{
    static const std::optional<std::string> v = detail::text(220, 0);
    return v;
}

/// GR8_GLM53F_KV_LEND_MS
inline int64_t glm53f_kv_lend_ms()
{
    static const int64_t v = *detail::int_(221, 0);
    return v;
}

/// GR8_GLM53F_MHC_FUSE
inline int64_t glm53f_mhc_fuse()
{
    static const int64_t v = *detail::int_(222, 0);
    return v;
}

/// GR8_GLM53F_MHC_FUSE_M1
inline std::optional<int64_t> glm53f_mhc_fuse_m1()
{
    static const std::optional<int64_t> v = detail::int_(223, 0);
    return v;
}

/// GR8_GLM53F_MHC_FUSE_M4
inline std::optional<int64_t> glm53f_mhc_fuse_m4()
{
    static const std::optional<int64_t> v = detail::int_(224, 0);
    return v;
}

/// GR8_GLM53F_PF_AMARK
inline bool glm53f_pf_amark()
{
    static const bool v = *detail::flag(225, 0);
    return v;
}

/// GR8_GLM53F_PF_BF16_EXACT
inline bool glm53f_pf_bf16_exact()
{
    static const bool v = *detail::flag(226, 0);
    return v;
}

/// GR8_GLM53F_PF_DRAFT_FULL
inline int64_t glm53f_pf_draft_full()
{
    static const int64_t v = *detail::int_(227, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_F16GEMM
inline bool glm53f_pf_dsa_f16gemm()
{
    static const bool v = *detail::flag(228, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_FULL_GV
inline bool glm53f_pf_dsa_full_gv()
{
    static const bool v = *detail::flag(229, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_LT
inline bool glm53f_pf_dsa_lt()
{
    static const bool v = *detail::flag(230, 0);
    return v;
}

/// GR8_GLM53F_PF_DSA_PROJ_P1
inline bool glm53f_pf_dsa_proj_p1()
{
    static const bool v = *detail::flag(231, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_GZF
inline bool glm53f_pf_kda_gzf()
{
    static const bool v = *detail::flag(232, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_TAIL
inline int64_t glm53f_pf_kda_tail()
{
    static const int64_t v = *detail::int_(233, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_TAIL_CHECK
inline bool glm53f_pf_kda_tail_check()
{
    static const bool v = *detail::flag(234, 0);
    return v;
}

/// GR8_GLM53F_PF_KDA_WS
inline bool glm53f_pf_kda_ws()
{
    static const bool v = *detail::flag(235, 0);
    return v;
}

/// GR8_GLM53F_PF_KP_TC
inline int64_t glm53f_pf_kp_tc()
{
    static const int64_t v = *detail::int_(236, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_FUSE
inline int64_t glm53f_pf_mhc_fuse()
{
    static const int64_t v = *detail::int_(237, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_PK
inline bool glm53f_pf_mhc_pk()
{
    static const bool v = *detail::flag(238, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_RR
inline bool glm53f_pf_mhc_rr()
{
    static const bool v = *detail::flag(239, 0);
    return v;
}

/// GR8_GLM53F_PF_MHC_STAGE
inline bool glm53f_pf_mhc_stage()
{
    static const bool v = *detail::flag(240, 0);
    return v;
}

/// GR8_GLM53F_PF_MLA_PROJ2
inline bool glm53f_pf_mla_proj2()
{
    static const bool v = *detail::flag(241, 0);
    return v;
}

/// GR8_GLM53F_PF_NOWIDE
inline bool glm53f_pf_nowide()
{
    static const bool v = *detail::flag(242, 0);
    return v;
}

/// GR8_GLM53F_PF_POOL_CAPTURE
inline const std::optional<std::string>& glm53f_pf_pool_capture()
{
    static const std::optional<std::string> v = detail::text(243, 0);
    return v;
}

/// GR8_GLM53F_PF_POOL_OWNER
inline std::optional<int64_t> glm53f_pf_pool_owner()
{
    static const std::optional<int64_t> v = detail::int_(244, 0);
    return v;
}

/// GR8_GLM53F_PF_PREWARM
inline std::optional<bool> glm53f_pf_prewarm()
{
    static const std::optional<bool> v = detail::flag(245, 0);
    return v;
}

/// GR8_GLM53F_PF_PUT_CTA
inline int64_t glm53f_pf_put_cta()
{
    static const int64_t v = *detail::int_(246, 0);
    return v;
}

/// GR8_GLM53F_PF_ROUTER1
inline bool glm53f_pf_router1()
{
    static const bool v = *detail::flag(247, 0);
    return v;
}

/// GR8_GLM53F_PF_SHFP8
inline std::optional<bool> glm53f_pf_shfp8()
{
    static const std::optional<bool> v = detail::flag(248, 0);
    return v;
}

/// GR8_GLM53F_PF_SP
inline bool glm53f_pf_sp()
{
    static const bool v = *detail::flag(249, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA
inline bool glm53f_pf_sparse_mla()
{
    static const bool v = *detail::flag(250, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA_512
inline bool glm53f_pf_sparse_mla_512()
{
    static const bool v = *detail::flag(251, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA_H16C
inline bool glm53f_pf_sparse_mla_h16c()
{
    static const bool v = *detail::flag(252, 0);
    return v;
}

/// GR8_GLM53F_PF_SPARSE_MLA_V2
inline std::optional<bool> glm53f_pf_sparse_mla_v2()
{
    static const std::optional<bool> v = detail::flag(253, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_AGD
inline bool glm53f_pf_sp_agd()
{
    static const bool v = *detail::flag(254, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_AGK
inline bool glm53f_pf_sp_agk()
{
    static const bool v = *detail::flag(255, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_AGKA
inline bool glm53f_pf_sp_agka()
{
    static const bool v = *detail::flag(256, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_ATTD
inline bool glm53f_pf_sp_attd()
{
    static const bool v = *detail::flag(257, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_FIND
inline bool glm53f_pf_sp_find()
{
    static const bool v = *detail::flag(258, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_PUT_CTA
inline int64_t glm53f_pf_sp_put_cta()
{
    static const int64_t v = *detail::int_(259, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_X16
inline bool glm53f_pf_sp_x16()
{
    static const bool v = *detail::flag(260, 0);
    return v;
}

/// GR8_GLM53F_PF_SP_XFRAG
inline bool glm53f_pf_sp_xfrag()
{
    static const bool v = *detail::flag(261, 0);
    return v;
}

/// GR8_GLM53F_PF_STAGE_ASYNC
inline std::optional<bool> glm53f_pf_stage_async()
{
    static const std::optional<bool> v = detail::flag(262, 0);
    return v;
}

/// GR8_GLM53F_PF_XFRAG
inline bool glm53f_pf_xfrag()
{
    static const bool v = *detail::flag(263, 0);
    return v;
}

/// GR8_GLM53F_PF_XTRACE
inline bool glm53f_pf_xtrace()
{
    static const bool v = *detail::flag(264, 0);
    return v;
}

/// GR8_GLM53F_RESET_ALL
inline bool glm53f_reset_all()
{
    static const bool v = *detail::flag(265, 0);
    return v;
}

/// GR8_GLM53F_RESET_CHECK
inline bool glm53f_reset_check()
{
    static const bool v = *detail::flag(266, 0);
    return v;
}

/// GR8_GLM53F_ROUTER_MF
inline bool glm53f_router_mf()
{
    static const bool v = *detail::flag(267, 0);
    return v;
}

/// GR8_GLM53F_ROUTER_NW
inline int64_t glm53f_router_nw()
{
    static const int64_t v = *detail::int_(268, 0);
    return v;
}

/// GR8_GLM53F_S2_COST
inline bool glm53f_s2_cost()
{
    static const bool v = *detail::flag(269, 0);
    return v;
}

/// GR8_GLM53F_STATE_BUDGET_MIB
inline int64_t glm53f_state_budget_mib()
{
    static const int64_t v = *detail::int_(270, 0);
    return v;
}

/// GR8_GLM53F_VISION
inline bool glm53f_vision()
{
    static const bool v = *detail::flag(271, 0);
    return v;
}

/// GR8_GV_MULTI
inline bool gv_multi()
{
    static const bool v = *detail::flag(272, 0);
    return v;
}

/// GR8_HF_ENDPOINT
inline const std::optional<std::string>& hf_endpoint()
{
    static const std::optional<std::string> v = detail::text(273, 0);
    return v;
}

/// GR8_HOME
inline const std::optional<std::string>& home()
{
    static const std::optional<std::string> v = detail::text(274, 0);
    return v;
}

/// GR8_HY4_DFLY
inline bool hy4_dfly()
{
    static const bool v = *detail::flag(275, 0);
    return v;
}

/// GR8_HY4_DFLY_TOKENS
inline int64_t hy4_dfly_tokens()
{
    static const int64_t v = *detail::int_(276, 0);
    return v;
}

/// GR8_HY4_ENGRAM_CACHE_MB
inline std::optional<int64_t> hy4_engram_cache_mb()
{
    static const std::optional<int64_t> v = detail::int_(277, 0);
    return v;
}

/// GR8_HY4_ENGRAM_DIRECT
inline bool hy4_engram_direct()
{
    static const bool v = *detail::flag(278, 0);
    return v;
}

/// GR8_HY4_ENGRAM_MMAP
inline bool hy4_engram_mmap()
{
    static const bool v = *detail::flag(279, 0);
    return v;
}

/// GR8_HY4_GRAPHS
inline bool hy4_graphs()
{
    static const bool v = *detail::flag(280, 0);
    return v;
}

/// GR8_HY4_KV_FORMAT
inline const std::string& hy4_kv_format()
{
    static const std::string v = *detail::text(281, 0);
    return v;
}

/// GR8_HY4_MX8_DENSE
inline bool hy4_mx8_dense()
{
    static const bool v = *detail::flag(282, 0);
    return v;
}

/// GR8_HY4_PROFILE
inline bool hy4_profile()
{
    static const bool v = *detail::flag(283, 0);
    return v;
}

/// GR8_IDXQ_SPLIT
inline std::optional<bool> idxq_split()
{
    static const std::optional<bool> v = detail::flag(284, 0);
    return v;
}

/// GR8_IDX_GRP
inline int64_t idx_grp()
{
    static const int64_t v = *detail::int_(285, 0);
    return v;
}

/// GR8_IDX_VEC
inline bool idx_vec()
{
    static const bool v = *detail::flag(286, 0);
    return v;
}

/// GR8_IMAGE
inline const std::optional<std::string>& image()
{
    static const std::optional<std::string> v = detail::text(287, 0);
    return v;
}

/// GR8_IMG_VMM
inline int64_t img_vmm()
{
    static const int64_t v = *detail::int_(288, 0);
    return v;
}

/// GR8_INK2_BATCH_COMMIT
inline bool ink2_batch_commit()
{
    static const bool v = *detail::flag(289, 0);
    return v;
}

/// GR8_INK2_CUBLAS_PREFILL
inline bool ink2_cublas_prefill()
{
    static const bool v = *detail::flag(290, 0);
    return v;
}

/// GR8_INK2_DEC_FAST_ATTN
inline bool ink2_dec_fast_attn()
{
    static const bool v = *detail::flag(291, 0);
    return v;
}

/// GR8_INK2_DF2_DP
inline bool ink2_df2_dp()
{
    static const bool v = *detail::flag(292, 0);
    return v;
}

/// GR8_INK2_DF2_EMBNORM
inline bool ink2_df2_embnorm()
{
    static const bool v = *detail::flag(293, 0);
    return v;
}

/// GR8_INK2_DF2_GRAPH
inline bool ink2_df2_graph()
{
    static const bool v = *detail::flag(294, 0);
    return v;
}

/// GR8_INK2_DF2_HEAD2
inline bool ink2_df2_head2()
{
    static const bool v = *detail::flag(295, 0);
    return v;
}

/// GR8_INK2_DF2_Q8
inline bool ink2_df2_q8()
{
    static const bool v = *detail::flag(296, 0);
    return v;
}

/// GR8_INK2_DF2_SEL_ALPHA
inline double ink2_df2_sel_alpha()
{
    static const double v = *detail::float_(297, 0);
    return v;
}

/// GR8_INK2_DFLASH2
inline const std::optional<std::string>& ink2_dflash2()
{
    static const std::optional<std::string> v = detail::text(298, 0);
    return v;
}

/// GR8_INK2_FAST_ALLREDUCE
inline bool ink2_fast_allreduce()
{
    static const bool v = *detail::flag(299, 0);
    return v;
}

/// GR8_INK2_FAST_GEMV
inline bool ink2_fast_gemv()
{
    static const bool v = *detail::flag(300, 0);
    return v;
}

/// GR8_INK2_FAST_GREEDY_HEAD
inline bool ink2_fast_greedy_head()
{
    static const bool v = *detail::flag(301, 0);
    return v;
}

/// GR8_INK2_FAST_RES
inline bool ink2_fast_res()
{
    static const bool v = *detail::flag(302, 0);
    return v;
}

/// GR8_INK2_FAST_ROUTER
inline bool ink2_fast_router()
{
    static const bool v = *detail::flag(303, 0);
    return v;
}

/// GR8_INK2_FUSED_QKV
inline bool ink2_fused_qkv()
{
    static const bool v = *detail::flag(304, 0);
    return v;
}

/// GR8_INK2_GEMM_C128
inline bool ink2_gemm_c128()
{
    static const bool v = *detail::flag(305, 0);
    return v;
}

/// GR8_INK2_GPU_HEAD_PACK
inline bool ink2_gpu_head_pack()
{
    static const bool v = *detail::flag(306, 0);
    return v;
}

/// GR8_INK2_GRAPH
inline bool ink2_graph()
{
    static const bool v = *detail::flag(307, 0);
    return v;
}

/// GR8_INK2_LOCAL_GRP
inline bool ink2_local_grp()
{
    static const bool v = *detail::flag(308, 0);
    return v;
}

/// GR8_INK2_MMA_NARROW_CTA
inline bool ink2_mma_narrow_cta()
{
    static const bool v = *detail::flag(309, 0);
    return v;
}

/// GR8_INK2_MSG_BF16
inline bool ink2_msg_bf16()
{
    static const bool v = *detail::flag(310, 0);
    return v;
}

/// GR8_INK2_MSG_PREFILL_FUSE
inline bool ink2_msg_prefill_fuse()
{
    static const bool v = *detail::flag(311, 0);
    return v;
}

/// GR8_INK2_MTP_ATTN_GRP
inline bool ink2_mtp_attn_grp()
{
    static const bool v = *detail::flag(312, 0);
    return v;
}

/// GR8_INK2_MTP_FAST_ARGMAX
inline bool ink2_mtp_fast_argmax()
{
    static const bool v = *detail::flag(313, 0);
    return v;
}

/// GR8_INK2_MTP_INT8_LM
inline bool ink2_mtp_int8_lm()
{
    static const bool v = *detail::flag(314, 0);
    return v;
}

/// GR8_INK2_MX8_DENSE
inline int64_t ink2_mx8_dense()
{
    static const int64_t v = *detail::int_(315, 0);
    return v;
}

/// GR8_INK2_MX8_LM_HEAD
inline int64_t ink2_mx8_lm_head()
{
    static const int64_t v = *detail::int_(316, 0);
    return v;
}

/// GR8_INK2_MX8_TC_MIN_M
inline int64_t ink2_mx8_tc_min_m()
{
    static const int64_t v = *detail::int_(317, 0);
    return v;
}

/// GR8_INK2_PF_FAST_ATTN
inline bool ink2_pf_fast_attn()
{
    static const bool v = *detail::flag(318, 0);
    return v;
}

/// GR8_INK2_PF_MX8_MIN_M
inline int64_t ink2_pf_mx8_min_m()
{
    static const int64_t v = *detail::int_(319, 0);
    return v;
}

/// GR8_INK2_PF_MX8_MOE
inline int64_t ink2_pf_mx8_moe()
{
    static const int64_t v = *detail::int_(320, 0);
    return v;
}

/// GR8_INK2_PROFILE
inline bool ink2_profile()
{
    static const bool v = *detail::flag(321, 0);
    return v;
}

/// GR8_INK2_PROFILE_EVERY
inline int64_t ink2_profile_every()
{
    static const int64_t v = *detail::int_(322, 0);
    return v;
}

/// GR8_INK2_S2_COST
inline const std::optional<std::vector<double>>& ink2_s2_cost()
{
    static const std::optional<std::vector<double>> v = detail::floats(323, 0);
    return v;
}

/// GR8_INK2_SPLITK
inline bool ink2_splitk()
{
    static const bool v = *detail::flag(324, 0);
    return v;
}

/// GR8_INK2_STOCK_MSG_GATE
inline bool ink2_stock_msg_gate()
{
    static const bool v = *detail::flag(325, 0);
    return v;
}

/// GR8_INK2_TC_PREFILL_ATTN
inline bool ink2_tc_prefill_attn()
{
    static const bool v = *detail::flag(326, 0);
    return v;
}

/// GR8_INK2_TR3_WIDE
inline bool ink2_tr3_wide()
{
    static const bool v = *detail::flag(327, 0);
    return v;
}

/// GR8_INK2_VARIANT
inline const std::optional<std::string>& ink2_variant()
{
    static const std::optional<std::string> v = detail::text(328, 0);
    return v;
}

/// GR8_INK2_VISION
inline bool ink2_vision()
{
    static const bool v = *detail::flag(329, 0);
    return v;
}

/// GR8_K1_FUSE_POLICY
inline bool k1_fuse_policy()
{
    static const bool v = *detail::flag(330, 0);
    return v;
}

/// GR8_K1_SPLIT
inline const std::optional<std::string>& k1_split()
{
    static const std::optional<std::string> v = detail::text(331, 0);
    return v;
}

/// GR8_K1_SPLIT_M1
inline const std::optional<std::vector<int64_t>>& k1_split_m1()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(332, 0);
    return v;
}

/// GR8_K1_SPLIT_M2
inline const std::optional<std::vector<int64_t>>& k1_split_m2()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(333, 0);
    return v;
}

/// GR8_K1_SPLIT_M4
inline const std::optional<std::vector<int64_t>>& k1_split_m4()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(334, 0);
    return v;
}

/// GR8_K1_TLBW
inline int64_t k1_tlbw()
{
    static const int64_t v = *detail::int_(335, 0);
    return v;
}

/// GR8_KNOBS_LIB
inline const std::optional<std::string>& knobs_lib()
{
    static const std::optional<std::string> v = detail::text(336, 0);
    return v;
}

/// GR8_KV4_CTX
inline int64_t kv4_ctx()
{
    static const int64_t v = *detail::int_(337, 0);
    return v;
}

/// GR8_KV4_FOLD
inline bool kv4_fold()
{
    static const bool v = *detail::flag(338, 0);
    return v;
}

/// GR8_KV4_FUSE_ROT
inline bool kv4_fuse_rot()
{
    static const bool v = *detail::flag(339, 0);
    return v;
}

/// GR8_KV4_QT
inline bool kv4_qt()
{
    static const bool v = *detail::flag(340, 0);
    return v;
}

/// GR8_KV4_SC
inline int64_t kv4_sc()
{
    static const int64_t v = *detail::int_(341, 0);
    return v;
}

/// GR8_KV4_SC_TR
inline int64_t kv4_sc_tr()
{
    static const int64_t v = *detail::int_(342, 0);
    return v;
}

/// GR8_KV4_TC
inline bool kv4_tc()
{
    static const bool v = *detail::flag(343, 0);
    return v;
}

/// GR8_KVB
inline const std::string& kvb()
{
    static const std::string v = *detail::text(344, 0);
    return v;
}

/// GR8_KVMAN_TRANSPORT
inline const std::string& kvman_transport()
{
    static const std::string v = *detail::text(345, 0);
    return v;
}

/// GR8_KV_CHUNKS
inline bool kv_chunks()
{
    static const bool v = *detail::flag(346, 0);
    return v;
}

/// GR8_KV_CHUNK_PAGES
inline int64_t kv_chunk_pages()
{
    static const int64_t v = *detail::int_(347, 0);
    return v;
}

/// GR8_KV_DRAIN_DELAY
inline const std::optional<std::string>& kv_drain_delay()
{
    static const std::optional<std::string> v = detail::text(348, 0);
    return v;
}

/// GR8_KV_REWIDTH
inline const std::optional<std::string>& kv_rewidth()
{
    static const std::optional<std::string> v = detail::text(349, 0);
    return v;
}

/// GR8_KV_SHARE_PAGES
inline bool kv_share_pages()
{
    static const bool v = *detail::flag(350, 0);
    return v;
}

/// GR8_KV_SHARE_WHEN
inline const std::string& kv_share_when()
{
    static const std::string v = *detail::text(351, 0);
    return v;
}

/// GR8_L2PF
inline int64_t l2pf()
{
    static const int64_t v = *detail::int_(352, 0);
    return v;
}

/// GR8_L2PF_GBPS
inline double l2pf_gbps()
{
    static const double v = *detail::float_(353, 0);
    return v;
}

/// GR8_L2PF_MB
inline double l2pf_mb()
{
    static const double v = *detail::float_(354, 0);
    return v;
}

/// GR8_L2PF_OPMB
inline double l2pf_opmb()
{
    static const double v = *detail::float_(355, 0);
    return v;
}

/// GR8_L2PF_T
inline int64_t l2pf_t()
{
    static const int64_t v = *detail::int_(356, 0);
    return v;
}

/// GR8_L2PF_US
inline const std::vector<double>& l2pf_us()
{
    static const std::vector<double> v = *detail::floats(357, 0);
    return v;
}

/// GR8_L2PF_WAVE
inline int64_t l2pf_wave()
{
    static const int64_t v = *detail::int_(358, 0);
    return v;
}

/// GR8_LOADER
inline const std::optional<std::string>& loader()
{
    static const std::optional<std::string> v = detail::text(359, 0);
    return v;
}

/// GR8_LOAD_CHUNK_MB
inline int64_t load_chunk_mb()
{
    static const int64_t v = *detail::int_(360, 0);
    return v;
}

/// GR8_LOAD_COMPACT
inline const std::optional<std::string>& load_compact()
{
    static const std::optional<std::string> v = detail::text(361, 0);
    return v;
}

/// GR8_LOAD_MERGE_KB
inline int64_t load_merge_kb()
{
    static const int64_t v = *detail::int_(362, 0);
    return v;
}

/// GR8_LOAD_MODE
inline const std::string& load_mode()
{
    static const std::string v = *detail::text(363, 0);
    return v;
}

/// GR8_LOAD_PROF
inline bool load_prof()
{
    static const bool v = *detail::flag(364, 0);
    return v;
}

/// GR8_LOAD_THREADS
inline int64_t load_threads()
{
    static const int64_t v = *detail::int_(365, 0);
    return v;
}

/// GR8_LOGIT_DUMP
inline const std::optional<std::string>& logit_dump()
{
    static const std::optional<std::string> v = detail::text(366, 0);
    return v;
}

/// GR8_LOGIT_DUMP_FROM
inline int64_t logit_dump_from()
{
    static const int64_t v = *detail::int_(367, 0);
    return v;
}

/// GR8_MEM_MIN
inline double mem_min()
{
    static const double v = *detail::float_(368, 0);
    return v;
}

/// GR8_MHC_BF16
inline int64_t mhc_bf16()
{
    static const int64_t v = *detail::int_(369, 0);
    return v;
}

/// GR8_MODEL
inline const std::optional<std::string>& model()
{
    static const std::optional<std::string> v = detail::text(370, 0);
    return v;
}

/// GR8_MOE_RTR_FIN_OPT
inline bool moe_rtr_fin_opt()
{
    static const bool v = *detail::flag(371, 0);
    return v;
}

/// GR8_MOE_TOPK
inline int64_t moe_topk()
{
    static const int64_t v = *detail::int_(372, 0);
    return v;
}

/// GR8_MOE_UNION_CAP
inline int64_t moe_union_cap()
{
    static const int64_t v = *detail::int_(373, 0);
    return v;
}

/// GR8_MOE_UNION_MODE
inline int64_t moe_union_mode()
{
    static const int64_t v = *detail::int_(374, 0);
    return v;
}

/// GR8_MTP_KSTOP
inline std::optional<int64_t> mtp_kstop()
{
    static const std::optional<int64_t> v = detail::int_(375, 0);
    return v;
}

/// GR8_MTP_KSTOPB
inline bool mtp_kstopb()
{
    static const bool v = *detail::flag(376, 0);
    return v;
}

/// GR8_MTP_KSTOPB_EXPLORE
inline int64_t mtp_kstopb_explore()
{
    static const int64_t v = *detail::int_(377, 0);
    return v;
}

/// GR8_MTP_KSTOPB_KMIN
inline int64_t mtp_kstopb_kmin()
{
    static const int64_t v = *detail::int_(378, 0);
    return v;
}

/// GR8_MTP_KSTOPB_TAU
inline const std::vector<double>& mtp_kstopb_tau()
{
    static const std::vector<double> v = *detail::floats(379, 0);
    return v;
}

/// GR8_MTP_KSTOPB_TAU_S<n>
inline std::optional<std::vector<double>> mtp_kstopb_tau_s(uint32_t n) { return detail::floats(380, n); }

/// GR8_MTP_KSTOPB_WIDTHS
inline const std::optional<std::vector<int64_t>>& mtp_kstopb_widths()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(381, 0);
    return v;
}

/// GR8_MTP_KSTOP_KMAX
inline std::optional<int64_t> mtp_kstop_kmax()
{
    static const std::optional<int64_t> v = detail::int_(382, 0);
    return v;
}

/// GR8_MTP_KSTOP_SEED3
inline double mtp_kstop_seed3()
{
    static const double v = *detail::float_(383, 0);
    return v;
}

/// GR8_MTP_KSTOP_TAU
inline const std::vector<double>& mtp_kstop_tau()
{
    static const std::vector<double> v = *detail::floats(384, 0);
    return v;
}

/// GR8_MTP_L2PF
inline int64_t mtp_l2pf()
{
    static const int64_t v = *detail::int_(385, 0);
    return v;
}

/// GR8_MTP_L2PF_US
inline double mtp_l2pf_us()
{
    static const double v = *detail::float_(386, 0);
    return v;
}

/// GR8_MTP_LASTROW
inline bool mtp_lastrow()
{
    static const bool v = *detail::flag(387, 0);
    return v;
}

/// GR8_MTP_LPLOG
inline int64_t mtp_lplog()
{
    static const int64_t v = *detail::int_(388, 0);
    return v;
}

/// GR8_MTP_SPLIT
inline std::optional<bool> mtp_split()
{
    static const std::optional<bool> v = detail::flag(389, 0);
    return v;
}

/// GR8_NGRAM
inline const std::optional<std::string>& ngram()
{
    static const std::optional<std::string> v = detail::text(390, 0);
    return v;
}

/// GR8_NGRAM_KMAX
inline std::optional<int64_t> ngram_kmax()
{
    static const std::optional<int64_t> v = detail::int_(391, 0);
    return v;
}

/// GR8_NGRAM_KMIN
inline int64_t ngram_kmin()
{
    static const int64_t v = *detail::int_(392, 0);
    return v;
}

/// GR8_NGRAM_LOG
inline bool ngram_log()
{
    static const bool v = *detail::flag(393, 0);
    return v;
}

/// GR8_NGRAM_MARGIN
inline double ngram_margin()
{
    static const double v = *detail::float_(394, 0);
    return v;
}

/// GR8_NOWIDE2
inline bool nowide2()
{
    static const bool v = *detail::flag(395, 0);
    return v;
}

/// GR8_NR
inline const std::optional<std::string>& nr()
{
    static const std::optional<std::string> v = detail::text(396, 0);
    return v;
}

/// GR8_NVCC_CACHE
inline bool nvcc_cache()
{
    static const bool v = *detail::flag(397, 0);
    return v;
}

/// GR8_NVCC_CACHE_DIR
inline const std::optional<std::string>& nvcc_cache_dir()
{
    static const std::optional<std::string> v = detail::text(398, 0);
    return v;
}

/// GR8_NVCC_CACHE_LOG
inline const std::optional<std::string>& nvcc_cache_log()
{
    static const std::optional<std::string> v = detail::text(399, 0);
    return v;
}

/// GR8_PF_ADUMP
inline const std::optional<std::string>& pf_adump()
{
    static const std::optional<std::string> v = detail::text(400, 0);
    return v;
}

/// GR8_PF_AOP
inline std::optional<bool> pf_aop()
{
    static const std::optional<bool> v = detail::flag(401, 0);
    return v;
}

/// GR8_PF_AOP_MMIN
inline int64_t pf_aop_mmin()
{
    static const int64_t v = *detail::int_(402, 0);
    return v;
}

/// GR8_PF_ATTN_OPT
inline bool pf_attn_opt()
{
    static const bool v = *detail::flag(403, 0);
    return v;
}

/// GR8_PF_ATTN_ROWS
inline std::optional<int64_t> pf_attn_rows()
{
    static const std::optional<int64_t> v = detail::int_(404, 0);
    return v;
}

/// GR8_PF_BLAS_ROWS
inline std::optional<int64_t> pf_blas_rows()
{
    static const std::optional<int64_t> v = detail::int_(405, 0);
    return v;
}

/// GR8_PF_DENSE
inline int64_t pf_dense()
{
    static const int64_t v = *detail::int_(406, 0);
    return v;
}

/// GR8_PF_DENSE_OPT
inline bool pf_dense_opt()
{
    static const bool v = *detail::flag(407, 0);
    return v;
}

/// GR8_PF_DRAFT_SHARE
inline std::optional<int64_t> pf_draft_share()
{
    static const std::optional<int64_t> v = detail::int_(408, 0);
    return v;
}

/// GR8_PF_DTR8
inline bool pf_dtr8()
{
    static const bool v = *detail::flag(409, 0);
    return v;
}

/// GR8_PF_EVENTS
inline bool pf_events()
{
    static const bool v = *detail::flag(410, 0);
    return v;
}

/// GR8_PF_F12
inline std::optional<bool> pf_f12()
{
    static const std::optional<bool> v = detail::flag(411, 0);
    return v;
}

/// GR8_PF_FC2ONE
inline bool pf_fc2one()
{
    static const bool v = *detail::flag(412, 0);
    return v;
}

/// GR8_PF_FC2V2
inline bool pf_fc2v2()
{
    static const bool v = *detail::flag(413, 0);
    return v;
}

/// GR8_PF_GEMMF32
inline bool pf_gemmf32()
{
    static const bool v = *detail::flag(414, 0);
    return v;
}

/// GR8_PF_GEMMG
inline int64_t pf_gemmg()
{
    static const int64_t v = *detail::int_(415, 0);
    return v;
}

/// GR8_PF_HASH
inline bool pf_hash()
{
    static const bool v = *detail::flag(416, 0);
    return v;
}

/// GR8_PF_HASH_GPU
inline bool pf_hash_gpu()
{
    static const bool v = *detail::flag(417, 0);
    return v;
}

/// GR8_PF_HEAD
inline bool pf_head()
{
    static const bool v = *detail::flag(418, 0);
    return v;
}

/// GR8_PF_HX
inline bool pf_hx()
{
    static const bool v = *detail::flag(419, 0);
    return v;
}

/// GR8_PF_HXF
inline bool pf_hxf()
{
    static const bool v = *detail::flag(420, 0);
    return v;
}

/// GR8_PF_IDDUMP
inline const std::optional<std::string>& pf_iddump()
{
    static const std::optional<std::string> v = detail::text(421, 0);
    return v;
}

/// GR8_PF_ISGRT_CLEAN
inline bool pf_isgrt_clean()
{
    static const bool v = *detail::flag(422, 0);
    return v;
}

/// GR8_PF_ISGRT_RG
inline int64_t pf_isgrt_rg()
{
    static const int64_t v = *detail::int_(423, 0);
    return v;
}

/// GR8_PF_MOE8
inline bool pf_moe8()
{
    static const bool v = *detail::flag(424, 0);
    return v;
}

/// GR8_PF_OWN
inline int64_t pf_own()
{
    static const int64_t v = *detail::int_(425, 0);
    return v;
}

/// GR8_PF_OWNPERM
inline bool pf_ownperm()
{
    static const bool v = *detail::flag(426, 0);
    return v;
}

/// GR8_PF_PDUMP
inline const std::optional<std::string>& pf_pdump()
{
    static const std::optional<std::string> v = detail::text(427, 0);
    return v;
}

/// GR8_PF_PDUMP_EVERY
inline int64_t pf_pdump_every()
{
    static const int64_t v = *detail::int_(428, 0);
    return v;
}

/// GR8_PF_PDUMP_LAYERS
inline const std::vector<int64_t>& pf_pdump_layers()
{
    static const std::vector<int64_t> v = *detail::ints(429, 0);
    return v;
}

/// GR8_PF_PROF
inline int64_t pf_prof()
{
    static const int64_t v = *detail::int_(430, 0);
    return v;
}

/// GR8_PF_PROF_TIMELINE
inline int64_t pf_prof_timeline()
{
    static const int64_t v = *detail::int_(431, 0);
    return v;
}

/// GR8_PF_Q8
inline bool pf_q8()
{
    static const bool v = *detail::flag(432, 0);
    return v;
}

/// GR8_PF_QO
inline const std::string& pf_qo()
{
    static const std::string v = *detail::text(433, 0);
    return v;
}

/// GR8_PF_QP_CTX
inline std::optional<int64_t> pf_qp_ctx()
{
    static const std::optional<int64_t> v = detail::int_(434, 0);
    return v;
}

/// GR8_PF_QP_EARLY
inline int64_t pf_qp_early()
{
    static const int64_t v = *detail::int_(435, 0);
    return v;
}

/// GR8_PF_QP_LAYERS
inline const std::optional<std::string>& pf_qp_layers()
{
    static const std::optional<std::string> v = detail::text(436, 0);
    return v;
}

/// GR8_PF_QP_MIN
inline int64_t pf_qp_min()
{
    static const int64_t v = *detail::int_(437, 0);
    return v;
}

/// GR8_PF_QP_ROLL
inline int64_t pf_qp_roll()
{
    static const int64_t v = *detail::int_(438, 0);
    return v;
}

/// GR8_PF_QP_ROLL_SER
inline bool pf_qp_roll_ser()
{
    static const bool v = *detail::flag(439, 0);
    return v;
}

/// GR8_PF_QP_WARM
inline bool pf_qp_warm()
{
    static const bool v = *detail::flag(440, 0);
    return v;
}

/// GR8_PF_QP_WIRE_E
inline std::optional<int64_t> pf_qp_wire_e()
{
    static const std::optional<int64_t> v = detail::int_(441, 0);
    return v;
}

/// GR8_PF_RCDUMP
inline const std::optional<std::string>& pf_rcdump()
{
    static const std::optional<std::string> v = detail::text(442, 0);
    return v;
}

/// GR8_PF_ROUTER_SPLIT
inline bool pf_router_split()
{
    static const bool v = *detail::flag(443, 0);
    return v;
}

/// GR8_PF_RS
inline bool pf_rs()
{
    static const bool v = *detail::flag(444, 0);
    return v;
}

/// GR8_PF_RSDIRECT
inline bool pf_rsdirect()
{
    static const bool v = *detail::flag(445, 0);
    return v;
}

/// GR8_PF_RSHASH
inline bool pf_rshash()
{
    static const bool v = *detail::flag(446, 0);
    return v;
}

/// GR8_PF_SELX_CTX
inline int64_t pf_selx_ctx()
{
    static const int64_t v = *detail::int_(447, 0);
    return v;
}

/// GR8_PF_SHR
inline bool pf_shr()
{
    static const bool v = *detail::flag(448, 0);
    return v;
}

/// GR8_PF_STAGGER
inline int64_t pf_stagger()
{
    static const int64_t v = *detail::int_(449, 0);
    return v;
}

/// GR8_PF_STREAMS
inline int64_t pf_streams()
{
    static const int64_t v = *detail::int_(450, 0);
    return v;
}

/// GR8_PF_STRIPE
inline std::optional<bool> pf_stripe()
{
    static const std::optional<bool> v = detail::flag(451, 0);
    return v;
}

/// GR8_PF_STRIPE_MIN
inline int64_t pf_stripe_min()
{
    static const int64_t v = *detail::int_(452, 0);
    return v;
}

/// GR8_PF_TILE
inline int64_t pf_tile()
{
    static const int64_t v = *detail::int_(453, 0);
    return v;
}

/// GR8_PF_TR_BM
inline int64_t pf_tr_bm()
{
    static const int64_t v = *detail::int_(454, 0);
    return v;
}

/// GR8_PF_WM
inline bool pf_wm()
{
    static const bool v = *detail::flag(455, 0);
    return v;
}

/// GR8_PF_WM2
inline bool pf_wm2()
{
    static const bool v = *detail::flag(456, 0);
    return v;
}

/// GR8_PF_WM_MIN
inline int64_t pf_wm_min()
{
    static const int64_t v = *detail::int_(457, 0);
    return v;
}

/// GR8_PF_WT
inline bool pf_wt()
{
    static const bool v = *detail::flag(458, 0);
    return v;
}

/// GR8_PF_WT_FUSE
inline bool pf_wt_fuse()
{
    static const bool v = *detail::flag(459, 0);
    return v;
}

/// GR8_PF_WT_GMAJOR
inline bool pf_wt_gmajor()
{
    static const bool v = *detail::flag(460, 0);
    return v;
}

/// GR8_POOL_LOG
inline bool pool_log()
{
    static const bool v = *detail::flag(461, 0);
    return v;
}

/// GR8_PROGRESS_ISOLATE
inline bool progress_isolate()
{
    static const bool v = *detail::flag(462, 0);
    return v;
}

/// GR8_PSEUDO_TOKENIZER
inline const std::optional<std::string>& pseudo_tokenizer()
{
    static const std::optional<std::string> v = detail::text(463, 0);
    return v;
}

/// GR8_PTAB
inline int64_t ptab()
{
    static const int64_t v = *detail::int_(464, 0);
    return v;
}

/// GR8_QA_SPLIT
inline std::optional<int64_t> qa_split()
{
    static const std::optional<int64_t> v = detail::int_(465, 0);
    return v;
}

/// GR8_QA_SPLIT_CPU
inline std::optional<int64_t> qa_split_cpu()
{
    static const std::optional<int64_t> v = detail::int_(466, 0);
    return v;
}

/// GR8_QA_SPLIT_TPC
inline int64_t qa_split_tpc()
{
    static const int64_t v = *detail::int_(467, 0);
    return v;
}

/// GR8_QO_CACHE
inline const std::optional<std::string>& qo_cache()
{
    static const std::optional<std::string> v = detail::text(468, 0);
    return v;
}

/// GR8_QO_DUMP
inline bool qo_dump()
{
    static const bool v = *detail::flag(469, 0);
    return v;
}

/// GR8_QO_NANINJ
inline bool qo_naninj()
{
    static const bool v = *detail::flag(470, 0);
    return v;
}

/// GR8_QO_ROPECHK
inline bool qo_ropechk()
{
    static const bool v = *detail::flag(471, 0);
    return v;
}

/// GR8_QUIT_FAST
inline bool quit_fast()
{
    static const bool v = *detail::flag(472, 0);
    return v;
}

/// GR8_QW38_ADAPTIVE
inline bool qw38_adaptive()
{
    static const bool v = *detail::flag(473, 0);
    return v;
}

/// GR8_QW38_CPU_FRAC
inline double qw38_cpu_frac()
{
    static const double v = *detail::float_(474, 0);
    return v;
}

/// GR8_QW38_CPU_MOE
inline bool qw38_cpu_moe()
{
    static const bool v = *detail::flag(475, 0);
    return v;
}

/// GR8_QW38_CPU_THREADS
inline int64_t qw38_cpu_threads()
{
    static const int64_t v = *detail::int_(476, 0);
    return v;
}

/// GR8_QW38_DENSE_FP8
inline bool qw38_dense_fp8()
{
    static const bool v = *detail::flag(477, 0);
    return v;
}

/// GR8_QW38_DFL
inline bool qw38_dfl()
{
    static const bool v = *detail::flag(478, 0);
    return v;
}

/// GR8_QW38_DFL_DP
inline int64_t qw38_dfl_dp()
{
    static const int64_t v = *detail::int_(479, 0);
    return v;
}

/// GR8_QW38_DFL_K
inline int64_t qw38_dfl_k()
{
    static const int64_t v = *detail::int_(480, 0);
    return v;
}

/// GR8_QW38_DFL_PACK
inline const std::optional<std::string>& qw38_dfl_pack()
{
    static const std::optional<std::string> v = detail::text(481, 0);
    return v;
}

/// GR8_QW38_EXPERT_CACHE_SLOTS
inline int64_t qw38_expert_cache_slots()
{
    static const int64_t v = *detail::int_(482, 0);
    return v;
}

/// GR8_QW38_EXPERT_PROFILE
inline const std::optional<std::string>& qw38_expert_profile()
{
    static const std::optional<std::string> v = detail::text(483, 0);
    return v;
}

/// GR8_QW38_FAST_AR
inline bool qw38_fast_ar()
{
    static const bool v = *detail::flag(484, 0);
    return v;
}

/// GR8_QW38_GRAPH
inline bool qw38_graph()
{
    static const bool v = *detail::flag(485, 0);
    return v;
}

/// GR8_QW38_HEAT_CAP
inline double qw38_heat_cap()
{
    static const double v = *detail::float_(486, 0);
    return v;
}

/// GR8_QW38_HEAT_DECAY
inline double qw38_heat_decay()
{
    static const double v = *detail::float_(487, 0);
    return v;
}

/// GR8_QW38_MIN_PROMOTE_HEAT
inline double qw38_min_promote_heat()
{
    static const double v = *detail::float_(488, 0);
    return v;
}

/// GR8_QW38_MTP_FAST
inline bool qw38_mtp_fast()
{
    static const bool v = *detail::flag(489, 0);
    return v;
}

/// GR8_QW38_NO_MTP
inline bool qw38_no_mtp()
{
    static const bool v = *detail::flag(490, 0);
    return v;
}

/// GR8_QW38_PF_CPU_FRAC
inline double qw38_pf_cpu_frac()
{
    static const double v = *detail::float_(491, 0);
    return v;
}

/// GR8_QW38_PF_CPU_MAX_M
inline int64_t qw38_pf_cpu_max_m()
{
    static const int64_t v = *detail::int_(492, 0);
    return v;
}

/// GR8_QW38_PF_D2D_PROMOTE
inline int64_t qw38_pf_d2d_promote()
{
    static const int64_t v = *detail::int_(493, 0);
    return v;
}

/// GR8_QW38_PF_MXFP8
inline bool qw38_pf_mxfp8()
{
    static const bool v = *detail::flag(494, 0);
    return v;
}

/// GR8_QW38_PF_STAGE_EXPS
inline int64_t qw38_pf_stage_exps()
{
    static const int64_t v = *detail::int_(495, 0);
    return v;
}

/// GR8_QW38_PF_WARM_HEAT
inline bool qw38_pf_warm_heat()
{
    static const bool v = *detail::flag(496, 0);
    return v;
}

/// GR8_QW38_PLE_ASYNC
inline bool qw38_ple_async()
{
    static const bool v = *detail::flag(497, 0);
    return v;
}

/// GR8_QW38_PLE_CACHE_MIB
inline int64_t qw38_ple_cache_mib()
{
    static const int64_t v = *detail::int_(498, 0);
    return v;
}

/// GR8_QW38_PLE_NO_WARM
inline bool qw38_ple_no_warm()
{
    static const bool v = *detail::flag(499, 0);
    return v;
}

/// GR8_QW38_PLE_PATH
inline const std::optional<std::string>& qw38_ple_path()
{
    static const std::optional<std::string> v = detail::text(500, 0);
    return v;
}

/// GR8_QW38_PLE_RAM
inline bool qw38_ple_ram()
{
    static const bool v = *detail::flag(501, 0);
    return v;
}

/// GR8_QW38_PROFILE
inline bool qw38_profile()
{
    static const bool v = *detail::flag(502, 0);
    return v;
}

/// GR8_QW38_PROMOTE_ASYNC
inline bool qw38_promote_async()
{
    static const bool v = *detail::flag(503, 0);
    return v;
}

/// GR8_QW38_PROMOTE_HYST
inline double qw38_promote_hyst()
{
    static const double v = *detail::float_(504, 0);
    return v;
}

/// GR8_QW38_PROMOTE_MAX
inline int64_t qw38_promote_max()
{
    static const int64_t v = *detail::int_(505, 0);
    return v;
}

/// GR8_QW38_STATIC_PIN_FRAC
inline double qw38_static_pin_frac()
{
    static const double v = *detail::float_(506, 0);
    return v;
}

/// GR8_QW38_STRATA
inline bool qw38_strata()
{
    static const bool v = *detail::flag(507, 0);
    return v;
}

/// GR8_QW38_TAP_DUMP
inline const std::optional<std::string>& qw38_tap_dump()
{
    static const std::optional<std::string> v = detail::text(508, 0);
    return v;
}

/// GR8_QW38_TRUNK_HEAD2
inline bool qw38_trunk_head2()
{
    static const bool v = *detail::flag(509, 0);
    return v;
}

/// GR8_QW38_VISION
inline bool qw38_vision()
{
    static const bool v = *detail::flag(510, 0);
    return v;
}

/// GR8_QW38_VRAM_RESERVE_MIB
inline int64_t qw38_vram_reserve_mib()
{
    static const int64_t v = *detail::int_(511, 0);
    return v;
}

/// GR8_QW38_WIDE
inline bool qw38_wide()
{
    static const bool v = *detail::flag(512, 0);
    return v;
}

/// GR8_QW38_WIDTHS
inline const std::optional<std::string>& qw38_widths()
{
    static const std::optional<std::string> v = detail::text(513, 0);
    return v;
}

/// GR8_REFUSAL_LINGER
inline std::optional<double> refusal_linger()
{
    static const std::optional<double> v = detail::float_(514, 0);
    return v;
}

/// GR8_RESID_DUMP
inline const std::optional<std::string>& resid_dump()
{
    static const std::optional<std::string> v = detail::text(515, 0);
    return v;
}

/// GR8_REUSE_ALIGN
inline std::optional<bool> reuse_align()
{
    static const std::optional<bool> v = detail::flag(516, 0);
    return v;
}

/// GR8_REUSE_GRID
inline std::optional<int64_t> reuse_grid()
{
    static const std::optional<int64_t> v = detail::int_(517, 0);
    return v;
}

/// GR8_REUSE_HDUMP
inline const std::optional<std::string>& reuse_hdump()
{
    static const std::optional<std::string> v = detail::text(518, 0);
    return v;
}

/// GR8_REUSE_HDUMP_SEQ
inline const std::optional<std::vector<int64_t>>& reuse_hdump_seq()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(519, 0);
    return v;
}

/// GR8_REUSE_NOEND
inline std::optional<bool> reuse_noend()
{
    static const std::optional<bool> v = detail::flag(520, 0);
    return v;
}

/// GR8_REUSE_POISON
inline bool reuse_poison()
{
    static const bool v = *detail::flag(521, 0);
    return v;
}

/// GR8_REUSE_STRICT
inline bool reuse_strict()
{
    static const bool v = *detail::flag(522, 0);
    return v;
}

/// GR8_REUSE_ZSCRATCH
inline bool reuse_zscratch()
{
    static const bool v = *detail::flag(523, 0);
    return v;
}

/// GR8_ROUTER_TPC
inline int64_t router_tpc()
{
    static const int64_t v = *detail::int_(524, 0);
    return v;
}

/// GR8_ROUTE_LOG
inline const std::optional<std::string>& route_log()
{
    static const std::optional<std::string> v = detail::text(525, 0);
    return v;
}

/// GR8_ROUTE_OVL
inline int64_t route_ovl()
{
    static const int64_t v = *detail::int_(526, 0);
    return v;
}

/// GR8_RT_CHECK
inline bool rt_check()
{
    static const bool v = *detail::flag(527, 0);
    return v;
}

/// GR8_RT_FMT
inline const std::string& rt_fmt()
{
    static const std::string v = *detail::text(528, 0);
    return v;
}

/// GR8_RT_L
inline std::optional<int64_t> rt_l()
{
    static const std::optional<int64_t> v = detail::int_(529, 0);
    return v;
}

/// GR8_RT_LOCAL
inline int64_t rt_local()
{
    static const int64_t v = *detail::int_(530, 0);
    return v;
}

/// GR8_RT_PF_SLICE
inline bool rt_pf_slice()
{
    static const bool v = *detail::flag(531, 0);
    return v;
}

/// GR8_RT_PLACE
inline std::optional<int64_t> rt_place()
{
    static const std::optional<int64_t> v = detail::int_(532, 0);
    return v;
}

/// GR8_RT_POOL
inline std::optional<int64_t> rt_pool()
{
    static const std::optional<int64_t> v = detail::int_(533, 0);
    return v;
}

/// GR8_RT_SELSTAT
inline bool rt_selstat()
{
    static const bool v = *detail::flag(534, 0);
    return v;
}

/// GR8_RT_SIZE
inline std::optional<int64_t> rt_size()
{
    static const std::optional<int64_t> v = detail::int_(535, 0);
    return v;
}

/// GR8_RT_STORE
inline const std::optional<std::string>& rt_store()
{
    static const std::optional<std::string> v = detail::text(536, 0);
    return v;
}

/// GR8_S2_ADAPT
inline bool s2_adapt()
{
    static const bool v = *detail::flag(537, 0);
    return v;
}

/// GR8_S2_ADMIT_HOLD
inline bool s2_admit_hold()
{
    static const bool v = *detail::flag(538, 0);
    return v;
}

/// GR8_S2_BELOW
inline std::optional<int64_t> s2_below()
{
    static const std::optional<int64_t> v = detail::int_(539, 0);
    return v;
}

/// GR8_S2_BIND_DELAY
inline const std::optional<std::string>& s2_bind_delay()
{
    static const std::optional<std::string> v = detail::text(540, 0);
    return v;
}

/// GR8_S2_DECODE_PF
inline int64_t s2_decode_pf()
{
    static const int64_t v = *detail::int_(541, 0);
    return v;
}

/// GR8_S2_DECODE_ROUNDS
inline int64_t s2_decode_rounds()
{
    static const int64_t v = *detail::int_(542, 0);
    return v;
}

/// GR8_S2_DSEL
inline bool s2_dsel()
{
    static const bool v = *detail::flag(543, 0);
    return v;
}

/// GR8_S2_DSEL_DWELL
inline int64_t s2_dsel_dwell()
{
    static const int64_t v = *detail::int_(544, 0);
    return v;
}

/// GR8_S2_DSEL_MARGIN
inline double s2_dsel_margin()
{
    static const double v = *detail::float_(545, 0);
    return v;
}

/// GR8_S2_DSEL_MODEL
inline bool s2_dsel_model()
{
    static const bool v = *detail::flag(546, 0);
    return v;
}

/// GR8_S2_DSEL_PROBE
inline int64_t s2_dsel_probe()
{
    static const int64_t v = *detail::int_(547, 0);
    return v;
}

/// GR8_S2_DSEL_TABLE
inline const std::optional<std::string>& s2_dsel_table()
{
    static const std::optional<std::string> v = detail::text(548, 0);
    return v;
}

/// GR8_S2_EVENTLOG
inline bool s2_eventlog()
{
    static const bool v = *detail::flag(549, 0);
    return v;
}

/// GR8_S2_FRESH_SPEC
inline bool s2_fresh_spec()
{
    static const bool v = *detail::flag(550, 0);
    return v;
}

/// GR8_S2_GROW
inline bool s2_grow()
{
    static const bool v = *detail::flag(551, 0);
    return v;
}

/// GR8_S2_NOTICE_MAX
inline int64_t s2_notice_max()
{
    static const int64_t v = *detail::int_(552, 0);
    return v;
}

/// GR8_S2_REPLAY_LOG
inline const std::optional<std::string>& s2_replay_log()
{
    static const std::optional<std::string> v = detail::text(553, 0);
    return v;
}

/// GR8_S2_ROUNDLOG
inline bool s2_roundlog()
{
    static const bool v = *detail::flag(554, 0);
    return v;
}

/// GR8_S2_SHARED_WATERFILL
inline bool s2_shared_waterfill()
{
    static const bool v = *detail::flag(555, 0);
    return v;
}

/// GR8_S2_SHARED_WF_MARGIN
inline double s2_shared_wf_margin()
{
    static const double v = *detail::float_(556, 0);
    return v;
}

/// GR8_S2_SHARE_MS
inline int64_t s2_share_ms()
{
    static const int64_t v = *detail::int_(557, 0);
    return v;
}

/// GR8_S2_TAIL_FIRST
inline bool s2_tail_first()
{
    static const bool v = *detail::flag(558, 0);
    return v;
}

/// GR8_S2_TIMELINE
inline int64_t s2_timeline()
{
    static const int64_t v = *detail::int_(559, 0);
    return v;
}

/// GR8_S2_TIMEOUT
inline int64_t s2_timeout()
{
    static const int64_t v = *detail::int_(560, 0);
    return v;
}

/// GR8_S2_TRACE
inline const std::optional<std::string>& s2_trace()
{
    static const std::optional<std::string> v = detail::text(561, 0);
    return v;
}

/// GR8_SESSIONS
inline bool sessions()
{
    static const bool v = *detail::flag(562, 0);
    return v;
}

/// GR8_SPEC_DRAFT
inline const std::vector<double>& spec_draft()
{
    static const std::vector<double> v = *detail::floats(563, 0);
    return v;
}

/// GR8_SPEC_K0
inline bool spec_k0()
{
    static const bool v = *detail::flag(564, 0);
    return v;
}

/// GR8_SPEC_TABLE
inline const std::optional<std::vector<double>>& spec_table()
{
    static const std::optional<std::vector<double>> v = detail::floats(565, 0);
    return v;
}

/// GR8_SPIN_TIMEOUT_S
inline std::optional<int64_t> spin_timeout_s()
{
    static const std::optional<int64_t> v = detail::int_(566, 0);
    return v;
}

/// GR8_STEP_TMR
inline bool step_tmr()
{
    static const bool v = *detail::flag(567, 0);
    return v;
}

/// GR8_STOP_GRACE
inline const std::optional<std::string>& stop_grace()
{
    static const std::optional<std::string> v = detail::text(568, 0);
    return v;
}

/// GR8_TESTS
inline const std::string& tests()
{
    static const std::string v = *detail::text(569, 0);
    return v;
}

/// GR8_TEST_CP_PF_BREAK
inline const std::optional<std::string>& test_cp_pf_break()
{
    static const std::optional<std::string> v = detail::text(570, 0);
    return v;
}

/// GR8_TEST_CP_PF_NET
inline bool test_cp_pf_net()
{
    static const bool v = *detail::flag(571, 0);
    return v;
}

/// GR8_TEST_DS41_KERNELS_DIR
inline const std::optional<std::string>& test_ds41_kernels_dir()
{
    static const std::optional<std::string> v = detail::text(572, 0);
    return v;
}

/// GR8_TEST_DS41_M4_PFW
inline int64_t test_ds41_m4_pfw()
{
    static const int64_t v = *detail::int_(573, 0);
    return v;
}

/// GR8_TEST_DS41_MGATE_DIAG
inline bool test_ds41_mgate_diag()
{
    static const bool v = *detail::flag(574, 0);
    return v;
}

/// GR8_TEST_DS41_TOKENIZER_DIR
inline const std::optional<std::string>& test_ds41_tokenizer_dir()
{
    static const std::optional<std::string> v = detail::text(575, 0);
    return v;
}

/// GR8_TEST_FUZZ_ROUNDS
inline int64_t test_fuzz_rounds()
{
    static const int64_t v = *detail::int_(576, 0);
    return v;
}

/// GR8_TEST_HY4_DUMP_FULL_LOGITS
inline bool test_hy4_dump_full_logits()
{
    static const bool v = *detail::flag(577, 0);
    return v;
}

/// GR8_TEST_HY4_ENGRAM_COMPACT_DIR
inline const std::optional<std::string>& test_hy4_engram_compact_dir()
{
    static const std::optional<std::string> v = detail::text(578, 0);
    return v;
}

/// GR8_TEST_IDX_DUMP
inline const std::optional<std::string>& test_idx_dump()
{
    static const std::optional<std::string> v = detail::text(579, 0);
    return v;
}

/// GR8_TEST_INK2_METADATA_DIR
inline const std::optional<std::string>& test_ink2_metadata_dir()
{
    static const std::optional<std::string> v = detail::text(580, 0);
    return v;
}

/// GR8_TEST_KVMAN_LEDGER
inline const std::string& test_kvman_ledger()
{
    static const std::string v = *detail::text(581, 0);
    return v;
}

/// GR8_TEST_KVSIM_ENTRIES
inline bool test_kvsim_entries()
{
    static const bool v = *detail::flag(582, 0);
    return v;
}

/// GR8_TEST_KV_HASH
inline bool test_kv_hash()
{
    static const bool v = *detail::flag(583, 0);
    return v;
}

/// GR8_TEST_KV_RANKS_CMP
inline bool test_kv_ranks_cmp()
{
    static const bool v = *detail::flag(584, 0);
    return v;
}

/// GR8_TEST_KV_TIER_CHECK
inline const std::optional<std::string>& test_kv_tier_check()
{
    static const std::optional<std::string> v = detail::text(585, 0);
    return v;
}

/// GR8_TEST_KV_TIER_CHECK_DROP
inline std::optional<int64_t> test_kv_tier_check_drop()
{
    static const std::optional<int64_t> v = detail::int_(586, 0);
    return v;
}

/// GR8_TEST_LEDGER_CRASH_AT
inline const std::optional<std::string>& test_ledger_crash_at()
{
    static const std::optional<std::string> v = detail::text(587, 0);
    return v;
}

/// GR8_TEST_MTP_LASTROW_CHECK
inline bool test_mtp_lastrow_check()
{
    static const bool v = *detail::flag(588, 0);
    return v;
}

/// GR8_TEST_PERTURB
inline double test_perturb()
{
    static const double v = *detail::float_(589, 0);
    return v;
}

/// GR8_TEST_PF_DRAFTDUMP
inline const std::optional<std::string>& test_pf_draftdump()
{
    static const std::optional<std::string> v = detail::text(590, 0);
    return v;
}

/// GR8_TEST_PF_HASH_POKE
inline const std::optional<std::vector<int64_t>>& test_pf_hash_poke()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(591, 0);
    return v;
}

/// GR8_TEST_PF_REPLAY_VERBOSE
inline bool test_pf_replay_verbose()
{
    static const bool v = *detail::flag(592, 0);
    return v;
}

/// GR8_TEST_PF_RESDUMP
inline const std::optional<std::string>& test_pf_resdump()
{
    static const std::optional<std::string> v = detail::text(593, 0);
    return v;
}

/// GR8_TEST_PF_TIMES
inline bool test_pf_times()
{
    static const bool v = *detail::flag(594, 0);
    return v;
}

/// GR8_TEST_ROW_LINES
inline bool test_row_lines()
{
    static const bool v = *detail::flag(595, 0);
    return v;
}

/// GR8_TEST_RT_ADOPT_SLOT
inline bool test_rt_adopt_slot()
{
    static const bool v = *detail::flag(596, 0);
    return v;
}

/// GR8_TEST_RT_POISON
inline bool test_rt_poison()
{
    static const bool v = *detail::flag(597, 0);
    return v;
}

/// GR8_TEST_S2_ATTN_PERSESSION
inline bool test_s2_attn_persession()
{
    static const bool v = *detail::flag(598, 0);
    return v;
}

/// GR8_TEST_S2_CLS
inline int64_t test_s2_cls()
{
    static const int64_t v = *detail::int_(599, 0);
    return v;
}

/// GR8_TEST_S2_GEN
inline const std::optional<std::vector<int64_t>>& test_s2_gen()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(600, 0);
    return v;
}

/// GR8_TEST_S2_GROW_POOL
inline std::optional<int64_t> test_s2_grow_pool()
{
    static const std::optional<int64_t> v = detail::int_(601, 0);
    return v;
}

/// GR8_TEST_S2_INSERT
inline const std::optional<std::string>& test_s2_insert()
{
    static const std::optional<std::string> v = detail::text(602, 0);
    return v;
}

/// GR8_TEST_S2_PAGES
inline const std::optional<std::vector<int64_t>>& test_s2_pages()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(603, 0);
    return v;
}

/// GR8_TEST_S2_PFBASE
inline const std::optional<std::vector<int64_t>>& test_s2_pfbase()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(604, 0);
    return v;
}

/// GR8_TEST_S2_PFNOFIX
inline bool test_s2_pfnofix()
{
    static const bool v = *detail::flag(605, 0);
    return v;
}

/// GR8_TEST_S2_PFPOISON
inline bool test_s2_pfpoison()
{
    static const bool v = *detail::flag(606, 0);
    return v;
}

/// GR8_TEST_S2_POOL
inline std::optional<int64_t> test_s2_pool()
{
    static const std::optional<int64_t> v = detail::int_(607, 0);
    return v;
}

/// GR8_TEST_S2_RELOC
inline bool test_s2_reloc()
{
    static const bool v = *detail::flag(608, 0);
    return v;
}

/// GR8_TEST_S2_RELOC_PAGES
inline std::optional<int64_t> test_s2_reloc_pages()
{
    static const std::optional<int64_t> v = detail::int_(609, 0);
    return v;
}

/// GR8_TEST_S2_ROWHASH
inline bool test_s2_rowhash()
{
    static const bool v = *detail::flag(610, 0);
    return v;
}

/// GR8_TEST_S2_RUNS
inline const std::optional<std::string>& test_s2_runs()
{
    static const std::optional<std::string> v = detail::text(611, 0);
    return v;
}

/// GR8_TEST_S2_SHRINK_AT
inline std::optional<int64_t> test_s2_shrink_at()
{
    static const std::optional<int64_t> v = detail::int_(612, 0);
    return v;
}

/// GR8_TEST_S2_SOLO1
inline bool test_s2_solo1()
{
    static const bool v = *detail::flag(613, 0);
    return v;
}

/// GR8_TEST_S2_XLIST
inline const std::optional<std::vector<int64_t>>& test_s2_xlist()
{
    static const std::optional<std::vector<int64_t>> v = detail::ints(614, 0);
    return v;
}

/// GR8_TEST_TOKENIZER_DIR
inline const std::optional<std::string>& test_tokenizer_dir()
{
    static const std::optional<std::string> v = detail::text(615, 0);
    return v;
}

/// GR8_TEST_VISION_CASES
inline const std::optional<std::string>& test_vision_cases()
{
    static const std::optional<std::string> v = detail::text(616, 0);
    return v;
}

/// GR8_TEST_ZERO_DELAY_MS
inline int64_t test_zero_delay_ms()
{
    static const int64_t v = *detail::int_(617, 0);
    return v;
}

/// TMPDIR
inline const std::string& tmpdir()
{
    static const std::string v = *detail::text(618, 0);
    return v;
}

/// GR8_TR3_W2ALL
inline bool tr3_w2all()
{
    static const bool v = *detail::flag(619, 0);
    return v;
}

/// GR8_TR3_WIDE4
inline bool tr3_wide4()
{
    static const bool v = *detail::flag(620, 0);
    return v;
}

/// GR8_V2_SHARED
inline const std::optional<std::string>& v2_shared()
{
    static const std::optional<std::string> v = detail::text(621, 0);
    return v;
}

/// GR8_WARM
inline int64_t warm()
{
    static const int64_t v = *detail::int_(622, 0);
    return v;
}

/// GR8_WEIGHT_CHECK
inline bool weight_check()
{
    static const bool v = *detail::flag(623, 0);
    return v;
}

/// GR8_WEIGHT_CHECK_FLIP
inline const std::optional<std::string>& weight_check_flip()
{
    static const std::optional<std::string> v = detail::text(624, 0);
    return v;
}

/// GR8_WIDTHS
inline const std::optional<std::string>& widths()
{
    static const std::optional<std::string> v = detail::text(625, 0);
    return v;
}

/// GR8_WORK_DIR
inline const std::optional<std::string>& work_dir()
{
    static const std::optional<std::string> v = detail::text(626, 0);
    return v;
}

/// GR8_XGRAMMAR
inline const std::optional<std::string>& xgrammar()
{
    static const std::optional<std::string> v = detail::text(627, 0);
    return v;
}

/// GR8_XGRAMMAR_LIB
inline const std::optional<std::string>& xgrammar_lib()
{
    static const std::optional<std::string> v = detail::text(628, 0);
    return v;
}

/// GR8_XTRACE
inline bool xtrace()
{
    static const bool v = *detail::flag(629, 0);
    return v;
}

/// GR8_XTRACE_OUT
inline const std::optional<std::string>& xtrace_out()
{
    static const std::optional<std::string> v = detail::text(630, 0);
    return v;
}

/// GR8_XTRACE_PF
inline bool xtrace_pf()
{
    static const bool v = *detail::flag(631, 0);
    return v;
}

/// GR8_XTRACE_QUIT
inline const std::optional<std::string>& xtrace_quit()
{
    static const std::optional<std::string> v = detail::text(632, 0);
    return v;
}

} // namespace knob
