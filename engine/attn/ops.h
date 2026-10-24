#pragma once
// The attention interface as data: each model fills one G6AttnOps (ops_glm.cu for GLM-5.3, ops_flash.cu for
// GLM-5.3-Flash), and the runtime calls the g6_attn_* wrappers below, which go through it. An op a model does not
// have stays null: its wrapper returns what a model without it gives (NOTSUP, 0 or nothing), or stops with the op's
// name when the runtime cannot do without it. So the runtime links only g6_attn_ops, and an op one model adds cannot
// break the other model's link. The implementations are the g6a_* functions the headers declare.
//
// Adding an op: declare its g6a_* implementation in the header for its area, add a member here and a wrapper below
// (a default when the runtime can do without it, g6_attn_missing when it cannot), and set the member in ops_glm.cu and
// ops_flash.cu: the model's g6a_* function, or nullptr.
#include "prefill.h"
#include "prefill-qo-bind.h"

#define G6_ATTN_NOTSUP 801   // cudaErrorNotSupported
struct G6FlashPoolOwner;   // prefill.h: Flash builds only

struct G6AttnOps {
    const char* model;
    int (*append)(const G6Attn*, void*);
    int (*cp_finish)(const G6Attn*, const G6AttnStepCell*, void*, size_t, const float*, void*);
    int (*cp_front)(const G6Attn*, const G6AttnStepCell*, void*, size_t, void*);
    int (*cp_group)(const G6Attn*, const G6AttnStepCell*, void*, size_t, const float*, const float*, float*, void*);
    int (*cp_groups)(const G6Attn*, const G6AttnStepCell*, void*, size_t, const float*, const float*, float*, int, int64_t, void*);
    size_t (*cp_groups_bytes)(int, int);
    int (*cp_groups_ok)(int);
    int (*cp_index)(const G6Attn*, const G6AttnStepCell*, void*, size_t, uint64_t*, void*);
    int (*cp_merge_index)(const G6Attn*, const G6AttnStepCell*, const uint64_t*, void*);
    int (*cp_qabs)(const G6Attn*, void*, const float**, const float**);
    int (*ctx_sp)();
    int (*index_scores)(const G6Attn*, void*, size_t, float*, void*);
    int (*posidx)();
    int (*posidx_takes)(const G6Attn*);
    int (*rope_init)(int, void*);
    int (*run)(const G6Attn*, void*, size_t, void*);
    size_t (*scratch_bytes)(const G6Attn*);
    int (*rt_pack_int4)(const void*, size_t, void*, size_t, int, void*);
    int (*rt_unpack_int4)(const void*, size_t, void*, size_t, int, void*);
    int (*bind_create)(G6AttnBinding*, const G6AttnPack*, int, int, int, int, int, void*);
    int (*bind_create_cp)(G6AttnBinding*, const G6AttnPack*, int, int, int, int, int, int, int, void*, void*, size_t);
    int (*bind_create_cp_phys)(G6AttnBinding*, const G6AttnPack*, int, int, int, int, int, int, int, int, void*, void*, size_t);
    int (*bind_create_ext)(G6AttnBinding*, const G6AttnPack*, int, int, int, int, int, void*, void*, size_t);
    int (*bind_prepare)(G6AttnBinding*, const G6AttnStep*, void*);
    int (*bind_release)(G6AttnBinding*);
    int (*bind_replay_init)(G6AttnBinding*, const G6AttnStep*, G6AttnStepCell*, int, G6AttnReplay*, void*);
    int (*bind_reset)(G6AttnBinding*, void*);
    int (*bind_run)(G6AttnBinding*, const G6AttnStep*, void*);
    size_t (*bind_scratch_need)(int, int, int, int);
    int (*kvb_bf16)(const void*, const float*, void*, int, void*);
    int (*bind_baseline_trace)(G6AttnBinding*, const G6AttnStep*, const G6AttnTrace*, void*);
    int (*bind_index_append)(G6AttnBinding*, const G6AttnStep*, void*);
    size_t (*bind_prefill_bytes)(const G6AttnPrefill*);
    int (*bind_prefill_create)(G6AttnPrefill**, G6AttnBinding*, void*);
    int (*bind_prefill_create_shared)(G6AttnPrefill**, G6AttnBinding*, void*, G6AttnPrefill*);
    int (*bind_prefill_get_config)(const G6AttnPrefill*, int*, int*);
    int (*bind_prefill_index_rows)(G6AttnPrefill*, G6AttnBinding*, const G6AttnStep*, int*, float*, void*);
    int (*bind_prefill_profile_read)(G6AttnPrefill*, G6AttnPrefillProfile*);
    int (*bind_prefill_release)(G6AttnPrefill*);
    int (*bind_prefill_run)(G6AttnPrefill*, G6AttnBinding*, const G6AttnStep*, void*);
    int (*bind_prefill_run_owned)(G6AttnPrefill*, G6AttnBinding*, const G6AttnStep*, void*, const G6FlashPoolOwner*);
    int (*bind_prefill_trace)(G6AttnPrefill*, G6AttnBinding*, const G6AttnStep*, const G6AttnTrace*, void*);
    size_t (*qo_bytes)(const G6AttnQo*);
    void (*qo_cap)(int);
    int (*qo_create)(G6AttnQo**, const G6AttnBinding*, const G6AttnTensor*, const G6AttnTensor*, int, void*, G6AttnQo*);
    int (*qo_load)(G6AttnQo*, const G6AttnTensor*, const G6AttnTensor*, void*);
    void (*qo_prof)(int, double*);
    int (*qo_release)(G6AttnQo*);
    int (*qo_run)(G6AttnQo*, G6AttnBinding*, const G6AttnQoStep*, void*);
    int (*qo_warm)(G6AttnQo*, int, void*);
    int (*replay_run)(const G6AttnReplay*, void*);
};

// The linked model's table (one per binary). The runtime reaches every op through it.
extern "C" const G6AttnOps* g6_attn_ops(void);
// Stops the process: the linked model has no `op` and the caller cannot go on without it.
extern "C" [[noreturn]] void g6_attn_missing(const char* op);

inline int g6_attn_append(const G6Attn* a0, void* a1) { auto f = g6_attn_ops()->append; if (!f) g6_attn_missing("append"); return f(a0, a1); }
inline int g6_attn_cp_finish(const G6Attn* a0, const G6AttnStepCell* a1, void* a2, size_t a3, const float* a4, void* a5) { auto f = g6_attn_ops()->cp_finish; return f ? f(a0, a1, a2, a3, a4, a5) : G6_ATTN_NOTSUP; }
inline int g6_attn_cp_front(const G6Attn* a0, const G6AttnStepCell* a1, void* a2, size_t a3, void* a4) { auto f = g6_attn_ops()->cp_front; return f ? f(a0, a1, a2, a3, a4) : G6_ATTN_NOTSUP; }
inline int g6_attn_cp_group(const G6Attn* a0, const G6AttnStepCell* a1, void* a2, size_t a3, const float* a4, const float* a5, float* a6, void* a7) { auto f = g6_attn_ops()->cp_group; return f ? f(a0, a1, a2, a3, a4, a5, a6, a7) : G6_ATTN_NOTSUP; }
inline int g6_attn_cp_groups(const G6Attn* a0, const G6AttnStepCell* a1, void* a2, size_t a3, const float* a4, const float* a5, float* a6, int a7, int64_t a8, void* a9) { auto f = g6_attn_ops()->cp_groups; return f ? f(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) : G6_ATTN_NOTSUP; }
inline size_t g6_attn_cp_groups_bytes(int a0, int a1) { auto f = g6_attn_ops()->cp_groups_bytes; return f ? f(a0, a1) : 0; }
inline int g6_attn_cp_groups_ok(int a0) { auto f = g6_attn_ops()->cp_groups_ok; return f ? f(a0) : 0; }
inline int g6_attn_cp_index(const G6Attn* a0, const G6AttnStepCell* a1, void* a2, size_t a3, uint64_t* a4, void* a5) { auto f = g6_attn_ops()->cp_index; return f ? f(a0, a1, a2, a3, a4, a5) : G6_ATTN_NOTSUP; }
inline int g6_attn_cp_merge_index(const G6Attn* a0, const G6AttnStepCell* a1, const uint64_t* a2, void* a3) { auto f = g6_attn_ops()->cp_merge_index; return f ? f(a0, a1, a2, a3) : G6_ATTN_NOTSUP; }
inline int g6_attn_cp_qabs(const G6Attn* a0, void* a1, const float** a2, const float** a3) { auto f = g6_attn_ops()->cp_qabs; return f ? f(a0, a1, a2, a3) : G6_ATTN_NOTSUP; }
inline int g6_attn_ctx_sp() { auto f = g6_attn_ops()->ctx_sp; return f ? f() : 0; }
inline int g6_attn_index_scores(const G6Attn* a0, void* a1, size_t a2, float* a3, void* a4) { auto f = g6_attn_ops()->index_scores; if (!f) g6_attn_missing("index_scores"); return f(a0, a1, a2, a3, a4); }
inline int g6_attn_posidx() { auto f = g6_attn_ops()->posidx; return f ? f() : 0; }
inline int g6_attn_posidx_takes(const G6Attn* a0) { auto f = g6_attn_ops()->posidx_takes; return f ? f(a0) : 0; }
inline int g6_attn_rope_init(int a0, void* a1) { auto f = g6_attn_ops()->rope_init; return f ? f(a0, a1) : 0; }
inline int g6_attn_run(const G6Attn* a0, void* a1, size_t a2, void* a3) { auto f = g6_attn_ops()->run; if (!f) g6_attn_missing("run"); return f(a0, a1, a2, a3); }
inline size_t g6_attn_scratch_bytes(const G6Attn* a0) { auto f = g6_attn_ops()->scratch_bytes; if (!f) g6_attn_missing("scratch_bytes"); return f(a0); }
inline int g6_rt_pack_int4(const void* a0, size_t a1, void* a2, size_t a3, int a4, void* a5) { auto f = g6_attn_ops()->rt_pack_int4; return f ? f(a0, a1, a2, a3, a4, a5) : G6_ATTN_NOTSUP; }
inline int g6_rt_unpack_int4(const void* a0, size_t a1, void* a2, size_t a3, int a4, void* a5) { auto f = g6_attn_ops()->rt_unpack_int4; return f ? f(a0, a1, a2, a3, a4, a5) : G6_ATTN_NOTSUP; }
inline int g6_attn_bind_create(G6AttnBinding* a0, const G6AttnPack* a1, int a2, int a3, int a4, int a5, int a6, void* a7) { auto f = g6_attn_ops()->bind_create; if (!f) g6_attn_missing("bind_create"); return f(a0, a1, a2, a3, a4, a5, a6, a7); }
inline int g6_attn_bind_create_cp(G6AttnBinding* a0, const G6AttnPack* a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, void* a9, void* a10, size_t a11) { auto f = g6_attn_ops()->bind_create_cp; if (!f) g6_attn_missing("bind_create_cp"); return f(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11); }
inline int g6_attn_bind_create_cp_phys(G6AttnBinding* a0, const G6AttnPack* a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, void* a10, void* a11, size_t a12) { auto f = g6_attn_ops()->bind_create_cp_phys; if (!f) g6_attn_missing("bind_create_cp_phys"); return f(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12); }
inline int g6_attn_bind_create_ext(G6AttnBinding* a0, const G6AttnPack* a1, int a2, int a3, int a4, int a5, int a6, void* a7, void* a8, size_t a9) { auto f = g6_attn_ops()->bind_create_ext; if (!f) g6_attn_missing("bind_create_ext"); return f(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }
inline int g6_attn_bind_prepare(G6AttnBinding* a0, const G6AttnStep* a1, void* a2) { auto f = g6_attn_ops()->bind_prepare; if (!f) g6_attn_missing("bind_prepare"); return f(a0, a1, a2); }
inline int g6_attn_bind_release(G6AttnBinding* a0) { auto f = g6_attn_ops()->bind_release; if (!f) g6_attn_missing("bind_release"); return f(a0); }
inline int g6_attn_bind_replay_init(G6AttnBinding* a0, const G6AttnStep* a1, G6AttnStepCell* a2, int a3, G6AttnReplay* a4, void* a5) { auto f = g6_attn_ops()->bind_replay_init; if (!f) g6_attn_missing("bind_replay_init"); return f(a0, a1, a2, a3, a4, a5); }
inline int g6_attn_bind_reset(G6AttnBinding* a0, void* a1) { auto f = g6_attn_ops()->bind_reset; if (!f) g6_attn_missing("bind_reset"); return f(a0, a1); }
inline int g6_attn_bind_run(G6AttnBinding* a0, const G6AttnStep* a1, void* a2) { auto f = g6_attn_ops()->bind_run; if (!f) g6_attn_missing("bind_run"); return f(a0, a1, a2); }
inline size_t g6_attn_bind_scratch_need(int a0, int a1, int a2, int a3) { auto f = g6_attn_ops()->bind_scratch_need; if (!f) g6_attn_missing("bind_scratch_need"); return f(a0, a1, a2, a3); }
inline int g6_attn_kvb_bf16(const void* a0, const float* a1, void* a2, int a3, void* a4) { auto f = g6_attn_ops()->kvb_bf16; return f ? f(a0, a1, a2, a3, a4) : G6_ATTN_NOTSUP; }
inline int g6_attn_bind_baseline_trace(G6AttnBinding* a0, const G6AttnStep* a1, const G6AttnTrace* a2, void* a3) { auto f = g6_attn_ops()->bind_baseline_trace; return f ? f(a0, a1, a2, a3) : G6_ATTN_NOTSUP; }
inline int g6_attn_bind_index_append(G6AttnBinding* a0, const G6AttnStep* a1, void* a2) { auto f = g6_attn_ops()->bind_index_append; if (!f) g6_attn_missing("bind_index_append"); return f(a0, a1, a2); }
inline size_t g6_attn_bind_prefill_bytes(const G6AttnPrefill* a0) { auto f = g6_attn_ops()->bind_prefill_bytes; if (!f) g6_attn_missing("bind_prefill_bytes"); return f(a0); }
inline int g6_attn_bind_prefill_create(G6AttnPrefill** a0, G6AttnBinding* a1, void* a2) { auto f = g6_attn_ops()->bind_prefill_create; if (!f) g6_attn_missing("bind_prefill_create"); return f(a0, a1, a2); }
inline int g6_attn_bind_prefill_create_shared(G6AttnPrefill** a0, G6AttnBinding* a1, void* a2, G6AttnPrefill* a3) { auto f = g6_attn_ops()->bind_prefill_create_shared; if (!f) g6_attn_missing("bind_prefill_create_shared"); return f(a0, a1, a2, a3); }
inline int g6_attn_bind_prefill_get_config(const G6AttnPrefill* a0, int* a1, int* a2) { auto f = g6_attn_ops()->bind_prefill_get_config; if (!f) g6_attn_missing("bind_prefill_get_config"); return f(a0, a1, a2); }
inline int g6_attn_bind_prefill_index_rows(G6AttnPrefill* a0, G6AttnBinding* a1, const G6AttnStep* a2, int* a3, float* a4, void* a5) { auto f = g6_attn_ops()->bind_prefill_index_rows; return f ? f(a0, a1, a2, a3, a4, a5) : G6_ATTN_NOTSUP; }
inline int g6_attn_bind_prefill_profile_read(G6AttnPrefill* a0, G6AttnPrefillProfile* a1) { auto f = g6_attn_ops()->bind_prefill_profile_read; if (!f) g6_attn_missing("bind_prefill_profile_read"); return f(a0, a1); }
inline int g6_attn_bind_prefill_release(G6AttnPrefill* a0) { auto f = g6_attn_ops()->bind_prefill_release; if (!f) g6_attn_missing("bind_prefill_release"); return f(a0); }
inline int g6_attn_bind_prefill_run(G6AttnPrefill* a0, G6AttnBinding* a1, const G6AttnStep* a2, void* a3) { auto f = g6_attn_ops()->bind_prefill_run; if (!f) g6_attn_missing("bind_prefill_run"); return f(a0, a1, a2, a3); }
inline int g6_attn_bind_prefill_run_owned(G6AttnPrefill* a0, G6AttnBinding* a1, const G6AttnStep* a2, void* a3, const G6FlashPoolOwner* a4) { auto f = g6_attn_ops()->bind_prefill_run_owned; if (!f) g6_attn_missing("bind_prefill_run_owned"); return f(a0, a1, a2, a3, a4); }
inline int g6_attn_bind_prefill_trace(G6AttnPrefill* a0, G6AttnBinding* a1, const G6AttnStep* a2, const G6AttnTrace* a3, void* a4) { auto f = g6_attn_ops()->bind_prefill_trace; return f ? f(a0, a1, a2, a3, a4) : G6_ATTN_NOTSUP; }
inline size_t g6_attn_qo_bytes(const G6AttnQo* a0) { auto f = g6_attn_ops()->qo_bytes; return f ? f(a0) : 0; }
inline void g6_attn_qo_cap(int a0) { if (auto f = g6_attn_ops()->qo_cap) f(a0); }
inline int g6_attn_qo_create(G6AttnQo** a0, const G6AttnBinding* a1, const G6AttnTensor* a2, const G6AttnTensor* a3, int a4, void* a5, G6AttnQo* a6) { auto f = g6_attn_ops()->qo_create; return f ? f(a0, a1, a2, a3, a4, a5, a6) : G6_ATTN_NOTSUP; }
inline int g6_attn_qo_load(G6AttnQo* a0, const G6AttnTensor* a1, const G6AttnTensor* a2, void* a3) { auto f = g6_attn_ops()->qo_load; return f ? f(a0, a1, a2, a3) : G6_ATTN_NOTSUP; }
inline void g6_attn_qo_prof(int a0, double* a1) { if (auto f = g6_attn_ops()->qo_prof) f(a0, a1); }
inline int g6_attn_qo_release(G6AttnQo* a0) { auto f = g6_attn_ops()->qo_release; return f ? f(a0) : 0; }
inline int g6_attn_qo_run(G6AttnQo* a0, G6AttnBinding* a1, const G6AttnQoStep* a2, void* a3) { auto f = g6_attn_ops()->qo_run; return f ? f(a0, a1, a2, a3) : G6_ATTN_NOTSUP; }
inline int g6_attn_qo_warm(G6AttnQo* a0, int a1, void* a2) { auto f = g6_attn_ops()->qo_warm; return f ? f(a0, a1, a2) : G6_ATTN_NOTSUP; }
inline int g6_attn_replay_run(const G6AttnReplay* a0, void* a1) { auto f = g6_attn_ops()->replay_run; if (!f) g6_attn_missing("replay_run"); return f(a0, a1); }
