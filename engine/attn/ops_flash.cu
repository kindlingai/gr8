// The GLM-5.3-Flash attention table (ops.h). The build links this file into every binary that runs the
// model through the runtime; a unit test that includes the model's .cu calls its g6a_* functions instead.
#include <cstdio>
#include <cstdlib>
#include "ops.h"

// the CP entry points exist only in G6_CP builds, and only CP code calls them
#if defined(G6_CP) && G6_CP
#define G6_CP_OP(f) f
#else
#define G6_CP_OP(f) nullptr
#endif

extern "C" const G6AttnOps* g6_attn_ops(void) {
    static const G6AttnOps ops = {
        "GLM-5.3-Flash",
        g6a_append,   // append
        nullptr,   // cp_finish
        nullptr,   // cp_front
        nullptr,   // cp_group
        nullptr,   // cp_groups
        nullptr,   // cp_groups_bytes
        nullptr,   // cp_groups_ok
        nullptr,   // cp_index
        nullptr,   // cp_merge_index
        nullptr,   // cp_qabs
        nullptr,   // ctx_sp
        g6a_index_scores,   // index_scores
        nullptr,   // posidx
        nullptr,   // posidx_takes
        nullptr,   // rope_init
        g6a_run,   // run
        g6a_scratch_bytes,   // scratch_bytes
        nullptr,   // rt_pack_int4
        nullptr,   // rt_unpack_int4
        g6a_bind_create,   // bind_create
        G6_CP_OP(g6a_bind_create_cp),   // bind_create_cp
        nullptr,   // bind_create_cp_phys
        g6a_bind_create_ext,   // bind_create_ext
        g6a_bind_prepare,   // bind_prepare
        g6a_bind_release,   // bind_release
        g6a_bind_replay_init,   // bind_replay_init
        g6a_bind_reset,   // bind_reset
        g6a_bind_run,   // bind_run
        g6a_bind_scratch_need,   // bind_scratch_need
        nullptr,   // kvb_bf16
        nullptr,   // bind_baseline_trace
        g6a_bind_index_append,   // bind_index_append
        g6a_bind_prefill_bytes,   // bind_prefill_bytes
        g6a_bind_prefill_create,   // bind_prefill_create
        g6a_bind_prefill_create_shared,   // bind_prefill_create_shared
        g6a_bind_prefill_get_config,   // bind_prefill_get_config
        nullptr,   // bind_prefill_index_rows
        g6a_bind_prefill_profile_read,   // bind_prefill_profile_read
        g6a_bind_prefill_release,   // bind_prefill_release
        g6a_bind_prefill_run,   // bind_prefill_run
        g6a_bind_prefill_run_owned,   // bind_prefill_run_owned
        nullptr,   // bind_prefill_trace
        nullptr,   // qo_bytes
        nullptr,   // qo_cap
        nullptr,   // qo_create
        nullptr,   // qo_load
        nullptr,   // qo_prof
        nullptr,   // qo_release
        nullptr,   // qo_run
        nullptr,   // qo_warm
        g6a_replay_run,   // replay_run
    };
    return &ops;
}
extern "C" void g6_attn_missing(const char* op) {
    fprintf(stderr, "g6: attention op %s is not in the GLM-5.3-Flash engine\n", op);
    abort();
}
