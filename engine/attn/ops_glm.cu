// The GLM-5.3 attention table (ops.h). The build links this file into every binary that runs the
// model through the runtime; a unit test that includes the model's .cu calls its g6a_* functions instead.
#include <cstdio>
#include <cstdlib>
#include "ops.h"

extern "C" const G6AttnOps* g6_attn_ops(void) {
    static const G6AttnOps ops = {
        "GLM-5.3",
        g6a_append,   // append
        g6a_cp_finish,   // cp_finish
        g6a_cp_front,   // cp_front
        g6a_cp_group,   // cp_group
        g6a_cp_groups,   // cp_groups
        g6a_cp_groups_bytes,   // cp_groups_bytes
        g6a_cp_groups_ok,   // cp_groups_ok
        g6a_cp_index,   // cp_index
        g6a_cp_merge_index,   // cp_merge_index
        g6a_cp_qabs,   // cp_qabs
        g6a_ctx_sp,   // ctx_sp
        g6a_index_scores,   // index_scores
        g6a_posidx,   // posidx
        g6a_posidx_takes,   // posidx_takes
        g6a_rope_init,   // rope_init
        g6a_run,   // run
        g6a_scratch_bytes,   // scratch_bytes
        g6a_rt_pack_int4,   // rt_pack_int4
        g6a_rt_unpack_int4,   // rt_unpack_int4
        g6a_bind_create,   // bind_create
        g6a_bind_create_cp,   // bind_create_cp
        g6a_bind_create_cp_phys,   // bind_create_cp_phys
        g6a_bind_create_ext,   // bind_create_ext
        g6a_bind_prepare,   // bind_prepare
        g6a_bind_release,   // bind_release
        g6a_bind_replay_init,   // bind_replay_init
        g6a_bind_reset,   // bind_reset
        g6a_bind_run,   // bind_run
        g6a_bind_scratch_need,   // bind_scratch_need
        g6a_kvb_bf16,   // kvb_bf16
        g6a_bind_baseline_trace,   // bind_baseline_trace
        g6a_bind_index_append,   // bind_index_append
        g6a_bind_prefill_bytes,   // bind_prefill_bytes
        g6a_bind_prefill_create,   // bind_prefill_create
        g6a_bind_prefill_create_shared,   // bind_prefill_create_shared
        g6a_bind_prefill_get_config,   // bind_prefill_get_config
        g6a_bind_prefill_index_rows,   // bind_prefill_index_rows
        g6a_bind_prefill_profile_read,   // bind_prefill_profile_read
        g6a_bind_prefill_release,   // bind_prefill_release
        g6a_bind_prefill_run,   // bind_prefill_run
        nullptr,   // bind_prefill_run_owned
        g6a_bind_prefill_trace,   // bind_prefill_trace
        g6a_qo_bytes,   // qo_bytes
        g6a_qo_cap,   // qo_cap
        g6a_qo_create,   // qo_create
        g6a_qo_load,   // qo_load
        g6a_qo_prof,   // qo_prof
        g6a_qo_release,   // qo_release
        g6a_qo_run,   // qo_run
        g6a_qo_warm,   // qo_warm
        g6a_replay_run,   // replay_run
    };
    return &ops;
}
extern "C" void g6_attn_missing(const char* op) {
    fprintf(stderr, "g6: attention op %s is not in the GLM-5.3 engine\n", op);
    abort();
}
