// The DeepSeek-V4.1-Flash-1017 attention table (ops.h), every slot nullptr: an op the table lacks answers G6_ATTN_NOTSUP,
// 0 or nothing, or aborts through g6_attn_missing, which is how the engine boots before all its ops exist (the Flash way).
// Owners: the attention block fills run / append / scratch_bytes / bind_* for decode (engine/attn/ds41_attn.cu), the indexer the index, prefill and
// KV-row slots (engine/attn/ds41_bind.cu); a slot is one line "g6a_x,   // x" replacing its "nullptr". docs/DEEPSEEK.md, file checklist.
#include <cstdio>
#include <cstdlib>
#include "ops.h"


extern "C" const G6AttnOps* g6_attn_ops(void) {
    static const G6AttnOps ops = {
        "DeepSeek-V4.1-Flash-1017",
        nullptr,   // append
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
        nullptr,   // index_scores
        nullptr,   // posidx
        nullptr,   // posidx_takes
        nullptr,   // rope_init
        nullptr,   // run
        nullptr,   // scratch_bytes
        nullptr,   // rt_pack_int4
        nullptr,   // rt_unpack_int4
        nullptr,   // bind_create
        nullptr,   // bind_create_cp
        nullptr,   // bind_create_cp_phys
        nullptr,   // bind_create_ext
        nullptr,   // bind_prepare
        nullptr,   // bind_release
        nullptr,   // bind_replay_init
        nullptr,   // bind_reset
        nullptr,   // bind_run
        nullptr,   // bind_scratch_need
        nullptr,   // kvb_bf16
        nullptr,   // bind_baseline_trace
        nullptr,   // bind_index_append
        nullptr,   // bind_prefill_bytes
        nullptr,   // bind_prefill_create
        nullptr,   // bind_prefill_create_shared
        nullptr,   // bind_prefill_get_config
        nullptr,   // bind_prefill_index_rows
        nullptr,   // bind_prefill_profile_read
        nullptr,   // bind_prefill_release
        nullptr,   // bind_prefill_run
        nullptr,   // bind_prefill_run_owned
        nullptr,   // bind_prefill_trace
        nullptr,   // qo_bytes
        nullptr,   // qo_cap
        nullptr,   // qo_create
        nullptr,   // qo_load
        nullptr,   // qo_prof
        nullptr,   // qo_release
        nullptr,   // qo_run
        nullptr,   // qo_warm
        nullptr,   // replay_run
        nullptr,   // bind_reset_pages
        nullptr,   // bind_check_clear
    };
    return &ops;
}
extern "C" void g6_attn_missing(const char* op) {
    fprintf(stderr, "g6: attention op %s is not in the DeepSeek-V4.1-Flash-1017 engine\n", op);
    abort();
}
