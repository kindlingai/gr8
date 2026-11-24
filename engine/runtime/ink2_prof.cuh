// Lightweight zero-overhead-when-off CUDA-event stage profiler for Inkling-2-Mini (GR8_INK2_PROFILE=1).
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ink2
{
enum ProfStage
{
    PS_H2D_META = 0,
    PS_EMBED,
    PS_LOOP_COPY_TAPS,
    PS_ATTN_RES,
    PS_LOOP_MSG_GATE,
    PS_ATTN_QKV_PROJ,
    PS_ATTN_SCONV_KVPREP,
    PS_ATTN_CORE_LOCAL,
    PS_ATTN_CORE_GLOBAL,
    PS_ATTN_WO,
    PS_POST_ATTN_SCONV_RES,
    PS_ROUTER,
    PS_MOE_FC1,
    PS_MOE_FC2,
    PS_MOE_COMBINE,
    PS_POST_MLP_SCONV_RES,
    PS_FINAL_RES,
    PS_TRUNK_LM_HEAD,
    PS_TRUNK_HEAD_PACK,
    PS_COMMIT_TRUNK,
    PS_COMMIT_MTP,
    PS_MTP_BODY,
    PS_MTP_BASE_LM,
    PS_MTP_RNN_FOLD,
    PS_MTP_HEAD_PACK,
    PS_COUNT
};

inline const char* prof_stage_name(int s)
{
    static const char* kNames[PS_COUNT] = {
        "h2d_meta",
        "embed",
        "loop_copy_taps",
        "attn_res",
        "loop_msg_gate",
        "attn_qkv_proj",
        "attn_sconv_kvprep",
        "attn_core_local",
        "attn_core_global",
        "attn_wo",
        "post_attn_sconv_res",
        "router",
        "moe_fc1",
        "moe_fc2",
        "moe_combine",
        "post_mlp_sconv_res",
        "final_res",
        "trunk_lm_head",
        "trunk_head_pack",
        "commit_trunk",
        "commit_mtp",
        "mtp_body",
        "mtp_base_lm",
        "mtp_rnn_fold",
        "mtp_head_pack",
    };
    return (s >= 0 && s < PS_COUNT) ? kNames[s] : "unknown";
}

struct ProfBucket
{
    long calls = 0;
    long total_rows = 0;
    double wall_ms = 0.0;
    double gpu_ms[PS_COUNT] = {};
    long stage_cnt[PS_COUNT] = {};
};

struct Prof
{
    bool enabled = false;
    int print_every = 20;
    long total_flushes = 0;
    // Buckets: 0 = prefill (!samples), 1 = decode (samples && M==1), 2 = verify (samples && M>1), 3 = commit, 4 = mtp_draft
    ProfBucket b[5];

    struct EvRec
    {
        int stage;
        cudaEvent_t t0, t1;
    };
    std::vector<cudaEvent_t> pool;
    std::vector<EvRec> cur;
    size_t pool_used = 0;
    int cur_bucket = -1;
    int cur_rows = 0;
    std::chrono::steady_clock::time_point wall_t0;

    void init(bool enable = false, int every = 20)
    {
        enabled = enable;
        if (every > 0) print_every = every;
        if (enabled)
        {
            // Pre-allocate enough events for a 48-layer forward (~1600 events)
            ensure_pool(2048);
        }
    }

    ~Prof()
    {
        if (enabled) dump("final");
        for (cudaEvent_t ev : pool) cudaEventDestroy(ev);
        pool.clear();
    }

    void ensure_pool(size_t need)
    {
        while (pool.size() < need)
        {
            cudaEvent_t ev = nullptr;
            if (cudaEventCreate(&ev) != cudaSuccess) break;
            pool.push_back(ev);
        }
    }

    cudaEvent_t alloc_ev()
    {
        if (pool_used >= pool.size()) ensure_pool(pool.size() + 512);
        return pool[pool_used++];
    }

    void begin_op(int bucket, int rows)
    {
        if (!enabled) return;
        cur.clear();
        pool_used = 0;
        cur_bucket = bucket;
        cur_rows = rows;
        wall_t0 = std::chrono::steady_clock::now();
    }

    int begin_stage(int stage, cudaStream_t st = 0)
    {
        if (!enabled || cur_bucket < 0) return -1;
        cudaEvent_t t0 = alloc_ev();
        cudaEvent_t t1 = alloc_ev();
        cudaEventRecord(t0, st);
        cur.push_back(EvRec{stage, t0, t1});
        return static_cast<int>(cur.size()) - 1;
    }

    void end_stage(int handle, cudaStream_t st = 0)
    {
        if (!enabled || handle < 0) return;
        cudaEventRecord(cur[static_cast<size_t>(handle)].t1, st);
    }

    void end_op(cudaStream_t st = 0)
    {
        if (!enabled || cur_bucket < 0) return;
        if (!cur.empty())
        {
            cudaEventSynchronize(cur.back().t1);
        }
        else
        {
            cudaStreamSynchronize(st);
        }
        const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_t0).count();
        ProfBucket& bk = b[cur_bucket];
        bk.calls += 1;
        bk.total_rows += cur_rows;
        bk.wall_ms += wms;
        for (const EvRec& r : cur)
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, r.t0, r.t1);
            bk.gpu_ms[r.stage] += static_cast<double>(ms);
            bk.stage_cnt[r.stage] += 1;
        }
        cur.clear();
        cur_bucket = -1;
        if (++total_flushes % print_every == 0) dump("periodic");
    }

    void dump(const char* tag) const
    {
        static const char* kBNames[5] = {"prefill", "decode_m1", "verify_mgt1", "commit", "mtp_draft"};
        fprintf(stderr, "\n=== GR8_INK2_PROFILE (%s) ===\n", tag);
        for (int bi = 0; bi < 5; ++bi)
        {
            const ProfBucket& bk = b[bi];
            if (bk.calls == 0) continue;
            double sum_gpu = 0.0;
            for (int s = 0; s < PS_COUNT; ++s) sum_gpu += bk.gpu_ms[s];
            const double avg_wall = bk.wall_ms / bk.calls;
            const double avg_gpu = sum_gpu / bk.calls;
            const double avg_rows = static_cast<double>(bk.total_rows) / bk.calls;
            fprintf(stderr, "[%s] calls=%ld avg_M=%.2f avg_wall=%.3f ms avg_gpu=%.3f ms host/gap=%.3f ms (%.1f%%)\n",
                    kBNames[bi], bk.calls, avg_rows, avg_wall, avg_gpu, avg_wall - avg_gpu,
                    avg_wall > 0 ? 100.0 * (avg_wall - avg_gpu) / avg_wall : 0.0);
            for (int s = 0; s < PS_COUNT; ++s)
            {
                if (bk.stage_cnt[s] == 0) continue;
                const double ms = bk.gpu_ms[s] / bk.calls;
                fprintf(stderr, "  %-22s %8.3f ms/call (%5.1f%% wall, %5.1f%% gpu, %.0f spans/call)\n",
                        prof_stage_name(s), ms,
                        avg_wall > 0 ? 100.0 * ms / avg_wall : 0.0,
                        avg_gpu > 0 ? 100.0 * ms / avg_gpu : 0.0,
                        static_cast<double>(bk.stage_cnt[s]) / bk.calls);
            }
        }
        fprintf(stderr, "=== END GR8_INK2_PROFILE ===\n");
        fflush(stderr);
    }
};

struct ProfScope
{
    Prof* p;
    cudaStream_t st;
    int h;
    ProfScope(Prof* p_, int stage, cudaStream_t st_ = 0) : p(p_), st(st_), h(p_ ? p_->begin_stage(stage, st_) : -1) {}
    ~ProfScope() { if (p && h >= 0) p->end_stage(h, st); }
};
}   // namespace ink2
