// The S2 runtime signature: the string every rank sends to the bootstrap agreement (s2/bootstrap.h), so ranks whose
// options, knobs or loaded files would plan differently refuse to start instead of diverging. One format per model.
// SigInputs holds what each format reads, including values main has only after the load (the checkpoint store's
// blocks, the vision tower's id, the QO table size) and the strings other builders make (knob::signature(),
// s2::ngram_sig, s2::kstopb_sig, s2::DSelTable::str). The formats are engine/scheduler-rs src/signature.rs; this is the
// facade over it. Host only.
#pragma once
#include <cstdint>
#include <string>
#include "../ds41_vision_id.h"
#include "../../scheduler-rs/g6sched.h"

namespace g6s2
{

enum class SigModel { Glm, Flash, Ds41 };   // signature.rs SIG_GLM, SIG_FLASH, SIG_DS41

struct SigInputs
{
    SigModel model = SigModel::Glm;
    int cap = 0;
    std::string layers_s;               // --layers as given
    int chunk = 0, nc = 0;              // Plan::chunk, --top-p-cands
    std::string set_id;
    bool dfl = false;                   // --dflash-manifest
    int dfl_k = 0, s2sess = 1;
    std::string knobs;                  // knob::signature(): the agree-scope knobs off their defaults
    int MP = 0, mtp = 0;                // GLM and Flash
    int dfl_reuse = 0, dfl_reuse_min = 0;
    int lanes = 0, gsplit = 0;          // Flash: prefill lanes, GR8_DEC_SPLIT
    std::string ckpt;                   // Flash: g6s2::flash_cache_name, empty when GR8_GLM53F_CACHE is off
    uint64_t ckpt_blocks = 0;           // the checkpoint store's blocks
    bool ckpt_persist = false;
    int s2_below = 0;                   // GLM
    bool two = false;                   // two prefill lanes
    int pf_own = 0;
    bool pf_qp = false;                 // GR8_PF_OWN=4
    int qp_ctx = 0, qp_min = 0;         // the QO table size (Prefill::qp_ctx), pf_qp_min(): read with pf_qp only
    bool grammar = false;               // g6rt::grammar_ready()
    std::string ngram;                  // s2::ngram_sig
    int kstop_mode = 0, kstop_kmax = 0;
    double kstop_tau[4] = {};
    double kstop_seed3 = 0;
    int draft_par = 0, qs_mode = 0;
    bool iq_split = false, mtp_split = false;
    int dhead_on = 0, dhead_tpc = 0, dhead_ks = 0;   // DraftHeadI4
    bool mtp_last = false;
    int dhead2 = 0;
    bool dfl_i4 = false, dfl_body_i4 = false;
    float dfl_alpha = 1.f;
    int dsel_model = 0;
    std::string dsel_tab;               // s2::DSelTable::str
    std::string kstopb;                 // s2::kstopb_sig
    int cp_step = 0;
    bool cp_seg = false;
    int ctx_sp = 0, posidx = 0;         // g6_attn_ctx_sp(), g6_attn_posidx()
    bool steer_shared = false;          // a loaded steering profile of shared scope: the S2 layers carry its MoE finalize
    int ds_mtp = 0;                     // DS41: GR8_DS_MTP
    bool sh_tc = false;                 // GR8_DS_SH_TC (W8A8 shared expert, not bit-identical to off)
    int l2hint = 0, fc2stg = 0, fc1bstg = 0;
    bool c8_pack = false, sess_drafts = false;
    int pf_lanes = 1;
    int tk_main = 0, tk_decode = 0;     // GR8_DS_MOE_TOPK and its decode cut
    int idx_pipe_b = 0, idx_q_trig = 0, idx_pipe_qr = 0;
    bool reuse = false;
    bool vis_on = false;                // the vision tower loaded (the capability, not the knob)
    std::string vis_id;                 // ds41v::file_id of the tower file
};

inline std::string runtime_signature(const SigInputs& in)
{
    g6sig_inputs c{};
    c.model = uint32_t(in.model);
    c.cap = in.cap; c.layers_s = in.layers_s.c_str(); c.chunk = in.chunk; c.nc = in.nc; c.set_id = in.set_id.c_str();
    c.dfl = in.dfl; c.dfl_k = in.dfl_k; c.s2sess = in.s2sess; c.knobs = in.knobs.c_str();
    c.mp = in.MP; c.mtp = in.mtp; c.dfl_reuse = in.dfl_reuse; c.dfl_reuse_min = in.dfl_reuse_min;
    c.lanes = in.lanes; c.gsplit = in.gsplit; c.ckpt = in.ckpt.c_str(); c.ckpt_blocks = in.ckpt_blocks; c.ckpt_persist = in.ckpt_persist;
    c.s2_below = in.s2_below; c.two = in.two; c.pf_own = in.pf_own; c.pf_qp = in.pf_qp; c.qp_ctx = in.qp_ctx; c.qp_min = in.qp_min;
    c.grammar = in.grammar; c.ngram = in.ngram.c_str(); c.kstop_mode = in.kstop_mode; c.kstop_kmax = in.kstop_kmax;
    for (int j = 0; j < 4; ++j) c.kstop_tau[j] = in.kstop_tau[j];
    c.kstop_seed3 = in.kstop_seed3; c.draft_par = in.draft_par; c.qs_mode = in.qs_mode; c.iq_split = in.iq_split;
    c.mtp_split = in.mtp_split; c.dhead_on = in.dhead_on; c.dhead_tpc = in.dhead_tpc; c.dhead_ks = in.dhead_ks;
    c.mtp_last = in.mtp_last; c.dhead2 = in.dhead2; c.dfl_i4 = in.dfl_i4; c.dfl_body_i4 = in.dfl_body_i4;
    c.dfl_alpha = in.dfl_alpha; c.dsel_model = in.dsel_model; c.dsel_tab = in.dsel_tab.c_str(); c.kstopb = in.kstopb.c_str();
    c.cp_step = in.cp_step; c.cp_seg = in.cp_seg; c.ctx_sp = in.ctx_sp; c.posidx = in.posidx; c.steer_shared = in.steer_shared;
    c.ds_mtp = in.ds_mtp; c.sh_tc = in.sh_tc; c.l2hint = in.l2hint; c.fc2stg = in.fc2stg; c.fc1bstg = in.fc1bstg;
    c.c8_pack = in.c8_pack; c.sess_drafts = in.sess_drafts; c.pf_lanes = in.pf_lanes; c.tk_main = in.tk_main;
    c.tk_decode = in.tk_decode; c.idx_pipe_b = in.idx_pipe_b; c.idx_q_trig = in.idx_q_trig; c.idx_pipe_qr = in.idx_pipe_qr;
    c.reuse = in.reuse; c.vis_on = in.vis_on; c.vis_id = in.vis_id.c_str();
    return g6sig_runtime(&c);
}

}   // namespace g6s2
