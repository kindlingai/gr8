// DS41 two-lane W prefill (GR8_DS_PF_LANES=2, ds41_lanes.h): lane B's Model view. Included by rank.cuh after ds41_step.cuh, inside namespace
// g6rt. Built once at boot and only with the flag (g6m4.cu); nothing here runs on the default path, and it adds no kernel.
//
// mB is a copy of the main Model m. It SHARES what ds41::lanes::table() calls SERIAL or CAUSAL (the weights, the window KV / compressed caches
// and rings, the attention scratch / split / prefill ring, the indexer stage buffers q_b / wproj / y512 / pre / klen, the Engram engine and its
// token rings, the DSpark main scratch and key rings), the per-layer error words (ORed) and the plan. It OWNS every LANE buffer: the step io
// (tok / pos / tgt / tok_out / sess / cells / stepp), the mHC trunk and hc_y, sc.part, its fabric (f, tickets, relay word, RS tickets, gather
// sum), the MoE scratch (scratch_of(&mB)), the indexer top-512 list and scratch (candidate blocks), the Engram window rows and per-step set
// (Engram::add_lane / use_lane), the DSpark tap rows, and the host per-chunk fields. Layer::ds41_attn / ds41_moe are pointers to per-layer
// state, so a plain copy would share them: each layer gets its own copy of both (the copies keep the shared pointers inside them).
// Report dir ds41-two-lane-20261029: AUDIT-shared-state.md is the source list for this file.
#pragma once
#if G6_MODEL_DS41

// Returns the device bytes it took (cudaMemGetInfo before / after; the fabric's pinned host slots are not in it).
inline size_t build_ds41_lane_b(Model& mB, Model& m, const g6::FabricDev& fB, cudaStream_t st)
{
    if (!m.ds41 || m.ds41_draft || m.hc_rows <= MX || m.layers.empty()) die("DS41 lane B: needs the main DS41 Model with W-row prefill (W > MX)");
    if (fB.nr != g6nr::NR || fB.ncoll != m.f.ncoll) die("DS41 lane B: bad lane-B fabric or missing B12 state collective");
    size_t free0 = 0, total = 0;
    G6RT_CK(cudaDeviceSynchronize());
    G6RT_CK(cudaMemGetInfo(&free0, &total));
    mB = m;
    const size_t HR = (size_t)m.hc_rows;
    // fabric and step io
    mB.f = fB;
    mB.stepp = dmalloc<u64>(1);
    mB.tickets = dmalloc<unsigned>((size_t)m.ncoll());
    mB.ds41_relay = m.ds41_relay ? dmalloc<u64>(1) : nullptr;
    mB.ds41_gsum = m.ds41_gsum ? dmalloc<__nv_bfloat16>(HR * HID) : nullptr;
    mB.ds41_rs_tickets = m.ds41_rs_k ? dmalloc<unsigned>((size_t)m.ds41_rs_k) : nullptr;
    // B12's owner-quantized y and bounded-replay state publish are live across
    // the two lanes' independent streams/fabrics; never alias lane A's storage.
    mB.ds41_y_codes = m.ds41_y_mxfp8 ? dmalloc<uint8_t>(HR * HID) : nullptr;
    mB.ds41_y_scales = m.ds41_y_mxfp8 ? dmalloc<uint8_t>(HR * HID / 32) : nullptr;
    mB.ds41_state_ticket = m.ds41_rowshard ? dmalloc<unsigned>(1) : nullptr;
    mB.tok = dmalloc<int>(HR); mB.pos = dmalloc<int>(HR); mB.tok_out = dmalloc<int>(HR); mB.tgt = dmalloc<int>(HR);
    mB.sess = dmalloc<int>(HR);
    mB.cells = dmalloc<G6AttnStepCell>((size_t)m.nl);
    // mHC trunk (build_ds41's sizes)
    mB.resid4 = dmalloc<ds_mhc::trunk_t>(HR * geom::HC * HID);
    mB.hc_mix = dmalloc<float>(HR * geom::HC_MIX);
    mB.hc_pre = dmalloc<float>(HR * geom::HC); mB.hc_post = dmalloc<float>(HR * geom::HC);
    mB.hc_comb = dmalloc<float>(HR * geom::HC * geom::HC);
    mB.hc_done = dmalloc<unsigned>((size_t)ds_mhc::done_words(m.hc_rows));
    mB.hc_part = dmalloc<float>(HR * (geom::HC_MIX + 2) * (HID / ds_mhc::PW));
    mB.hc_y = dmalloc<__nv_bfloat16>(HR * HID);
    mB.resid = dmalloc<float>((size_t)MX * HID);
    mB.x = dmalloc<__half>((size_t)MX * HID); mB.x2 = dmalloc<__half>((size_t)MX * HID);
    mB.scratch = dmalloc<char>(bind::scratch_bytes()); mB.sc = bind::scratch_at(mB.scratch);
    mB.sc.part[0] = dmalloc<uint16_t>(HR * HID);
    mB.sc.part[1] = dmalloc<uint16_t>(HR * HID);
    // Engram: the window rows; the token rings (ehist) and the engine are shared, the engine gets a second per-step set
    if (m.engram)
    {
        mB.win = dmalloc<int32_t>(HR * 4);
        std::string err;
        if (!m.engram->add_lane(&err)) die("DS41 lane B: " + err);
    }
    // DSpark: the tap rows (main layers 37..39 -> ST_DS_MAIN); MainS and the key rings stay shared (ST_DS_MAIN is a chain section)
    if (m.dsp.tap) mB.dsp.tap = dmalloc<__nv_bfloat16>((size_t)ds41dsp_ops::tap_cap(m.dsp) * 3 * 5120);
    mB.dsp.main_lo = 0; mB.dsp.main_hi = -1;
    // host per-chunk state
    mB.ds41_pf_late = 0; mB.ds_pf_slot = 0; mB.ds_ns = 1; mB.ds_moe_decode_step = false;
    mB.ds41_dchunk = false; mB.ds41_dseq = 0; mB.ds41_dir_used[0] = mB.ds41_dir_used[1] = false;
    mB.ds41_rowshard_chunk = false; mB.ds41_rowshard_state = false;
    mB.ds41_y_quant_ready = false; mB.ds41_rowshard_pending_c = -1;
    mB.ds41_pack = nullptr; mB.ds41_b12_kernel_launches = 0;
    // per-layer state: own copies (attention: S.sess -> mB.sess, topk below; MoE: the scratch of &mB)
    for (int i = 0; i < m.nl; ++i)
    {
        Layer& y = mB.layers[i];
        if (const Ds41AttnLayer* a = ds41_attn_state(m, i))
        {
            Ds41AttnLayer* b = new Ds41AttnLayer(*a);
            if (a->S.sess) b->S.sess = mB.sess;
            y.ds41_attn = b;
        }
        if (const auto* s = (const ds41moe::LayerState*)m.layers[i].ds41_moe)
        {
            auto* t = new ds41moe::LayerState(*s);
            t->sc = ds41moe::scratch_of(&mB);
            y.ds41_moe = t;
        }
    }
    // indexer: a registry entry for &mB, a copy of m's with its own top-512 list and scratch (the layer-20 candidate blocks); lane B runs one
    // session per chunk (ds_ns = 1), so the per-session decode scratch sscr is never its own: null, a use would fault instead of sharing m's
    if (Ds41IdxModel* SA = ds41_idx_detail::find(m, false); SA && !SA->lay.empty())
    {
        Ds41IdxModel* SB = ds41_idx_detail::find(mB, true);
        if (!SB || !SB->lay.empty()) die("DS41 lane B: indexer registry full or lane B registered twice");
        *SB = *SA;
        const size_t tk_bytes = (size_t)SA->rcap * geom::IDX_TOPK * sizeof(int32_t);
        SB->topk = (int32_t*)ds41_idx_detail::zalloc(tk_bytes);
        G6RT_CK(cudaMemset(SB->topk, 0xFF, tk_bytes));   // int32 -1 = "no key", as ds41_idx_bind
        SB->scratch = ds41_idx_detail::zalloc(ds41_idx_scratch_bytes(SA->rcap, SA->lim1));
        for (void*& p : SB->sscr) p = nullptr;
        SB->chunk_end = 0;
        SB->cand_ready = false;
        for (int i = 0; i < m.nl; ++i)
        {
            const Ds41AttnLayer* a = ds41_attn_state(m, i);
            Ds41AttnLayer* b = ds41_attn_state(mB, i);
            if (a && b && a->topk == SA->topk) b->topk = SB->topk;
        }
    }
    G6RT_CK(cudaMemsetAsync(mB.hc_done, 0, (size_t)ds_mhc::done_words(m.hc_rows) * sizeof(unsigned), st));
    G6RT_CK(cudaStreamSynchronize(st));
    G6RT_CK(cudaDeviceSynchronize());
    size_t free1 = 0;
    G6RT_CK(cudaMemGetInfo(&free1, &total));
    return free0 > free1 ? free0 - free1 : 0;
}

#endif
