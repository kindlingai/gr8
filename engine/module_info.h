// What the host side needs to know about the engine build it runs in: the model, the width, the rows of a step, the
// session slots, the geometry the host reads, the KV row format, and the model behaviours host code branches on
// (docs/HOST-CORE.md, R1). Host code reads these instead of testing G6_MODEL_*; every value comes from the build's own
// constants, so this is one build's description, the same on every rank of an engine.
//
// The struct is C: a later host core outside the model's compile reads it across the module boundary. abi changes with
// any change to the layout.
#pragma once
#include <stdint.h>

#define G6_MODULE_ABI 3

enum G6ModelId { G6_MODEL_ID_GLM53 = 1, G6_MODEL_ID_GLM53F = 2, G6_MODEL_ID_DS41 = 3, G6_MODEL_ID_INK2 = 4 };

// model behaviours host code branches on (G6ModuleInfo::caps)
enum G6ModuleCap
{
    G6_CAP_MTP0 = 1u << 0,           // a session may serve with no draft head (MTP 0)
    G6_CAP_PACK = 1u << 1,           // several sessions' prompt rows in one prefill pack (Config::c8_pack)
    G6_CAP_PARK = 1u << 2,           // a finished chat may stay parked in its slot (Config::park)
    G6_CAP_PROMPT_DECODE = 1u << 3,  // prompts run as decode steps unless prefill_min admits them (Sessions::payload)
    G6_CAP_DSPARK = 1u << 4,         // DSpark drafts: their keys come from the main step's rows (Dispatch)
    G6_CAP_DFLASH2 = 1u << 5,        // the DFlash2 drafter
    G6_CAP_FLASH_COST = 1u << 6,     // the Flash solo cost table (S2Cost, GR8_GLM53F_S2_COST)
    G6_CAP_KDA = 1u << 7,            // linear-attention (KDA) state per session beside the paged KV
    G6_CAP_GRAMMAR = 1u << 8,        // the model's requests may carry grammars (the engine also needs XGrammar: G6_GRAMMAR)
    G6_CAP_KV_STORE = 1u << 9,       // shared page + boundary-state tiered stores
};

struct G6ModuleInfo
{
    uint32_t abi;                    // G6_MODULE_ABI
    uint32_t model;                  // G6ModelId
    char name[16];                   // the model's name in logs: "glm-5.3", "glm-5.3-flash", "ds41", "ink2"
    int32_t nr;                      // ranks (tensor-parallel width)
    int32_t mx;                      // rows of one step, at most
    int32_t s2n;                     // session slots of a step's S2In
    int32_t max_sessions;            // session slots the scheduler holds
    int32_t hidden, layers, mtp_layers, vocab, experts, topk;
    int32_t kv_row_bytes;            // one g6kv::ROW: one layer's latent row on one rank (a DS41 ca row covers 2 tokens on ratio-2 layers)
    int32_t kv_page;                 // tokens in a KV page
    char kv_format[16];              // g6kv::NAME
    uint32_t caps;                   // G6ModuleCap bits
    int32_t dfl_block;               // DFlash2 block rows, the bonus row and its drafts (dflash.cuh dfl::B); the planner sizes by it
    int32_t pack_members;            // sessions in one prefill pack at most (ds41::pack::MAX_SEGMENTS); the scheduler sizes by it
    // what the option parse (scheduler-rs options.rs) reads besides the above; g6m4.cu checks each against its device header
    int32_t dfl_drafts;              // drafts a DFlash2 round verifies at most (dflash.cuh dfl::K)
    int32_t dfl_rep;                 // 1: the drafter body runs replicated at this width (dfl::REP)
    int32_t tp_sort;                 // top-p candidates all ranks hold together, at most (glue.cuh glue::TP_SORT)
    int32_t kv4;                     // 1: the int4 KV build (G6_KV4)
    int32_t xgrammar;                // 1: built with XGrammar (G6_GRAMMAR)
    int32_t ds_markov_maxs;          // DS41: sessions of one DSpark markov launch (ds41dsp::MARKOV_MAXS); else 0
    int32_t ds_block;                // DS41: the DSpark draft block rows (Ds41RoundLogic::BLOCK); else 0
};

#ifdef __cplusplus
#include "g6mx.h"
#include "g6nr.h"
#include "model_geom.h"
#include "attn/kvfmt.h"
#include "runtime/s2_in.h"

namespace g6mod {

constexpr uint32_t this_model()
{
#if G6_MODEL_DS41
    return G6_MODEL_ID_DS41;
#elif G6_MODEL_INK2
    return G6_MODEL_ID_INK2;
#elif G6_MODEL_FLASH
    return G6_MODEL_ID_GLM53F;
#else
    return G6_MODEL_ID_GLM53;
#endif
}

constexpr G6ModuleInfo make()
{
    G6ModuleInfo m{};
    m.abi = G6_MODULE_ABI;
    m.model = this_model();
    const char* name = m.model == G6_MODEL_ID_INK2 ? "ink2" : m.model == G6_MODEL_ID_DS41 ? "ds41" : m.model == G6_MODEL_ID_GLM53F ? "glm-5.3-flash" : "glm-5.3";
    for (int i = 0; name[i] && i < 15; ++i) m.name[i] = name[i];
    m.nr = g6nr::NR;
    m.mx = g6mx::MX;
    m.s2n = m.max_sessions = g6rt::S2N;   // the scheduler holds as many sessions as a step's S2In lays out
    m.hidden = geom::HID;
    m.layers = geom::L_MAIN;
#if G6_MODEL_DS41 || G6_MODEL_INK2
    m.mtp_layers = geom::N_NEXTN;
#else
    m.mtp_layers = 1;
#endif
    m.vocab = geom::VOCAB;
    m.experts = geom::NEXP;
    m.topk = geom::TOPK;
    m.kv_row_bytes = g6kv::ROW;
    m.kv_page = 64;
    for (int i = 0; g6kv::NAME[i] && i < 15; ++i) m.kv_format[i] = g6kv::NAME[i];
    m.dfl_block = 8;      // dflash.cuh checks
    m.pack_members = 8;   // s2/ds41_prefill.h checks
    m.dfl_drafts = 7;
    m.dfl_rep = 8 % g6nr::NR != 0;   // NKV = 8
    m.tp_sort = 2048;
#if defined(G6_KV4) && G6_KV4
    m.kv4 = 1;
#endif
#if defined(G6_GRAMMAR) && G6_GRAMMAR
    m.xgrammar = 1;
#endif
    if (m.model == G6_MODEL_ID_DS41) { m.ds_markov_maxs = 8; m.ds_block = 5; }
    switch (m.model)
    {
    case G6_MODEL_ID_DS41: m.caps = G6_CAP_MTP0 | G6_CAP_PACK | G6_CAP_PARK | G6_CAP_PROMPT_DECODE | G6_CAP_DSPARK | G6_CAP_KV_STORE; break;
    // MTP 0..7: the drafter's context rows come from the main step's rows, as the DSpark keys do
    case G6_MODEL_ID_INK2: m.caps = G6_CAP_MTP0 | G6_CAP_DSPARK | G6_CAP_KV_STORE; break;
    case G6_MODEL_ID_GLM53F: m.caps = G6_CAP_DFLASH2 | G6_CAP_FLASH_COST | G6_CAP_KDA | G6_CAP_GRAMMAR; break;
    default: m.caps = G6_CAP_DFLASH2 | G6_CAP_GRAMMAR | G6_CAP_KV_STORE; break;
    }
    return m;
}

// this build's description
inline constexpr G6ModuleInfo INFO = make();
static_assert(INFO.max_sessions == INFO.s2n && (INFO.s2n == 8 || INFO.s2n == 32), "session slots");
constexpr bool has(uint32_t cap) { return (INFO.caps & cap) != 0; }

}   // namespace g6mod
#endif
