// GR8_DS_ENGRAM's boot reads, host only, so g6m4's option parse (engine_options.h) sizes the decode fabric with
// ds41_engram_colls before the Model exists. ds41_engram_ops.cuh holds the rest of the Engram wiring.
#pragma once
#if G6_MODEL_DS41
#include <string>
#include <vector>
#include "rank_host.h"
#include "../engram/engram_boot.h"

namespace g6rt
{

// GR8_DS_ENGRAM and its knobs. Off (default) reads nothing else, so a stray knob cannot change an Engram-off run.
inline ds41::engram::Knobs ds41_engram_knobs()
{
    ds41::engram::Knobs k;
    std::string err;
    if (!knob::ds_engram()) return k;
    if (!ds41::engram::read_knobs(&k, &err)) die(err);
    if (k.prefix.empty()) die("GR8_DS_ENGRAM=1 needs GR8_DS_ENGRAM_PREFIX (PREFIX.engram.json, written at boot, and PREFIX.engram.cmap)");
    return k;
}
// Extra collectives of the decode fabric (g6m4.cu, before the Model exists): 0 off, else kNE = 2 after the head slot. Both Engram layers must be served.
inline int ds41_engram_colls(const std::vector<int>& ids)
{
    if (!ds41_engram_knobs().on) return 0;
    for (int l : ids) if (l == geom::ENGRAM_L0)
        for (int l2 : ids) if (l2 == geom::ENGRAM_L1) return ds41::engram::kNE;
    die("DS41 build: GR8_DS_ENGRAM=1 needs layers " + std::to_string(geom::ENGRAM_L0) + " and " + std::to_string(geom::ENGRAM_L1) + " in --layers");
    return 0;
}

}   // namespace g6rt
#endif
