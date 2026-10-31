// The order a DS41 decode step rings its collectives in (host-only, no CUDA). The fabric proxy (comms/fabric.cpp) serves a step's
// collectives in the order of the Job's order vector (empty = c ascending, the identity). With Engram on, the two Engram reduces are
// c = 2*nl+1+e, past the head slot, but the step runs each of them in the middle of the layers, so the identity order stalls the
// proxy on a doorbell that rings late (error word 0x610051pp after the 10 s device spin).
// Where the step puts them (ds41_ops.h plan(), engram = true): ST_ENGRAM(i) opens layer i, i.e. it sits in U_CONS_F of layer i-1
// after ST_CONSUME_FFN (c = 2i-1) and before the publish of c = 2i (U_ATTN(i)). For the layer ids 0..39: Engram e=0 (layer 1) runs
// between c=1 and c=2, e=1 (layer 14) between c=27 and c=28 -> [0, 1, 81, 2 .. 27, 82, 28 .. 79, 80].
#pragma once
#include <vector>
#include "../model_geom.h"

namespace ds41
{
// ids: the served layer ids (Model::layers order). ecoll: Engram collectives (0 off, 2 on; as ds41_engram_colls). head: false for a
// step that skips the head (a non-final prompt chunk), whose order then leaves out c = 2*nl. Empty when ecoll == 0: no order is
// passed and the proxy keeps its identity loop.
inline std::vector<int> coll_exec_order(const std::vector<int>& ids, int ecoll, bool head = true)
{
    std::vector<int> o;
    if (ecoll <= 0) return o;
    const int nl = (int)ids.size();
    o.reserve((size_t)2 * nl + 1 + ecoll);
    for (int i = 0; i < nl; ++i)
    {
        if (geom::is_engram(ids[i])) o.push_back(2 * nl + 1 + (ids[i] == geom::ENGRAM_L0 ? 0 : 1));   // ds41_op_engram's e
        o.push_back(2 * i);
        o.push_back(2 * i + 1);
    }
    if (head) o.push_back(2 * nl);
    return o;
}
}
