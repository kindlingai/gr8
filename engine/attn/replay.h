#pragma once
#include "attn.h"

// Allocate with cudaMalloc, not pinned host or managed memory. A glue kernel
// updates the first four words before every attention graph invocation.
// Positions, block-table entries and projection inputs live at fixed device
// addresses in G6Attn and are updated in place, not baked-in scalar values.
#ifdef DEC_S2
// Batched step (S2, one call for the rows of several sessions): per step row, the bounds and KV run of the session the
// row belongs to. Device data, written before the launch (k_s2_rows), read by every KV access of the row.
struct G6AttnRow {
    const int* table;                    // the session's block table (logical page -> physical page, G6_CP_REMOTE, -1)
    int context, page_count, table_size; // the session's live bounds, as the step cell's fields
    int ibase;                           // CP: first page of the session's index-cache run (logical-page addressed)
    int sess;                            // session id: positions must be distinct only within one session
};
#endif
struct G6AttnStepCell {
    int context, page_count, table_size, reuse_rows;
    unsigned valid; // written by captured validation kernel; not caller input
#ifdef DEC_S2
    // Batched step: [G6Attn::rows] per-row bounds / tables (device). nullptr = every row uses the cell's own fields (one
    // sequence). Set once when the cell is made and never changed. With rows, context / page_count / table_size are the
    // largest of the rows' (they bound the captured allocation checks), reuse_rows = the number of rows (shared layers) or 0.
    const G6AttnRow* rows;
#endif
};
struct G6AttnReplay {
    G6Attn fixed;   // context/page_count/table_size are allocation limits
    G6AttnStepCell* cell;
    void* scratch;
    size_t scratch_bytes;
};

// Capture/replay entry: topology is fixed by rows, reuse mode and capacity.
// All per-step validation/clearing uses a captured device kernel. No allocation,
// host reads of live control words, graph update or recapture is required.
extern "C" int g6_attn_replay_run(const G6AttnReplay*, void* stream);
