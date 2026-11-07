// DS41 prefill row budget and the shared decode/prefill fabric's payload. The rules are engine/scheduler-rs
// src/plan_sizes.rs.
#pragma once
#include <cstddef>
#include "../scheduler-rs/g6plan.h"

// GR8_DS_PREFILL_W: zero means a bad explicit value; unset leaves the decode allocation unchanged.
inline int ds41_prefill_rowcap(const char* value, int decode_rows) { return g6plan_ds41_prefill_rowcap(value, decode_rows); }

// The maximum payload, not the physical stride: Fabric::init appends the 8-byte sequence tail and rounds to 256 B.
inline size_t ds41_prefill_fabric_payload(int rows, int hidden, int engram_dim) { return g6plan_ds41_prefill_payload(rows, hidden, engram_dim); }
