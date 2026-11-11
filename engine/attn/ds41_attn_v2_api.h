// Kernel-free v2 attention entry points for the conductor's rank TU.
#pragma once
#include "ds41_attn_v2_types.h"
extern "C" {
void ds41_v2_input_project(const ds41a::Weights*, ds41a::v2::InputBuf*, const void*, int, int, void*);
void ds41_v2_input_pack(const ds41a::v2::InputBuf*, uint8_t*, int, int, int, void*);
void ds41_v2_input_unpack(const uint8_t*, const ds41a::v2::InputBuf*, int, int, void*);
void ds41_v2_project_rest(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const ds41a::v2::InputBuf*, int, int, void*, const int*);
void ds41_v2_comp_rest(const ds41a::Weights*, ds41a::Cache*, const ds41a::Scratch*, const ds41a::v2::InputBuf*, int, void*, const int*);
void ds41_v2_output_a(const ds41a::Weights*, const ds41a::Scratch*, int, int, void*, const int*);
void ds41_v2_output_pack(const ds41a::Scratch*, uint8_t*, int, int, int, void*);
void ds41_v2_output_b(const ds41a::Fp8Mat*, const uint8_t*, void*, int, uint32_t*, void*);
void ds41_v2_output_b_split(const ds41a::Fp8Mat*, const uint8_t*, const uint8_t*, void*, int, uint32_t*, void*);
void ds41_v2_output_sum(const void*, void*, int, int, int, void*);
}
