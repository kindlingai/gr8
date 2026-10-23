// Decode profile marks for the Flash-only kernels (kda.cuh, flash_mlp.cuh). rank.cuh's dmark is not visible in
// those headers (they are included before it is declared), so they call this hook instead; launch_step_flash sets
// it to flash_mark (flash_step.cuh), which maps these ids onto the DecStage enum. Null hook = no-op, so tests and
// the GLM path are untouched.
#pragma once
inline void (*g6_flash_mark)(int mark, void* stream) = nullptr;
// Mark ids; the mapping in flash_step.cuh owns the DecStage values (keep the two in sync).
enum FlashMark { FM_K_PROJ, FM_K_PREP, FM_K_STEP, FM_K_OUT, FM_K_OPROJ,
                 FM_MLP_ROUTER, FM_MLP_PLAN, FM_MLP_XHALF, FM_MLP_XFRAG, FM_K1_FC1, FM_K1_FC2, FM_K1_FIN,
                 FM_D_GATE, FM_D_DOWN, FM_D_OUT };
#define FMARK(k, st) do { if (g6_flash_mark) g6_flash_mark((k), (void*)(st)); } while (0)
