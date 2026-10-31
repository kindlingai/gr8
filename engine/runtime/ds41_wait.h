// GR8_DS_WAIT_K: who waits for the peers' payload before a DS41 consume kernel. Host-only, no CUDA; ds41_step.cuh's ds41_consume_wait calls it,
// engine/tests/test_ds41_wait.cpp checks it. Must stay outside namespace g6rt (like ds41_ops.h).
//   off (default): the consume kernel waits itself (Ds41Block::wait, wait_payload = bytes of the slot payload; the tail sits right behind them)
//   on:            one CTA waits first (glue::k_wait_peers(c, payload)), the consume kernel gets wait_payload = 0 and skips its own wait
//   relay (K=2):   no wait kernel; the consume kernel starts at once, CTA (0,0) waits for the peers (glue::wait_peers) and releases a device word, every
//                  other CTA waits for that word instead of the three host-memory tails (Ds41RelayBlock, ds41_step.cuh). wait_payload = the slot payload
#pragma once
#include <cstddef>

namespace ds41 {

struct ConsumeWait
{
    bool kernel;              // launch glue::k_wait_peers(c, kernel_payload) ahead of the consume kernel
    size_t kernel_payload;    // its `payload` argument (0 when no wait kernel is launched)
    size_t block_payload;     // Ds41Block::wait_payload (0 = the consume kernel does not wait)
    bool relay;               // the consume kernel is the relay block (K=2): block_payload is the bytes CTA (0,0) waits for
};

inline ConsumeWait consume_wait(bool waitk, int rows, int hid)
{
    const size_t payload = (size_t)rows * hid * 2;   // rows x HID BF16, what put_publish writes (rows * (HID * 2 / 16) uint4)
    return waitk ? ConsumeWait{true, payload, 0, false} : ConsumeWait{false, 0, payload, false};
}

// GR8_DS_WAIT_K value -> the mode that runs: 0 off, 1 wait kernel, 2 relay. 2 needs the main Model (the draft Model has its own small fabric and
// keeps K=1) and no GR8_DRAFT_PAR (a second stream's kernels may take the SMs: CTAs that wait on CTA (0,0) need it resident); otherwise it falls back to 1.
// Any other non-zero value is 1 (atoi(...) != 0).
inline int wait_mode(int env_k, bool draft_model, bool draft_par)
{
    if (env_k == 0) return 0;
    if (env_k == 2 && !draft_model && !draft_par) return 2;
    return 1;
}
inline ConsumeWait consume_wait_mode(int mode, int rows, int hid)
{
    ConsumeWait w = consume_wait(mode == 1, rows, hid);
    if (mode == 2) w.block_payload = (size_t)rows * hid * 2, w.relay = true;
    return w;
}

}   // namespace ds41
