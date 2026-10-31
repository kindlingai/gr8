// GR8_L2PF for the DS41 step, CPU only (no GPU, no launch): the targets ds41l2pf::add_attn (q_a + kv, q_b) and ds41l2pf::add_moe give a layer (the gate as
// T_ROUTER, the shared expert's sh_w1 / sh_w3 / sh_w2 as T_SHARED) and the queue l2pf::queue makes of them: ranges, 16 B
// alignment, chunk counts, read order, target masks, a dense layer (no targets), the DSpark draft's 128-expert gate.
// build + run: nvcc -O2 -std=c++17 -I engine/kernels -o ds41_l2pf_queue_test engine/kernels/tests/ds41_l2pf_queue_test.cu
//   (the flags of l2pf_queue_test in scripts/build.sh: $GEN $K1H2_FLAGS)
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include "../ds41_l2pf.h"

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static const uint8_t* at(size_t off) { return reinterpret_cast<const uint8_t*>(uintptr_t(0x100000000ull) + off); }
static uint32_t chunks(size_t bytes) { return uint32_t((bytes + l2pf::CHUNK - 1) / l2pf::CHUNK); }

int main()
{
    constexpr int HID = 5120, IW = 576;                       // the verified tiling (ds41_moe_ops.cuh static_assert)
    const unsigned all = (1u << l2pf::NTG) - 1;
    const size_t gate384 = size_t(384) * HID * 2, fc1 = size_t(IW / 16) * (HID / 32) * 528, fc2 = size_t(HID / 16) * (IW / 32) * 528;
    EXPECT(gate384 == 3932160 && fc1 == 3041280 && fc2 == 3041280);   // 3.93 MB gate, 3 x 3.04 MB shared

    // 1. a 384-expert layer: four ranges, in the order the step reads them
    {
        l2pf::Plan P;
        ds41l2pf::add_moe(P, 384, HID, IW, at(0), at(size_t(8) << 20), at(size_t(16) << 20), at(size_t(24) << 20));
        EXPECT(P.n[l2pf::T_ROUTER][0] == gate384 && P.n[l2pf::T_ROUTER][1] == 0);
        EXPECT(P.n[l2pf::T_SHARED][0] == fc1 && P.n[l2pf::T_SHARED][1] == fc1 && P.n[l2pf::T_SHARED][2] == fc2 && P.n[l2pf::T_SHARED][3] == 0);
        for (int t = 0; t < l2pf::NTG; ++t)
            if (t != l2pf::T_ROUTER && t != l2pf::T_SHARED) EXPECT(P.n[t][0] == 0);   // no other target
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 4);
        EXPECT(q.p[0] == at(0) && q.p[1] == at(size_t(8) << 20) && q.p[2] == at(size_t(16) << 20) && q.p[3] == at(size_t(24) << 20));
        EXPECT(q.n[0] == gate384 && q.n[1] == fc1 && q.n[2] == fc1 && q.n[3] == fc2);
        for (int i = 0; i < q.count; ++i) EXPECT((uintptr_t(q.p[i]) & 15) == 0 && (q.n[i] & 15) == 0);
        EXPECT(q.end[0] == chunks(gate384) && chunks(gate384) == 480);
        EXPECT(q.end[1] == 480 + chunks(fc1) && chunks(fc1) == 372);
        EXPECT(q.end[3] == 480 + 3 * 372 && q.end[3] == 1596);                      // 1596 chunks of 8 KiB = 13.07 MB a layer
        EXPECT(q.from[l2pf::T_ROUTER] == 0 && q.from[l2pf::T_SHARED] == 480);
        EXPECT(q.from[l2pf::T_XIN] == 0 && q.from[l2pf::T_HEAD] == 1596);           // a window from T_HEAD (W_H) has nothing to issue
        // the window helper starts at T_ROUTER: the whole queue
        EXPECT(q.from[l2pf::T_ROUTER] < q.end[q.count - 1]);
        // the default W_A budget (30 us at 175 GB/s, 10 MiB cap): the gate and the start of sh_w1 (the pace bounds it, not MAXR)
        const l2pf::Cfg d{};
        const double ns = l2pf::CHUNK / d.gbps, bu = d.us[l2pf::W_A] * 1000.0 / ns, bm = d.mb * 1048576.0 / l2pf::CHUNK;
        const uint32_t budget = uint32_t(bu < bm ? bu : bm);
        EXPECT(budget > 480 && budget < 1596);
        printf("default W_A budget: %u chunks = %.2f MB of %.2f MB (gate %.2f MB first)\n", budget, budget * l2pf::CHUNK / 1e6,
               q.end[3] * l2pf::CHUNK / 1e6, gate384 / 1e6);
    }
    // 2. target masks (GR8_L2PF_T): router only, shared only; the other bits change nothing here
    {
        l2pf::Plan P;
        ds41l2pf::add_moe(P, 384, HID, IW, at(0), at(size_t(8) << 20), at(size_t(16) << 20), at(size_t(24) << 20));
        const l2pf::Queue r = l2pf::queue(P, 1u << l2pf::T_ROUTER), s = l2pf::queue(P, 1u << l2pf::T_SHARED);
        EXPECT(r.count == 1 && r.n[0] == gate384 && r.end[0] == 480);
        EXPECT(s.count == 3 && s.p[0] == at(size_t(8) << 20) && s.from[l2pf::T_SHARED] == 0 && s.end[2] == 3 * 372);
        const l2pf::Queue none = l2pf::queue(P, (1u << l2pf::T_XIN) | (1u << l2pf::T_UK));
        EXPECT(none.count == 0);
    }
    // 3. a base that is not 16 B aligned loses its edges (the queue trims to 16 B), nothing else
    {
        l2pf::Plan P;
        ds41l2pf::add_moe(P, 384, HID, IW, at(8), nullptr, nullptr, nullptr);
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 1 && q.p[0] == at(16) && q.n[0] == gate384 - 16 && (uintptr_t(q.p[0]) & 15) == 0);
    }
    // 4. a layer without MoE state adds nothing; null records are skipped
    {
        l2pf::Plan P;
        ds41l2pf::add_moe(P, 384, HID, IW, nullptr, nullptr, nullptr, nullptr);
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 0 && q.from[l2pf::T_ROUTER] >= (q.count ? q.end[q.count - 1] : 0));   // l2pf::window_us returns before any launch
    }
    // 5. the DSpark draft layers (128 experts): a 1.31 MB gate, the same shared expert
    {
        l2pf::Plan P;
        ds41l2pf::add_moe(P, 128, HID, IW, at(0), at(size_t(8) << 20), at(size_t(16) << 20), at(size_t(24) << 20));
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 4 && q.n[0] == size_t(128) * HID * 2 && q.end[0] == 160 && q.end[3] == 160 + 3 * 372);
    }
    // 6. attention targets (add_attn): q_a + kv as T_XIN, q_b as T_QB, then the MoE targets; the whole layer queue in read order
    {
        constexpr int HEADS = 16;                                         // heads of one rank at TP4 (wq_b N = heads * 512)
        const size_t qa = size_t(1280 / 16) * (HID / 32) * 528, kv = size_t(512 / 16) * (HID / 32) * 528, qb = size_t(HEADS * 512 / 16) * (1280 / 32) * 528;
        EXPECT(qa == 6758400 && kv == 2703360 && qb == 10813440);         // 6.76 + 2.70 + 10.81 MB (attention per layer)
        l2pf::Plan P;
        ds41l2pf::add_attn(P, at(0), 1280, HID, at(size_t(8) << 20), 512, HID, at(size_t(16) << 20), HEADS * 512, 1280);
        ds41l2pf::add_moe(P, 384, HID, IW, at(size_t(32) << 20), at(size_t(40) << 20), at(size_t(48) << 20), at(size_t(56) << 20));
        EXPECT(P.n[l2pf::T_XIN][0] == qa && P.n[l2pf::T_XIN][1] == kv && P.n[l2pf::T_XIN][2] == 0 && P.n[l2pf::T_QB][0] == qb && P.n[l2pf::T_QB][1] == 0);
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 7);
        EXPECT(q.p[0] == at(0) && q.p[1] == at(size_t(8) << 20) && q.p[2] == at(size_t(16) << 20) && q.p[3] == at(size_t(32) << 20));   // q_a, kv, q_b, gate
        EXPECT(q.n[0] == qa && q.n[1] == kv && q.n[2] == qb && q.n[3] == gate384 && q.n[4] == fc1 && q.n[5] == fc1 && q.n[6] == fc2);
        for (int i = 0; i < q.count; ++i) EXPECT((uintptr_t(q.p[i]) & 15) == 0 && (q.n[i] & 15) == 0);
        EXPECT(q.from[l2pf::T_XIN] == 0 && q.from[l2pf::T_QB] == q.end[1] && q.from[l2pf::T_ROUTER] == q.end[2] && q.from[l2pf::T_SHARED] == q.end[3]);
        EXPECT(q.end[2] == chunks(qa) + chunks(kv) + chunks(qb));
        EXPECT(q.end[6] == q.end[2] + 480 + 3 * 372);                      // the MoE part is the same 1596 chunks as test 1
        printf("layer queue: attention %.2f MB (q_a %.2f kv %.2f q_b %.2f) then gate + shared %.2f MB; W_M from chunk %u, W_A from chunk %u of %u\n",
               (q.end[2]) * l2pf::CHUNK / 1e6, qa / 1e6, kv / 1e6, qb / 1e6, 1596 * l2pf::CHUNK / 1e6, q.from[l2pf::T_XIN], q.from[l2pf::T_ROUTER], q.end[6]);
        // GR8_L2PF_T = 0x05 (T_XIN | T_QB): attention only; 0xc0 (T_ROUTER | T_SHARED): gate + shared from chunk 0
        const l2pf::Queue a = l2pf::queue(P, (1u << l2pf::T_XIN) | (1u << l2pf::T_QB)), g = l2pf::queue(P, (1u << l2pf::T_ROUTER) | (1u << l2pf::T_SHARED));
        EXPECT(a.count == 3 && a.end[2] == q.end[2] && a.from[l2pf::T_ROUTER] == a.end[2]);   // a W_A window (from T_ROUTER) has nothing to issue
        EXPECT(g.count == 4 && g.from[l2pf::T_XIN] == 0 && g.from[l2pf::T_ROUTER] == 0 && g.end[3] == 1596 && g.p[0] == at(size_t(32) << 20));
    }
    // 7. a layer without attention records (gv null: no .mx.gv in the pack) adds nothing for them; a layer with no state at all adds nothing
    {
        l2pf::Plan P;
        ds41l2pf::add_attn(P, nullptr, 1280, HID, at(0), 512, HID, nullptr, 8192, 1280);
        const l2pf::Queue q = l2pf::queue(P, all);
        EXPECT(q.count == 1 && q.n[0] == size_t(512 / 16) * (HID / 32) * 528 && P.n[l2pf::T_QB][0] == 0);
        l2pf::Plan Z;
        ds41l2pf::add_attn(Z, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, 0);
        EXPECT(l2pf::queue(Z, all).count == 0);
    }
    printf("ds41_l2pf_queue_test: %s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
