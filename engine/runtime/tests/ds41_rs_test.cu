// F10 GPU gate of the DS41 prefill RS+AG kernels (GR8_DS_PF_RS=1, runtime/ds41_rs.cuh). One GPU plays all NR ranks: each rank has its own
// pinned fabric region (the real slot layout, fab_slot_layout; send, recv, doorbell, error word) and its own device sum. The host plays the
// proxies exactly as fabric.cpp does: after doorbell c, peer p gets slice p (send + p ss, Q H 2 bytes) + the seq tail written right after
// it, into its recv slot (c, me) at offset 0; after doorbell c_ag, every peer gets the AG payload + tail. Then:
//   k_put_publish_rs (every rank) -> host RS -> k_wait_peers + k_rs_reduce (every rank) -> host AG -> k_wait_peers + k_rs_gather.
// Every rank's device sum must equal, byte for byte, rn16 of the FP32 sum p = 0..NR-1 from 0.f of the partials (the value the all-reduce
// consumers use; ds41_gather_test shows the consume kernels then give identical outputs from it). Also: the error word stays 0.
// Partials as ds41_gather_test: random BF16 incl. +-0, subnormals, large magnitudes, rows where the partials cancel. M % NR != 0 cases.
// Build: nvcc -O2 -std=c++17 -DG6_MODEL_DS41=1 -arch=sm_121a -I engine -I engine/runtime -o ds41_rs_test engine/runtime/tests/ds41_rs_test.cu
// Exit 0 = last line "ds41 rs: ALL IDENTICAL".
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "../../kernels/glue.cuh"
#include "../ds41_rs.h"
#include "../ds41_gather.cuh"
#include "../ds41_rs.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static const int H = HID, C = 0, CAG = 1, NCOLL = 2;   // collective 0 = the RS one, 1 = its AG collective
static unsigned short bf(float v) { unsigned u; memcpy(&u, &v, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float fl(unsigned short b) { unsigned u = (unsigned)b << 16; float v; memcpy(&v, &u, 4); return v; }

struct Rank { g6::FabricDev f{}; char* region = nullptr; uint4* staging = nullptr; __nv_bfloat16* gsum = nullptr; unsigned* tickets = nullptr; };

int main() {
    std::mt19937 rng(20261026);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    const int NR = g6nr::NR;
    int bad = 0;
    for (int M : {2048, 2047, 846, 129, 50, 49}) {
        const int q = ds41::rs::quarter(M, NR);
        const size_t rb = (size_t)H * 2, ss = ds41::rs::slice_stride(q, rb), pay = (size_t)q * rb;
        // the layout g6m4.cu builds: run 0 holds NR slices; run 3 (here: from c = 1) one slice payload
        g6::FabricConfig cfg; cfg.slot_bytes = (size_t)NR * ss - 8; cfg.run1 = 1; cfg.run2 = 1; cfg.slot_bytes1 = 8; cfg.run3 = 1; cfg.slot_bytes3 = pay;
        const u64 step = 7;   // parity 1
        u64* dstep; CK(cudaMalloc(&dstep, 8)); CK(cudaMemcpy(dstep, &step, 8, cudaMemcpyHostToDevice));
        std::vector<Rank> rk(NR);
        std::vector<std::vector<unsigned short>> part(NR, std::vector<unsigned short>((size_t)M * H));
        for (size_t k = 0; k < (size_t)M * H; ++k) {
            const int kind = rng() % 16;
            for (int p = 0; p < NR; ++p) {
                float v = u(rng) * (kind == 0 ? 3e4f : kind == 1 ? 1e-39f : 4.f);
                if (kind == 2) v = p == 0 ? 0.f : -0.f;
                part[p][k] = bf(v);
            }
            if (kind == 3) { part[1][k] = part[0][k] ^ 0x8000; part[3 % NR][k] = part[2 % NR][k] ^ 0x8000; }
        }
        for (int r = 0; r < NR; ++r) {
            Rank& R = rk[r];
            R.f.nr = NR; R.f.rank = r;
            if (!g6::fab_slot_layout(cfg, NCOLL, &R.f)) { printf("layout refused\n"); return 1; }
            const size_t half = g6::fab_slot_base(R.f, NCOLL), send_b = 2 * half, recv_b = 2 * half * NR;
            CK(cudaHostAlloc((void**)&R.region, send_b + recv_b + 64 + 8, cudaHostAllocMapped | cudaHostAllocPortable));
            memset(R.region, 0xEE, send_b + recv_b);   // stale bytes everywhere
            memset(R.region + send_b + recv_b, 0, 64 + 8);
            R.f.send = R.region; R.f.recv = R.region + send_b;
            R.f.doorbell = (unsigned long long*)(R.region + send_b + recv_b);
            R.f.error = (unsigned*)(R.region + send_b + recv_b + 64);
            CK(cudaMalloc(&R.staging, (size_t)M * rb)); CK(cudaMemcpy(R.staging, part[r].data(), (size_t)M * rb, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&R.gsum, (size_t)M * rb)); CK(cudaMemset(R.gsum, 0xEE, (size_t)M * rb));
            CK(cudaMalloc(&R.tickets, 8)); CK(cudaMemset(R.tickets, 0, 8));
        }
        const unsigned long long sc = g6::coll_seq(step, C, NCOLL), sa = g6::coll_seq(step, CAG, NCOLL);
        // 1. publish
        for (int r = 0; r < NR; ++r) k_put_publish_rs<<<16, 128>>>(rk[r].staging, rk[r].f, dstep, C, M, H, q, ss, rk[r].tickets + C);
        CK(cudaDeviceSynchronize());
        // 2. host proxy, RS form (fabric.cpp: tail at sp + nb in the send slot, then nb + 8 bytes to recv (c, me) of peer p at 0)
        for (int r = 0; r < NR; ++r) {
            if (rk[r].f.doorbell[C] != sc) { printf("M=%d rank %d: doorbell c not rung\n", M, r); ++bad; }
            char* src = rk[r].f.send + g6::fab_send_off(rk[r].f, 1, C);
            for (int p = 0; p < NR; ++p) {
                if (p == r) continue;
                char* sp = src + (size_t)p * ss;
                memcpy(sp + pay, &sc, 8);
                memcpy(rk[p].f.recv + g6::fab_recv_off(rk[p].f, 1, C, r), sp, pay + 8);
            }
        }
        // 3. reduce
        for (int r = 0; r < NR; ++r) {
            glue::k_wait_peers<<<1, 32>>>(rk[r].f, dstep, C, pay, glue::ERR_AR);
            k_rs_reduce<<<dim3(H / 8 / DS41_GATHER_T, q), DS41_GATHER_T>>>(rk[r].f, dstep, C, CAG, H, M, q, ss, rk[r].gsum, rk[r].tickets + CAG);
        }
        CK(cudaDeviceSynchronize());
        // 4. host proxy, AG (a plain collective: tail at src + bytes, bytes + 8 to every peer)
        for (int r = 0; r < NR; ++r) {
            if (rk[r].f.doorbell[CAG] != sa) { printf("M=%d rank %d: doorbell c_ag not rung\n", M, r); ++bad; }
            char* src = rk[r].f.send + g6::fab_send_off(rk[r].f, 1, CAG);
            memcpy(src + pay, &sa, 8);
            for (int p = 0; p < NR; ++p) if (p != r) memcpy(rk[p].f.recv + g6::fab_recv_off(rk[p].f, 1, CAG, r), src, pay + 8);
        }
        // 5. gather
        for (int r = 0; r < NR; ++r) {
            glue::k_wait_peers<<<1, 32>>>(rk[r].f, dstep, CAG, pay, glue::ERR_AR);
            k_rs_gather<<<dim3(H / 8 / DS41_GATHER_T, q, NR - 1), DS41_GATHER_T>>>(rk[r].f, dstep, CAG, H, M, q, rk[r].gsum);
        }
        CK(cudaDeviceSynchronize());
        // 6. every rank's device sum vs the host all-reduce value
        std::vector<unsigned short> want((size_t)M * H);
        for (size_t k = 0; k < want.size(); ++k) { float s = 0.f; for (int p = 0; p < NR; ++p) s += fl(part[p][k]); want[k] = bf(s); }
        for (int r = 0; r < NR; ++r) {
            std::vector<unsigned short> have((size_t)M * H);
            CK(cudaMemcpy(have.data(), rk[r].gsum, (size_t)M * rb, cudaMemcpyDeviceToHost));
            size_t n = 0; for (size_t k = 0; k < have.size(); ++k) n += have[k] != want[k];
            const unsigned err = *rk[r].f.error;
            printf("rs+ag sum vs all-reduce rank=%d M=%d Q=%d identical=%s error=0x%x%s\n", r, M, q, n ? "no" : "yes", err,
                   n ? (" (" + std::to_string(n) + " elements differ)").c_str() : "");
            if (n || err) ++bad;
        }
        for (Rank& R : rk) { CK(cudaFreeHost(R.region)); CK(cudaFree(R.staging)); CK(cudaFree(R.gsum)); CK(cudaFree(R.tickets)); }
        CK(cudaFree(dstep));
    }
    printf("ds41 rs: %s\n", bad ? "FAIL" : "ALL IDENTICAL");
    return bad ? 1 : 0;
}
