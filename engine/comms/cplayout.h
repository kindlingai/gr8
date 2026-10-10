// Variable slot layout for a fabric whose collectives differ in size and fan-out (CP fabric `fabc`).
// Shared by the RDMA proxy (fabric.cpp) and the one-GPU loopback (m4sim.cu), so both address the same bytes.
//
// Per collective c: cap[c] = max payload bytes; dst_k[c] = 0 -> every peer gets the payload (all-gather form),
// dst_k[c] = k in 1..nr-1 -> only peer (rank + k) % nr gets it (single destination; the receiver's one source is
// (rank - k) % nr). stride[c] = round_up(cap[c] + 8, 256); the u64 seq tail sits right after the payload.
// send: [2 parity][c] stride[c] each. recv: [2 parity][c] (nr - 1 slots for dst_k = 0, 1 slot otherwise) x stride[c].
// All-gather recv slots are compact: rank r keeps sender s's copy in slot (s - r + nr) % nr - 1 (no self slot).
// Every rank builds the same layout from the same caps, so a sender computes the receiver's offsets locally.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace g6 {

// What a kernel needs to publish / wait on collective c of a VLayout fabric, both parities resolved on the host
// (the step parity is only known on the device). recv[par][j]: j-th source slot (j < nsrc), src[j]: its sender rank.
struct VSlotDev
{
    char* send[2];
    const char* recv[2][7];
    unsigned long long* doorbell;   // this collective's doorbell word
    unsigned int* error;
    int c, ncoll, nsrc;
    int src[7];
};

struct VLayout
{
    int nr = 0, ncoll = 0;
    std::vector<uint32_t> cap;
    std::vector<uint8_t> dst_k;
    std::vector<size_t> stride, soff, roff;   // offsets within one parity half
    size_t send_half = 0, recv_half = 0;      // bytes per parity

    static size_t round_stride(size_t cap) { return (cap + 8 + 255) & ~size_t(255); }
    int nslot(int c) const { return dst_k[c] ? 1 : nr - 1; }

    void make(int nr_, const std::vector<uint32_t>& cap_, const std::vector<uint8_t>& dst_k_)
    {
        nr = nr_; ncoll = (int)cap_.size(); cap = cap_; dst_k = dst_k_;
        stride.resize(ncoll); soff.resize(ncoll); roff.resize(ncoll);
        send_half = recv_half = 0;
        for (int c = 0; c < ncoll; ++c)
        {
            stride[c] = round_stride(cap[c]);
            soff[c] = send_half; send_half += stride[c];
            roff[c] = recv_half; recv_half += (size_t)nslot(c) * stride[c];
        }
    }
    size_t send_bytes() const { return 2 * send_half; }
    size_t recv_bytes() const { return 2 * recv_half; }
    // byte offsets from the send / recv base
    size_t send_at(int c, int par) const { return (size_t)par * send_half + soff[c]; }
    size_t recv_at(int c, int par, int slot) const { return (size_t)par * recv_half + roff[c] + (size_t)slot * stride[c]; }
    // slot at receiver r holding sender s's copy of collective c (-1: s does not send c to r)
    int slot_of(int c, int r, int s) const
    {
        if (s == r) return -1;
        if (dst_k[c]) return (s + dst_k[c]) % nr == r ? 0 : -1;
        return (s - r + nr) % nr - 1;
    }
    // kernel view of collective c on rank me, given that rank's region bases (RDMA fabric and loopback alike)
    VSlotDev vslot(int c, int me, char* send_base, char* recv_base, unsigned long long* doorbell, unsigned int* error) const
    {
        VSlotDev v{};
        for (int par = 0; par < 2; ++par) v.send[par] = send_base + send_at(c, par);
        v.doorbell = doorbell + c; v.error = error; v.c = c; v.ncoll = ncoll;
        for (int s = 0; s < nr; ++s)
        {
            const int j = slot_of(c, me, s);
            if (j < 0) continue;
            for (int par = 0; par < 2; ++par) v.recv[par][v.nsrc] = recv_base + recv_at(c, par, j);
            v.src[v.nsrc++] = s;
        }
        return v;
    }
};

// CP decode collectives, numbered in GPU publish order (the proxy posts strictly in c
// order): per CP layer i, q(i) = q_abs all-gather, cand(i) = (score, pos) candidates all-gather (full-index layers only),
// part(i, k) for k = 1..nr-1 = the partial (m, l, c) of head group (rank + k) % nr, single destination.
// Capacities scale with mmax (the most verify rows a step may carry); a step with M rows sends M x row bytes.
struct CpColls
{
    static constexpr int HEADS = 64, KV_RANK = 512, QK_DIM = 576, TOPK = 2048;
    int nr = 0, nl = 0, mmax = 0;
    uint32_t q_row = 0, cand_row = 0, part_row = 0;   // payload bytes per verify row
    std::vector<int> q, cand, part;                   // [layer], [layer] (-1 = none), [layer * (nr - 1) + k - 1]
    std::vector<uint32_t> cap;
    std::vector<uint8_t> dst_k;

    void make(const std::vector<bool>& full, int nr_, int mmax_)
    {
        nr = nr_; nl = (int)full.size(); mmax = mmax_;
        // rows sized for the WIDEST head group (NR=6: 11 of 11/11/11/11/10/10; NR=4: 16), rounded to 16 B so
        // every M x row is a whole number of uint4 (k_vdeliver / k_vput); a group with fewer heads sends fewer bytes,
        // the tail stays at M x row (the wire size)
        const int hpr = (HEADS + nr - 1) / nr;
        q_row = (uint32_t)hpr * QK_DIM * 4;
        cand_row = (uint32_t)TOPK * 8;
        part_row = ((uint32_t)hpr * (KV_RANK + 2) * 4 + 15) & ~15u;
        q.assign(nl, -1); cand.assign(nl, -1); part.assign((size_t)nl * (nr - 1), -1);
        cap.clear(); dst_k.clear();
        auto add = [&](uint32_t row, int k) { cap.push_back(row * (uint32_t)mmax); dst_k.push_back((uint8_t)k); return (int)cap.size() - 1; };
        for (int i = 0; i < nl; ++i)
        {
            q[i] = add(q_row, 0);
            if (full[i]) cand[i] = add(cand_row, 0);
            for (int k = 1; k < nr; ++k) part[(size_t)i * (nr - 1) + k - 1] = add(part_row, k);
        }
    }
    int ncoll() const { return (int)cap.size(); }
    // proxy sizes of one step with M rows (every collective of the step)
    std::vector<uint32_t> sizes(int M) const
    {
        std::vector<uint32_t> s(cap.size());
        for (size_t c = 0; c < cap.size(); ++c) s[c] = cap[c] / (uint32_t)mmax * (uint32_t)M;
        return s;
    }
};

}  // namespace g6
