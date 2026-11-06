// Persistent host-control sockets for the opt-in S2 server. No CUDA/fabric I/O.
// Call AFTER local model load. A failed handshake never publishes ready.s2.
// The bootstrap's sockets and exchanges are Rust (engine/scheduler-rs src/control/bootstrap.rs); the settings, their
// agreement string and the S2 files stay here.
#pragma once
#include "control_io.h"
#include "ingress.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <poll.h>
#include <fcntl.h>

namespace g6s2 {
struct BootstrapConfig {
    Config scheduler;
    unsigned rank = 0, ranks = 6, timeout = 120;
    std::string head_ip, directory, runtime_signature;
    unsigned control_version = CONTROL_VERSION;   // scheduler.h; tests set another to act as a rank of another build
    uint32_t prefill_min = 0;                     // DS41: prompts of at least this many rows prefill in W-row chunks (ds41_prefill.h); 0 = every prompt runs as decode steps
    uint16_t port = 0;
    void validate() const {
        scheduler.validate();
        require(ranks >= 1 && ranks <= 6 && rank < ranks, "invalid control rank count");   // 1: one box, no workers to link
        ctl::check(g6ctl_boot_check(reinterpret_cast<const uint8_t*>(head_ip.c_str()), std::strlen(head_ip.c_str()), port, timeout));
        require(!directory.empty() && !runtime_signature.empty(), "missing bootstrap directory/runtime signature");
    }
    std::string agreement() const {
        return capability(scheduler, ranks) + std::to_string(scheduler.max_sessions) + " " +
            std::to_string(scheduler.prefill_rows) + " " + std::to_string(scheduler.two_lanes) +
            (scheduler.decode_prefill_rows ? " dpf " + std::to_string(scheduler.decode_prefill_rows) : std::string()) +
            (scheduler.decode_rounds > 1 ? " dr " + std::to_string(scheduler.decode_rounds) : std::string()) +
            (scheduler.cp_step_rows ? " cpsr " + std::to_string(scheduler.cp_step_rows) + (scheduler.cp_seg ? " seg" : "") +
                                      " after " + std::to_string(scheduler.cp_after) : std::string()) +
            (scheduler.r2_rows ? " r2 " + std::to_string(scheduler.r2_rows) : std::string()) +
            (scheduler.tail_first ? " tf" : std::string()) +
            (scheduler.c8_pack ? " c8pack" : std::string()) +
            (scheduler.park ? " park" : std::string()) +
            (scheduler.notice_max != NOTICE_MAX ? " nmax " + std::to_string(scheduler.notice_max) : std::string()) +
            (prefill_min ? " pfmin " + std::to_string(prefill_min) : std::string()) +
            " ctl " + std::to_string(control_version) + "\n" + runtime_signature;
    }
};
// The first word where a worker's agreement differs from the head's, for the refusal: "rank 2 has 'G6_X=1', the head
// has 'G6_X=2'". Empty when they are equal.
inline std::string agreement_difference(unsigned rank, const std::string& head, const std::string& worker) {
    std::string out;
    g6ctl_agreement_difference(rank, ctl::bytes(head), head.size(), ctl::bytes(worker), worker.size(), ctl::grow_string, &out);
    return out;
}
inline void atomic_file(const std::string& path, const std::string& bytes) {
    const auto temp = path + ".tmp";
    { std::ofstream f(temp, std::ios::binary | std::ios::trunc); require(bool(f), "cannot create S2 file");
      f << bytes; f.close(); require(bool(f), "S2 file write failed"); }
    require(!::rename(temp.c_str(), path.c_str()), "S2 file publication failed");
}
struct ControlLinks {
    std::vector<std::unique_ptr<ControlIO>> links; // head: rank1..NR-1; worker: only head
    std::vector<ControlIO*> peers() const {
        std::vector<ControlIO*> p; for (const auto& x : links) p.push_back(x.get()); return p;
    }
    std::vector<const g6ctl_io*> handles() const {
        std::vector<const g6ctl_io*> p; for (const auto& x : links) p.push_back(x ? x->handle() : nullptr); return p;
    }
    void shutdown() { for (auto& p : links) if (p) p->shutdown(); }
};
namespace ctl {
inline g6ctl_boot boot(const BootstrapConfig& c, const std::string& agreement) {
    g6ctl_boot b{};
    b.rank = c.rank; b.ranks = c.ranks; b.timeout = c.timeout; b.port = c.port;
    b.head_ip = bytes(c.head_ip); b.head_ip_n = std::strlen(c.head_ip.c_str());
    b.directory = bytes(c.directory); b.directory_n = std::strlen(c.directory.c_str());
    b.agreement = bytes(agreement); b.agreement_n = agreement.size();
    return b;
}
} // namespace ctl
// The directory must be fresh (no ready, ready.s2, abort.s2, quit, cmd.0 or req.0). The head returns with every rank's
// identity and agreement acknowledged. Its caller starts the control/compute service before publishing readiness.
inline ControlLinks bootstrap(const BootstrapConfig& c) {
    c.validate(); sch::check_abi(); ctl::check_sys();
    const auto agree = c.agreement();
    const auto b = ctl::boot(c, agree);
    std::vector<g6ctl_io*> out(std::max(1u, c.ranks - 1), nullptr);
    size_t n = 0;
    const int32_t rc = g6ctl_bootstrap(&b, out.data(), &n);
    ControlLinks links;
    for (size_t i = 0; i < n; ++i) links.links.push_back(std::make_unique<ControlIO>(ControlIO::Adopt{}, out[i]));
    ctl::check(rc);
    return links;
}
// After bootstrap, every rank sends its lines (one stored entry each). The head keeps the lines every rank has, in its own
// order, and sends them back. Every rank then starts from the same stored entries (pool.h keep_agreed). idle: wait for
// the other ranks with no receive timeout, as capture_barrier (Flash agrees after the graph capture and its checkpoint
// checks, whose times differ by rank)
// A worker sends its lines before the head reads them, under the link's send timeout (BootstrapConfig timeout,
// GR8_S2_TIMEOUT, 120 s). Lines past the socket buffers (about 110 bytes an entry, a few thousand entries) wait for the
// head. If the head works longer than the timeout first (Flash entry verification), the worker fails with "control send
// failed". Widen the timeout then.
inline std::string agree_lines(const BootstrapConfig& c, ControlLinks& links, const std::string& mine, bool idle = false) {
    const auto h = links.handles();
    std::string out;
    ctl::check(g6ctl_agree_lines(c.rank, h.data(), h.size(), ctl::bytes(mine), mine.size(), idle, ctl::grow_string, &out));
    return out;
}
// All-rank barrier after the boot graph capture (native.cuh capture_all, which runs after bootstrap): every worker reports
// "captured" to the head and the head waits for all of them before it publishes ready.s2 (Service::run). The wait is
// idle (no receive timeout, as the event reader): the capture time of the ranks may differ by more than the bootstrap
// timeout, and a rank that dies mid-capture closes its link (recv 0) or fails the TCP keepalive.
inline void capture_barrier(const BootstrapConfig& c, ControlLinks& links) {
    const auto agree = c.agreement();
    const auto b = ctl::boot(c, agree);
    const auto h = links.handles();
    ctl::check(g6ctl_capture_barrier(&b, h.data(), h.size()));
}
inline void publish_ready(const BootstrapConfig& c, const ControlLinks& links) {
    require(c.rank == 0 && links.links.size() == c.ranks - 1, "ready before all-rank bootstrap");
    for (auto& p : links.links) require(bool(p), "ready with missing rank");
    require(!exists(c.directory + "/abort.s2"), "aborted before ready");
    atomic_file(c.directory + "/ready.s2", capability(c.scheduler, c.ranks));
}
} // namespace g6s2
