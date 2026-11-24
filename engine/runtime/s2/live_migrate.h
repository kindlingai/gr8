// Host-only two-phase live bulk-then-tail session migration for EL4c (#160).
// Migrates active S2 sessions (SESS <= 8) across arbitrary TP widths & memberships (e.g. 4<->3, 4<->6, 1<->2):
//   Phase 1 (Bulk, while decoding): streams immutable prefix KV rows [0, bulk_end_row) from source CP/replicated owners
//     to target owners over a peer transport (TCP loopback in host tests, g6::Fabric bulk proxy on cluster), keeping
//     every page already owned by the same physical box in place with ZERO network transfer.
//   Phase 2 (Tail, at S2 event fence): once decoding pauses at `final_row >= bulk_end_row`, transfers ONLY the tiny
//     delta rows [bulk_end_row, final_row) plus per-session metadata (`CanonSessionMeta`) and optional opaque
//     draft/KDA state envelope (`#161`), minimizing cutover pause (<200 ms target).
//   Rollback: any peer error or abort during Phase 1 fails closed without disturbing active decoding sessions.
// Zero CUDA dependency; tested under ASan+UBSan in engine/runtime/s2/tests/live_migrate_test.cpp.
#pragma once

#include "canon_mover.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace g6s2 {

struct MigrateLayout {
    std::vector<int> members; // pool box index per rank (0..nr-1)
    uint32_t cp_nr = 0;       // 0 = replicated (all ranks hold all pages), >0 = CP width (== members.size())
    uint32_t below_rows = 0;  // CP R in rows (multiple of 64; 0 if replicated)

    int rank_of_box(int box) const {
        for (size_t r = 0; r < members.size(); ++r)
            if (members[r] == box) return int(r);
        return -1;
    }
};

struct PageTransferOp {
    uint32_t binding = 0;
    uint32_t lp = 0;          // logical page index
    int src_box = -1;         // pool box index of source owner
    int dst_box = -1;         // pool box index of destination owner
    bool local = false;       // true iff src_box == dst_box (0 network bytes!)
};

struct PageTransferPlan {
    std::vector<PageTransferOp> ops;
    size_t local_pages = 0;
    size_t remote_pages = 0;
};

// Returns whether `box` holds logical page `lp` of a binding with `flag` (bit 1 = CP) in `lay`.
inline bool layout_box_holds_page(const MigrateLayout& lay, int box, uint8_t flag, uint32_t lp) {
    const int r = lay.rank_of_box(box);
    if (r < 0) return false;
    const bool is_cp = (flag & 2u) != 0 && lay.cp_nr > 0;
    if (!is_cp) return true;
    const uint32_t rp = lay.below_rows / 64u;
    if (lp < rp) return true;
    return (lp % lay.cp_nr) == uint32_t(r);
}

// Picks the deterministic source pool box holding logical page `lp` for destination box `dst_box`.
// Prefers `dst_box` itself whenever `dst_box` already holds `lp` in `src` (0 network transfer!).
inline int pick_source_box(const MigrateLayout& src, int dst_box, uint8_t flag, uint32_t lp) {
    if (layout_box_holds_page(src, dst_box, flag, lp)) return dst_box;
    const bool is_cp = (flag & 2u) != 0 && src.cp_nr > 0;
    if (!is_cp) return src.members.empty() ? -1 : src.members[0];
    const uint32_t rp = src.below_rows / 64u;
    if (lp < rp) return src.members.empty() ? -1 : src.members[0];
    const uint32_t owner_r = lp % src.cp_nr;
    return owner_r < src.members.size() ? src.members[owner_r] : -1;
}

// Plans per-binding, per-page ownership transfers for rows [first_row, end_row) from `src` to `dst`.
inline PageTransferPlan plan_page_transfers(const MigrateLayout& src, const MigrateLayout& dst,
                                            const std::vector<uint8_t>& src_flags,
                                            const std::vector<uint8_t>& dst_flags,
                                            uint64_t first_row, uint64_t end_row) {
    PageTransferPlan plan;
    if (end_row <= first_row) return plan;
    if (src_flags.size() != dst_flags.size())
        throw std::runtime_error("plan_page_transfers: binding count mismatch");
    const uint32_t first_lp = uint32_t(first_row / 64u);
    const uint32_t end_lp = uint32_t((end_row + 63u) / 64u);
    const uint32_t nb = uint32_t(dst_flags.size());
    for (uint32_t b = 0; b < nb; ++b) {
        for (uint32_t lp = first_lp; lp < end_lp; ++lp) {
            for (int dbox : dst.members) {
                if (!layout_box_holds_page(dst, dbox, dst_flags[b], lp)) continue;
                const int sbox = pick_source_box(src, dbox, src_flags[b], lp);
                if (sbox < 0) throw std::runtime_error("plan_page_transfers: no source box owns page");
                const bool loc = (sbox == dbox);
                plan.ops.push_back(PageTransferOp{b, lp, sbox, dbox, loc});
                if (loc) ++plan.local_pages;
                else ++plan.remote_pages;
            }
        }
    }
    return plan;
}

// Wire packet carrying one logical page slice [row0, row1) of one binding of one session, OR tail metadata.
enum class MigrateMsgKind : uint8_t {
    PageChunk = 1,
    TailMeta = 2,
    PhaseDone = 3,
};

struct MigratePagePayload {
    uint64_t session_id = 0;
    uint32_t binding = 0;
    uint32_t lp = 0;
    uint32_t row_off = 0; // first row inside page lp (0..63)
    uint32_t row_cnt = 0; // number of rows carried (1..64 - row_off)
    uint32_t mla_row = 0;
    uint8_t has_idx = 0;
    std::vector<uint8_t> mla; // row_cnt * mla_row bytes
    std::vector<uint8_t> idx; // has_idx ? row_cnt * 132 : 0 bytes
};

struct MigrateTailMeta {
    uint64_t session_id = 0;
    uint64_t final_rows = 0;
    CanonSessionMeta meta;
    std::vector<uint8_t> state_envelope; // optional EL4d (#161) KDA/DFlash2/MTP canonical state
};

inline std::vector<uint8_t> encode_page_msg(const MigratePagePayload& p) {
    std::vector<uint8_t> out;
    out.reserve(32 + p.mla.size() + p.idx.size());
    out.push_back(uint8_t(MigrateMsgKind::PageChunk));
    out.push_back(p.has_idx);
    out.push_back(0); out.push_back(0);
    canon_put_le64(out, p.session_id);
    canon_put_le32(out, p.binding);
    canon_put_le32(out, p.lp);
    canon_put_le32(out, p.row_off);
    canon_put_le32(out, p.row_cnt);
    canon_put_le32(out, p.mla_row);
    out.insert(out.end(), p.mla.begin(), p.mla.end());
    if (p.has_idx) out.insert(out.end(), p.idx.begin(), p.idx.end());
    return out;
}

inline MigratePagePayload decode_page_msg(const uint8_t* data, size_t n) {
    if (n < 32 || data[0] != uint8_t(MigrateMsgKind::PageChunk))
        throw std::runtime_error("decode_page_msg: invalid header");
    MigratePagePayload p;
    p.has_idx = data[1];
    p.session_id = canon_le64(data + 4);
    p.binding = canon_le32(data + 12);
    p.lp = canon_le32(data + 16);
    p.row_off = canon_le32(data + 20);
    p.row_cnt = canon_le32(data + 24);
    p.mla_row = canon_le32(data + 28);
    const size_t mb = size_t(p.row_cnt) * size_t(p.mla_row);
    const size_t xb = p.has_idx ? size_t(p.row_cnt) * 132u : 0u;
    if (p.row_off + p.row_cnt > 64 || 32u + mb + xb != n)
        throw std::runtime_error("decode_page_msg: payload length mismatch");
    p.mla.assign(data + 32, data + 32 + mb);
    if (p.has_idx) p.idx.assign(data + 32 + mb, data + 32 + mb + xb);
    return p;
}

inline std::vector<uint8_t> encode_tail_meta(const MigrateTailMeta& tm) {
    std::vector<uint8_t> out;
    out.push_back(uint8_t(MigrateMsgKind::TailMeta));
    out.push_back(0); out.push_back(0); out.push_back(0);
    canon_put_le64(out, tm.session_id);
    canon_put_le64(out, tm.final_rows);
    canon_put_le64(out, tm.meta.request_id);
    canon_put_le64(out, tm.meta.export_id);
    canon_put_le32(out, tm.meta.prompt_tokens);
    canon_put_le32(out, tm.meta.emitted_tokens);
    canon_put_le32(out, tm.meta.committed_rows);
    canon_put_le32(out, tm.meta.remaining_max_tokens);
    canon_put_le32(out, uint32_t(tm.meta.req_header.size()));
    canon_put_le32(out, uint32_t(tm.meta.tokens.size()));
    canon_put_le32(out, uint32_t(tm.state_envelope.size()));
    out.insert(out.end(), tm.meta.req_header.begin(), tm.meta.req_header.end());
    for (int32_t t : tm.meta.tokens) canon_put_le32(out, uint32_t(t));
    out.insert(out.end(), tm.state_envelope.begin(), tm.state_envelope.end());
    return out;
}

inline MigrateTailMeta decode_tail_meta(const uint8_t* data, size_t n) {
    if (n < 64 || data[0] != uint8_t(MigrateMsgKind::TailMeta))
        throw std::runtime_error("decode_tail_meta: invalid header");
    MigrateTailMeta tm;
    tm.session_id = canon_le64(data + 4);
    tm.final_rows = canon_le64(data + 12);
    tm.meta.request_id = canon_le64(data + 20);
    tm.meta.export_id = canon_le64(data + 28);
    tm.meta.prompt_tokens = canon_le32(data + 36);
    tm.meta.emitted_tokens = canon_le32(data + 40);
    tm.meta.committed_rows = canon_le32(data + 44);
    tm.meta.remaining_max_tokens = canon_le32(data + 48);
    const uint32_t hdr_n = canon_le32(data + 52);
    const uint32_t tok_n = canon_le32(data + 56);
    const uint32_t env_n = canon_le32(data + 60);
    const size_t expect = 64u + size_t(hdr_n) + size_t(tok_n) * 4u + size_t(env_n);
    if (expect != n) throw std::runtime_error("decode_tail_meta: length mismatch");
    size_t off = 64;
    tm.meta.req_header.assign(reinterpret_cast<const char*>(data + off), hdr_n);
    off += hdr_n;
    tm.meta.tokens.resize(tok_n);
    for (uint32_t i = 0; i < tok_n; ++i) {
        tm.meta.tokens[i] = int32_t(canon_le32(data + off));
        off += 4;
    }
    tm.state_envelope.assign(data + off, data + off + env_n);
    return tm;
}

inline std::vector<uint8_t> encode_phase_done(uint32_t phase_id) {
    std::vector<uint8_t> out(8, 0);
    out[0] = uint8_t(MigrateMsgKind::PhaseDone);
    out[4] = uint8_t(phase_id & 0xff);
    out[5] = uint8_t((phase_id >> 8) & 0xff);
    out[6] = uint8_t((phase_id >> 16) & 0xff);
    out[7] = uint8_t((phase_id >> 24) & 0xff);
    return out;
}

// Pluggable point-to-point transport between pool boxes (loopback TCP in host tests, Fabric bulk proxy on cluster).
struct MigrateTransport {
    virtual ~MigrateTransport() = default;
    virtual bool send_frame(int dst_box, const uint8_t* data, size_t n, std::string& err) = 0;
    virtual bool recv_frame(int src_box, std::vector<uint8_t>& out, int timeout_ms, std::string& err) = 0;
};

// Framed TCP transport over pre-connected peer sockets (one socket per (src_box, dst_box) pair).
class SocketMigrateTransport : public MigrateTransport {
    std::map<int, int> send_fds_; // dst_box -> fd
    std::map<int, int> recv_fds_; // src_box -> fd

    static constexpr int kSendFlags =
#ifdef MSG_NOSIGNAL
        MSG_NOSIGNAL
#else
        0
#endif
        ;

    static void no_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
        int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#else
        (void)fd;
#endif
    }

    static bool send_all(int fd, const uint8_t* p, size_t n, std::string& err) {
        size_t off = 0;
        while (off < n) {
            const ssize_t k = ::send(fd, p + off, n - off, kSendFlags);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) {
                err = std::string("migrate send failed: ") + std::strerror(errno);
                return false;
            }
            off += size_t(k);
        }
        return true;
    }

    static bool recv_all(int fd, uint8_t* p, size_t n, int timeout_ms, std::string& err) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        size_t off = 0;
        while (off < n) {
            if (timeout_ms >= 0) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now()).count();
                if (left <= 0) {
                    err = "migrate recv timed out";
                    return false;
                }
                pollfd pf{fd, POLLIN, 0};
                const int rc = ::poll(&pf, 1, int(left));
                if (rc < 0 && errno == EINTR) continue;
                if (rc == 0) { err = "migrate recv timed out"; return false; }
                if (rc < 0) { err = std::string("migrate poll failed: ") + std::strerror(errno); return false; }
            }
            const ssize_t k = ::recv(fd, p + off, n - off, 0);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) {
                err = "migrate peer closed connection";
                return false;
            }
            off += size_t(k);
        }
        return true;
    }

public:
    ~SocketMigrateTransport() override { close_all(); }

    void add_send_fd(int dst_box, int fd) { no_sigpipe(fd); send_fds_[dst_box] = fd; }
    void add_recv_fd(int src_box, int fd) { no_sigpipe(fd); recv_fds_[src_box] = fd; }

    void close_all() {
        for (auto& [b, fd] : send_fds_) if (fd >= 0) ::close(fd);
        for (auto& [b, fd] : recv_fds_) if (fd >= 0) ::close(fd);
        send_fds_.clear();
        recv_fds_.clear();
    }

    bool send_frame(int dst_box, const uint8_t* data, size_t n, std::string& err) override {
        auto it = send_fds_.find(dst_box);
        if (it == send_fds_.end() || it->second < 0) {
            err = "migrate: no send link to box " + std::to_string(dst_box);
            return false;
        }
        uint8_t hdr[4] = {uint8_t(n & 0xff), uint8_t((n >> 8) & 0xff),
                          uint8_t((n >> 16) & 0xff), uint8_t((n >> 24) & 0xff)};
        return send_all(it->second, hdr, 4, err) && (n == 0 || send_all(it->second, data, n, err));
    }

    bool recv_frame(int src_box, std::vector<uint8_t>& out, int timeout_ms, std::string& err) override {
        auto it = recv_fds_.find(src_box);
        if (it == recv_fds_.end() || it->second < 0) {
            err = "migrate: no recv link from box " + std::to_string(src_box);
            return false;
        }
        uint8_t hdr[4];
        if (!recv_all(it->second, hdr, 4, timeout_ms, err)) return false;
        const uint32_t n = canon_le32(hdr);
        if (n > (256u << 20)) { err = "migrate: oversized frame"; return false; }
        out.resize(n);
        if (n > 0 && !recv_all(it->second, out.data(), n, timeout_ms, err)) return false;
        return true;
    }
};

struct MigrateSessionSpec {
    uint64_t session_id = 0;
    uint32_t mla_row = 0;
    std::vector<uint8_t> src_flags;
    std::vector<uint8_t> dst_flags;
    uint64_t bulk_end_row = 0;   // Phase 1 bulk copies [0, bulk_end_row) while live decode continues
    uint64_t final_end_row = 0;  // Phase 2 tail copies [bulk_end_row, final_end_row) at step fence
};

struct MigrateStats {
    size_t bulk_local_pages = 0;
    size_t bulk_remote_pages = 0;
    size_t bulk_remote_bytes = 0;
    size_t tail_local_pages = 0;
    size_t tail_remote_pages = 0;
    size_t tail_remote_bytes = 0;
    double bulk_ms = 0.0;
    double tail_ms = 0.0;
};

// Callbacks connecting the migration coordinator to a box's local KV storage (CPU buffer in host tests,
// GPU bindings via canon_mover.cuh in g6m4).
struct MigrateBoxIO {
    // Reads rows [row_off, row_off + row_cnt) of logical page `lp` on `binding` of `session_id` from local source KV.
    std::function<void(uint64_t session_id, uint32_t binding, uint32_t lp, uint32_t row_off, uint32_t row_cnt,
                       uint8_t* mla_out, uint8_t* idx_out)> read_rows;
    // Writes rows [row_off, row_off + row_cnt) of logical page `lp` on `binding` of `session_id` into target KV.
    std::function<void(uint64_t session_id, uint32_t binding, uint32_t lp, uint32_t row_off, uint32_t row_cnt,
                       const uint8_t* mla_in, const uint8_t* idx_in)> write_rows;
    // Fast local in-place retention when src_box == dst_box (0 network bytes; optional — falls back to read+write).
    std::function<void(uint64_t session_id, uint32_t binding, uint32_t lp, uint32_t row_off, uint32_t row_cnt)> copy_local;
    // Supplies tail metadata + optional EL4d state envelope on the source head box (src.members[0]).
    std::function<MigrateTailMeta(uint64_t session_id)> get_tail_meta;
    // Receives tail metadata + state envelope on each destination box.
    std::function<void(const MigrateTailMeta&)> on_tail_meta;
};

// Runs one phase (bulk: phase_id=1 over [0, bulk_end_row); tail: phase_id=2 over [bulk_end_row, final_end_row))
// on pool box `my_box`. Sends all pages `my_box` owns to remote destination boxes, applies all local-keep ops
// with 0 network transfer, and receives all pages destined for `my_box` from remote source boxes.
// Fails closed immediately if `cancel` is set or any peer disconnects / times out.
inline bool run_migrate_phase(int my_box, uint32_t phase_id, const MigrateLayout& src, const MigrateLayout& dst,
                              const std::vector<MigrateSessionSpec>& sessions, MigrateTransport& xport,
                              const MigrateBoxIO& io, const std::atomic<bool>* cancel, int timeout_ms,
                              size_t& out_local_pages, size_t& out_remote_pages, size_t& out_remote_bytes,
                              std::string& err) {
    out_local_pages = out_remote_pages = out_remote_bytes = 0;

    struct PlannedSlice {
        const MigrateSessionSpec* sess = nullptr;
        PageTransferOp op;
        uint32_t row_off = 0, row_cnt = 0;
    };
    std::vector<PlannedSlice> local_ops;
    std::map<int, std::vector<PlannedSlice>> sends_by_dst;
    std::map<int, size_t> recvs_by_src;

    for (const auto& s : sessions) {
        const uint64_t first = (phase_id == 1) ? 0ull : s.bulk_end_row;
        const uint64_t end = (phase_id == 1) ? s.bulk_end_row : s.final_end_row;
        if (end <= first) continue;
        const PageTransferPlan p = plan_page_transfers(src, dst, s.src_flags, s.dst_flags, first, end);
        for (const auto& op : p.ops) {
            const uint64_t page_lo = uint64_t(op.lp) * 64u;
            const uint64_t lo = std::max(first, page_lo);
            const uint64_t hi = std::min(end, page_lo + 64u);
            const uint32_t roff = uint32_t(lo - page_lo);
            const uint32_t rcnt = uint32_t(hi - lo);
            if (op.local) {
                if (op.src_box == my_box) local_ops.push_back({&s, op, roff, rcnt});
            } else {
                if (op.src_box == my_box) sends_by_dst[op.dst_box].push_back({&s, op, roff, rcnt});
                if (op.dst_box == my_box) recvs_by_src[op.src_box] += 1;
            }
        }
    }

    // Tail phase also carries MigrateTailMeta from src.members[0] to every destination box.
    const int meta_src = src.members.empty() ? -1 : src.members[0];
    const bool send_meta = (phase_id == 2 && my_box == meta_src && io.get_tail_meta);
    const bool recv_meta = (phase_id == 2 && dst.rank_of_box(my_box) >= 0 && my_box != meta_src && meta_src >= 0);
    if (send_meta) {
        for (int dbox : dst.members) {
            if (dbox == my_box) {
                if (io.on_tail_meta) {
                    for (const auto& s : sessions) io.on_tail_meta(io.get_tail_meta(s.session_id));
                }
            } else {
                (void)sends_by_dst[dbox]; // ensure peer receives meta + PhaseDone
            }
        }
    }
    if (recv_meta) recvs_by_src[meta_src] += sessions.size();

    // 1. Apply local-keep pages in place (0 network transfer!).
    for (const auto& ps : local_ops) {
        if (cancel && cancel->load()) { err = "migrate aborted"; return false; }
        if (io.copy_local) {
            io.copy_local(ps.sess->session_id, ps.op.binding, ps.op.lp, ps.row_off, ps.row_cnt);
        } else {
            const bool has_idx = (ps.sess->src_flags[ps.op.binding] & 1u) != 0;
            std::vector<uint8_t> m(size_t(ps.row_cnt) * ps.sess->mla_row);
            std::vector<uint8_t> x(has_idx ? size_t(ps.row_cnt) * 132u : 0u);
            io.read_rows(ps.sess->session_id, ps.op.binding, ps.op.lp, ps.row_off, ps.row_cnt,
                         m.data(), has_idx ? x.data() : nullptr);
            io.write_rows(ps.sess->session_id, ps.op.binding, ps.op.lp, ps.row_off, ps.row_cnt,
                          m.data(), has_idx ? x.data() : nullptr);
        }
        ++out_local_pages;
    }

    // 2. Send all outbound pages (and tail metadata) concurrently with receiving via per-peer order:
    //    to avoid TCP buffer deadlock when boxes both send and receive, send in a helper loop or chunked non-blocking,
    //    or since SocketMigrateTransport has dedicated directional fds, run sends first in a worker or interleaved.
    //    Here we spawn one short sender thread per box call when both sending and receiving, keeping deadlock impossible!
    std::string send_err;
    bool send_ok = true;
    auto do_sends = [&]() {
        for (const auto& [dbox, list] : sends_by_dst) {
            for (const auto& ps : list) {
                if (cancel && cancel->load()) { send_err = "migrate aborted"; send_ok = false; return; }
                const bool has_idx = (ps.sess->src_flags[ps.op.binding] & 1u) != 0;
                MigratePagePayload mp;
                mp.session_id = ps.sess->session_id;
                mp.binding = ps.op.binding;
                mp.lp = ps.op.lp;
                mp.row_off = ps.row_off;
                mp.row_cnt = ps.row_cnt;
                mp.mla_row = ps.sess->mla_row;
                mp.has_idx = has_idx ? 1u : 0u;
                mp.mla.resize(size_t(ps.row_cnt) * ps.sess->mla_row);
                if (has_idx) mp.idx.resize(size_t(ps.row_cnt) * 132u);
                io.read_rows(mp.session_id, mp.binding, mp.lp, mp.row_off, mp.row_cnt,
                             mp.mla.data(), has_idx ? mp.idx.data() : nullptr);
                const auto wire = encode_page_msg(mp);
                if (!xport.send_frame(dbox, wire.data(), wire.size(), send_err)) {
                    send_ok = false;
                    return;
                }
                ++out_remote_pages;
                out_remote_bytes += wire.size();
            }
            if (send_meta && dbox != my_box) {
                for (const auto& s : sessions) {
                    const auto wire = encode_tail_meta(io.get_tail_meta(s.session_id));
                    if (!xport.send_frame(dbox, wire.data(), wire.size(), send_err)) {
                        send_ok = false;
                        return;
                    }
                    out_remote_bytes += wire.size();
                }
            }
            const auto done = encode_phase_done(phase_id);
            if (!xport.send_frame(dbox, done.data(), done.size(), send_err)) {
                send_ok = false;
                return;
            }
        }
    };

    // Receive from every source box that sends to `my_box` until PhaseDone arrives from each.
    auto do_recvs = [&]() -> bool {
        std::vector<uint8_t> frame;
        for (const auto& [sbox, expected_cnt] : recvs_by_src) {
            (void)expected_cnt;
            for (;;) {
                if (cancel && cancel->load()) { err = "migrate aborted"; return false; }
                if (!xport.recv_frame(sbox, frame, timeout_ms, err)) return false;
                if (frame.empty()) { err = "migrate: empty frame"; return false; }
                const auto kind = MigrateMsgKind(frame[0]);
                if (kind == MigrateMsgKind::PhaseDone) break;
                if (kind == MigrateMsgKind::PageChunk) {
                    const MigratePagePayload mp = decode_page_msg(frame.data(), frame.size());
                    io.write_rows(mp.session_id, mp.binding, mp.lp, mp.row_off, mp.row_cnt,
                                  mp.mla.data(), mp.has_idx ? mp.idx.data() : nullptr);
                } else if (kind == MigrateMsgKind::TailMeta) {
                    const MigrateTailMeta tm = decode_tail_meta(frame.data(), frame.size());
                    if (io.on_tail_meta) io.on_tail_meta(tm);
                } else {
                    err = "migrate: unknown message kind";
                    return false;
                }
            }
        }
        return true;
    };

    if (!sends_by_dst.empty() && !recvs_by_src.empty()) {
        std::thread st(do_sends);
        const bool rok = do_recvs();
        st.join();
        if (!rok) return false;
        if (!send_ok) { err = send_err; return false; }
    } else if (!sends_by_dst.empty()) {
        do_sends();
        if (!send_ok) { err = send_err; return false; }
    } else if (!recvs_by_src.empty()) {
        if (!do_recvs()) return false;
    }
    return true;
}

} // namespace g6s2
