// File ingress for opt-in S2 shim. Control thread only, no per-round file polling.
#pragma once
#include "scheduler.h"
#include "../serve.h"
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace g6s2 {
inline uint64_t payload_hash(const std::string& bytes) {
    uint64_t h=14695981039346656037ull;
    for (unsigned char c : bytes) { h ^= c; h *= 1099511628211ull; }
    return h;
}
inline bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
inline std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary); require(bool(f), "cannot read S2 file");
    std::ostringstream bytes; bytes << f.rdbuf(); require(!f.bad(), "S2 read failed"); return bytes.str();
}
inline uint64_t number(const std::string& s) {
    require(!s.empty(), "empty number"); uint64_t n=0;
    for (char c:s) { require(c>='0' && c<='9' && n <= (UINT64_MAX-unsigned(c-'0'))/10, "bad/overflow number"); n=n*10+unsigned(c-'0'); }
    return n;
}
struct Command { bool admit=false; uint64_t epoch=0, request=0, hash=0; };
inline Command parse_command(const std::string& text, uint64_t epoch) {
    std::istringstream in(text); std::string kind,e,r,h,extra;
    require(bool(in>>kind>>e>>r>>h) && !(in>>extra), "invalid ingress command");
    require(kind=="ADMIT" || kind=="CANCEL", "unknown ingress command");
    Command c{kind=="ADMIT",number(e),number(r),number(h)};
    require(c.epoch==epoch && (c.admit || c.hash==0), "ingress epoch/cancel mismatch"); return c;
}
inline std::string capability(Config c, unsigned ranks) {
    c.validate(); require(ranks==3 || ranks==4 || ranks==6, "invalid rank count");
    const std::string base=std::to_string(c.epoch)+" "+std::to_string(ranks)+" "+std::to_string(c.pool_rows)+" "+
        std::to_string(c.mtp)+" "+std::to_string(c.guard())+" "+std::to_string(c.max_rows)+" "+std::to_string(g6rt::LP_MAX);
    // Version 3 adds insert rules. Version 4 appends the immutable steering receipt.
    if (!c.steering.loaded) return "G6S2 3 "+base+"\n";
    const auto& s=c.steering;
    return "G6S2 4 "+base+" "+s.profile_name+" "+g6rt::steering::digest_hex(s.profile_digest)+" "+
        g6rt::steering::scope_name(s.scope)+" "+std::to_string(s.active_layers)+" "+std::to_string(s.forced)+" "+
        std::to_string(s.forced_scale_bits)+"\n";
}
class Ingress {
    std::string dir_; uint64_t epoch_, next_=0;
public:
    Ingress(std::string dir, uint64_t epoch):dir_(std::move(dir)),epoch_(epoch){require(epoch,"invalid epoch");}
    std::optional<Command> next() {
        require(!exists(dir_+"/abort.s2"), "shim aborted epoch");
        auto p=dir_+"/cmd."+std::to_string(next_);
        if (!exists(p)) return {};
        auto c=parse_command(read_file(p),epoch_);
        require(next_!=UINT64_MAX,"command sequence exhausted");++next_;return c;
    }
    std::string request(const Command& c) const {
        require(c.admit && c.epoch==epoch_,"not this epoch's admission");
        auto bytes=read_file(dir_+"/req."+std::to_string(c.request));
        require(payload_hash(bytes)==c.hash,"published request payload mismatch");return bytes;
    }
};
} // namespace g6s2
