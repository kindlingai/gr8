// Opt-in diagnostics only. The engine does not include this without
// -DGR8_ATTN_SWEEP_CAPTURE=1. No timing from a capture-enabled request is valid.
#pragma once
#include "capture_io.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <map>
#include <set>

namespace sweep_capture {
struct Context {
    int prompt_tokens=0, chunk_position=0, request_slot=0, rank=0;
    std::string request_id, phase;
};
struct Input {
    Metadata metadata;
    const int32_t* topk=nullptr;
    const int32_t* positions=nullptr;
    const int32_t* sessions=nullptr;
    int start=0;
};
class Hook {
    bool enabled_=false;
    std::string directory_, revision_, knobs_;
    std::set<int> layers_, chunks_, lengths_, ranks_;
    size_t max_records_=0, max_bytes_=0, records_=0, bytes_=0;
    std::set<std::string> seen_;
    std::map<int,std::string> selected_requests_; // first matching request per prompt length
    std::mutex mutex_;
    static std::set<int> integers(const g6v::Json& j) {
        require(j.t==g6v::Json::Arr&&!j.a.empty(),"capture selection must be a nonempty array");
        std::set<int> out;
        for(const auto& v:j.a) {auto n=v.i64();require(n>=0&&n<=1048576,"capture selection out of bounds");out.insert(int(n));}
        return out;
    }
public:
    // Empty config = OFF. Testable with host copy/sync callbacks; constructor
    // does not touch the GPU, output directory, or files in the disabled case.
    explicit Hook(const std::string& config={}) {
        if(config.empty())return;
        auto j=g6v::Json::parse(config);
        require(j.at("schema").i64()==1,"capture config schema invalid");
        directory_=j.at("output_dir").str();revision_=j.at("engine_revision").str();
        require(!directory_.empty()&&!revision_.empty(),"capture config provenance missing");
        auto nr=j.at("max_records").i64(),nb=j.at("max_bytes").i64();
        require(nr>0&&nr<=4096&&nb>0&&nb<=int64_t(8ULL<<30),"capture budget invalid");
        max_records_=size_t(nr);max_bytes_=size_t(nb);
        layers_=integers(j.at("layers"));chunks_=integers(j.at("chunk_positions"));
        lengths_=integers(j.at("prompt_lengths"));ranks_=integers(j.at("ranks"));
        require(j.at("knob_manifest").t==g6v::Json::Obj&&j.at("knob_manifest").size()>0,"missing capture knob manifest");
        knobs_=json_dump(j.at("knob_manifest"));enabled_=true;
    }
    bool enabled() const{return enabled_;}
    // copy(dst, device_src, bytes) and sync() are supplied by the engine.
    // Copies and sync are entirely outside any benchmark timing region.
    template<class Copy,class Sync> bool capture(const Context& ctx, Input in, Copy copy, Sync sync) {
        if(!enabled_||ctx.phase.empty())return false;
        std::lock_guard<std::mutex> guard(mutex_);
        auto& m=in.metadata;
        if(!layers_.count(m.layer)||!chunks_.count(ctx.chunk_position)||!lengths_.count(ctx.prompt_tokens)||!ranks_.count(ctx.rank))return false;
        const auto selected=selected_requests_.find(ctx.prompt_tokens);
        if(selected!=selected_requests_.end()&&selected->second!=ctx.request_id)return false;
        if(records_>=max_records_)return false;
        require(m.n>0&&m.n<=8192&&m.ldtk>=512&&m.ldtk<=1048576,"capture input dimensions invalid");
        std::string identity="r"+std::to_string(ctx.rank)+"_q"+ctx.request_id+"_c"+std::to_string(ctx.chunk_position)+"_l"+std::to_string(m.layer)+"_"+ctx.phase;
        require(identity.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"unsafe capture identity");
        if(seen_.count(identity))return false;
        m.prompt_tokens=ctx.prompt_tokens;m.chunk_position=ctx.chunk_position;m.rank=ctx.rank;
        m.request_id=ctx.request_id;m.request_slot=ctx.request_slot;m.dispatch_phase=ctx.phase;
        m.engine_revision=revision_;m.knob_manifest=knobs_;m.origin="engine";
        const auto meta=metadata_json(m);
        const size_t expected=80+meta.size()+size_t(m.n)*514*4;
        if(expected>max_bytes_-bytes_)return false;
        Record r;r.metadata=m;
        r.topk.assign(size_t(m.n)*512,-1);r.positions.resize(m.n);r.sessions.assign(m.n,0);
        if(m.ratio) {
            require(in.topk!=nullptr,"capture missing top-k pointer");
            if(m.ldtk==512)copy(r.topk.data(),in.topk,r.topk.size()*4);
            else for(int row=0;row<m.n;++row)copy(r.topk.data()+size_t(row)*512,in.topk+size_t(row)*m.ldtk,512*4);
        }
        if(in.sessions) {
            require(in.positions!=nullptr,"capture sessions need per-row positions");
            copy(r.sessions.data(),in.sessions,size_t(m.n)*4);
            copy(r.positions.data(),in.positions,size_t(m.n)*4);
        } else if(in.positions) copy(r.positions.data(),in.positions,4);
        else r.positions[0]=in.start;
        sync(); // staging must complete before deriving positions or hashing
        if(!in.sessions)for(int row=1;row<m.n;++row)r.positions[row]=r.positions[0]+row;
        auto data=encode(r);
        require(data.size()==expected,"capture budget accounting mismatch");
        const std::string path=directory_+"/"+identity+".asw";
        FILE* f=std::fopen(path.c_str(),"wbx");require(f!=nullptr,"cannot exclusively create capture (directory missing or file exists)");
        const bool wrote=std::fwrite(data.data(),1,data.size(),f)==data.size();
        const int closed=std::fclose(f);require(wrote&&closed==0,"capture write failed");
        ++records_;bytes_+=data.size();seen_.insert(identity);selected_requests_[ctx.prompt_tokens]=ctx.request_id;
        std::fprintf(stderr,"# attention capture %s sha256=%s bytes=%zu operands=unavailable NOT_TIMING_DATA\n",path.c_str(),sha256(data).c_str(),data.size());
        return true;
    }
};
inline std::string config_from_file(const std::string& path) {
    if(path.empty())return {};
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot open capture config");
    auto n=f.tellg();require(n>=0&&uint64_t(n)<=max_metadata,"capture config too large");
    std::string data(size_t(n),'\0');f.seekg(0);f.read(data.data(),data.size());require(bool(f),"short config read");return data;
}
} // namespace sweep_capture
