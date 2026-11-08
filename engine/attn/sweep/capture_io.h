// Host-only attention-index capture v1. No CUDA, no external libraries.
#pragma once
#include "../../runtime/json.h"
#include "topk_fixture.h"
#include <array>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>

namespace sweep_capture {
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
// SHA-256, expressed directly from the standard round equations.
inline std::string sha256(const std::string& input) {
    static constexpr uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::string data=input; data.push_back(char(0x80));
    while (data.size()%64 != 56) data.push_back(0);
    uint64_t bits=uint64_t(input.size())*8;
    for(int i=7;i>=0;--i) data.push_back(char(bits>>(i*8)));
    auto rotr=[](uint32_t x,int n){return (x>>n)|(x<<(32-n));};
    for(size_t off=0;off<data.size();off+=64) {
        uint32_t w[64];
        for(int i=0;i<16;++i) {w[i]=0;for(int j=0;j<4;++j) w[i]=(w[i]<<8)|uint8_t(data[off+i*4+j]);}
        for(int i=16;i<64;++i) {
            auto a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(rotr(a,7)^rotr(a,18)^(a>>3))+w[i-7]+(rotr(b,17)^rotr(b,19)^(b>>10));
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for(int i=0;i<64;++i) {
            uint32_t t1=v+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            v=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    }
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto x:h)out<<std::setw(8)<<x;return out.str();
}
inline std::string quote(const std::string& s) {
    std::ostringstream o;o<<'"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\')o<<'\\'<<c;
        else if(c<32)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
        else o<<c;
    }
    o<<'"';return o.str();
}
inline std::string json_dump(const g6v::Json& j) {
    switch(j.t) {
    case g6v::Json::Null:return "null";
    case g6v::Json::Bool:return j.b?"true":"false";
    case g6v::Json::Num:{std::ostringstream o;o<<std::setprecision(17)<<j.n;return o.str();}
    case g6v::Json::Str:return quote(j.s);
    case g6v::Json::Arr:{std::string s="[";for(const auto& v:j.a){if(s.size()>1)s+=",";s+=json_dump(v);}return s+"]";}
    case g6v::Json::Obj:{std::string s="{";for(const auto& v:j.o){if(s.size()>1)s+=",";s+=quote(v.first)+":"+json_dump(v.second);}return s+"}";}
    } throw std::runtime_error("JSON type");
}
struct Metadata {
    int n=0, prompt_tokens=0, chunk_position=0, layer=-1, index_source=-1, ratio=0;
    int ring_rows=0, key_rows=0, slot_count=1, request_slot=0, rank=0;
    int heads=16, wl=128, floor=0, ldtk=512;
    uint64_t ck_stride_bytes=0;
    std::string request_id, dispatch_phase, format, session_form, engine_revision;
    std::string origin="engine", knob_manifest="{}";
};
struct Record {
    Metadata metadata;
    std::vector<int32_t> topk, positions, sessions;
    bool is_captured=false; // engine indices only; operands below are unavailable
    std::string source_hash;
    sweep_topk::Metrics metrics;
};
inline std::string metadata_json(const Metadata& m) {
    std::ostringstream o;
    o<<"{\"schema\":1,\"kind\":\"attention_indices\",\"timing_data\":false";
#define SWEEP_INT(k) o<<",\"" #k "\":"<<m.k
    SWEEP_INT(n);SWEEP_INT(prompt_tokens);SWEEP_INT(chunk_position);SWEEP_INT(layer);SWEEP_INT(index_source);SWEEP_INT(ratio);
    SWEEP_INT(ring_rows);SWEEP_INT(key_rows);SWEEP_INT(slot_count);SWEEP_INT(request_slot);SWEEP_INT(rank);
    SWEEP_INT(heads);SWEEP_INT(wl);SWEEP_INT(floor);SWEEP_INT(ldtk);SWEEP_INT(ck_stride_bytes);
#undef SWEEP_INT
#define SWEEP_STR(k) o<<",\"" #k "\":"<<quote(m.k)
    SWEEP_STR(request_id);SWEEP_STR(dispatch_phase);SWEEP_STR(format);SWEEP_STR(session_form);SWEEP_STR(engine_revision);SWEEP_STR(origin);
#undef SWEEP_STR
    const auto knobs=json_dump(g6v::Json::parse(m.knob_manifest));
    o<<",\"knob_manifest\":"<<knobs<<",\"knob_manifest_sha256\":"<<quote(sha256(knobs));
    o<<",\"knob_manifest_source\":\"operator_supplied_not_runtime_verified\"";
    o<<",\"operands\":{\"q\":\"unavailable\",\"window_kv\":\"unavailable\",\"compressed_kv\":\"unavailable\",\"sink\":\"unavailable\"}}";
    return o.str();
}
inline Metadata parse_metadata(const std::string& text) {
    auto j=g6v::Json::parse(text);Metadata m;
    require(j.at("schema").i64()==1 && j.at("kind").str()=="attention_indices","unknown capture schema");
    require(j.at("timing_data").t==g6v::Json::Bool && !j.at("timing_data").b,"capture is not timing data");
#define SWEEP_INT(k) {auto v=j.at(#k).i64();require(v>=INT32_MIN&&v<=INT32_MAX,"metadata integer overflow");m.k=int(v);}
    SWEEP_INT(n);SWEEP_INT(prompt_tokens);SWEEP_INT(chunk_position);SWEEP_INT(layer);SWEEP_INT(index_source);SWEEP_INT(ratio);
    SWEEP_INT(ring_rows);SWEEP_INT(key_rows);SWEEP_INT(slot_count);SWEEP_INT(request_slot);SWEEP_INT(rank);
    SWEEP_INT(heads);SWEEP_INT(wl);SWEEP_INT(floor);SWEEP_INT(ldtk);
#undef SWEEP_INT
    auto stride=j.at("ck_stride_bytes").i64();require(stride>=0,"negative stride");m.ck_stride_bytes=uint64_t(stride);
#define SWEEP_STR(k) m.k=j.at(#k).str()
    SWEEP_STR(request_id);SWEEP_STR(dispatch_phase);SWEEP_STR(format);SWEEP_STR(session_form);SWEEP_STR(engine_revision);SWEEP_STR(origin);
#undef SWEEP_STR
    require(j.at("knob_manifest").t==g6v::Json::Obj && j.at("knob_manifest").size()>0,"missing knob manifest");
    m.knob_manifest=json_dump(j.at("knob_manifest"));
    require(j.at("knob_manifest_sha256").str()==sha256(m.knob_manifest),"knob manifest SHA256 mismatch");
    require(j.at("knob_manifest_source").str()=="operator_supplied_not_runtime_verified","unsupported knob manifest source");
    for(const char* key:{"q","window_kv","compressed_kv","sink"})
        require(j.at("operands").at(key).str()=="unavailable","v1 does not carry operands");
    return m;
}
inline void validate(Record& r) {
    const auto& m=r.metadata;
    require(m.n>0&&m.n<=8192&&m.prompt_tokens>0&&m.prompt_tokens<=1048576,"capture shape out of bounds");
    require(m.ratio>=0&&m.ratio<=2&&m.layer>=0&&m.layer<40&&m.index_source>=-1&&m.index_source<40,"capture layer/ratio invalid");
    require(m.ring_rows>0&&m.ring_rows<=1048704&&m.key_rows>=0&&m.key_rows<=1048576,"capture capacity invalid");
    require(m.slot_count>0&&m.slot_count<=65536&&m.request_slot>=0&&m.request_slot<65536,"capture slot geometry invalid");
    require(m.rank>=0&&m.rank<8&&m.heads>0&&m.heads<=128&&m.wl>0&&m.wl<=128&&m.ring_rows>=m.wl&&m.ldtk>=512&&m.ldtk<=1048576,"capture launch shape invalid");
    require(m.chunk_position>=0&&m.chunk_position<m.prompt_tokens&&m.floor>=0&&m.floor<m.prompt_tokens,"capture position/floor invalid");
    require(m.session_form=="implicit_single"||m.session_form=="per_row_slots","capture session form invalid");
    require(m.dispatch_phase=="prefill_v2"||m.dispatch_phase=="prefill_legacy"||m.dispatch_phase=="late_replay"||m.dispatch_phase=="main","capture phase invalid");
    require(m.format=="fp4_e2m1_e4m3_288"||m.format=="bf16_512"||m.format=="window_only","capture format invalid");
    require((m.ratio==0)==(m.format=="window_only"),"capture format/ratio mismatch");
    require(m.origin=="engine"||m.origin=="synthetic_test","capture origin invalid");
    require(!m.request_id.empty()&&!m.engine_revision.empty(),"capture provenance missing");
    if(m.ratio) require(m.key_rows>0 && m.index_source>=0 && m.ck_stride_bytes==uint64_t(m.key_rows)*(m.format=="bf16_512"?1024:288),"capture compressed stride invalid");
    require(r.positions.size()==size_t(m.n)&&r.sessions.size()==size_t(m.n)&&r.topk.size()==size_t(m.n)*512,"capture tensor dimensions invalid");
    for(int row=0;row<m.n;++row) {
        auto pos=r.positions[row],slot=r.sessions[row];
        require(pos>=0&&pos<m.prompt_tokens&&slot>=0&&slot<m.slot_count,"capture position/session out of bounds");
        if(m.session_form=="implicit_single")require(slot==0,"implicit slot must be zero (base already shifted)");
        int visible=m.ratio?std::min(m.key_rows,(pos+1)/m.ratio):0, count=0,prev=-1;
        bool tail=false;
        for(int col=0;col<512;++col) {
            int value=r.topk[size_t(row)*512+col];
            if(value==-1){tail=true;continue;}
            require(!tail&&value>=0&&value<visible&&value>prev,"capture top-k not unique/sorted/visible or padding malformed");
            prev=value;++count;
        }
        require(count==std::min(512,visible),"capture top-k active count mismatch");
    }
    r.is_captured=m.origin=="engine";
    r.metrics=sweep_topk::metrics(r.topk,r.sessions);
}
inline void put32(std::string& s,uint32_t x){for(int i=0;i<4;++i)s.push_back(char(x>>(8*i)));}
inline uint32_t get32(const std::string& s,size_t off){require(off+4<=s.size(),"short capture");uint32_t x=0;for(int i=0;i<4;++i)x|=uint32_t(uint8_t(s[off+i]))<<(8*i);return x;}
constexpr size_t max_metadata=1024*1024, max_payload=8192*514*4;
// Magic (8), metadata length LE32, payload length LE32, SHA256 ASCII (64), JSON,
// then topk[n][512], positions[n], sessions[n], all signed LE32, in original order.
inline std::string encode(Record r) {
    auto meta=metadata_json(r.metadata);r.metadata=parse_metadata(meta);validate(r);
    std::string payload;payload.reserve(r.topk.size()*4+r.positions.size()*8);
    for(const auto* v:{&r.topk,&r.positions,&r.sessions})for(auto x:*v)put32(payload,uint32_t(x));
    require(meta.size()<=max_metadata&&payload.size()<=max_payload,"capture too large");
    std::string file="ASWEEP1\n";put32(file,uint32_t(meta.size()));put32(file,uint32_t(payload.size()));
    return file+sha256(meta+payload)+meta+payload;
}
inline Record decode(const std::string& file) {
    require(file.size()>=80&&file.compare(0,8,"ASWEEP1\n")==0,"invalid capture magic");
    size_t ml=get32(file,8),pl=get32(file,12);
    require(ml<=max_metadata&&pl<=max_payload&&file.size()==80+ml+pl,"capture length invalid");
    auto body=file.substr(80);require(sha256(body)==file.substr(16,64),"capture SHA256 mismatch");
    Record r;r.metadata=parse_metadata(body.substr(0,ml));
    require(r.metadata.n>0&&r.metadata.n<=8192&&pl==size_t(r.metadata.n)*514*4,"capture payload dimensions invalid");
    size_t offset=80+ml;
    for(auto* v:{&r.topk,&r.positions,&r.sessions}) {
        size_t count=v==&r.topk?size_t(r.metadata.n)*512:size_t(r.metadata.n);
        v->reserve(count);
        for(size_t i=0;i<count;++i,offset+=4){uint32_t x=get32(file,offset);v->push_back(x==UINT32_MAX?-1:int32_t(x<=INT32_MAX?x:throw std::runtime_error("invalid negative index")));}
    }
    r.source_hash=sha256(file);validate(r);return r;
}
inline Record load(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot open capture");
    auto size=f.tellg();require(size>=0&&uint64_t(size)<=80+max_metadata+max_payload,"capture file too large");
    std::string data(size_t(size),'\0');f.seekg(0);f.read(data.data(),data.size());require(bool(f),"short capture read");return decode(data);
}
} // namespace sweep_capture
