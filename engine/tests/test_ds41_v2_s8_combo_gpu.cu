// Synthetic multi-epoch State API integration. The transport below is test-only:
// it moves mapped slots on a CPU thread, not RDMA. No speed claim.
#include "ds41_v2_test_support.cuh"
#include "../attn/ds41_attn_v2_api.h"
#include "../kernels/ds41_v2_moe.cuh"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
using namespace v2test;
extern "C" void ds41_attn_set_tc(int);
namespace g6 {
struct Fabric::Impl {
    Arena memory;
    struct Job { unsigned long long epoch; std::vector<uint32_t> sizes; Rs rs; };
    std::deque<Job> jobs;
};
}
namespace transport {
std::mutex mutex;
std::condition_variable changed;
g6::Fabric* ranks[4]{};
g6::Fabric::Impl* stores[4]{};
bool stopped=false;
std::thread worker;
void start() {
    stopped=false;
    worker=std::thread([] {
        std::unique_lock<std::mutex> lock(mutex);
        while(!stopped) {
            bool progress=false;
            for(int r=0;r<4;++r)if(stores[r] && !stores[r]->jobs.empty()) {
                auto& job=stores[r]->jobs.front(); const int c=job.rs.order.at(0);
                auto f=ranks[r]->dev(); auto seq=g6::coll_seq(job.epoch,c,f.ncoll);
                auto seen=__atomic_load_n(f.doorbell+c,__ATOMIC_ACQUIRE);
                if(seen<seq)continue;
                insist(seen==seq,"proxy did not skip an epoch");
                const char* source=f.send+g6::fab_send_off(f,int(job.epoch&1),c);
                for(int peer=0;peer<4;++peer)if(peer!=r) {
                    auto target=ranks[peer]->dev(); bool scatter=job.rs.mask.bit(c);
                    size_t n=scatter?job.rs.bytes.at(peer):job.sizes.at(c);
                    char* dest=target.recv+g6::fab_recv_off(target,int(job.epoch&1),c,r);
                    memcpy(dest,source+(scatter?size_t(peer)*job.rs.ss:0),n);
                    __atomic_store_n(reinterpret_cast<unsigned long long*>(dest+n),seq,__ATOMIC_RELEASE);
                }
                stores[r]->jobs.pop_front(); progress=true; changed.notify_all();
            }
            if(!progress)changed.wait_for(lock,std::chrono::microseconds(50));
        }
    });
}
void stop() {
    { std::lock_guard<std::mutex> lock(mutex); stopped=true; }
    changed.notify_all(); worker.join();
}
}
namespace g6 {
void Fabric::init(const FabricConfig& config) {
    cfg_=config; insist(!impl_,"test fabric initialized once"); impl_=new Impl;
    insist(fab_slot_layout(cfg_,cfg_.ncoll,&dev_),"mapped fabric layout");
    dev_.rank=cfg_.rank; dev_.nr=4; dev_.compact=1;
    auto& a=impl_->memory;
    size_t send_bytes=2*fab_slot_base(dev_,dev_.ncoll),recv_bytes=fab_recv_bytes(dev_);
    dev_.send=a.make<char>(send_bytes,true); dev_.recv=a.make<char>(recv_bytes,true);
    dev_.doorbell=a.make<unsigned long long>(dev_.ncoll,true); dev_.error=a.make<unsigned>(1,true);
    memset(dev_.send,0,send_bytes); memset(dev_.recv,0,recv_bytes); memset(dev_.doorbell,0,dev_.ncoll*8); *dev_.error=0;
    if(cfg_.xtrace) { dev_.xt=a.make<unsigned long long>(size_t(XT_STEPS)*dev_.ncoll*XT_W); check(cudaMemset(dev_.xt,0,size_t(XT_STEPS)*dev_.ncoll*XT_W*8)); }
    transport::ranks[cfg_.rank]=this; transport::stores[cfg_.rank]=impl_;
}
Fabric::~Fabric() {
    if(impl_) { insist(impl_->jobs.empty(),"fabric destructor drained"); transport::stores[cfg_.rank]=nullptr; transport::ranks[cfg_.rank]=nullptr; delete impl_; }
}
void Fabric::enqueue_step(unsigned long long epoch,const std::vector<uint32_t>& sizes,const Rs& rs) {
    insist(rs.order.size()==1,"one public API job per collective");
    { std::lock_guard<std::mutex> lock(transport::mutex); impl_->jobs.push_back({epoch,sizes,rs}); }
    transport::changed.notify_all();
}
bool Fabric::wait_proxy_idle(double seconds) {
    std::unique_lock<std::mutex> lock(transport::mutex);
    return transport::changed.wait_for(lock,std::chrono::duration<double>(seconds),[&]{return impl_->jobs.empty();});
}
}
namespace {
constexpr int H=128, C=16;
constexpr size_t tw[6]={64,96,16,16,64,128};
struct Rank {
    State state;
    ds41a::Scratch attention{};
    uint8_t *packed,*remote,*partial,*own_out,*record,*ckv_source,*ckv_all;
    Segment ffn[4],tail[6],input;
    __nv_bfloat16 *rs_source[4],*rs_own[4];
    uint8_t* compact[6];
    uint32_t* xf;
};
using Result=std::vector<Bytes>;
void trace_check(const State& s,uint64_t epoch,const std::vector<int>& order) {
    for(int c:order) {
        auto raw=download(s.wire.xt+((epoch%g6::XT_STEPS)*s.wire.ncoll+c)*g6::XT_W,g6::XT_W*8);
        unsigned long long stamp[g6::XT_W]; memcpy(stamp,raw.data(),sizeof(stamp));
        insist(stamp[g6::XT_SEQ]==g6::coll_seq(epoch,c,s.wire.ncoll),"trace epoch and collective");
        insist(stamp[g6::XT_PRODUCED]>0 && stamp[g6::XT_PRODUCED]<=stamp[g6::XT_PUB] && stamp[g6::XT_PUB]<=stamp[g6::XT_DB],"trace produced/published/doorbell");
        insist(stamp[g6::XT_WAIT_BEGIN]>0 && stamp[g6::XT_RED]>=stamp[g6::XT_WAIT_BEGIN],"trace wait/consume");
        for(int p=0;p<8;++p) {
            if(p==s.g.rank || p>=4)insist(stamp[g6::XT_TAIL+p]==0,"trace unused peers cleared");
            else insist(stamp[g6::XT_TAIL+p]>=stamp[g6::XT_WAIT_BEGIN] && stamp[g6::XT_TAIL+p]<=stamp[g6::XT_RED],"trace peer consumed before red");
        }
        if(c==int(Coll::AttnOut) || c>=int(Coll::Ffn0))
            insist(stamp[g6::XT_JOIN_BEGIN]>0 && stamp[g6::XT_JOIN_END]>=stamp[g6::XT_JOIN_BEGIN] && stamp[g6::XT_JOIN_END]>=stamp[g6::XT_RED],"real cross-stream join event");
    }
}
// Test-only compact packing: the production heads wrapper requires K >= 1024.
__global__ void pack_compact(const uint8_t* oaq,const uint8_t* oas,uint8_t* packed,int M,int P,int K) {
    const size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x, row=K+K/32;
    if(idx>=size_t(P)*row)return;
    const size_t r=idx/row,c=idx%row;
    packed[idx]=(r<size_t(M)) ? (c<size_t(K) ? oaq[r*size_t(K)+c] : oas[r*size_t(K/32)+(c-size_t(K))]) : uint8_t(0);
}
// Compact wide-row projection uses the same ordered FP32 rank sum.
template<int N>
__global__ void sum_compact(const __nv_bfloat16* in,__nv_bfloat16* out,int rows,int q) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*N) { float x=0; for(int r=0;r<4;++r)x+=__bfloat162float(in[size_t(r)*q*N+i]); out[i]=__float2bfloat16_rn(x); }
}
template<int K=1024,int N=5120,int D=512>
Result exercise(int M,bool skip,bool trace,int mode,bool flip=false) {
    fprintf(stderr,"RUN M=%d skip=%d trace=%d mode=%d flip=%d\n",M,skip,trace,mode,flip);
    Arena memory; Inputs immutable; g6::Fabric fabric[4]; Rank rank[4];
    int Q=(M+3)/4,P=4*Q; size_t row=K+K/32, part=size_t(Q)*N*2;
    auto wq=memory.make<uint8_t>(size_t(N)*K),ws=memory.make<uint8_t>(size_t(N)*K/32);
    fp8(wq,size_t(N)*K,31); fp8(ws,size_t(N)*K/32,32,true);
    immutable.add(wq,size_t(N)*K); immutable.add(ws,size_t(N)*K/32);
    ds41a::Fp8Mat weights{wq,ws,N,K};
    const size_t fw[]={H,H/32,24,24};
    size_t ffn_payload=0; for(auto width:fw)ffn_payload+=aligned(size_t(Q)*width,16);
    for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state;
        g6::FabricConfig config{}; config.rank=r; config.ips.resize(4); config.ncoll=int(Coll::Count); config.compact=true;
        config.run1=int(Coll::AttnOut); config.run2=int(Coll::Ffn0);
        config.slot_bytes=std::max({size_t(65536-8),size_t(Q)*D*2,ffn_payload}); config.slot_bytes1=4*aligned(size_t(Q)*row+8,256)-8;
        config.slot_bytes2=4*aligned(size_t(Q)*C*2+8,256)-8; config.xtrace=trace;
        fabric[r].init(config); check(cudaStreamCreateWithFlags(&s.st,cudaStreamNonBlocking)); init(s,fabric[r],s.st);
        s.l20_tail.on=true; s.send_ffn_opt.mode=mode;
        b.attention.oaq=memory.make<uint8_t>(size_t(M)*K); b.attention.oas=memory.make<uint8_t>(size_t(M)*K/32);
        fp8(b.attention.oaq,size_t(M)*K,r+41); fp8(b.attention.oas,size_t(M)*K/32,r+51,true);
        immutable.add(b.attention.oaq,size_t(M)*K); immutable.add(b.attention.oas,size_t(M)*K/32);
        b.packed=memory.make<uint8_t>(size_t(P)*row); b.remote=memory.make<uint8_t>(size_t(P)*row);
        b.partial=memory.make<uint8_t>(4*part); b.own_out=memory.make<uint8_t>(part); b.xf=memory.make<uint32_t>(K*4);
        b.record=memory.make<uint8_t>(size_t(P)*H);
        for(int k=0;k<4;++k) {
            auto source=memory.make<uint8_t>(size_t(Q)*fw[k]); auto bytes=data(size_t(Q)*fw[k],r*13+k+61);
            if(flip && r==0 && k==0)bytes[0]^=1;
            upload(source,bytes); immutable.add(source,bytes.size());
            b.ffn[k]={source,memory.make<uint8_t>(size_t(P)*fw[k]),fw[k]};
            b.rs_source[k]=memory.make<__nv_bfloat16>(size_t(M)*C); b.rs_own[k]=memory.make<__nv_bfloat16>(size_t(Q)*C);
            std::vector<__nv_bfloat16> values(size_t(M)*C,__float2bfloat16_rn(float(r+k+1)));
            check(cudaMemcpy(b.rs_source[k],values.data(),values.size()*2,cudaMemcpyHostToDevice)); immutable.add(b.rs_source[k],values.size()*2);
        }
        auto input=memory.make<uint8_t>(size_t(Q)*32); upload(input,data(size_t(Q)*32,r+81)); immutable.add(input,size_t(Q)*32);
        b.input={input,memory.make<uint8_t>(size_t(P)*32),32};
        b.ckv_source=memory.make<uint8_t>(size_t(Q)*D*2); b.ckv_all=memory.make<uint8_t>(size_t(P)*D*2);
        for(int k=0;k<6;++k) {
            auto source=memory.make<uint8_t>(128*tw[k]); Bytes bytes(128*tw[k],0);
            auto slice=tail_slice(geometry(M,4,r)); auto full=data(size_t(M)*tw[k],k+91);
            if(slice.count)memcpy(bytes.data(),full.data()+size_t(std::max(0,M-128)+slice.destination)*tw[k],size_t(slice.count)*tw[k]);
            upload(source,bytes); immutable.add(source,bytes.size());
            b.tail[k]={source,memory.make<uint8_t>(512*tw[k]),tw[k]}; b.compact[k]=memory.make<uint8_t>(size_t(std::min(M,128))*tw[k]);
        }
    }
    fprintf(stderr,"stage initialized\n");
    transport::start();
    for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state; begin_layer(s,geometry(M,4,r),false,false,false);
        insist(s.epoch==1,"Range A first epoch"); all_gather(s,Coll::AttnIn,&b.input,1,s.st);
    }
    for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state; size_t offset=size_t(s.g.row0);
        if(s.send_ffn_opt.attn_opt()) {
            attn_out_a2a_opt(s,b.attention.oaq,b.attention.oas,K,b.remote);
            ds41_v2_output_b_split(&weights,b.attention.oaq+offset*K,b.attention.oas+offset*(K/32),b.partial+size_t(r)*part,s.g.owned,b.xf,s.st);
        } else {
            if constexpr(K>=1024) ds41_v2_output_pack(&b.attention,b.packed,M,P,K/128,s.st);
            else pack_compact<<<(size_t(P)*row+255)/256,256,0,s.st>>>(b.attention.oaq,b.attention.oas,b.packed,M,P,K);
            fork(s,Coll::AttnOut); Segment seg{b.packed,b.remote,row}; all_to_all(s,Coll::AttnOut,&seg,1,s.cs); record_received(s,Coll::AttnOut);
            ds41_v2_output_b(&weights,b.packed+offset*row,b.partial+size_t(r)*part,s.g.owned,b.xf,s.st);
        }
        join(s,Coll::AttnOut);
        for(int p=0;p<4;++p)if(p!=r)ds41_v2_output_b(&weights,b.remote+size_t(p)*Q*row,b.partial+size_t(p)*part,s.g.owned,b.xf,s.st);
        if constexpr(N==5120) ds41_v2_output_sum(b.partial,b.own_out,s.g.owned,Q,4,s.st);
        else if(s.g.owned) sum_compact<N><<<(size_t(s.g.owned)*N+255)/256,256,0,s.st>>>(reinterpret_cast<const __nv_bfloat16*>(b.partial),reinterpret_cast<__nv_bfloat16*>(b.own_out),s.g.owned,Q);
    }
    for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state;
        if(s.send_ffn_opt.ffn_opt())all_gather_ffn_in(s,b.ffn,4,b.record,H,s.st);
        else {
            all_gather(s,Coll::FfnIn,b.ffn,4,s.st);
            ds41v2moe::k_relayout<true><<<(size_t(M)*H+255)/256,256,0,s.st>>>(static_cast<const uint8_t*>(b.ffn[0].dst),b.record,M,H);
        }
    }
    for(int k=0;k<4;++k)for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state; fork(s,ffn_group(k));
        reduce_scatter(s,k,b.rs_source[k],b.rs_own[k],C,s.cs); record_received(s,ffn_group(k)); join(s,ffn_group(k));
    }
    for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state; end_layer(s);
        // No State reset or replacement: transition immediately on the same streams,
        // events and Fabric, after Range-A output production.
        begin_l20_tail(s,geometry(M,4,r),!skip); insist(s.epoch==2,"hoisted epoch follows Range A");
        if(s.g.owned)check(cudaMemcpy2DAsync(b.ckv_source,D*2,b.own_out,N*2,D*2,s.g.owned,cudaMemcpyDeviceToDevice,s.st));
        Segment seg{b.ckv_source,b.ckv_all,D*2}; all_gather(s,Coll::AttnIn,&seg,1,s.st);
    }
    if(!skip)for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state; auto saved=s.g; s.g=geometry(512,4,r);
        all_gather(s,Coll::Tail,b.tail,6,s.st); s.g=saved;
        for(int p=0;p<4;++p) {
            auto slice=tail_slice(geometry(M,4,p));
            if(slice.count)for(int k=0;k<6;++k)check(cudaMemcpyAsync(b.compact[k]+size_t(slice.destination)*tw[k],static_cast<uint8_t*>(b.tail[k].dst)+size_t(p)*128*tw[k],size_t(slice.count)*tw[k],cudaMemcpyDeviceToDevice,s.st));
        }
    }
    fprintf(stderr,"stage all issued\n");
    Result result;
    for(int r=0;r<4;++r) {
        auto& b=rank[r]; auto& s=b.state; end_layer(s); check(cudaStreamSynchronize(s.st)); check(cudaStreamSynchronize(s.cs));
        insist(fabric[r].wait_proxy_idle(30),"CPU proxy drain"); if(*s.wire.error)fprintf(stderr,"rank=%d M=%d mode=%d trace=%d error=0x%x\n",r,M,mode,trace,*s.wire.error); insist(*s.wire.error==0,"zero timeout/error word");
        if(trace) { trace_check(s,1,layer_order(false,false,false)); trace_check(s,2,l20_tail_order(!skip)); }
        result.push_back(download(b.own_out,part));
        result.push_back(download(b.input.dst,size_t(P)*32));
        for(int k=0;k<4;++k)result.push_back(download(b.ffn[k].dst,size_t(P)*fw[k]));
        result.push_back(download(b.record,size_t(P)*H));
        for(int k=0;k<4;++k) {
            result.push_back(download(b.rs_own[k],size_t(Q)*C*2));
            auto expected=__float2bfloat16_rn(float(10+4*k));
            for(int i=0;i<s.g.owned*C;++i)insist(memcmp(result.back().data()+i*2,&expected,2)==0,"RS CPU oracle");
        }
        result.push_back(download(b.ckv_all,size_t(P)*D*2));
        if(!skip)for(int k=0;k<6;++k) {
            result.push_back(download(b.compact[k],size_t(std::min(M,128))*tw[k]));
            auto full=data(size_t(M)*tw[k],k+91);
            insist(result.back()==Bytes(full.begin()+size_t(std::max(0,M-128))*tw[k],full.end()),"tail_slice compact handover CPU oracle");
            result.push_back(download(b.tail[k].dst,512*tw[k]));
        }
    }
    // Hoisted ckv1 equals concatenated own-row BF16 projections, not poison.
    Bytes ckv(size_t(M)*D*2);
    for(int r=0;r<4;++r) {
        auto g=geometry(M,4,r); auto source=download(rank[r].own_out,part);
        for(int i=0;i<g.owned;++i)memcpy(ckv.data()+size_t(g.row0+i)*D*2,source.data()+size_t(i)*N*2,D*2);
    }
    for(int r=0;r<4;++r)equal(rank[r].ckv_all,ckv,"hoisted BF16 gather CPU oracle");
    immutable.verify();
    fprintf(stderr,"stage verified\n");
    for(auto& b:rank) { release(b.state); check(cudaStreamDestroy(b.state.st)); }
    transport::stop();
    return result;
}
}
int main() {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);
    setenv("CUDA_MODULE_LOADING","EAGER",1);
    setenv("GR8_DS_PF_V2","0",1); setenv("GR8_DS_PF_V2_TIMING","0",1);
    // Resolve every optional traced kernel before any peer wait can be resident.
    // Loading a first-use CUDA kernel may synchronize a context under lazy loading.
    auto load=[](auto kernel) { cudaFuncAttributes attr{}; check(cudaFuncGetAttributes(&attr,kernel)); };
    load(detail::trace_produced); load(detail::trace_stamp);
    load(detail::publish<true>); load(detail::wait<true>);
    load(detail::unpack<true>); load(detail::unpack_remote<true>);
    load(detail::unpack_ffn_in<true>); load(detail::reduce<true>); load(detail::reduce_vector<true>);
    ds41_attn_set_tc(1);
    for(int M:{1,5,127,128,129,257,511})for(bool skip:{false,true})for(bool trace:{false,true}) {
        Result reference=exercise(M,skip,trace,0);
        for(int mode:{1,2,3}) {
            auto got=exercise(M,skip,trace,mode); insist(got.size()==reference.size(),"same output set");
            for(size_t i=0;i<got.size();++i)insist(got[i].size()==reference[i].size() && memcmp(got[i].data(),reference[i].data(),got[i].size())==0,"all ranks/all arrays bit-exact mode 0 vs candidate");
        }
        auto changed=exercise(M,skip,trace,3,true); size_t differences=0;
        for(size_t i=0;i<changed.size();++i)for(size_t j=0;j<changed[i].size();++j)differences+=changed[i][j]!=reference[i][j];
        insist(differences>0,"input-byte flip negative control");
        printf("PASS combo M=%d skip_late=%d trace=%d modes=0,1,2,3 memcmp=0 negative_byte_diffs=%zu\n",M,skip,trace,differences);
    }
    ds41_attn_set_tc(0);
    for(int M:{8257,12288})for(bool skip:{false,true})for(bool trace:{false,true}) {
        auto reference=exercise<256,256,64>(M,skip,trace,0);
        for(int mode:{1,2,3})insist(exercise<256,256,64>(M,skip,trace,mode)==reference,"wide all arrays exact");
        insist(exercise<256,256,64>(M,skip,trace,3,true)!=reference,"wide input flip rejected");
        printf("PASS wide combo M=%d K=256 N=256 D=64 skip_late=%d trace=%d modes=0,1,2,3 memcmp=0 negative_control=rejected\n",M,skip,trace);
    }
    summary("s8_combo");
}
