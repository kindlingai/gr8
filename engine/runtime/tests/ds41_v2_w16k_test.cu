// Bounded synthetic API regression; no checkpoint, network, or speed claim.
#include "../../tests/ds41_v2_test_support.cuh"
#include "../ds41_v2_ffn.cuh"
#include "../../engram/engram_dev.cuh"
#include "../ds41_v2_engram.cuh"
#include <cmath>
#include <memory>
using namespace v2test;
namespace f=ds41::v2::ffn;
namespace eg=ds41::engram;
namespace ew=ds41::v2::engram_wire;
using BF=__nv_bfloat16;
namespace {
void sync() { check(cudaGetLastError()); check(cudaDeviceSynchronize()); }
template<class T> void put(T* dst,const std::vector<T>& src) {
    check(cudaMemcpy(dst,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice));
}
void prepare_contract() {
    f::Scratch q;
    insist(f::prepare(q,8193,false)==cudaErrorInvalidValue && q.allocations.empty(),"8193 opt-in required");
    insist(f::prepare(q,16385,true)==cudaErrorInvalidValue && q.allocations.empty(),"16385 rejected");
    // Account for each internal allocation BEFORE invoking production prepare.
    const size_t m=64,own=16,r=8*dsr::pw_tile_bound(64*f::TK,f::NE);
    const bool resident=knob::ds_pf_v2() && knob::ds_pf_v2_fc2();
    std::vector<size_t> sizes={sizeof(f::Plan),f::NE*(f::MM/32)*sizeof(unsigned),
        f::NE*sizeof(int),own*f::NE*sizeof(float),own*f::H,own*(f::H/32),
        own*f::TK*sizeof(int),own*f::TK*sizeof(float),m*f::H,m*(f::H/32),
        m*f::TK*sizeof(int),m*f::TK*sizeof(float),m*f::H,r*f::I,r*mxtc::ASP,r*f::H*sizeof(__half),
        m*2*f::I*sizeof(BF),m*f::KP,m*(f::KP/32),m*f::COLS*sizeof(BF)};
    for(int c=0;c<4;++c) { sizes.push_back(m*f::COLS*sizeof(BF)); sizes.push_back(own*f::COLS*sizeof(BF)); }
    if(resident)sizes.push_back(f::NE*sizeof(int));
    size_t bytes=0; for(auto n:sizes)bytes+=n;
    insist(live+bytes<budget,"prepare internal allocation budget"); live+=bytes; peak=std::max(peak,live);
    check(f::prepare(q,64,false));
    insist(q.allocations.size()==sizes.size() && q.plan && !q.plan16k && q.capacity==64,"small prepare default planner");
    f::release(q); live-=bytes; insist(q.allocations.empty() && !q.capacity,"release reset");
    printf("PASS prepare guards/small64/release internal_bytes=%zu\n",bytes);
}
void route_api(int M) {
    Arena a; Inputs input; f::Scratch q; f::Layer w;
    auto g=geometry(M,4,3); q.capacity=g.padded(); q.own_capacity=g.Q;
    q.plan16k=a.make<f::PlanW16K>(1); q.masks=a.make<unsigned>(f::NE*(f::MM_W16K/32));
    q.logits=a.make<float>(g.Q*f::NE); q.own_q=a.make<uint8_t>(size_t(g.Q)*f::H);
    q.own_s=a.make<uint8_t>(size_t(g.Q)*(f::H/32)); q.own_idx=a.make<int>(g.Q*f::TK); q.own_wt=a.make<float>(g.Q*f::TK);
    auto gate=a.make<BF>(size_t(f::NE)*f::H),y=a.make<BF>(size_t(g.Q)*f::H); bf16(gate,size_t(f::NE)*f::H,17); bf16(y,size_t(g.Q)*f::H,29);
    auto bias=a.make<float>(f::NE); std::vector<float> b(f::NE); for(int i=0;i<f::NE;++i)b[i]=float(i%17)/32; put(bias,b);
    // supported() requires all weights, but router_own reads only gate/bias.
    auto sentinel=a.make<uint8_t>(1); w={gate,bias,nullptr,sentinel,sentinel,sentinel,sentinel,sentinel,sentinel,10.f};
    input.add(gate,size_t(f::NE)*f::H*2); input.add(y,size_t(g.Q)*f::H*2); input.add(bias,f::NE*4);
    auto ref=a.make<f::Plan>(1); auto mask=a.make<unsigned>(f::NE*(f::MM/32));
    auto iq=a.make<uint8_t>(size_t(g.Q)*f::H),is=a.make<uint8_t>(size_t(g.Q)*(f::H/32));
    auto ix=a.make<int>(g.Q*f::TK); auto wt=a.make<float>(g.Q*f::TK);
    for(int cut:{3,6}) {
        check(cudaMemset(q.own_q,0xd3,size_t(g.Q)*f::H)); check(cudaMemset(q.own_s,0xd3,size_t(g.Q)*(f::H/32)));
        check(cudaMemset(iq,0xd3,size_t(g.Q)*f::H)); check(cudaMemset(is,0xd3,size_t(g.Q)*(f::H/32)));
        check(f::router_own(g,w,q,y,nullptr,cut,nullptr));
        // The reference uses the old-width planner on the <=4096 owned rows.
        check(cudaMemset(mask,0,f::NE*(f::MM/32)*sizeof(unsigned)));
        if(cut==3)dsr::k_route_ro<f::NE,f::TK,f::MM,3><<<(g.owned+7)/8,256>>>(q.logits,bias,g.owned,ref,mask);
        else dsr::k_route_ro<f::NE,f::TK,f::MM,6><<<(g.owned+7)/8,256>>>(q.logits,bias,g.owned,ref,mask);
        ds41v2moe::k_export_routes<f::TK,f::MM><<<(g.Q*f::TK+255)/256,256>>>(ref,g.owned,g.Q,ix,wt);
        insist(ds41_sh_act_quant(y,f::H,iq,f::H,is,f::H/32,g.owned,f::H,nullptr)==0,"reference activation quantization"); sync();
        equal(q.own_idx,download(ix,g.Q*f::TK*4),"router indices old vs wide planner");
        equal(q.own_wt,download(wt,g.Q*f::TK*4),"router weights old vs wide planner");
        equal(q.own_q,download(iq,size_t(g.Q)*f::H),"router quantized codes including padding");
        equal(q.own_s,download(is,size_t(g.Q)*(f::H/32)),"router scales including padding");
        auto expected=download(ix,g.Q*f::TK*4); expected[0]^=1;
        insist(download(q.own_idx,expected.size())!=expected,"router negative control rejected");
        printf("PASS router_own M=%d rank=3 cut=%d actual_API old_plan_reference memcmp=0\n",M,cut);
    }
    input.verify();
}
void packed_weight(uint8_t* p,size_t n,unsigned seed) {
    auto b=data(n,seed);
    for(size_t i=0;i<n;++i)b[i]=i%mx4::GR<512?uint8_t((b[i]&0x33)|((b[i]>>2)&0x88)):uint8_t(119+b[i]%3);
    upload(p,b);
}
int record_offset(int k) {
    return (k & ~63) | ((k & 6)<<3) | ((k & 32)>>2) | ((k & 16)>>2) | ((k & 8)>>2) | (k & 1);
}
void wide_ffn(int M) {
    Arena a; Inputs input; f::Scratch q; f::Layer w;
    const auto g=geometry(M,4,0); q.capacity=g.padded(); q.own_capacity=g.Q;
    q.plan16k=a.make<f::PlanW16K>(1); q.masks=a.make<unsigned>(f::NE*(f::MM_W16K/32)); q.tile0=a.make<int>(f::NE);
    q.all_q=a.make<uint8_t>(size_t(M)*f::H); q.record_q=a.make<uint8_t>(size_t(M)*f::H); q.all_s=a.make<uint8_t>(size_t(M)*(f::H/32));
    q.all_idx=a.make<int>(size_t(M)*f::TK); q.all_wt=a.make<float>(size_t(M)*f::TK);
    // One expert, at most three live tokens -> exactly one 8-token tile.
    q.aq=a.make<uint8_t>(8*f::I); q.as=a.make<uint8_t>(8*mxtc::ASP); q.partial=a.make<__half>(8*f::H);
    q.sh13=a.make<BF>(size_t(M)*2*f::I); q.sh_q=a.make<uint8_t>(size_t(M)*f::KP); q.sh_s=a.make<uint8_t>(size_t(M)*(f::KP/32));
    q.shared=a.make<BF>(size_t(M)*f::COLS); auto out=a.make<BF>(size_t(g.padded())*f::COLS);
    for(int c=0;c<4;++c)q.send[c]=out; // API calls are drained before buffer reuse.
    auto fc1=a.make<uint8_t>(size_t(f::I/16)*(f::H/64)*2*mx4::GR);
    auto fc2=a.make<uint8_t>(size_t(f::H/16)*(f::I/64)*mx4::GR);
    packed_weight(fc1,size_t(f::I/16)*(f::H/64)*2*mx4::GR,37); packed_weight(fc2,size_t(f::H/16)*(f::I/64)*mx4::GR,41);
    auto w13=a.make<uint8_t>(size_t(2*f::I)*f::H),s13=a.make<uint8_t>(size_t(2*f::I)*(f::H/32));
    auto w2=a.make<uint8_t>(size_t(f::H)*f::KP),s2=a.make<uint8_t>(size_t(f::H)*(f::KP/32));
    fp8(w13,size_t(2*f::I)*f::H,43); fp8(s13,size_t(2*f::I)*(f::H/32),47,true);
    fp8(w2,size_t(f::H)*f::KP,53); fp8(s2,size_t(f::H)*(f::KP/32),59,true);
    w.fc1=fc1; w.fc2=fc2; w.w13=w13; w.sw13=s13; w.w2=w2; w.sw2=s2;
    fp8(q.all_q,size_t(M)*f::H,61); fp8(q.all_s,size_t(M)*(f::H/32),67,true);
    std::vector<int> ix(size_t(M)*f::TK,-1); std::vector<float> wt(ix.size(),0);
    for(int r:{0,8192,M-1}) { ix[size_t(r)*f::TK]=0; wt[size_t(r)*f::TK]=0.75f; }
    put(q.all_idx,ix); put(q.all_wt,wt);
    for(auto b:a.blocks)if(b.p+red==fc1 || b.p+red==fc2 || b.p+red==w13 || b.p+red==s13 || b.p+red==w2 || b.p+red==s2 || b.p+red==q.all_q || b.p+red==q.all_s || b.p+red==reinterpret_cast<uint8_t*>(q.all_idx) || b.p+red==reinterpret_cast<uint8_t*>(q.all_wt))input.add(b.p+red,b.n);
    check(cudaFuncSetAttribute(mxtc::k_fc1_tc<f::H,f::I,DS41_TC_CW,DS41_TC_NTG1,DS41_TC_ST1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,mxtc::fc1_smem<f::H,DS41_TC_CW,DS41_TC_NTG1,DS41_TC_ST1>()));
    check(cudaFuncSetAttribute(mxtc::k_fc2_tc<f::I,f::H,DS41_TC_W2,DS41_TC_WT,DS41_TC_TPC,DS41_TC_ST2,4,ds41v2moe::Args>,cudaFuncAttributeMaxDynamicSharedMemorySize,mxtc::fc2_smem<DS41_TC_W2,DS41_TC_TPC,DS41_TC_ST2>()));
    std::vector<Bytes> baseline; Bytes record;
    for(bool skip:{false,true}) {
        check(cudaMemset(q.record_q,0xd3,size_t(M)*f::H));
        if(skip)upload(q.record_q,record);
        check(cudaMemset(q.aq,0xd3,8*f::I)); check(cudaMemset(q.as,0xd3,8*mxtc::ASP));
        check(cudaMemset(q.sh13,0xd3,size_t(M)*2*f::I*2)); check(cudaMemset(q.sh_q,0xd3,size_t(M)*f::KP)); check(cudaMemset(q.sh_s,0xd3,size_t(M)*(f::KP/32)));
        check(f::up(g,w,q,nullptr,nullptr,skip)); sync();
        if(!skip) {
            // Independent host permutation, not the production relayout kernel.
            auto plain=download(q.all_q,size_t(M)*f::H); record.resize(plain.size());
            for(size_t i=0;i<plain.size();++i)record[i-i%f::H+record_offset(i%f::H)]=plain[i];
        }
        equal(q.record_q,record,"record coordinates and skip preservation");
        auto plan=std::make_unique<f::PlanW16K>(); check(cudaMemcpy(plan.get(),q.plan16k,sizeof(*plan),cudaMemcpyDeviceToHost));
        insist(plan->n_tiles==1 && plan->n_slots==1 && plan->slot_expert[0]==0,"sparse wide planner single expert tile");
        std::vector<Bytes> result={download(q.aq,8*f::I),download(q.as,8*mxtc::ASP),download(q.sh13,size_t(M)*2*f::I*2),download(q.sh_q,size_t(M)*f::KP),download(q.sh_s,size_t(M)*(f::KP/32))};
        for(int c=0;c<4;++c) {
            check(cudaMemset(out,0xd3,size_t(g.padded())*f::COLS*2)); check(cudaMemset(q.partial,0xd3,8*f::H*2)); check(cudaMemset(q.shared,0xd3,size_t(M)*f::COLS*2));
            check(f::down_group(g,w,q,c,nullptr)); sync();
            auto raw=download(q.partial,8*f::H*2),shared=download(q.shared,size_t(M)*f::COLS*2),actual=download(out,size_t(g.padded())*f::COLS*2);
            std::vector<__half> part(8*f::H); memcpy(part.data(),raw.data(),raw.size());
            std::vector<BF> sh(size_t(M)*f::COLS),want(size_t(g.padded())*f::COLS); memcpy(sh.data(),shared.data(),shared.size());
            size_t nonzero=0;
            for(int m=0;m<g.padded();++m)for(int col=0;col<f::COLS;++col) {
                float sum=0;
                if(m<M) { for(int k=0;k<f::TK;++k)if(plan->tok_slot[m][k]>=0)sum+=__half2float(part[(size_t(plan->tok_slot[m][k])*8+plan->tok_col[m][k])*f::H+c*f::COLS+col]); sum+=__bfloat162float(sh[size_t(m)*f::COLS+col]); }
                insist(std::isfinite(sum),"finite nondegenerate outputs"); nonzero+=sum!=0;
                want[size_t(m)*f::COLS+col]=__float2bfloat16_rn(sum);
            }
            insist(nonzero>0 && !memcmp(actual.data(),want.data(),actual.size()),"down_group CPU ordered combine exact");
            result.push_back(std::move(actual));
        }
        if(!skip)baseline=std::move(result); else insist(result==baseline,"up false/true and all down groups byte exact");
    }
    check(cudaMemset(q.record_q,0,f::H)); check(cudaMemset(q.aq,0xd3,8*f::I));
    check(f::up(g,w,q,nullptr,nullptr,true)); sync();
    insist(download(q.aq,8*f::I)!=baseline[0],"skip relayout corrupt active record negative control");
    input.verify(); printf("PASS up/down_group M=%d H=5120 skip=false,true 4groups memcmp=0 sparse_routes nonzero negative_control=rejected live_bytes=%zu\n",M,live);
}
void engram_admission() {
    for(bool v2:{false,true})for(bool wide:{false,true})for(int m:{0,4096,4097,8192,8193,12288,16384,16385}) {
        eg::Config config; config.prefix="/nonexistent/wide_api_contract"; config.max_rows=m;
        config.prefill_v2=v2; config.prefill_v2_w16k=wide;
        eg::Engram instance; std::string error; insist(!instance.init(config,&error),"nonexistent sidecar cannot initialize");
        const int cap=v2?(wide?16384:8192):4096;
        if(m<1 || m>cap)insist(error=="engram: max_rows outside 1.."+std::to_string(cap),"Engram admission bound");
        else insist(error.find("wide_api_contract")!=std::string::npos && error.find("max_rows outside")==std::string::npos,"valid capacity reaches sidecar lookup");
    }
    printf("PASS Engram admission 32 cases zero device allocation\n");
}
constexpr int side_rows=193;
uint8_t code(int row,int cell,int k,int peer) { return uint8_t(16+(row*7+cell*13+k*3+peer*29)%80); }
uint8_t scale(int local,int k,int peer) { return uint8_t(120+(local+k+peer)%4); }
void engram_wire_case(int M) {
    Arena a; Inputs input; const auto g=geometry(M,4,3);
    const size_t entries=size_t(M)*24;
    auto ids=a.make<uint32_t>(entries); std::vector<uint32_t> host_ids(entries);
    for(int r=0;r<M;++r)for(int c=0;c<24;++c)host_ids[size_t(r)*24+c]=((r+c)%5)*side_rows+(r*17+c*11)%side_rows;
    put(ids,host_ids); input.add(ids,entries*4);
    auto recv=a.make<uint8_t>(size_t(4)*g.Q*ew::ROW_BYTES);
    {
        Arena packing; auto stage=packing.make<uint8_t>(entries*256); auto send=packing.make<uint8_t>(size_t(M)*ew::ROW_BYTES);
        auto scales=packing.make<uint8_t>(side_rows*8);
        Bytes host_stage(entries*256),host_scales(side_rows*8),reference(size_t(M)*ew::ROW_BYTES);
        for(int peer=0;peer<4;++peer) {
            for(int loc=0;loc<side_rows;++loc)for(int k=0;k<8;++k)host_scales[loc*8+k]=scale(loc,k,peer);
            for(int r=0;r<M;++r)for(int c=0;c<24;++c)for(int k=0;k<256;++k)host_stage[(size_t(r)*24+c)*256+k]=code(r,c,k,peer);
            upload(stage,host_stage); upload(scales,host_scales); check(cudaMemset(send,0xd3,reference.size()));
            ew::pack<<<M,256>>>(ids,stage,scales,peer*side_rows,side_rows,M,send); sync();
            std::fill(reference.begin(),reference.end(),0);
            for(int r=0;r<M;++r) {
                uint32_t header[4]={}; uint8_t* row=reference.data()+size_t(r)*ew::ROW_BYTES;
                for(int c=0;c<24;++c) {
                    int loc=int(host_ids[size_t(r)*24+c])-peer*side_rows;
                    if(loc<0 || loc>=side_rows)continue;
                    ++header[0]; header[1]|=1u<<c;
                    memcpy(row+16+c*264,host_stage.data()+(size_t(r)*24+c)*256,256);
                    memcpy(row+16+c*264+256,host_scales.data()+loc*8,8);
                }
                memcpy(row,header,sizeof(header));
            }
            equal(send,reference,"every wide Engram wire byte CPU oracle");
            equal(stage,host_stage,"immutable Engram stage"); equal(scales,host_scales,"immutable Engram scales");
            check(cudaMemcpy(recv+size_t(peer)*g.Q*ew::ROW_BYTES,send+size_t(g.row0)*ew::ROW_BYTES,size_t(g.owned)*ew::ROW_BYTES,cudaMemcpyDeviceToDevice));
        }
        printf("PASS Engram pack M=%d every_byte 4peers live_bytes=%zu\n",M,live);
    }
    auto output=a.make<BF>(size_t(g.Q)*6144); input.add(recv,size_t(4)*g.Q*ew::ROW_BYTES);
    ew::unpack<<<dim3(g.owned,24),256>>>(recv,g.Q,g.owned,4,output); sync();
    Bytes expected(size_t(g.Q)*6144*2,0xd3);
    for(int r=0;r<g.owned;++r)for(int c=0;c<24;++c)for(int k=0;k<256;++k) {
        int global=g.row0+r,owner=host_ids[size_t(global)*24+c]/side_rows,local=host_ids[size_t(global)*24+c]%side_rows;
        uint16_t bits=owner<4?eg::dequant1(code(global,c,k,owner),scale(local,k/32,owner)):0;
        memcpy(expected.data()+((size_t(r)*24+c)*256+k)*2,&bits,2);
    }
    equal(output,expected,"Engram unpack CPU dequant and ragged padding"); input.verify();
    int cell=0; while(host_ids[size_t(g.row0)*24+cell]/side_rows>=4)++cell;
    int peer=host_ids[size_t(g.row0)*24+cell]/side_rows;
    auto location=recv+size_t(peer)*g.Q*ew::ROW_BYTES+16+cell*264;
    auto changed=download(location,1); changed[0]^=0x80; upload(location,changed);
    ew::unpack<<<dim3(g.owned,24),256>>>(recv,g.Q,g.owned,4,output); sync();
    insist(download(output,expected.size())!=expected,"Engram corrupted wire negative control rejected");
    printf("PASS Engram unpack M=%d rank=3 CPU_exact padded_poison negative_control=rejected\n",M);
}
} // namespace
int main() {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);
    setenv("GR8_DS_PF_V2","0",1); setenv("GR8_DS_PF_V2_TIMING","0",1);
    setenv("GR8_DS_PF_V2_GATE_RO","0",1);
    prepare_contract();
    for(int M:{8193,12288,16384}) { route_api(M); wide_ffn(M); }
    engram_admission(); engram_wire_case(16381); engram_wire_case(16384);
    summary("w16k_actual_api");
}
