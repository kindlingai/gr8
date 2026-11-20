#include "ds41_v2_test_support.cuh"
#include "../attn/ds41_attn_v2_api.h"
#include "../kernels/ds41_v2_moe.cuh"
using namespace v2test;
extern "C" void ds41_attn_set_tc(int);

template<bool Trace> void attention(int M,int heads,bool offset) {
    Arena a; Inputs inputs;
    int K=ds41a::v2::output_width(heads), Q=(M+3)/4, P=4*Q;
    size_t row=K+K/32;
    auto q=a.make<uint8_t>(size_t(M)*K+offset)+offset;
    auto s=a.make<uint8_t>(size_t(M)*K/32+offset)+offset;
    auto packed=a.make<uint8_t>(size_t(P)*row), dest=a.make<uint8_t>(size_t(P)*row);
    ds41a::Scratch scratch{}; scratch.oaq=q; scratch.oas=s;
    Segment seg{packed,dest,row}; auto layout=wire_layout(Q,&seg,1);
    Wire wire(a,4*layout.stride,Trace);
    std::array<Bytes,4> refs,opts;
    for(int rank=0;rank<4;++rank) {
        upload(q,data(size_t(M)*K,17+rank)); upload(s,data(size_t(M)*K/32,47+rank));
        Inputs immutable; immutable.add(q,size_t(M)*K); immutable.add(s,size_t(M)*K/32);
        auto g=geometry(M,4,rank); auto args=detail::segments(g,&seg,1);
        for(int variant=0;variant<2;++variant) {
            wire.clear(rank);
            if constexpr(Trace)detail::trace_produced<<<1,32>>>(wire.f,1,0);
            if(variant)detail::pack_attn_out_remote<<<32,256>>>(wire.f,1,0,g,q,s,K,layout.stride);
            else {
                check(cudaMemset(packed,0x5e,size_t(P)*row));
                ds41_v2_output_pack(&scratch,packed,M,P,heads,nullptr);
                detail::pack<<<8,256>>>(wire.f,1,0,g,args,true);
            }
            detail::publish<Trace><<<1,32>>>(wire.f,1,0);
            (variant?opts:refs)[rank]=wire.snapshot();
        }
        for(int peer=0;peer<4;++peer) {
            size_t off=size_t(peer)*layout.stride;
            if(peer!=rank)insist(memcmp(refs[rank].data()+off,opts[rank].data()+off,layout.payload)==0,"remote wire bytes");
            else for(size_t i=0;i<layout.stride;++i)insist(opts[rank][off+i]==0xa7,"own wire quarter untouched");
        }
        immutable.verify();
    }
    for(int rank=0;rank<4;++rank) {
        auto g=geometry(M,4,rank); auto args=detail::segments(g,&seg,1); Bytes ref;
        for(int variant=0;variant<2;++variant) {
            wire.clear(rank); wire.deliver(variant?opts:refs,layout,true);
            check(cudaMemset(dest,0x5e,size_t(P)*row));
            detail::wait<Trace><<<1,32>>>(wire.f,1,0,layout.payload);
            if(variant)detail::unpack_remote<Trace><<<32,256>>>(wire.f,1,0,g,args);
            else detail::unpack<Trace><<<8,256>>>(wire.f,1,0,g,args,true);
            auto result=download(dest,size_t(P)*row);
            if(!variant)ref=result;
            else for(int peer=0;peer<4;++peer)for(size_t i=0;i<size_t(Q)*row;++i) {
                size_t j=size_t(peer)*Q*row+i;
                insist(result[j]==(peer==rank?0x5e:ref[j]),"remote unpack and padding");
            }
            insist(*wire.f.error==0,"wire no timeout");
        }
    }
    auto bad=opts[0]; bad[size_t(1)*layout.stride]^=1;
    insist(memcmp(bad.data()+layout.stride,refs[0].data()+layout.stride,layout.payload)!=0,"wire negative control");
    printf("PASS attention M=%d heads=%d trace=%d unaligned=%d\n",M,heads,Trace,offset);
}

template<bool Trace> void ffn(int M) {
    Arena a; constexpr int H=5120; int Q=(M+3)/4,P=Q*4;
    const size_t widths[]={H,H/32,24,24}; Segment seg[4];
    for(int k=0;k<4;++k)seg[k]={a.make<uint8_t>(Q*widths[k]),a.make<uint8_t>(P*widths[k]),widths[k]};
    auto record=a.make<uint8_t>(size_t(P)*H); auto layout=wire_layout(Q,seg,4);
    Wire wire(a,layout.stride,Trace); std::array<Bytes,4> messages;
    for(int rank=0;rank<4;++rank) {
        wire.clear(rank); Inputs immutable;
        for(int k=0;k<4;++k) { upload(const_cast<void*>(seg[k].src),data(Q*widths[k],rank*9+k)); immutable.add(seg[k].src,Q*widths[k]); }
        auto args=detail::segments(geometry(M,4,rank),seg,4);
        detail::pack<<<8,256>>>(wire.f,1,0,geometry(M,4,rank),args,false);
        detail::publish<Trace><<<1,32>>>(wire.f,1,0); messages[rank]=wire.snapshot(); immutable.verify();
    }
    for(int rank=0;rank<4;++rank) {
        auto g=geometry(M,4,rank); auto args=detail::segments(g,seg,4);
        std::array<Bytes,5> expected;
        for(int variant=0;variant<2;++variant) {
            wire.clear(rank); wire.deliver(messages,layout,false);
            for(int k=0;k<4;++k)check(cudaMemset(seg[k].dst,0x5e,P*widths[k]));
            check(cudaMemset(record,0x5e,size_t(P)*H));
            detail::wait<Trace><<<1,32>>>(wire.f,1,0,layout.payload);
            if(variant)detail::unpack_ffn_in<Trace><<<32,256>>>(wire.f,1,0,g,args,record,H);
            else {
                detail::unpack<Trace><<<8,256>>>(wire.f,1,0,g,args,false);
                ds41v2moe::k_relayout<true><<<(size_t(M)*H+255)/256,256>>>(static_cast<const uint8_t*>(seg[0].dst),record,M,H);
            }
            for(int k=0;k<5;++k) {
                auto p=k==4?record:seg[k].dst; size_t n=k==4?size_t(P)*H:P*widths[k];
                if(variant)equal(p,expected[k],"FFN dual unpack, all five arrays incl padding"); else expected[k]=download(p,n);
            }
            insist(*wire.f.error==0,"FFN no timeout");
        }
        insist(expected[0]!=expected[4],"negative control missing relayout detected");
    }
    printf("PASS FFN M=%d trace=%d\n",M,Trace);
}

void projection(int rows,int heads,int tc) {
    Arena a; int Q=std::max(rows,1), M=4*Q, K=ds41a::v2::output_width(heads), N=5120;
    size_t row=K+K/32, chunk=size_t(Q)*N*2;
    ds41a::Scratch scratch{}; scratch.oaq=a.make<uint8_t>(size_t(M)*K); scratch.oas=a.make<uint8_t>(size_t(M)*K/32);
    auto packed=a.make<uint8_t>(size_t(M)*row), wq=a.make<uint8_t>(size_t(N)*K), ws=a.make<uint8_t>(size_t(N)*K/32);
    auto xf=a.make<uint32_t>(size_t(K)*4); auto parts=a.make<uint8_t>(4*chunk), out=a.make<uint8_t>(chunk);
    ds41a::Fp8Mat weight{wq,ws,N,K}; ds41_attn_set_tc(tc);
    for(int rank=0;rank<4;++rank) {
        Bytes expected,partials;
        for(int variant=0;variant<2;++variant) {
            check(cudaMemset(parts,0x5e,4*chunk)); check(cudaMemset(out,0x5e,chunk));
            for(int j=0;j<4;++j) {
                int peer=(rank+j)%4;
                fp8(scratch.oaq,size_t(M)*K,peer+3); fp8(scratch.oas,size_t(M)*K/32,peer+7,true);
                fp8(wq,size_t(N)*K,peer+11); fp8(ws,size_t(N)*K/32,peer+13,true);
                Inputs immutable; immutable.add(scratch.oaq,size_t(M)*K); immutable.add(scratch.oas,size_t(M)*K/32);
                immutable.add(wq,size_t(N)*K); immutable.add(ws,size_t(N)*K/32);
                ds41_v2_output_pack(&scratch,packed,M,M,heads,nullptr);
                if(variant && peer==rank)ds41_v2_output_b_split(&weight,scratch.oaq+size_t(rank)*Q*K,scratch.oas+size_t(rank)*Q*(K/32),parts+peer*chunk,rows,xf,nullptr);
                else ds41_v2_output_b(&weight,packed+size_t(rank)*Q*row,parts+peer*chunk,rows,xf,nullptr);
                immutable.verify();
            }
            ds41_v2_output_sum(parts,out,rows,Q,4,nullptr);
            if(variant) { equal(parts,partials,"all projection partials"); equal(out,expected,"own plus remote sum"); }
            else { partials=download(parts,4*chunk); expected=download(out,chunk); }
        }
    }
    printf("PASS projection rows=%d heads=%d tc=%d\n",rows,heads,tc);
}
int main() {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);
    for(int heads:{8,16,32})for(int M:{1,5,127,128,129,257,511,2048}) {
        attention<false>(M,heads,false); attention<true>(M,heads,false);
    }
    for(int heads:{8,16,32}) { attention<false>(17,heads,true); attention<true>(17,heads,true); }
    for(int M:{1,5,127,128,129,257,511,2048,8191,8192}) { ffn<false>(M); ffn<true>(M); }
    for(int tc:{0,1})for(int heads:{8,16,32})for(int rows:{0,1,9,128})projection(rows,heads,tc);
    summary("send_ffn_opt");
}
