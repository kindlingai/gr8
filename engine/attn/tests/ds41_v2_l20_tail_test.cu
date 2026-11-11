// Full-width index numerics with serialized ranks and bounded mapped transport.
#include "../../tests/ds41_v2_test_support.cuh"
#include "../ds41_idx_api.h"
#include "../ds41_idx_layout.h"
using namespace v2test;
void index_rows(int M,bool skip) {
    Arena a; Inputs immutable;
    constexpr int D=512; int Q=(M+3)/4, pos0=skip?192:384, rpp=(pos0+M+63)&~63;
    const size_t ca_size=size_t(3)*rpp*288, sh_size=size_t(3)*rpp*D*2, ic_size=size_t(3)*rpp*ds41i::ICA_ROW_BYTES;
    auto w=a.make<__nv_bfloat16>(size_t(D)*HID), norm=a.make<__nv_bfloat16>(D);
    auto iw=a.make<__nv_bfloat16>(size_t(geom::IDX_DIM)*D), inorm=a.make<__nv_bfloat16>(geom::IDX_DIM);
    auto y=a.make<__nv_bfloat16>(size_t(M)*HID);
    bf16(w,size_t(D)*HID,1); bf16(norm,D,2); bf16(iw,size_t(geom::IDX_DIM)*D,3); bf16(inorm,geom::IDX_DIM,4); bf16(y,size_t(M)*HID,5);
    immutable.add(w,size_t(D)*HID*2); immutable.add(norm,D*2); immutable.add(iw,size_t(geom::IDX_DIM)*D*2); immutable.add(inorm,geom::IDX_DIM*2); immutable.add(y,size_t(M)*HID*2);
    auto positions=a.make<int>(M), pages=a.make<int>(M); std::vector<int> hp(M),hs(M,1);
    for(int i=0;i<M;++i)hp[i]=pos0+i;
    check(cudaMemcpy(positions,hp.data(),M*4,cudaMemcpyHostToDevice)); check(cudaMemcpy(pages,hs.data(),M*4,cudaMemcpyHostToDevice));
    immutable.add(positions,M*4); immutable.add(pages,M*4);
    auto latent=a.make<__nv_bfloat16>(size_t(M)*D), own=a.make<__nv_bfloat16>(size_t(Q)*D), pre=a.make<__nv_bfloat16>(size_t(M)*D);
    auto ca=a.make<uint8_t>(ca_size), sh=a.make<uint8_t>(sh_size), ic=a.make<uint8_t>(ic_size);
    auto init_ca=data(ca_size,11),init_sh=data(sh_size,12),init_ic=data(ic_size,13);
    auto consume=[&]() {
        upload(ca,init_ca); upload(sh,init_sh); upload(ic,init_ic); check(cudaMemset(pre,0x5e,size_t(M)*D*2));
        ds41_idx_ca_row_r1(latent,norm,positions,ca,pages,rpp,pre,reinterpret_cast<__nv_bfloat16*>(sh),M,nullptr);
        ds41_idx_ica_row(pre,iw,inorm,positions,1,ic,pages,rpp,M,nullptr);
    };
    check(cudaMemset(latent,0x5e,size_t(M)*D*2));
    ds41_idx_gemv(y,HID,w,D,HID,latent,D,M,nullptr); consume();
    auto ref_latent=download(latent,size_t(M)*D*2),ref_ca=download(ca,ca_size),ref_sh=download(sh,sh_size),ref_ic=download(ic,ic_size);
    Segment seg{own,latent,D*2}; auto layout=wire_layout(Q,&seg,1);
    Wire wire(a,layout.stride); std::array<Bytes,4> sent;
    for(int rank=0;rank<4;++rank) {
        auto g=geometry(M,4,rank); wire.clear(rank); check(cudaMemset(own,0x5e,size_t(Q)*D*2));
        if(g.owned)ds41_idx_gemv(y+size_t(g.row0)*HID,HID,w,D,HID,own,D,g.owned,nullptr);
        detail::pack<<<8,256>>>(wire.f,1,0,g,detail::segments(g,&seg,1),false);
        detail::publish<<<1,32>>>(wire.f,1,0); sent[rank]=wire.snapshot();
    }
    for(int rank=0;rank<4;++rank) {
        auto g=geometry(M,4,rank); wire.clear(rank); wire.deliver(sent,layout,false);
        check(cudaMemset(latent,0x5e,size_t(M)*D*2));
        detail::wait<<<1,32>>>(wire.f,1,0,layout.payload);
        detail::unpack<<<8,256>>>(wire.f,1,0,g,detail::segments(g,&seg,1),false);
        consume(); equal(latent,ref_latent,"full vs owned projection/BF16 gather"); equal(ca,ref_ca,"CA all slots"); equal(sh,ref_sh,"shadow all slots"); equal(ic,ref_ic,"ICA all slots");
        insist(*wire.f.error==0,"index no timeout");
    }
    immutable.verify();
    // Perturb actual gather input, rerun consumer, and demand a changed latent.
    sent[0][0]^=1; wire.clear(0); wire.deliver(sent,layout,false);
    check(cudaMemset(latent,0x5e,size_t(M)*D*2)); auto g=geometry(M,4,0);
    detail::unpack<<<8,256>>>(wire.f,1,0,g,detail::segments(g,&seg,1),false); consume();
    insist(download(latent,size_t(M)*D*2)!=ref_latent,"input perturbation detected");
    printf("PASS L20 projection/CA/ICA M=%d skip_late=%d four ranks\n",M,skip);
}
void handover(int M) {
    Arena a; const size_t widths[]={size_t(4)*HID*4,96,16,16,64,size_t(HID)*2};
    Segment seg[6]; std::array<Bytes,6> original;
    for(int k=0;k<6;++k) {
        seg[k]={a.make<uint8_t>(128*widths[k]),a.make<uint8_t>(512*widths[k]),widths[k]};
        original[k]=data(size_t(M)*widths[k],k+71);
    }
    auto layout=wire_layout(128,seg,6); Wire wire(a,layout.stride); std::array<Bytes,4> sent;
    for(int rank=0;rank<4;++rank) {
        wire.clear(rank); auto slice=tail_slice(geometry(M,4,rank)); Inputs immutable;
        for(int k=0;k<6;++k) {
            Bytes source(128*widths[k],0);
            if(slice.count)memcpy(source.data(),original[k].data()+size_t(std::max(0,M-128)+slice.destination)*widths[k],size_t(slice.count)*widths[k]);
            upload(const_cast<void*>(seg[k].src),source); immutable.add(seg[k].src,source.size());
        }
        auto g=geometry(512,4,rank);
        detail::pack<<<8,256>>>(wire.f,1,0,g,detail::segments(g,seg,6),false);
        detail::publish<<<1,32>>>(wire.f,1,0); sent[rank]=wire.snapshot(); immutable.verify();
    }
    // One compacted array at a time; mapped and device allocations share one cap.
    for(int rank=0;rank<4;++rank) {
        wire.clear(rank); wire.deliver(sent,layout,false);
        for(int k=0;k<6;++k)check(cudaMemset(seg[k].dst,0x5e,512*widths[k]));
        auto g=geometry(512,4,rank);
        detail::wait<<<1,32>>>(wire.f,1,0,layout.payload);
        detail::unpack<<<8,256>>>(wire.f,1,0,g,detail::segments(g,seg,6),false);
        for(int k=0;k<6;++k) {
            Arena scratch; auto full=scratch.make<uint8_t>(size_t(M)*widths[k]); check(cudaMemset(full,0x5e,size_t(M)*widths[k]));
            int late=std::min(M,128),off=M-late;
            for(int peer=0;peer<4;++peer) {
                auto sl=tail_slice(geometry(M,4,peer));
                if(sl.count)check(cudaMemcpy(full+size_t(off+sl.destination)*widths[k],static_cast<uint8_t*>(seg[k].dst)+size_t(peer)*128*widths[k],size_t(sl.count)*widths[k],cudaMemcpyDeviceToDevice));
            }
            if(off)for(int r=0;r<late;r+=off)check(cudaMemcpy(full+size_t(r)*widths[k],full+size_t(r+off)*widths[k],size_t(std::min(off,late-r))*widths[k],cudaMemcpyDeviceToDevice));
            equal(full,Bytes(original[k].begin()+size_t(off)*widths[k],original[k].end()),"six-array tail compaction");
        }
        insist(*wire.f.error==0,"tail no timeout");
    }
    printf("PASS Tail six full-width arrays M=%d four ranks\n",M);
}
int main() {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS","32",1);
    for(int M:{65,129,512})for(bool skip:{false,true}) { index_rows(M,skip); if(!skip)handover(M); }
    summary("l20_tail");
}
