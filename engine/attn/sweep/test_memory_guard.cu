// Build: nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a -DG6_MODEL_DS41=1 test_memory_guard.cu -o test_memory_guard
// Run under the shared GPU lock. No modes intentionally perform invalid CUDA
// memory accesses: negative writes stay inside the enclosing guarded allocation.
#include "adapter.cuh"
#include "variants/off/variant.cuh"
#include <cassert>
#include <utility>

static void require(bool condition, const char* reason) {
    if (!condition) { fprintf(stderr,"TEST_FAILURE: %s\n",reason); std::exit(3); }
}
static std::vector<std::string> shape_args(int ratio=2, int floor=0) {
    return {"3","16",std::to_string(ratio),std::to_string(floor),"16","32"};
}
static void flip(unsigned char* pointer, cudaStream_t st) {
    unsigned char x;
    CK(cudaMemcpyAsync(&x,pointer,1,cudaMemcpyDeviceToHost,st)); CK(cudaStreamSynchronize(st));
    x^=1;
    CK(cudaMemcpyAsync(pointer,&x,1,cudaMemcpyHostToDevice,st)); CK(cudaStreamSynchronize(st));
}
static void off_pair(sweep_adapter::Fixture& f, sweep_adapter::Fixture& dest, cudaStream_t st) {
    auto ref=sweep_adapter::params_with_output(f,dest,true);
    auto cand=sweep_adapter::params_with_output(f,dest,false);
    const auto bytes=sweep_adapter::output_bytes(dest);
    CK(cudaMemsetAsync(ref.out,0xA5,bytes,st)); CK(cudaMemsetAsync(cand.out,0x5A,bytes,st));
    sweep_variant::launch(ref,st); sweep_variant::launch(cand,st);
    CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
    const auto a=back(reinterpret_cast<uint8_t*>(ref.out),bytes), b=back(reinterpret_cast<uint8_t*>(cand.out),bytes);
    require(a==b,"off outputs not exact");
}
static int negative(const std::string& mode, cudaStream_t st) {
    auto f=sweep_adapter::make_case(shape_args(),123);
    CK(sweep_variant::init(sweep_adapter::params(f,false)));
    sweep_adapter::snapshot_inputs(f,st);
    off_pair(f,f,st);
    require(sweep_adapter::check_integrity(f,st)==0,"negative control baseline not clean");
    auto& allocations=f.memory->allocations;
    if (mode=="missing-snapshot") f.memory->invalidate_snapshot();
    else {
        const std::string name=mode=="mutate-input" ? "q" : mode=="scratch-overrun" ? "integrity_hash_scratch" : "candidate_output";
        for (auto& a:allocations) if (a.name==name) {
            auto* ptr=mode=="underrun" ? a.logical-1 : mode=="mutate-input" ? a.logical : a.logical+a.bytes;
            flip(ptr,st);
        }
    }
    const auto corrupt=sweep_adapter::check_integrity(f,st);
    const auto counts=sweep_adapter::integrity_counts(f);
    printf("mode=%s output_status=EXACT status=%s corruption_count=%zu guard_bytes=%zu immutable_chunks=%zu missing_snapshots=%zu\n",
           mode.c_str(),corrupt?"MEMORY_CORRUPTION":"CLEAN",corrupt,counts.guard_bytes,counts.immutable_chunks,counts.missing_snapshots);
    sweep_adapter::destroy(f);
    return corrupt ? 1 : 0;  // deliberately bad modes MUST return nonzero
}
static void positive(cudaStream_t st) {
    for (const auto config : {std::pair<int,int>{0,0},{1,0},{2,0},{2,8}}) {
        const auto args=shape_args(config.first,config.second);
        std::vector<sweep_adapter::Fixture> slots;
        for (int i=0;i<3;++i) slots.push_back(sweep_adapter::make_case(args,123+i));
        // init ONCE despite vector relocation and later input/output rotation.
        CK(sweep_variant::init(sweep_adapter::params(slots[0],false)));
        for (auto& f:slots) {
            require(f.memory->device_bytes()==sweep_adapter::required_bytes(args),"VRAM accounting mismatch");
            require(f.memory->allocations.size()==10,"allocation missing from registry");
            for (const auto& a:f.memory->allocations)
                require(uintptr_t(a.logical)%256==0,"guard offset lost allocation alignment");
            sweep_adapter::snapshot_inputs(f,st);
        }
        std::vector<std::vector<sweep_memory::Digest>> previous;
        for (uint64_t nonce=1;nonce<=3;++nonce) {
            auto& f=slots[0];
            const auto tk=back(f.b.d_topk,size_t(f.b.p.n)*512), positions=back(f.b.d_pos,size_t(f.b.p.n));
            sweep_adapter::refresh_inputs(f,nonce,st);
            sweep_adapter::snapshot_inputs(f,st);
            // All refreshed data classes change with the nonce, but metadata stays valid.
            std::vector<std::vector<sweep_memory::Digest>> current;
            for (const auto& a:f.memory->allocations) {
                if (a.name=="q" || a.name=="window_kv" || a.name=="keys_packed" || a.name=="keys_bf16" || a.name=="sink")
                    current.push_back(a.expected);
            }
            if (!previous.empty()) for (size_t i=0;i<current.size();++i)
                require(std::memcmp(previous[i].data(),current[i].data(),current[i].size()*sizeof(sweep_memory::Digest))!=0,"refresh left an input unchanged");
            previous=current;
            require(tk==back(f.b.d_topk,size_t(f.b.p.n)*512),"refresh changed top-k");
            require(positions==back(f.b.d_pos,size_t(f.b.p.n)),"refresh changed positions");
            const auto packed=back(f.keys.pk,f.keys.rows*CQ_ROW);
            const auto bf=back(reinterpret_cast<uint16_t*>(f.keys.bf),f.keys.rows*HD);
            std::vector<uint16_t> decoded(f.keys.rows*HD);
            for (size_t r=0;r<f.keys.rows;++r) h_decode(packed.data()+r*CQ_ROW,decoded.data()+r*HD);
            require(bf==decoded,"refreshed key forms disagree");
            // Correctness launch, then several timing-style launches without a
            // re-init or refresh. Integrity is checked after BOTH phases.
            off_pair(f,slots[nonce%3],st);
            for (auto& slot:slots) require(sweep_adapter::check_integrity(slot,st)==0,"off correctness corrupted memory");
            cudaEvent_t begin,end; CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
            CK(cudaEventRecord(begin,st));
            for (int i=0;i<3;++i) sweep_variant::launch(sweep_adapter::params_with_output(f,slots[i],false),st);
            CK(cudaEventRecord(end,st)); CK(cudaEventSynchronize(end));
            for (auto& slot:slots) require(sweep_adapter::check_integrity(slot,st)==0,"off timing corrupted memory");
            CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
        }
        printf("off ratio=%d floor=%d transitions=3 slots=3 status=EXACT integrity=CLEAN device_bytes=%zu host_snapshot_bytes=%zu\n",
               config.first,config.second,slots[0].memory->device_bytes(),slots[0].memory->host_snapshot_bytes());
        for (auto& f:slots) { sweep_adapter::destroy(f); require(!f.memory,"destroy retained registry"); }
    }
}
static void exhaustive(cudaStream_t st) {
    auto f=sweep_adapter::make_case(shape_args(),321);
    sweep_adapter::snapshot_inputs(f,st);
    size_t guard_tests=0,input_tests=0;
    for (auto& a:f.memory->allocations) {
        // Inner and outer boundaries of BOTH guard regions, on EVERY allocation.
        for (auto* ptr : {a.logical-sweep_memory::redzone_bytes,a.logical-1,a.logical+a.bytes,a.logical+a.bytes+sweep_memory::redzone_bytes-1}) {
            flip(ptr,st);
            require(sweep_adapter::check_integrity(f,st)>0,"guard mutation escaped detection");
            require(sweep_adapter::integrity_counts(f).guard_bytes==1,"wrong guard count");
            flip(ptr,st); ++guard_tests;
        }
        if (a.immutable) for (size_t offset : {size_t(0),a.bytes/2,a.bytes-1}) {
            flip(a.logical+offset,st);
            require(sweep_adapter::check_integrity(f,st)>0,"input mutation escaped detection");
            require(sweep_adapter::integrity_counts(f).immutable_chunks==1,"wrong immutable chunk count");
            flip(a.logical+offset,st); ++input_tests;
        }
    }
    require(sweep_adapter::check_integrity(f,st)==0,"restored fixture not clean");
    printf("coverage guard_mutations=%zu immutable_mutations=%zu status=PASS\n",guard_tests,input_tests);
    sweep_adapter::destroy(f);
}
int main(int argc,char** argv) {
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    int result=0;
    const std::string mode=argc>1?argv[1]:"sanity";
    if (mode=="sanity") { positive(st); exhaustive(st); }
    else if (mode=="mutate-input" || mode=="overrun" || mode=="underrun" || mode=="scratch-overrun" || mode=="missing-snapshot") result=negative(mode,st);
    else { fprintf(stderr,"unknown mode\n"); result=2; }
    CK(cudaStreamDestroy(st));
    return result;
}
