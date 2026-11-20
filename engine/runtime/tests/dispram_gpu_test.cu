// Run on a GPU host: verifies every word and the imported-to-created VMM seam.
#include "../../dispram.h"
#include <cassert>
#include <vector>
#include <thread>
#include <sys/wait.h>
using namespace g6dispram;
static void check(cudaError_t rc) { if (rc!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(rc)); std::exit(2); } }
__host__ __device__ static uint64_t pattern(size_t global_word) { return 0x9e3779b97f4a7c15ULL ^ (uint64_t(global_word)*0xd6e8feb86659fd93ULL); }
__device__ static uint64_t load_bypass_l2(const uint64_t* p) {
    unsigned long long v = 0;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return static_cast<uint64_t>(v);
}
__global__ void fill(uint64_t* p,size_t words,size_t word_base) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=size_t(blockDim.x)*gridDim.x) p[i]=pattern(word_base+i);
}
__global__ void verify(const uint64_t* p,size_t words,size_t word_base,unsigned long long* errors) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=size_t(blockDim.x)*gridDim.x)
        if (load_bypass_l2(p+i)!=pattern(word_base+i)) atomicAdd(errors,1ULL);
}
static unsigned long long verify_span_count(const void* p,size_t bytes,size_t word_base) {
    unsigned long long* bad=nullptr; check(cudaMalloc(&bad,sizeof(*bad))); check(cudaMemset(bad,0,sizeof(*bad)));
    verify<<<1024,256>>>(static_cast<const uint64_t*>(p),bytes/8,word_base,bad); check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    unsigned long long n=0; check(cudaMemcpy(&n,bad,sizeof(n),cudaMemcpyDeviceToHost)); check(cudaFree(bad)); return n;
}
static void verify_span(const void* p,size_t bytes,size_t word_base) { assert(verify_span_count(p,bytes,word_base)==0); }
static void alias_negative_test() {
    int count=0; CudaBackend backend; size_t gmin=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1) { puts("SKIP dispram alias negative test (VMM unavailable)"); return; }
    check(cudaSetDevice(0)); check(cudaFree(nullptr));
    if (!backend.granularity(&gmin)) { puts("SKIP dispram alias negative test (VMM unavailable)"); return; }
    const size_t gran=lcm_gran(GRANULE,gmin); const auto& vmm=vmm_api(); auto prop=backend.prop;
    CUmemGenericAllocationHandle h{}; CUdeviceptr va=0;
    assert(vmm.create(&h,gran,&prop,0)==CUDA_SUCCESS);
    assert(vmm.reserve(&va,2*gran,gran,0,0)==CUDA_SUCCESS);
    assert(vmm.map(va,gran,0,h,0)==CUDA_SUCCESS);
    assert(vmm.map(va+gran,gran,0,h,0)==CUDA_SUCCESS);
    CUmemAccessDesc access{}; access.location=prop.location; access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    assert(vmm.access(va,2*gran,&access,1)==CUDA_SUCCESS);
    fill<<<256,256>>>(reinterpret_cast<uint64_t*>(va),gran/8,0); check(cudaGetLastError());
    fill<<<256,256>>>(reinterpret_cast<uint64_t*>(va+gran),gran/8,gran/8); check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    assert(verify_span_count(reinterpret_cast<const void*>(va),gran,0)==gran/8);
    assert(verify_span_count(reinterpret_cast<const void*>(va+gran),gran,gran/8)==0);
    assert(vmm.unmap(va,gran)==CUDA_SUCCESS && vmm.unmap(va+gran,gran)==CUDA_SUCCESS);
    assert(vmm.free_va(va,2*gran)==CUDA_SUCCESS && vmm.release(h)==CUDA_SUCCESS);
    puts("PASS dispram alias negative test: every aliased word rejected");
}
static void child(char* self,const char* mode,int expected,bool required=false) {
    pid_t p=fork(); assert(p>=0);
    if (!p) {
        setenv("GR8_DISPRAM_MIN_BYTES",!strcmp(mode,"shortfall")?"2145386496":"0",1);
        setenv("GR8_DISPRAM",!strcmp(mode,"disabled")?"0":"1",1); setenv("DISPRAM_DISABLE","0",1);
        if (!strcmp(mode,"default_off")) { unsetenv("GR8_DISPRAM"); unsetenv("G6_DISPRAM"); }
        if (strcmp(mode,"partial") && strcmp(mode,"live_seam") && strcmp(mode,"default_off"))
            setenv("DISPRAM_SOCKET","/tmp/nonexistent-dispram.sock",1);
        char* args[]={self,const_cast<char*>(mode),required?const_cast<char*>("--require-dispram"):nullptr,nullptr}; execv(self,args); _exit(99);
    }
    int st=0; assert(waitpid(p,&st,0)==p); assert(WIFEXITED(st));
    if (!strcmp(mode,"transport_loss") && WEXITSTATUS(st)==77 && !required) {
        puts("SKIP dispram transport-loss VMM test (VMM unavailable)"); return;
    }
    assert(WEXITSTATUS(st)==expected);
}
static Info info_now() { Client c; Info i; assert(c.open_conn(socket_path()) && c.query_info(&i)); return i; }
static void partial_test() {
    Client probe; Info info;
    if (!probe.open_conn(socket_path()) || !probe.query_info(&info) || info.free_bytes<4*GRANULE) {
        puts("SKIP dispram partial-lease test (no available daemon)"); return;
    }
    probe.close_conn();
    void *a=nullptr,*b=nullptr; check(alloc_kv(&a,2*GRANULE)); assert(claimed_bytes()==2*GRANULE);
    set_fail_vmm_after_spans_for_test(1); check(alloc_kv(&b,2*GRANULE));
    assert(has_open_conn_for_test() && claimed_bytes()==2*GRANULE);
    fill<<<256,256>>>(static_cast<uint64_t*>(a),2*GRANULE/8,0); check(cudaGetLastError());
    fill<<<256,256>>>(static_cast<uint64_t*>(b),2*GRANULE/8,2*GRANULE/8); check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    verify_span(a,2*GRANULE,0); verify_span(b,2*GRANULE,2*GRANULE/8);
    set_fail_vmm_after_spans_for_test(-1); check(free_kv(b)); check(free_kv(a));
    assert(!has_open_conn_for_test() && claimed_bytes()==0 && mapped_bytes()==0 && info_now().free_bytes==info.free_bytes);
}

static int transport_loss_test() {
#ifdef SOCK_SEQPACKET
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1 || !vmm_api().ok) return 77;
    check(cudaSetDevice(0)); check(cudaFree(nullptr));
    CudaBackend backend; size_t gran=0;
    if (!backend.granularity(&gran) || lcm_gran(GRANULE,gran)!=GRANULE) return 77;
    decltype(&cuMemExportToShareableHandle) export_fd=nullptr;
    void* fn=nullptr; cudaDriverEntryPointQueryResult query;
    if (cudaGetDriverEntryPointByVersion("cuMemExportToShareableHandle",&fn,12000,cudaEnableDefault,&query)!=cudaSuccess ||
        query!=cudaDriverEntryPointSuccess) return 77;
    export_fd=reinterpret_cast<decltype(export_fd)>(fn);
    char path[]="/tmp/gr8-dispram-gpu-XXXXXX"; int tmp=mkstemp(path); assert(tmp>=0); close(tmp); unlink(path);
    setenv("DISPRAM_SOCKET",path,1); setenv("GR8_DISPRAM_MIN_BYTES","0",1);
    int listener=socket(AF_UNIX,SOCK_SEQPACKET,0); assert(listener>=0);
    sockaddr_un addr{}; addr.sun_family=AF_UNIX; memcpy(addr.sun_path,path,sizeof(path));
    assert(posix_bind(listener,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0 && listen(listener,1)==0);
    std::thread server([&] {
        check(cudaSetDevice(0));
        int fd=accept(listener,nullptr,nullptr); assert(fd>=0); unlink(path); close(listener);
        char request[4096]; assert(recv(fd,request,sizeof(request),0)>0);
        const char info[]="{\"free\":4194304,\"size\":4194304,\"base\":0,\"largest\":4194304}";
        assert(send(fd,info,sizeof(info)-1,0)==sizeof(info)-1);
        assert(recv(fd,request,sizeof(request),0)>0);
        auto prop=backend.prop; prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmemGenericAllocationHandle h{}; const auto& api=vmm_api();
        assert(api.create(&h,GRANULE,&prop,0)==CUDA_SUCCESS);
        int share=-1; assert(export_fd(&share,h,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0)==CUDA_SUCCESS);
        const char reply[]="{\"ok\":true,\"size\":2097152}";
        iovec io{const_cast<char*>(reply),sizeof(reply)-1}; msghdr msg{}; msg.msg_iov=&io; msg.msg_iovlen=1;
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
        msg.msg_control=control; msg.msg_controllen=sizeof(control);
        cmsghdr* c=CMSG_FIRSTHDR(&msg); c->cmsg_level=SOL_SOCKET; c->cmsg_type=SCM_RIGHTS; c->cmsg_len=CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(c),&share,sizeof(share)); assert(sendmsg(fd,&msg,0)==sizeof(reply)-1);
        close(share); assert(api.release(h)==CUDA_SUCCESS);
        assert(recv(fd,request,sizeof(request),0)>0); close(fd);
    });
    void *a=nullptr,*b=nullptr;
    check(alloc_kv(&a,GRANULE)); assert(claimed_bytes()==GRANULE);
    fill<<<64,256>>>(static_cast<uint64_t*>(a),GRANULE/8,0); check(cudaGetLastError());
    (void)alloc_kv(&b,GRANULE); // Must synchronize/unmap and exit 75, even with minimum zero.
    server.join(); assert(false && "transport loss continued with live mappings"); return 1;
#else
    return 77;
#endif
}
static int live_seam_test(bool required) {
    alias_negative_test();
    Client probe; Info info;
    bool live=probe.open_conn(socket_path()) && probe.query_info(&info) && info.free_bytes>=4*GRANULE;
    probe.close_conn();
    if (!live) { if (required) { fprintf(stderr,"dispramd with at least 8 MiB free required\n"); return 1; } puts("SKIP dispram live carveout seam test (no dispramd socket)"); return 0; }
    assert(enabled()); int dev=0; check(cudaGetDevice(&dev));
    CUmemAllocationProp prop{}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=dev;
    size_t gran=0; const auto& api=vmm_api(); assert(api.ok && api.gran(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM)==CUDA_SUCCESS);
    assert(lcm_gran(GRANULE,gran)==GRANULE && info.free_bytes%GRANULE==0);
    struct Allocation { void* p; size_t bytes; size_t word_base; }; std::vector<Allocation> spans;
    size_t remain=info.free_bytes,seam=0,word_base=0;
    while (remain) {
        size_t carve=std::min(remain,size_t(32<<20)); bool last=carve==remain;
        size_t bytes=carve+(last?(8<<20):0); void* p=nullptr; check(alloc_kv(&p,bytes)); spans.push_back({p,bytes,word_base}); word_base+=bytes/8; remain-=carve;
        if (last) seam=carve;
    }
    assert(claimed_bytes()==info.free_bytes && mapped_bytes()==info.free_bytes+(8<<20));
    for (auto a:spans) { fill<<<1024,256>>>(static_cast<uint64_t*>(a.p),a.bytes/8,a.word_base); check(cudaGetLastError()); }
    check(cudaDeviceSynchronize());
    for (auto a:spans) verify_span(a.p,a.bytes,a.word_base);
    const auto& last=spans.back(); std::vector<uint64_t> seam_words((128<<10)/8);
    check(cudaMemcpy(seam_words.data(),static_cast<char*>(last.p)+seam-(64<<10),128<<10,cudaMemcpyDeviceToHost));
    for (size_t i=0;i<seam_words.size();++i) assert(seam_words[i]==pattern(last.word_base+(seam-(64<<10))/8+i));
    for (auto a:spans) check(free_kv(a.p));
    assert(claimed_bytes()==0 && mapped_bytes()==0 && !has_open_conn_for_test());
    assert(info_now().free_bytes==info.free_bytes);
    puts("PASS dispram GPU: full carveout, every word, VMM seam, reclamation, pre-allocation shortfall");
    return 0;
}

int main(int argc,char** argv) {
    if (argc>1 && !strcmp(argv[1],"transport_loss")) return transport_loss_test();
    bool required=knob::dispram_min_bytes()>0;
    for (int n=1;n<argc;++n) if (!strcmp(argv[n],"--require-dispram")) required=true;
    if (argc>1 && !strcmp(argv[1],"partial")) { partial_test(); return 0; }
    if (argc>1 && !strcmp(argv[1],"live_seam")) return live_seam_test(required);
    if (argc>1 && (!strcmp(argv[1],"default_off") || !strcmp(argv[1],"disabled") || !strcmp(argv[1],"missing") || !strcmp(argv[1],"shortfall"))) {
        if (!strcmp(argv[1],"default_off")) assert(!enabled());
        void* p=nullptr; check(alloc_kv(&p,4*GRANULE)); assert(claimed_bytes()==0); check(free_kv(p)); return 0;
    }
    // Re-exec before any parent CUDA initialization.
    child(argv[0],"default_off",0);
    child(argv[0],"transport_loss",75,required);
    child(argv[0],"disabled",0); child(argv[0],"missing",0); child(argv[0],"shortfall",75); child(argv[0],"partial",0);
    child(argv[0],"live_seam",0,required);
}
