// Host-only, real O_DIRECT parity, including ring wraparound and short final blocks.
// g++ -std=c++17 -O2 -pthread engine/engram/tests/test_uring.cpp -o /tmp/test_uring
// /tmp/test_uring [--time] [--require-uring]
// --time uses a 1 GiB local-disk file; these are NOT fleet timings.
// Syscall fault injection is test-local: no production environment or test hook.
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/io_uring.h>
#include <cerrno>

static int setup_error = 0, setup_calls = 0, enter_calls = 0;
static bool inject_interrupt = false, inject_partial = false, inject_enter_error = false, inject_cqe_eintr = false;
static io_uring_params last_params{};
static long test_syscall(long op, ...)
{
    va_list ap; va_start(ap, op);
    long rc;
    if (op == __NR_io_uring_setup) {
        ++setup_calls;
        const unsigned n = va_arg(ap, unsigned);
        auto* p = va_arg(ap, io_uring_params*);
        if (setup_error) { errno = setup_error; rc = -1; }
        else { rc = syscall(op, n, p); if (rc >= 0) last_params = *p; }
    } else {
        assert(op == __NR_io_uring_enter);
        ++enter_calls;
        const int fd = va_arg(ap, int);
        unsigned n = va_arg(ap, unsigned);
        const int min = va_arg(ap, int), flags = va_arg(ap, int);
        // Interrupt once before submission, then periodically submit only one SQE.
        if (inject_interrupt) { inject_interrupt = false; errno = EINTR; rc = -1; }
        else if (inject_enter_error) {
            inject_enter_error = false;
            // Model a fatal enter after one accepted read, with other SQEs pending.
            assert(n > 1 && syscall(op, fd, 1u, 0u, 0u, nullptr, 0) == 1);
            errno = EIO; rc = -1;
        } else {
            if (inject_partial && n > 1) n = 1;
            rc = syscall(op, fd, n, min, flags, nullptr, 0);
            if (rc >= 0 && inject_cqe_eintr) {
                // Test-only alias of the kernel CQ: make one completed read report
                // EINTR. The production reader must retry its unchanged task/slot.
                const auto& o=last_params.cq_off;
                const size_t len=o.cqes+last_params.cq_entries*sizeof(io_uring_cqe);
                auto* cq=(unsigned char*)mmap(nullptr,len,PROT_READ|PROT_WRITE,MAP_SHARED,fd,IORING_OFF_CQ_RING);
                assert(cq!=MAP_FAILED);
                const unsigned h=__atomic_load_n((unsigned*)(cq+o.head),__ATOMIC_ACQUIRE);
                const unsigned t=__atomic_load_n((unsigned*)(cq+o.tail),__ATOMIC_ACQUIRE);
                if(h!=t) {
                    auto* entries=(io_uring_cqe*)(cq+o.cqes);
                    entries[h & *(unsigned*)(cq+o.ring_mask)].res=-EINTR;
                    inject_cqe_eintr=false;
                }
                assert(!munmap(cq,len));
            }
        }
    }
    va_end(ap); return rc;
}
#define syscall test_syscall
#include "../rowstore.h"
#undef syscall
using namespace ds41::engram;

struct Temp
{
    char path[64] = "/tmp/gr8-uring-XXXXXX";
    int fd = mkstemp(path);
    Temp() { assert(fd >= 0); }
    ~Temp() { close(fd); unlink(path); }
};
static uint8_t value(uint32_t row, unsigned col)
{ return (uint8_t)((row * 31u) ^ (row >> 8) ^ (col * 17u)); }
static void fill(Temp& f, size_t rows)
{
    std::vector<uint8_t> b(16384 * kRowBytes);
    assert(!ftruncate(f.fd, 0));
    for (size_t off = 0; off < rows; off += 16384) {
        const size_t n = std::min<size_t>(16384, rows - off);
        for (size_t i = 0; i < n; ++i) for (unsigned j = 0; j < kRowBytes; ++j) b[i*kRowBytes+j] = value(off+i,j);
        assert(pwrite(f.fd, b.data(), n*kRowBytes, off*kRowBytes) == (ssize_t)(n*kRowBytes));
    }
    assert(!fsync(f.fd));
}
static StoreOpts options(int qd)
{
    StoreOpts o; o.io_threads = 4; o.io_uring = qd; o.cache_bytes = 32ull << 20; return o;
}
static void open_store(RowStore& s, const TableFile& f, int qd, bool require)
{
    std::string e;
    if (!s.open({f}, options(qd), &e)) { fprintf(stderr, "open: %s\n", e.c_str()); abort(); }
    assert(s.direct(0));  // never pass by silently testing buffered I/O
    if (require && qd) assert(s.using_uring());
    if (!qd || setup_error) assert(!s.using_uring());
}
static void stats_equal(const FetchStats& a, const FetchStats& b)
{
    assert(a.req == b.req && a.foreign == b.foreign && a.hits == b.hits);
    assert(a.miss_req == b.miss_req && a.miss_uniq == b.miss_uniq && a.blocks == b.blocks);
    assert(a.bytes == b.bytes && a.err == b.err);
}
static bool owned(const TableFile& f, uint32_t id)
{ return f.columns.enabled ? f.columns.at(id) >= 0 : id >= f.first && int64_t(id) < f.first+f.valid; }
static void parity(const TableFile& f, size_t total, int qd, bool require)
{
    for (size_t n : {size_t(1),size_t(16),size_t(17),size_t(4096),size_t(25508)}) {
        RowStore pread, uring;
        open_store(pread,f,0,false); open_store(uring,f,qd,require);
        std::mt19937 rng(617);
        std::vector<uint32_t> ids(n);
        for (auto& id : ids) id = rng()%total;
        // Adjacent rows, duplicates, run edges and guaranteed foreign IDs.
        const uint32_t special[] = {uint32_t(f.first),uint32_t(f.first+1),uint32_t(f.first),
                                   uint32_t(total-1),uint32_t(total),0xffffffffu,16,17};
        for (size_t i=0;i<std::min(n,sizeof special/sizeof special[0]);++i) ids[i]=special[i];
        for (int round=0;round<2;++round) {
            std::vector<uint8_t> a(n*kRowBytes,0xa5),b(a);
            FetchStats x,y;
            assert(!pread.fetch(0,ids.data(),n,a.data(),&x));
            assert(!uring.fetch(0,ids.data(),n,b.data(),&y));
            assert(!memcmp(a.data(),b.data(),a.size())); stats_equal(x,y);
            for(size_t i=0;i<n;++i) for(unsigned j=0;j<kRowBytes;++j)
                assert(b[i*kRowBytes+j]==(owned(f,ids[i])?value(ids[i],j):0xa5));
            if(round) assert(!x.miss_req && !y.miss_req);
        }
        printf("PASS parity qd=%d n=%zu col=%d actual=%s\n",qd,n,f.columns.enabled,uring.using_uring()?"uring":"fallback");
    }
    // Cold null-dst warm, then shared store calls from a different fetch thread.
    RowStore warm; open_store(warm,f,qd,require);
    std::vector<uint32_t> ids;
    for(uint32_t i=0;i<total;i+=97) ids.push_back(i);
    assert(!warm.fetch(0,ids.data(),ids.size(),nullptr));
    std::thread other([&]{
        FetchStats st; std::vector<uint8_t> out(ids.size()*kRowBytes,0xa5);
        assert(!warm.fetch(0,ids.data(),ids.size(),out.data(),&st)); assert(!st.miss_req);
        for(size_t i=0;i<ids.size();++i) for(unsigned j=0;j<kRowBytes;++j)
            assert(out[i*kRowBytes+j]==(owned(f,ids[i])?value(ids[i],j):0xa5));
        uint32_t fresh = f.first+2;
        assert(!warm.fetch(0,&fresh,1,out.data()));
        for(unsigned j=0;j<kRowBytes;++j) assert(out[j]==value(fresh,j));
    }); other.join();
    FetchStats z; assert(!warm.fetch(0,nullptr,0,nullptr,&z)); assert(z.req==0);
}
static void short_read(int qd, bool require)
{
    Temp f; fill(f,65);
    TableFile file{f.path,0,65,{}};
    RowStore pread,uring; open_store(pread,file,0,false); open_store(uring,file,qd,require);
    uint32_t tail=64; std::vector<uint8_t>a(256),b(256);
    // A short *block* containing a complete requested row is legal, exactly as pread.
    assert(!pread.fetch(0,&tail,1,a.data())); assert(!uring.fetch(0,&tail,1,b.data())); assert(a==b);
    // Truncate after open; the requested row in the last block is now incomplete.
    assert(!ftruncate(f.fd,33*256+128));
    uint32_t ids[]={0,33}; a.assign(512,0xa5);b=a; FetchStats x,y;
    assert(pread.fetch(0,ids,2,a.data(),&x)==EIO); assert(uring.fetch(0,ids,2,b.data(),&y)==EIO);
    assert(a==b && a==std::vector<uint8_t>(512,0xa5)); stats_equal(x,y);
    // No rows from a failed fetch enter the cache; restore then retry on the same ring.
    fill(f,65);
    assert(!pread.fetch(0,ids,2,a.data(),&x)); assert(!uring.fetch(0,ids,2,b.data(),&y));
    assert(a==b && x.hits==0 && y.hits==0); stats_equal(x,y);
    printf("PASS short read qd=%d (legal tail, EIO, no partial dst/cache, retry)\n",qd);
}
int main(int argc,char** argv)
{
    bool timing=false,require=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"--time")) timing=true; else if(!strcmp(argv[i],"--require-uring"))require=true; else return 2; }
    Temp f; const size_t rows=65541; fill(f,rows);
    // The disabled path makes no setup syscall.
    const TableFile full{f.path,0,(int64_t)rows,{}};
    { RowStore off;open_store(off,full,0,false);assert(setup_calls==0); }
    RowStore probe;open_store(probe,full,128,require);
    printf("io_uring availability: %s\n",probe.using_uring()?"real kernel ring":"UNAVAILABLE; real-ring coverage skipped");
    // Foreign IDs in both contiguous and column-sharded layouts; split unaligned runs.
    TableFile contiguous{"",16,(int64_t)rows-32,{{f.path,16,1001,16*256},{f.path,1017,(int64_t)rows-1033,1017*256}}, {}};
    HashParams hp{};
    for(int e=0;e<kNE;++e) for(int j=0;j<kIds;++j) { hp.prime[e][j]=2701+j; hp.offset[e][j]=j?hp.offset[e][j-1]+hp.prime[e][j-1]:0; }
    const auto col=ColShard::make(hp,0,0,4);
    TableFile columns{"",0,col.rows,{},col};
    for(int j=0;j<kIds;j+=4) columns.runs.push_back({f.path,hp.offset[0][j],hp.prime[0][j],uint64_t(hp.offset[0][j])*256});
    for(int qd : {1,3,17,128}) {
        parity(contiguous,rows,qd,require);parity(columns,rows,qd,require);short_read(qd,require);
    }
    if(probe.using_uring()) {
        inject_interrupt=true;inject_partial=true;inject_cqe_eintr=true;
        parity(contiguous,rows,17,true);
        assert(!inject_interrupt && !inject_cqe_eintr);inject_partial=false;
        puts("PASS enter/CQE EINTR and partial submission via test-local syscall adapter");
        UringReader invalid;assert(!invalid.start(3,4096,4096));
        int failures=0;
        assert(!invalid.run(20,[](size_t,io_uring_sqe& e){e.fd=-1;},[&](size_t,const uint8_t*,int r){assert(r==-EBADF);++failures;}));
        assert(failures==20);puts("PASS negative CQE errors drained over multiple queue wraps");
        RowStore broken;open_store(broken,full,17,true);
        uint32_t ids[]={0,32,64};std::vector<uint8_t> dst(3*256,0xa5);
        inject_enter_error=true;
        assert(broken.fetch(0,ids,3,dst.data())==EIO && !inject_enter_error);
        assert(dst==std::vector<uint8_t>(3*256,0xa5));
        assert(broken.fetch(0,ids,3,dst.data())==EIO);
        puts("PASS fatal enter after accepted I/O: drain before teardown, no partial dst, fail-closed retry");
    }
    for(int e : {EPERM,ENOSYS}) {
        setup_error=e;parity(contiguous,rows,128,false);short_read(128,false);setup_error=0;
        printf("PASS forced setup errno=%d -> pread fallback\n",e);
    }
    if(timing) {
        Temp big; const size_t nrows=4u<<20;fill(big,nrows);
        TableFile file{big.path,0,(int64_t)nrows,{}};
        std::mt19937 rng(2026);std::set<uint32_t> unique;
        while(unique.size()<25508) unique.insert(rng()%nrows);
        std::vector<uint32_t> ids(unique.begin(),unique.end());
        std::vector<uint8_t> a(ids.size()*256),b(a);
        for(int qd : {0,128}) {
            RowStore s; std::string e;auto o=options(qd);o.io_threads=64;
            assert(s.open({file},o,&e));assert(s.direct(0));FetchStats st;
            assert(!s.fetch(0,ids.data(),ids.size(),qd?b.data():a.data(),&st));
            printf("local-disk requested_qd=%d actual=%s file_bytes=%zu rows=%zu blocks=%u io_ms=%.3f total_ms=%.3f\n",qd,s.using_uring()?"uring":"pread",nrows*256,ids.size(),st.blocks,st.io_us/1000,st.total_us/1000);
        }
        assert(a==b);
    }
    printf("PASS uring rowstore (setup=%d enter=%d)\n",setup_calls,enter_calls);
}
