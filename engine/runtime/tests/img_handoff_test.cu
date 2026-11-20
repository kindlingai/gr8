// Run on a GB10: can a weight image outlive the process that made it? (Phase B probe P0, gr8-private#156)
// Three processes over a Unix socket, the allocations' file descriptors passed with SCM_RIGHTS:
//   export SOCK GIB   cuMemCreate chunks (exportable as POSIX fds), mapped and filled, each exported and sent to the
//                     holder; then it exits without unmapping
//   hold SOCK         no CUDA: takes the exporter's fds, hands them to the importer, exits when the importer is done
//   import SOCK GIB   imports every fd, maps them into one range, verifies every word, prints the import + map time
// With no role (or `run GIB`), the test runs all three and checks MemAvailable around them. Five PASS lines, one per
// condition: export succeeds; the memory stays allocated after the exporter exits; 0 bad words; import + map under
// 2 s; the memory comes back once importer and holder exit. Exit 0 = all PASS, 1 = a FAIL, 2 = CUDA error,
// 0 with SKIP = no VMM or no fd export on this driver.
//   TEST=img_handoff_test scripts/c1_tests.sh [bin_dir] [GIB]      (64 by default)
#include "../../vmm.h"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

static void check(cudaError_t rc)
{
    if (rc != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc)); std::exit(2); }
}
// the fill and verify kernels of dispram_gpu_test.cu
__host__ __device__ static uint64_t pattern(size_t w) { return 0x9e3779b97f4a7c15ULL ^ (uint64_t(w) * 0xd6e8feb86659fd93ULL); }
__device__ static uint64_t load_bypass_l2(const uint64_t* p)
{
    unsigned long long v = 0;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return static_cast<uint64_t>(v);
}
__global__ void fill(uint64_t* p, size_t words)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += size_t(blockDim.x) * gridDim.x) p[i] = pattern(i);
}
__global__ void verify(const uint64_t* p, size_t words, unsigned long long* errors)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += size_t(blockDim.x) * gridDim.x)
        if (load_bypass_l2(p + i) != pattern(i)) atomicAdd(errors, 1ULL);
}

// the driver's fd export and import, fetched as vmm.h fetches the rest (no -lcuda)
static CUresult (*export_fd)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) = nullptr;
static CUresult (*import_fd)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType) = nullptr;
static bool share_api()
{
    auto get = [](const char* s, auto& fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        if (cudaGetDriverEntryPointByVersion(s, &p, 12000, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess) return false;
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
        return true;
    };
    return g6vmm::api().ok && get("cuMemExportToShareableHandle", export_fd) && get("cuMemImportFromShareableHandle", import_fd);
}

static CUmemAllocationProp share_prop(int dev)
{
    CUmemAllocationProp p = g6vmm::prop(dev);
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return p;
}

// the chunk size: the recommended granularity, larger only when this process may not hold one fd per chunk
static size_t chunk_for(size_t bytes, size_t gmin, size_t grec)
{
    rlimit r{};
    getrlimit(RLIMIT_NOFILE, &r);
    r.rlim_cur = r.rlim_max;
    setrlimit(RLIMIT_NOFILE, &r);
    getrlimit(RLIMIT_NOFILE, &r);
    const size_t room = r.rlim_cur > 128 ? r.rlim_cur - 64 : 64;
    size_t chunk = grec;
    if (bytes / chunk > room) chunk = (bytes / room + gmin) / gmin * gmin;
    return chunk;
}

// ---- the socket: a header line, then the fds, at most FDS_PER_MSG to a message --------------------------------------
static const size_t FDS_PER_MSG = 250;   // under the kernel's SCM_MAX_FD (253)
static bool send_all(int s, const void* b, size_t n)
{
    const char* p = static_cast<const char*>(b);
    while (n) { ssize_t k = send(s, p, n, MSG_NOSIGNAL); if (k <= 0) return false; p += k; n -= size_t(k); }
    return true;
}
static bool recv_all(int s, void* b, size_t n)
{
    char* p = static_cast<char*>(b);
    while (n) { ssize_t k = recv(s, p, n, 0); if (k <= 0) return false; p += k; n -= size_t(k); }
    return true;
}
struct Header { uint64_t count, chunk; };
static bool send_fds(int s, const std::vector<int>& fds, size_t chunk)
{
    Header h{fds.size(), chunk};
    if (!send_all(s, &h, sizeof(h))) return false;
    for (size_t at = 0; at < fds.size(); at += FDS_PER_MSG)
    {
        const size_t n = std::min(FDS_PER_MSG, fds.size() - at);
        char byte = 'f';
        iovec io{&byte, 1};
        std::vector<char> control(CMSG_SPACE(sizeof(int) * n));
        msghdr m{};
        m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = control.data(); m.msg_controllen = control.size();
        cmsghdr* c = CMSG_FIRSTHDR(&m);
        c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int) * n);
        memcpy(CMSG_DATA(c), fds.data() + at, sizeof(int) * n);
        if (sendmsg(s, &m, MSG_NOSIGNAL) != 1) return false;
    }
    return true;
}
static bool recv_fds(int s, std::vector<int>& fds, size_t& chunk)
{
    Header h{};
    if (!recv_all(s, &h, sizeof(h))) return false;
    chunk = h.chunk;
    while (fds.size() < h.count)
    {
        const size_t n = std::min<size_t>(FDS_PER_MSG, h.count - fds.size());
        char byte = 0;
        iovec io{&byte, 1};
        std::vector<char> control(CMSG_SPACE(sizeof(int) * n));
        msghdr m{};
        m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = control.data(); m.msg_controllen = control.size();
        if (recvmsg(s, &m, 0) != 1 || (m.msg_flags & MSG_CTRUNC)) return false;
        cmsghdr* c = CMSG_FIRSTHDR(&m);
        if (!c || c->cmsg_type != SCM_RIGHTS || c->cmsg_len != CMSG_LEN(sizeof(int) * n)) return false;
        const size_t at = fds.size();
        fds.resize(at + n);
        memcpy(fds.data() + at, CMSG_DATA(c), sizeof(int) * n);
    }
    return true;
}
static sockaddr_un addr_of(const char* path)
{
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    strncpy(a.sun_path, path, sizeof(a.sun_path) - 1);
    return a;
}
static int connect_to(const char* path)
{
    const sockaddr_un a = addr_of(path);
    for (int i = 0; i < 100; ++i)
    {
        int s = socket(AF_UNIX, SOCK_STREAM, 0);
        if (connect(s, reinterpret_cast<const sockaddr*>(&a), sizeof(a)) == 0) return s;
        close(s);
        usleep(100000);
    }
    return -1;
}

// ---- the roles ------------------------------------------------------------------------------------------------------
static int role_export(const char* sock, double gib)
{
    check(cudaSetDevice(0));
    check(cudaFree(nullptr));
    if (!share_api()) { puts("SKIP img handoff: no VMM or no fd export entry points"); return 77; }
    size_t gmin = 0, grec = 0;
    g6vmm::granularity(0, gmin, grec);
    if (!gmin) { puts("SKIP img handoff: no VMM granularity"); return 77; }
    const size_t want = size_t(gib * double(1ull << 30));
    const size_t chunk = chunk_for(want, gmin, grec), n = (want + chunk - 1) / chunk, total = n * chunk;
    const auto& a = g6vmm::api();
    const CUmemAllocationProp p = share_prop(0);
    CUdeviceptr base = 0;
    if (CUresult r = a.reserve(&base, total, chunk, 0, 0)) { printf("FAIL export: cuMemAddressReserve %d\n", int(r)); return 1; }
    std::vector<int> fds;
    for (size_t i = 0; i < n; ++i)
    {
        CUmemGenericAllocationHandle h{};
        CUresult r = a.create(&h, chunk, &p, 0);
        if (!r) r = a.map(base + i * chunk, chunk, 0, h, 0);
        int fd = -1;
        CUresult e = r ? r : export_fd(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        if (e)
        {
            printf("FAIL export: chunk %zu of %zu: %s %d\n", i, n, r ? "cuMemCreate/cuMemMap" : "cuMemExportToShareableHandle", int(e));
            return 1;
        }
        a.release(h);   // the mapping and the fd keep the memory
        fds.push_back(fd);
    }
    CUmemAccessDesc d{};
    d.location = p.location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (CUresult r = a.access(base, total, &d, 1)) { printf("FAIL export: cuMemSetAccess %d\n", int(r)); return 1; }
    fill<<<4096, 256>>>(reinterpret_cast<uint64_t*>(base), total / 8);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    int s = connect_to(sock);
    if (s < 0 || !send_fds(s, fds, chunk)) { printf("FAIL export: could not hand the fds to the holder (%s)\n", strerror(errno)); return 1; }
    char ack = 0;
    recv_all(s, &ack, 1);   // the holder has them
    printf("PASS export: %zu chunks of %zu MiB (%.1f GiB) created, filled and exported as POSIX fds (CUDA_SUCCESS)\n", n,
           chunk >> 20, double(total) / double(1ull << 30));
    fflush(stdout);
    _exit(0);   // no unmap, no release, no context teardown by hand: what a process that ends leaves
}

static int role_hold(const char* sock)
{
    rlimit r{};
    getrlimit(RLIMIT_NOFILE, &r);
    r.rlim_cur = r.rlim_max;
    setrlimit(RLIMIT_NOFILE, &r);
    unlink(sock);
    int l = socket(AF_UNIX, SOCK_STREAM, 0);
    const sockaddr_un a = addr_of(sock);
    if (bind(l, reinterpret_cast<const sockaddr*>(&a), sizeof(a)) || listen(l, 2)) { perror("hold: bind"); return 1; }
    std::vector<int> fds;
    size_t chunk = 0;
    int e = accept(l, nullptr, nullptr);
    if (e < 0 || !recv_fds(e, fds, chunk)) { puts("FAIL hold: no fds from the exporter"); return 1; }
    char ack = 'k';
    send_all(e, &ack, 1);
    close(e);
    int i = accept(l, nullptr, nullptr);
    if (i < 0 || !send_fds(i, fds, chunk)) { puts("FAIL hold: could not hand the fds to the importer"); return 1; }
    for (int fd : fds) close(fd);
    char done = 0;
    recv_all(i, &done, 1);   // the importer has imported, verified and released them
    close(i);
    close(l);
    unlink(sock);
    return 0;
}

static int role_import(const char* sock)
{
    check(cudaSetDevice(0));
    check(cudaFree(nullptr));
    if (!share_api()) { puts("SKIP img handoff: no VMM or no fd import entry points"); return 77; }
    int s = connect_to(sock);
    std::vector<int> fds;
    size_t chunk = 0;
    if (s < 0 || !recv_fds(s, fds, chunk)) { puts("FAIL import: no fds from the holder"); return 1; }
    const auto& a = g6vmm::api();
    const size_t total = fds.size() * chunk;
    const auto t0 = std::chrono::steady_clock::now();
    CUdeviceptr base = 0;
    if (CUresult r = a.reserve(&base, total, chunk, 0, 0)) { printf("FAIL import: cuMemAddressReserve %d\n", int(r)); return 1; }
    for (size_t i = 0; i < fds.size(); ++i)
    {
        CUmemGenericAllocationHandle h{};
        CUresult r = import_fd(&h, reinterpret_cast<void*>(static_cast<uintptr_t>(fds[i])), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        if (!r) r = a.map(base + i * chunk, chunk, 0, h, 0);
        if (r) { printf("FAIL import: chunk %zu: %s %d\n", i, "cuMemImportFromShareableHandle/cuMemMap", int(r)); return 1; }
        a.release(h);
        close(fds[i]);
    }
    CUmemAccessDesc d{};
    d.location = g6vmm::prop(0).location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (CUresult r = a.access(base, total, &d, 1)) { printf("FAIL import: cuMemSetAccess %d\n", int(r)); return 1; }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long* bad = nullptr;
    check(cudaMalloc(&bad, sizeof(*bad)));
    check(cudaMemset(bad, 0, sizeof(*bad)));
    verify<<<4096, 256>>>(reinterpret_cast<const uint64_t*>(base), total / 8, bad);
    check(cudaGetLastError());
    unsigned long long n = 0;
    check(cudaMemcpy(&n, bad, sizeof(n), cudaMemcpyDeviceToHost));
    int rc = 0;
    if (n) { printf("FAIL import: %llu of %zu words differ\n", n, total / 8); rc = 1; }
    else printf("PASS import: every word of %.1f GiB as the exporter wrote it (0 bad words)\n", double(total) / double(1ull << 30));
    if (secs < 2.0) printf("PASS import + map of %zu chunks in %.3f s (under 2 s)\n", fds.size(), secs);
    else { printf("FAIL import + map of %zu chunks in %.3f s (2 s or more)\n", fds.size(), secs); rc = 1; }
    check(cudaFree(bad));
    g6vmm::unmap(base, total, chunk);
    a.free_va(base, total);
    char done = 'd';
    send_all(s, &done, 1);
    close(s);
    return rc;
}

// ---- the run: the three roles as children, MemAvailable around them ------------------------------------------------
static double mem_available_gib()
{
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    double kb = -1;
    while (fgets(line, sizeof(line), f))
        if (!strncmp(line, "MemAvailable:", 13)) { kb = atof(line + 13); break; }
    fclose(f);
    return kb < 0 ? -1 : kb / (1 << 20);
}
static pid_t spawn(char* self, std::vector<std::string> args)
{
    pid_t p = fork();
    if (p) return p;
    std::vector<char*> v{self};
    for (auto& s : args) v.push_back(const_cast<char*>(s.c_str()));
    v.push_back(nullptr);
    execv(self, v.data());
    _exit(99);
}
static int wait_rc(pid_t p)
{
    int st = 0;
    if (waitpid(p, &st, 0) != p || !WIFEXITED(st)) return 98;
    return WEXITSTATUS(st);
}
static int run(char* self, double gib)
{
    char sock[] = "/tmp/gr8-img-handoff-XXXXXX";
    int t = mkstemp(sock);
    close(t);
    unlink(sock);
    const std::string g = std::to_string(gib);
    const double m0 = mem_available_gib();
    pid_t holder = spawn(self, {"hold", sock});
    int rc = wait_rc(spawn(self, {"export", sock, g}));
    if (rc == 77) { kill(holder, SIGKILL); waitpid(holder, nullptr, 0); unlink(sock); return 0; }
    if (rc) { kill(holder, SIGKILL); waitpid(holder, nullptr, 0); unlink(sock); return rc == 2 ? 2 : 1; }
    sleep(1);
    const double m1 = mem_available_gib();
    int fails = 0;
    if (m0 - m1 >= 0.9 * gib) printf("PASS the memory outlives the exporter: MemAvailable %.1f -> %.1f GiB after it exited\n", m0, m1);
    else { printf("FAIL the memory went with the exporter: MemAvailable %.1f -> %.1f GiB (want %.1f GiB less)\n", m0, m1, gib); ++fails; }
    fflush(stdout);
    rc = wait_rc(spawn(self, {"import", sock}));
    if (rc == 2) return 2;
    if (rc) ++fails;
    if (wait_rc(holder)) ++fails;
    sleep(1);
    const double m2 = mem_available_gib();
    if (m0 - m2 < 0.1 * gib) printf("PASS the memory comes back once importer and holder exit: MemAvailable %.1f GiB (%.1f before)\n", m2, m0);
    else { printf("FAIL the memory stays held: MemAvailable %.1f GiB (%.1f before)\n", m2, m0); ++fails; }
    printf("img_handoff_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::string role = argc > 1 ? argv[1] : "run";
    if (role == "export" && argc > 3) return role_export(argv[2], atof(argv[3]));
    if (role == "hold" && argc > 2) return role_hold(argv[2]);
    if (role == "import" && argc > 2) return role_import(argv[2]);
    if (role == "run") return run(argv[0], argc > 2 ? atof(argv[2]) : 64.0);
    if (argc > 1 && atof(argv[1]) > 0) return run(argv[0], atof(argv[1]));   // c1_tests.sh passes GIB alone
    fprintf(stderr, "usage: %s [run] [GIB] | export SOCK GIB | hold SOCK | import SOCK\n", argv[0]);
    return 2;
}
