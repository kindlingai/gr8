#pragma once
// Host memory after load (#4). On GB10 host memory is GPU memory: every MiB the host keeps is a MiB the KV pool and the
// rest of the boot cannot have.
//
// The loaders already keep the pack files out of the page cache (O_DIRECT, or POSIX_FADV_DONTNEED behind buffered reads:
// loader/devimage.cpp, loader/g6pack.cpp, loader-rs). The big host buffers are cudaHostAlloc staging or mmap-backed and
// go back on free. What stays behind is fragments of the engine's own heap: manifests, plans, name tables and vectors
// freed between live allocations sit in glibc's arenas, counted in the process's RSS, until malloc_trim hands their
// whole pages back. Expect tens to hundreds of MiB, not GiB. host_reclaim() trims and logs MemAvailable, the file
// cache (Cached - Shmem) and the process's RSS before and after, so a boot log shows where the memory went.
//
// Nothing computed changes: malloc_trim only returns free heap pages to the kernel. Without glibc it only logs.
#include <cstdio>
#include <cstring>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace g6rt
{

struct HostMem
{
    long avail_kb = -1;   // MemAvailable (/proc/meminfo)
    long cached_kb = -1;  // Cached - Shmem (/proc/meminfo): the file page cache
    long rss_kb = -1;     // VmRSS (/proc/self/status)
};

// One "<key>: <n> kB" field out of a /proc text; -1 if it is not there.
inline long proc_kb(const char* text, const char* key)
{
    const size_t n = strlen(key);
    for (const char* p = text; p && *p;)
    {
        if (!strncmp(p, key, n) && p[n] == ':')
        {
            long v;
            if (sscanf(p + n + 1, " %ld", &v) == 1) return v;
            return -1;
        }
        p = strchr(p, '\n');
        if (p) ++p;
    }
    return -1;
}

inline bool proc_read(const char* path, char* buf, size_t cap)
{
    FILE* f = fopen(path, "r");
    if (!f) return false;
    const size_t got = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[got] = 0;
    return true;
}

inline HostMem host_mem()
{
    HostMem m;
    char buf[8192];                       // on the stack: callers may be on any thread
    if (proc_read("/proc/meminfo", buf, sizeof buf))
    {
        m.avail_kb = proc_kb(buf, "MemAvailable");
        const long c = proc_kb(buf, "Cached"), sh = proc_kb(buf, "Shmem");
        m.cached_kb = c < 0 ? -1 : sh < 0 ? c : c > sh ? c - sh : 0;
    }
    if (proc_read("/proc/self/status", buf, sizeof buf)) m.rss_kb = proc_kb(buf, "VmRSS");
    return m;
}

inline double kb_gib(long kb) { return kb < 0 ? -1.0 : kb / 1048576.0; }

inline void host_mem_log(int rank, const char* when)
{
    const HostMem m = host_mem();
    fprintf(stderr, "# rank %d: host %s: MemAvailable %.2f GiB, file cache %.2f GiB, engine RSS %.2f GiB\n", rank, when,
            kb_gib(m.avail_kb), kb_gib(m.cached_kb), kb_gib(m.rss_kb));
}

// malloc_trim(0), with the numbers before and after. Returns the RSS it gave back, in kB (0 if none or unknown).
inline long host_reclaim(int rank, const char* when)
{
    const HostMem a = host_mem();
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
    const HostMem b = host_mem();
    const long freed = a.rss_kb >= 0 && b.rss_kb >= 0 && a.rss_kb > b.rss_kb ? a.rss_kb - b.rss_kb : 0;
    fprintf(stderr, "# rank %d: host %s: malloc_trim gave back %.0f MiB; MemAvailable %.2f -> %.2f GiB, file cache %.2f GiB, "
            "engine RSS %.2f -> %.2f GiB\n", rank, when, freed / 1024.0, kb_gib(a.avail_kb), kb_gib(b.avail_kb),
            kb_gib(b.cached_kb), kb_gib(a.rss_kb), kb_gib(b.rss_kb));
    return freed;
}

}  // namespace g6rt
