// Progress-thread CPU isolation (host only, no CUDA, no verbs, so a host test can drive it: tools/health/test_progress_cpus.cpp).
// The fabric proxy threads are the RDMA progress threads: they spin on a doorbell and post the writes, so a proxy that
// shares its core with another busy thread is delayed on every collective. Each proxy pins itself to one CPU
// (FabricConfig::proxy_cpu); this header does the other half: it keeps every other thread of the process off those CPUs.
#pragma once
#include <dirent.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace g6 {

// CPUs the running proxy threads are pinned to (one entry per started proxy; two proxies on one CPU are counted twice),
// and the thread ids of those proxy threads. The tids are how the isolation tells a proxy from any other thread: a
// thread that merely has a one-CPU mask on a reserved CPU is NOT a proxy and is moved.
inline std::mutex& progress_cpu_mu() { static std::mutex m; return m; }
inline std::vector<int>& progress_cpu_list() { static std::vector<int> v; return v; }
inline std::vector<int>& progress_tid_list() { static std::vector<int> v; return v; }
inline void progress_tid_add(int tid) { std::lock_guard<std::mutex> g(progress_cpu_mu()); progress_tid_list().push_back(tid); }
inline void progress_tid_remove(int tid)
{
    std::lock_guard<std::mutex> g(progress_cpu_mu());
    auto& v = progress_tid_list();
    auto it = std::find(v.begin(), v.end(), tid);
    if (it != v.end()) v.erase(it);
}
inline std::vector<int> progress_tids()
{
    std::lock_guard<std::mutex> g(progress_cpu_mu());
    return progress_tid_list();
}
inline void progress_cpu_add(int cpu) { std::lock_guard<std::mutex> g(progress_cpu_mu()); progress_cpu_list().push_back(cpu); }
inline void progress_cpu_remove(int cpu)
{
    std::lock_guard<std::mutex> g(progress_cpu_mu());
    auto& v = progress_cpu_list();
    auto it = std::find(v.begin(), v.end(), cpu);
    if (it != v.end()) v.erase(it);
}
// sorted, unique
inline std::vector<int> progress_cpus()
{
    std::lock_guard<std::mutex> g(progress_cpu_mu());
    std::vector<int> v = progress_cpu_list();
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// `mask` without the `reserved` CPUs. -> false (and `out` = `mask`) when the result would be empty or nothing changes.
inline bool mask_without(const cpu_set_t& mask, const std::vector<int>& reserved, cpu_set_t* out)
{
    cpu_set_t r = mask;
    for (int c : reserved)
        if (c >= 0 && c < CPU_SETSIZE) CPU_CLR(c, &r);
    const bool changed = !CPU_EQUAL(&r, &mask);
    if (!changed || CPU_COUNT(&r) == 0) { *out = mask; return false; }
    *out = r;
    return true;
}

// Take every thread of this process (the calling thread included) off the reserved CPUs, except the registered proxy
// threads (`proxy_tids`, exact thread ids). A thread whose mask holds only reserved CPUs (another thread pinned to a
// proxy CPU) is moved to the calling thread's mask without the reserved CPUs. New threads inherit the calling thread's
// mask, so threads created later (CUDA driver, request handlers) stay off too.
// `proc_task_dir` is /proc/self/task (a parameter for the host test). -> number of threads moved, -1 if nothing could be read.
inline int isolate_progress_cpus(const std::vector<int>& reserved, const std::vector<int>& proxy_tids,
                                 const char* proc_task_dir = "/proc/self/task")
{
    if (reserved.empty()) return 0;
    cpu_set_t self, fallback;
    CPU_ZERO(&self);
    if (sched_getaffinity(0, sizeof self, &self) != 0) return -1;
    if (!mask_without(self, reserved, &fallback)) fallback = self;       // the caller's mask without the reserved CPUs
    DIR* d = opendir(proc_task_dir);
    if (!d) return -1;
    int moved = 0;
    while (dirent* e = readdir(d))
    {
        const int tid = atoi(e->d_name);
        if (tid <= 0) continue;
        if (std::find(proxy_tids.begin(), proxy_tids.end(), tid) != proxy_tids.end()) continue;
        cpu_set_t cur, next;
        if (sched_getaffinity(tid, sizeof cur, &cur) != 0) continue;
        if (!mask_without(cur, reserved, &next))
        {
            // nothing to remove, or the thread's mask is only reserved CPUs: then it takes the caller's mask
            cpu_set_t only = cur;
            for (int c : reserved) if (c >= 0 && c < CPU_SETSIZE) CPU_CLR(c, &only);
            if (CPU_COUNT(&only) != 0 || CPU_COUNT(&fallback) == 0 || CPU_EQUAL(&fallback, &self)) continue;
            next = fallback;
        }
        if (sched_setaffinity(tid, sizeof next, &next) == 0) ++moved;
        else fprintf(stderr, "[progress] could not move thread %d off the progress CPUs\n", tid);
    }
    closedir(d);
    return moved;
}

}  // namespace g6
