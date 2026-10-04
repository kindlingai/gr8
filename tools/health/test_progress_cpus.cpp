// Host test for engine/comms/progress_cpus.h (no GPU, no verbs):
//   g++ -std=c++20 -O1 -pthread -I../../engine/comms test_progress_cpus.cpp -o /tmp/test_progress_cpus && /tmp/test_progress_cpus
// Needs a host with at least 3 CPUs in the process's affinity mask.
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

#include "progress_cpus.h"

static std::vector<int> cpus_of(const cpu_set_t& m)
{
    std::vector<int> v;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &m)) v.push_back(c);
    return v;
}

static cpu_set_t mask_of(pid_t tid)
{
    cpu_set_t m;
    CPU_ZERO(&m);
    if (sched_getaffinity(tid, sizeof m, &m) != 0) { perror("sched_getaffinity"); exit(2); }
    return m;
}

int main()
{
    const std::vector<int> all = cpus_of(mask_of(0));
    if (all.size() < 3) { printf("SKIP: need 3 CPUs, have %zu\n", all.size()); return 0; }
    const int p0 = all[all.size() - 1], p1 = all[all.size() - 2];   // the two "proxy" CPUs

    // mask_without: removes reserved, refuses an empty result and a no-op
    cpu_set_t a, out;
    CPU_ZERO(&a); CPU_SET(all[0], &a); CPU_SET(p0, &a);
    assert(g6::mask_without(a, {p0}, &out));
    assert(cpus_of(out) == std::vector<int>{all[0]});
    CPU_ZERO(&a); CPU_SET(p0, &a);
    assert(!g6::mask_without(a, {p0}, &out) && cpus_of(out) == std::vector<int>{p0});      // would be empty
    CPU_ZERO(&a); CPU_SET(all[0], &a);
    assert(!g6::mask_without(a, {p0}, &out));                                              // nothing to remove
    // registry
    g6::progress_cpu_add(p0); g6::progress_cpu_add(p1); g6::progress_cpu_add(p0);
    assert((g6::progress_cpus() == std::vector<int>{std::min(p0, p1), std::max(p0, p1)}));
    g6::progress_cpu_remove(p0);
    assert(g6::progress_cpus().size() == 2);          // p0 is still registered once
    g6::progress_cpu_remove(p0);
    assert((g6::progress_cpus() == std::vector<int>{p1}));
    g6::progress_cpu_remove(p1);
    assert(g6::progress_cpus().empty());
    g6::progress_tid_add(11); g6::progress_tid_add(12);
    g6::progress_tid_remove(11);
    assert((g6::progress_tids() == std::vector<int>{12}));
    g6::progress_tid_remove(12); g6::progress_tid_remove(12);
    assert(g6::progress_tids().empty());

    // end to end: a registered proxy thread pinned to p0, an impostor (NOT registered) also pinned to p0 alone, a thread
    // pinned to p1 together with another CPU, and a free worker; isolate with reserved {p0, p1}
    std::atomic<bool> stop{false};
    std::atomic<int> ptid{0}, itid{0}, wtid{0}, ttid{0};
    auto pinned = [&](std::atomic<int>& tid, std::vector<int> cpus) {
        return std::thread([&, cpus] {
            cpu_set_t m; CPU_ZERO(&m);
            for (int c : cpus) CPU_SET(c, &m);
            pthread_setaffinity_np(pthread_self(), sizeof m, &m);
            tid = (int)gettid();
            while (!stop) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        });
    };
    std::thread proxy = pinned(ptid, {p0});
    std::thread impostor = pinned(itid, {p0});
    std::thread two = pinned(ttid, {p1, all[0]});
    std::thread worker([&] {
        wtid = (int)gettid();
        while (!stop) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
    while (!ptid || !itid || !wtid || !ttid) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const int moved = g6::isolate_progress_cpus({p0, p1}, {ptid.load()});
    assert(moved >= 4);                                                    // main + worker + impostor + two-cpu thread
    for (pid_t t : {(pid_t)gettid(), (pid_t)wtid.load(), (pid_t)itid.load(), (pid_t)ttid.load()}) {
        const cpu_set_t m = mask_of(t);
        assert(!CPU_ISSET(p0, &m) && !CPU_ISSET(p1, &m) && CPU_COUNT(&m) >= 1);
    }
    const cpu_set_t pm = mask_of(ptid);
    assert(CPU_COUNT(&pm) == 1 && CPU_ISSET(p0, &pm));                     // the registered proxy stays pinned
    // a thread created after the call inherits the isolated mask
    std::thread late([&] { const cpu_set_t m = mask_of(0); assert(!CPU_ISSET(p0, &m) && !CPU_ISSET(p1, &m)); });
    late.join();
    assert(g6::isolate_progress_cpus({p0, p1}, {ptid.load()}) == 0);       // idempotent
    assert(g6::isolate_progress_cpus({}, {}) == 0);
    assert(g6::isolate_progress_cpus({p0}, {}, "/nonexistent") == -1);
    stop = true;
    proxy.join(); impostor.join(); two.join(); worker.join();
    printf("PASS: progress_cpus (reserved %d,%d of %zu CPUs, %d threads moved)\n", p0, p1, all.size(), moved);
    return 0;
}
