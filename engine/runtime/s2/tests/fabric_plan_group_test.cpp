// EL6 (#154): fabric_plan_apply_group over real loopback g6::FabricPool groups.
// Checks that:
// 1. Applying group {0..NR-1} (id 1, group_port[1] == eo.cfg.port) reproduces the fixed-fabric FabricPlan bit for bit
//    across every planned fabric kind (including late Gather +64, CpStage +48, DraftGather +72).
// 2. Alternating groups (id 1 -> 2 -> 3) alternate fabric ports (+0 <-> +100) and S2 ports (7341 <-> 7342) cleanly with
//    zero overlap with PoolConfig::port.
// 3. A ring fabric plan (--shape ring) is refused by fabric_plan_apply_group.
#include <cassert>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "../../fabric_plan.h"

int main() {
    using namespace g6;
    using namespace g6rt;

    const int NR = g6nr::NR;
    if (NR < 2) { printf("PASS fabric_plan_group_test (NR=%d skip)\n", NR); return 0; }

    EngineOptions eo;
    eo.cfg.rank = 0;
    for (int i = 0; i < NR; ++i) {
        eo.cfg.ips.push_back("127.0.0.1");
        eo.cfg.ips2.push_back("127.0.1." + std::to_string(i + 1));
    }
    eo.cfg.port = 18600;
    eo.s2port = 7341;
    eo.MP = 4096;
    eo.lanes = 2;
    eo.mtp = 1;
    eo.s2mode = true;
    eo.s2sess = 4;
    eo.cap = 65536;
    eo.K = 1;

    PlanInputs in;
    in.dfl_ncoll = 14;
    in.full_index = {true, false, false, false};
    in.cp_build = true;
    in.k.cp = true;
    in.k.cp_r = 4096;
    in.gather = true;
    in.draft_gather = true;

    const FabricPlan fixed = fabric_plan(eo, in);
    assert(fixed.refusal.empty() && !fixed.fabrics.empty());
    assert(fixed.find(FabKind::Gather) && fixed.find(FabKind::DraftGather));
    const std::string fixed_desc = describe(fixed);

    // Boot a real loopback FabricPool of NR threads and announce 3 alternating groups
    const int base_port = 26000 + (int)(getpid() % 1000) * 4;
    auto make_pc = [&](int idx) {
        PoolConfig pc;
        pc.index = idx;
        pc.ips = eo.cfg.ips;
        pc.ips2 = eo.cfg.ips2;
        pc.port = base_port;
        pc.group_port[1] = eo.cfg.port;
        pc.group_port[0] = eo.cfg.port + 100;
        pc.s2_port[1] = eo.s2port;
        pc.s2_port[0] = eo.s2port + 1;
        pc.timeout_s = 15;
        return pc;
    };

    std::vector<int> all(NR);
    std::iota(all.begin(), all.end(), 0);
    std::vector<std::vector<Group>> seen(NR);
    std::vector<std::thread> th;
    for (int r = 0; r < NR; ++r) {
        th.emplace_back([&, r] {
            FabricPool pool;
            pool.init(make_pc(r));
            for (int k = 0; k < 3; ++k)
                seen[r].push_back(r == 0 ? pool.announce(all) : pool.follow(15000));
            pool.close();
        });
    }
    for (auto& t : th) t.join();

    const PoolConfig pc0 = make_pc(0);
    // Group 1 (id 1): every fabric matches the fixed plan bit for bit
    {
        FabricPlan p = fixed;
        fabric_plan_apply_group(p, pc0, seen[0][0]);
        assert(describe(p) == fixed_desc);
        assert(seen[0][0].s2_port == 7341 && seen[0][0].s2_head_ip == "127.0.0.1");
    }
    // Group 2 (id 2): alternates to +100 fabric range and 7342 S2 port, preserving every port_off (incl. +64, +72)
    {
        FabricPlan p = fixed;
        fabric_plan_apply_group(p, pc0, seen[0][1]);
        assert(p.fabrics.size() == fixed.fabrics.size());
        for (size_t i = 0; i < p.fabrics.size(); ++i)
            assert(p.fabrics[i].cfg.port == fixed.fabrics[i].cfg.port + 100);
        assert(seen[0][1].s2_port == 7342);
    }
    // Group 3 (id 3): alternates back to the first range
    {
        FabricPlan p = fixed;
        fabric_plan_apply_group(p, pc0, seen[0][2]);
        assert(describe(p) == fixed_desc);
        assert(seen[0][2].s2_port == 7341);
    }
    // Ring plan is refused by fabric_plan_apply_group
    {
        EngineOptions reo = eo;
        reo.cfg.ring = true;
        reo.cfg.ips2.clear();
        FabricPlan rp = fabric_plan(reo, in);
        assert(rp.refusal.empty());
        bool threw = false;
        try { fabric_plan_apply_group(rp, pc0, seen[0][0]); } catch (const std::exception&) { threw = true; }
        assert(threw);
    }
    printf("PASS fabric_plan_group_test (NR=%d, %zu fabrics)\n", NR, fixed.fabrics.size());
    return 0;
}
