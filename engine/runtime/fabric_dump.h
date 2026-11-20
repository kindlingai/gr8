// Every field of a g6::FabricConfig on one line: g6m4's GR8_FABRIC_DUMP prints one per fabric, and
// s2/tests/fabric_plan_vectors_test.cpp hashes the same lines. Host only.
#pragma once
#include <string>
#include <vector>
#include "../comms/fabric.h"

namespace g6rt
{
inline std::string describe_fabric(const char* name, const g6::FabricConfig& c)
{
    std::string s = std::string("fabric ") + name;
    auto num = [&](const char* k, long long v) { s += ' '; s += k; s += '='; s += std::to_string(v); };
    auto strs = [&](const char* k, const std::vector<std::string>& v) {
        s += ' '; s += k; s += '=';
        for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    };
    auto list = [&](const char* k, const auto& v) {
        s += ' '; s += k; s += '=';
        for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string((unsigned long long)v[i]);
    };
    num("rank", c.rank); strs("ips", c.ips); strs("ips2", c.ips2); num("stripe_min", (long long)c.stripe_min);
    num("slot_bytes", (long long)c.slot_bytes); num("run1", c.run1); num("run2", c.run2);
    num("slot_bytes1", (long long)c.slot_bytes1); num("slot_bytes2", (long long)c.slot_bytes2); num("run3", c.run3);
    num("slot_bytes3", (long long)c.slot_bytes3); num("pool", c.pool); num("sge2", c.sge2); num("compact", c.compact);
    num("ncoll", c.ncoll); num("port", c.port); num("proxy_cpu", c.proxy_cpu); list("vcap", c.vcap); list("vdst_k", c.vdst_k);
    num("bulk_bytes", (long long)c.bulk_bytes); num("bulk_flags", c.bulk_flags); num("xtrace", c.xtrace); num("ring", c.ring);
    strs("ring_devs", c.ring_devs); num("relay_cpu", c.relay_cpu);
    return s;
}
}   // namespace g6rt
