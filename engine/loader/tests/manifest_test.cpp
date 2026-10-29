// Pack::open_manifest against manifests whose numbers wrap: a negative offset, a shape whose element count overflows,
// a negative dimension. Each must be refused, so no tensor can point outside the image. And the routed-expert formats:
// each codebook gr8 decodes maps to its number, and any other expert_format is refused (never loaded as mcg).
//   g++ -std=c++20 -O2 -pthread -I engine/loader engine/loader/tests/manifest_test.cpp engine/loader/g6pack.cpp -o t && ./t
#include "g6pack.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

static int fails = 0;

static std::string manifest(const std::string& shape, const std::string& offset, const std::string& nbytes,
                            const std::string& xf = "")
{
    return "{\"format\":\"g6pack-v1\",\"rank\":0,\"align\":4096,\"pack_bytes\":16384,\"tensors\":[{\"name\":\"t\","
           "\"dtype\":\"bf16\",\"shape\":" + shape + ",\"offset\":" + offset + ",\"nbytes\":" + nbytes +
           ",\"sha256\":\"" + std::string(64, '0') + "\"" + (xf.empty() ? "" : ",\"expert_format\":\"" + xf + "\"") + "}]}";
}

// want_fmt >= 0: the entry's expert_format number
static void expect(const char* what, const std::string& text, const char* want, int want_fmt = -1)
{
    char path[] = "/tmp/g6pack-manifest-test.XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { std::perror("mkstemp"); std::exit(2); }
    close(fd);
    std::ofstream(path) << text;
    g6::Pack p;
    std::string err;
    const bool ok = p.open_manifest(path, err);
    unlink(path);
    bool pass = want ? !ok && err.find(want) != std::string::npos : ok;
    if (ok && want_fmt >= 0) pass = pass && p.find("t") && p.find("t")->expert_format == want_fmt;
    if (!pass) fails++;
    std::printf("  %-4s %-28s %s\n", pass ? "ok" : "FAIL", what, ok ? "accepted" : err.c_str());
}

int main()
{
    expect("a plain tensor", manifest("[2,2048]", "4096", "8192"), nullptr);
    expect("offset -4096 (wraps)", manifest("[4096]", "-4096", "8192"), "negative offset");
    expect("element count overflows", manifest("[4294967296,4294967296,2]", "0", "0"), "bad shape");
    expect("a negative dimension", manifest("[-1,2]", "0", "4"), "bad shape");
    expect("runs past pack_bytes", manifest("[8192]", "4096", "16384"), "past pack_bytes");
    expect("no expert_format: TR3", manifest("[2,2048]", "4096", "8192"), nullptr, 0);
    expect("tr3", manifest("[2,2048]", "4096", "8192", "tr3"), nullptr, 0);
    expect("exlr8-e4m3 (mcg)", manifest("[2,2048]", "4096", "8192", "exlr8-e4m3"), nullptr, 1);
    expect("exlr8-e4m3-mul1", manifest("[2,2048]", "4096", "8192", "exlr8-e4m3-mul1"), nullptr, 2);
    expect("exlr8-e4m3-pmcgx", manifest("[2,2048]", "4096", "8192", "exlr8-e4m3-pmcgx"), nullptr, 3);
    expect("exlr8-e4m3-unchecked", manifest("[2,2048]", "4096", "8192", "exlr8-e4m3-unchecked"), nullptr, 4);
    expect("an unknown codebook", manifest("[2,2048]", "4096", "8192", "exlr8-e4m3-gauss"), "unknown expert_format");
    std::printf("manifest_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
