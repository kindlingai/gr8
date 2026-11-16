// Host-only test of the same diagnostic used by Ds41Pool::on_match under ck_mu.
#include "s2/ds41_park_miss.h"
#include <array>
#include <cassert>
#include <string>
using namespace g6s2;
struct Parked { bool on = false; uint32_t pos = 0; std::vector<int> tokens; std::array<uint8_t, 2> ns{}; };
static std::string contents(FILE* f) {
    std::fflush(f); std::rewind(f); std::string s; char b[256];
    while (std::fgets(b, sizeof b, f)) s += b;
    std::fseek(f, 0, SEEK_END); return s;
}
int main() {
    FILE* f = std::tmpfile(); assert(f);
    std::optional<Ds41ParkMissKey> last_miss;
    std::array<uint8_t, 2> ns{};
    std::array<Parked, 32> parked{};
    std::vector<int> prompt{1,2,3,4,9};
    auto miss = [&](int q=-1, int limit=3, bool park=true) {
        ds41_park_miss(park,q,parked,8,ns,prompt,limit,0,last_miss,f);
    };
    miss(); assert(contents(f).empty()); // nothing parked
    parked[0] = {true,4,{1,2,3,4},ns};
    parked[1] = {true,5,{1,2,3,4,8},ns};
    parked[2] = parked[1]; // same lcp/pos: lowest slot wins
    parked[3] = {true,100,{1,2},ns}; // higher pos cannot beat higher lcp
    parked[4] = {true,5,prompt,{1,0}}; // wrong namespace excluded
    parked[5] = {false,5,prompt,ns};
    parked[8] = {true,5,prompt,ns}; // outside the configured slots excluded
    parked[6] = {true,0,prompt,ns};
    miss(0); miss(2); miss(-1,3,false); assert(contents(f).empty()); // resumes, other lanes parked
    miss();
    const std::string one = "# rank 0 ds41 park_miss slot 1 lcp 4 pos 5 limit 3 prompt 5\n";
    assert(contents(f)==one); // includes pos > limit
    miss(); miss(); assert(contents(f)==one); // repeated control try-fit/grant suppressed
    miss(-1,2); assert(contents(f)==one+"# rank 0 ds41 park_miss slot 1 lcp 4 pos 5 limit 2 prompt 5\n");
    prompt.push_back(10); miss(-1,2);
    assert(contents(f).find("limit 2 prompt 6\n")!=std::string::npos); // distinct key
    std::fclose(f);
    std::puts("ds41_park_miss_test: PASS (empty/resume/namespace/limit/ties/dedupe/distinct)");
}
