// CPU-only strict G6_PF_ROUTER_SPLIT parse: unset/0 = off, 1 = on, every other spelling fails.
// g++ -std=c++17 -O1 -o /tmp/t_rs_env engine/tests/test_flash_route_split_env.cpp && /tmp/t_rs_env
#include "../runtime/envparse.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
int main() {
    constexpr const char* name = "G6_PF_ROUTER_SPLIT";
    std::string err;
    unsetenv(name);
    assert(env_choice(name, 1, 0, &err) == 0);
    setenv(name, "0", 1); assert(env_choice(name, 1, 0, &err) == 0);
    setenv(name, "1", 1); assert(env_choice(name, 1, 0, &err) == 1);
    int bad = 0;
    for (const char* s : {"", "2", "01", "-1", "1junk", " 1", "true"}) {
        setenv(name, s, 1);
        assert(env_choice(name, 1, 0, &err) == -1 && err.find(name) != std::string::npos);
        ++bad;
    }
    unsetenv(name);
    std::printf("test_flash_route_split_env: off/on and %d bad values verified\n", bad);
}
