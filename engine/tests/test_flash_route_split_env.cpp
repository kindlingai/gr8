// CPU-only strict GR8_PF_ROUTER_SPLIT parse: unset/0 = off, 1 = on, every other spelling stops the process with exit 2.
// A knob is read once per process, so each value runs in a child of its own.
// cargo build --release --features capi --manifest-path engine/knobs-rs/Cargo.toml && \
// g++ -std=c++17 -O1 -o /tmp/t_rs_env engine/tests/test_flash_route_split_env.cpp engine/knobs-rs/target/release/libgr8knobs.a \
//     -lpthread -ldl -lm && /tmp/t_rs_env
#include "../knobs-rs/gr8_knobs.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
// the exit code of a child that reads the knob with GR8_PF_ROUTER_SPLIT = v (nullptr: unset): 0 off, 1 on, 2 refused
static int run(const char* v) {
    const pid_t pid = fork();
    if (pid == 0) {
        unsetenv("GR8_PF_ROUTER_SPLIT");
        if (v) setenv("GR8_PF_ROUTER_SPLIT", v, 1);
        else unsetenv("GR8_PF_ROUTER_SPLIT");
        _exit(knob::pf_router_split() ? 1 : 0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status));
    return WEXITSTATUS(status);
}
int main() {
    assert(run(nullptr) == 0);
    assert(run("") == 0);   // an empty value counts as unset
    assert(run("0") == 0);
    assert(run("1") == 1);
    int bad = 0;
    for (const char* s : {"2", "01", "-1", "1junk", " 1", "true"}) {
        assert(run(s) == 2);
        ++bad;
    }
    std::printf("test_flash_route_split_env: off/on and %d bad values verified\n", bad);
}
