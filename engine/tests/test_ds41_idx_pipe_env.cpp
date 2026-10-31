// GR8_DS41_IDX_PIPE_B through the knob table (knobs/ds41-attn.yaml: flag, default 0). Unset and empty read 0, "0" and "1"
// read as given, and any other word stops the process (exit 2) at the first read. A knob is read once per process and a
// refused value ends it, so each case runs in its own child.
//
// build and run (from the repo root):
//   g++ -O2 -std=c++17 engine/tests/test_ds41_idx_pipe_env.cpp engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm
//       -o /tmp/test_ds41_idx_pipe_env && /tmp/test_ds41_idx_pipe_env
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "../knobs-rs/gr8_knobs.h"

// knob::ds41_idx_pipe_b() in a child with GR8_DS41_IDX_PIPE_B=val (null: unset): 0 or 1, or -2 when the table refuses the value
static int pipe_b(const char* val)
{
    fflush(stdout);
    const pid_t p = fork();
    if (!p)
    {
        unsetenv("G6_DS41_IDX_PIPE_B");   // the old name, still read
        if (val) setenv("GR8_DS41_IDX_PIPE_B", val, 1); else unsetenv("GR8_DS41_IDX_PIPE_B");
        const int null = open("/dev/null", O_WRONLY);
        if (null >= 0) dup2(null, 2);        // the refusal's message
        _exit(10 + (knob::ds41_idx_pipe_b() ? 1 : 0));
    }
    int st = 0;
    waitpid(p, &st, 0);
    if (!WIFEXITED(st)) return -1;
    return WEXITSTATUS(st) == 2 ? -2 : WEXITSTATUS(st) - 10;
}

int main()
{
    int fails = 0;
    auto ck = [&](const char* val, int want) {
        const int got = pipe_b(val);
        if (got != want) { printf("FAIL GR8_DS41_IDX_PIPE_B=%s: got %d, want %d\n", val ? val : "(unset)", got, want); ++fails; }
    };
    ck(nullptr, 0);
    ck("", 0);
    ck("0", 0);
    ck("1", 1);
    for (const char* bad : {"2", "-1", "01", "1junk", " 1", "yes"}) ck(bad, -2);
    if (fails) return 1;
    printf("test_ds41_idx_pipe_env: ok\n");
    return 0;
}
