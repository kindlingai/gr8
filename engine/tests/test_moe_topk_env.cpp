// GR8_MOE_TOPK through the knob table (engine/knobs-rs/src/table.rs MOE_TOPK: int 1..8, unset = 8, the model's top-k).
// A cut of 0, values above 8, words, suffixes, spaces and negative numbers must stop the process (exit 2) at the first
// read, not fall back to a default. A knob is read once per process and a refused value ends it, so each case runs in
// its own child.
//
// build and run:  g++ -O2 -std=c++17 engine/tests/test_moe_topk_env.cpp engine/knobs-rs/target/release/libgr8knobs.a \
//                     -lpthread -ldl -lm -o /tmp/test_moe_topk_env && /tmp/test_moe_topk_env
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "../knobs-rs/gr8_knobs.h"

static int fails = 0, checks = 0;
static void ck(int ok, const char* what)
{
    ++checks;
    if (!ok) { printf("FAIL %s\n", what); ++fails; }
    else printf("ok   %s\n", what);
}

// knob::moe_topk() in a child with GR8_MOE_TOPK=val (null: unset): the cut, or -2 when the table refuses the value
static int topk(const char* val)
{
    fflush(stdout);
    const pid_t p = fork();
    if (!p)
    {
        if (val) setenv("GR8_MOE_TOPK", val, 1); else unsetenv("GR8_MOE_TOPK");
        const int null = open("/dev/null", O_WRONLY);
        if (null >= 0) dup2(null, 2);        // the refusal's message
        _exit(10 + (int)knob::moe_topk());
    }
    int st = 0;
    waitpid(p, &st, 0);
    if (!WIFEXITED(st)) return -1;
    return WEXITSTATUS(st) == 2 ? -2 : WEXITSTATUS(st) - 10;
}

int main()
{
    ck(topk(nullptr) == 8, "unset -> the model default 8");
    ck(topk("") == 8, "empty -> unset -> 8");
    ck(topk("1") == 1, "'1' -> 1 (the smallest valid cut)");
    ck(topk("4") == 4, "'4' -> 4");
    ck(topk("6") == 6, "'6' -> 6");
    ck(topk("8") == 8, "'8' -> 8 (= the model top-k, explicit)");
    ck(topk("0") == -2, "'0' refused (a cut of 0 is not valid)");
    ck(topk("9") == -2, "'9' refused (above the model top-k)");
    ck(topk("x") == -2, "'x' refused");
    ck(topk("4x") == -2, "'4x' refused (suffix)");
    ck(topk(" 4") == -2, "' 4' refused (space)");
    ck(topk("2.5") == -2, "'2.5' refused (not an integer)");
    ck(topk("-1") == -2, "'-1' refused");
    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
