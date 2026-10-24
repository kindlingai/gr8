// G6_MOE_TOPK env parsing (engine/runtime/envparse.h): strict single digit 1..TOPK, unset = the model's top-k.
// A cut of 0, values above TOPK, signs, spaces, leading zeros, suffixes and the empty string must all fail CLOSED
// with a message naming the variable and the bad value (atoi would fail open). moe_topk_cut caches its first read in
// a function-local static, so each of its environments is checked in its own forked child.
//
// build and run:  g++ -O2 -std=c++17 -Iengine/runtime engine/tests/test_moe_topk_env.cpp -o /tmp/test_moe_topk_env &&
//                 /tmp/test_moe_topk_env
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <string>

#include "envparse.h"

static int fails = 0, checks = 0;
static void ck(int ok, const char* what)
{
    ++checks;
    if (!ok) { printf("FAIL %s\n", what); ++fails; }
    else printf("ok   %s\n", what);
}

// env_digit is stateless: every case runs in-process
static bool digit_case(const char* val, int want, bool want_err)
{
    std::string e;
    if (val) setenv("G6_MOE_TOPK", val, 1); else unsetenv("G6_MOE_TOPK");
    const int got = env_digit("G6_MOE_TOPK", 1, 8, 8, &e);
    const bool err_ok = want_err ? e.find("G6_MOE_TOPK") != std::string::npos && (!val || e.find(val) != std::string::npos)
                                 : e.empty();
    return got == want && err_ok;
}

// moe_topk_cut reads the environment once per process: fork per case
static bool cut_case(const char* val, int want)
{
    if (val) setenv("G6_MOE_TOPK", val, 1); else unsetenv("G6_MOE_TOPK");
    const pid_t p = fork();
    if (!p) _exit(moe_topk_cut(8) == want ? 0 : 1);
    int st = 0;
    waitpid(p, &st, 0);
    return WIFEXITED(st) && !WEXITSTATUS(st);
}

int main()
{
    ck(digit_case(nullptr, 8, false), "unset -> the model default 8");
    ck(digit_case("1", 1, false), "'1' -> 1 (the smallest valid cut)");
    ck(digit_case("4", 4, false), "'4' -> 4");
    ck(digit_case("6", 6, false), "'6' -> 6");
    ck(digit_case("8", 8, false), "'8' -> 8 (= the model top-k, explicit)");
    ck(digit_case("0", -1, true), "'0' rejected (a cut of 0 is not valid), error names var + value");
    ck(digit_case("9", -1, true), "'9' rejected (above TOPK)");
    ck(digit_case("", -1, true), "empty string rejected");
    ck(digit_case("x", -1, true), "'x' rejected");
    ck(digit_case("08", -1, true), "'08' rejected (two chars)");
    ck(digit_case("4x", -1, true), "'4x' rejected (suffix)");
    ck(digit_case(" 4", -1, true), "' 4' rejected (space)");
    ck(digit_case("+6", -1, true), "'+6' rejected (sign)");
    ck(digit_case("-1", -1, true), "'-1' rejected");
    ck(cut_case("6", 6), "moe_topk_cut(8) with G6_MOE_TOPK=6 -> 6 (own process)");
    ck(cut_case(nullptr, 8), "moe_topk_cut(8) unset -> 8 (own process)");
    ck(cut_case("bad", 8), "moe_topk_cut(8) bad value -> fallback 8 (boot already died; own process)");
    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
