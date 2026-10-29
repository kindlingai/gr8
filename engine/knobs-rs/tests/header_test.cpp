// gr8_knobs.h over the C API: defaults, optionals, a family, lists, the signature and init's error count.
//   cargo build --release --features capi && g++ -std=c++17 -I engine/knobs-rs engine/knobs-rs/tests/header_test.cpp \
//     engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm
// It sets its own environment and runs itself again.
#include "gr8_knobs.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        // a clean environment with a few knobs set, one through its old name
        char* env[] = {(char*)"GR8_ATTN_CTX2=3", (char*)"G6_ATTN_CTX3=1", (char*)"GR8_MTP_KSTOPB_TAU_S2=0.5,0.75",
                       (char*)"GR8_KVB=fp8", (char*)"GR8_S2_GEN=4,5", nullptr};
        execve(argv[0], (char* const[]){argv[0], (char*)"child", nullptr}, env);
        perror("execve");
        return 1;
    }
    if (!strcmp(argv[1], "child"))
    {
        assert(knob::init() == 0);
        assert(knob::attn_ctx2() == 3 && knob::attn_ctx3() == 1 && knob::attn_ctx3_hg() == 0);
        assert(knob::kvb() == "fp8" && knob::mtp_kstopb_tau() == (std::vector<double>{0.6, 0.88, 0.95}));
        assert(knob::mtp_kstopb_tau_s(2) == (std::vector<double>{0.5, 0.75}) && !knob::mtp_kstopb_tau_s(3));
        assert(knob::s2_gen() == (std::vector<int64_t>{4, 5}) && !knob::dfl_reuse() && !knob::rt_store());
        assert(knob::signature() == " env ATTN_CTX2=3 ATTN_CTX3=1 KVB=fp8 MTP_KSTOPB_TAU_S2=0.5,0.75");
        char* env[] = {(char*)"GR8_ATTN_CTX2=9", (char*)"GR8_NO_SUCH=1", nullptr};
        execve(argv[0], (char* const[]){argv[0], (char*)"bad", nullptr}, env);
        perror("execve");
        return 1;
    }
    // a bad value and an unknown name: two errors, and reading the bad knob stops the process with status 2
    assert(knob::init() == 2);
    if (fork() == 0) knob::attn_ctx2();
    int status;
    wait(&status);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 2);
    printf("header_test: ok\n");
}
