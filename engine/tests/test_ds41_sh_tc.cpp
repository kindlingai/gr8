// Host tests of the tensor-core shared expert (GR8_DS_SH_TC). No CUDA.
//   H1  repack round trip: random codes + E8M0 bytes -> the forward packer (a host copy of k_kda_mx_record's loop, engine/kda/kda.cuh) -> the
//       inverse k_mx_unrecord runs (its index map attn/ds41_sh_tc_map.h, the same function the kernel calls) -> bytes equal. Shapes of bind_moe:
//       W13 = gate rows 0..575 + up rows 576..1151 at K 5120; W2p = down [5120][640] from K 576, the pad k 576..639 (codes) and blocks 18..19 (bytes) 0.
//   H2  GR8_DS_SH_TC through the knob table (knob::ds_sh_tc(), what bind_moe reads): unset / empty / 0 / 1 accepted, anything else stops the
//       process (exit 2) at the first read. A knob is read once per process, so each case runs in its own child.
//   plus the ds41_gemm_mx_tc shape rule (tc_ok: K % 128, N % 64, lda % 16, lsa % 4) for the two GEMMs of shared_ro_tc.
// build:  g++ -std=c++17 -O2 -Wall -Wextra -I engine engine/tests/test_ds41_sh_tc.cpp engine/knobs-rs/target/release/libgr8knobs.a
//             -lpthread -ldl -lm -o /tmp/test_ds41_sh_tc
// run:    /tmp/test_ds41_sh_tc
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "attn/ds41_sh_tc_map.h"
#include "knobs-rs/gr8_knobs.h"

static int fails = 0, checks = 0;
static void ck(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) { printf("FAIL %s\n", what.c_str()); ++fails; }
    else printf("ok   %s\n", what.c_str());
}

// host copy of k_kda_mx_record (engine/kda/kda.cuh): codes [16T][K] + scales [16T][K/32] -> [T][K/32][528]
static void fwd_record(const uint8_t* codes, const uint8_t* scales, int T, int K, uint8_t* out)
{
    const int ng = K / 32;
    for (int b = 0; b < T * ng; ++b)
    {
        const int t = b / ng, g = b % ng;
        uint8_t* rec = out + (size_t)b * 528;
        for (int i = 0; i < 256; ++i)
        {
            const int l = i >> 3, e = i & 7, r = l >> 2, c = l & 3;
            const int row = r + ((e & 2) ? 8 : 0), col = 2 * c + (e & 1) + ((e & 4) ? 8 : 0);
            for (int h = 0; h < 2; ++h) rec[h * 256 + i] = codes[(size_t)(16 * t + row) * K + 32 * g + 16 * h + col];
            if (i < 8)
            {
                rec[512 + 2 * i] = scales[(size_t)(16 * t + i) * ng + g];
                rec[512 + 2 * i + 1] = scales[(size_t)(16 * t + i + 8) * ng + g];
            }
        }
    }
}
// the inverse, as k_mx_unrecord runs it (one "CTA" b, "threads" i), with the shared index map
static void unrecord(const uint8_t* rec_all, int T, int K, uint8_t* codes, int ldc, uint8_t* scales, int lds)
{
    const int ng = K / 32;
    for (int b = 0; b < T * ng; ++b)
    {
        const int t = b / ng, g = b % ng;
        const uint8_t* rec = rec_all + (size_t)b * ds41a::MX_REC_B;
        for (int i = 0; i < 256; ++i)
        {
            int row, col;
            ds41a::mx_rec_rc(i, row, col);
            for (int h = 0; h < 2; ++h) codes[(size_t)(16 * t + row) * ldc + 32 * g + 16 * h + col] = rec[h * 256 + i];
            if (i < 8)
            {
                scales[(size_t)(16 * t + i) * lds + g] = rec[512 + 2 * i];
                scales[(size_t)(16 * t + i + 8) * lds + g] = rec[512 + 2 * i + 1];
            }
        }
    }
}

// knob::ds_sh_tc() in a child with GR8_DS_SH_TC=val (null: unset): 0 or 1, or -2 when the table refuses the value
static int sh_tc(const char* val)
{
    fflush(stdout);
    const pid_t p = fork();
    if (!p)
    {
        unsetenv("G6_DS_SH_TC");   // the old name, still read
        if (val) setenv("GR8_DS_SH_TC", val, 1); else unsetenv("GR8_DS_SH_TC");
        const int null = open("/dev/null", O_WRONLY);
        if (null >= 0) dup2(null, 2);        // the refusal's message
        _exit(10 + (knob::ds_sh_tc() ? 1 : 0));
    }
    int st = 0;
    waitpid(p, &st, 0);
    if (!WIFEXITED(st)) return -1;
    return WEXITSTATUS(st) == 2 ? -2 : WEXITSTATUS(st) - 10;
}

int main()
{
    constexpr int HID = 5120, IW = 576, KP = (IW + 127) / 128 * 128, N13 = 2 * IW;   // KP: kernels/ds41_moe_ops.cuh SH_KP
    std::mt19937 rng(12);
    auto fill = [&](std::vector<uint8_t>& v) { for (auto& x : v) x = (uint8_t)(rng() & 0xff); };

    // ---- H1: the index map hits every byte of a record's 512 code bytes exactly once (a bijection onto 16 x 32)
    {
        std::vector<int> hit(16 * 32, 0);
        for (int i = 0; i < 256; ++i)
        {
            int row, col;
            ds41a::mx_rec_rc(i, row, col);
            for (int h = 0; h < 2; ++h) hit[row * 32 + 16 * h + col]++;
        }
        bool one = true;
        for (int x : hit) one = one && x == 1;
        ck(one, "H1 map: 512 code bytes of a record -> 16 rows x 32 k, each once");
    }
    // ---- H1: W13 (gate + up at K 5120, the bind_moe offsets)
    {
        std::vector<uint8_t> g(IW * HID), u(IW * HID), gs(IW * (HID / 32)), us(IW * (HID / 32));
        fill(g); fill(u); fill(gs); fill(us);
        std::vector<uint8_t> rg((size_t)(IW / 16) * (HID / 32) * 528), ru(rg.size());
        fwd_record(g.data(), gs.data(), IW / 16, HID, rg.data());
        fwd_record(u.data(), us.data(), IW / 16, HID, ru.data());
        std::vector<uint8_t> w13((size_t)N13 * HID, 0), sw13((size_t)N13 * (HID / 32), 0);
        unrecord(rg.data(), IW / 16, HID, w13.data(), HID, sw13.data(), HID / 32);
        unrecord(ru.data(), IW / 16, HID, w13.data() + (size_t)IW * HID, HID, sw13.data() + (size_t)IW * (HID / 32), HID / 32);
        const bool cw = !memcmp(w13.data(), g.data(), g.size()) && !memcmp(w13.data() + g.size(), u.data(), u.size());
        const bool cs = !memcmp(sw13.data(), gs.data(), gs.size()) && !memcmp(sw13.data() + gs.size(), us.data(), us.size());
        ck(cw, "H1 W13 codes [1152][5120] == gate rows 0..575 + up rows 576..1151, bytewise");
        ck(cs, "H1 SW13 bytes [1152][160] == gate + up scales, bytewise");
    }
    // ---- H1: W2p (down, K 576 -> padded 640)
    {
        std::vector<uint8_t> d((size_t)HID * IW), ds((size_t)HID * (IW / 32));
        fill(d); fill(ds);
        std::vector<uint8_t> rd((size_t)(HID / 16) * (IW / 32) * 528);
        fwd_record(d.data(), ds.data(), HID / 16, IW, rd.data());
        std::vector<uint8_t> w2((size_t)HID * KP, 0), sw2((size_t)HID * (KP / 32), 0);   // zeroed, as dmalloc does
        unrecord(rd.data(), HID / 16, IW, w2.data(), KP, sw2.data(), KP / 32);
        bool body = true, pad = true, sbody = true, spad = true;
        for (int n = 0; n < HID; ++n)
        {
            for (int k = 0; k < KP; ++k)
            {
                const uint8_t v = w2[(size_t)n * KP + k];
                if (k < IW) body = body && v == d[(size_t)n * IW + k];
                else pad = pad && v == 0;
            }
            for (int b = 0; b < KP / 32; ++b)
            {
                const uint8_t v = sw2[(size_t)n * (KP / 32) + b];
                if (b < IW / 32) sbody = sbody && v == ds[(size_t)n * (IW / 32) + b];
                else spad = spad && v == 0;
            }
        }
        ck(KP == 640, "H1 SH_KP = 640 (576 rounded up to 128)");
        ck(body, "H1 W2p codes k 0..575 == down codes, bytewise");
        ck(pad, "H1 W2p codes k 576..639 == 0 (pad)");
        ck(sbody, "H1 SW2p bytes blocks 0..17 == down scales");
        ck(spad, "H1 SW2p bytes blocks 18..19 == 0 (pad)");
    }
    // ---- the two GEMMs of shared_ro_tc pass tc_ok (ds41_attn.cuh: K % 128, N % 64, lda % 16, lsa % 4)
    {
        auto tc_ok = [](int N, int K, int lda, int lsa) { return K % 128 == 0 && N % 64 == 0 && lda % 16 == 0 && lsa % 4 == 0; };
        ck(tc_ok(N13, HID, HID, HID / 32), "tc_ok fc1: N 1152, K 5120, lda 5120, lsa 160");
        ck(tc_ok(HID, KP, KP, KP / 32), "tc_ok fc2: N 5120, K 640, lda 640, lsa 20");
        ck(!tc_ok(HID, IW, IW, IW / 32), "tc_ok refuses the unpadded fc2 (K 576): the pad is needed");
    }
    // ---- H2: GR8_DS_SH_TC through the knob table
    {
        struct C { const char* v; int want; };
        const C cs[] = {{nullptr, 0}, {"", 0}, {"0", 0}, {"1", 1}, {"2", -2}, {"x", -2}, {"01", -2}, {"1 ", -2}, {" 1", -2}, {"-1", -2}, {"10", -2}, {"true", -2}};
        for (const C& c : cs)
        {
            const int got = sh_tc(c.v);
            ck(got == c.want, std::string("H2 GR8_DS_SH_TC=") + (c.v ? std::string("'") + c.v + "'" : "unset") + " -> " +
                                  (got == -2 ? std::string("refused") : std::to_string(got)));
        }
    }
    printf("%s: %d checks, %d failed\n", fails ? "DS41 SH_TC host FAIL" : "DS41 SH_TC host PASS", checks, fails);
    return fails ? 1 : 0;
}
