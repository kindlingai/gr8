// Cross-check driver for tests/ds41.rs: runs builder B6's host packer (engine/kernels/mxfp4_pack.h) on a file of
// matrices and writes its records, so the Rust port (src/ds41.rs mx4g_fc1 / mx4g_fc2) can be compared byte for byte.
//   usage: ds41_mx4g_cpp IN OUT
//   IN : i32 R, i32 K, i32 KFULL, i32 C0, then gp[R*K/2] ge[R*K/32] up[R*K/2] ue[R*K/32]   (fc1 inputs, R rows of K)
//        then w2p[R*KFULL/2] w2e[R*KFULL/32]                                                 (fc2 input, column slice C0..C0+K)
//   OUT: pack_fc1(gp,ge,up,ue,R,K) then pack_matrix(slice_cols(w2p,w2e,R,KFULL,C0,K),R,K)
#include "mxfp4_pack.h"
#include <cstdio>
#include <vector>

static std::vector<uint8_t> rd(FILE* f, size_t n)
{
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
    return v;
}

int main(int argc, char** argv)
{
    if (argc < 3) return 1;
    FILE* f = fopen(argv[1], "rb");
    int32_t h[4];
    if (!f || fread(h, 4, 4, f) != 4) return 2;
    const int R = h[0], K = h[1], KF = h[2], C0 = h[3];
    auto gp = rd(f, (size_t)R * K / 2), ge = rd(f, (size_t)R * K / 32), up = rd(f, (size_t)R * K / 2), ue = rd(f, (size_t)R * K / 32);
    auto w2p = rd(f, (size_t)R * KF / 2), w2e = rd(f, (size_t)R * KF / 32);
    fclose(f);
    auto fc1 = mx4pack::pack_fc1(gp.data(), ge.data(), up.data(), ue.data(), R, K);
    std::vector<uint8_t> sp, se;
    mx4pack::slice_cols(sp, se, w2p.data(), w2e.data(), R, KF, C0, K);
    auto fc2 = mx4pack::pack_matrix(sp.data(), se.data(), R, K);
    FILE* o = fopen(argv[2], "wb");
    if (!o) return 3;
    fwrite(fc1.data(), 1, fc1.size(), o);
    fwrite(fc2.data(), 1, fc2.size(), o);
    fclose(o);
    return 0;
}
