// Cross-check driver for tests/nvfp4.rs: runs the host NVFP4 code (engine/kernels/nvfp4_pack.h) on a file of matrices
// and writes its bytes, so the Rust twin (src/nvfp4.rs) can be compared byte for byte.
//   usage: nvfp4_cpp IN OUT
//   IN : i32 R, K, KFULL, C0, NQ, KQ, then gp[R*K/2] gs[R*K/16] up[R*K/2] us[R*K/16]       (fc1 inputs)
//        then w2p[R*KFULL/2] w2s[R*KFULL/16]                                               (fc2 input, columns C0..C0+K)
//        then f32 x[NQ*KQ], f32 fixed
//   OUT: pack_fc1, pack_fc2, then for rule in (Static6, FourOverSixMae):
//          per row of x: codes[KQ/2] scales[KQ/16] f32 outer               (own outer scale per row)
//          all of x under the fixed outer scale: codes scales f32 outer
#include "nvfp4_pack.h"
#include <cstdio>
#include <cstdlib>
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
    int32_t h[6];
    if (!f || fread(h, 4, 6, f) != 6) return 2;
    const int R = h[0], K = h[1], KF = h[2], C0 = h[3], NQ = h[4], KQ = h[5];
    auto gp = rd(f, (size_t)R * K / 2), gs = rd(f, (size_t)R * K / 16), up = rd(f, (size_t)R * K / 2), us = rd(f, (size_t)R * K / 16);
    auto w2p = rd(f, (size_t)R * KF / 2), w2s = rd(f, (size_t)R * KF / 16);
    std::vector<float> x((size_t)NQ * KQ);
    float fixed;
    if (fread(x.data(), 4, x.size(), f) != x.size() || fread(&fixed, 4, 1, f) != 1) return 2;
    fclose(f);
    FILE* o = fopen(argv[2], "wb");
    if (!o) return 3;
    auto fc1 = nv4pack::pack_fc1(gp.data(), gs.data(), up.data(), us.data(), R, K);
    auto fc2 = nv4pack::pack_fc2(w2p.data(), w2s.data(), R, KF, C0, K);
    fwrite(fc1.data(), 1, fc1.size(), o);
    fwrite(fc2.data(), 1, fc2.size(), o);
    for (auto rule : {nv4pack::ScaleRule::Static6, nv4pack::ScaleRule::FourOverSixMae})
    {
        std::vector<uint8_t> c((size_t)NQ * KQ / 2), s((size_t)NQ * KQ / 16);
        for (int n = 0; n < NQ; ++n)
        {
            const float outer = nv4pack::quantize_row(x.data() + (size_t)n * KQ, KQ, rule, c.data(), s.data());
            fwrite(c.data(), 1, KQ / 2, o);
            fwrite(s.data(), 1, KQ / 16, o);
            fwrite(&outer, 4, 1, o);
        }
        const float outer = nv4pack::quantize(x.data(), NQ, KQ, rule, &fixed, c.data(), s.data());
        fwrite(c.data(), 1, c.size(), o);
        fwrite(s.data(), 1, s.size(), o);
        fwrite(&outer, 4, 1, o);
    }
    fclose(o);
    return 0;
}
