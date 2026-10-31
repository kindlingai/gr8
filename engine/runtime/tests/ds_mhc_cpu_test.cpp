// CPU-only gate: the host oracle engine/mhc/ds_cpu.h against the torch reference engine/mhc/ds_ref.py on one case file.
//   python3 engine/mhc/ds_ref.py gen /tmp/ds_case.bin 3 256 6 50 1e-20 1 && ds_mhc_cpu_test /tmp/ds_case.bin
#include "../../mhc/ds_cpu.h"
#include "ds_mhc_case.h"
#include <cstdio>
using namespace dscase;
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    Case c;
    if (!load(argv[1], c)) return 2;
    const int M = c.M, H = c.H;
    printf("ds_mhc_cpu_test: M=%d H=%d sublayers=%d V=%d eps=%g\n", M, H, c.NS, c.V, c.eps);
    std::vector<float> T((size_t)M*4*H), pre(M*4), post(M*4), comb(M*16), y((size_t)M*H), h((size_t)M*H);
    ds_mhc::embed_cpu(c.tok.data(), c.embed.data(), M, H, T.data(), pre.data());
    bool ok = true;
    for (int s = 0; s < c.NS; ++s) {
        const Layer& l = c.L[s]; const Expect& e = c.E[s];
        ds_mhc::prepare_cpu(T.data(), pre.data(), l.fn.data(), l.base.data(), l.scale.data(), l.norm.data(), M, H, c.eps,
                            post.data(), comb.data(), y.data());
        printf("sub-layer %d\n", s);
        ok &= gate_bf16("y", cmp(y.data(), e.y.data(), y.size()), 0.02, 4e-3);
        ok &= gate_f32("post", cmp(post.data(), e.post.data(), post.size()), 2e-4);
        ok &= gate_f32("comb", cmp(comb.data(), e.comb.data(), comb.size()), 2e-4);
        ok &= gate_f32("pre_next", cmp(pre.data(), e.pre_next.data(), pre.size()), 2e-4);
        ds_mhc::consume_cpu(T.data(), l.block.data(), post.data(), comb.data(), M, H);
    }
    ds_mhc::final_cpu(T.data(), pre.data(), c.final_norm.data(), M, H, c.eps, h.data());
    printf("end of model\n");
    ok &= gate_bf16("trunk (last)", cmp(T.data(), c.streams_last.data(), T.size()), 0.03, 6e-3);
    ok &= gate_bf16("final h", cmp(h.data(), c.final_h.data(), h.size()), 0.03, 6e-3);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
