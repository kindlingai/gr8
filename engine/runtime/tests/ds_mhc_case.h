// Reader for the case files written by engine/mhc/ds_ref.py gen (layout in that file's docstring) + error metrics.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace dscase {
struct Layer { std::vector<float> fn, base, scale, norm, block; };
struct Expect { std::vector<float> y, post, comb, pre_next; };
struct Case {
    int M = 0, H = 0, NS = 0, V = 0;
    float eps = 0;
    std::vector<int> tok;
    std::vector<float> embed, final_norm, streams_last, final_h;
    std::vector<Layer> L;
    std::vector<Expect> E;
};
inline bool rd(FILE* f, void* p, size_t bytes) { return fread(p, 1, bytes, f) == bytes; }
inline std::vector<float> rdf(FILE* f, size_t n, bool& ok) { std::vector<float> v(n); ok = ok && rd(f, v.data(), n*4); return v; }
inline bool load(const char* path, Case& c) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return false; }
    int32_t hd[4]; bool ok = rd(f, hd, 16) && rd(f, &c.eps, 4);
    if (!ok) return false;
    c.M = hd[0]; c.H = hd[1]; c.NS = hd[2]; c.V = hd[3];
    c.tok.resize(c.M); ok = ok && rd(f, c.tok.data(), (size_t)c.M*4);
    c.embed = rdf(f, (size_t)c.V*c.H, ok);
    c.L.resize(c.NS);
    for (auto& l : c.L) {
        l.fn = rdf(f, (size_t)24*4*c.H, ok); l.base = rdf(f, 24, ok); l.scale = rdf(f, 3, ok);
        l.norm = rdf(f, c.H, ok); l.block = rdf(f, (size_t)c.M*c.H, ok);
    }
    c.final_norm = rdf(f, c.H, ok);
    c.E.resize(c.NS);
    for (auto& e : c.E) {
        e.y = rdf(f, (size_t)c.M*c.H, ok); e.post = rdf(f, (size_t)c.M*4, ok);
        e.comb = rdf(f, (size_t)c.M*16, ok); e.pre_next = rdf(f, (size_t)c.M*4, ok);
    }
    c.streams_last = rdf(f, (size_t)c.M*4*c.H, ok); c.final_h = rdf(f, (size_t)c.M*c.H, ok);
    fclose(f);
    if (!ok) printf("short read in %s\n", path);
    return ok;
}
struct Err { double rel_l2 = 0, max_abs = 0, mismatch = 0; };
inline Err cmp(const float* a, const float* ref, size_t n) {
    double num = 0, den = 0, mx = 0; size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - ref[i];
        num += d*d; den += (double)ref[i]*ref[i];
        if (std::fabs(d) > mx) mx = std::fabs(d);
        if (a[i] != ref[i]) ++bad;
    }
    return {std::sqrt(num / (den > 0 ? den : 1.0)), mx, (double)bad / (n ? n : 1)};
}
// Gate for BF16-valued tensors after rounding-order differences: mismatch fraction and relative L2.
inline bool gate_bf16(const char* what, const Err& e, double max_mismatch, double max_rel) {
    const bool ok = e.mismatch <= max_mismatch && e.rel_l2 <= max_rel;
    printf("  %-18s rel_l2 %.3e  max_abs %.3e  mismatch %.4f%%  %s\n", what, e.rel_l2, e.max_abs, 100*e.mismatch, ok ? "ok" : "FAIL");
    return ok;
}
inline bool gate_f32(const char* what, const Err& e, double max_abs) {
    const bool ok = e.max_abs <= max_abs;
    printf("  %-18s rel_l2 %.3e  max_abs %.3e  %s\n", what, e.rel_l2, e.max_abs, ok ? "ok" : "FAIL");
    return ok;
}
} // namespace dscase
