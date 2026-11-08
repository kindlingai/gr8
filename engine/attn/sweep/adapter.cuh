#pragma once
#include "fixtures.cuh"
#include <stdexcept>
namespace sweep_adapter {
using Params = ds41a::AttnP;
struct Shape { int n, pos, ratio, floor, wl, keys; };
inline Shape shape(const std::vector<std::string>& a) {
    if (a.size()!=6) throw std::runtime_error("attention needs rows,pos,ratio,floor,WL,key_rows");
    Shape s{std::stoi(a[0]),std::stoi(a[1]),std::stoi(a[2]),std::stoi(a[3]),std::stoi(a[4]),std::stoi(a[5])};
    if (s.n<1 || s.n>8192 || s.pos<0 || s.pos>1048576-s.n || s.ratio<0 || s.ratio>2 ||
        s.floor<0 || s.floor>s.pos || s.wl<1 || s.wl>128 || s.keys<1 || s.keys>1048576 ||
        (s.floor && !s.ratio) || (s.ratio && s.keys<(s.pos+s.n)/s.ratio))
        throw std::runtime_error("invalid attention shape");
    return s;
}
inline size_t required_bytes(const std::vector<std::string>& a) {
    const auto s=shape(a);
    // Three BF16 tensors, packed + BF16 key fixtures, window ring, top-k and metadata.
    return size_t(s.n)*16*512*2*3 + size_t(s.keys)*(512*2+288) +
           size_t(128+s.n-1)*512*2 + size_t(s.n)*(512*4+4) + 4096;
}
struct Fixture { Built b; Keys keys; };
inline Fixture make_case(const std::vector<std::string>& a, uint64_t seed) {
    const auto s=shape(a); rng.seed(seed);
    Fixture f; f.keys=make_keys(s.keys,1);
    Case c{"sweep",s.ratio ? s.ratio : 2,s.n,1,s.pos,s.floor,s.wl};
    f.b=build(c,16,f.keys,false);
    if (!s.ratio) f.b.p.TK=0;
    set_rows(f.b.p,f.keys,s.ratio!=0,f.b.out_b);
    return f;
}
inline Params params(Fixture& f,bool ref) { auto p=f.b.p; p.out=ref?f.b.out_a:f.b.out_b; return p; }
inline void* output(Fixture& f,bool ref) { return ref?f.b.out_a:f.b.out_b; }
inline size_t output_bytes(const Fixture& f) { return f.b.out_n*2; }
inline void destroy(Fixture& f) { free_built(f.b); free_keys(f.keys); }
}
