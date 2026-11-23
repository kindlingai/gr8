// Host C++ test of EL4d (#161) C FFI in engine/kvman-rs/g6kvman.h:
// - g6kv_canon_heads_out / g6kv_canon_heads_in
// - g6kv_canon_dfl_export / g6kv_canon_dfl_import (split TP=1/2/4/8 <-> replicated TP=3/6, ring wrap-around, bulk+tail)
// - g6kv_canon_kda_rank_bytes / g6kv_canon_kda_export / g6kv_canon_kda_import cross-checked against kda::heads/nhmax
// - g6kv_canon_session_pack / g6kv_canon_session_unpack
#include "../../../kda/math.h"
#include "../../../kvman-rs/g6kvman.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint64_t rng_next(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::vector<uint8_t> rand_bytes(uint64_t& s, size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(rng_next(s));
    return v;
}

void test_heads_ffi() {
    uint64_t s = 17;
    const uint32_t heads = 64;
    const size_t outer = 3, inner = 24;
    const auto whole = rand_bytes(s, outer * heads * inner);
    for (uint32_t nr : {1u, 2u, 3u, 4u, 6u, 8u}) {
        std::vector<std::vector<uint8_t>> parts(nr);
        std::vector<const uint8_t*> ptrs(nr);
        std::vector<size_t> lens(nr);
        for (uint32_t g = 0; g < nr; ++g) {
            size_t need = 0;
            int rc = g6kv_canon_heads_in(whole.data(), whole.size(), heads, outer, inner, nr, g, nullptr, 0, &need);
            assert(rc == 1 && need > 0);
            parts[g].resize(need);
            rc = g6kv_canon_heads_in(whole.data(), whole.size(), heads, outer, inner, nr, g, parts[g].data(),
                                     parts[g].size(), &need);
            assert(rc == 0 && need == parts[g].size());
            ptrs[g] = parts[g].data();
            lens[g] = parts[g].size();
        }
        size_t need = 0;
        int rc = g6kv_canon_heads_out(ptrs.data(), lens.data(), nr, heads, outer, inner, nullptr, 0, &need);
        assert(rc == 1 && need == whole.size());
        std::vector<uint8_t> gathered(need);
        rc = g6kv_canon_heads_out(ptrs.data(), lens.data(), nr, heads, outer, inner, gathered.data(), gathered.size(),
                                  &need);
        assert(rc == 0 && gathered == whole);
    }
    // Invalid call reports an error string
    size_t need = 0;
    assert(g6kv_canon_heads_in(whole.data(), 1, heads, outer, inner, 4, 0, nullptr, 0, &need) == -1);
    assert(std::strlen(g6kv_canon_error()) > 0);
}

void test_dfl_ffi() {
    uint64_t s = 29;
    const uint32_t L = 6, NKV = 8, HD = 16, EB = 2, RING = 64;
    const uint64_t lo = 50, mid = 72, hi = 94; // wraps around RING=64
    const size_t row = size_t(HD) * EB, rows = size_t(hi - lo);
    const auto k_truth = rand_bytes(s, size_t(L) * NKV * rows * row);
    const auto v_truth = rand_bytes(s, size_t(L) * NKV * rows * row);

    auto build_rank = [&](uint32_t nr, int rep, uint32_t rank, uint8_t fill) {
        const uint32_t nkr = rep ? NKV : NKV / nr;
        const uint32_t h0 = rep ? 0 : rank * nkr;
        const size_t half = size_t(L) * nkr * RING * row;
        std::vector<uint8_t> k(half, fill), v(half, fill);
        for (uint32_t l = 0; l < L; ++l)
            for (uint32_t hl = 0; hl < nkr; ++hl)
                for (size_t o = 0; o < rows; ++o) {
                    const size_t slot = size_t((lo + o) % RING);
                    const size_t src = ((size_t(l) * NKV + h0 + hl) * rows + o) * row;
                    const size_t dst = ((size_t(l) * nkr + hl) * RING + slot) * row;
                    std::memcpy(k.data() + dst, k_truth.data() + src, row);
                    std::memcpy(v.data() + dst, v_truth.data() + src, row);
                }
        return std::make_pair(std::move(k), std::move(v));
    };

    // Export at TP=4 (split) in two pieces (bulk + tail), import at TP=6 (replicated) and TP=2 (split)
    std::vector<std::vector<uint8_t>> rk(4), rv(4);
    std::vector<const uint8_t*> kp(4), vp(4);
    std::vector<size_t> kb(4), vb(4);
    for (uint32_t r = 0; r < 4; ++r) {
        auto [k, v] = build_rank(4, 0, r, 0xCC);
        rk[r] = std::move(k);
        rv[r] = std::move(v);
        kp[r] = rk[r].data();
        vp[r] = rv[r].data();
        kb[r] = rk[r].size();
        vb[r] = rv[r].size();
    }
    auto export_range = [&](uint64_t a, uint64_t b) {
        size_t need = 0;
        int rc = g6kv_canon_dfl_export(L, NKV, HD, EB, 4, RING, 0, kp.data(), kb.data(), vp.data(), vb.data(), a, b, 1,
                                       nullptr, 0, &need);
        assert(rc == 1 && need > 0);
        std::vector<uint8_t> blob(need);
        rc = g6kv_canon_dfl_export(L, NKV, HD, EB, 4, RING, 0, kp.data(), kb.data(), vp.data(), vb.data(), a, b, 1,
                                   blob.data(), blob.size(), &need);
        assert(rc == 0 && need == blob.size());
        return blob;
    };
    const auto bulk = export_range(lo, mid);
    const auto tail = export_range(mid, hi);

    struct Target { uint32_t nr; int rep; };
    for (Target t : {Target{6, 1}, Target{3, 1}, Target{2, 0}, Target{1, 0}, Target{4, 0}}) {
        for (uint32_t r = 0; r < t.nr; ++r) {
            auto [wk, wv] = build_rank(t.nr, t.rep, r, 0x33);
            std::vector<uint8_t> gk(wk.size(), 0x33), gv(wv.size(), 0x33);
            uint64_t w1 = 0, w2 = 0;
            assert(g6kv_canon_dfl_import(bulk.data(), bulk.size(), L, NKV, HD, EB, t.nr, RING, t.rep, r, gk.data(),
                                         gk.size(), gv.data(), gv.size(), &w1) == 0);
            assert(g6kv_canon_dfl_import(tail.data(), tail.size(), L, NKV, HD, EB, t.nr, RING, t.rep, r, gk.data(),
                                         gk.size(), gv.data(), gv.size(), &w2) == 0);
            assert(w1 + w2 == rows);
            assert(gk == wk && gv == wv);
        }
    }
}

void test_kda_ffi_matches_kda_math_h() {
    // Cross-check Rust kda_rank_heads/nhmax against engine/kda/math.h kda::heads/nhmax directly!
    uint64_t s = 43;
    const uint32_t heads = kda::HEADS, dk = 8, dv = 8, nhist = kda::NHIST, n_streams = 3, eb = 4;
    std::vector<uint32_t> layer_ids;
    for (int li = 0; li < kda::N_KDA; ++li) layer_ids.push_back(uint32_t(li + li / 3));

    const size_t s_head = size_t(dk) * dv * eb, c_head = size_t(dk) * eb, c_rows = size_t(nhist) * n_streams;
    const size_t canon_bytes = layer_ids.size() * (size_t(heads) * s_head + c_rows * heads * c_head);
    const auto truth = rand_bytes(s, canon_bytes);

    // Reference writer using kda::heads and kda::nhmax from engine/kda/math.h
    auto ref_rank = [&](uint32_t nr, uint32_t rank) {
        const size_t nhm = size_t(kda::nhmax(int(nr)));
        const size_t r_s = nhm * s_head, r_c = c_rows * nhm * c_head;
        const size_t total = layer_ids.size() * (r_s + r_c);
        assert(g6kv_canon_kda_rank_bytes(heads, dk, dv, nhist, n_streams, eb, uint32_t(layer_ids.size()), nr) == total);
        std::vector<uint8_t> buf(total, 0);
        const size_t c_s = size_t(heads) * s_head, c_c = c_rows * heads * c_head;
        for (size_t li = 0; li < layer_ids.size(); ++li) {
            int h0 = 0, nh = 0;
            kda::heads(int(layer_ids[li]), int(rank), int(nr), h0, nh);
            const size_t src = li * (c_s + c_c);
            const size_t dst = li * (r_s + r_c);
            std::memcpy(buf.data() + dst, truth.data() + src + size_t(h0) * s_head, size_t(nh) * s_head);
            for (size_t k = 0; k < c_rows; ++k)
                std::memcpy(buf.data() + dst + r_s + k * nhm * c_head,
                            truth.data() + src + c_s + (k * heads + size_t(h0)) * c_head, size_t(nh) * c_head);
        }
        return buf;
    };

    // Export from TP=3 (uneven heads + nhmax padding) and import into TP=4, TP=2, TP=1, TP=6
    std::vector<std::vector<uint8_t>> r3(3);
    std::vector<const uint8_t*> ptrs(3);
    std::vector<size_t> lens(3);
    for (uint32_t r = 0; r < 3; ++r) {
        r3[r] = ref_rank(3, r);
        ptrs[r] = r3[r].data();
        lens[r] = r3[r].size();
    }
    size_t need = 0;
    int rc = g6kv_canon_kda_export(heads, dk, dv, nhist, n_streams, eb, layer_ids.data(), uint32_t(layer_ids.size()), 3,
                                   ptrs.data(), lens.data(), nullptr, 0, &need);
    assert(rc == 1 && need > canon_bytes);
    std::vector<uint8_t> kda_blob(need);
    rc = g6kv_canon_kda_export(heads, dk, dv, nhist, n_streams, eb, layer_ids.data(), uint32_t(layer_ids.size()), 3,
                               ptrs.data(), lens.data(), kda_blob.data(), kda_blob.size(), &need);
    assert(rc == 0 && need == kda_blob.size());

    for (uint32_t dst_nr : {4u, 2u, 1u, 6u, 3u}) {
        for (uint32_t r = 0; r < dst_nr; ++r) {
            const auto want = ref_rank(dst_nr, r);
            std::vector<uint8_t> got(want.size(), 0xEE);
            size_t written = 0;
            assert(g6kv_canon_kda_import(kda_blob.data(), kda_blob.size(), heads, dk, dv, nhist, n_streams, eb,
                                         layer_ids.data(), uint32_t(layer_ids.size()), dst_nr, r, got.data(),
                                         got.size(), &written) == 0);
            assert(written == want.size() && got == want);
        }
    }

    // Session envelope pack + unpack with KDA + extra (Flash raw rings)
    const auto extra = rand_bytes(s, 256);
    size_t ses_need = 0;
    assert(g6kv_canon_session_pack(nullptr, 0, nullptr, 0, kda_blob.data(), kda_blob.size(), extra.data(), extra.size(),
                                   nullptr, 0, &ses_need) == 1);
    std::vector<uint8_t> ses(ses_need);
    assert(g6kv_canon_session_pack(nullptr, 0, nullptr, 0, kda_blob.data(), kda_blob.size(), extra.data(), extra.size(),
                                   ses.data(), ses.size(), &ses_need) == 0);
    const uint8_t *ukv = nullptr, *udfl = nullptr, *ukda = nullptr, *uex = nullptr;
    size_t ukv_n = 1, udfl_n = 1, ukda_n = 0, uex_n = 0;
    assert(g6kv_canon_session_unpack(ses.data(), ses.size(), &ukv, &ukv_n, &udfl, &udfl_n, &ukda, &ukda_n, &uex,
                                     &uex_n) == 0);
    assert(ukv == nullptr && ukv_n == 0);
    assert(udfl == nullptr && udfl_n == 0);
    assert(ukda != nullptr && ukda_n == kda_blob.size() && std::memcmp(ukda, kda_blob.data(), ukda_n) == 0);
    assert(uex != nullptr && uex_n == extra.size() && std::memcmp(uex, extra.data(), uex_n) == 0);

    // Corrupt sub-blob rejected by pack and unpack
    std::vector<uint8_t> bad_kda = kda_blob;
    bad_kda[0] ^= 1;
    assert(g6kv_canon_session_pack(nullptr, 0, nullptr, 0, bad_kda.data(), bad_kda.size(), nullptr, 0, nullptr, 0,
                                   &ses_need) == -1);
    std::vector<uint8_t> bad_ses = ses;
    bad_ses[40] ^= 1;
    assert(g6kv_canon_session_unpack(bad_ses.data(), bad_ses.size(), &ukv, &ukv_n, &udfl, &udfl_n, &ukda, &ukda_n, &uex,
                                     &uex_n) == -1);
}

} // namespace

int main() {
    test_heads_ffi();
    test_dfl_ffi();
    test_kda_ffi_matches_kda_math_h();
    std::printf("PASS canon_state_test\n");
    return 0;
}
