// CPU test of the Engram hash against the official NgramHashState.
//   test_hash <PREFIX.engram.cmap> <vectors.bin>        (both written by engine/engram/gen_engram.py)
// Checks: cmap file header + sha256 text matches hash_consts.h is the generator's job; here we check that every id of every
// position of every vector sequence is bit-identical to the reference, building `win` the way the driver does (raw tokens,
// -1 before the start), and the table-row bounds against the model geometry.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../hash_consts.h"
#include "../../model_geom.h"

using namespace ds41::engram;

static std::vector<unsigned char> slurp(const char* p)
{
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    std::vector<unsigned char> b;
    unsigned char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    fclose(f);
    return b;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: test_hash cmap vectors\n"); return 2; }
    // geometry agreement (model_geom_ds41.h is the engine's one definition)
    static_assert(geom::ENGRAM_ROWS_L0 == 384006168 && geom::ENGRAM_ROWS_L1 == 384016682, "rows");
    if (kRows[0] != geom::ENGRAM_ROWS_L0 || kRows[1] != geom::ENGRAM_ROWS_L1) { fprintf(stderr, "FAIL rows vs geom\n"); return 1; }
    if (kLayerIds[0] != geom::ENGRAM_L0 || kLayerIds[1] != geom::ENGRAM_L1) { fprintf(stderr, "FAIL layer ids vs geom\n"); return 1; }
    if (kCVocab != (uint32_t)geom::ENGRAM_CVOCAB || kPadTok != (uint32_t)geom::ENGRAM_PAD) { fprintf(stderr, "FAIL vocab/pad vs geom\n"); return 1; }
    if (kEmbDim != geom::ENGRAM_DIM * geom::NGRAM_TABLES) { fprintf(stderr, "FAIL emb dim vs geom\n"); return 1; }

    auto cm = slurp(argv[1]);
    if (cm.size() < 24 || memcmp(cm.data(), "G6ECMAP1", 8) != 0) { fprintf(stderr, "FAIL cmap header\n"); return 1; }
    uint32_t hdr[4];
    memcpy(hdr, cm.data() + 8, 16);
    if (hdr[0] != kVocab || hdr[1] != kCVocab || hdr[2] != kPadId || cm.size() != 24 + 4ull * kVocab) { fprintf(stderr, "FAIL cmap fields\n"); return 1; }
    std::vector<uint32_t> cmap(kVocab);
    memcpy(cmap.data(), cm.data() + 24, 4ull * kVocab);
    uint32_t mx = 0;
    for (uint32_t v : cmap) if (v > mx) mx = v;
    if (mx + 1 != kCVocab) { fprintf(stderr, "FAIL cmap count %u != %u\n", mx + 1, kCVocab); return 1; }
    if (cmap[kPadTok] != kPadId) { fprintf(stderr, "FAIL pad id\n"); return 1; }

    auto vb = slurp(argv[2]);
    size_t o = 0;
    auto rd = [&](void* d, size_t n) { if (o + n > vb.size()) { fprintf(stderr, "FAIL short vectors\n"); exit(1); } memcpy(d, vb.data() + o, n); o += n; };
    char magic[8];
    rd(magic, 8);
    if (memcmp(magic, "G6EVEC01", 8) != 0) { fprintf(stderr, "FAIL vectors magic\n"); return 1; }
    uint32_t nseq;
    rd(&nseq, 4);
    long long npos = 0, nid = 0, bad = 0;
    for (uint32_t s = 0; s < nseq; ++s)
    {
        uint32_t n;
        rd(&n, 4);
        std::vector<int32_t> tok(n);
        std::vector<uint32_t> want((size_t)n * kNE * kIds);
        rd(tok.data(), 4ull * n);
        rd(want.data(), 4ull * want.size());
        for (uint32_t m = 0; m < n; ++m)
        {
            int32_t win[kNGram];
            for (int k = 0; k < kNGram; ++k) win[k] = (int32_t)m - k >= 0 ? tok[m - k] : -1;
            uint32_t got[kNE][kIds];
            if (hash_row(kHashParams, cmap.data(), kVocab, win, got)) { fprintf(stderr, "FAIL seq %u pos %u flagged bad\n", s, m); return 1; }
            for (int e = 0; e < kNE; ++e)
                for (int j = 0; j < kIds; ++j)
                {
                    ++nid;
                    const uint32_t w = want[((size_t)m * kNE + e) * kIds + j];
                    if (got[e][j] != w || (int64_t)got[e][j] >= kRows[e])
                    {
                        if (bad++ < 10) fprintf(stderr, "MISMATCH seq %u pos %u layer %d id %d: got %u want %u\n", s, m, e, j, got[e][j], w);
                    }
                }
            ++npos;
        }
    }
    // an out-of-range id is reported, not read
    {
        int32_t win[kNGram] = {(int32_t)kVocab, 1, 2, 3};
        uint32_t ids[kNE][kIds];
        if (!hash_row(kHashParams, cmap.data(), kVocab, win, ids)) { fprintf(stderr, "FAIL out-of-range token not flagged\n"); return 1; }
    }
    if (bad) { fprintf(stderr, "FAIL %lld mismatches of %lld ids\n", bad, nid); return 1; }
    printf("PASS hash: %u sequences, %lld positions, %lld ids bit-identical to the official NgramHashState; cmap %u -> %u ids\n", nseq, npos, nid, kVocab, kCVocab);
    return 0;
}
