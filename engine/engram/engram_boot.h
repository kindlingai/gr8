// Engram boot checks (host only, no CUDA): the GR8_DS_ENGRAM knobs, and the sidecar of one rank loaded fail-closed.
//   PREFIX.engram.json        the manifest (g6engram-v1): format, rank, tp, per table layer/rows/part/first/valid/dim/sdim/w_bytes/s_bytes
//   PREFIX.engram.L###.w8     E4M3 rows [part][256]   (size only: `ds41plan --verify PREFIX` checks its sha256, 47 GiB per rank)
//   PREFIX.engram.L###.s8     E8M0 scales [part][8]   (size only here; the whole file is read to the device at boot)
//   PREFIX.engram.cmap        "G6ECMAP1" token map written by gen_engram.py; sha256 of the file must equal kCmapSha256 (hash_consts.h)
// Anything missing, the wrong size, from another rank / tp / table shape, or a cmap that is not the one the constants were generated
// from, is an error string and no Engram: GR8_DS_ENGRAM=1 then stops the boot (docs/DEEPSEEK-ENGRAM.md "Boot and knobs").
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "hash_consts.h"
#include "rowstore.h"
#include "sha256.h"
#include "../runtime/json.h"
#include "../knobs-rs/gr8_knobs.h"

namespace ds41 { namespace engram {

struct Knobs
{
    bool on = false;          // GR8_DS_ENGRAM
    StoreOpts store;          // GR8_DS_ENGRAM_CACHE_MIB / _WAYS / _IO_THREADS
    bool verbose = false;     // GR8_DS_ENGRAM_VERBOSE: one stats line per fetch on stderr
    bool lookahead = false;   // GR8_DS_ENGRAM_LOOKAHEAD: inter-chunk L001 cache warm into shared RowStore; default off
    std::string prefix;       // GR8_DS_ENGRAM_PREFIX: the sidecar PREFIX (PREFIX.engram.json / .cmap / .L###.w8 / .s8); the step driver requires it when on
};

// The knob table stops the process on a value outside its range, so this returns true; err stays for the callers.
inline bool read_knobs(Knobs* k, std::string* err)
{
    (void)err;
    k->on = knob::ds_engram();
    k->store.cache_bytes = (size_t)knob::ds_engram_cache_mib() << 20;
    k->store.ways = (int)knob::ds_engram_ways();
    k->store.io_threads = (int)knob::ds_engram_io_threads();
    k->verbose = knob::ds_engram_verbose();
    k->lookahead = knob::ds_engram_lookahead();
    if (const auto& pf = knob::ds_engram_prefix()) k->prefix = *pf;
    return true;
}

struct TableInfo
{
    int64_t rows = 0, part = 0, first = 0, valid = 0;
    std::string w8, s8;       // paths
};

struct Sidecar
{
    TableInfo t[kNE];
    std::vector<uint32_t> cmap;   // kVocab entries
};

inline bool slurp(const std::string& path, std::string* out, std::string* err)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { *err = path + ": cannot open"; return false; }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    const bool bad = ferror(f);
    fclose(f);
    if (bad) { *err = path + ": read error"; return false; }
    return true;
}

inline bool file_size(const std::string& path, uint64_t* n, std::string* err)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) { *err = path + ": missing"; return false; }
    *n = (uint64_t)st.st_size;
    return true;
}

inline std::string table_path(const std::string& prefix, int layer, const char* ext)
{
    char b[32];
    snprintf(b, sizeof b, ".engram.L%03d.%s", layer, ext);
    return prefix + b;
}

// Loads and checks everything the engine needs from disk except the table bytes. rank/nr: this rank and the TP width the image was planned for.
inline bool load_sidecar(const std::string& prefix, int rank, int nr, Sidecar* sc, std::string* err)
{
    if (nr < 1 || rank < 0 || rank >= nr) { *err = "engram: bad rank/tp " + std::to_string(rank) + "/" + std::to_string(nr); return false; }
    const std::string mp = prefix + ".engram.json";
    std::string text;
    if (!slurp(mp, &text, err)) return false;
    try
    {
        const g6v::Json man = g6v::Json::parse(text);
        if (man.at("format").str() != "g6engram-v1") { *err = mp + ": format is not g6engram-v1"; return false; }
        if (man.at("rank").i64() != rank || man.at("tp").i64() != nr)
        {
            *err = mp + ": planned for rank " + std::to_string(man.at("rank").i64()) + " of " + std::to_string(man.at("tp").i64()) +
                   ", this process is rank " + std::to_string(rank) + " of " + std::to_string(nr);
            return false;
        }
        const g6v::Json& tabs = man.at("tables");
        for (int e = 0; e < kNE; ++e)
        {
            const g6v::Json* mine = nullptr;
            for (size_t i = 0; i < tabs.size(); ++i)
                if (tabs.at(i).at("layer").i64() == kLayerIds[e]) mine = &tabs.at(i);
            if (!mine) { *err = mp + ": no table for layer " + std::to_string(kLayerIds[e]); return false; }
            TableInfo& t = sc->t[e];
            t.rows = mine->at("rows").i64(); t.part = mine->at("part").i64(); t.first = mine->at("first").i64(); t.valid = mine->at("valid").i64();
            const int64_t dim = mine->at("dim").i64(), sdim = mine->at("sdim").i64();
            const int64_t wb = mine->at("w_bytes").i64(), sb = mine->at("s_bytes").i64();
            const int64_t part = (kRows[e] + nr - 1) / nr, first = std::min<int64_t>(rank * part, kRows[e]);
            const int64_t valid = std::min<int64_t>(part, kRows[e] - first);
            const std::string tag = mp + " layer " + std::to_string(kLayerIds[e]) + ": ";
            if (t.rows != kRows[e]) { *err = tag + "rows " + std::to_string(t.rows) + " != " + std::to_string(kRows[e]) + " (engram_num_embeddings)"; return false; }
            if (t.part != part || t.first != first || t.valid != valid)
            {
                *err = tag + "part/first/valid " + std::to_string(t.part) + "/" + std::to_string(t.first) + "/" + std::to_string(t.valid) +
                       " != " + std::to_string(part) + "/" + std::to_string(first) + "/" + std::to_string(valid) + " for rank " + std::to_string(rank) + " of " + std::to_string(nr);
                return false;
            }
            if (dim != kRowBytes || sdim != kScaleBytes || wb != part * kRowBytes || sb != part * kScaleBytes) { *err = tag + "row shape or byte counts are not [part][256] / [part][8]"; return false; }
            t.w8 = table_path(prefix, kLayerIds[e], "w8");
            t.s8 = table_path(prefix, kLayerIds[e], "s8");
            uint64_t n;
            if (!file_size(t.w8, &n, err)) return false;
            if ((int64_t)n != wb) { *err = t.w8 + ": " + std::to_string(n) + " bytes, the manifest says " + std::to_string(wb); return false; }
            if (!file_size(t.s8, &n, err)) return false;
            if ((int64_t)n != sb) { *err = t.s8 + ": " + std::to_string(n) + " bytes, the manifest says " + std::to_string(sb); return false; }
        }
    }
    catch (const std::exception& ex) { *err = mp + ": " + ex.what(); return false; }

    const std::string cp = prefix + ".engram.cmap";
    std::string cm;
    if (!slurp(cp, &cm, err)) { *err += " (write it with engine/engram/gen_engram.py --out-cmap)"; return false; }
    Sha256 h;
    h.update(cm.data(), cm.size());
    const std::string sha = h.hex();
    if (sha != kCmapSha256) { *err = cp + ": sha256 " + sha + " is not the " + kCmapSha256 + " the hash constants were generated from"; return false; }
    uint32_t hdr[4];
    if (cm.size() != 24 + 4ull * kVocab || memcmp(cm.data(), "G6ECMAP1", 8) != 0) { *err = cp + ": bad header or size"; return false; }
    memcpy(hdr, cm.data() + 8, 16);
    if (hdr[0] != kVocab || hdr[1] != kCVocab || hdr[2] != kPadId) { *err = cp + ": vocab/cvocab/pad fields differ from the constants"; return false; }
    sc->cmap.resize(kVocab);
    memcpy(sc->cmap.data(), cm.data() + 24, 4ull * kVocab);
    for (uint32_t v : sc->cmap) if (v >= kCVocab) { *err = cp + ": an id is outside the compressed vocabulary"; return false; }
    if (sc->cmap[kPadTok] != kPadId) { *err = cp + ": the pad token does not map to the pad id"; return false; }
    return true;
}

}}  // namespace ds41::engram
