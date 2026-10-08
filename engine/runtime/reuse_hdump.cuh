// K1REUSE diag arms (W38b, test only, default off; no new kernel, the default path is unchanged):
// G6_REUSE_HDUMP=<dir> + G6_REUSE_HDUMP_SEQ=<q[,q..]>: after the prefill loop of a listed seq (before any decode step),
//   every layer's cache bytes for rows [0, n) (n = the prefill end, p.first) -> <dir>/hd.<q>.r<rank>.bin, raw, page by
//   page: the target model (tag 'm') then the MTP draft model (tag 'd'). tools/reuse/hdump.py hashes the rows and
//   merges the ranks by GLOBAL row (latent rows live on their owner rank only; index rows are replicated under CP).
// G6_REUSE_ZSCRATCH=1 (with --reuse-prefix): on a hit, every layer's per-seq attention scratch is cleared before the
//   suffix prefill: indices := -1 (0xff bytes), output := 0 (max_rows rows of each binding).
// File format (little endian): per layer a header of 12 int32
//   {0x31444448 'HDD1', tag, layer index, n, page_size, np = ceil(n / page_size), table_size, page_count, cp_nr,
//    has_index, latent row bytes 656, index row bytes 132}
//   then np page records: int32 pg (block_table[lp]: >= 0 local physical page, < 0 not on this rank), int32 ip (the index
//   page read, -1 = none), then page_size * 656 latent bytes if 0 <= pg < page_count, then page_size * 132 index bytes
//   (keys [ps][128], then fp32 scales [ps]) if ip >= 0.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "rank.cuh"

namespace g6rt {
struct HDump
{
    std::string dir;                  // empty = off
    std::vector<long> seqs;
    bool zscratch = false;
    bool listed(size_t q) const
    {
        if (dir.empty()) return false;
        for (long s : seqs) if (s == (long)q) return true;
        return false;
    }
};

// strict: G6_REUSE_HDUMP unset = off; set -> non-empty dir and a non-empty list of seq numbers >= 0, else exit
inline HDump hdump_env()
{
    HDump h;
    const char* d = getenv("G6_REUSE_HDUMP");
    const char* s = getenv("G6_REUSE_HDUMP_SEQ");
    if (!d) { if (s) die("G6_REUSE_HDUMP_SEQ without G6_REUSE_HDUMP"); return h; }
    if (!*d || !s || !*s) die("G6_REUSE_HDUMP needs a dir and G6_REUSE_HDUMP_SEQ=<q[,q..]>");
    h.dir = d;
    for (const char* p = s; *p;)
    {
        char* e = nullptr;
        const long v = strtol(p, &e, 10);
        if (e == p || v < 0 || (*e && *e != ',')) die(std::string("G6_REUSE_HDUMP_SEQ: bad list '") + s + "'");
        h.seqs.push_back(v);
        p = *e ? e + 1 : e;
    }
    return h;
}

inline void hdump_put(FILE* f, const void* p, size_t n)
{
    if (fwrite(p, 1, n, f) != n) die("G6_REUSE_HDUMP: write failed");
}

// one model's layers, rows [0, n); returns bytes written
inline size_t hdump_model(FILE* f, Model& m, int tag, int n)
{
    size_t bytes = 0;
    std::vector<char> lat, ix;
    for (size_t i = 0; i < m.layers.size(); ++i)
    {
        const G6Attn& a = m.layers[i].ab.args;
        const int ps = a.page_size, np = (n + ps - 1) / ps;
#if defined(G6_CP) && G6_CP
        const int cp = a.cp_nr;
#else
        const int cp = 0;
#endif
        if (np > a.table_size) die("G6_REUSE_HDUMP: rows past the block table");
        std::vector<int> bt(a.table_size);
        G6RT_CK(cudaMemcpy(bt.data(), a.block_table, (size_t)a.table_size * sizeof(int), cudaMemcpyDeviceToHost));
        const int hdr[12] = {0x31444448, tag, (int)i, n, ps, np, a.table_size, a.page_count, cp, a.index_cache ? 1 : 0, 656, 132};
        hdump_put(f, hdr, sizeof hdr); bytes += sizeof hdr;
        lat.resize((size_t)ps * 656); ix.resize((size_t)ps * 132);
        for (int lp = 0; lp < np; ++lp)
        {
            const int pg = bt[lp];
            const int ip0 = cp ? lp : pg;     // as k_reuse_poison: CP index keys are replicated, logical-page addressed
            const int ip = (a.index_cache && ip0 >= 0 && ip0 < (cp ? a.table_size : a.page_count)) ? ip0 : -1;
            const int rec[2] = {pg, ip};
            hdump_put(f, rec, sizeof rec); bytes += sizeof rec;
            if (pg >= 0 && pg < a.page_count)
            {
                G6RT_CK(cudaMemcpy(lat.data(), static_cast<const char*>(a.mla_cache) + (size_t)pg * ps * 656, lat.size(), cudaMemcpyDeviceToHost));
                hdump_put(f, lat.data(), lat.size()); bytes += lat.size();
            }
            if (ip >= 0)
            {
                G6RT_CK(cudaMemcpy(ix.data(), static_cast<const char*>(a.index_cache) + (size_t)ip * ps * 132, ix.size(), cudaMemcpyDeviceToHost));
                hdump_put(f, ix.data(), ix.size()); bytes += ix.size();
            }
        }
    }
    return bytes;
}

// after the prefill loop of seq q (st synchronized first); d = the MTP draft model or nullptr
inline void hdump_seq(const HDump& h, Model& m, Model* d, size_t q, int rank, int n, cudaStream_t st)
{
    G6RT_CK(cudaStreamSynchronize(st));
    const std::string path = h.dir + "/hd." + std::to_string(q) + ".r" + std::to_string(rank) + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) die("G6_REUSE_HDUMP: cannot open " + path);
    size_t b = hdump_model(f, m, 'm', n);
    if (d) b += hdump_model(f, *d, 'd', n);
    if (fclose(f) != 0) die("G6_REUSE_HDUMP: close failed " + path);
    fprintf(stderr, "# hdump seq %zu rank %d rows %d file %s bytes %zu\n", q, rank, n, path.c_str(), b);
}

// G6_REUSE_ZSCRATCH: on a hit, before the suffix prefill
inline void zscratch(Model& m, cudaStream_t st)
{
    for (Layer& y : m.layers)
    {
        const G6Attn& a = y.ab.args;
        const size_t R = (size_t)y.ab.max_rows;
        G6RT_CK(cudaMemsetAsync(a.indices, 0xff, R * 2048 * sizeof(int), st));
        G6RT_CK(cudaMemsetAsync(a.output, 0, R * a.heads * 256 * 2, st));
    }
}
}   // namespace g6rt
