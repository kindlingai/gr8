// gr8 M4 tensor provider: pack image in host memory -> per-tensor device copies on first use, as the
// g6rt::Find that rank.cuh binds from. One DevStore may serve several packs (m4sim's six simulated ranks): tensors
// with the same sha256 and size share one device copy, so replicated tensors are held once.
// M5 replaces this by the builder's g6::DevImage (the pack streamed straight into one device allocation).
#pragma once
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>
#include "rank.cuh"
#include "../knobs-rs/gr8_knobs.h"
#include "../loader/g6pack.h"
#include "../loader/devimage.h"
#include "../vmm.h"
#ifdef G6_RS_LOADER
#include "weight_check.cuh"
#endif

namespace g6rt
{

struct DevStore
{
    std::map<std::string, void*> by_sha;
    size_t bytes = 0, shared_bytes = 0;

    const void* get(const g6::Pack& pack, const g6::Entry& e)
    {
        const std::string key = std::string((const char*)e.sha256, 32) + ":" + std::to_string(e.nbytes);
        auto it = by_sha.find(key);
        if (it != by_sha.end()) { shared_bytes += e.nbytes; return it->second; }
        void* d;
        G6RT_CK(cudaMalloc(&d, e.nbytes));
        G6RT_CK(g6h2d::copy(d, pack.base() + e.offset, e.nbytes));   // pageable source: pinned staging (h2d.h)
        by_sha[key] = d;
        bytes += e.nbytes;
        return d;
    }
};

// the pack and the store must outlive the returned function
inline Find pack_find(const g6::Pack& pack, DevStore& store)
{
    return [&pack, &store](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        t.p = store.get(pack, *e);
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}

// the same from a device image of the whole pack (no host copy, no per-tensor copies): what a Spark rank uses
inline Find image_find(const g6::Pack& pack, const g6::DevImage& img)
{
    return [&pack, &img](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        t.p = img.ptr(*e);
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}
// exlr8 v2 experts from the raw width files (--v2 DIR --v2-slices DIR --v2-widths FILE). With dense (--v2-dense DIR),
// every other tensor comes from the release's dense and vocab files too, with no base pack. trellis keeps the trellis
// matrices as the release's entries (engine/kernels/dtr.cuh, --v2-dense-form trellis); false rounds them to a
// converted pack's MXFP8 / int4 forms. qo adds the GR8_PF_OWN=3 full-head tensors to the image. layout (--layout FILE)
// says which pieces each rank holds (engine/loader-rs/src/layout.rs); empty: the formulas.
struct V2Src
{
    std::string dir, slices, widths, dense, layout;
    bool trellis = true, qo = false;
};

#ifdef G6_RS_LOADER
#include <mutex>
#include "../loader-rs/g6load.h"
#if G6_MODEL_DS41
#include "../loader-rs/ds41.h"
#endif

// The Rust loader's sink: pinned staging buffers and one stream per worker into the device image. Each chunk is summed
// in staging for the weight check (weight_check.cuh).
struct RsSink
{
    uint8_t* dev = nullptr;
    std::mutex mu;
    std::vector<cudaStream_t> streams;
    wcheck::Loaded sums;
    bool check = false;

    static void* alloc(void*, size_t n)
    {
        void* p = nullptr;
        return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
    }
    static void release(void*, void* p) { cudaFreeHost(p); }
    static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n)
    {
        RsSink& s = *(RsSink*)ctx;
        cudaStream_t st;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            while ((int)s.streams.size() <= w)
            {
                cudaStream_t x;
                if (cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking) != cudaSuccess) return 2;
                s.streams.push_back(x);
            }
            st = s.streams[w];
        }
        if (s.check) s.sums.add(off, src, n);
        if (cudaMemcpyAsync(s.dev + off, src, n, cudaMemcpyHostToDevice, st) != cudaSuccess) return 3;
        return cudaStreamSynchronize(st) == cudaSuccess ? 0 : 4;
    }
};

// pack -> one device image through the Rust loader; load(sink, stats, err) runs the load
template <class F> void rs_fill(const g6::Pack& pack, g6::DevImage& img, int rank, const char* what, F load)
{
    RsSink sink;
    void* dev = nullptr;
    // GR8_IMG_VMM=1 (A/B): the image as cuMemCreate chunks of the recommended granularity mapped into one range
    // (engine/vmm.h), =N: chunks of N MB. Same bytes at the same offsets; only the physical
    // backing (the GPU's page size) can differ. Falls back to cudaMalloc, with a line saying why.
    static const long vmm = (long)knob::img_vmm();
    void (*release)(void*) = nullptr;
    if (vmm > 0)
    {
        int d = 0;
        size_t gmin = 0, grec = 0;
        cudaGetDevice(&d);
        g6vmm::granularity(d, gmin, grec);
        std::string why;
        const size_t chunk = vmm > 1 ? (size_t)vmm << 20 : 0;
        if ((dev = g6vmm::alloc(pack.bytes(), chunk, why))) release = [](void* p) { g6vmm::release(p); };
        fprintf(stderr, "# rank %d: GR8_IMG_VMM=%ld: %s (granularity min %zu KB, recommended %zu KB)\n", rank, vmm,
                dev ? ("image in chunks of " + std::to_string((chunk ? (chunk + gmin - 1) / gmin * gmin : grec) >> 20) + " MB").c_str()
                    : ("cudaMalloc instead: " + why).c_str(), gmin >> 10, grec >> 10);
    }
    if (!dev && cudaMalloc(&dev, pack.bytes()) != cudaSuccess) die("cudaMalloc of " + std::to_string(pack.bytes()) + " bytes failed");
    sink.dev = (uint8_t*)dev;
    sink.check = wcheck::enabled();
    if (sink.check) sink.sums.begin(pack.bytes());
    const g6l_sink s{&sink, &RsSink::alloc, &RsSink::release, &RsSink::copy};
    g6l_stats st{};
    char err[1024] = "";
    if (load(&s, &st, err, sizeof err)) die(std::string("Rust loader: ") + err);
    for (cudaStream_t x : sink.streams) cudaStreamDestroy(x);
    img.adopt(pack, (uint8_t*)dev, pack.bytes(), release);
    fprintf(stderr, "# rank %d: %s %.2f GB -> device image (Rust loader, %s, %u threads, %llu ranges, %.2f GB read)"
            " in %.1f s (%.2f GB/s)\n", rank, what, st.image_bytes / 1e9, st.mode ? "read" : "mmap", st.threads,
            (unsigned long long)st.ranges, st.read_bytes / 1e9, st.seconds, st.image_bytes / 1e9 / st.seconds);
    if (sink.check) wcheck::record(what, pack, (const uint8_t*)dev, sink.sums, rank);
}

// The pack -> one device image with the Rust loader: parallel reads, no checksum (the files are checked offline).
// GR8_LOADER=cpp, or a pack streamed on stdin, takes the C++ path instead.
inline bool rs_load_image(g6::Pack& pack, g6::DevImage& img, const std::string& file, int rank)
{
    if (knob::loader() == "cpp" || file == "-") return false;
    std::vector<g6l_extent> ext;
    for (const g6::Entry& e : pack.entries()) ext.push_back({e.offset, e.nbytes});
    rs_fill(pack, img, rank, "pack", [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
        return g6l_load(file.c_str(), pack.bytes(), ext.data(), ext.size(), "", "", s, st, err, n);
    });
    return true;
}

// The planned manifest of a rank image: a new file of this process in $TMPDIR (default /tmp), made with mkstemps, so
// another account on the box can neither hold the name nor swap the file between the loader's write and our read.
// The caller removes it once it has been read.
inline std::string planned_manifest(int rank, const char* what)
{
    const std::string sfx = ".manifest.json";
    std::string p = knob::tmpdir() + "/g6m4-rank" + std::to_string(rank) + "." + what +
                    ".XXXXXX" + sfx;
    const int fd = mkstemps(&p[0], (int)sfx.size());
    if (fd < 0) die(p + ": " + strerror(errno));
    close(fd);
    return p;
}

// The base pack with its routed experts replaced by exlr8 v2 experts read from the raw width files (--v2): the Rust
// loader plans the image from the base manifest, the index slices and the width manifest, and writes the image's
// manifest to $TMPDIR (default /tmp) for `pack` to read. Its sha256 fields are zero, so only this loader loads it.
inline void rs_load_v2(g6::Pack& pack, g6::DevImage& img, const std::string& manifest, const std::string& file, int rank,
                       const V2Src& v2)
{
    const std::string planned = planned_manifest(rank, "v2");
    char err[1024] = "";
    void* h = g6l_v2_plan(manifest.c_str(), file.c_str(), v2.dir.c_str(), v2.slices.c_str(), v2.widths.c_str(),
                          planned.c_str(), err, sizeof err);
    if (!h) { unlink(planned.c_str()); die(std::string("v2 plan: ") + err); }
    std::string e;
    const bool opened = pack.open_manifest(planned, e);
    unlink(planned.c_str());
    if (!opened) die(planned + ": " + e);
    if (pack.rank() != rank) die(manifest + ": manifest says rank " + std::to_string(pack.rank()));
    rs_fill(pack, img, rank, "base pack + v2 files", [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
        return g6l_plan_load(h, s, st, err, n);
    });
    g6l_plan_free(h);
}

// Plans rank `rank`'s image from the release alone (--v2-dense) and opens its manifest in pack. Returns the plan handle.
inline void* rs_plan_release(g6::Pack& pack, int rank, const V2Src& v2)
{
    const std::string planned = planned_manifest(rank, "release");
    char err[1024] = "";
#if G6_MODEL_DS41
    // DeepSeek-V4.1-Flash-1017: --v2-dense names the exlr8 native release; the planner also writes the Engram tables'
    // runs, to GR8_DS_ENGRAM_PREFIX.engram.json for the Engram boot, or to a file it removes again
    const auto eprefix = knob::ds_engram_prefix();
    const std::string engram = eprefix ? *eprefix + ".engram.json" : planned_manifest(rank, "engram");
    void* h = g6l_ds41_plan(v2.dense.c_str(), g6nr::NR, rank, planned.c_str(), engram.c_str(), err, sizeof err);
    if (!eprefix) unlink(engram.c_str());
#else
    void* h = g6l_release_plan(v2.dense.c_str(), v2.dir.c_str(), v2.slices.c_str(), v2.widths.c_str(), g6nr::NR, rank,
                               v2.qo, v2.trellis, v2.layout.c_str(), planned.c_str(), err, sizeof err);
#endif
    if (!h) { unlink(planned.c_str()); die(std::string("release plan: ") + err); }
    std::string e;
    const bool opened = pack.open_manifest(planned, e);
    unlink(planned.c_str());
    if (!opened) die(planned + ": " + e);
    return h;
}

inline void rs_load_release(g6::Pack& pack, g6::DevImage& img, int rank, const V2Src& v2)
{
    void* h = rs_plan_release(pack, rank, v2);
    rs_fill(pack, img, rank, "exlr8 release", [&](const g6l_sink* s, g6l_stats* st, char* err, size_t n) {
        return g6l_plan_load(h, s, st, err, n);
    });
    g6l_plan_free(h);
}

// m4sim: one planned image per simulated rank, its tensors read from the release one at a time on first use. Tensors
// with the same bytes share one device copy (the replicated ones), as DevStore does for packs.
struct PlanStore
{
    std::vector<void*> plans;
    std::map<std::string, void*> by_name, by_content;
    size_t bytes = 0, shared_bytes = 0;

    const void* get(int r, const g6::Entry& e)
    {
        void*& got = by_name[std::to_string(r) + ":" + e.name];
        if (!got) got = read(r, e);
        return got;
    }
    void* read(int r, const g6::Entry& e)
    {
        std::vector<uint8_t> h(e.nbytes);
        char err[1024] = "";
        if (e.nbytes && g6l_plan_read(plans[r], e.offset, e.nbytes, h.data(), err, sizeof err)) die(e.name + ": " + err);
        uint64_t x = 0xcbf29ce484222325ull;       // FNV-1a over the bytes, plus the size, as the sharing key
        for (uint8_t b : h) x = (x ^ b) * 0x100000001b3ull;
        const std::string key = std::to_string(x) + ":" + std::to_string(e.nbytes);
        auto it = by_content.find(key);
        if (it != by_content.end()) { shared_bytes += e.nbytes; return it->second; }
        void* d;
        G6RT_CK(cudaMalloc(&d, e.nbytes ? e.nbytes : 1));
        G6RT_CK(g6h2d::copy(d, h.data(), e.nbytes));
        by_content[key] = d;
        bytes += e.nbytes;
        return d;
    }
};

inline Find plan_find(const g6::Pack& pack, PlanStore& store, int r)
{
    return [&pack, &store, r](const std::string& name) {
        TensorRef t;
        const g6::Entry* e = pack.find(name);
        if (!e) return t;
        if (e->shape.size() > 4) die("too many dims: " + name);
        t.p = store.get(r, *e);
        t.nd = (int)e->shape.size();
        for (int i = 0; i < t.nd; ++i) t.shape[i] = e->shape[i];
        t.bf16 = e->dtype == g6::DType::BF16;
        t.fp32 = e->dtype == g6::DType::F32;
        t.fmt = e->expert_format;
        return t;
    };
}
#endif

// manifest + pack stream ("-" = stdin) -> device image
inline void load_image(g6::Pack& pack, g6::DevImage& img, const std::string& manifest, const std::string& file, int rank,
                       bool hash = true, const V2Src& v2 = {})
{
#if G6_MODEL_DS41
    if (!v2.dir.empty() || !v2.dense.empty())   // DS41: --v2-dense RELEASE_DIR alone (rs_plan_release -> g6l_ds41_plan)
#else
    if (!v2.dir.empty())
#endif
    {
#ifdef G6_RS_LOADER
        if (!v2.dense.empty()) return rs_load_release(pack, img, rank, v2);
        return rs_load_v2(pack, img, manifest, file, rank, v2);
#else
        die("--v2 needs the Rust loader (scripts/build.sh links it when cargo is installed)");
#endif
    }
    std::string err;
    if (!pack.open_manifest(manifest, err)) die(err);
    if (pack.rank() != rank) die(manifest + ": manifest says rank " + std::to_string(pack.rank()));
#ifdef G6_RS_LOADER
    if (rs_load_image(pack, img, file, rank)) return;
#endif
    g6::LoadOpts lo;
    g6::LoadStats st;
    if (!hash) lo.hash_threads = 0;
    if (!img.load(pack, file, lo, st, err)) die(err);
    fprintf(stderr, "# rank %d: pack %.2f GB -> device image%s in", rank, st.bytes / 1e9, hash ? ", sha256-checked," : " (NO checksum)");
    fprintf(stderr, " %.1f s (%.2f GB/s; copy %.1f s, stall %.1f s)\n", st.seconds, st.gb_per_s, st.copy_seconds, st.stall_seconds);
}

inline void load_pack(g6::Pack& pack, const std::string& manifest, const std::string& file, int rank)
{
    std::string err;
    if (!pack.open_manifest(manifest, err)) die(err);
    if (pack.rank() != rank) die(manifest + ": manifest says rank " + std::to_string(pack.rank()));
    g6::LoadOpts lo;
    g6::LoadStats st;
    if (!pack.load(file, lo, st, err)) die(err);
    fprintf(stderr, "# rank %d: pack %.2f GB loaded + sha256-checked in %.1f s (%.2f GB/s)\n", rank, st.bytes / 1e9, st.seconds,
            st.gb_per_s);
}

// comma-separated integers; "@file" reads them (comma or white space separated) from a file; "all" = 0..77
inline std::vector<int> parse_ints(std::string s)
{
    std::vector<int> v;
    if (s == "all") { for (int i = 0; i < geom::L_MAIN; ++i) v.push_back(i); return v; }
    if (!s.empty() && s[0] == '@')
    {
        FILE* f = fopen(s.c_str() + 1, "r");
        if (!f) die("cannot read " + s.substr(1));
        int x;
        while (fscanf(f, "%d%*[, \n\t]", &x) == 1) v.push_back(x);
        fclose(f);
        return v;
    }
    for (size_t p = 0; p < s.size();)
    {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        v.push_back(atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

}   // namespace g6rt
