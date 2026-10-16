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
#include "../loader/g6pack.h"
#include "../loader/devimage.h"
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
// converted pack's MXFP8 / int4 forms. qo adds the G6_PF_OWN=3 full-head tensors to the image.
struct V2Src
{
    std::string dir, slices, widths, dense;
    bool trellis = true, qo = false;
    const std::vector<int>* partial = nullptr;   // --v2-partial (TEST ONLY): plan these layers + globals, not the model
};

#ifdef G6_RS_LOADER
#include <atomic>
#include <chrono>
#include <mutex>
#include "../loader-rs/g6load.h"

// The Rust loader's sink: pinned staging buffers and one stream per worker into the device image. Each chunk is summed
// in staging for the weight check (weight_check.cuh).
struct RsSink
{
    uint8_t* dev = nullptr;
    std::mutex mu;
    std::vector<cudaStream_t> streams;
    std::vector<int> step;
    wcheck::Loaded sums;
    bool check = false;
    std::atomic<uint64_t> wcheck_cpu_ns{0};
    std::atomic<uint64_t> h2d_memcpy_ns{0};

    static void* alloc(void*, size_t n)
    {
        void* p = nullptr;
        return cudaHostAlloc(&p, n, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
    }
    static void release(void* ctx, void* p)
    {
        RsSink& s = *(RsSink*)ctx;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            for (cudaStream_t x : s.streams) cudaStreamSynchronize(x);
        }
        cudaFreeHost(p);
    }
    static int copy(void* ctx, int w, uint64_t off, const void* src, size_t n)
    {
        RsSink& s = *(RsSink*)ctx;
        cudaStream_t st_cur, st_prev;
        {
            std::lock_guard<std::mutex> lk(s.mu);
            while ((int)s.streams.size() <= 2 * w + 1)
            {
                cudaStream_t x;
                if (cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking) != cudaSuccess) return 2;
                s.streams.push_back(x);
            }
            if ((int)s.step.size() <= w) s.step.resize(w + 1, 0);
            const int cur = s.step[w] & 1;
            s.step[w]++;
            st_cur = s.streams[2 * w + cur];
            st_prev = s.streams[2 * w + (cur ^ 1)];
        }
        const auto th = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(s.dev + off, src, n, cudaMemcpyHostToDevice, st_cur) != cudaSuccess) return 3;
        s.h2d_memcpy_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - th).count(),
            std::memory_order_relaxed);
        if (s.check)
        {
            const auto t0 = std::chrono::steady_clock::now();
            s.sums.add(off, src, n);
            s.wcheck_cpu_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                std::memory_order_relaxed);
        }
        const auto ts = std::chrono::steady_clock::now();
        const int rc = cudaStreamSynchronize(st_prev) == cudaSuccess ? 0 : 4;
        s.h2d_memcpy_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - ts).count(),
            std::memory_order_relaxed);
        return rc;
    }
};

// pack -> one device image through the Rust loader; load(sink, stats, err) runs the load
template <class F> void rs_fill(const g6::Pack& pack, g6::DevImage& img, int rank, const char* what, F load)
{
    RsSink sink;
    void* dev = nullptr;
    const auto t_malloc = std::chrono::steady_clock::now();
    if (cudaMalloc(&dev, pack.bytes()) != cudaSuccess) die("cudaMalloc of " + std::to_string(pack.bytes()) + " bytes failed");
    const double malloc_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_malloc).count();
    sink.dev = (uint8_t*)dev;
    sink.check = wcheck::enabled();
    if (sink.check) sink.sums.begin(pack.bytes());
    const g6l_sink s{&sink, &RsSink::alloc, &RsSink::release, &RsSink::copy};
    g6l_stats st{};
    char err[1024] = "";
    const auto t_load = std::chrono::steady_clock::now();
    if (load(&s, &st, err, sizeof err)) die(std::string("Rust loader: ") + err);
    const double load_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
    for (cudaStream_t x : sink.streams) { cudaStreamSynchronize(x); cudaStreamDestroy(x); }
    img.adopt(pack, (uint8_t*)dev, pack.bytes());
    fprintf(stderr, "# rank %d: %s %.2f GB -> device image (Rust loader, %s, %u threads, %llu ranges, %.2f GB read)"
            " in %.1f s (%.2f GB/s)\n", rank, what, st.image_bytes / 1e9, st.mode ? "read" : "mmap", st.threads,
            (unsigned long long)st.ranges, st.read_bytes / 1e9, st.seconds, st.image_bytes / 1e9 / st.seconds);
    double wcheck_gpu_s = 0.0;
    if (sink.check)
    {
        const auto tw = std::chrono::steady_clock::now();
        wcheck::record(what, pack, (const uint8_t*)dev, sink.sums, rank);
        wcheck_gpu_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    }
    fprintf(stderr, "# rank %d: boot-prof rs_fill: cudaMalloc %.3f s, plan_load_wall %.3f s (sink thread-sums: wcheck_cpu %.3f s, h2d_memcpy %.3f s), wcheck_gpu %.3f s\n",
            rank, malloc_s, load_wall_s, sink.wcheck_cpu_ns.load() / 1e9, sink.h2d_memcpy_ns.load() / 1e9, wcheck_gpu_s);
}

// The pack -> one device image with the Rust loader: parallel reads, no checksum (the files are checked offline).
// G6_LOADER=cpp, or a pack streamed on stdin, takes the C++ path instead.
inline bool rs_load_image(g6::Pack& pack, g6::DevImage& img, const std::string& file, int rank)
{
    const char* sel = getenv("G6_LOADER");
    if ((sel && std::string(sel) == "cpp") || file == "-") return false;
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
    const char* tmp = getenv("TMPDIR");
    const std::string sfx = ".manifest.json";
    std::string p = std::string(tmp && *tmp ? tmp : "/tmp") + "/g6m4-rank" + std::to_string(rank) + "." + what +
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
inline void* rs_plan_release(g6::Pack& pack, int rank, const V2Src& v2, const std::vector<int>* selected = nullptr)
{
    const auto t_plan = std::chrono::steady_clock::now();
    const std::string planned = planned_manifest(rank, "release");
    char err[1024] = "";
    std::vector<uint32_t> ids;
    if (selected) {
        for (int l : *selected) {
            if (l < 0 || l >= geom::NLAYERS) die("partial release: invalid layer ID");
            ids.push_back((uint32_t)l);
        }
    }
    void* h = selected
        ? g6l_release_plan_selected(v2.dense.c_str(), v2.dir.c_str(), v2.slices.c_str(), v2.widths.c_str(), g6nr::NR, rank,
                                    v2.qo, v2.trellis, ids.data(), ids.size(), planned.c_str(), err, sizeof err)
        : g6l_release_plan(v2.dense.c_str(), v2.dir.c_str(), v2.slices.c_str(), v2.widths.c_str(), g6nr::NR, rank,
                           v2.qo, v2.trellis, planned.c_str(), err, sizeof err);
    const double g6l_plan_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_plan).count();
    if (!h) { unlink(planned.c_str()); die(std::string("release plan: ") + err); }
    const auto t_man = std::chrono::steady_clock::now();
    std::string e;
    const bool opened = pack.open_manifest(planned, e);
    unlink(planned.c_str());
    if (!opened) die(planned + ": " + e);
    const double open_man_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_man).count();
    fprintf(stderr, "# rank %d: boot-prof rs_plan_release: g6l_release_plan %.3f s, open_manifest %.3f s\n",
            rank, g6l_plan_s, open_man_s);
    return h;
}

inline void rs_load_release(g6::Pack& pack, g6::DevImage& img, int rank, const V2Src& v2)
{
    void* h = rs_plan_release(pack, rank, v2, v2.partial);
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
    if (!v2.dir.empty())
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
