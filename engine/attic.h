// spark-attic (github.com/bertholomus/spark-attic): the GB10's display carveout, the ~2 GiB the driver reserves for a display and never uses on
// a headless box, lent by atticd as whole 2 MiB slices over a socket. GR8_ATTIC_SOCK names atticd's socket; unset, nothing
// here runs and nothing is printed. With it, the rank asks its own box's atticd for one slice that spans the weight
// image's G.embed (the full vocabulary in BF16 on every rank, 1,815 MiB, of which a step reads only its tokens' rows),
// imports the slice's descriptor and maps the image around it (engine/vmm.h alloc_around): the image takes that much
// less ordinary memory, so a larger --cap fits in the same MemAvailable. Same bytes at the same offsets; the KV pool
// stays in ordinary memory. Fail open: no daemon, a refusal, a short lend, a key we cannot send or a slice the driver
// will not map build the image as GR8_IMG_VMM says, after one line saying why.
// The protocol is KindlingAI's dispramd socket protocol (Matt Mastracci's dispram, part of Kindling Spark OS), which
// atticd speaks; the requests are byte for byte those of spark-attic's own clients. AF_UNIX SOCK_SEQPACKET, one flat
// JSON object a message, the slice's descriptor beside the
// alloc reply (SCM_RIGHTS). The socket stays open for the life of the process: atticd frees a client's slices when its
// socket closes and the memory is unmapped. The driver API comes through cudaGetDriverEntryPointByVersion, as in vmm.h.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "attic_proto.h"
#include "knobs-rs/gr8_knobs.h"
#include "loader/g6pack.h"
#include "vmm.h"

namespace g6attic
{

// cuMemImportFromShareableHandle (nullptr without), and the minimum granularity of memory shared through a POSIX
// descriptor (0 without)
using ImportFn = CUresult (*)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
inline ImportFn import_fn()
{
    static const ImportFn f = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        if (cudaGetDriverEntryPointByVersion("cuMemImportFromShareableHandle", &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess)
            return ImportFn(nullptr);
        return reinterpret_cast<ImportFn>(p);
    }();
    return f;
}
inline size_t fd_granularity(int dev)
{
    if (!g6vmm::api().ok) return 0;
    CUmemAllocationProp p = g6vmm::prop(dev);
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t g = 0;
    return g6vmm::api().gran(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS ? g : 0;
}

// The image outside the slice, in physical allocations cut at multiples of this (engine/vmm.h alloc_around): on a GB10
// each one costs the kernel ~25 KB of slab, so chunks of the granularity would take back half of what the slice frees.
constexpr size_t PIECE = size_t(1) << 30;

// The lend this process holds: atticd's socket, open until exit (closing it would give the slice back)
inline Client& held()
{
    static Client c;
    return c;
}

// The image of `pack` with [lo, hi) (the whole slices that span G.embed) mapped from one slice atticd lends, or nullptr
// and why. Nothing is held on failure: the socket is closed and every mapping undone.
inline void* place(const g6::Pack& pack, const g6::Entry& e, const std::string& path, size_t& lo, size_t& hi, std::string& why)
{
    int dev = 0;
    cudaGetDevice(&dev);
    size_t gmin = 0, grec = 0;
    g6vmm::granularity(dev, gmin, grec);
    const size_t gfd = fd_granularity(dev);
    if (!gmin || !gfd || !import_fn()) { why = "no VMM support for memory shared through a descriptor"; return nullptr; }
    const size_t chunk = (grec + gmin - 1) / gmin * gmin;        // the granularity g6vmm::alloc_around cuts at
    const size_t g = std::max(std::max(chunk, gfd), SLICE);      // powers of two: the largest is a multiple of the rest
    if (g % chunk || g % gfd || g % SLICE) { why = "the granularities do not divide each other"; return nullptr; }
    lo = e.offset / g * g;
    hi = (e.offset + e.nbytes + g - 1) / g * g;
    std::string key;
    if (!key_of(key, why)) return nullptr;
    Client c;
    if (!c.connect(path, why)) return nullptr;
    std::string rep;
    std::vector<uint64_t> got;
    if (!c.ask(request(key, "info"), rep, nullptr, why) || !ok_reply(rep, {"largest"}, got, why)) { c.close(); return nullptr; }
    if (got[0] < hi - lo)
    {
        why = "a short lend: atticd's largest slice is " + std::to_string(got[0] >> 20) + " MiB, G.embed needs " +
              std::to_string((hi - lo) >> 20);
        c.close();
        return nullptr;
    }
    int fd = -1;
    if (!c.ask(request(key, "alloc", hi - lo), rep, &fd, why)) { c.close(); return nullptr; }
    if (!ok_reply(rep, {"size"}, got, why) || got[0] != hi - lo || fd < 0)
    {
        if (why.empty())
            why = fd < 0 ? std::string("atticd's alloc reply carried no descriptor")
                         : "atticd lent " + std::to_string(got[0]) + " bytes, we asked for " + std::to_string(hi - lo);
        if (fd >= 0) ::close(fd);
        c.close();
        return nullptr;
    }
    CUmemGenericAllocationHandle h = 0;
    const CUresult r = import_fn()(&h, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    ::close(fd);                         // the handle holds the memory, the socket the lend
    if (r) { why = "cuMemImportFromShareableHandle " + std::to_string(r); c.close(); return nullptr; }
    void* p = g6vmm::alloc_around(pack.bytes(), lo, hi, h, PIECE, why);
    if (!p) { c.close(); return nullptr; }
    held() = c;
    return p;
}

// The device image of `pack` with G.embed in a slice of the display carveout (see the top), with `release` set to free it,
// or nullptr: GR8_ATTIC_SOCK unset (nothing is printed), a pack without G.embed (the drafter's and the QO packs read the
// target's), an image after the first (one lend a process), or a failed lend (one line says why).
inline void* image(const g6::Pack& pack, int rank, void (*&release)(void*))
{
    static const auto sock = knob::attic_sock();
    if (!sock) return nullptr;
    const g6::Entry* e = pack.find("G.embed");
    static bool asked = false;
    if (!e || asked) return nullptr;
    asked = true;
    size_t lo = 0, hi = 0;
    std::string why;
    void* dev = place(pack, *e, *sock, lo, hi, why);
    if (!dev)
    {
        fprintf(stderr, "# rank %d: attic: no display carveout lent (%s): the image as GR8_IMG_VMM says\n", rank, why.c_str());
        return nullptr;
    }
    release = [](void* p) { g6vmm::release(p); };
    fprintf(stderr, "# rank %d: attic: G.embed (%.1f MiB at image offset %llu) in a %zu MiB slice of the display carveout from "
            "%s, image bytes [%zu, %zu) MiB; the rest of the %.2f GB image in %zu allocations of up to %zu MiB\n", rank,
            e->nbytes / 1048576.0, (unsigned long long)e->offset, (hi - lo) >> 20, sock->c_str(), lo >> 20, hi >> 20,
            pack.bytes() / 1e9, g6vmm::mappings(dev) - 1, PIECE >> 20);
    return dev;
}

}   // namespace g6attic
