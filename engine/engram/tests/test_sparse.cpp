// Host regression for the exact owned-data guard called by load_sidecar before cmap/table reads.
// Small synthetic ranges; no release weights, GPU or authentic cmap needed.
#include "engram/engram_boot.h"
#include <cassert>
#include <limits>
using namespace ds41::engram;

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string path = argv[1];
    constexpr uint64_t MiB = 1ull << 20, woff = MiB, soff = 3 * MiB, rows = 512;
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, 4 * MiB) == 0);
    std::vector<uint8_t> bytes(4 * MiB, 0x35);
    auto write = [&](uint64_t off, size_t n) {
        assert(pwrite(fd, bytes.data(), n, (off_t)off) == (ssize_t)n);
        assert(fsync(fd) == 0);
    };
    Sidecar sc;
    for (auto& t : sc.t) {
        t.columns.enabled = true;
        t.runs = {{path, 0, rows, woff}};
        t.sruns = {{path, 0, rows, soff}};
    }
    std::string err;
    auto refused = [&](const char* needle) {
        err.clear(); assert(!check_column_data(sc, &err));
        assert(err.find(needle) != std::string::npos);
        assert(err.find("restage full release files") != std::string::npos);
        assert(err.find("GR8_DS_ENGRAM_COL_SHARD=0") != std::string::npos);
    };
    refused("sparse/hole");
    write(woff, rows * kRowBytes);
    refused("scale range"); // Weight bytes alone do not make a column safe.
    write(soff, rows * kScaleBytes);
    assert(check_column_data(sc, &err)); // Holes outside owned ranges are harmless.
    // A run beginning in data but crossing into a hole must also be rejected.
    sc.t[0].runs[0].rows = MiB / kRowBytes + 1;
    refused("sparse/hole");
    sc.t[0].runs[0].rows = rows;
    write(0, bytes.size());
    assert(check_column_data(sc, &err)); // Fully materialized file passes the boot guard.
    sc.t[1].sruns[0].off = 4 * MiB - rows * kScaleBytes;
    assert(check_column_data(sc, &err)); // Hole at EOF is outside the half-open range.
    assert(ftruncate(fd, 4 * MiB - 1) == 0);
    refused("truncated");
    sc.t[1].sruns[0].off = std::numeric_limits<uint64_t>::max();
    refused("truncated");
    for (auto& t : sc.t) t.columns.enabled = false;
    assert(check_column_data(sc, &err)); // Explicit legacy mode keeps its previous behavior.
    close(fd);
    puts("PASS column data boot guard: sparse weights/scales and crossing-hole ranges refused; owned-only/full data accepted; EOF/short/overflow checks; legacy bypass; both fixes named");
}
