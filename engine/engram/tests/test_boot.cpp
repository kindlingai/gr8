// Host test of the Engram boot checks (engram_boot.h): a good fake sidecar loads; every corruption is refused with a message.
// The knob reads and the sha256 known answers need no files.
//   test_boot [PREFIX.engram.cmap scratchdir]      (the cmap is the one gen_engram.py wrote; files are written, never deleted;
//                                                 without the two arguments the sidecar checks are skipped)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <initializer_list>
#include <utility>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include "engram/engram_boot.h"
using namespace ds41::engram;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

static void write_file(const std::string& p, const std::string& s)
{
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", p.c_str()); exit(2); }
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
}
static void sparse(const std::string& p, uint64_t n)
{
    int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, (off_t)n) != 0) { printf("cannot create %s\n", p.c_str()); exit(2); }
    close(fd);
}

struct Case
{
    int rank = 1, tp = 4;
    int64_t rows0 = kRows[0], rows1 = kRows[1], w_extra = 0;
    bool with_s8 = true, with_json = true, with_cmap = true, flip_cmap = false;
    const char* format = "g6engram-v1";
    int json_rank = 1, json_tp = 4;
};

// writes a sidecar for `rank` of `tp` under prefix; returns the manifest-side numbers through the files
static void build(const std::string& prefix, const std::string& cmap, const Case& c)
{
    std::string man = std::string("{\n \"format\": \"") + c.format + "\",\n \"rank\": " + std::to_string(c.json_rank) + ",\n \"tp\": " + std::to_string(c.json_tp) + ",\n \"tables\": [";
    const int64_t R[kNE] = {c.rows0, c.rows1};
    for (int e = 0; e < kNE; ++e)
    {
        const int64_t part = (R[e] + c.tp - 1) / c.tp, first = std::min<int64_t>(c.rank * part, R[e]), valid = std::min<int64_t>(part, R[e] - first);
        man += std::string(e ? "," : "") + "\n  {\"layer\": " + std::to_string(kLayerIds[e]) + ", \"rows\": " + std::to_string(R[e]) + ", \"part\": " + std::to_string(part) +
               ", \"first\": " + std::to_string(first) + ", \"valid\": " + std::to_string(valid) + ", \"dim\": 256, \"sdim\": 8, \"w_bytes\": " + std::to_string(part * 256) +
               ", \"s_bytes\": " + std::to_string(part * 8) + ", \"w_sha256\": \"00\", \"s_sha256\": \"00\"}";
        sparse(table_path(prefix, kLayerIds[e], "w8"), (uint64_t)(part * 256 + (e == 0 ? c.w_extra : 0)));
        if (c.with_s8) sparse(table_path(prefix, kLayerIds[e], "s8"), (uint64_t)(part * 8));
    }
    man += "\n ]\n}\n";
    if (c.with_json) write_file(prefix + ".engram.json", man);
    if (c.with_cmap)
    {
        std::string cm;
        std::string err;
        if (!slurp(cmap, &cm, &err)) { printf("%s\n", err.c_str()); exit(2); }
        if (c.flip_cmap) cm[100] ^= 1;
        write_file(prefix + ".engram.cmap", cm);
    }
}

static void expect_fail(const char* name, const std::string& dir, const std::string& cmap, const Case& c, const char* needle, int rank = 1, int tp = 4)
{
    const std::string prefix = dir + "/" + name;
    build(prefix, cmap, c);
    Sidecar sc;
    std::string err;
    const bool ok = load_sidecar(prefix, rank, tp, &sc, &err);
    CHECK(!ok, "%s: a corrupt sidecar was accepted", name);
    CHECK(err.find(needle) != std::string::npos, "%s: message '%s' lacks '%s'", name, err.c_str(), needle);
    printf("  %-18s refused: %s\n", name, err.c_str());
}

// read_knobs in a child with the GR8_DS_ENGRAM knobs set to env (name, value pairs; the rest unset): 0 when check(k) holds,
// 1 when it does not, 2 when the knob table refuses a value. A knob is read once per process, so each case runs in its own child.
static const char* const kKnobNames[] = {"GR8_DS_ENGRAM", "GR8_DS_ENGRAM_CACHE_MIB", "GR8_DS_ENGRAM_WAYS", "GR8_DS_ENGRAM_IO_THREADS",
                                         "GR8_DS_ENGRAM_VERBOSE", "GR8_DS_ENGRAM_LOOKAHEAD", "GR8_DS_ENGRAM_PREFIX"};
static int knob_case(std::initializer_list<std::pair<const char*, const char*>> env, bool (*check)(const Knobs&))
{
    fflush(stdout);
    const pid_t p = fork();
    if (!p)
    {
        for (const char* n : kKnobNames) { unsetenv(n); unsetenv((std::string("G6_") + (n + 4)).c_str()); }
        for (const auto& kv : env) setenv(kv.first, kv.second, 1);
        const int null = open("/dev/null", O_WRONLY);
        if (null >= 0) dup2(null, 2);   // the refusal's message
        Knobs k;
        std::string err;
        _exit(read_knobs(&k, &err) && check(k) ? 0 : 1);
    }
    int st = 0;
    waitpid(p, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

int main(int argc, char** argv)
{
    if (argc != 1 && argc != 3) { printf("usage: %s [PREFIX.engram.cmap scratchdir]\n", argv[0]); return 2; }

    // sha256 known answers
    {
        Sha256 h; h.update("abc", 3);
        CHECK(h.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
        Sha256 e;
        CHECK(e.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
        Sha256 m; std::string s(1000, 'a'); m.update(s.data(), 300); m.update(s.data() + 300, 700);
        CHECK(m.hex() == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3", "sha256 1000 x 'a' in two updates");
    }

    // knobs
    {
        CHECK(knob_case({{"GR8_DS_ENGRAM", "1"}, {"GR8_DS_ENGRAM_CACHE_MIB", "512"}, {"GR8_DS_ENGRAM_WAYS", "8"}, {"GR8_DS_ENGRAM_IO_THREADS", "32"},
                         {"GR8_DS_ENGRAM_LOOKAHEAD", "1"}, {"GR8_DS_ENGRAM_PREFIX", "/x/p"}},
                        [](const Knobs& k) { return k.on && k.lookahead && k.store.cache_bytes == (512ull << 20) && k.store.ways == 8 && k.store.io_threads == 32 && k.prefix == "/x/p"; }) == 0,
              "knobs set");
        CHECK(knob_case({{"GR8_DS_ENGRAM_LOOKAHEAD", "2"}}, [](const Knobs&) { return true; }) == 2, "lookahead 2 refused");
        CHECK(knob_case({{"GR8_DS_ENGRAM_WAYS", "17"}}, [](const Knobs&) { return true; }) == 2, "ways 17 refused");
        CHECK(knob_case({}, [](const Knobs& k) { return !k.on && !k.lookahead && !k.verbose && k.store.cache_bytes == (4ull << 30) && k.store.ways == 16 &&
                                                        k.store.io_threads == 32 && k.prefix.empty(); }) == 0,
              "knob defaults (off, lookahead off, 4 GiB, 16 ways, 32 threads, no prefix)");
    }

    if (argc == 1)
    {
        if (fails) { printf("FAILED %d checks\n", fails); return 1; }
        printf("PASS boot: knob defaults and bounds, sha256 known answers (sidecar checks skipped: no cmap given)\n");
        return 0;
    }
    const std::string cmap = argv[1], dir = argv[2];

    // good: every rank of TP4 and TP1
    for (int tp : {4, 1})
        for (int r = 0; r < tp; ++r)
        {
            Case c; c.rank = c.json_rank = r; c.tp = c.json_tp = tp;
            const std::string prefix = dir + "/good_tp" + std::to_string(tp) + "_r" + std::to_string(r);
            build(prefix, cmap, c);
            Sidecar sc; std::string err;
            CHECK(load_sidecar(prefix, r, tp, &sc, &err), "good tp%d rank %d refused: %s", tp, r, err.c_str());
            if (err.empty())
            {
                CHECK(sc.cmap.size() == kVocab && sc.cmap[kPadTok] == kPadId, "cmap loaded");
                CHECK(sc.t[0].first == std::min<int64_t>(r * ((kRows[0] + tp - 1) / tp), kRows[0]) && sc.t[0].valid > 0, "table 0 slice");
                CHECK(sc.t[1].rows == kRows[1] && sc.t[1].w8.find("L014.w8") != std::string::npos, "table 1");
            }
        }
    printf("  good: TP4 ranks 0..3 and TP1 load\n");

    { Case c; c.with_json = false; expect_fail("no_json", dir, cmap, c, "engram.json"); }
    { Case c; c.json_rank = 2; expect_fail("wrong_rank", dir, cmap, c, "rank 2 of 4"); }
    { Case c; c.json_tp = 2; expect_fail("wrong_tp", dir, cmap, c, "rank 1 of 2"); }
    { Case c; c.format = "g6engram-v0"; expect_fail("wrong_format", dir, cmap, c, "format"); }
    { Case c; c.rows0 = kRows[0] + 1; expect_fail("wrong_rows", dir, cmap, c, "rows"); }
    { Case c; c.w_extra = 4096; expect_fail("w8_size", dir, cmap, c, "bytes, the manifest says"); }
    { Case c; c.with_s8 = false; expect_fail("no_s8", dir, cmap, c, "L001.s8"); }
    { Case c; c.with_cmap = false; expect_fail("no_cmap", dir, cmap, c, "engram.cmap"); }
    { Case c; c.flip_cmap = true; expect_fail("bad_cmap_sha", dir, cmap, c, "sha256"); }
    { Case c; expect_fail("asked_rank", dir, cmap, c, "rank 1 of 4", 3, 4); }   // manifest of rank 1, process is rank 3

    if (fails) { printf("FAILED %d checks\n", fails); return 1; }
    printf("PASS boot: sidecar loads for TP4 ranks 0..3 and TP1, 10 corruptions refused with a message, knob defaults and bounds, sha256 known answers\n");
    return 0;
}
