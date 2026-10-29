"""Compile the actual serve prefill lambda against CPU mocks: where its fences and error checks fall.

The draft's cache fill (default) checks once after the last chunk; the GR8_GLM53F_DRAFT_KV=0 draft step checks after the
main prefill and after every chunk, as before. Every case: an error word set by the main prefill or by any draft chunk
still fails the request before the lambda returns, the fences are the same in both modes (one after the main prefill,
one per chunk), and the request's work is the same calls in the same order. The words check() reads stay set until the
next request's reset (nothing in the cache fill clears one), which the mock models as one sticky word. Each case runs
with one session and with several: the prefill runs with s2on off and a clean one turns it back on.
GR8_GLM53F_PF_DRAFT_FULL (default on, with the cache fill) fills each lane's draft cache in one call: one chunk and one fence
per lane, still one check after the last. The rows each request writes reach the partial reset's dirty bound.
"""
from pathlib import Path
import subprocess
import tempfile


def main():
    source = (Path(__file__).resolve().parents[1] / 'runtime/s2/flash_native.cuh').read_text()
    start = source.index('    h.dispatch.prefill = [&](')
    prefill = source[start:source.index('\n    };', start) + len('\n    };')]
    prefill = prefill.replace('h.dispatch.prefill = [&]', 'auto prefill = [&]', 1)
    program = r'''
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
using u64 = unsigned long long;
constexpr int HID = 8, MX = 4;
struct StepIn { u64 step; int M, context; int pos[MX]; };
struct FlashPrefill { int lane; };
struct Fabric { void enqueue_step(u64, const std::vector<uint32_t>&) {} };
struct S2 { std::vector<std::vector<int>> run{{0, 1}}; } s2m;
namespace bind { struct Launch {}; struct Layer { Launch kv_a, idx_k, idx_g; }; inline void launch(const Launch&, int) {} }
struct DLayer { std::vector<bind::Layer> L = std::vector<bind::Layer>(MX + 1); };
struct DModel { int nc = 1, cur = 0; bool s2on = false; S2* s2 = &s2m; std::vector<DLayer> layers = std::vector<DLayer>(1); };
using Model = DModel;
struct Draft { DModel d; };
struct Main { int nl = 3; bool s2on = false; S2* s2 = &s2m; };
StepIn draft_in{};
struct Context {
    Main main; Draft draft; FlashPrefill prefill{0}, prefill2{1};
    Fabric prefill_fabric, prefill_fabric2, draft_fabric; int stream = 0; StepIn* draft_input = &draft_in;
} c;
struct Config { int prefill_rows = 8; bool two_lanes = true; } sc;
struct Slot { struct { u64 id; } req; struct { unsigned slot; } key; };
bool multi = false;
struct { int base(int q) const { return 2 * q; } } ks;
template <class M> void s2_solo_pages(M&, const std::vector<int>&) {}
struct Payload { std::vector<int> prompt; };
namespace glue { inline size_t head_bytes(int, int) { return 8; } }
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
unsigned word = 0;                 // every error word check() reads: set by a kernel, cleared only by the next reset
int inject = -1, chunk = 0;        // inject: -2 = the main prefill sets a word, k >= 0 = draft chunk k does
std::string events;                // M main prefill, D draft chunk, F fence, C check
int bucket_of(DModel&, int) { return 0; }
void pf_inputs_flash(FlashPrefill&, int, const int*, const int*, int) {}
void pf_enqueue_flash(Fabric&, const Main&, const FlashPrefill&, u64, int, int) {}
void main_prefill() { events += 'M'; if (inject == -2) word = 1; }
void pf_step_flash(Main&, FlashPrefill&, int, int, u64, int) { main_prefill(); }
void pf_step2_flash(Main&, FlashPrefill&, FlashPrefill&, int, int, int, int, u64, u64, int) { main_prefill(); }
void draft_chunk() { events += 'D'; if (chunk++ == inject) word = 1; }
void pf_draft_cache_flash(Draft&, int, int, int) { draft_chunk(); }
void launch_draft_step(Draft&, int, int, bool, int) { draft_chunk(); }
// the knobs the lambda reads, set per case (the engine reads each once per process): GR8_GLM53F_DRAFT_KV, and
// GR8_GLM53F_PF_DRAFT_FULL, the lane's whole draft cache fill in one call (=2, the engine's compare arm, is not run here)
namespace knob { bool draft_kv = true; inline bool glm53f_draft_kv() { return draft_kv; } }
int draft_full_knob = 1;
int pf_draft_full() { return draft_full_knob; }
bool pf_draft_full_ok(Draft&) { return true; }
unsigned diff_word = 0;
struct DraftFull { unsigned* diff = &diff_word; } draft_full;
DraftFull& pf_draft_cache_full(Draft&, FlashPrefill&, int, int, const int*, bool, int) { draft_chunk(); return draft_full; }
void launch_step_in(Model&, int) {}
void launch_draft_embed(Draft&, int, bool, int) {}
void pf_draft_full_cmp(Draft&, DraftFull&, int, int, int) {}
enum cudaMemcpyKind { cudaMemcpyDeviceToHost };
int cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return 0; }
#define G6RT_CK(x) (void)(x)
void pf_draft_flash(Draft&, FlashPrefill&, int M, int pos0, const int*, u64& dstep, const std::function<void(const StepIn&)>& run)
{
    for (int m0 = 0; m0 < M; m0 += MX) {
        const int r = std::min(MX, M - m0);
        StepIn si{++dstep, r, pos0 + m0 + r, {}};
        for (int k = 0; k < r; ++k) si.pos[k] = pos0 + m0 + k;
        run(si);
    }
}
int main() {
    u64 pfseq = 1, pfseq2 = 1, dseq = 0;
    auto fence = [&] { events += 'F'; };
    auto check = [&] { events += 'C'; assert(!c.main.s2on && !c.draft.d.s2on); require(!word, "Flash serve fabric/attention error"); };
    uint32_t capture_at = 0;          // GR8_GLM53F_CACHE: the live request's checkpoint point (0: none)
    std::vector<int> capture_tokens;
    u64 capture_req = 0;
    int dirty = 0;                    // the partial reset's bound: the end of the rows written since the last reset
    auto wrote = [&](int end) { if (dirty >= 0) dirty = std::max(dirty, end); };
    auto wrote_step = [&](const StepIn& si) {
        int e = si.context;
        for (int r = 0; r < si.M; ++r) e = std::max(e, si.pos[r] + 1);
        wrote(e);
    };
''' + prefill + r'''
    int cases = 0;
    const Slot slot{{7}, {0}};
    Payload p; p.prompt.assign(64, 1);
    struct Shape { int prefill_rows; bool two_lanes; unsigned rows; int steps, lanes; };   // steps: MX-row draft steps
    for (const Shape s : {Shape{8, false, 4, 1, 1}, Shape{16, false, 13, 4, 1}, Shape{8, true, 13, 4, 2},
                          Shape{8, true, 16, 4, 2}})
        for (const char* kv : {(const char*)nullptr, "1", "0"})
        for (const char* full : {(const char*)nullptr, "0"})
        for (const bool several : {false, true})
        {
            const bool cache = !kv || std::atoi(kv);
            if (!cache && full) continue;     // GR8_GLM53F_PF_DRAFT_FULL applies to the cache fill only
            const int chunks = cache && !full ? s.lanes : s.steps;
            knob::draft_kv = cache;
            draft_full_knob = full ? std::atoi(full) : 1;
            sc.prefill_rows = s.prefill_rows; sc.two_lanes = s.two_lanes;
            std::string clean_work;
            for (int at = -1; at < chunks; ++at)
                for (int main_err : {0, 1})
                {
                    if (main_err && at >= 0) continue;
                    word = 0; chunk = 0; events.clear(); inject = main_err ? -2 : at; dirty = 0;
                    multi = c.main.s2on = c.draft.d.s2on = several;
                    std::string err;
                    try { prefill(slot, p, 0, s.rows); } catch (const std::runtime_error& e) { err = e.what(); }
                    std::string work;   // the calls without the checks: the same in both modes
                    for (char e : events) if (e != 'C') work += e;
                    const int checks = (int)std::count(events.begin(), events.end(), 'C');
                    if (inject == -1)
                    {
                        assert(err.empty());
                        std::string want = "MF";
                        for (int k = 0; k < chunks; ++k) want += "DF";
                        assert(work == want);
                        assert(checks == (cache ? 1 : 1 + chunks));
                        assert(dirty == int(s.rows));
                        assert(events.back() == 'C');
                        assert(c.main.s2on == several && c.draft.d.s2on == several);
                        clean_work = work;
                    }
                    else
                    {
                        // the request fails before the lambda returns
                        assert(err == "Flash serve fabric/attention error");
                        if (cache) assert(work == clean_work && checks == 1);
                        else
                        {
                            // per chunk: the failure stops the request at the first check after the word was set
                            const int done = inject == -2 ? 0 : inject + 1;
                            std::string want = "MF";
                            for (int k = 0; k < done; ++k) want += "DF";
                            assert(work == want && checks == 1 + done);
                        }
                    }
                    ++cases;
                }
        }
    knob::draft_kv = true; draft_full_knob = 1;
    // GR8_GLM53F_CACHE: the quantum that ends at the checkpoint point notes its tokens [0, p] and the request; another
    // quantum leaves it pending
    word = 0; inject = -1; sc.prefill_rows = 8; sc.two_lanes = false;
    for (int i = 0; i < 64; ++i) p.prompt[i] = 100 + i;
    capture_at = 12;
    prefill(slot, p, 0, 4);
    assert(capture_at == 12 && capture_tokens.empty());
    prefill(slot, p, 8, 4);
    assert(capture_at == 0 && capture_req == 7 && capture_tokens.size() == 13 && capture_tokens.back() == 112);
    ++cases;
    std::printf("flash_serve_prefill_check: PASS %d CPU mock cases (actual serve prefill lambda)\n", cases);
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        cpp, binary = Path(tmp) / 'prefill.cpp', Path(tmp) / 'prefill'
        cpp.write_text(program)
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
                        str(cpp), '-o', str(binary)], check=True)
        run = subprocess.run([str(binary)], capture_output=True, text=True)   # stderr: the lambda's timing lines
        if run.returncode:
            raise SystemExit(run.stderr + run.stdout + f'flash_serve_prefill_check: exit {run.returncode}')
        print(run.stdout, end='')


if __name__ == '__main__':
    main()
