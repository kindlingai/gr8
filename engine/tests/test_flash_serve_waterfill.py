"""Compile the solo and Flash S2 draft graph-capture and draft_lp hooks against CPU mocks and libg6host.a."""
from pathlib import Path
import subprocess
import tempfile


def main():
    engine = Path(__file__).resolve().parents[1]
    repo = engine.parent
    g6m4 = (engine / 'runtime/g6m4.cu').read_text()
    flash_native = (engine / 'runtime/s2/flash_native.cuh').read_text()

    # 1. Verify FlashModule gates o.draft_lp on knob::s2_shared_waterfill() in serve_flash.
    reg_line = '    if (knob::s2_shared_waterfill()) o.draft_lp = mod::draft_lp<M, &M::on_draft_lp>;'
    assert reg_line in flash_native

    # 2. Extract the solo draft graph-capture block from g6m4.cu (s2on == false).
    solo_anchor = '        if (mtp && D.d.nbk != m.nbk) die("draft and target context buckets differ");'
    solo_start = g6m4.index(solo_anchor)
    solo_end = g6m4.index('\n    }\n    FILE* dfl_dump', solo_start)
    solo_block = g6m4[solo_start:solo_end]

    # 3. Extract capture_all and FlashModule::on_launch / on_draft_lp from flash_native.cuh.
    cap_start = flash_native.index('    auto capture_all = [&] {')
    cap_end = flash_native.index('\n    };\n    auto release =', cap_start) + len('\n    };')
    cap_block = flash_native[cap_start:cap_end]

    launch_head = '        std::vector<int> on_launch(const Launch& l) {'
    launch_start = flash_native.index(launch_head)
    launch_end = flash_native.index('\n        g6rt::LpRow on_logprob', launch_start)
    launch_block = flash_native[launch_start:launch_end].replace(
        launch_head, '    auto on_launch = [&](const Launch& l) -> std::vector<int> {', 1
    ) + ';'

    dlp_head = '        float on_draft_lp(int row) {'
    dlp_start = flash_native.index(dlp_head)
    dlp_end = flash_native.index('\n        void on_prefill', dlp_start)
    dlp_block = flash_native[dlp_start:dlp_end].replace(
        dlp_head, '    auto on_draft_lp = [&](int row) -> float {', 1
    ) + ';'

    host_lib = engine / 'target/release/libg6host.a'
    if not host_lib.exists():
        subprocess.run(
            ['cargo', 'build', '--release', '--quiet', '--manifest-path', str(engine / 'Cargo.toml')],
            cwd=str(repo),
            check=True,
        )

    program = f'''
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "{engine / 'runtime/draft_shape.h'}"
#include "{engine / 'runtime/s2/flash_ring.h'}"

using g6s2::FlashShapes;
using g6s2::flash_shapes;
using u64 = unsigned long long;
using cudaGraphExec_t = int;
constexpr int MX = 32, NBK = 2, S2N = 8, HID = 8;

void die(const char* msg) {{ throw std::runtime_error(std::string("g6: ") + msg); }}
void require(bool ok, const char* msg) {{ if (!ok) throw std::runtime_error(msg); }}
void require(bool ok, const std::string& msg) {{ if (!ok) throw std::runtime_error(msg); }}
#define G6RT_CK(x) (void)(x)
int cudaMemGetInfo(size_t* f, size_t* t) {{ *f = 1024; *t = 2048; return 0; }}
int cudaGraphLaunch(cudaGraphExec_t, int) {{ return 0; }}

namespace s2 {{
    struct Step {{
        enum Kind {{ MAIN = 0, DRAFT = 1, CHAIN = 2 }};
        static constexpr int TAIL = 3;
        int kind = MAIN;
        std::vector<int> sess, row0, rows;
    }};
    inline int row_class(int m) {{ return m; }}
}}
namespace knob {{
    bool wf = true;
    bool ng = false;
    inline bool s2_adapt() {{ return false; }}
    inline bool s2_shared_waterfill() {{ return wf; }}
    inline bool ngram() {{ return ng; }}
}}
namespace mod {{
    template <class T, float (T::*F)(int)>
    float call_draft_lp(void* self, int row) {{ return (static_cast<T*>(self)->*F)(row); }}
    template <class T, float (T::*F)(int)>
    constexpr float (*draft_lp)(void*, int) = &call_draft_lp<T, F>;
}}
namespace glue {{ inline size_t head_bytes(int m, int) {{ return size_t(m) * 8; }} }}
namespace g6rt {{ inline void evprof_read(int, int, int) {{}} }}

struct S2In {{ int ns = 0, cls = 0; int sess[S2N]{{}}, row0[S2N]{{}}, rows[S2N]{{}}, ctx[S2N]{{}}; }};
struct S2State {{ S2In h{{}}; int bucket = 0; int bk[S2N]{{}}; int pool_pages = 16; }};
struct Model {{
    int nbk = NBK, cur = 0, nc = 1, rank = 0;
    int bk[NBK] = {{2048, 8192}};
    bool s2on = false;
    int* dfl_tap = nullptr;
    S2State* s2 = nullptr;
    int ncoll() const {{ return 3; }}
}};
struct Draft {{ Model d; }};
struct StepIn {{ u64 step = 0; int M = 0, context = 0, page_bound = 0; int pos[MX]{{}}, sess[MX]{{}}; }};
struct StepOut {{ unsigned attn_err = 0; int tok[MX]{{}}; float lp[MX]{{}}; }};
struct LpStep {{ int dummy = 0; }};
struct Sampling {{ int lp = -1, plp = -1; }};
struct Fabric {{ void enqueue_step(u64, const std::vector<uint32_t>&) {{}} }};
struct Launch {{
    s2::Step step;
    int solo = -1, bucket = 0;
    std::vector<int> buckets{{0}}, events{{0}};
    std::string graph_key;
    std::vector<Sampling> samples;
}};
void s2_prepare(Model&, int, int) {{}}
void launch_step(Model&, int, int, int, int, int) {{}}
void launch_draft_step(Draft& D, int M, int, bool chain, int) {{
    const int ns = D.d.s2on ? D.d.s2->h.ns : 0;
    if (const char* e = draft_shape_error(M, chain, D.d.s2on, ns)) die(e);
}}
void native_inputs(const Launch& l, StepIn& si, S2In& h2) {{
    int total = 0;
    h2.ns = int(l.step.sess.size());
    for (size_t j = 0; j < l.step.sess.size(); ++j) {{
        h2.sess[j] = l.step.sess[j];
        h2.row0[j] = l.step.row0[j];
        h2.rows[j] = l.step.rows[j];
        total += l.step.rows[j];
    }}
    si.M = total; si.context = 64;
}}
uint32_t head_size(const StepIn& si, int) {{ return uint32_t(si.M * 8); }}
void lp_fetch(Model&, const StepOut&, const StepIn&, LpStep&) {{}}

int main() {{
    // Part A: solo capture in g6m4.cu (s2on == false) with SESS=4, MTP=3, GR8_S2_SHARED_WATERFILL=1.
    S2State s2_main{{}}, s2_draft{{}};
    Model m; m.s2 = &s2_main;
    Draft D; D.d.s2 = &s2_draft;
    const int mtp = 3, st = 0, s2sess = 4;
    const bool shared_waterfill = true;
    (void)s2sess;
    (void)shared_waterfill;
    const std::vector<int> draft_rows{{1, 2, 3, 4}};
    cudaGraphExec_t exec_d[NBK][2][MX + 1]{{}};
    int next_id = 1;
    auto capture_solo = [&](const std::function<void()>& body, cudaGraphExec_t* out, const char*, int) {{
        body();
        *out = next_id++;
    }};
    {{
        auto capture = capture_solo;
{solo_block}
    }}
    for (int k = 0; k < NBK; ++k) {{
        assert(exec_d[k][1][1] != 0);
        for (int M = 2; M <= MX; ++M) assert(exec_d[k][1][M] == 0);
    }}
    // Negative control: M >= 2 on solo chain draft (s2on == false) fails with the exact Boot-0 message.
    for (int bad_m : {{2, 3, 4}}) {{
        std::string got;
        try {{ launch_draft_step(D, bad_m, 0, true, st); }} catch (const std::runtime_error& e) {{ got = e.what(); }}
        assert(got == "g6: a chain draft step has one row");
    }}

    // Part B: S2 multi-session capture_all in flash_native.cuh over real g6s2::flash_shapes (libg6host.a).
    struct Graphs {{ cudaGraphExec_t a = 0, b = 0; }};
    std::map<std::string, Graphs> graphs;
    auto key = [](char kind, int M, int bucket) {{ return kind + std::to_string(M) + ":b" + std::to_string(bucket); }};
    auto fence = [] {{}};
    auto check = [] {{}};
    auto capture = [&](const std::function<void()>& body) -> cudaGraphExec_t {{ body(); return next_id++; }};
    struct Sc {{ unsigned limit() const {{ return 4; }} unsigned mtp = 3; }} sc;
    StepIn main_in{{}}, draft_in{{}};
    StepOut main_out{{}}, draft_out{{}};
    struct Ctx {{
        Model main; Draft draft; int stream = 0, split = 0, cp_R = 0;
        StepIn *input = nullptr, *draft_input = nullptr;
        StepOut *output = nullptr, *draft_output = nullptr;
        Fabric main_fabric, draft_fabric;
        std::function<void(int, bool, int)> draft_graph = [](int, bool, int) {{}};
        std::function<void(int, int)> main_graph = [](int, int) {{}};
        std::function<void(u64, int, int)> cp_enqueue = [](u64, int, int) {{}};
    }} c{{}};
    c.main = m; c.draft = D; c.input = &main_in; c.draft_input = &draft_in; c.output = &main_out; c.draft_output = &draft_out;
    const int chunk = 16;
{cap_block}
    capture_all();
    for (int k = 0; k < NBK; ++k) {{
        for (int M = 1; M <= 16; ++M) {{
            assert(graphs.count(key('m', M, k)) == 1);
            assert(graphs.count(key('d', M, k)) == 1);
        }}
        for (int M = 1; M <= 4; ++M) assert(graphs.count(key('c', M, k)) == 1);
    }}

    // Part C: FlashModule::on_launch populates dlp on draft/chain steps, on_draft_lp bounds-checks rows,
    // and o.draft_lp registration is gated on knob::s2_shared_waterfill() so off-mode leaves o.draft_lp null.
    bool multi = true;
    S2In layout_buf{{}}, draft_layout_buf{{}};
    S2In *layout = &layout_buf, *draft_layout = &draft_layout_buf;
    u64 dseq = 0;
    LpStep lp{{}};
    int lp_rows = 0;
    float dlp[MX] = {{}};
    int dlp_rows = 0;
    auto wrote_step = [](const StepIn&) {{}};
    auto widen = [](const s2::Step&, const std::vector<int>&) {{}};
    struct Ks {{ void step(StepIn&, const s2::Step&, const std::vector<int>&, int) {{}} }} ks;
{launch_block}
{dlp_block}

    struct M {{ float on_draft_lp(int r) {{ return float(r); }} }};
    struct Ops {{ float (*draft_lp)(void*, int) = nullptr; }};
    knob::wf = false;
    Ops o_off{{}};
    {{
        Ops& o = o_off;
{reg_line}
    }}
    assert(o_off.draft_lp == nullptr);
    knob::wf = true;
    Ops o_on{{}};
    {{
        Ops& o = o_on;
{reg_line}
    }}
    assert(o_on.draft_lp != nullptr);

    for (int r = 0; r < 4; ++r) {{
        draft_out.tok[r] = 100 + r;
        draft_out.lp[r] = -0.125f * float(r + 1);
    }}
    Launch ld;
    ld.step.kind = s2::Step::DRAFT;
    ld.step.sess = {{0, 1}};
    ld.step.row0 = {{0, 2}};
    ld.step.rows = {{2, 2}};
    ld.solo = -1;
    ld.bucket = 0;
    ld.graph_key = "d4:b0";
    const auto dtoks = on_launch(ld);
    assert(dtoks.size() == 4 && dlp_rows == 4);
    for (int r = 0; r < 4; ++r) assert(std::fabs(on_draft_lp(r) - (-0.125f * float(r + 1))) < 1e-6f);
    for (int bad : {{-1, 4}}) {{
        std::string got;
        try {{ (void)on_draft_lp(bad); }} catch (const std::runtime_error& e) {{ got = e.what(); }}
        assert(got == "draft logprob row outside the last draft step");
    }}

    // A narrower chain step updates dlp_rows to the new step width.
    draft_out.lp[0] = -0.5f; draft_out.lp[1] = -0.75f;
    Launch lc;
    lc.step.kind = s2::Step::CHAIN;
    lc.step.sess = {{0, 1}};
    lc.step.row0 = {{0, 1}};
    lc.step.rows = {{1, 1}};
    lc.solo = -1;
    lc.bucket = 0;
    lc.graph_key = "c2:b0";
    (void)on_launch(lc);
    assert(dlp_rows == 2 && std::fabs(on_draft_lp(1) - (-0.75f)) < 1e-6f);
    {{
        std::string got;
        try {{ (void)on_draft_lp(2); }} catch (const std::runtime_error& e) {{ got = e.what(); }}
        assert(got == "draft logprob row outside the last draft step");
    }}

    std::printf("flash_serve_waterfill: PASS (solo chain M=1 guard, real flash_shapes 1..4 capture, gated Flash draft_lp)\\n");
    return 0;
}}
'''
    with tempfile.TemporaryDirectory() as tmp:
        cpp, binary = Path(tmp) / 'wf.cpp', Path(tmp) / 'wf'
        cpp.write_text(program)
        subprocess.run(
            [
                'g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                str(cpp), str(host_lib), '-lpthread', '-ldl', '-lm', '-o', str(binary),
            ],
            check=True,
        )
        run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        print(run.stdout, end='')


if __name__ == '__main__':
    main()
