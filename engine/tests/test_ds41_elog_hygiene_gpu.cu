// Synthetic router/log hygiene only: no weights pack, fabric, or expert execution.
#include "ds41_v2_test_support.cuh"
#include <limits>

namespace elogtest {
constexpr size_t cap = 32ULL << 20;
struct BoundedArena : v2test::Arena {
    static size_t live_bytes() { return v2test::live; }
    template<typename T> T* make(size_t count) {
        v2test::insist(count <= (std::numeric_limits<size_t>::max() - 2 * v2test::red) / sizeof(T),
                       "allocation size does not overflow");
        const size_t total = std::max(size_t(1), count * sizeof(T)) + 2 * v2test::red;
        v2test::insist(total < cap && live_bytes() < cap - total, "strict 32 MiB allocation cap");
        return v2test::Arena::make<T>(count);
    }
};
BoundedArena* active = nullptr;
template<typename T> cudaError_t allocate(T** p, size_t bytes) {
    v2test::insist(active != nullptr, "allocator is active");
    *p = reinterpret_cast<T*>(active->make<unsigned char>(bytes));
    return cudaSuccess;
}
}

// scratch_of owns its registry and initializes base scratch itself. Route those
// allocations through the same guarded Arena as the fixture, without changing
// the production allocator or calling bind_moe/prepare_moe (expert buffers).
#define cudaMalloc elogtest::allocate
#include "../runtime/rank.cuh"
#undef cudaMalloc

using namespace g6rt;
using v2test::insist;
using ds41::v2::check;
constexpr int NE = 384, TK = 6, M = 48;
static_assert(MX == M && dsr::NEXP == NE && dsr::TOPK == TK, "fixture geometry");
static_assert(M > ds41moe::MTMAX && M < ds41moe::RO_FROM, "exercise router_pw");
struct Header { unsigned cursor; int win_start; };
static Header header(const dsr::ExpertLog* log) {
    Header h{};
    check(cudaMemcpy(&h, log, sizeof(h), cudaMemcpyDeviceToHost));
    return h;
}
static bool empty_log(Model& m) {
    std::vector<dsr::ExpertLogRec> recs;
    return ds41moe::elog_read(&m, recs, false) == 0 && recs.empty() &&
           header(ds41moe::scratch_of(&m)->elog).cursor == 0;
}
static void begin(Model& m, int first, bool reset) {
    ds41moe::elog_begin(&m, first, reset);
    check(cudaGetLastError());
    // The serve path also orders this legacy-stream copy before nonblocking work.
    check(cudaStreamSynchronize(nullptr));
}

int main() {
    insist(setenv("GR8_DS_MOE_ELOG", "1", 1) == 0, "enable expert log");
    insist(setenv("GR8_DS_MOE_TOPK", "6", 1) == 0, "prefill top-k");
    insist(setenv("GR8_DS_MOE_TOPK_DECODE", "6", 1) == 0, "decode top-k");
    {
        elogtest::BoundedArena a;
        elogtest::active = &a;
        Model m{};
        m.nl = 1;
        m.layers.resize(1);
        auto* sc = ds41moe::scratch_of(&m);
        ds41moe::PwScratch pw{};
        sc->pw = &pw;
        pw.logits = a.make<float>(M * NE);
        pw.plan = a.make<char>(sizeof(dsr::PlanMT<TK, 128>));
        constexpr int nt = ((M + 31) / 32) * 4;
        pw.xfrag = a.make<u32>(size_t(nt) * HID * 4);
        m.hc_y = a.make<__nv_bfloat16>(M * HID);
        auto* gate = a.make<__nv_bfloat16>(NE * HID);
        auto* bias = a.make<float>(NE);
        v2test::bf16(m.hc_y, M * HID, 19);
        v2test::bf16(gate, NE * HID, 31);
        check(cudaMemset(bias, 0, NE * sizeof(float)));
        v2test::Inputs inputs;
        inputs.add(m.hc_y, M * HID * sizeof(__nv_bfloat16));
        inputs.add(gate, NE * HID * sizeof(__nv_bfloat16));
        inputs.add(bias, NE * sizeof(float));
        ds41moe::LayerState layer{};
        layer.ne = NE; layer.tk = TK; layer.layer = 0;
        layer.sc = sc; layer.gate = gate; layer.bias = bias;
        m.layers[0].ds41_moe = &layer;
        cudaStream_t st;
        check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        check(cudaDeviceSynchronize());
        insist(ds41moe::elog_on(), "log enabled before capture");

        m.ds_moe_decode_step = true;
        begin(m, 0, true);
        check(static_cast<cudaError_t>(ds41moe::op_router(&m, 0, M, st)));
        check(cudaStreamSynchronize(st));
        insist(empty_log(m), "eager decode leaves cursor zero");
        printf("PASS eager decode M=48 cursor=0\n");

        cudaGraph_t graph;
        cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        check(static_cast<cudaError_t>(ds41moe::op_router(&m, 0, M, st)));
        check(cudaStreamEndCapture(st, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            check(cudaGraphLaunch(exec, st));
            check(cudaStreamSynchronize(st));
            insist(empty_log(m), "graph decode leaves cursor zero");
        }
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
        printf("PASS captured decode M=48 replays=2 cursor=0\n");

        // Main/verification graphs are captured before the decode flag is set.
        // A host flag changed at replay cannot remove a captured log node.
        m.ds_moe_decode_step = false;
        begin(m, 0, true);
        check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        check(static_cast<cudaError_t>(ds41moe::op_router(&m, 0, M, st)));
        check(cudaStreamEndCapture(st, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (bool decode : {true, false}) {
            m.ds_moe_decode_step = decode;
            for (int replay = 0; replay < 2; ++replay) {
                check(cudaGraphLaunch(exec, st));
                check(cudaStreamSynchronize(st));
                insist(empty_log(m), "main/verification graph leaves cursor zero");
            }
        }
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
        printf("PASS capture flag=false; replay flags=true,false replays=2 each cursor=0\n");

        m.ds_moe_decode_step = false;
        begin(m, 12288, true);
        check(static_cast<cudaError_t>(ds41moe::op_router(&m, 0, M, st)));
        check(cudaStreamSynchronize(st));
        std::vector<dsr::ExpertLogRec> recs;
        insist(ds41moe::elog_read(&m, recs, false) == 1 && recs.size() == 1,
               "prefill writes one record");
        insist(recs[0].rows == M && recs[0].start == 12288, "prefill rows and start");
        insist(!empty_log(m), "negative control: prefill rejected by empty-log predicate");
        printf("PASS prefill n=1 rows=48 start=12288; empty-log negative control rejected\n");

        const unsigned stale = 4096;
        check(cudaMemcpy(sc->elog, &stale, sizeof(stale), cudaMemcpyHostToDevice));
        begin(m, 24576, false);
        Header h = header(sc->elog);
        insist(h.cursor == stale && h.win_start == 24576, "non-reset preserves stale cursor");
        ds41moe::elog_begin(&m, 24576); // the two-argument default is still non-reset
        check(cudaStreamSynchronize(nullptr));
        h = header(sc->elog);
        insist(h.cursor == stale && h.win_start == 24576, "default preserves stale cursor");
        begin(m, 36864, true);
        h = header(sc->elog);
        insist(h.cursor == 0 && h.win_start == 36864, "reset clears stale cursor and sets start");
        check(static_cast<cudaError_t>(ds41moe::op_router(&m, 0, M, st)));
        check(cudaStreamSynchronize(st));
        insist(ds41moe::elog_read(&m, recs, true) == 1 && recs.size() == 1,
               "reset removes stale records before new prefill");
        insist(recs[0].rows == M && recs[0].start == 36864, "fresh record after reset");
        insist(empty_log(m), "resetting read drains cursor");
        printf("PASS stale cursor=4096 preserved; reset cursor=0 start=36864; fresh n=1 drained\n");
        inputs.verify();
        a.verify();
        check(cudaStreamDestroy(st));
        delete sc; // its device blocks belong to Arena; registry is not used again
        elogtest::active = nullptr;
    }
    insist(elogtest::BoundedArena::live_bytes() == 0 && v2test::peak < elogtest::cap,
           "all device allocations released below 32 MiB");
    printf("PASS test_ds41_elog_hygiene_gpu SYNTHETIC correctness only; peak_bytes=%zu live_bytes=%zu; guards/input_hashes PASS\n",
           v2test::peak, elogtest::BoundedArena::live_bytes());
}
