// g6tts: Native CUDA TP=1 Drama 3 (fish_omni) TTS engine in reference mode.
// Speaks PROTOCOL.md v1 over a Unix domain socket (`speech`, `audio`, `encode_audio`, `codes`, `cancel`, `metrics`).
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "tts_backbone.cuh"
#include "tts_config.h"
#include "tts_proto.h"
#include "tts_sample.cuh"
#include "tts_tower.cuh"
#include "tts_weights.h"

#if __has_include("tts_codec.cuh")
#include "tts_codec.cuh"
#else
// Stub interface matching the agreed `g6tts::TtsCodec` API until Wave 1.7 `tts_codec.cuh` lands.
namespace g6tts {
struct TtsCodec {
    bool loaded_ = false;
    bool init(const std::string&, int, int = 4) { loaded_ = false; return false; }
    bool loaded() const { return loaded_; }
    void release() { loaded_ = false; }
    void reset_slot(int, cudaStream_t) {}
    void decode_frames(const int32_t*, int M, int stride, const int*, int16_t* d_pcm, cudaStream_t s) {
        cudaMemsetAsync(d_pcm, 0, size_t(M) * stride * geom::FRAME_SAMPLES * sizeof(int16_t), s);
    }
    int encode_pcm(const float*, int, std::vector<int32_t>& out_codes_10xF, cudaStream_t) {
        out_codes_10xF.clear();
        return -1; // codec unavailable
    }
    void decode_whole(const int32_t*, int F, std::vector<int16_t>& out_pcm, cudaStream_t) {
        out_pcm.assign(size_t(F) * geom::FRAME_SAMPLES, 0);
    }
    void decode_streamed(const int32_t*, int F, const std::vector<int>&, std::vector<int16_t>& out_pcm, cudaStream_t) {
        out_pcm.assign(size_t(F) * geom::FRAME_SAMPLES, 0);
    }
};
} // namespace g6tts
#endif

#include "tts_graph.cuh"
#include "tts_prefix_cache.cuh"

namespace {

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct CliOpts {
    std::string model_dir;
    std::string manifest;
    std::string codec_path;
    std::string socket_path = "/run/g6/engine.sock";
    std::string model_name_override;
    std::string voc_strides = "1,1,2,4";
    int streams = 8;
    int cap = 4096;
    int prefix_cache_entries = 8;
    bool verify_checksums = true;
    bool dry_run_load = false;
    bool use_graph = true;
    g6tts::WeightMode weight_mode = g6tts::WeightMode::Ref;
};

void usage() {
    std::fprintf(stderr,
                 "usage: g6tts --model-dir DIR [--manifest NAME] [--socket PATH] [--model-name NAME]\n"
                 "             [--streams N] [--cap N] [--prefix-cache-entries N]\n"
                 "             [--weight-mode ref|ref-fp16|native]\n"
                 "             [--codec-path PATH] [--voc-strides 1,1,2,4] [--graph|--no-graph]\n"
                 "             [--no-hash] [--dry-run-load]\n");
}

bool parse_cli(int argc, char** argv, CliOpts& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* f) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "g6tts: %s requires an argument\n", f);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") {
            const char* v = need("--model-dir");
            if (!v) return false;
            o.model_dir = v;
        } else if (a == "--manifest") {
            const char* v = need("--manifest");
            if (!v) return false;
            o.manifest = v;
        } else if (a == "--socket") {
            const char* v = need("--socket");
            if (!v) return false;
            o.socket_path = v;
        } else if (a == "--model-name") {
            const char* v = need("--model-name");
            if (!v) return false;
            o.model_name_override = v;
        } else if (a == "--streams") {
            const char* v = need("--streams");
            if (!v) return false;
            o.streams = std::max(1, std::atoi(v));
        } else if (a == "--cap") {
            const char* v = need("--cap");
            if (!v) return false;
            o.cap = std::max(64, std::atoi(v));
        } else if (a == "--prefix-cache-entries") {
            const char* v = need("--prefix-cache-entries");
            if (!v) return false;
            o.prefix_cache_entries = std::max(0, std::atoi(v));
        } else if (a == "--codec-path") {
            const char* v = need("--codec-path");
            if (!v) return false;
            o.codec_path = v;
        } else if (a == "--voc-strides") {
            const char* v = need("--voc-strides");
            if (!v) return false;
            o.voc_strides = v;
        } else if (a == "--weight-mode") {
            const char* v = need("--weight-mode");
            if (!v) return false;
            if (std::strcmp(v, "ref") == 0) o.weight_mode = g6tts::WeightMode::Ref;
            else if (std::strcmp(v, "ref-fp16") == 0) o.weight_mode = g6tts::WeightMode::RefFp16;
            else if (std::strcmp(v, "native") == 0) o.weight_mode = g6tts::WeightMode::Native;
            else {
                std::fprintf(stderr, "g6tts: unknown --weight-mode %s (expected ref, ref-fp16, or native)\n", v);
                return false;
            }
        } else if (a == "--graph") {
            o.use_graph = true;
        } else if (a == "--no-graph") {
            o.use_graph = false;
        } else if (a == "--no-hash") {
            o.verify_checksums = false;
        } else if (a == "--dry-run-load") {
            o.dry_run_load = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "g6tts: unknown option %s\n", a.c_str());
            return false;
        }
    }
    if (o.model_dir.empty()) {
        std::fprintf(stderr, "g6tts: --model-dir is required\n");
        return false;
    }
    return true;
}

std::vector<int> parse_voc_strides(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const int v = std::atoi(tok.c_str());
        if (v == 1 || v == 2 || v == 4) out.push_back(v);
    }
    if (out.empty()) out = {1, 1, 2, 4};
    return out;
}

struct SlotState {
    bool active = false;
    g6tts::SpeechRequest req;
    int pos = 0;            // current sequence position of `last_frame`
    int frames_done = 0;    // generated audio frames so far
    int ring_read_pos = 0;  // next unread frame index in device code ring
    int ring_unflushed = 0; // frames in device code ring not yet decoded/emitted
    size_t stride_idx = 0;
    std::vector<int32_t> pending_codes; // buffered [ring_unflushed, 10] for EmitMode::Codes/Both
    std::chrono::steady_clock::time_point t_start;
    double prefill_s = 0.0;
};

struct TtsEngine {
    g6tts::TtsWeights weights;
    g6tts::BackboneRunner bb;
    g6tts::TowerRunner tower;
    g6tts::TtsCodec codec;
    g6tts::TtsGraphRunner<g6tts::TtsCodec> graph;
    g6tts::TtsPrefixCache prefix_cache;
    std::vector<int> voc_strides;

    __nv_bfloat16* d_prefill_hid = nullptr; // [H] scratch for last prefill hidden state

    uint64_t next_auto_seed = 0x123456789abcdef0ull;
    cudaStream_t stream = nullptr;

    ~TtsEngine() { release(); }

    void release() {
        prefix_cache.release();
        graph.release();
        if (d_prefill_hid) { cudaFree(d_prefill_hid); d_prefill_hid = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }

    void init(const CliOpts& opts) {
        weights = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, opts.weight_mode);
        weights.cfg.max_streams = opts.streams;
        weights.cfg.cap = opts.cap;
        if (!opts.model_name_override.empty()) weights.cfg.model_name = opts.model_name_override;

        voc_strides = parse_voc_strides(opts.voc_strides);
        TTS_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        // Allocate `opts.streams + 1` slots so `pad_slot = opts.streams` is valid for CUDA graph bucket padding.
        const int total_slots = opts.streams + 1;
        bb.init(weights, total_slots, opts.cap, 128);
        tower.init(weights, total_slots, opts.cap, 128);

        const std::string cpath = opts.codec_path.empty()
            ? (opts.model_dir + "/codec/exlr8-codec.safetensors")
            : opts.codec_path;
        if (codec.init(cpath, total_slots, 4)) {
            std::fprintf(stderr, "g6tts: loaded native CUDA codec from %s\n", cpath.c_str());
        } else {
            std::fprintf(stderr, "g6tts: codec not loaded (%s); PCM output will be silent until codec is present\n",
                         cpath.c_str());
        }

        graph.init(weights, bb, tower, codec, opts.streams, opts.cap, opts.use_graph);
        prefix_cache.init(weights.cfg, opts.prefix_cache_entries);
        TTS_CK(cudaMalloc(&d_prefill_hid, size_t(weights.cfg.hidden_size) * sizeof(__nv_bfloat16)));
        if (opts.use_graph) {
            graph.warmup_graphs(stream);
            std::fprintf(stderr, "g6tts: pre-captured CUDA graphs for streams<=%d (prefix_cache=%d)\n",
                         opts.streams, opts.prefix_cache_entries);
        }
    }

    void prefill_slot(int slot, SlotState& st) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& c = weights.cfg;
        const int T = st.req.num_rows;
        const int rw = c.row_width();
        const int chunk = bb.max_chunk;
        const int32_t* rows = st.req.rows.data();

        codec.reset_slot(slot, stream);

        int hit_P = 0;
        const int hit_idx = prefix_cache.find_longest_prefix(rows, T, rw, hit_P);
        if (hit_idx >= 0) {
            prefix_cache.restore_to_slot(hit_idx, slot, bb, tower, d_prefill_hid, stream);
            if (hit_P < T) {
                // Bridge tower position `hit_P - 1` using restored `bb_hid[hit_P - 1]` and new `rows[hit_P]`.
                tower.prefill_chunk(d_prefill_hid,
                                    rows + size_t(hit_P - 1) * rw,
                                    rows + size_t(hit_P) * rw,
                                    slot, hit_P - 1, 1, nullptr, stream);
            }
        } else {
            ++prefix_cache.misses;
        }

        if (hit_P < T) {
            const int lcp = prefix_cache.find_recent_common_prefix(rows, T, rw);
            const int split_P = (lcp >= g6tts::TtsPrefixCache::MIN_PREFIX_LEN && lcp < T && lcp > hit_P) ? lcp : 0;

            auto run_range = [&](int from, int to) {
                for (int start = from; start < to; start += chunk) {
                    const int end = std::min(start + chunk, to);
                    const int len = end - start;
                    bb.forward(rows + size_t(start) * rw, len,
                               true, slot, start, nullptr, nullptr, false, false, stream);
                    const int paired_end = std::min(end, T - 1);
                    if (paired_end > start) {
                        tower.prefill_chunk(bb.d_final_bf16,
                                            rows + size_t(start) * rw,
                                            rows + size_t(start + 1) * rw,
                                            slot, start, paired_end - start, nullptr, stream);
                    }
                    if (end == to) {
                        TTS_CK(cudaMemcpyAsync(d_prefill_hid,
                                               bb.d_final_bf16 + size_t(len - 1) * c.hidden_size,
                                               size_t(c.hidden_size) * sizeof(__nv_bfloat16),
                                               cudaMemcpyDeviceToDevice, stream));
                    }
                }
            };

            if (split_P > hit_P) {
                run_range(hit_P, split_P);
                prefix_cache.save_from_slot(rows, split_P, rw, slot, bb, tower, d_prefill_hid, stream);
                run_range(split_P, T);
            } else {
                run_range(hit_P, T);
            }
            prefix_cache.save_from_slot(rows, T, rw, slot, bb, tower, d_prefill_hid, stream);
        }
        prefix_cache.record_prompt(rows, T, rw);

        // Initialize slot's 16-entry RAS ring from the last min(16, T) prompt main tokens
        int32_t ring[g6tts::geom::RAS_WINDOW];
        std::fill(std::begin(ring), std::end(ring), -1);
        const int take = std::min(T, g6tts::geom::RAS_WINDOW);
        for (int i = 0; i < take; ++i) {
            ring[i] = rows[size_t(T - take + i) * rw];
        }

        graph.admit_slot(slot, d_prefill_hid,
                         rows + size_t(T - 1) * rw,
                         T - 1, ring, take,
                         st.req.temperature, st.req.top_p, st.req.top_k, st.req.seed,
                         stream);

        st.pos = T - 1;
        st.frames_done = 0;
        st.ring_read_pos = 0;
        st.ring_unflushed = 0;
        st.stride_idx = 0;
        st.pending_codes.clear();
        st.t_start = t0;
        st.prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        st.active = true;
    }

    // Flushes any remaining unflushed frames for `slot` (e.g. tail on EOS or max_frames) via codec graphs
    // decomposing `rem` greedily into supported strides {4, 2, 1}.
    void flush_slot_tail(int slot, SlotState& st,
                         const std::function<bool(const std::string&)>& safe_send) {
        const int n_frames = st.ring_unflushed;
        if (n_frames <= 0) return;
        const bool want_pcm = (st.req.emit == g6tts::EmitMode::Pcm || st.req.emit == g6tts::EmitMode::Both);
        const bool want_codes = (st.req.emit == g6tts::EmitMode::Codes || st.req.emit == g6tts::EmitMode::Both);
        std::vector<int16_t> pcm;
        if (want_pcm) {
            pcm.resize(size_t(n_frames) * g6tts::geom::FRAME_SAMPLES, 0);
            if (codec.loaded()) {
                int done = 0;
                while (done < n_frames) {
                    const int rem = n_frames - done;
                    const int s = (rem >= 4) ? 4 : ((rem >= 2) ? 2 : 1);
                    graph.decode_codec_batch({slot}, {st.ring_read_pos + done}, s, stream);
                    std::memcpy(pcm.data() + size_t(done) * g6tts::geom::FRAME_SAMPLES,
                                graph.h_codec_pcm,
                                size_t(s) * g6tts::geom::FRAME_SAMPLES * sizeof(int16_t));
                    done += s;
                }
            }
        }
        safe_send(g6tts::make_audio_json(
            st.req.id, n_frames,
            want_pcm ? pcm.data() : nullptr,
            want_codes ? st.pending_codes.data() : nullptr));
        st.ring_read_pos += n_frames;
        st.ring_unflushed = 0;
        st.pending_codes.clear();
    }

    void run_loop(g6tts::TtsQueue& queue,
                  const std::function<bool(const std::string&)>& safe_send,
                  g6tts::TtsMetrics& metrics) {
        const auto& c = weights.cfg;
        const int S = c.max_streams;
        const int rw = c.row_width();
        std::vector<SlotState> slots(static_cast<size_t>(S));
        std::vector<int> active_slots;

        for (;;) {
            // 1. Service pending encode_audio requests (between frames)
            g6tts::EncodeAudioRequest enc_req;
            while (queue.pop_encode(enc_req)) {
                metrics.waiting.store(queue.queued_count());
                if (queue.is_cancelled(enc_req.id)) {
                    queue.mark_done(enc_req.id);
                    metrics.req_abort.fetch_add(1);
                    safe_send(g6tts::make_finish_json(enc_req.id, "abort", 0, 0));
                    continue;
                }
                if (!codec.loaded()) {
                    queue.mark_done(enc_req.id);
                    metrics.req_error.fetch_add(1);
                    safe_send(g6tts::make_finish_json(enc_req.id, "error", 0, 0, -1,
                                                      "codec not loaded in engine", 503));
                    continue;
                }
                std::vector<int32_t> codes_10xF;
                const int F = codec.encode_pcm(enc_req.pcm.data(), static_cast<int>(enc_req.pcm.size()),
                                               codes_10xF, stream);
                queue.mark_done(enc_req.id);
                if (F < 0) {
                    metrics.req_error.fetch_add(1);
                    safe_send(g6tts::make_finish_json(enc_req.id, "error", 0, 0, -1, "codec encode failed", 500));
                } else {
                    std::vector<int32_t> codes_Fx10(size_t(F) * g6tts::geom::NCB);
                    for (int f = 0; f < F; ++f) {
                        for (int cb = 0; cb < g6tts::geom::NCB; ++cb) {
                            codes_Fx10[size_t(f) * g6tts::geom::NCB + cb] = codes_10xF[size_t(cb) * F + f];
                        }
                    }
                    metrics.req_stop.fetch_add(1);
                    if (safe_send(g6tts::make_codes_json(enc_req.id, F, codes_Fx10.data()))) {
                        safe_send(g6tts::make_finish_json(enc_req.id, "stop", 0, F));
                    }
                }
            }

            // 2. Admit queued speech requests into free slots (between frames, outside CUDA graph)
            for (int s = 0; s < S; ++s) {
                if (slots[size_t(s)].active) continue;
                g6tts::SpeechRequest sreq;
                if (!queue.pop_speech(sreq)) break;
                metrics.waiting.store(queue.queued_count());
                if (queue.is_cancelled(sreq.id)) {
                    queue.mark_done(sreq.id);
                    metrics.req_abort.fetch_add(1);
                    safe_send(g6tts::make_finish_json(sreq.id, "abort", sreq.num_rows, 0));
                    --s;
                    continue;
                }
                if (!sreq.has_seed) {
                    sreq.seed = next_auto_seed;
                    next_auto_seed = next_auto_seed * 6364136223846793005ull + 1442695040888963407ull;
                }
                slots[size_t(s)].req = std::move(sreq);
                try {
                    prefill_slot(s, slots[size_t(s)]);
                } catch (const std::exception& e) {
                    const std::string id = slots[size_t(s)].req.id;
                    const int pt = slots[size_t(s)].req.num_rows;
                    slots[size_t(s)].active = false;
                    queue.mark_done(id);
                    metrics.req_error.fetch_add(1);
                    safe_send(g6tts::make_finish_json(id, "error", pt, 0, -1, e.what(), 500));
                }
            }

            // 3. Collect active slots
            active_slots.clear();
            for (int s = 0; s < S; ++s) {
                if (slots[size_t(s)].active) active_slots.push_back(s);
            }
            metrics.running.store(active_slots.size());
            if (active_slots.empty()) {
                if (queue.is_closed()) break;
                if (!queue.wait_for_work()) break;
                continue;
            }

            // 4. Launch frame step:
            //    When all active slots are on frame 0 and first stride is 1, split frame 0 into
            //    (tower+sample -> immediate stride-1 codec + safe_send -> pipelined backbone forward)
            //    so first-audio arrives before the 40L backbone runs for frame 1 (bit-identical state).
            bool all_frame0 = (!voc_strides.empty() && voc_strides[0] == 1);
            if (all_frame0) {
                for (int s : active_slots) {
                    if (slots[size_t(s)].frames_done != 0) {
                        all_frame0 = false;
                        break;
                    }
                }
            }
            const int step_M = all_frame0
                ? graph.step_frame_tower_only(active_slots, stream)
                : graph.step_frame(active_slots, stream);
            const int n_act = static_cast<int>(active_slots.size());

            // 5. Inspect pinned [n_act, 11] results; group codec-ready slots by stride in {1, 2, 4}.
            std::array<std::vector<int>, 3> ready_slots;
            std::array<std::vector<int>, 3> ready_starts;
            std::vector<int> finish_length_slots;

            for (int m = 0; m < n_act; ++m) {
                const int s = active_slots[size_t(m)];
                SlotState& st = slots[size_t(s)];

                if (queue.is_cancelled(st.req.id)) {
                    graph.deactivate_slot(s, stream);
                    st.active = false;
                    queue.mark_done(st.req.id);
                    metrics.req_abort.fetch_add(1);
                    safe_send(g6tts::make_finish_json(st.req.id, "abort", st.req.num_rows, st.frames_done));
                    continue;
                }

                const int32_t* row = graph.h_out_rows + size_t(m) * rw;
                const int32_t main_tok = row[0];
                if (main_tok == c.eos_token_id) {
                    flush_slot_tail(s, st, safe_send);
                    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - st.t_start).count();
                    const double dec_s = std::max(0.0, tot_s - st.prefill_s);
                    st.active = false;
                    queue.mark_done(st.req.id);
                    metrics.req_stop.fetch_add(1);
                    metrics.prompt_tokens.fetch_add(uint64_t(st.req.num_rows));
                    metrics.gen_frames.fetch_add(uint64_t(st.frames_done));
                    safe_send(g6tts::make_finish_json(st.req.id, "stop", st.req.num_rows, st.frames_done,
                                                      c.eos_token_id, "", 0, st.prefill_s, dec_s));
                    continue;
                }

                for (int cb = 0; cb < g6tts::geom::NCB; ++cb) {
                    st.pending_codes.push_back(row[1 + cb]);
                }
                ++st.frames_done;
                ++st.pos;
                ++st.ring_unflushed;

                const int target_stride = voc_strides[std::min(st.stride_idx, voc_strides.size() - 1)];
                if (st.ring_unflushed >= target_stride) {
                    const int si = g6tts::stride_index(target_stride);
                    const bool want_pcm = (st.req.emit == g6tts::EmitMode::Pcm || st.req.emit == g6tts::EmitMode::Both);
                    if (si >= 0 && want_pcm && codec.loaded()) {
                        ready_slots[size_t(si)].push_back(s);
                        ready_starts[size_t(si)].push_back(st.ring_read_pos);
                    } else {
                        const bool want_codes = (st.req.emit == g6tts::EmitMode::Codes || st.req.emit == g6tts::EmitMode::Both);
                        std::vector<int16_t> zeros;
                        if (want_pcm) zeros.assign(size_t(st.ring_unflushed) * g6tts::geom::FRAME_SAMPLES, 0);
                        safe_send(g6tts::make_audio_json(
                            st.req.id, st.ring_unflushed,
                            want_pcm ? zeros.data() : nullptr,
                            want_codes ? st.pending_codes.data() : nullptr));
                        st.ring_read_pos += st.ring_unflushed;
                        st.ring_unflushed = 0;
                        st.pending_codes.clear();
                        if (st.stride_idx + 1 < voc_strides.size()) ++st.stride_idx;
                    }
                }

                if (st.frames_done >= st.req.max_frames) {
                    finish_length_slots.push_back(s);
                }
            }

            // 6. Launch batched codec stride CUDA graphs for each stride in {1, 2, 4} with ready slots.
            for (int si = 0; si < 3; ++si) {
                if (ready_slots[size_t(si)].empty()) continue;
                const int stride = g6tts::CODEC_STRIDES[si];
                graph.decode_codec_batch(ready_slots[size_t(si)], ready_starts[size_t(si)], stride, stream);
                const size_t n_ready = ready_slots[size_t(si)].size();
                for (size_t i = 0; i < n_ready; ++i) {
                    const int s = ready_slots[size_t(si)][i];
                    SlotState& st = slots[size_t(s)];
                    const bool want_codes = (st.req.emit == g6tts::EmitMode::Codes || st.req.emit == g6tts::EmitMode::Both);
                    const int16_t* pcm_ptr = graph.h_codec_pcm + i * size_t(stride) * g6tts::geom::FRAME_SAMPLES;
                    safe_send(g6tts::make_audio_json(
                        st.req.id, stride, pcm_ptr,
                        want_codes ? st.pending_codes.data() : nullptr));
                    st.ring_read_pos += stride;
                    st.ring_unflushed -= stride;
                    if (st.ring_unflushed <= 0) {
                        st.ring_unflushed = 0;
                        st.pending_codes.clear();
                    } else {
                        st.pending_codes.erase(st.pending_codes.begin(),
                                               st.pending_codes.begin() + ptrdiff_t(stride * g6tts::geom::NCB));
                    }
                    if (st.stride_idx + 1 < voc_strides.size()) ++st.stride_idx;
                }
            }

            // 7. Retire slots that reached max_frames (flush any remaining tail < target_stride).
            for (int s : finish_length_slots) {
                SlotState& st = slots[size_t(s)];
                if (!st.active) continue;
                flush_slot_tail(s, st, safe_send);
                graph.deactivate_slot(s, stream);
                const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - st.t_start).count();
                const double dec_s = std::max(0.0, tot_s - st.prefill_s);
                st.active = false;
                queue.mark_done(st.req.id);
                metrics.req_length.fetch_add(1);
                metrics.prompt_tokens.fetch_add(uint64_t(st.req.num_rows));
                metrics.gen_frames.fetch_add(uint64_t(st.frames_done));
                safe_send(g6tts::make_finish_json(st.req.id, "length", st.req.num_rows, st.frames_done,
                                                  -1, "", 0, st.prefill_s, dec_s));
            }

            // 8. Complete pipelined frame-0 backbone forward after first audio chunk has been sent.
            if (all_frame0) {
                bool any_still_active = false;
                for (int s : active_slots) {
                    if (slots[size_t(s)].active) {
                        any_still_active = true;
                        break;
                    }
                }
                if (any_still_active) {
                    graph.step_frame_backbone_only(step_M, stream);
                }
            }
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    CliOpts opts;
    if (!parse_cli(argc, argv, opts)) {
        usage();
        return 2;
    }

    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    std::signal(SIGPIPE, SIG_IGN);

    try {
        if (opts.dry_run_load) {
            auto w = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, opts.weight_mode);
            std::fprintf(stderr, "g6tts: --dry-run-load OK (%s, mode=%s)\n",
                         w.cfg.model_name.c_str(), g6tts::weight_mode_name(w.mode));
            return 0;
        }

        TtsEngine engine;
        engine.init(opts);

        const int lfd = g6tts::listen_unix_socket(opts.socket_path);
        if (lfd < 0) {
            std::fprintf(stderr, "g6tts: failed to bind Unix socket %s\n", opts.socket_path.c_str());
            return 1;
        }
        std::fprintf(stderr, "g6tts: state=%.2f GiB, listening on %s (model=%s, streams=%d, cap=%d, graph=%s)\n",
                     double(engine.bb.state_bytes + engine.tower.state_bytes) / double(1ull << 30),
                     opts.socket_path.c_str(), engine.weights.cfg.model_name.c_str(),
                     opts.streams, opts.cap, opts.use_graph ? "on" : "off");

        g6tts::TtsMetrics metrics;
        while (!g_stop) {
            const int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break;
            }
            g6tts::serve_tts_connection(
                cfd, engine.weights.cfg, metrics,
                [&](g6tts::TtsQueue& q, const std::function<bool(const std::string&)>& send, g6tts::TtsMetrics& m) {
                    engine.run_loop(q, send, m);
                });
            ::close(cfd);
        }

        ::close(lfd);
        ::unlink(opts.socket_path.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "g6tts: fatal: %s\n", e.what());
        return 1;
    }
}
