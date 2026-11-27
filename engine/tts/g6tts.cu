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
    bool verify_checksums = true;
    bool dry_run_load = false;
    g6tts::WeightMode weight_mode = g6tts::WeightMode::Ref;
};

void usage() {
    std::fprintf(stderr,
                 "usage: g6tts --model-dir DIR [--manifest NAME] [--socket PATH] [--model-name NAME]\n"
                 "             [--streams N] [--cap N] [--weight-mode ref|ref-fp16|native]\n"
                 "             [--codec-path PATH] [--voc-strides 1,1,2,4] [--no-hash] [--dry-run-load]\n");
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
        if (v >= 1 && v <= 4) out.push_back(v);
    }
    if (out.empty()) out = {1, 1, 2, 4};
    return out;
}

struct SlotState {
    bool active = false;
    g6tts::SpeechRequest req;
    int pos = 0;            // current sequence position of `last_frame`
    int frames_done = 0;    // generated audio frames so far
    size_t stride_idx = 0;
    std::vector<int32_t> last_frame; // [11]
    std::vector<int32_t> pending_codes; // buffered [k, 10] waiting for vocoder stride
    std::chrono::steady_clock::time_point t_start;
    double prefill_s = 0.0;
};

struct TtsEngine {
    g6tts::TtsWeights weights;
    g6tts::BackboneRunner bb;
    g6tts::TowerRunner tower;
    g6tts::TtsCodec codec;
    std::vector<int> voc_strides;

    // Device state per slot for continuous decode + sampling
    __half* d_slot_hidden = nullptr;   // [max_slots, H]
    __half* d_step_hidden = nullptr;   // [max_slots, H]
    int32_t* d_ras_ring = nullptr;     // [max_slots, 16]
    int32_t* d_ras_pos = nullptr;      // [max_slots]
    int32_t* d_step_slots = nullptr;   // [max_slots]
    int32_t* d_step_pos = nullptr;     // [max_slots]
    int32_t* d_step_frames = nullptr;  // [max_slots, 11]
    float* d_step_temp = nullptr;      // [max_slots]
    float* d_step_top_p = nullptr;     // [max_slots]
    int32_t* d_step_top_k = nullptr;   // [max_slots]
    uint64_t* d_step_seed = nullptr;   // [max_slots]
    uint32_t* d_step_fidx = nullptr;   // [max_slots]
    int32_t* d_out_main = nullptr;     // [max_slots]
    int32_t* d_out_c0 = nullptr;       // [max_slots]
    uint8_t* d_out_sem = nullptr;      // [max_slots]
    int32_t* d_out_code = nullptr;     // [max_slots]
    int32_t* d_codec_codes = nullptr;  // [max_slots * 4 * 10]
    int16_t* d_codec_pcm = nullptr;    // [max_slots * 4 * 2048]
    int32_t* d_codec_slot = nullptr;   // [max_slots]

    uint64_t next_auto_seed = 0x123456789abcdef0ull;
    cudaStream_t stream = nullptr;

    ~TtsEngine() { release(); }

    void release() {
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        freep(d_slot_hidden); freep(d_step_hidden); freep(d_ras_ring); freep(d_ras_pos);
        freep(d_step_slots); freep(d_step_pos); freep(d_step_frames);
        freep(d_step_temp); freep(d_step_top_p); freep(d_step_top_k);
        freep(d_step_seed); freep(d_step_fidx);
        freep(d_out_main); freep(d_out_c0); freep(d_out_sem); freep(d_out_code);
        freep(d_codec_codes); freep(d_codec_pcm); freep(d_codec_slot);
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }

    void init(const CliOpts& opts) {
        weights = g6tts::load_tts_weights(opts.model_dir, opts.manifest, opts.verify_checksums, opts.weight_mode);
        weights.cfg.max_streams = opts.streams;
        weights.cfg.cap = opts.cap;
        if (!opts.model_name_override.empty()) weights.cfg.model_name = opts.model_name_override;

        voc_strides = parse_voc_strides(opts.voc_strides);
        TTS_CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bb.init(weights, opts.streams, opts.cap, 128);
        tower.init(weights, opts.streams, opts.cap, 128);

        const std::string cpath = opts.codec_path.empty()
            ? (opts.model_dir + "/codec/exlr8-codec.safetensors")
            : opts.codec_path;
        if (codec.init(cpath, opts.streams, 4)) {
            std::fprintf(stderr, "g6tts: loaded native CUDA codec from %s\n", cpath.c_str());
        } else {
            std::fprintf(stderr, "g6tts: codec not loaded (%s); PCM output will be silent until codec is present\n",
                         cpath.c_str());
        }

        const int S = opts.streams;
        const int H = weights.cfg.hidden_size;
        TTS_CK(cudaMalloc(&d_slot_hidden, size_t(S) * H * sizeof(__half)));
        TTS_CK(cudaMalloc(&d_step_hidden, size_t(S) * H * sizeof(__half)));
        TTS_CK(cudaMalloc(&d_ras_ring, size_t(S) * g6tts::geom::RAS_WINDOW * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_ras_pos, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_step_slots, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_step_pos, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_step_frames, size_t(S) * weights.cfg.row_width() * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_step_temp, size_t(S) * sizeof(float)));
        TTS_CK(cudaMalloc(&d_step_top_p, size_t(S) * sizeof(float)));
        TTS_CK(cudaMalloc(&d_step_top_k, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_step_seed, size_t(S) * sizeof(uint64_t)));
        TTS_CK(cudaMalloc(&d_step_fidx, size_t(S) * sizeof(uint32_t)));
        TTS_CK(cudaMalloc(&d_out_main, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_out_c0, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_out_sem, size_t(S) * sizeof(uint8_t)));
        TTS_CK(cudaMalloc(&d_out_code, size_t(S) * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_codec_codes, size_t(S) * 4 * g6tts::geom::NCB * sizeof(int32_t)));
        TTS_CK(cudaMalloc(&d_codec_pcm, size_t(S) * 4 * g6tts::geom::FRAME_SAMPLES * sizeof(int16_t)));
        TTS_CK(cudaMalloc(&d_codec_slot, size_t(S) * sizeof(int32_t)));
    }

    void prefill_slot(int slot, SlotState& st) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& c = weights.cfg;
        const int T = st.req.num_rows;
        const int rw = c.row_width();
        const int chunk = bb.max_chunk;

        codec.reset_slot(slot, stream);

        for (int start = 0; start < T; start += chunk) {
            const int end = std::min(start + chunk, T);
            const int len = end - start;
            bb.forward(st.req.rows.data() + size_t(start) * rw, len,
                       true, slot, start, nullptr, nullptr, false, false, stream);
            const int paired_end = std::min(end, T - 1);
            if (paired_end > start) {
                tower.prefill_chunk(bb.d_final_f16,
                                    st.req.rows.data() + size_t(start) * rw,
                                    st.req.rows.data() + size_t(start + 1) * rw,
                                    slot, start, paired_end - start, nullptr, stream);
            }
            if (end == T) {
                // Save last prompt row's backbone hidden state [H]
                TTS_CK(cudaMemcpyAsync(d_slot_hidden + size_t(slot) * c.hidden_size,
                                       bb.d_final_f16 + size_t(len - 1) * c.hidden_size,
                                       size_t(c.hidden_size) * sizeof(__half),
                                       cudaMemcpyDeviceToDevice, stream));
            }
        }

        // Initialize slot's 16-entry RAS ring from the last min(16, T) prompt main tokens
        int32_t ring[g6tts::geom::RAS_WINDOW];
        std::fill(std::begin(ring), std::end(ring), -1);
        const int take = std::min(T, g6tts::geom::RAS_WINDOW);
        for (int i = 0; i < take; ++i) {
            ring[i] = st.req.rows[size_t(T - take + i) * rw];
        }
        const int32_t rpos = take;
        TTS_CK(cudaMemcpyAsync(d_ras_ring + size_t(slot) * g6tts::geom::RAS_WINDOW,
                               ring, sizeof(ring), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaMemcpyAsync(d_ras_pos + slot, &rpos, sizeof(rpos), cudaMemcpyHostToDevice, stream));
        TTS_CK(cudaStreamSynchronize(stream));

        st.pos = T - 1;
        st.frames_done = 0;
        st.stride_idx = 0;
        st.pending_codes.clear();
        st.last_frame.assign(st.req.rows.data() + size_t(T - 1) * rw,
                             st.req.rows.data() + size_t(T) * rw);
        st.t_start = t0;
        st.prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        st.active = true;
    }

    void flush_slot_audio(int slot, SlotState& st,
                          const std::function<bool(const std::string&)>& safe_send) {
        const int n_frames = static_cast<int>(st.pending_codes.size()) / g6tts::geom::NCB;
        if (n_frames <= 0) return;
        std::vector<int16_t> pcm;
        const bool want_pcm = (st.req.emit == g6tts::EmitMode::Pcm || st.req.emit == g6tts::EmitMode::Both);
        const bool want_codes = (st.req.emit == g6tts::EmitMode::Codes || st.req.emit == g6tts::EmitMode::Both);
        if (want_pcm) {
            pcm.resize(size_t(n_frames) * g6tts::geom::FRAME_SAMPLES, 0);
            if (codec.loaded()) {
                int done = 0;
                while (done < n_frames) {
                    const int s = std::min(4, n_frames - done);
                    TTS_CK(cudaMemcpyAsync(d_codec_codes,
                                           st.pending_codes.data() + size_t(done) * g6tts::geom::NCB,
                                           size_t(s) * g6tts::geom::NCB * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, stream));
                    TTS_CK(cudaMemcpyAsync(d_codec_slot, &slot, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    codec.decode_frames(d_codec_codes, 1, s, d_codec_slot, d_codec_pcm, stream);
                    TTS_CK(cudaMemcpyAsync(pcm.data() + size_t(done) * g6tts::geom::FRAME_SAMPLES,
                                           d_codec_pcm,
                                           size_t(s) * g6tts::geom::FRAME_SAMPLES * sizeof(int16_t),
                                           cudaMemcpyDeviceToHost, stream));
                    TTS_CK(cudaStreamSynchronize(stream));
                    done += s;
                }
            }
        }
        safe_send(g6tts::make_audio_json(
            st.req.id, n_frames,
            want_pcm ? pcm.data() : nullptr,
            want_codes ? st.pending_codes.data() : nullptr));
        st.pending_codes.clear();
    }

    void run_loop(g6tts::TtsQueue& queue,
                  const std::function<bool(const std::string&)>& safe_send,
                  g6tts::TtsMetrics& metrics) {
        const auto& c = weights.cfg;
        const int S = c.max_streams;
        const int H = c.hidden_size;
        const int rw = c.row_width();
        std::vector<SlotState> slots(static_cast<size_t>(S));

        std::vector<int> active_slots;
        std::vector<int32_t> h_slots(static_cast<size_t>(S)), h_pos(static_cast<size_t>(S)), h_frames(static_cast<size_t>(S) * rw);
        std::vector<float> h_temp(static_cast<size_t>(S)), h_top_p(static_cast<size_t>(S));
        std::vector<int32_t> h_top_k(static_cast<size_t>(S));
        std::vector<uint64_t> h_seed(static_cast<size_t>(S));
        std::vector<uint32_t> h_fidx(static_cast<size_t>(S));
        std::vector<int32_t> h_next_slots(static_cast<size_t>(S)), h_next_pos(static_cast<size_t>(S)), h_next_rows(static_cast<size_t>(S) * rw);

        for (;;) {
            // 1. Service pending encode_audio requests
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
                    // Transpose codebook-major [10, F] -> wire frame-major [F, 10]
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

            // 2. Admit queued speech requests into free slots
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

            // 4. Batched decode step over M = active_slots.size()
            const int M = static_cast<int>(active_slots.size());
            for (int m = 0; m < M; ++m) {
                const int s = active_slots[size_t(m)];
                const SlotState& st = slots[size_t(s)];
                h_slots[size_t(m)] = s;
                h_pos[size_t(m)] = st.pos;
                h_temp[size_t(m)] = st.req.temperature;
                h_top_p[size_t(m)] = st.req.top_p;
                h_top_k[size_t(m)] = st.req.top_k;
                h_seed[size_t(m)] = st.req.seed;
                h_fidx[size_t(m)] = static_cast<uint32_t>(st.frames_done);
                std::memcpy(h_frames.data() + size_t(m) * rw, st.last_frame.data(), size_t(rw) * sizeof(int32_t));
                TTS_CK(cudaMemcpyAsync(d_step_hidden + size_t(m) * H,
                                       d_slot_hidden + size_t(s) * H,
                                       size_t(H) * sizeof(__half),
                                       cudaMemcpyDeviceToDevice, stream));
            }
            TTS_CK(cudaMemcpyAsync(d_step_slots, h_slots.data(), size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_pos, h_pos.data(), size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_frames, h_frames.data(), size_t(M) * rw * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_temp, h_temp.data(), size_t(M) * sizeof(float), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_top_p, h_top_p.data(), size_t(M) * sizeof(float), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_top_k, h_top_k.data(), size_t(M) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_seed, h_seed.data(), size_t(M) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
            TTS_CK(cudaMemcpyAsync(d_step_fidx, h_fidx.data(), size_t(M) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

            // 4a. Sliced lm_head + main token sampling with RAS
            TTS_BLAS_CK(cublasSetStream(bb.blas, stream));
            g6tts::gemm_f16(bb.blas, d_step_hidden, weights.lm_head_cand, bb.d_main_logits,
                            M, c.main_candidates(), H, true, 0.0f);
            g6tts::k_sample_main_ras<<<M, 256, 0, stream>>>(
                bb.d_main_logits, weights.main_cand_ids_dev, d_ras_ring, d_ras_pos,
                d_step_slots, d_step_temp, d_step_top_p, d_step_top_k, d_step_seed, d_step_fidx,
                d_out_main, d_out_c0, d_out_sem,
                M, c.main_candidates(), c.semantic_start_id, c.semantic_end_id, c.tower_vocab);

            // 4b. 4 shared tower layers
            __half* cur_hid = tower.run_shared(d_step_hidden, d_out_c0, d_out_sem,
                                               d_step_frames, d_step_slots, d_step_pos, M, stream);

            // Stage main + c0 to host
            std::vector<int32_t> tmp_main(static_cast<size_t>(M)), tmp_c(static_cast<size_t>(M));
            TTS_CK(cudaMemcpyAsync(tmp_main.data(), d_out_main, size_t(M) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            TTS_CK(cudaMemcpyAsync(tmp_c.data(), d_out_c0, size_t(M) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

            // 4c. 9 residual tower layers + heads + sampling
            const int32_t* prev_code_dev = d_out_c0;
            std::vector<std::vector<int32_t>> res_codes(static_cast<size_t>(c.tower_res), std::vector<int32_t>(static_cast<size_t>(M)));
            for (int i = 0; i < c.tower_res; ++i) {
                cur_hid = tower.run_residual_step(i, cur_hid, prev_code_dev, d_out_sem,
                                                  d_step_slots, d_step_pos, M, g6tts::geom::RES_CB_SIZE, true, stream);
                g6tts::k_sample_residual<<<M, 256, 0, stream>>>(
                    tower.d_head_logits, d_step_temp, d_step_top_p, d_step_top_k,
                    d_step_seed, d_step_fidx, d_out_code, M, g6tts::geom::RES_CB_SIZE, i + 1);
                TTS_CK(cudaMemcpyAsync(res_codes[size_t(i)].data(), d_out_code,
                                       size_t(M) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                prev_code_dev = d_out_code;
            }
            TTS_CK(cudaStreamSynchronize(stream));

            // 5. Commit sampled frames, emit audio/finish, collect continuing slots for backbone step
            int M_next = 0;
            for (int m = 0; m < M; ++m) {
                const int s = active_slots[size_t(m)];
                SlotState& st = slots[size_t(s)];

                if (queue.is_cancelled(st.req.id)) {
                    st.active = false;
                    queue.mark_done(st.req.id);
                    metrics.req_abort.fetch_add(1);
                    safe_send(g6tts::make_finish_json(st.req.id, "abort", st.req.num_rows, st.frames_done));
                    continue;
                }

                const int32_t main_tok = tmp_main[size_t(m)];
                if (main_tok == c.eos_token_id) {
                    flush_slot_audio(s, st, safe_send);
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

                st.last_frame[0] = main_tok;
                st.last_frame[1] = tmp_c[size_t(m)];
                for (int i = 0; i < c.tower_res; ++i) st.last_frame[size_t(2 + i)] = res_codes[size_t(i)][size_t(m)];
                for (int cb = 0; cb < g6tts::geom::NCB; ++cb) st.pending_codes.push_back(st.last_frame[size_t(1 + cb)]);
                ++st.frames_done;
                ++st.pos;

                const int target_stride = voc_strides[std::min(st.stride_idx, voc_strides.size() - 1)];
                const int cur_buf = static_cast<int>(st.pending_codes.size()) / g6tts::geom::NCB;
                if (cur_buf >= target_stride || st.frames_done >= st.req.max_frames) {
                    flush_slot_audio(s, st, safe_send);
                    if (st.stride_idx + 1 < voc_strides.size()) ++st.stride_idx;
                }

                if (st.frames_done >= st.req.max_frames) {
                    const double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - st.t_start).count();
                    const double dec_s = std::max(0.0, tot_s - st.prefill_s);
                    st.active = false;
                    queue.mark_done(st.req.id);
                    metrics.req_length.fetch_add(1);
                    metrics.prompt_tokens.fetch_add(uint64_t(st.req.num_rows));
                    metrics.gen_frames.fetch_add(uint64_t(st.frames_done));
                    safe_send(g6tts::make_finish_json(st.req.id, "length", st.req.num_rows, st.frames_done,
                                                      -1, "", 0, st.prefill_s, dec_s));
                    continue;
                }

                h_next_slots[size_t(M_next)] = s;
                h_next_pos[size_t(M_next)] = st.pos;
                std::memcpy(h_next_rows.data() + size_t(M_next) * rw, st.last_frame.data(), size_t(rw) * sizeof(int32_t));
                ++M_next;
            }

            // 6. Batched backbone forward for continuing slots -> update `d_slot_hidden`
            if (M_next > 0) {
                bb.forward(h_next_rows.data(), M_next, false, 0, 0,
                           h_next_slots.data(), h_next_pos.data(), false, false, stream);
                for (int i = 0; i < M_next; ++i) {
                    const int s = h_next_slots[size_t(i)];
                    TTS_CK(cudaMemcpyAsync(d_slot_hidden + size_t(s) * H,
                                           bb.d_final_f16 + size_t(i) * H,
                                           size_t(H) * sizeof(__half),
                                           cudaMemcpyDeviceToDevice, stream));
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
        std::fprintf(stderr, "g6tts: state=%.2f GiB, listening on %s (model=%s, streams=%d, cap=%d)\n",
                     double(engine.bb.state_bytes + engine.tower.state_bytes) / double(1ull << 30),
                     opts.socket_path.c_str(), engine.weights.cfg.model_name.c_str(),
                     opts.streams, opts.cap);

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
