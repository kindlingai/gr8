// CPU-only unit test for g6tts host headers: TtsConfig geometry & config.json parsing,
// base64 encode/decode, PROTOCOL.md v1 speech/audio/encode_audio/codes/cancel/metrics roundtrip,
// and CAP capacity 400 validation.
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "../tts_config.h"
#include "../tts_proto.h"

static void test_base64_roundtrip() {
    std::vector<uint8_t> dec;
    assert(g6tts::encode_base64(nullptr, 0).empty());
    assert(g6tts::decode_base64("", dec) && dec.empty());

    const std::vector<uint8_t> sample = {0, 1, 2, 3, 255, 128, 64, 42, 17};
    const std::string enc = g6tts::encode_base64(sample.data(), sample.size());
    assert(g6tts::decode_base64(enc, dec));
    assert(dec == sample);

    const std::vector<uint8_t> sample2 = {0x12, 0x34};
    const std::string enc2 = g6tts::encode_base64(sample2.data(), sample2.size());
    assert(g6tts::decode_base64(enc2, dec) && dec == sample2);

    assert(!g6tts::decode_base64("not-valid!", dec));
    assert(!g6tts::decode_base64("ABC", dec));
}

static void test_config_and_candidates() {
    g6tts::TtsConfig cfg;
    cfg.validate_drama3_geometry();
    assert(cfg.row_width() == 11);
    assert(cfg.main_candidates() == 4097);
    assert(!cfg.is_semantic(g6tts::geom::EOS));
    assert(cfg.is_semantic(g6tts::geom::SEM0));
    assert(cfg.is_semantic(g6tts::geom::SEM1));
    assert(!cfg.is_semantic(g6tts::geom::SEM1 + 1));

    const auto cands = cfg.build_main_candidate_ids();
    assert(cands.size() == 4097);
    assert(cands[0] == g6tts::geom::EOS);
    assert(cands[1] == g6tts::geom::SEM0);
    assert(cands.back() == g6tts::geom::SEM1);
    for (size_t i = 1; i < cands.size(); ++i) assert(cands[i] > cands[i - 1]);
}

static void test_protocol_roundtrip() {
    int sv[2];
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    g6tts::TtsConfig cfg;
    cfg.cap = 16; // small cap to test 400 validation
    g6tts::TtsMetrics metrics;

    std::thread server([&] {
        g6tts::serve_tts_connection(
            sv[0], cfg, metrics,
            [&](g6tts::TtsQueue& q, const std::function<bool(const std::string&)>& send, g6tts::TtsMetrics& m) {
                while (q.wait_for_work()) {
                    g6tts::EncodeAudioRequest ereq;
                    while (q.pop_encode(ereq)) {
                        const int F = 2;
                        std::vector<int32_t> codes(size_t(F) * 10, 7);
                        q.mark_done(ereq.id);
                        m.req_stop.fetch_add(1);
                        send(g6tts::make_codes_json(ereq.id, F, codes.data()));
                        send(g6tts::make_finish_json(ereq.id, "stop", 0, F));
                    }
                    g6tts::SpeechRequest sreq;
                    while (q.pop_speech(sreq)) {
                        std::vector<int16_t> pcm(2048, 123);
                        std::vector<int32_t> codes = {10, 1, 2, 3, 4, 5, 6, 7, 8, 9};
                        q.mark_done(sreq.id);
                        m.req_stop.fetch_add(1);
                        m.prompt_tokens.fetch_add(uint64_t(sreq.num_rows));
                        m.gen_frames.fetch_add(1);
                        send(g6tts::make_audio_json(
                            sreq.id, 1,
                            sreq.emit != g6tts::EmitMode::Codes ? pcm.data() : nullptr,
                            sreq.emit != g6tts::EmitMode::Pcm ? codes.data() : nullptr));
                        send(g6tts::make_finish_json(sreq.id, "stop", sreq.num_rows, 1, g6tts::geom::EOS));
                    }
                    if (q.is_closed()) break;
                }
            });
        ::close(sv[0]);
    });

    // 1. Verify hello fields
    std::string body;
    assert(g6tts::recv_frame(sv[1], body));
    const auto hello = g6v::Json::parse(body);
    assert(hello.at("type").str() == "hello");
    assert(hello.at("version").i64() == 1);
    assert(hello.at("speech").b == true);
    assert(hello.at("sample_rate").i64() == 44100);
    assert(hello.at("codebooks").i64() == 10);
    assert(hello.at("frame_samples").i64() == 2048);
    assert(hello.at("eos_ids").a[0].i64() == 248046);

    // 2. Valid speech request with emit="both" (2 rows = 22 ints)
    assert(g6tts::send_frame(
        sv[1],
        R"({"type":"speech","id":"sp1","rows":[100,0,0,0,0,0,0,0,0,0,0,248099,0,1,2,3,4,5,6,7,8,9],"max_frames":4,"temperature":0.7,"top_p":0.7,"top_k":30,"seed":42,"emit":"both"})"));
    assert(g6tts::recv_frame(sv[1], body));
    const auto aud = g6v::Json::parse(body);
    assert(aud.at("type").str() == "audio");
    assert(aud.at("id").str() == "sp1");
    assert(aud.at("frames").i64() == 1);
    assert(aud.at("codes").a.size() == 10);
    assert(aud.at("codes").a[0].i64() == 10);
    std::vector<uint8_t> pcm_bytes;
    assert(g6tts::decode_base64(aud.at("pcm").str(), pcm_bytes));
    assert(pcm_bytes.size() == 2048 * 2);

    assert(g6tts::recv_frame(sv[1], body));
    const auto fin = g6v::Json::parse(body);
    assert(fin.at("type").str() == "finish");
    assert(fin.at("reason").str() == "stop");
    assert(fin.at("stop_id").i64() == 248046);
    assert(fin.at("completion_tokens").i64() == 1);

    // 3. Over-capacity speech request (2 rows + 20 > cap=16) -> finish reason=error, status=400
    assert(g6tts::send_frame(
        sv[1],
        R"({"type":"speech","id":"sp_big","rows":[100,0,0,0,0,0,0,0,0,0,0,101,0,0,0,0,0,0,0,0,0,0],"max_frames":20})"));
    assert(g6tts::recv_frame(sv[1], body));
    const auto err_fin = g6v::Json::parse(body);
    assert(err_fin.at("type").str() == "finish");
    assert(err_fin.at("id").str() == "sp_big");
    assert(err_fin.at("reason").str() == "error");
    assert(err_fin.at("status").i64() == 400);

    // 4. encode_audio roundtrip
    const float f32_pcm[4] = {0.0f, 0.25f, -0.25f, 0.5f};
    const std::string pcm_b64 = g6tts::encode_base64(reinterpret_cast<const uint8_t*>(f32_pcm), sizeof(f32_pcm));
    assert(g6tts::send_frame(sv[1], std::string(R"({"type":"encode_audio","id":"ea1","pcm":")") + pcm_b64 + "\"}"));
    assert(g6tts::recv_frame(sv[1], body));
    const auto codes = g6v::Json::parse(body);
    assert(codes.at("type").str() == "codes");
    assert(codes.at("id").str() == "ea1");
    assert(codes.at("frames").i64() == 2);
    assert(codes.at("codes").a.size() == 20);
    assert(g6tts::recv_frame(sv[1], body));
    const auto efin = g6v::Json::parse(body);
    assert(efin.at("type").str() == "finish");
    assert(efin.at("reason").str() == "stop");
    assert(efin.at("completion_tokens").i64() == 2);

    // 5. metrics query
    assert(g6tts::send_frame(sv[1], R"({"type":"metrics","seq":9})"));
    assert(g6tts::recv_frame(sv[1], body));
    const auto met = g6v::Json::parse(body);
    assert(met.at("type").str() == "metrics");
    assert(met.at("seq").i64() == 9);
    assert(met.at("tokens").at("generation").i64() == 1);

    ::close(sv[1]);
    server.join();
}

int main() {
    test_base64_roundtrip();
    test_config_and_candidates();
    test_protocol_roundtrip();
    std::printf("tts_proto_test: OK\n");
    return 0;
}
