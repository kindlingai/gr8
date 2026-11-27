// CPU-only unit test for g6embed host logic: PROTOCOL.md v1 socket framing, base64 RGB decode,
// EmbedConfig inference, 3D M-RoPE position builder, dynamic batch queue, and libg6load embed C-ABI.
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../embed_config.h"
#include "../embed_proto.h"

static void test_base64() {
    std::vector<uint8_t> out;
    assert(g6emb::decode_base64("", out) && out.empty());
    assert(g6emb::decode_base64("AAECAw==", out) && out == (std::vector<uint8_t>{0, 1, 2, 3}));
    assert(g6emb::decode_base64("AQID", out) && out == (std::vector<uint8_t>{1, 2, 3}));
    assert(!g6emb::decode_base64("not-valid!", out));
    assert(!g6emb::decode_base64("ABC", out));
}

static void test_mrope_positions() {
    // 1. Pure text: isotropic (p, p, p)
    std::vector<g6emb::MropePos> pos(5);
    g6emb::build_qwen3vl_mrope_positions(5, {}, pos.data());
    for (int i = 0; i < 5; ++i) {
        assert(pos[size_t(i)].t == i && pos[size_t(i)].h == i && pos[size_t(i)].w == i);
    }

    // 2. Single 4x6 patch image (2x3 merged = 6 tokens) starting at pos=3 inside length-12 prompt:
    //   tokens 0..2: (0,0,0), (1,1,1), (2,2,2)
    //   tokens 3..8 (image): t=3, h=3+{0,1}, w=3+{0,1,2} -> max dim = 3, delta after image = 3 - 6 = -3
    //   tokens 9..11 (text after image): pos + delta = 6, 7, 8!
    g6emb::EmbedImage im;
    im.pos = 3;
    im.vit_h = 4;
    im.vit_w = 6;
    im.span = 6;
    std::vector<g6emb::MropePos> mp(12);
    g6emb::build_qwen3vl_mrope_positions(12, {im}, mp.data());
    assert(mp[2].t == 2 && mp[2].h == 2 && mp[2].w == 2);
    assert(mp[3].t == 3 && mp[3].h == 3 && mp[3].w == 3);
    assert(mp[5].t == 3 && mp[5].h == 3 && mp[5].w == 5);
    assert(mp[6].t == 3 && mp[6].h == 4 && mp[6].w == 3);
    assert(mp[8].t == 3 && mp[8].h == 4 && mp[8].w == 5);
    assert(mp[9].t == 6 && mp[9].h == 6 && mp[9].w == 6);
    assert(mp[11].t == 8 && mp[11].h == 8 && mp[11].w == 8);
}

static void test_batch_queue_packing() {
    g6emb::EmbedBatchQueue q(10);
    g6emb::EmbedRequest r1{"a", {1, 2, 3, 4}, 0, true, {}};
    g6emb::EmbedRequest r2{"b", {1, 2, 3, 4, 5}, 0, true, {}};
    g6emb::EmbedRequest r3{"c", {1, 2, 3}, 0, true, {}};
    assert(q.push(r1));
    assert(!q.push(r1)); // duplicate id refused
    assert(q.push(r2));
    assert(q.push(r3));
    assert(q.cancel_queued("b"));
    assert(!q.cancel_queued("nonexistent"));

    // Remaining: a (4 tok) + c (3 tok) = 7 <= 10 -> packed in one batch!
    auto b1 = q.pop_batch();
    assert(b1.size() == 2 && b1[0].id == "a" && b1[1].id == "c");
    q.mark_done("a");
    q.mark_done("c");
    q.close();
    assert(q.pop_batch().empty());

    // Enforce max_batch cap even when total tokens easily fit within max_batched_tokens.
    g6emb::EmbedBatchQueue qcap(16384, 3);
    for (int i = 0; i < 5; ++i) {
        assert(qcap.push(g6emb::EmbedRequest{"req-" + std::to_string(i), {1, 2}, 0, true, {}}));
    }
    auto c1 = qcap.pop_batch();
    assert(c1.size() == 3 && c1[0].id == "req-0" && c1[2].id == "req-2");
    auto c2 = qcap.pop_batch();
    assert(c2.size() == 2 && c2[0].id == "req-3" && c2[1].id == "req-4");
    for (const auto& r : c1) qcap.mark_done(r.id);
    for (const auto& r : c2) qcap.mark_done(r.id);
    qcap.close();
    assert(qcap.pop_batch().empty());
}

static void test_protocol_roundtrip() {
    int sv[2];
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    g6emb::EmbedConfig cfg;
    cfg.model_name = "qwen3-vl-embedding-2b";
    cfg.vocab_size = 1000;
    cfg.embedding_dim = 8;
    cfg.has_vision = true;
    cfg.matryoshka = true;
    cfg.max_model_len = 64;

    g6emb::EmbedMetrics metrics;
    std::thread server([&] {
        g6emb::serve_connection(sv[0], cfg, 16, metrics, [&](const std::vector<g6emb::EmbedRequest>& batch) {
            std::vector<g6emb::EmbedOutput> outs;
            for (const auto& r : batch) {
                const int d = r.dimensions > 0 ? r.dimensions : cfg.embedding_dim;
                g6emb::EmbedOutput o;
                o.id = r.id;
                o.prompt_tokens = static_cast<int>(r.prompt.size());
                o.embedding.assign(size_t(d), 1.0f / std::sqrt(float(d)));
                o.prefill_s = 0.001;
                outs.push_back(std::move(o));
            }
            return outs;
        });
        ::close(sv[0]);
    });

    // 1. Read hello
    std::string body;
    assert(g6emb::recv_frame(sv[1], body));
    const auto hello = g6v::Json::parse(body);
    assert(hello.at("type").str() == "hello");
    assert(hello.at("version").i64() == 1);
    assert(hello.at("embeddings").b == true);
    assert(hello.at("embedding_dim").i64() == 8);
    assert(hello.at("matryoshka").b == true);
    assert(hello.at("vision").b == true);

    // 2. Send valid embed request with MRL dimensions=4
    assert(g6emb::send_frame(sv[1], R"({"type":"embed","id":"e1","prompt":[1,2,3],"dimensions":4,"normalize":true})"));
    assert(g6emb::recv_frame(sv[1], body));
    const auto emb = g6v::Json::parse(body);
    assert(emb.at("type").str() == "embedding");
    assert(emb.at("id").str() == "e1");
    assert(emb.at("prompt_tokens").i64() == 3);
    assert(emb.at("embedding").a.size() == 4);
    assert(std::fabs(emb.at("embedding").a[0].num() - 0.5) < 1e-5);

    assert(g6emb::recv_frame(sv[1], body));
    const auto fin = g6v::Json::parse(body);
    assert(fin.at("type").str() == "finish");
    assert(fin.at("id").str() == "e1");
    assert(fin.at("reason").str() == "stop");
    assert(fin.at("completion_tokens").i64() == 0);

    // 3. Send invalid dimensions=99 -> finish reason=error directly
    assert(g6emb::send_frame(sv[1], R"({"type":"embed","id":"e2","prompt":[1,2],"dimensions":99})"));
    assert(g6emb::recv_frame(sv[1], body));
    const auto err_fin = g6v::Json::parse(body);
    assert(err_fin.at("type").str() == "finish");
    assert(err_fin.at("id").str() == "e2");
    assert(err_fin.at("reason").str() == "error");

    // 4. Send metrics query
    assert(g6emb::send_frame(sv[1], R"({"type":"metrics","seq":7})"));
    assert(g6emb::recv_frame(sv[1], body));
    const auto met = g6v::Json::parse(body);
    assert(met.at("type").str() == "metrics");
    assert(met.at("seq").i64() == 7);
    assert(met.at("tokens").at("prompt").i64() == 3);

    ::close(sv[1]);
    server.join();
}

int main() {
    test_base64();
    test_mrope_positions();
    test_batch_queue_packing();
    test_protocol_roundtrip();
    std::printf("embed_proto_test: OK\n");
    return 0;
}
