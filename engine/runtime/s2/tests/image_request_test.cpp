// Image requests, host side: the img header fields of an S2 request (s2/sessions.h parse_request, image_input.h), the
// prompt checks, the image digest in the prefix-cache namespace and the control protocol, and the refusals
// (Service::images_ok). The DS41 geometry first, then GLM-5.3-Flash's (image_model() set to GLM5V_IMAGES).
#include "s2/dispatch.h"
#include "s2/control_protocol.h"
#include <cstdio>

using namespace g6s2;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::string b64(const std::vector<uint8_t>& b) {
    static const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string s;
    for (size_t i = 0; i < b.size(); i += 3) {
        uint32_t w = uint32_t(b[i]) << 16 | (i + 1 < b.size() ? uint32_t(b[i + 1]) << 8 : 0) | (i + 2 < b.size() ? b[i + 2] : 0);
        s += A[w >> 18 & 63]; s += A[w >> 12 & 63];
        s += i + 1 < b.size() ? A[w >> 6 & 63] : '='; s += i + 2 < b.size() ? A[w & 63] : '=';
    }
    return s;
}
// a 1 x 1 patch grid (span 4) of one grey level
static std::string field(int pos, uint8_t grey, int vh = 1, int vw = 1) {
    return "img " + std::to_string(pos) + ":" + std::to_string(vh) + ":" + std::to_string(vw) + ":0123456789abcdef:" +
           b64(std::vector<uint8_t>(size_t(vh) * vw * 14 * 14 * 3, grey));
}
static std::string prompt(std::initializer_list<int> ids) {
    std::string s; for (int i : ids) s += (s.empty() ? "" : ",") + std::to_string(i); return s;
}
static bool refused(const std::string& bytes) {
    try { parse_request(bytes); return false; } catch (const std::exception&) { return true; }
}

int main() {
    constexpr int T = g6rt::TOK_IMAGE;
    const std::string head = "n_gen 4 temp 0.0 seed 1 min 0 stop - top_p 1.0";
    const std::string p6 = prompt({5, T, T, T, T, 6});
    // one image: parsed, pixels decoded, identity carries the digest
    auto r = parse_request(head + " " + field(1, 7) + "\n" + p6 + "\n");
    CHECK(r.images.size() == 1 && r.images[0].pos == 1 && r.images[0].span == 4 && r.images[0].rgb->size() == 588);
    CHECK((*r.images[0].rgb)[0] == 7 && r.steering.images && !r.steering.enabled);
    const auto ns = g6rt::steering::cache_namespace(r.steering);
    CHECK(ns[0] == 2 && std::equal(r.steering.image_digest.begin(), r.steering.image_digest.end(), ns.begin() + 1));
    // a text request is unchanged: stock namespace, no images
    auto t = parse_request(head + "\n" + prompt({5, 6}) + "\n");
    CHECK(t.images.empty() && !t.steering.images && g6rt::steering::cache_namespace(t.steering) == decltype(ns){});
    // other pixels or another position: another namespace; the same: the same
    auto r2 = parse_request(head + " " + field(1, 8) + "\n" + p6 + "\n");
    auto r3 = parse_request(head + " " + field(1, 7) + "\n" + p6 + "\n");
    CHECK(r2.steering != r.steering && r3.steering == r.steering && g6rt::steering::cache_namespace(r3.steering) == ns);
    auto r4 = parse_request(head + " " + field(2, 7) + "\n" + prompt({5, 6, T, T, T, T, 6}) + "\n");
    CHECK(r4.steering != r.steering);
    uint64_t h0 = 1469598103934665603ull, h1 = h0;
    g6rt::steering::hash_identity(h0, r.steering); g6rt::steering::hash_identity(h1, r2.steering); CHECK(h0 != h1);
    // two images, in order
    auto two = parse_request(head + " " + field(1, 1) + " " + field(5, 2) + "\n" + prompt({5, T, T, T, T, T, T, T, T, 6}) + "\n");
    CHECK(two.images.size() == 2 && two.images[1].pos == 5);
    // malformed or inconsistent fields end the request at parse (the shim checks the same first)
    for (const std::string& bad : {
             head + " " + field(2, 7) + "\n" + p6 + "\n",                      // the span runs onto a text token
             head + " " + field(1, 7) + " " + field(3, 7) + "\n" + prompt({5, T, T, T, T, T, T, 6}) + "\n",   // overlap
             head + " " + field(1, 7, 0, 1) + "\n" + p6 + "\n",               // bad grid
             head + " img 1:1:1:0123456789abcdef:AAAA\n" + p6 + "\n",          // wrong size
             head + " img 1:1:1:XYZ:" + b64(std::vector<uint8_t>(588)) + "\n" + p6 + "\n",
             head + " img 1:1:1:0123456789abcdef:" + b64(std::vector<uint8_t>(588)).substr(1) + "\n" + p6 + "\n",
             head + " " + field(1, 7) + " lp 2\n" + p6 + "\n",                  // img is last
         })
        CHECK(refused(bad));
    // the control protocol carries the digest; text identities keep their bytes
    for (const auto& id : {r.steering, t.steering}) {
        Encoder e(Tag::Request); e.steering(id);
        Decoder d(e.b); CHECK(d.steering() == id); d.end();
    }
    { Encoder e(Tag::Request); e.steering(t.steering); CHECK(e.b.size() == 3 * 8); }
    // refusals: no tower, a decode-step prompt, a prompt that ends inside a span
    g6rt::vision_flag() = false;
    CHECK(*g6rt::img::refusal(r.prompt, r.images, false));
    g6rt::vision_flag() = true;
    CHECK(!*g6rt::img::refusal(r.prompt, r.images, false));
    CHECK(*g6rt::img::refusal(r.prompt, r.images, true));
    auto end = parse_request(head + " " + field(1, 7) + "\n" + prompt({5, T, T, T, T}) + "\n");
    CHECK(*g6rt::img::refusal(end.prompt, end.images, false));
    CHECK(!*g6rt::img::refusal(t.prompt, t.images, true));
    // GLM-5.3-Flash: <|image|> rows, one per 2 x 2 patches, 16 to 8000 tokens
    g6rt::image_model() = g6rt::GLM5V_IMAGES;
    {
        constexpr int G = g6rt::TOK_IMAGE_GLM5V;
        std::vector<int> ids{5, 154830};
        ids.insert(ids.end(), 16, G);
        ids.insert(ids.end(), {154831, 6, 7});
        std::string p;
        for (int i : ids) p += (p.empty() ? "" : ",") + std::to_string(i);
        auto f = parse_request(head + " " + field(2, 9, 8, 8) + "\n" + p + "\n");
        CHECK(f.images.size() == 1 && f.images[0].span == 16 && f.images[0].rgb->size() == size_t(8 * 8 * 14 * 14 * 3));
        CHECK(f.steering.images && !*g6rt::img::refusal(f.prompt, f.images, false));
        CHECK(*g6rt::img::refusal(f.prompt, f.images, true));
        for (const std::string& bad : {
                 head + " " + field(1, 9, 8, 8) + "\n" + p + "\n",            // the span starts on <|begin_of_image|>
                 head + " " + field(2, 9, 6, 6) + "\n" + p + "\n",            // 9 tokens: below the floor of 16
                 head + " " + field(2, 9, 7, 8) + "\n" + p + "\n",            // an odd side
                 head + " " + field(2, 9, 2, 16002) + "\n" + p + "\n",        // past 8000 tokens
                 head + " " + field(1, 7) + "\n" + p6 + "\n",                 // DS41's token and grid
             })
            CHECK(refused(bad));
    }
    g6rt::image_model() = g6rt::DS41_IMAGES;
    if (fails) { std::printf("%d FAILED\n", fails); return 1; }
    std::printf("image_request_test OK\n");
    return 0;
}
