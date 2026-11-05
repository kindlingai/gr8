// DS41 image identity in the replicated scheduler state (engine/scheduler-rs): a grant whose steering Identity carries
// images + image_digest goes through the Rust Scheduler (Scheduler::apply -> g6sch_apply); the slot it fills keeps both,
// and Scheduler::digest (g6sch_digest_head + g6sch_digest_slots, src/scheduler.rs hash_identity) changes with the flag
// and with every digest byte, steering on or off. src/scheduler.rs hashes images before its `!enabled` return, so the
// steering-off case is the one a reorder or a dropped block would lose without another test noticing. Scheduler::digest is
// an event diagnostic (never called by next()/complete()), so no serving path would notice a hole in it either; the vector
// transcripts (scheduler_rs_vectors.h) never set images. Each case is a FRESH scheduler with the same config, key, request
// and run, so only the identity differs between the two digests compared. Host only.
#include "s2/scheduler.h"
#include <cstdio>
#include <string>

using namespace g6s2;
namespace st = g6rt::steering;
static int fails = 0;
#define CHECK(x, ...) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #x); \
    std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); ++fails; } } while (0)

// one grant of a request with this identity into a fresh scheduler; the slot it filled and the state digest after it
struct After { Slot slot; uint64_t digest; };
static After grant(const st::Identity& id)
{
    Config cfg; cfg.epoch = 9; cfg.pool_rows = 4096; cfg.max_rows = 8; cfg.max_sessions = 2; cfg.mtp = 2; cfg.prefill_rows = 16;
    Scheduler s(cfg);
    const Grant g{s.next_key(), Request{1, 8, 4, id}, Run{0, 1, 1}, 0};
    Commit c; c.epoch = 9; c.event = 1; c.before_hash = s.digest(); c.admit = g;
    s.apply(c);
    CHECK(s.slots()[g.key.slot].key == g.key && s.slots()[g.key.slot].req.id == 1, "the grant was not admitted into slot %u", unsigned(g.key.slot));
    return {s.slots()[g.key.slot], s.digest()};
}
static std::string hex(uint64_t v) { char b[24]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v); return b; }

int main()
{
    std::array<uint8_t, 32> d{};
    for (unsigned i = 0; i < 32; ++i) d[i] = uint8_t(0xa0 + i);   // nonzero in every byte

    st::Identity text;                               // steering off, no images
    st::Identity img = text; img.images = true; img.image_digest = d;
    st::Identity steer;                              // steering on (a profile), no images
    steer.enabled = true; steer.scale_bits = st::canonical_scale_bits(1.0); steer.slot = 0;
    for (unsigned i = 0; i < 32; ++i) { steer.profile_digest[i] = uint8_t(i); steer.schedule_digest[i] = uint8_t(63 - i); }
    st::Identity steer_img = steer; steer_img.images = true; steer_img.image_digest = d;

    for (const auto* id : {&img, &steer_img})
    {
        const After a = grant(*id), base = grant(id->enabled ? steer : text);
        const char* on = id->enabled ? "steering on" : "steering off";
        // (A) the image identity survives the Rust grant into the slot
        CHECK(a.slot.req.steering.images, "%s: the slot lost images", on);
        CHECK(a.slot.req.steering.image_digest == d, "%s: the slot's image_digest is not the grant's", on);
        CHECK(a.slot.req.steering == *id, "%s: the slot's identity differs from the grant's", on);
        CHECK(grant(*id).digest == a.digest, "%s: the same grant twice gave two digests", on);
        // (B) the flag and each digest byte change the state digest; nothing else differs between the two schedulers
        CHECK(a.digest != base.digest, "%s: images = true + a nonzero image_digest leave the state digest at %s (the text "
              "grant's): scheduler-rs hash_identity does not hash the images", on, hex(a.digest).c_str());
        st::Identity flag = *id; flag.image_digest = {};
        CHECK(grant(flag).digest != base.digest, "%s: images = true (zero digest) leaves the state digest at the text grant's", on);
        for (unsigned i = 0; i < 32; ++i)
        {
            st::Identity other = *id; other.image_digest[i] ^= 0x01;
            const uint64_t o = grant(other).digest;
            CHECK(o != a.digest, "%s: image_digest byte %u changed, the state digest stayed %s", on, i, hex(o).c_str());
        }
        std::printf("image_state_test %s: text %s, images %s\n", on, hex(base.digest).c_str(), hex(a.digest).c_str());
    }
    std::printf(fails ? "image_state_test: %d FAIL\n" : "image_state_test: PASS\n", fails);
    return fails != 0;
}
