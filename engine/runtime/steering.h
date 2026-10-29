// Host-visible steering contracts shared by the profile loader, request protocol and CPU tests.
// CUDA storage and fused kernels live with the runtime paths that consume them.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace g6rt::steering {

constexpr float SCALE_MAX = 4.0f;
constexpr uint32_t PROFILES_MAX = 8;    // profiles one boot may load
constexpr size_t MANIFEST_MAX_BYTES = 1u << 20;
constexpr size_t DIRECTIONS_MAX_BYTES = 64u << 20;

enum class Scope : uint8_t { Shared = 1, Full = 2 };

inline const char* scope_name(Scope s) {
    return s == Scope::Shared ? "shared" : s == Scope::Full ? "full" : "invalid";
}

inline bool valid_profile_name(const std::string& s) {
    auto alnum = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); };
    if (s.empty() || s.size() > 128 || !alnum(s.front())) return false;
    for (char c : s) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return false;
    return true;
}

inline bool hex_digest(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

inline uint32_t float_bits(float v) {
    uint32_t out;
    static_assert(sizeof out == sizeof v);
    std::memcpy(&out, &v, sizeof out);
    return out;
}

inline float bits_float(uint32_t v) {
    float out;
    std::memcpy(&out, &v, sizeof out);
    return out;
}

inline uint32_t parse_scale_bits(const std::string& s) {
    if (s.size() != 8) throw std::runtime_error("steering scale bits must be eight lowercase hex digits");
    uint32_t bits = 0;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            throw std::runtime_error("steering scale bits must be eight lowercase hex digits");
        bits = bits * 16 + uint32_t(c <= '9' ? c - '0' : c - 'a' + 10);
    }
    const float scale = bits_float(bits);
    if (scale == 0.0f) return 0; // includes -0
    if (!std::isfinite(scale) || scale < 0.0f || scale > SCALE_MAX)
        throw std::runtime_error("steering scale bits are outside [0, 4]");
    return bits;
}

inline std::string digest_hex(const std::array<uint8_t, 32>& d) {
    static const char* h = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < d.size(); ++i) { out[2*i] = h[d[i] >> 4]; out[2*i+1] = h[d[i] & 15]; }
    return out;
}

// The request wire and cache key use the exact canonical FP32 bits.  Conversion to FP32 happens once at the API
// boundary.  -0 is the sole non-identity representation that is normalized.
inline uint32_t canonical_scale_bits(double v) {
    if (!std::isfinite(v) || v < 0.0 || v > double(SCALE_MAX))
        throw std::runtime_error("steering scale must be finite and in [0, 4]");
    const float f = float(v);
    if (!std::isfinite(f) || f < 0.0f || f > SCALE_MAX)
        throw std::runtime_error("steering scale is not representable in [0, 4]");
    return f == 0.0f ? 0u : float_bits(f);
}

struct Identity {
    bool enabled = false;
    std::array<uint8_t, 32> profile_digest{};
    Scope scope = Scope::Full;
    std::array<uint8_t, 32> schedule_digest{};
    uint32_t scale_bits = 0;
    uint32_t slot = 0;               // index of the profile in the boot's load order; the digest also identifies it

    static Identity off() { return {}; }
    bool operator==(const Identity& b) const {
        if (enabled != b.enabled) return false;
        return !enabled || (profile_digest == b.profile_digest && scope == b.scope &&
                            schedule_digest == b.schedule_digest && scale_bits == b.scale_bits && slot == b.slot);
    }
    bool operator!=(const Identity& b) const { return !(*this == b); }
};

// Exact prefix-cache namespace. Stock is all zeroes; a selected profile carries every identity byte, including the
// canonical FP32 scale. This is passed through kvman's retained-run key rather than compressed into a collision-prone
// integer hash.
constexpr size_t CACHE_NAMESPACE_BYTES = 70;
inline std::array<uint8_t, CACHE_NAMESPACE_BYTES> cache_namespace(const Identity& s) {
    std::array<uint8_t, CACHE_NAMESPACE_BYTES> out{};
    if (!s.enabled) return out;
    out[0] = 1;
    std::copy(s.profile_digest.begin(), s.profile_digest.end(), out.begin() + 1);
    out[33] = uint8_t(s.scope);
    std::copy(s.schedule_digest.begin(), s.schedule_digest.end(), out.begin() + 34);
    for (unsigned i = 0; i < 4; ++i) out[66 + i] = uint8_t(s.scale_bits >> (8 * i));
    return out;
}

// FNV-1a is the runtime's replicated-state hash. This helper preserves its byte order and makes steering part of the
// same fail-stop agreement check without treating FNV as the profile's cryptographic identity.
inline void hash_identity(uint64_t& h, const Identity& s) {
    auto add = [&](uint8_t v) { h = (h ^ v) * 1099511628211ull; };
    add(s.enabled ? 1 : 0);
    if (!s.enabled) return;
    for (uint8_t v : s.profile_digest) add(v);
    add(uint8_t(s.scope));
    for (uint8_t v : s.schedule_digest) add(v);
    for (unsigned i = 0; i < 4; ++i) add(uint8_t(s.scale_bits >> (8 * i)));
}

struct Profile {
    uint32_t schema_version = 0;
    std::string name;
    std::string model_name, model_revision, weight_set;
    std::string tokenizer_revision, chat_template_sha256;
    uint32_t hidden_size = 0, layer_count = 0;
    Scope scope = Scope::Full;
    std::vector<uint32_t> layers;
    std::vector<float> alphas;
    std::vector<float> directions; // layer-major validated FP32; immutable after boot
    std::array<uint8_t, 32> manifest_digest{}, tensor_digest{}, schedule_digest{};
    std::string manifest_json;

};

struct RequestFields {
    bool has_profile = false, has_scale = false;
    std::string profile;
    double scale = 0.0;
    bool any() const { return has_profile || has_scale; }
};

// The immutable profiles loaded at boot, in load order (a request's slot). All share one scope. Forced mode needs
// exactly one profile and makes every request use it.
struct BootConfig {
    struct Slot {
        std::string name;
        std::array<uint8_t, 32> profile_digest{}, schedule_digest{};
        uint32_t active_layers = 0;
    };
    bool loaded = false, forced = false;
    std::vector<Slot> profiles;
    Scope scope = Scope::Full;
    uint32_t forced_scale_bits = 0;

    static BootConfig from_profiles(const std::vector<Profile>& ps) {
        BootConfig b;
        for (const auto& p : ps) {
            if (b.loaded && p.scope != b.scope) throw std::runtime_error("steering: loaded profiles differ in scope");
            for (const auto& q : b.profiles)
                if (q.name == p.name) throw std::runtime_error("steering: two loaded profiles are named " + p.name);
            if (b.profiles.size() == PROFILES_MAX)
                throw std::runtime_error("steering: at most " + std::to_string(PROFILES_MAX) + " profiles");
            b.loaded = true; b.scope = p.scope;
            b.profiles.push_back({p.name, p.manifest_digest, p.schedule_digest, uint32_t(p.layers.size())});
        }
        return b;
    }
    static BootConfig from_profile(const Profile& p) { return from_profiles({p}); }
    Identity selected(uint32_t bits, uint32_t slot = 0) const {
        bits = canonical_scale_bits(bits_float(bits));
        if (!bits) return Identity::off();
        if (!loaded) throw std::runtime_error("no steering profile is loaded");
        const Slot& p = profiles.at(slot);
        return {true, p.profile_digest, scope, p.schedule_digest, bits, slot};
    }
    Identity resolve(const RequestFields& f) const {
        uint32_t bits = 0, slot = 0;
        if (f.has_profile) {
            if (!loaded) throw std::runtime_error("steering is not enabled on this server");
            while (slot < profiles.size() && profiles[slot].name != f.profile) ++slot;
            if (slot == profiles.size()) throw std::runtime_error("unknown steering_profile");
            bits = f.has_scale ? canonical_scale_bits(f.scale) : canonical_scale_bits(1.0);
        } else if (f.has_scale) {
            bits = canonical_scale_bits(f.scale);
            if (bits) throw std::runtime_error("steering_scale requires steering_profile");
        }
        const Identity requested = selected(bits, slot);
        if (!forced) return requested;
        const Identity imposed = selected(forced_scale_bits);
        if (f.any() && requested != imposed) throw std::runtime_error("request steering conflicts with forced mode");
        return imposed;
    }
};

inline void hash_boot(uint64_t& h, const BootConfig& b) {
    auto add = [&](uint8_t v) { h = (h ^ v) * 1099511628211ull; };
    add(b.loaded); add(b.forced);
    if (!b.loaded) return;
    for (const auto& p : b.profiles) {
        for (char c : p.name) add(uint8_t(c));
        add(0);
        for (uint8_t v : p.profile_digest) add(v);
        add(uint8_t(b.scope));
        for (uint8_t v : p.schedule_digest) add(v);
        for (unsigned i = 0; i < 4; ++i) add(uint8_t(b.forced_scale_bits >> (8 * i)));
        for (unsigned i = 0; i < 4; ++i) add(uint8_t(p.active_layers >> (8 * i)));
    }
}

// CPU oracle for the writer-side operator. A zero effective alpha performs no loads or stores through `row`.
inline void project(float* row, const float* direction, size_t width, float effective_alpha) {
    if (effective_alpha == 0.0f) return;
    double dot = 0.0;
    for (size_t i = 0; i < width; ++i) dot += double(row[i]) * double(direction[i]);
    const double k = double(effective_alpha) * dot;
    for (size_t i = 0; i < width; ++i) row[i] = float(double(row[i]) - k * double(direction[i]));
}

} // namespace g6rt::steering
