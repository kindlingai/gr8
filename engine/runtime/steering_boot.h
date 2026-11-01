// Host-only verified local-profile boot preparation; no device state changes on failure.
#pragma once
#include "steering_profile.h"
#include "steering_glp.h"
#include <algorithm>
#include <cstdlib>

namespace g6rt::steering {
struct PreparedBoot {
    std::vector<Profile> profiles;   // load order = request slot
    BootConfig config;
};
inline void check_layers(const Profile& p, const std::vector<int>& layers) {
    std::set<int> unique(layers.begin(), layers.end());
    if (unique.size() != layers.size()) throw std::runtime_error("steering: duplicate runtime layer");
    for (uint32_t l : p.layers)
        if (!unique.count(int(l))) throw std::runtime_error("steering: profile layer " + std::to_string(l) + " is absent from --layers");
}
// `directories`: one profile directory or GLP GGUF vector, or several
// separated by commas (each entry's container is detected by magic bytes).
inline PreparedBoot prepare_boot(const std::string& directories, const ExpectedModel& expected,
                                  const std::string& served_template, const std::vector<int>& layers,
                                  const std::string& force = {}) {
    PreparedBoot out;
    if (directories.empty()) {
        if (!force.empty()) throw std::runtime_error("--steering-force needs --steering-profile");
        return out; // No profile means no profile/template file I/O.
    }
    if (served_template.empty()) throw std::runtime_error("--steering-profile needs --steering-chat-template");
    const auto bytes = read_file(served_template, 16u << 20);
    if (digest_hex(sha256(bytes.data(), bytes.size())) != expected.chat_template_sha256)
        throw std::runtime_error("steering: served chat template does not match expected identity");
    for (size_t at = 0;;) {
        const size_t comma = directories.find(',', at);
        const std::string dir = directories.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (dir.empty()) throw std::runtime_error("--steering-profile: empty directory in the list");
        out.profiles.push_back(load_profile_any(dir, expected));
        check_layers(out.profiles.back(), layers);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    out.config = BootConfig::from_profiles(out.profiles);
    if (!force.empty()) {
        if (out.profiles.size() != 1) throw std::runtime_error("--steering-force needs exactly one --steering-profile");
        char* end = nullptr; errno = 0;
        const double value = std::strtod(force.c_str(), &end);
        if (errno || end == force.c_str() || *end) throw std::runtime_error("--steering-force: want a finite number in (0, 4]");
        out.config.forced_scale_bits = canonical_scale_bits(value);
        if (!out.config.forced_scale_bits) throw std::runtime_error("--steering-force: want a positive FP32 scale");
        out.config.forced = true;
    }
    return out;
}
} // namespace g6rt::steering
