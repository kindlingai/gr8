// Host-only verified local-profile boot preparation; no device state changes on failure.
#pragma once
#include "steering_profile.h"
#include <algorithm>
#include <cstdlib>

namespace g6rt::steering {
struct PreparedBoot {
    Profile profile;
    BootConfig config;
};
inline void check_layers(const Profile& p, const std::vector<int>& layers) {
    std::set<int> unique(layers.begin(), layers.end());
    if (unique.size() != layers.size()) throw std::runtime_error("steering: duplicate runtime layer");
    for (uint32_t l : p.layers)
        if (!unique.count(int(l))) throw std::runtime_error("steering: profile layer " + std::to_string(l) + " is absent from --layers");
}
inline PreparedBoot prepare_boot(const std::string& directory, const ExpectedModel& expected,
                                  const std::string& served_template, const std::vector<int>& layers,
                                  const std::string& force = {}) {
    PreparedBoot out;
    if (directory.empty()) {
        if (!force.empty()) throw std::runtime_error("--steering-force needs --steering-profile");
        return out; // No profile means no profile/template file I/O.
    }
    if (served_template.empty()) throw std::runtime_error("--steering-profile needs --steering-chat-template");
    const auto bytes = read_file(served_template, 16u << 20);
    if (digest_hex(sha256(bytes.data(), bytes.size())) != expected.chat_template_sha256)
        throw std::runtime_error("steering: served chat template does not match expected identity");
    out.profile = load_profile(directory, expected);
    check_layers(out.profile, layers);
    out.config = BootConfig::from_profile(out.profile);
    if (!force.empty()) {
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
