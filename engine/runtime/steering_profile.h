// Strict schema-v1 steering profile loader. Host only; callers provide the expected model identity.
// The profile digest is SHA-256 of the exact manifest bytes, which themselves commit to
// the directions tensor digest and byte count.
#pragma once

#include "steering.h"
#include "json.h"
#include "../loader/g6pack.h"
#include <cerrno>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>

namespace g6rt::steering {

inline std::array<uint8_t, 32> sha256(const void* data, size_t size) {
    g6::Sha256 hash;
    hash.update(static_cast<const uint8_t*>(data), size);
    std::array<uint8_t, 32> digest{};
    hash.final(digest.data());
    return digest;
}

struct ExpectedModel {
    std::string name, revision, weight_set, tokenizer_revision, chat_template_sha256;
    uint32_t hidden_size = 0, layer_count = 0;
};

inline std::string read_file(const std::string& path, size_t limit = std::numeric_limits<size_t>::max()) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("steering profile: cannot open " + path);
    f.seekg(0, std::ios::end);
    const auto end = f.tellg();
    if (end < 0 || uint64_t(end) > limit) throw std::runtime_error("steering profile: invalid/oversize file " + path);
    std::string out(size_t(end), '\0');
    f.seekg(0);
    if (!out.empty() && !f.read(out.data(), std::streamsize(out.size())))
        throw std::runtime_error("steering profile: short read " + path);
    return out;
}

inline std::array<uint8_t, 32> parse_digest(const std::string& s, const char* field) {
    if (!hex_digest(s)) throw std::runtime_error(std::string("steering profile: invalid ") + field);
    std::array<uint8_t, 32> out{};
    auto nibble = [](char c) { return uint8_t(c <= '9' ? c - '0' : c - 'a' + 10); };
    for (size_t i = 0; i < out.size(); ++i) out[i] = uint8_t(nibble(s[2 * i]) << 4 | nibble(s[2 * i + 1]));
    return out;
}

inline void object_unique(const g6v::Json& j, const char* where) {
    if (j.t != g6v::Json::Obj) throw std::runtime_error(std::string("steering profile: ") + where + " is not an object");
    std::set<std::string> keys;
    for (const auto& kv : j.o)
        if (!keys.insert(kv.first).second) throw std::runtime_error(std::string("steering profile: duplicate key ") + where + "." + kv.first);
}

inline void object_known(const g6v::Json& j, const char* where, std::initializer_list<const char*> allowed) {
    object_unique(j, where);
    std::set<std::string> names;
    for (const char* key : allowed) names.insert(key);
    for (const auto& kv : j.o)
        if (!names.count(kv.first)) throw std::runtime_error(std::string("steering profile: unknown key ") + where + "." + kv.first);
}

inline uint32_t u32_field(const g6v::Json& j, const char* key) {
    const int64_t v = j.at(key).i64();
    if (v < 0 || uint64_t(v) > UINT32_MAX) throw std::runtime_error(std::string("steering profile: invalid ") + key);
    return uint32_t(v);
}

inline float finite_alpha(const g6v::Json& j) {
    const double v = j.num();
    if (!std::isfinite(v) || v < 0.0 || v > double(SCALE_MAX))
        throw std::runtime_error("steering profile: alpha must be finite and in [0, 4]");
    const float f = float(v);
    if (!std::isfinite(f) || f < 0.0f || f > SCALE_MAX)
        throw std::runtime_error("steering profile: alpha is not representable in [0, 4]");
    return f;
}

inline std::array<uint8_t, 32> schedule_hash(Scope scope, const std::vector<uint32_t>& layers,
                                             const std::vector<float>& alphas) {
    g6::Sha256 h;
    const uint8_t s = uint8_t(scope); h.update(&s, 1);
    for (size_t i = 0; i < layers.size(); ++i) {
        uint8_t b[8]; const uint32_t l = layers[i], a = float_bits(alphas[i]);
        for (int k = 0; k < 4; ++k) { b[k] = uint8_t(l >> (8 * k)); b[4 + k] = uint8_t(a >> (8 * k)); }
        h.update(b, sizeof b);
    }
    std::array<uint8_t, 32> digest{};
    h.final(digest.data());
    return digest;
}

inline Profile load_profile(const std::string& directory, const ExpectedModel& expected) {
    Profile p;
    p.manifest_json = read_file(directory + "/manifest.json", MANIFEST_MAX_BYTES);
    p.manifest_digest = sha256(p.manifest_json.data(), p.manifest_json.size());
    const auto root = g6v::Json::parse(p.manifest_json);
    object_known(root, "root", {"schema_version", "profile_name", "model", "scope", "layers", "alphas",
                                "directions", "provenance"});
    p.schema_version = u32_field(root, "schema_version");
    if (p.schema_version != 1) throw std::runtime_error("steering profile: unsupported schema_version");
    p.name = root.at("profile_name").str();
    if (!valid_profile_name(p.name)) throw std::runtime_error("steering profile: invalid profile_name");

    const auto& model = root.at("model");
    object_known(model, "model", {"name", "revision", "weight_set", "tokenizer_revision", "chat_template_sha256",
                                   "hidden_size", "layer_count"});
    p.model_name = model.at("name").str();
    p.model_revision = model.at("revision").str();
    p.weight_set = model.at("weight_set").str();
    p.tokenizer_revision = model.at("tokenizer_revision").str();
    p.chat_template_sha256 = model.at("chat_template_sha256").str();
    p.hidden_size = u32_field(model, "hidden_size");
    p.layer_count = u32_field(model, "layer_count");
    if (p.model_name.empty() || p.model_revision.empty() || p.tokenizer_revision.empty() ||
        !p.hidden_size || !p.layer_count)
        throw std::runtime_error("steering profile: model identity fields and dimensions must be nonempty");
    if (!hex_digest(p.weight_set)) throw std::runtime_error("steering profile: invalid weight_set");
    if (!hex_digest(p.chat_template_sha256)) throw std::runtime_error("steering profile: invalid chat_template_sha256");
    if (p.model_name != expected.name || p.model_revision != expected.revision || p.weight_set != expected.weight_set ||
        p.tokenizer_revision != expected.tokenizer_revision || p.chat_template_sha256 != expected.chat_template_sha256 ||
        p.hidden_size != expected.hidden_size || p.layer_count != expected.layer_count)
        throw std::runtime_error("steering profile: model identity mismatch");

    const std::string scope = root.at("scope").str();
    if (scope == "shared") p.scope = Scope::Shared;
    else if (scope == "full") p.scope = Scope::Full;
    else throw std::runtime_error("steering profile: unknown scope");

    const auto& layers = root.at("layers");
    const auto& alphas = root.at("alphas");
    if (layers.t != g6v::Json::Arr || alphas.t != g6v::Json::Arr || layers.a.empty() || layers.a.size() != alphas.a.size())
        throw std::runtime_error("steering profile: inconsistent layer/alpha counts");
    for (size_t i = 0; i < layers.a.size(); ++i) {
        const int64_t l = layers.a[i].i64();
        if (l < 0 || uint64_t(l) >= p.layer_count || (i && uint64_t(l) <= p.layers.back()))
            throw std::runtime_error("steering profile: layers must be unique, ordered, and in range");
        p.layers.push_back(uint32_t(l));
        p.alphas.push_back(finite_alpha(alphas.a[i]));
    }

    const auto& directions = root.at("directions");
    object_known(directions, "directions", {"file", "dtype", "byte_count", "sha256"});
    if (directions.at("file").str() != "directions.f32" || directions.at("dtype").str() != "float32-le")
        throw std::runtime_error("steering profile: directions must be directions.f32 in float32-le");
    const uint64_t bytes = uint64_t(u32_field(directions, "byte_count"));
    const uint64_t want = uint64_t(p.layers.size()) * p.hidden_size * sizeof(float);
    if (bytes != want || want > DIRECTIONS_MAX_BYTES || want > SIZE_MAX) throw std::runtime_error("steering profile: directions byte_count mismatch");
    p.tensor_digest = parse_digest(directions.at("sha256").str(), "directions sha256");

    const auto& provenance = root.at("provenance");
    object_known(provenance, "provenance", {"extraction_tool", "extraction_commit", "dataset_manifests",
                                             "capture_manifest_sha256", "fit_sha256", "family", "direction_layer"});
    if (provenance.at("extraction_tool").str().empty() || provenance.at("extraction_commit").str().empty())
        throw std::runtime_error("steering profile: missing extraction provenance");
    const auto& datasets = provenance.at("dataset_manifests");
    object_known(datasets, "dataset_manifests", {"train", "validation", "final"});
    for (const char* split : {"train", "validation", "final"})
        (void)parse_digest(datasets.at(split).str(), split);
    for (const char* field : {"capture_manifest_sha256", "fit_sha256"})
        if (const auto* value = provenance.find(field)) (void)parse_digest(value->str(), field);
    if (const auto* family = provenance.find("family"); family && family->t != g6v::Json::Null) {
        const std::string& value = family->str();
        if (value != "global-band" && value != "per-layer")
            throw std::runtime_error("steering profile: unknown provenance.family");
    }
    if (const auto* layer = provenance.find("direction_layer")) {
        if (layer->t != g6v::Json::Null) {
            const int64_t value = layer->i64();
            if (value < 0 || uint64_t(value) >= p.layer_count)
                throw std::runtime_error("steering profile: invalid provenance.direction_layer");
        }
    }

    const std::string raw = read_file(directory + "/directions.f32", size_t(want));
    if (raw.size() != want) throw std::runtime_error("steering profile: short/trailing directions payload");
    if (sha256(raw.data(), raw.size()) != p.tensor_digest)
        throw std::runtime_error("steering profile: directions digest mismatch");
    p.directions.resize(size_t(p.layers.size()) * p.hidden_size);
    for (size_t i = 0; i < p.directions.size(); ++i) {
        const auto* q = reinterpret_cast<const uint8_t*>(raw.data()) + 4 * i;
        const uint32_t bits = uint32_t(q[0]) | uint32_t(q[1]) << 8 | uint32_t(q[2]) << 16 | uint32_t(q[3]) << 24;
        p.directions[i] = bits_float(bits);
    }
    for (size_t l = 0; l < p.layers.size(); ++l) {
        double norm2 = 0.0;
        for (uint32_t i = 0; i < p.hidden_size; ++i) {
            const float v = p.directions[l * p.hidden_size + i];
            if (!std::isfinite(v)) throw std::runtime_error("steering profile: non-finite direction");
            norm2 += double(v) * double(v);
        }
        const double norm = std::sqrt(norm2);
        if (!(norm >= 0.999 && norm <= 1.001)) throw std::runtime_error("steering profile: direction norm outside [0.999, 1.001]");
    }
    p.schedule_digest = schedule_hash(p.scope, p.layers, p.alphas);
    return p;
}

} // namespace g6rt::steering
