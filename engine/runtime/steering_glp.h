// Native GLP (GGUF Layer Projection) control-vector input for steering.
// Reads a GLP vector file and validates it into the same Profile the
// schema-v1 directory loader produces, through the same validation path
// (profile_from_parts): one identity, digest and norm semantics, whatever
// container the directions arrive in.
//
// Reader conformance follows the GLP format contract (weightless,
// https://weightle.ss/ , https://github.com/msuiche/weightless,
// spec: https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md):
// glp.mode must be "project" (absent means "add" per the
// contract and is fatal here), glp.spec_version 1-3, direction.0 rejected,
// uniform F32 widths, glp.content_sha256 re-verified over the tensor bytes.
// The hook gate is the gr8 apply site: the reference operator projects the
// complete MLP writer output (scope full), so only hooks naming that site
// family are accepted -- residual_stream_post_layer (which materialises at
// the combined MLP output on this mhc architecture, per the serving
// adapters) and ffn_out_pre_residual. attn_out_pre_residual and the
// spec-3 hc fold hooks name different tensors and are refused, never
// re-sited silently.
#pragma once

#include "steering_profile.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace g6rt::steering {

// ---------------------------------------------------------------- GGUF v3
// Minimal bounded reader: metadata scalars/strings and F32 tensor payloads,
// which is all a control vector carries. Everything is range-checked before
// allocation or dereference.

struct GgufTensor {
    std::vector<uint64_t> dims;
    uint32_t type = 0;   // 0 = F32
    uint64_t offset = 0; // relative to the data region
};

struct GgufFile {
    std::map<std::string, std::string> strings;
    std::map<std::string, uint64_t> uints;
    std::map<std::string, double> floats;
    std::map<std::string, bool> bools;
    std::map<std::string, GgufTensor> tensors;
    size_t data_base = 0;
};

inline uint32_t gguf_u32(const std::string& b, size_t& at) {
    if (at + 4 > b.size()) throw std::runtime_error("glp: truncated file");
    const auto* p = reinterpret_cast<const uint8_t*>(b.data()) + at;
    at += 4;
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

inline uint64_t gguf_u64(const std::string& b, size_t& at) {
    const uint64_t lo = gguf_u32(b, at), hi = gguf_u32(b, at);
    return lo | hi << 32;
}

inline std::string gguf_str(const std::string& b, size_t& at) {
    const uint64_t n = gguf_u64(b, at);
    if (n > (1u << 26) || at + n > b.size()) throw std::runtime_error("glp: truncated file");
    std::string out = b.substr(at, size_t(n));
    at += size_t(n);
    return out;
}

inline GgufFile parse_gguf(const std::string& bytes) {
    if (bytes.size() < 24 || std::memcmp(bytes.data(), "GGUF", 4) != 0)
        throw std::runtime_error("glp: not a GGUF file");
    size_t at = 4;
    const uint32_t version = gguf_u32(bytes, at);
    if (version != 3) throw std::runtime_error("glp: GGUF version must be 3");
    const uint64_t n_tensors = gguf_u64(bytes, at), n_kv = gguf_u64(bytes, at);
    if (n_tensors > 4096 || n_kv > 4096) throw std::runtime_error("glp: unreasonable GGUF counts");
    GgufFile g;
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = gguf_str(bytes, at);
        const uint32_t type = gguf_u32(bytes, at);
        switch (type) {
            case 0: case 1: case 7: { // uint8 / int8 / bool
                if (at + 1 > bytes.size()) throw std::runtime_error("glp: truncated file");
                const auto v = uint8_t(bytes[at++]);
                if (type == 7) g.bools[key] = v != 0; else g.uints[key] = v;
                break;
            }
            case 2: case 3: { // uint16 / int16
                if (at + 2 > bytes.size()) throw std::runtime_error("glp: truncated file");
                g.uints[key] = uint8_t(bytes[at]) | uint16_t(uint8_t(bytes[at + 1])) << 8;
                at += 2;
                break;
            }
            case 4: case 5: g.uints[key] = gguf_u32(bytes, at); break;
            case 6: { // float32
                const uint32_t bits = gguf_u32(bytes, at);
                g.floats[key] = double(bits_float(bits));
                break;
            }
            case 8: g.strings[key] = gguf_str(bytes, at); break;
            case 10: g.uints[key] = gguf_u64(bytes, at); break;
            case 11: (void)gguf_u64(bytes, at); break;
            case 12: { // float64
                const uint64_t bits = gguf_u64(bytes, at);
                double v;
                std::memcpy(&v, &bits, 8);
                g.floats[key] = v;
                break;
            }
            case 9: { // array: skipped by size, control vectors do not use them
                const uint32_t elem = gguf_u32(bytes, at);
                const uint64_t count = gguf_u64(bytes, at);
                uint64_t elem_size = 0;
                switch (elem) {
                    case 0: case 1: case 7: elem_size = 1; break;
                    case 2: case 3: elem_size = 2; break;
                    case 4: case 5: case 6: elem_size = 4; break;
                    case 10: case 11: case 12: elem_size = 8; break;
                    case 8: // string array: skip element by element
                        for (uint64_t s = 0; s < count; ++s) (void)gguf_str(bytes, at);
                        break;
                    default: throw std::runtime_error("glp: unknown GGUF array element type");
                }
                if (elem_size) {
                    if (count > (uint64_t(1) << 32) || at + count * elem_size > bytes.size())
                        throw std::runtime_error("glp: truncated file");
                    at += size_t(count * elem_size);
                }
                break;
            }
            default: throw std::runtime_error("glp: unknown GGUF metadata type");
        }
    }
    uint64_t alignment = 32;
    if (const auto it = g.uints.find("general.alignment"); it != g.uints.end()) {
        alignment = it->second;
        if (!alignment || alignment > 4096 || (alignment & (alignment - 1)))
            throw std::runtime_error("glp: invalid general.alignment");
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_str(bytes, at);
        const uint32_t n_dims = gguf_u32(bytes, at);
        if (n_dims > 4) throw std::runtime_error("glp: tensor rank above 4");
        GgufTensor t;
        for (uint32_t d = 0; d < n_dims; ++d) t.dims.push_back(gguf_u64(bytes, at));
        t.type = gguf_u32(bytes, at);
        t.offset = gguf_u64(bytes, at);
        if (!g.tensors.emplace(name, std::move(t)).second)
            throw std::runtime_error("glp: duplicate tensor name");
    }
    g.data_base = (at + alignment - 1) / alignment * alignment;
    if (g.data_base > bytes.size()) throw std::runtime_error("glp: truncated file");
    return g;
}

// ---------------------------------------------------------------- GLP gates

inline std::string glp_meta(const GgufFile& g, const std::string& key, bool required = true) {
    if (const auto it = g.strings.find(key); it != g.strings.end()) return it->second;
    if (required) throw std::runtime_error("glp: missing " + key);
    return {};
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            default:
                if (uint8_t(c) < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof esc, "\\u%04x", c);
                    out += esc;
                } else out += c;
        }
    }
    return out;
}

inline std::string alpha_json(float f) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", double(f));
    std::string out = buf;
    if (out.find_first_not_of("+-0123456789") == std::string::npos) out += ".0";
    return out;
}

// Load a GLP control vector as a steering profile. `expected` is the served
// model identity exactly as for directory profiles; the file's own base
// revision must match it (GLP vectors are checkpoint-pinned).
inline Profile load_profile_gguf(const std::string& path, const ExpectedModel& expected) {
    // Control vectors are small; cap the whole file at the schema's payload
    // cap plus header slack, before any allocation in the parser.
    const std::string bytes = read_file(path, DIRECTIONS_MAX_BYTES + (1u << 20));
    const GgufFile g = parse_gguf(bytes);

    // --- conformance gates ------------------------------------------------
    const std::string mode = glp_meta(g, "glp.mode");
    if (mode != "project")
        throw std::runtime_error("glp: glp.mode must be 'project' (absent means 'add' per the GLP contract)");
    uint64_t spec_version = 1;
    if (const auto it = g.uints.find("glp.spec_version"); it != g.uints.end()) spec_version = it->second;
    if (spec_version < 1 || spec_version > 3)
        throw std::runtime_error("glp: glp.spec_version not implemented (1-3 are)");
    const std::string hook = glp_meta(g, "glp.hook_point");
    if (hook != "residual_stream_post_layer" && hook != "ffn_out_pre_residual")
        throw std::runtime_error("glp: hook_point '" + hook + "' is not this engine's apply site "
                                 "(the complete MLP writer output); refusing to re-site the vector");
    if (const auto it = g.uints.find("glp.rank"); it != g.uints.end() && it->second != 1)
        throw std::runtime_error("glp: rank > 1 needs a multi-direction operator this engine does not have");
    if (g.strings.count("glp.dir_scales"))
        throw std::runtime_error("glp: dir_scales is rank-k machinery; refusing a rank-mismatched file");

    // Base revision pin: a GLP vector is only defined on its own checkpoint.
    std::string base_revision = glp_meta(g, "glp.base_revision", false);
    if (const auto it = g.strings.find("general.base_model.0.version"); it != g.strings.end()) {
        if (!base_revision.empty() && base_revision != it->second)
            throw std::runtime_error("glp: conflicting base revisions in metadata");
        base_revision = it->second;
    }
    if (base_revision.empty()) throw std::runtime_error("glp: no base model revision pin");
    if (base_revision != expected.revision)
        throw std::runtime_error("glp: vector is pinned to base revision " + base_revision +
                                 ", expected " + expected.revision);

    // --- directions -------------------------------------------------------
    struct Row { uint32_t layer; const GgufTensor* t; };
    std::vector<Row> rows;
    for (const auto& [name, t] : g.tensors) {
        if (name.rfind("direction.", 0) != 0)
            throw std::runtime_error("glp: unexpected tensor " + name);
        const char* suffix = name.c_str() + 10;
        char* end = nullptr;
        const unsigned long layer = std::strtoul(suffix, &end, 10);
        if (end == suffix || *end || layer == 0 || layer > UINT32_MAX)
            throw std::runtime_error("glp: invalid tensor name " + name + " (direction.0 is invalid per contract)");
        if (t.type != 0 || t.dims.size() != 1)
            throw std::runtime_error("glp: tensor " + name + " must be 1-D F32");
        rows.push_back({uint32_t(layer), &t});
    }
    if (rows.empty()) throw std::runtime_error("glp: no direction tensors");
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.layer < b.layer; });
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i && rows[i].layer == rows[i - 1].layer) throw std::runtime_error("glp: duplicate layer");
        if (rows[i].layer >= expected.layer_count) throw std::runtime_error("glp: layer past model depth");
        if (rows[i].t->dims[0] != expected.hidden_size || rows[i].t->dims[0] > (1u << 22))
            throw std::runtime_error("glp: direction width mismatch");
    }

    // Payload in ascending layer order (also the content_sha256 order).
    const size_t width = expected.hidden_size;
    std::string payload(rows.size() * width * 4, '\0');
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint64_t at = g.data_base + rows[i].t->offset;
        if (rows[i].t->offset > bytes.size() || at + width * 4 > bytes.size())
            throw std::runtime_error("glp: tensor payload out of range");
        std::memcpy(payload.data() + i * width * 4, bytes.data() + at, width * 4);
    }
    const auto content = glp_meta(g, "glp.content_sha256");
    if (sha256(payload.data(), payload.size()) != parse_digest(content, "glp.content_sha256"))
        throw std::runtime_error("glp: content_sha256 does not cover the shipped tensor bytes");

    // --- effective alphas ---------------------------------------------------
    const auto alpha_it = g.floats.find("glp.alpha_default");
    if (alpha_it == g.floats.end()) throw std::runtime_error("glp: missing glp.alpha_default");
    const double alpha_default = alpha_it->second;
    if (!std::isfinite(alpha_default) || alpha_default < 0.0 || alpha_default > 4.0)
        throw std::runtime_error("glp: alpha_default outside this engine's [0, 4] band");
    std::map<uint32_t, double> layer_scales;
    if (const auto it = g.strings.find("glp.layer_scales"); it != g.strings.end()) {
        for (const auto& entry : [&]() {
                std::vector<std::string> out; size_t at = 0;
                for (size_t c = it->second.find(',');; c = it->second.find(',', at)) {
                    out.push_back(it->second.substr(at, c == std::string::npos ? c : c - at));
                    if (c == std::string::npos) break;
                    at = c + 1;
                }
                return out; }()) {
            const auto colon = entry.find(':');
            if (colon == std::string::npos) throw std::runtime_error("glp: malformed layer_scales");
            const uint32_t layer = uint32_t(std::strtoul(entry.c_str(), nullptr, 10));
            const double m = std::strtod(entry.c_str() + colon + 1, nullptr);
            layer_scales[layer] = m;
        }
    }
    std::vector<float> alphas;
    for (const auto& r : rows) {
        double a = alpha_default;
        if (const auto it = layer_scales.find(r.layer); it != layer_scales.end()) a *= it->second;
        if (!std::isfinite(a) || a < 0.0 || a > 4.0)
            throw std::runtime_error("glp: effective alpha outside this engine's [0, 4] band");
        alphas.push_back(float(a));
    }

    // --- synthesize the schema-v1 manifest --------------------------------
    std::string name = glp_meta(g, "general.name", false);
    if (!valid_profile_name(name)) name = "glp-import";
    std::string generator = glp_meta(g, "glp.generator", false);
    if (generator.empty()) generator = "unknown";
    const std::string content_l = content;
    const std::string file_sha = digest_hex(sha256(bytes.data(), bytes.size()));

    std::string m = "{\"schema_version\":1,\"profile_name\":\"" + json_escape(name) + "\",\"model\":{";
    m += "\"name\":\"" + json_escape(expected.name) + "\",";
    m += "\"revision\":\"" + json_escape(expected.revision) + "\",";
    m += "\"weight_set\":\"" + json_escape(expected.weight_set) + "\",";
    m += "\"tokenizer_revision\":\"" + json_escape(expected.tokenizer_revision) + "\",";
    m += "\"chat_template_sha256\":\"" + json_escape(expected.chat_template_sha256) + "\",";
    m += "\"hidden_size\":" + std::to_string(expected.hidden_size) + ",";
    m += "\"layer_count\":" + std::to_string(expected.layer_count) + "},";
    m += "\"scope\":\"full\",\"layers\":[";
    for (size_t i = 0; i < rows.size(); ++i) m += (i ? "," : "") + std::to_string(rows[i].layer);
    m += "],\"alphas\":[";
    for (size_t i = 0; i < alphas.size(); ++i) m += (i ? "," : "") + alpha_json(alphas[i]);
    m += "],\"directions\":{\"file\":\"directions.f32\",\"dtype\":\"float32-le\",\"byte_count\":" +
         std::to_string(payload.size()) + ",\"sha256\":\"" + digest_hex(sha256(payload.data(), payload.size())) + "\"},";
    // GLP files carry no dataset manifests; the content hash (which commits
    // to the exact direction bytes) stands in for all three splits, and the
    // GLP provenance keys stay visible in extraction_tool/extraction_commit.
    m += "\"provenance\":{\"extraction_tool\":\"glp:" + json_escape(generator) + "\",";
    m += "\"extraction_commit\":\"" + json_escape(content_l) + "\",";
    m += "\"dataset_manifests\":{\"train\":\"" + content_l + "\",\"validation\":\"" + content_l +
         "\",\"final\":\"" + content_l + "\"},";
    m += "\"fit_sha256\":\"" + file_sha + "\",\"family\":\"per-layer\"}}";

    return profile_from_parts(m, payload, expected);
}

// Directory or GLP GGUF, decided by magic bytes: one flag value, both
// containers. Anything that is not a GGUF goes through the directory loader.
inline Profile load_profile_any(const std::string& path, const ExpectedModel& expected) {
    std::ifstream f(path, std::ios::binary);
    char magic[4] = {};
    f.read(magic, sizeof magic);
    if (f.gcount() == std::streamsize(sizeof magic) && std::memcmp(magic, "GGUF", 4) == 0)
        return load_profile_gguf(path, expected);
    return load_profile(path, expected);
}

} // namespace g6rt::steering
