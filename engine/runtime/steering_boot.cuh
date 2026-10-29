// Materialize the verified immutable profiles before any layer binding or graph capture.
#pragma once
#include "rank.cuh"
#include "steering_boot.h"

namespace g6rt {
inline void materialize_steering(Model& m, const std::vector<steering::Profile>& ps, cudaStream_t st) {
    if (m.nl != int(m.layers.size()) || m.nl <= 0 || !m.steering_scale || !m.steering_dir.empty())
        throw std::runtime_error("steering: materialization requires a fresh built model");
    const size_t np = ps.size();
    if (!np || np > steering::PROFILES_MAX) throw std::runtime_error("steering: invalid profile count");
    for (const auto& p : ps)
        if (p.hidden_size != HID || p.layers.empty() || p.alphas.size() != p.layers.size() ||
            p.directions.size() != p.layers.size() * HID || p.scope != ps[0].scope)
            throw std::runtime_error("steering: invalid profile materialization shape");
    std::vector<int> ids;
    for (const auto& layer : m.layers) {
        ids.push_back(layer.l);
        for (bool built : layer.built)
            if (built) throw std::runtime_error("steering: activate before layer binding");
    }
    for (const auto& p : ps) steering::check_layers(p, ids);
    // host tables: [nl][profiles] alphas, and each steered layer's [profiles][HID] directions (zero where a profile
    // leaves the layer out; its alpha 0 keeps those rows on the stock path)
    m.steering_profiles = int(np);
    m.steering_alphas.assign(size_t(m.nl) * np, 0.f);
    std::vector<std::vector<float>> dirs(m.nl);
    for (size_t s = 0; s < np; ++s)
        for (size_t k = 0; k < ps[s].layers.size(); ++k) {
            const size_t i = size_t(std::find(ids.begin(), ids.end(), int(ps[s].layers[k])) - ids.begin());
            if (dirs[i].empty()) dirs[i].assign(np * HID, 0.f);
            std::copy_n(ps[s].directions.data() + k * HID, HID, dirs[i].data() + s * HID);
            m.steering_alphas[i * np + s] = ps[s].alphas[k];
        }
    m.steering_dir.assign(m.nl, nullptr);
    m.steering_alpha.assign(m.nl, 0.f);
    m.steering_scope = ps[0].scope;
    G6RT_CK(cudaMemsetAsync(m.steering_scale, 0, MX * sizeof(float), st));
    G6RT_CK(cudaMemsetAsync(m.steering_slot, 0, MX * sizeof(int), st));
    if (m.steering_scope == steering::Scope::Shared) {
        m.steering_part = dmalloc<float>(size_t(MX) * 48);
        m.steering_count = dmalloc<unsigned>(MX);
        G6RT_CK(cudaMemsetAsync(m.steering_count, 0, MX * sizeof(unsigned), st));
    }
    size_t floats = 0;
    for (const auto& d : dirs) floats += d.size();
    float* device = dmalloc<float>(floats + (np > 1 ? size_t(m.nl) * np : 0));
    float* at = device;
    for (int i = 0; i < m.nl; ++i) {
        m.steering_alpha[i] = m.steering_alphas[size_t(i) * np];
        if (dirs[i].empty()) continue;
        G6RT_CK(cudaMemcpyAsync(at, dirs[i].data(), dirs[i].size() * sizeof(float), cudaMemcpyHostToDevice, st));
        m.steering_dir[i] = at;
        at += dirs[i].size();
    }
    if (np > 1) {
        G6RT_CK(cudaMemcpyAsync(at, m.steering_alphas.data(), m.steering_alphas.size() * sizeof(float),
                                cudaMemcpyHostToDevice, st));
        for (int i = 0; i < m.nl; ++i) m.steering_slot_alpha.push_back(at + size_t(i) * np);
    }
    G6RT_CK(cudaStreamSynchronize(st));   // the host tables above are the copies' sources
}
inline void materialize_steering(Model& m, const steering::Profile& p, cudaStream_t st) {
    materialize_steering(m, std::vector<steering::Profile>{p}, st);
}
} // namespace g6rt
