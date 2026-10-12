// Materialize one verified immutable profile before any layer binding or graph capture.
#pragma once
#include "rank.cuh"
#include "steering_boot.h"

namespace g6rt {
inline void materialize_steering(Model& m, const steering::Profile& p, cudaStream_t st) {
    if (m.nl != int(m.layers.size()) || m.nl <= 0 || !m.steering_scale || !m.steering_dir.empty())
        throw std::runtime_error("steering: materialization requires a fresh built model");
    if (p.hidden_size != HID || p.layers.empty() || p.alphas.size() != p.layers.size() ||
        p.directions.size() != p.layers.size() * HID)
        throw std::runtime_error("steering: invalid profile materialization shape");
    std::vector<int> ids;
    for (const auto& layer : m.layers) {
        ids.push_back(layer.l);
        for (bool built : layer.built)
            if (built) throw std::runtime_error("steering: activate before layer binding");
    }
    steering::check_layers(p, ids);
    float* device = dmalloc<float>(p.directions.size());
    G6RT_CK(cudaMemcpyAsync(device, p.directions.data(), p.directions.size() * sizeof(float), cudaMemcpyHostToDevice, st));
    m.steering_dir.assign(m.nl, nullptr);
    m.steering_alpha.assign(m.nl, 0.f);
    m.steering_scope = p.scope;
    G6RT_CK(cudaMemsetAsync(m.steering_scale, 0, MX * sizeof(float), st));
    if (p.scope == steering::Scope::Shared) {
        m.steering_part = dmalloc<float>(size_t(MX) * 48);
        m.steering_count = dmalloc<unsigned>(MX);
        G6RT_CK(cudaMemsetAsync(m.steering_count, 0, MX * sizeof(unsigned), st));
    }
    for (size_t k = 0; k < p.layers.size(); ++k) {
        const size_t i = size_t(std::find(ids.begin(), ids.end(), int(p.layers[k])) - ids.begin());
        m.steering_dir[i] = device + k * HID;
        m.steering_alpha[i] = p.alphas[k];
    }
}
} // namespace g6rt
