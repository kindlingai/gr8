// Host analogue of the CUDA mover for integration tests: the same descriptor and real block-store FFI.
#pragma once
#include "s2/kv_state_desc.h"
namespace g6s2 {
class KvStateTiers {
    KvStateDesc d_;
    std::vector<g6kv_tier*> stores_;
public:
    explicit KvStateTiers(KvStateDesc d) : d_(std::move(d)) {}
    ~KvStateTiers() { for (auto* t : stores_) g6kv_tier_free(t); }
    const KvStateDesc& desc() const { return d_; }
    size_t size() const { return stores_.size(); }
    uint64_t add(const std::string& spec, const std::string& name, uint64_t bytes) {
        g6kv_tier* t = nullptr;
        auto n = bytes / d_.slot_bytes();
        if (g6kv_tier_open(spec.c_str(), name.c_str(), d_.raw_block_bytes(), n, &t))
            throw std::runtime_error(g6kv_tier_error());
        stores_.push_back(t);
        return n;
    }
    uint64_t add_group(const std::string&, const std::string&, const std::string&, uint32_t, uint32_t,
                       uint64_t, uint64_t, uint32_t) { throw std::runtime_error("group test needs a node"); }
    std::string describe(size_t i) const {
        char text[256]; g6kv_tier_describe(stores_.at(i), text, sizeof(text)); return text;
    }
    void run(const std::vector<g6kv_io>& io, cudaStream_t) {
        std::vector<uint8_t> block(d_.slot_bytes());
        for (auto x : io) {
            void* p = block.data();
            if (x.kind != 0 && g6kv_tier_read_many(stores_.at(x.tier), 1, &x.slot, &p))
                throw std::runtime_error(g6kv_tier_error());
            if (x.kind == 0) {
                if (x.part == 0) d_.pack_page_host(x.page, block.data());
                else d_.pack_state_block_host(x.page, x.sub, block.data());
            }
            if (x.kind == 2) {
                if (x.part == 0) d_.unpack_page_host(x.page, block.data());
                else d_.unpack_state_block_host(x.page, x.sub, block.data());
            } else {
                const auto tier = x.kind == 1 ? x.to : x.tier;
                const auto slot = x.kind == 1 ? x.to_slot : x.slot;
                if (g6kv_tier_write_many(stores_.at(tier), 1, &slot, &p))
                    throw std::runtime_error(g6kv_tier_error());
            }
        }
    }
};
} // namespace g6s2
