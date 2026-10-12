// Host-only opaque prefix-cache identity. Consumers define its bytes; cache code compares them exactly.
#pragma once
#include "../kvman-rs/g6kvman.h"
#include <array>

namespace g6rt {
using CacheNamespace = std::array<uint8_t, G6KV_NAMESPACE_BYTES>;
inline constexpr CacheNamespace STOCK_CACHE_NAMESPACE{};
}
