// ibv_mtu wire values: 1..5 represent 256..4096 bytes; reject invalid bootstrap metadata.
#pragma once
#include <cstdint>
namespace g6 {
constexpr bool valid_fabric_mtu(uint32_t mtu) { return mtu >= 1 && mtu <= 5; }
constexpr uint32_t fabric_mtu_min(uint32_t local, uint32_t peer) { return local < peer ? local : peer; }
}
