#pragma once
namespace g6kv4_dispatch {
inline constexpr bool qt_enable(bool tc, bool fold, bool qt_on, bool heads, int sc) {
    return !tc && !fold && qt_on && heads && (sc == 1 || sc == 2 || sc == 4);
}
inline constexpr bool fuse_rot_enable(bool fold, int fuse_rot) {
    return !fold && (fuse_rot != 0);
}
} // namespace g6kv4_dispatch
