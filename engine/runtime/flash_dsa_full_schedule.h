#pragma once
#include <cstddef>
#if defined(__CUDACC__)
#define FLASH_DSA_HD __host__ __device__
#else
#define FLASH_DSA_HD
#endif

// The opt-in DSA input path changes only the launch schedule. A grid row owns
// exactly the same 32 token rows as one launch of the existing NT=4 GEMV.
namespace flash_dsa_full_schedule {
constexpr int rows_per_block = 32;
FLASH_DSA_HD constexpr int grid_rows(int rows) { return (rows + rows_per_block - 1) / rows_per_block; }
FLASH_DSA_HD constexpr int first_row(int grid_y) { return grid_y * rows_per_block; }
FLASH_DSA_HD constexpr int active_rows(int rows, int grid_y) {
    const int left = rows - first_row(grid_y);
    return left < rows_per_block ? left : rows_per_block;
}
FLASH_DSA_HD constexpr std::size_t fragment_words(int grid_y, int k) {
    // Four 8-row fragment tables, each K/16 * 64 words.
    return static_cast<std::size_t>(first_row(grid_y) / 8) * (k / 16) * 64;
}
FLASH_DSA_HD constexpr std::size_t output_elements(int grid_y, int ld) {
    return static_cast<std::size_t>(first_row(grid_y)) * ld;
}
FLASH_DSA_HD constexpr std::size_t full_fragment_words(int rows, int k) {
    return static_cast<std::size_t>((rows + 7) / 8) * (k / 16) * 64;
}
} // namespace flash_dsa_full_schedule
#undef FLASH_DSA_HD
