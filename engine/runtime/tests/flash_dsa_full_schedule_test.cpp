#include <cassert>
#include <cstdio>
#include <initializer_list>
#include "../flash_dsa_full_schedule.h"

int main() {
    using namespace flash_dsa_full_schedule;
    for (int rows = 1; rows <= 32768; ++rows) {
        int next = 0;
        for (int y = 0; y < grid_rows(rows); ++y) {
            assert(first_row(y) == next);
            const int n = active_rows(rows, y);
            assert(n >= 1 && n <= rows_per_block);
            next += n;
            for (int k : {1536, 4096}) {
                assert(fragment_words(y, k) == static_cast<std::size_t>(first_row(y) / 8) * (k / 16) * 64);
                assert(fragment_words(y, k) + static_cast<std::size_t>((n + 7) / 8) * (k / 16) * 64 <= full_fragment_words(rows, k));
            }
            for (int ld : {32, 128, 576, 1536, 4096})
                assert(output_elements(y, ld) == static_cast<std::size_t>(next - n) * ld);
        }
        assert(next == rows);
    }
    std::puts("FLASH_DSA_FULL_SCHEDULE_CPU_PASS rows=1..32768");
}
