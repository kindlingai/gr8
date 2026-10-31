// Host-only: g++ -std=c++17 -O2 -I engine engine/tests/test_ds41_prefill_rows.cpp -o /tmp/test_ds41_prefill_rows && /tmp/test_ds41_prefill_rows
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include "runtime/ds41_prefill_rows.h"

int main() {
    static_assert(g6mx::MX == 8 || g6mx::MX == 32, "test both default and DEC_S2");
    for (int rows : {-1, 0, g6mx::MX + 1, 33}) assert(!ds41_prepare_rows_supported(rows));
    for (int rows = 1; rows <= g6mx::MX; ++rows) assert(ds41_prepare_rows_supported(rows));
    if (g6mx::MX == 32) for (int rows : {9, 16, 24, 32}) assert(ds41_prepare_rows_supported(rows));
    else assert(!ds41_prepare_rows_supported(9));
    std::printf("DS41 prefill-row contract PASS MX=%d\n", g6mx::MX);
}
