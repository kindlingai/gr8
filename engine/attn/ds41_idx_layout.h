// Physical ICA page stride for DS41 index-K: 64 code bytes + 4 scale bytes per row.
// The pipe-B reader loads a uint4 from the code plane, so every S2 slot's page
// base must remain 16-byte aligned. Round the row capacity, not the row size:
// both the writer and every reader receive the same rows_per_page value.
#pragma once

namespace ds41i {
constexpr int ICA_ROW_BYTES = 68;
constexpr int ica_aligned_rows(int nominal_rows) { return (nominal_rows + 3) & ~3; }
constexpr int ica_page_bytes(int rows_per_page) { return rows_per_page * ICA_ROW_BYTES; }
}
