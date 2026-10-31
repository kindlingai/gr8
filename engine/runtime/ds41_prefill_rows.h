// Host-testable DS41 main-step row contract. W-row prefill will have a separate budget.
#pragma once
#include "../g6mx.h"

constexpr bool ds41_prepare_rows_supported(int rows) {
    return rows >= 1 && rows <= ::g6mx::MX;
}
