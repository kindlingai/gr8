// A text knob (DS41_TEXT(knob::name())) as a C string for the DS41 parsers that take one (nullptr when unset). The value lives for the process.
#pragma once
#include "../knobs-rs/gr8_knobs.h"

#define DS41_TEXT(expr) ([]() -> const char* {                                                    \
    static const std::optional<std::string> v = expr;                                    \
    return v ? v->c_str() : nullptr;                                                              \
}())
