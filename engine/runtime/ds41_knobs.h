// A text knob (DS41_TEXT(knob::name())) as a C string for the DS41 parsers that take one (nullptr when unset). The value lives for the process.
#pragma once
#include <cstring>
#include "../knobs-rs/gr8_knobs.h"

#define DS41_TEXT(expr) ([]() -> const char* {                                                    \
    static const std::optional<std::string> v = expr;                                    \
    return v ? v->c_str() : nullptr;                                                              \
}())

// The DSpark knobs by name, from the knob table: the engine's Get for ds41::dsp::parse_knobs, which does the parsing and
// refuses a bad value. nullptr = unset (the parser's default). GR8_DS_ENGRAM is a flag, so it reads back as "0" / "1".
inline const char* ds41_dspark_knob(const char* n)
{
    if (!std::strcmp(n, "GR8_DS_MTP")) return DS41_TEXT(knob::ds_mtp());
    if (!std::strcmp(n, "GR8_DS_BLOCK")) return DS41_TEXT(knob::ds_block());
    if (!std::strcmp(n, "GR8_DS_RELAXED_ACCEPT")) return DS41_TEXT(knob::ds_relaxed_accept());
    if (!std::strcmp(n, "GR8_DS_PREFILL_KEEP")) return DS41_TEXT(knob::ds_prefill_keep());
    if (!std::strcmp(n, "GR8_DS_VERIFY_CAP")) return DS41_TEXT(knob::ds_verify_cap());
    if (!std::strcmp(n, "GR8_DS_CONF_TEMP")) return DS41_TEXT(knob::ds_conf_temp());
    if (!std::strcmp(n, "GR8_DS_ENGRAM")) return knob::ds_engram() ? "1" : "0";
    return nullptr;
}
