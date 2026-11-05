// Does a prompt run as MAIN decode steps (true) or prefill (false), on a model whose prompts decode unless prefill_min
// admits them (G6_CAP_PROMPT_DECODE, module_info.h)? Prompt logprobs (plp) score each prompt row against the next prompt
// token, which only a decode step produces; steering scales target MAIN rows. Both stay on the decode path.
#pragma once
#include <cstddef>
#include <cstdint>

namespace g6s2 {
inline bool ds41_decode_prompt(uint32_t prefill_min, size_t prompt_tokens, int plp, bool steering) {
    return prefill_min == 0 || prompt_tokens < prefill_min || plp >= 0 || steering;
}
}   // namespace g6s2
