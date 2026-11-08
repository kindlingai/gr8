// Diagnostic build switch. Normal builds compile out every engine hook.
// A dedicated capture artifact may set this to 1 in its own recorded commit,
// or pass -DGR8_ATTN_SWEEP_CAPTURE=1 to every affected translation unit.
#pragma once
#ifndef GR8_ATTN_SWEEP_CAPTURE
#define GR8_ATTN_SWEEP_CAPTURE 0
#endif
// Empty = disabled even in a capture-capable build. Dedicated artifact only:
// set to an absolute path writable/readable inside that deployment container.
#ifndef GR8_ATTN_SWEEP_CAPTURE_CONFIG_PATH
#define GR8_ATTN_SWEEP_CAPTURE_CONFIG_PATH ""
#endif
