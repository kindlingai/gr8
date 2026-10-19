#!/bin/bash
# Boot the TP6 engine on six Sparks and serve an OpenAI-compatible API on the head: runs scripts/launch_s2.sh (up to
# SESS requests at once; MTP, and with DFLASH=1 the DFlash2 drafter, its body replicated on every rank). KNOBS.md lists the knobs.
set -eu
[ "${PLAIN:-0}" = 1 ] && { echo "PLAIN=1 (the one-request loop) is retired at TP=6: unset it; TP=6 serves through launch_s2.sh"; exit 2; }
exec bash "$(dirname "$0")/launch_s2.sh"
