#!/bin/bash
# Boot the TP4 engine on four Sparks and serve an OpenAI-compatible API on the head: runs scripts/launch_s2.sh (up to
# SESS requests at once, DFlash2 for a request that runs alone). KNOBS.md lists the knobs.
set -eu
[ "${PLAIN:-0}" = 1 ] && { echo "PLAIN=1 (the one-request loop) is retired at TP=4: unset it; TP=4 serves through launch_s2.sh"; exit 2; }
exec bash "$(dirname "$0")/launch_s2.sh"
