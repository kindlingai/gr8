#!/bin/bash
# Build and run the NVMe random-read bench (host only, no GPU). Writes one file (default 24 GiB) into <scratch dir>; nothing is deleted.
#   run_nvme_bench.sh <scratch dir> [--gib N] [--secs S] [--quick]
# The binary lands in <scratch dir>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ENG="$HERE/../.."
SCR="${1:?scratch dir}"; shift || true
mkdir -p "$SCR"
g++ -std=c++17 -O2 -Wall -Wextra -pthread -DG6_MODEL_DS41=1 -I"$ENG" -o "$SCR/bench_nvme_randread" "$HERE/bench_nvme_randread.cpp"
echo "== host =="; hostname; uname -r; echo "cpus $(nproc)"; grep -E "MemTotal|MemAvailable" /proc/meminfo; echo "loadavg $(cat /proc/loadavg)"
SRC="$(df --output=source "$SCR" | tail -1)"; PK="$(lsblk -no pkname "$SRC" 2>/dev/null | head -1 || true)"; [ -n "$PK" ] || PK="$(basename "$SRC")"
df -hT "$SCR" | tail -2
lsblk -d -o NAME,MODEL,SIZE,TRAN,ROTA "/dev/$PK" 2>&1 || true
for f in device/model device/firmware_rev queue/scheduler queue/nr_requests queue/read_ahead_kb queue/max_sectors_kb; do
    echo "$PK/$f: $(cat /sys/block/$PK/$f 2>&1 | head -1)"
done
echo "== run =="
"$SCR/bench_nvme_randread" "$SCR" "$@"
