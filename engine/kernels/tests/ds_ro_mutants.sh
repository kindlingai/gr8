#!/bin/bash
# Mutation check of the read-once host gate (ds_ro_host_test.cpp): each mutant edits a copy of the headers (sed), the gate must then fail (non-zero exit).
#   bash ds_ro_mutants.sh [workdir]        (run from anywhere; default workdir /tmp/ds_ro_mut)
set -u
K="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-/tmp/ds_ro_mut}"
mkdir -p "$W"
declare -a NAMES=(
  "m1 group step NTG-1 (ro_tile)|ds_ro.h|return tile0 + g \* ntg + t;|return tile0 + g * (ntg - 1) + t;"
  "m2 drop the last partial group (ro_ngroups)|ds_ro.h|return (nt + ntg - 1) / ntg; }|return nt / ntg; }"
  "m3 fc2 output-tile overlap (ro_fc2_work, warps-1)|ds_ro.h|ot = (block % ups) \* warps + warp;|ot = (block % ups) * (warps - 1) + warp;"
  "m4 wrong tile0 offset (ro_tile +1)|ds_ro.h|return tile0 + g \* ntg + t;|return tile0 + 1 + g * ntg + t;"
  "m5 rank word off by one (mt_rank_w)|ds_plan_mt.h|for (int i = 0; i < (m >> 5); ++i) r += mt_popc(em\[i\]);|for (int i = 0; i < (m >> 5) - 1 + (m < 32); ++i) r += mt_popc(em[i]);"
  "m6 dead-column fill start +1 (ro_last_live)|ds_ro.h|return n - 8 \* (mt_ntiles(n) - 1); }|return n - 8 * (mt_ntiles(n) - 1) + 1; }"
  "m7 fc1 gate/up record swap (ro_fc1_rec)|ds_ro.h|\* 2 + up;|* 2 + (1 - up);"
  "m8 group tile count without remainder (ro_group_nt)|ds_ro.h|return left < ntg ? left : ntg; }|return ntg; }"
  "m9 fc1 chunk map duplicates (ro_fc1_work)|ds_ro.h|chunk = (block % ups) \* cw + warp / ks;|chunk = (block % ups) * cw + warp % cw;"
  "m10 tile count (mt_ntiles (n+8)>>3)|ds_plan_mt.h|return (n + 7) >> 3; }|return (n + 8) >> 3; }"
  "m12 counter counts experts with >= 2 routes (elog_stats)|ds_ro.h|distinct += cnt\[e\] > 0;|distinct += cnt[e] > 1;"
  "m11 two experts share weight records (ro_fc1_rec expert\&~1)|ds_ro.h|(static_cast<size_t>(expert) \* nch + chunk)|(static_cast<size_t>(expert \& ~1) * nch + chunk)"
  "m13 ro_pos_slot ignores the order array|ds_ro.h|return order ? order\[pos\] : pos; }|return pos; }"
  "m14 class tie goes to the light expert (ro_ord_before)|ds_ro.h|if (ca != cb) return ca < cb;|if (ca != cb) return ca > cb;"
  "m15 light weight nt instead of 1 (ro_ord_key)|ds_ro.h|const int wj = hs ? nt\[j\] : 1;|const int wj = nt[j];"
  "m16 heavy threshold 6 instead of 5|ds_ro.h|constexpr int RO_ORD_HEAVY_NT = 5;|constexpr int RO_ORD_HEAVY_NT = 6;"
  "m17 heavy list ascending instead of descending (ro_ord_key)|ds_ro.h|nt\[j\] > nt\[s\]|nt[j] < nt[s]"
  "m18 heavy tie goes to the higher slot (ro_ord_key)|ds_ro.h|nt\[j\] == nt\[s\] \&\& j < s|nt[j] == nt[s] \&\& j > s"
  "m19 keys sorted descending (ro_ord_before)|ds_ro.h|if (x != y) return x < y;|if (x != y) return x > y;"
  "m20 key denominator off by one (ro_ord_key)|ds_ro.h|num = before; den = total;|num = before; den = total + 1;"
  "m21 prefetch target behind instead of ahead (ro_pf_target)|ds_ro.h|bp = block + la;|bp = block - la;"
  "m22 target one slot group past the last (ro_pf_target <=)|ds_ro.h|bp / ups < ns;|bp / ups <= ns;"
  "m23 chunk offset +16 (ro_pf_chunk)|ds_ro.h|off = static_cast<unsigned>(c) \* RO_PF_CHUNK;|off = static_cast<unsigned>(c) * RO_PF_CHUNK + 16;"
  "m24 chunk size 4096 instead of 8192 (RO_PF_CHUNK)|ds_ro.h|constexpr unsigned RO_PF_CHUNK = 8192;|constexpr unsigned RO_PF_CHUNK = 4096;"
  "m25 last chunk not clipped (ro_pf_chunk)|ds_ro.h|bytes - off < RO_PF_CHUNK ? bytes - off : RO_PF_CHUNK;|RO_PF_CHUNK;"
  "m26 fc1 range starts at chunk cgrp, not cgrp * cw (ro_fc1_pf_rec)|ds_ro.h|expert, cgrp \* cw, 0, 0|expert, cgrp, 0, 0"
  "m27 fc1 range misses the up half (ro_fc1_pf_recs)|ds_ro.h|static_cast<size_t>(cw) \* ng \* 2; }|static_cast<size_t>(cw) * ng; }"
  "m28 fc2 range one output tile short (ro_fc2_pf_recs)|ds_ro.h|static_cast<size_t>(warps) \* ng; }|static_cast<size_t>(warps - 1) * ng; }"
  "m29 prefetch for la <= 0 (ro_pf_target)|ds_ro.h|return la > 0 && bp / ups < ns;|return bp / ups < ns;"
  "m30 fc2 range starts at tile ogrp, not ogrp * warps (ro_fc2_pf_rec)|ds_ro.h|ro_fc2_rec(expert, ogrp \* warps, 0|ro_fc2_rec(expert, ogrp, 0"
  "m31 empty chunk at an exact multiple (ro_pf_chunk off > bytes)|ds_ro.h|if (off >= bytes) return false;|if (off > bytes) return false;"
)
if [ ! -x "$W/base" ]; then :; fi
mkdir -p "$W/base/tests"
cp "$K/ds_plan_mt.h" "$K/ds_ro.h" "$W/base/"
cp "$K/tests/ds_ro_host_test.cpp" "$W/base/tests/"
g++ -O2 -std=c++17 -Wall -o "$W/base/tests/ht" "$W/base/tests/ds_ro_host_test.cpp" || { echo "base build failed"; exit 2; }
"$W/base/tests/ht" > "$W/base.out" 2>&1; echo "base (unmutated): exit $? -> $(tail -1 "$W/base.out")"
caught=0; total=0
for entry in "${NAMES[@]}"; do
  IFS='|' read -r label file pat rep <<< "$entry"
  id="${label%% *}"
  d="$W/$id"; mkdir -p "$d/tests"
  cp "$K/ds_plan_mt.h" "$K/ds_ro.h" "$d/"; cp "$K/tests/ds_ro_host_test.cpp" "$d/tests/"
  sed -i "s|$pat|$rep|" "$d/$file"
  if cmp -s "$d/$file" "$K/$file"; then echo "$label: SED DID NOT CHANGE THE FILE"; total=$((total+1)); continue; fi
  total=$((total+1))
  if ! g++ -O2 -std=c++17 -o "$d/tests/ht" "$d/tests/ds_ro_host_test.cpp" 2> "$d/build.err"; then echo "$label: build error (not counted as caught)"; continue; fi
  "$d/tests/ht" > "$d/out" 2>&1; rc=$?
  if [ $rc -ne 0 ]; then caught=$((caught+1)); echo "$label: CAUGHT (exit $rc; $(grep -m1 '^FAIL' "$d/out" | cut -c1-150))"; else echo "$label: SURVIVED (exit 0)"; fi
done
echo "mutants caught: $caught / $total"
[ "$caught" -eq "$total" ]
