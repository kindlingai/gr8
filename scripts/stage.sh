#!/bin/bash
# Copy the engine binary and the packs to every node (run on the head).
#   scripts/stage.sh <BIN_DIR> <PACK_DIR>    BIN_DIR has g6m4 (and lib/, if built with BUNDLE_LIBS=1; and gpuprobe, for the rank health probe);
#                                            PACK_DIR has rank{0..NR-1}.* and qo.*
# Rank r gets ONLY its own rank pack (~62 GB, ~90 GB at TP4) plus the shared qo pack (~13 GB). Packs must sit on each node's
# own disk: the launcher loads them locally (streaming a pack over ssh at boot was 2-3x slower for us).
# SHARED=1: PACK_DIR is a shared mount (NFS) at the same path on every node; each node copies its own files from it
# instead of the head pushing them over ssh.
set -eu
. "$(dirname "$0")/common.sh"
BIN=$1; P=$2
for ((r = 0; r < NRANK; r++)); do
  h=${HOST[$r]}
  on $r "mkdir -p $G6_HOME/bin $G6_HOME/packs $G6_HOME/serve $G6_HOME/out"
  rsync -a "$BIN/g6m4" "$h:$G6_HOME/bin/" &
  [ -f "$BIN/gpuprobe" ] && rsync -a "$BIN/gpuprobe" "$h:$G6_HOME/bin/" &
  if [ -d "$BIN/lib" ]; then rsync -a "$BIN/lib" "$h:$G6_HOME/bin/" & fi
  if [ "${SHARED:-0}" = 1 ]; then
    on $r "rsync -a --partial $P/rank$r.pack $P/rank$r.manifest.json $P/qo.pack $P/qo.manifest.json $G6_HOME/packs/" &
  else
    rsync -a --partial "$P/rank$r.pack" "$P/rank$r.manifest.json" "$P/qo.pack" "$P/qo.manifest.json" "$h:$G6_HOME/packs/" &
  fi
done
wait
for ((r = 0; r < NRANK; r++)); do
  echo "rank $r ${HOST[$r]}: $(on $r "md5sum $G6_HOME/bin/g6m4 | cut -c1-12 | tr -d '\n'; echo -n ' '; ls $G6_HOME/packs | tr '\n' ' '; df -h $G6_HOME | tail -1 | awk '{print \"free \" \$4}'")"
done
