#!/bin/bash
# nvcc_cache.sh: a content-addressed cache in front of nvcc or g++ (#32). build_m4.sh and scripts/build.sh run their
# compiles through it, so a rebuild after an edit compiles only the translation units the edit reaches.
#
#   nvcc_cache.sh <compiler> <args..>      as `<compiler> <args..>`, with the output taken from the cache when it can be
#
# The key is the SHA-256 of: the compiler's path and `--version`, every argument except the -o path, every header and
# source the compile reads (the compiler's own -M list for each source argument, with each file's contents), and the
# contents of every other input file named on the command line (objects, archives). Equal keys mean the compiler saw
# the same bytes with the same flags, so the cached output is the one it would write. A hit copies the output into
# place; a miss runs the compiler and stores what it wrote. Runs that do not name exactly one -o output, or that read
# from stdin, run the compiler as they are.
# Not in the key: libraries given with -l (only their names are; the toolkit's are covered by --version) and the
# environment, so a rebuild against another libibverbs or a changed CPATH needs G6_NVCC_CACHE=0 or an empty cache.
#
#   G6_NVCC_CACHE=0          off: every call runs the compiler (build_m4.sh then does not call this script at all)
#   G6_NVCC_CACHE_DIR=DIR    where outputs are kept (default ~/.cache/gr8-nvcc). Delete it to empty the cache.
#   G6_NVCC_CACHE_LOG=FILE   one line per call: hit|miss|pass <output>; build_m4.sh prints the counts
set -o pipefail
CC=$1; shift
DIR=${G6_NVCC_CACHE_DIR:-${HOME:-/tmp}/.cache/gr8-nvcc}
log() { [ -n "${G6_NVCC_CACHE_LOG:-}" ] && echo "$1 $2" >> "$G6_NVCC_CACHE_LOG"; return 0; }
pass() { log pass "${OUTF:--}"; exec "$CC" "$@"; }
[ "${G6_NVCC_CACHE:-1}" = 0 ] && exec "$CC" "$@"

OUTF=""; NOUT=0; ARGS=(); SRCS=(); INS=()
args=("$@"); i=0
while [ $i -lt ${#args[@]} ]; do
  a=${args[$i]}
  case $a in
    -o) i=$((i + 1)); OUTF=${args[$i]}; NOUT=$((NOUT + 1)) ;;
    -o*) OUTF=${a#-o}; NOUT=$((NOUT + 1)) ;;
    -) pass "$@" ;;
    -M|-MM|-MD|-MMD|-E|-M?*|--preprocess|--generate-dependencies) pass "$@" ;;
    *.cu|*.cpp|*.cc|*.cxx|*.c) ARGS+=("$a"); SRCS+=("$a") ;;
    -*) ARGS+=("$a") ;;
    *) ARGS+=("$a"); [ -f "$a" ] && INS+=("$a") ;;
  esac
  i=$((i + 1))
done
[ "$NOUT" = 1 ] || pass "$@"

# the flags a -M run takes: every argument but the sources, objects, -c and the output
MFLAGS=()
for a in "${ARGS[@]}"; do
  case $a in -c) ;; *.cu|*.cpp|*.cc|*.cxx|*.c) ;; *) [ -f "$a" ] || MFLAGS+=("$a") ;; esac
done
deps() {   # deps <source>: the files the compiler reads for it, one per line
  "$CC" "${MFLAGS[@]}" -M "$1" 2> /dev/null | sed -e 's/\\$//' -e 's/^[^:]*://' | tr ' ' '\n' | sed '/^$/d'
}
KEY=$(
  {
    command -v "$CC"; "$CC" --version 2>&1
    printf 'arg\0%s\n' "${ARGS[@]}"
    for s in "${SRCS[@]}"; do
      d=$(deps "$s") || exit 1
      [ -n "$d" ] || exit 1
      echo "deps of $s"
      echo "$d" | sort -u | while read -r f; do printf '%s ' "$f"; sha256sum < "$f" || exit 1; done || exit 1
    done
    for f in "${INS[@]}"; do printf 'in %s ' "$f"; sha256sum < "$f" || exit 1; done
  } | sha256sum | cut -c1-64
) || pass "$@"
[ ${#KEY} = 64 ] || pass "$@"
C=$DIR/${KEY:0:2}/$KEY
if [ -f "$C" ]; then
  # copy, then rename: a reader never sees half a file, and the cache entry is not tied to the output's inode
  if cp "$C" "$OUTF.nc$$" 2> /dev/null && chmod --reference="$C" "$OUTF.nc$$" && mv -f "$OUTF.nc$$" "$OUTF"; then
    log hit "$OUTF"; exit 0
  fi
  rm -f "$OUTF.nc$$"
fi
"$CC" "$@" || exit $?
mkdir -p "$DIR/${KEY:0:2}" && cp "$OUTF" "$C.tmp$$" && mv -f "$C.tmp$$" "$C" || rm -f "$C.tmp$$"
log miss "$OUTF"
exit 0
