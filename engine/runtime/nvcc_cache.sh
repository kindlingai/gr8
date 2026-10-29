#!/bin/bash
# nvcc_cache.sh: a content-addressed cache in front of nvcc or g++ (#32). build_m4.sh and scripts/build.sh run their
# compiles through it, so a rebuild after an edit compiles only the translation units the edit reaches.
#
#   nvcc_cache.sh <compiler> <args..>      as `<compiler> <args..>`, with the output taken from the cache when it can be
#
# The key is the SHA-256 of: the compiler's path and `--version`, the host compiler's (`-ccbin`, else c++ on PATH) and
# ld's `--version`, the environment the compiler reads (CPATH, include and library paths, NVCC_PREPEND/APPEND_FLAGS),
# every argument except the -o path, every file the compile reads (the compiler's own -M list for each source, with each
# file's contents), the contents of every other input file named on the command line (objects, archives), and for each
# -l library the file the linker would take (contents for an archive; path, size and time for a shared library). Equal
# keys mean the compiler saw the same bytes with the same flags, so the cached output is the one it would write.
# A hit copies the output into place and replays the compiler's stderr (its warnings); a miss runs the compiler and
# stores what it wrote. These run the compiler as they are: no -o or two, stdin, -x, response files (@F, -optf),
# dependency and preprocess runs (-M*, -E), a -l library not found, a path with a space.
# docs/BUILD-CACHE.md has the details.
#
#   GR8_NVCC_CACHE=0          off: every call runs the compiler
#   GR8_NVCC_CACHE_DIR=DIR    where outputs are kept (default ~/.cache/gr8-nvcc). Delete it to empty the cache.
#   GR8_NVCC_CACHE_LOG=FILE   one line per call: hit|miss|pass <output>; build_m4.sh prints the counts
set -o pipefail
CC=$1; shift
[ "${GR8_NVCC_CACHE:-1}" = 0 ] && exec "$CC" "$@"
DIR=${GR8_NVCC_CACHE_DIR:-${HOME:-/tmp}/.cache/gr8-nvcc}
log() { [ -n "${GR8_NVCC_CACHE_LOG:-}" ] && echo "$1 $2" >> "$GR8_NVCC_CACHE_LOG"; return 0; }

OUTF=""; NOUT=0; ARGS=(); MFLAGS=(); SRCS=(); INS=(); LIBS=(); LDIRS=(); HOSTCC=""; PASS=""; HASC=0
args=("$@"); i=0; n=${#args[@]}
while [ $i -lt $n ]; do
  a=${args[$i]}
  case $a in
    *' '*) PASS=space ;;
    # before -o*: -optf and -odir start with -o
    -|-x|-x*|--x|--x=*|-E|--preprocess|-M*|--generate-dependencies*|--generate-nonsystem-dependencies*|@*|-optf|-optf=*|\
    --options-file*|-odir|-odir=*|--output-directory*)
      PASS=$a ;;
    -o) i=$((i + 1)); OUTF=${args[$i]:-}; NOUT=$((NOUT + 1)) ;;
    -o*) OUTF=${a#-o}; NOUT=$((NOUT + 1)) ;;
    -c) HASC=1; ARGS+=("$a") ;;
    # options whose value is the next argument: flags, never inputs
    -I|-isystem|-iquote|-idirafter|-include|-imacros|-L|-Xcompiler|-Xlinker|-Xptxas|-Xnvlink|-Xfatbin|-arch|-code|-gencode|-ccbin|--compiler-bindir)
      v=${args[$((i + 1))]:-}
      case $v in *' '*) PASS=space ;; esac
      ARGS+=("$a" "$v"); MFLAGS+=("$a" "$v")
      [ "$a" = -L ] && LDIRS+=("$v")
      case $a in -ccbin|--compiler-bindir) HOSTCC=$v ;; esac
      i=$((i + 1)) ;;
    -L*) ARGS+=("$a"); MFLAGS+=("$a"); LDIRS+=("${a#-L}") ;;
    -ccbin=*|--compiler-bindir=*) ARGS+=("$a"); MFLAGS+=("$a"); HOSTCC=${a#*=} ;;
    -lineinfo|-link|-lto|-lib) ARGS+=("$a"); MFLAGS+=("$a") ;;
    -l*) ARGS+=("$a"); LIBS+=("${a#-l}") ;;
    -*) ARGS+=("$a"); MFLAGS+=("$a") ;;
    *.cu|*.cpp|*.cc|*.cxx|*.c) ARGS+=("$a"); SRCS+=("$a") ;;
    *) ARGS+=("$a"); if [ -f "$a" ]; then INS+=("$a"); else MFLAGS+=("$a"); fi ;;
  esac
  i=$((i + 1))
done
[ -z "$PASS" ] && [ "$NOUT" != 1 ] && PASS=output
if [ -n "$PASS" ]; then log pass "${OUTF:--}"; exec "$CC" "$@"; fi

libfile() {   # libfile <name>: the file -l<name> resolves to, as the linker searches (-L dirs, then the CUDA and system dirs)
  local d f
  for d in "${LDIRS[@]}" /usr/local/cuda/lib64 /usr/local/cuda/targets/*/lib /usr/local/lib /usr/lib/*-linux-gnu /usr/lib \
           /lib/*-linux-gnu /lib; do
    for f in "$d/lib$1.so" "$d/lib$1.a"; do [ -e "$f" ] && { echo "$f"; return 0; }; done
  done
  return 1
}
deps() {   # deps <source>: the files the compiler reads for it, one per line
  "$CC" "${MFLAGS[@]}" -M "$1" 2> /dev/null | sed -e 's/\\$//' -e 's/^[^:]*://' | tr ' ' '\n' | sed '/^$/d'
}
KEY=$(
  {
    command -v "$CC"; "$CC" --version 2>&1
    if [ -n "$HOSTCC" ]; then echo "host $HOSTCC"; "$HOSTCC" --version 2>&1
    else for h in c++ g++ gcc; do echo "host $h"; "$h" --version 2>&1; done; fi
    ld --version 2>&1 | head -1
    for v in CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS NVCC_CCBIN; do
      printf 'env %s=%s\n' "$v" "${!v-}"
    done
    printf 'arg\0%s\n' "${ARGS[@]}"
    for s in "${SRCS[@]}"; do
      d=$(deps "$s") || exit 1
      [ -n "$d" ] || exit 1
      printf 'src %s ' "$s"; sha256sum < "$s" || exit 1
      echo "deps of $s"
      echo "$d" | sort -u | while read -r f; do printf '%s ' "$f"; sha256sum < "$f" || exit 1; done || exit 1
    done
    for f in "${INS[@]}"; do printf 'in %s ' "$f"; sha256sum < "$f" || exit 1; done
    if [ "$HASC" = 0 ]; then
      for l in "${LIBS[@]}"; do
        f=$(libfile "$l") || exit 1
        case $f in
          (*.a) printf 'lib %s ' "$f"; sha256sum < "$f" || exit 1 ;;
          (*) printf 'lib %s ' "$f"; stat -L -c '%s %Y' "$f" || exit 1 ;;
        esac
      done
    fi
  } | sha256sum | cut -c1-64
) || KEY=""
if [ ${#KEY} != 64 ]; then log pass "$OUTF"; exec "$CC" "$@"; fi
C=$DIR/${KEY:0:2}/$KEY
if [ -f "$C" ]; then
  # copy, then rename: a reader never sees half a file, and the cache entry is not tied to the output's inode
  if cp "$C" "$OUTF.nc$$" 2> /dev/null && chmod --reference="$C" "$OUTF.nc$$" && mv -f "$OUTF.nc$$" "$OUTF"; then
    [ -s "$C.err" ] && cat "$C.err" >&2
    log hit "$OUTF"; exit 0
  fi
  rm -f "$OUTF.nc$$"
fi
mkdir -p "$DIR/${KEY:0:2}" 2> /dev/null && [ -w "$DIR/${KEY:0:2}" ] || { log pass "$OUTF"; exec "$CC" "$@"; }
E=$DIR/${KEY:0:2}/.err$$
"$CC" "$@" 2> "$E"
rc=$?
cat "$E" >&2
if [ $rc != 0 ]; then rm -f "$E"; exit $rc; fi
# the stderr first, then the output: a reader that finds the output finds its warnings too
{ mv -f "$E" "$C.err" && cp "$OUTF" "$C.tmp$$" && mv -f "$C.tmp$$" "$C"; } 2> /dev/null || rm -f "$C.tmp$$" "$E"
log miss "$OUTF"
exit 0
