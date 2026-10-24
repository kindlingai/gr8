#!/bin/bash
# nvcc_cache.sh (#32) with g++ standing in for nvcc: hits only when the compile would see the same bytes and flags.
#   engine/runtime/tests/nvcc_cache_test.sh        (scripts/host_tests.sh runs it)
set -eu
C=$(cd "$(dirname "$0")/.." && pwd)/nvcc_cache.sh
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
export G6_NVCC_CACHE_DIR=$T/cache G6_NVCC_CACHE_LOG=$T/log
cd "$T"
fail() { echo "nvcc_cache_test FAILED: $1"; exit 1; }
last() { tail -1 "$G6_NVCC_CACHE_LOG" | cut -d' ' -f1; }
printf '#pragma once\n#define V 1\n' > h.h
printf '#include "h.h"\nint f() { return V; }\n' > a.cpp
printf 'int f();\nint main() { return f() == 1 ? 0 : 1; }\n' > m.cpp
"$C" g++ -O2 -c a.cpp -o a.o;            [ "$(last)" = miss ] || fail "first compile is not a miss"
cp a.o a.first
"$C" g++ -O2 -c a.cpp -o a.o;            [ "$(last)" = hit ] || fail "same compile is not a hit"
cmp -s a.o a.first || fail "hit gives other bytes"
"$C" g++ -O2 -c a.cpp -o b.o;            [ "$(last)" = hit ] || fail "another -o path is not a hit"
"$C" g++ -O1 -c a.cpp -o a.o;            [ "$(last)" = miss ] || fail "a changed flag hits"
"$C" g++ -O2 -DX=1 -c a.cpp -o a.o;      [ "$(last)" = miss ] || fail "a new -D hits"
printf '#pragma once\n#define V 2\n' > h.h
"$C" g++ -O2 -c a.cpp -o a.o;            [ "$(last)" = miss ] || fail "an edited header hits"
cmp -s a.o a.first && fail "an edited header gives the old object"
printf '#pragma once\n#define V 1\n' > h.h
"$C" g++ -O2 -c a.cpp -o a.o;            [ "$(last)" = hit ] || fail "the header back as it was is not a hit"
cmp -s a.o a.first || fail "the header back as it was gives other bytes"
# a header found through -I DIR (a separate argument), and one through -include FILE
mkdir -p inc && printf '#define W 1\n' > inc/w.h && printf '#define P 1\n' > pre.h
printf '#include "w.h"\nint g() { return W + P; }\n' > g.cpp
"$C" g++ -O2 -I inc -include pre.h -c g.cpp -o g.o;  [ "$(last)" = miss ] || fail "-I / -include: not a miss"
"$C" g++ -O2 -I inc -include pre.h -c g.cpp -o g.o;  [ "$(last)" = hit ] || fail "-I / -include: not a hit"
printf '#define W 2\n' > inc/w.h
"$C" g++ -O2 -I inc -include pre.h -c g.cpp -o g.o;  [ "$(last)" = miss ] || fail "an edited -I header hits"
printf '#define P 2\n' > pre.h
"$C" g++ -O2 -I inc -include pre.h -c g.cpp -o g.o;  [ "$(last)" = miss ] || fail "an edited -include file hits"
# warnings come back on a hit
printf 'int h() { int u; return 0; }\n' > w.cpp
"$C" g++ -Wall -c w.cpp -o w.o 2> w1.err; "$C" g++ -Wall -c w.cpp -o w.o 2> w2.err
[ "$(last)" = hit ] || fail "warning compile: not a hit"
grep -q unused w1.err && cmp -s w1.err w2.err || fail "a hit does not replay the warnings"
# the environment the compiler reads is in the key
CPATH=inc "$C" g++ -O2 -c a.cpp -o a.o;  [ "$(last)" = miss ] || fail "a changed CPATH hits"
# a link: the inputs are objects, so their bytes are in the key
"$C" g++ -o m m.cpp a.o;                 [ "$(last)" = miss ] || fail "first link is not a miss"
./m || fail "linked binary does not run"
"$C" g++ -o m m.cpp a.o;                 [ "$(last)" = hit ] || fail "same link is not a hit"
[ -x m ] || fail "a hit loses the executable bit"
printf '#pragma once\n#define V 3\n' > h.h
"$C" g++ -O2 -c a.cpp -o a.o
"$C" g++ -o m m.cpp a.o;                 [ "$(last)" = miss ] || fail "a link with a changed object hits"
./m && fail "the link used the old object"
# a failing compile: its status comes back and nothing is stored
n=$(find "$G6_NVCC_CACHE_DIR" -type f | wc -l)
printf 'int g( {\n' > bad.cpp
if "$C" g++ -c bad.cpp -o bad.o 2> /dev/null; then fail "a failing compile succeeds"; fi
[ "$(find "$G6_NVCC_CACHE_DIR" -type f | wc -l)" = "$n" ] || fail "a failing compile stored something"
# not cacheable: no -o, -x, a response file
"$C" g++ -c a.cpp;                       [ "$(last)" = pass ] || fail "no -o is cached"
cp a.cpp a.inl
"$C" g++ -x c++ -c a.inl -o x.o;         [ "$(last)" = pass ] || fail "-x is cached"
echo -O2 > rsp
"$C" g++ @rsp -c a.cpp -o r.o;           [ "$(last)" = pass ] || fail "a response file is cached"
# eight compiles of one key at once: one entry, the same bytes everywhere
printf 'int k() { return 7; }\n' > k.cpp
for j in 1 2 3 4 5 6 7 8; do "$C" g++ -O2 -c k.cpp -o k$j.o & done; wait
for j in 2 3 4 5 6 7 8; do cmp -s k1.o k$j.o || fail "concurrent compiles of one key differ"; done
"$C" g++ -O2 -c k.cpp -o k9.o;           [ "$(last)" = hit ] || fail "after concurrent compiles: not a hit"
cmp -s k1.o k9.o || fail "concurrent compiles stored other bytes"
# off: the compiler alone, nothing logged
: > "$G6_NVCC_CACHE_LOG"
G6_NVCC_CACHE=0 "$C" g++ -O2 -c a.cpp -o a.o
[ ! -s "$G6_NVCC_CACHE_LOG" ] || fail "G6_NVCC_CACHE=0 still goes through the cache"
echo "nvcc_cache_test: OK"
