# The build's object cache

`engine/runtime/nvcc_cache.sh` sits in front of every `nvcc` and `g++` call in `engine/runtime/build_m4.sh` and
`scripts/build.sh` (#32). An edit then recompiles only the objects it reaches. The rest come from the cache, byte for
byte the files nvcc wrote the first time.

## Key

A compile's key is the SHA-256 of:

- the compiler's path and its `--version` output;
- every argument except the `-o` path (so `-D`, `-O`, `-gencode`, `-I` all count);
- for each source on the command line, the list of files the compiler itself reports reading (`-M`, with the same
  flags), and the contents of each one;
- the contents of every other input file on the command line (objects and archives in a link).

A hit copies the stored output to `-o` and keeps its mode. A miss runs the compiler and stores the output only if the
compile succeeds. Calls with no `-o` or with two, calls that read stdin, and dependency or preprocess runs (`-M`, `-E`)
go straight to the compiler.

Not in the key: the environment (`CPATH`, `PATH` past the compiler, `NVCC_PREPEND_FLAGS`) and `-l` libraries, which
only appear by name. A CUDA toolkit upgrade changes `--version`. A new libibverbs or cuBLAS of the same toolkit does
not, so build once with `G6_NVCC_CACHE=0` after one, or empty the cache.

## Knobs

| variable | default | |
|---|---|---|
| `G6_NVCC_CACHE` | 1 | 0: plain `nvcc` / `g++`, the cache is neither read nor written |
| `G6_NVCC_CACHE_DIR` | `~/.cache/gr8-nvcc` | where outputs are kept, one file per key; `rm -rf` it to empty |
| `G6_NVCC_CACHE_LOG` | set by `build_m4.sh` | one `hit`/`miss`/`pass` line per call; `build_m4.sh` prints the counts |

The cache never evicts. One full build (every width) adds about the size of `bin/obj` plus the two binaries; delete
the directory when it gets large.

## Test

`engine/runtime/tests/nvcc_cache_test.sh` (and `scripts/host_tests.sh`) checks, with g++: a repeat compile hits and
gives the same bytes; a changed flag, a new `-D` or an edited header misses; the header put back hits again; a link
with a changed object misses; a failing compile returns its status and stores nothing; `G6_NVCC_CACHE=0` bypasses it.
