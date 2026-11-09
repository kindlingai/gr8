# The build's object cache

`engine/runtime/nvcc_cache.sh` sits in front of every `nvcc` and `g++` call in `engine/runtime/build_m4.sh` and
`scripts/build.sh` (#32). An edit then recompiles only the objects it reaches, and the rest come from the cache.
`scripts/c1_tests.sh` and `scripts/host_tests.sh` still call the compilers directly.

## Key

A compile's key is the SHA-256 of:

- the compiler's path and its `--version` output, the host compiler's `--version` (`-ccbin`, else `c++` on `PATH`),
  and the first line of `ld --version`;
- the environment the compilers read: `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`,
  `NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS` and `NVCC_CCBIN`;
- every argument except the `-o` path (`-D`, `-O`, `-gencode`, `-I` and the rest);
- for each source on the command line, the files the compiler reports reading (`-M` with the same flags, so `-I` and
  `-include` count), with the contents of each one;
- the contents of every other input file on the command line (objects and archives in a link);
- in a link (no `-c`), the file each `-l` library resolves to: the contents of an archive, or the path, size and
  modification time of a shared library.

A hit copies the stored output to `-o`, keeps its mode, prints the compiler's stored stderr, so warnings still show, and
sets the entry's time to now, so an entry's time is its last use.
A miss runs the compiler, and stores the output and stderr only if the compile succeeds. Entries are written to a
temporary name and then renamed, so parallel jobs on one key are safe.

These calls go straight to the compiler and are never cached: no `-o`, or more than one; input from stdin; `-x`;
response files (`@FILE`, `-optf`, `--options-file`); dependency and preprocess runs (`-M*`, `-E`); a `-l` library the
script cannot find; any argument with a space in it.

Limits:

- `nvcc -M` lists headers as the host pass sees them, so a header included only under `#if __CUDA_ARCH__` would not
  be in the key. No header in `engine/` is included that way today.
- The key does not cover the driver or the GPU. It does not need to, because compiles do not read them.

## Knobs

| variable | default | |
|---|---|---|
| `GR8_NVCC_CACHE` | 1 | 0: plain `nvcc` / `g++`, the cache is neither read nor written |
| `GR8_NVCC_CACHE_DIR` | `~/.cache/gr8-nvcc` | where outputs are kept, one file per key; `rm -rf` it to empty it |
| `GR8_NVCC_CACHE_LOG` | set by `build_m4.sh` and `build.sh` | one `hit`/`miss`/`pass` line per call; the builds print the counts |

The cache never evicts. One full build (every width) adds about the size of `bin/obj` plus the binaries, so delete the
directory when it gets large, or delete what no build used lately: `find ~/.cache/gr8-nvcc -type f -mtime +30 -delete`.

## In the image build and CI

Each engine stage of the `Dockerfile` mounts the cache at `/var/cache/gr8-nvcc` (`GR8_NVCC_CACHE_DIR`), as a BuildKit
cache mount with one id a stage: `gr8-nvcc-fp8`, `-int4`, `-flash`, `-flash-int4`, `-ds41` and `-tests`. After its
build the stage deletes the entries older than the build's start, the ones it did not read, so the mount holds one
build. A local `docker buildx build` keeps the mount between builds on the same builder.

CI (`.github/workflows/ci.yml`, the engine job) keeps each leg's mount in the Actions cache. Every run restores the
newest copy for its leg (key `gr8-nvcc-<leg>-<hash of the compile inputs>`) and loads it into the builder with
reproducible-containers/buildkit-cache-dance. Pushes to main save it, once for each set of inputs. The layer cache still
comes first: when the engine layer is cached, nothing compiles. The compile cache matters when that layer has to run
again: an edit to one engine file recompiles the objects that include it and takes the rest from the cache. The build
log's `nvcc cache: N hit, M miss` line shows what it did.

## Test

`engine/runtime/tests/nvcc_cache_test.sh` (also run by `scripts/host_tests.sh`) checks these cases with g++:

- a repeat compile hits and gives the same bytes;
- a changed flag, a new `-D`, or an edited header (direct, through `-I DIR`, or through `-include`) misses;
- the header put back hits again;
- a changed `CPATH` misses;
- a hit replays the warnings;
- a link with a changed object misses;
- eight compiles of one key at once give one entry and the same bytes;
- a failing compile returns its status and stores nothing;
- `-x`, response files and calls with no `-o` pass through;
- `GR8_NVCC_CACHE=0` bypasses the cache.
