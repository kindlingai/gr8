# Compact expert copies

The optional `GR8_LOAD_COMPACT` directory lets the Rust loader read a rank's expert
ranges from dense copies instead of the sparse release files. Unset or empty keeps
the normal loader path. This changes where bytes are read, not their values.

## Prepare, then boot

Use a different output directory for each release, rank, TP size, variant and layout.
On each rank, set the same absolute directory for prepare and boot (paths below are placeholders):

```sh
export EXTRA_ENV="GR8_LOAD_COMPACT=$COMPACT_DIR"
./gr8 glm53 prepare --rank "$RANK" --tp 4 --model "$MODEL" --model-cache "$CACHE" --variant home
./gr8 glm53 up --tp 4 --model "$MODEL" --model-cache "$CACHE" --variant home
```

Keep any other `EXTRA_ENV` assignments your configuration needs. `prepare` first
fetches/checks the release normally, then writes only this rank's expert ranges.
Local releases work too. Custom ownership follows `--layout` on both commands.
Boot does not write or refresh compact files.

Preparation writes to a sibling staging directory, reads every requested half back
through the loader's compact map, checks its index-slice xxhash64, then publishes the
directory. An existing directory is checked and reused, never overwritten. If it is
stale or corrupt, prepare fails; select a new destination to rebuild it. Preparation
needs additional disk space for the copies. Interrupted preparation may leave a
`.partial-<pid>` sibling; it is not used by the loader.

The standalone fetch helper can do the same for an already-fetched release:

```sh
g6load compact --v2 "$EXPERTS" --slices "$SLICES" --manifest "$WIDTHS" \
    --tp 4 --rank "$RANK" --out "$COMPACT_DIR"
```

Do not remove the originals. They still provide metadata, non-expert weights,
generated tensors and any unmapped ranges. Keep the normal boot weight hash check
on: the reader checks source sizes and bounds, but the map itself is not a content
identity. Copies must match the release, rank and layout being loaded.

## Map format

For source file `<name>`, the directory contains `<name>` (dense bytes) and
`<name>.g6map` (text):

```text
g6map-v1 <original-file-size>
<source-offset> <length> <copy-offset>
```

Rows are sorted, nonoverlapping source ranges. The writer coalesces touching or
overlapping ranges and writes them contiguously. The reader also accepts partial
maps: a Copy or Item128 segment is remapped only when a single row covers it fully;
otherwise it reads the original source. Generated tensors are unaffected. Both read
and mmap modes support the maps. With compact loading enabled, the loader raises its
soft open-file limit to the existing hard limit before opening sources and copies.
