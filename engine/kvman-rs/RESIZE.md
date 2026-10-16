# A KV pool that grows and shrinks at runtime (#153)

A narrower layout puts more weights on each box, and that memory has to come out of the KV pool; a wider layout gives
it back. This is the pool side and the memory side of that. The engine does not call either yet: the switch (#156)
does, and until then every boot runs with every chunk online, which is the pool it always was.

## Chunks

`Pool::chunks(k)` cuts the pages below `place` into chunks of `k` pages (the last may be shorter). Pages past `place`
(the drafter's tail class) never go offline. A chunk is online or offline. An offline chunk:

- holds no page of a live run and no page of a retained run;
- is not free, not room and not counted by the Banker's check: `free_pages`, `room`, `fits`, `grow`, `match_prefix`
  and `take` all see only the online pages below `place` (`place_online()`);
- for an exclusive run on the tables as built (pages 0, 1, ..), cuts the range: such a run fits only in the online
  pages before the first offline chunk.

With every chunk online the pool gives the same runs, room and digest as one without chunks
(`tests/resize.rs`, `chunks_online_change_nothing`), so nothing changes at a fixed size.

## Shrinking

`Pool::shrink(n)` takes `n` chunks offline, or waits:

- it picks the highest online chunks that no live run holds a page of. It never evicts or moves a live run;
- every retained run with a page in those chunks is evicted (and spills to the stores, when there are stores; the
  engine runs the queued copies before it releases the memory);
- it waits (`Ok(None)`, nothing changed) while an exclusive run is resident, when fewer than `n` chunks are free of
  live runs, or when the live runs could not all reach their worst cases in what would stay online. The caller tries
  again after sessions finish: the switch waits instead of evicting a live session;
- it refuses (`BadSize`) a size no state allows: fewer than `n` chunks online, or less than R / 64 pages left online.

## Growing

`Pool::grow_pool(n)` brings `n` offline chunks back, the lowest first. The pool is in a safe state before (every state
it reaches is one) and only gains free pages, so it stays safe. The engine maps the memory before this call.

## Memory

`engine/attn/kvchunks.cu` holds a cache (a binding's MLA rows or its index keys) in reserved virtual memory, backed
granule by granule with `cuMemCreate` / `cuMemMap` (2 MiB granules on GB10, at either host page size). `g6_kvc_sync`
backs exactly the granules that hold a byte of an online chunk or of a page past `place`, and releases the rest. The
address never moves, so block tables, kernels and captured graphs keep working. A granule two chunks share stays while
either chunk is online, so a chunk gives back all its bytes only when it is a whole number of granules of every cache
it covers. A page is 64 rows: 41,984 B of FP8 MLA rows (`ROW` 656), 26,624 B of INT4 rows (416) and 8,448 B of index
keys (132), so a 2 MiB granule is never a whole number of pages. 8,192 pages is a whole number of granules of all three
(328, 208 and 66 MiB); with smaller chunks the boundary granules stay, at most two per cache per chunk.

`g6_kvc_sync` synchronizes the whole context before it unmaps anything, since g6m4 runs prefill lanes, decode graphs
and store copies on several streams. A remote store registers KV pages with the RDMA device; a resize must happen
with no such copy in flight, and #156 must register the regions again after a grow (granules that come back are new
memory at the same address).

Order for a shrink: `shrink` -> run the queued spills (`io()` / `io_done()`) -> `g6_kvc_sync` for every cache (memory
goes). For a grow: `g6_kvc_sync` with the chunks marked online (memory comes, zeroed) -> `grow_pool`.

## Not here yet

- **Engine wiring.** `bind.cu` still `cudaMalloc`s each binding's storage in one piece. Moving the KV caches into
  `kvchunks` regions (one per cache, the rest of the binding as now) and the S2 control commands for a resize belong
  with the switch (#156).
- **Limits per layout.** `CAP` and `SHARE_LIMIT` stay per boot in `scripts/engine_args.sh`. Per layout, they become
  the pool's chunk count online at that layout (`place_online`) and its R; what the plan file or the switch command
  carries is #150's and #156's format.
- **Speed of a VMM-backed pool.** Each granule is its own `cuMemCreate` and mapping (about 50,000 for a 100 GB pool).
  The attention kernels read the same addresses, and `kv_resize_test` checks they give the same bits, but whether
  decode and prefill keep their numbers with the caches in mapped granules instead of one `cudaMalloc` is measured
  when #156 moves the engine's caches into it (one allocation per chunk is the alternative if they do not).
- **The reserve.** How much of the pool a deployment holds back so it can shrink is decision 4 on #147 (people).
  Nothing here picks a number: a shrink takes what is free of live runs and waits for the rest.
