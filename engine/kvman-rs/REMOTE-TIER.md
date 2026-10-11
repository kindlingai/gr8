# A remote tier for retained runs: proof of concept

Status: a benchmark and tests only. Nothing in the engine calls `src/remote`, and it has no C entry points.
[SSD-TIER.md](SSD-TIER.md) has the problem both tiers address: a busy pool evicts retained runs, and a request that
loses its run prefills its whole prompt again.

## What it is

A KV node is a box that locks some of its memory and serves it to the cluster as page blocks. A page block is the SSD
tier's unit (one 64-row page of every layer, 3,502,592 B, 3,506,176 B with 4 KiB rounding).

- `kvman-remote` is the node. It `mlock()`s `--gb` GiB, serves it as leases, and stops when locking would leave less
  than `--min-free-gb` (10) of MemAvailable or when RLIMIT_MEMLOCK is too low.
- `RemoteStore` (`src/remote/client.rs`) is a client's lease as a `tier::BlockStore`. `SlotFile` implements the same
  trait. `Tier` runs over either. The Tier decides which slot holds which page, lowest free first, so both stores
  follow the same lock-step allocation.
- Transports: RDMA (RoCE v2 RC, one-sided reads and writes from the client into the node's registered memory) when
  both ends of the connection have a RoCE v2 GID on its addresses. TCP otherwise. Each end reads its GID index from
  sysfs for the address the connection uses: the index differs per box. libibverbs is loaded with `dlopen`, so the
  crate builds without rdma-core, and a box without it uses TCP.
- Leases: a client names its lease (a box and rank, say), and a name has one live holder. The node checks every TCP
  offset against the lease. For RDMA each lease has its own protection domain, holding one memory region over exactly
  its range and its clients' QPs. The NIC refuses another lease's rkey and any address past the end. A lease is zeroed
  when granted and freed when its last connection closes. The earlier prototype had one region for everyone, and a
  unit test once overwrote a live stack's rows.
- `./gr8 kv up --gb N` starts a node like a model (KNOBS.md, "`./gr8 kv`"). The node joins the mentat group as an
  agent without GPUs, announcing the service `gr8-kv`, its size and its RDMA addresses. `mentat.kv_nodes()` lists the
  group's nodes. `./gr8 kv status` shows a node's leases, and `status all` every node in the group.

## Results: kvman-remote-bench

Two GB10 boxes (ASUS GX10): the node on one with 8 GiB locked (0.2 s to lock), the client on the other. RDMA runs used a
direct 200G cable between them, one PCIe root unless stated. TCP ran over the same cable and over the 10G LAN. Leases of
2-4 GB, buffers registered with the device unless stated. GB/s are 10^9 bytes.

Page blocks, GB/s and ms per block (p50 / p99):

| transport | threads | write | write ms | read | read ms |
|---|---|---|---|---|---|
| RDMA | 1 | 13.33 | 0.262 / 0.263 | 13.24 | 0.264 / 0.265 |
| RDMA | 2-16 | 13.97 | 0.50 / 0.51 at 2 | 13.97 | 0.50 / 0.51 at 2 |
| RDMA, staging copy (`--copy`) | 1 | 10.17 | 0.344 / 0.350 | 9.23 | 0.379 / 0.383 |
| RDMA, staging copy | 4-8 | 13.95 | | 13.94 | |
| RDMA, both roots, two clients at once | 4-8 each | 12.24 + 12.24 | | 12.23 + 12.23 | |
| TCP over the 200G cable | 1 | 3.42 | 0.96 / 1.96 | 2.43 | 1.27 / 5.20 |
| TCP over the 200G cable | 8 | 13.74 | 1.98 / 3.13 | 13.82 | 2.02 / 2.58 |
| TCP over the 10G LAN | 1 | 0.88 | 3.85 / 6.39 | 0.98 | 3.36 / 6.70 |
| TCP over the 10G LAN | 2-8 | 1.15-1.18 | | 1.17-1.18 | |

13.97 GB/s is 112 Gb/s, the most one PCIe root has moved on these boxes (109-112 Gb/s with `ib_write_bw`). Both roots
together moved 24.5 GB/s.

Rows of 2,624 B (one token's MLA rows of one indexer group) at random places, one thread per read in flight:

| transport | queue depth | reads/s | GB/s | p50 us | p99 us | p99.9 us |
|---|---|---|---|---|---|---|
| RDMA | 1 | 259,088 | 0.68 | 3.8 | 4.0 | 4.5 |
| RDMA | 8 | 1,901,939 | 4.99 | 4.1 | 4.9 | 5.7 |
| RDMA | 16 | 3,139,904 | 8.24 | 4.9 | 6.9 | 8.0 |
| RDMA | 32 | 3,406,187 | 8.94 | 5.7 | 8.6 | 3,004 |
| TCP, 200G cable | 1 | 1,682 | 0.00 | 559 | 1,445 | 1,739 |
| TCP, 200G cable | 64 | 698,823 | 1.83 | 82 | 215 | 519 |
| TCP, 10G LAN | 1 | 2,072 | 0.01 | 405 | 1,367 | 1,732 |
| TCP, 10G LAN | 64 | 168,717 | 0.44 | 360 | 581 | 729 |

Each RDMA reader polls its completion queue. At depth 32 the client runs more polling threads than the box has cores,
which is where the 3 ms tail comes from. TCP latency on these boxes is wake-up latency: an idle `ping` between them
takes 0.7-1.5 ms on either link.

Gathers, N random rows in one request, one thread:

| transport | rows | p50 ms | p99 ms | GB/s |
|---|---|---|---|---|
| RDMA | 256 | 0.086 | 0.087 | 7.76 |
| RDMA | 2,048 | 0.649 | 0.654 | 8.19 |
| RDMA | 4,096 | 1.293 | 1.298 | 8.23 |
| RDMA, staging copy | 2,048 | 0.829 | 0.870 | 6.42 |
| TCP, 200G cable | 2,048 | 1.94 | 3.08 | 2.74 |
| TCP, 10G LAN | 2,048 | 4.92 | 5.55 | 1.08 |

`--isolation`: lease A's QPs read their own region, and the NIC refused both a read of lease B's region with B's own
rkey and a read one block past A's end (remote access error).

A third box without a RoCE GID picked TCP on its own (`--transport auto`) and matched the LAN numbers above (1.17 GB/s,
466 us at depth 1).

Not tested here: the path through the switch. The node box's switch-facing port is 100G, and the cluster boxes were
busy. The earlier prototype ran that path: one box at 95 Gb/s, four boxes together 96.8 Gb/s, and write p99 tails (no
PFC, 200G into 100G).

## Against the SSD tier

| | local NVMe (SSD-TIER.md) | KV node, direct cable | KV node through a 100G switch port | KV node, 10G LAN |
|---|---|---|---|---|
| page blocks, write / read | 5.7 / 7.0 GB/s | 14.0 / 14.0 GB/s per root | about 12 GB/s for the cluster, about 3 GB/s each when four boxes move at once | 1.18 / 1.18 GB/s |
| a 40,000-token run (2.19 GB), write / read | 0.38 / 0.31 s | 0.16 / 0.16 s | about 0.7 s each when four ranks move at once | 1.9 / 1.9 s |
| one 2,624 B row, depth 1, p50 | 43 us (4 KiB) | 3.8 us | not tested | 405 us |
| small reads/s at most | 832,555 (depth 128) | 3.4 M (depth 32) | not tested | 169,000 |
| capacity | the drive (the simulation used 32-256 GB a rank) | what the node locks | the same | the same |
| drive writes | 92-179 GB a rank in 36-46 min of replay | none | none | none |

What the numbers show:

- Large transfers: a direct cable beats the local NVMe 2-2.5x. Through the node's 100G switch port, four boxes
  spilling at once get about 3 GB/s each, below their own NVMe. The 10G LAN is 4.8-5.9x slower than the NVMe, and
  still restores a 40,000-token run in 1.9 s against about 31 s to prefill it again.
- Small reads: RDMA reads one row in 3.8 us, 11x faster than the NVMe at depth 1, and a 2,048-row gather takes 0.65
  ms. That is the remote tier's clearest case: the sparse row cache estimate in SSD-TIER.md needs 1,800-22,000 row
  reads a token. At 3.4 M reads/s that is 0.5-6.4 ms a token, where the NVMe needs 2.2-26 ms. Over TCP a small read
  takes longer than on the NVMe.
- Capacity: below R every rank holds a session's whole KV. A TP=4 cluster stores each retained run four times. 8 GiB
  on the node is 2,450 page blocks: 612 pages a rank, about 39,000 tokens of retained runs for the cluster. That holds
  one long agent session. The simulation used 32 GB a rank. If the ranks' copies are byte-identical, one copy would
  serve every rank, and the same 8 GiB would hold about 157,000 tokens. Stage B checks that first.
- Wear: every byte the node holds is a byte the NVMe does not write. The SSD simulation wrote 120-300 GB an hour per
  rank.

## Running it

    ./gr8 kv up --gb 8                    # on the box with spare memory; ./gr8 kv status, ./gr8 kv down
    kvman-remote-bench --server HOST      # from a client box: --transport auto|tcp|rdma, --isolation, --copy

`kvman-remote` alone: `kvman-remote --gb 8 [--port 7350] [--min-free-gb 10] [--no-rdma]`. For a test outside the
launcher, `sudo systemd-run --unit=kv-node -p LimitMEMLOCK=infinity -p User=$USER kvman-remote --gb 8` is easy to stop.

## Stage B: the engine spills to a tier (plan)

Off by default, and the same for both stores.

- Switch: `KV_TIER` (`--kv-tier`), a cluster option the handshake compares: `off`, `ssd:PATH:GB`, or
  `remote[:HOST:PORT]:GB`. Without a host the supervisor takes a node from `mentat.kv_nodes()`. Each rank names its
  lease `<cluster digest>/<rank>`, so two ranks or two clusters never share one.
- Pool: the pool keeps the stored runs: each one's kept age, reuse state and slots, lowest free first, from a
  `SlotAlloc` of the store's size. `fill()` and the exclusive branch of `reserve` / `take` already hold the evicted
  runs (`gone`). Each one goes to the store when it fits, else the oldest stored run drops. `match` tries the pool's
  retained runs first, then the stored ones with the same `begin` rule. A hit on a stored run makes `take` return the
  grant's pages and the slots to read into the first ceil(skip / 64) of them.
- Lock-step: every decision is a function of pool state alone, the same on every rank whatever the I/O does. The head
  decides at the event, and the grant carries the restored run (its age and page count). Each worker repeats `match` /
  `take` at its grant and requires the same run and the same restore, as G6S2E2 and G6S2E3 did for page lists and
  growth. The pool digest folds in the stored runs' ages, slots and the free count, so a rank whose store differs
  fails at the next event. The frame version moves to G6S2E4.
- I/O: each event yields two lists: spills (page to slot) and restores (slot to page). A spill gathers the 101
  per-layer pieces of a page (79 x 41,984 B and 22 x 8,448 B, `bind.cu create`) device to host into pinned staging,
  registered with `RemoteStore::register` so RDMA skips the copy. The device copy must finish before the grant's
  prefill overwrites the page. The store write can trail it. A restore reads into staging and scatters back before the
  prefill of the rest of the prompt. A hit on a run whose writes have not landed waits for them on that rank, so
  `match` stays independent of timing. The first version treats a failed store write or read as fatal for the rank.
  That stops the cluster, as any rank failure does.
- Open first: whether the ranks' page blocks are byte-identical below R: hash every rank's blocks of one run in m4sim.
  If the copies are identical, one lease serves every rank and the capacity grows 4x at TP=4. The DFlash2 drafter's KV
  is not in a page block, so a restored run has to rebuild it or store it too. Exclusive runs past R hold different
  pages on each rank (the CP map). The switch path with four boxes needs a run once the cluster is free.
- Tests: pool property tests with stored runs (every slot held once, ranks agree), a `remote` policy in `kvman-sim`
  with the rates above (14, 3 and 1.1 GB/s), and m4sim with a store on a loopback node.
