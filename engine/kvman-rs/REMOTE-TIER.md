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

## Stage B: stores below the pool

Experimental. Off unless `--kv-store` is given. Without it the pool has no store, counts nothing, queues no copies, and its digest is
what it was. #37 shipped this as `--kv-tier STORE,size=N,keep=evicted|shared|all` per tier; that flag and
`--kv-shared-after` remain for one release as deprecated aliases (a tier that kept evicted runs, the old default, maps to
`--kv-keep all`, else `system`).

### Configuration

    --kv-store ssd:/nvme/gr8kv,size=200G       # where: repeatable, tried in order
                                               # ,persist=on keeps an SSD store across restarts
    --kv-store remote:auto,size=8G
    --kv-keep system|sessions|all              # what: default system
    --kv-system-after 2                        # default 2
    --kv-write-limit 0.25                      # optional cap, off by default

The launcher's `KV_STORE` holds the stores separated by `;`, with `KV_KEEP`, `KV_SYSTEM_AFTER` and `KV_WRITE_LIMIT`
(KNOBS.md). A store is `ssd:PATH` (a folder: rank r keeps `PATH/r<r>/blocks` and `PATH/r<r>/index`) or
`remote:HOST:PORT`. `remote:auto` is the launcher's: each box's supervisor names the KV node mentat announces, with an
RDMA address on one of the box's rails first. `size` is per rank, binary (G = 2^30, T = 2^40). An SSD store is cleared
at every boot unless it has `persist=on` (below, Persistence).

### What the stores keep

- `system` (the default): the system prompt and tool definitions of each conversation, once per distinct span, after
  `--kv-system-after` conversations started with it. The front ends send the span (`system_tokens`, PROTOCOL.md): the
  prompt's tokens up to where the first message after the leading system messages begins, as the chat template renders
  them. A new conversation counts once toward every grid-cell prefix (4,096 rows served) of its prompt within the span;
  without a span (the completions API) toward every cell of its prompt. A request that continues a live, retained or
  stored run (its prompt holds all of that run's prompt) counts toward nothing, so one long chat never reaches N. When a
  run retires, the longest counted prefix it holds within its span goes to the first store, unless a store has it. The
  key of a prefix of S rows is its first S + 1 tokens, since the MTP layer's row S - 1 depends on token S. The count
  table keeps the 4,096 most recently counted prefixes, and the first `persist=on` SSD store's index keeps it across
  restarts.
  Counting cells rather than whole spans matters for agents: opencode writes the workspace folder into its system
  prompt 29% of the way in, so agents in different folders share 6,926 of their 12,637-token spans, and only the
  cells before the folder.
- `sessions`: those, plus every run the pool evicts that continued its conversation at least once (a store policy,
  `min_hits` 1: a one-shot never spills). A run past R also keeps its tail (below).
- `all`: every run the pool evicts, and the system prefixes. For benchmarks.
- Every policy: a run whose leading pages another run in the pool holds (live or retained, the same tokens) stores
  that prefix once as a prefix entry, and itself as a reference to it plus its own pages; a run that starts with a
  stored prefix stores only the pages after it. A restore takes the prefix from a retained run in the pool when one
  holds it (no copy), else from the store.
- `--kv-write-limit F`: the stores write at most F blocks per page of KV the engine computes (a grant's prefill pages
  and growth). Every rank earns and spends the same credit, never by the clock. A write needs credit above zero and
  may take it below; the credit holds at most 4,096 blocks. Without credit a spill is refused and a move drops its
  entry.
- Stores are tried in order, for lookups and for placement. A full store frees its least recently used entries, and
  each goes on to the next store that takes it (a block-to-block move), or is dropped. Rank 0 logs, at most once a
  minute, what a store did once it fills, passes entries on, drops or refuses them; `status.s2` carries each store's
  pages and bytes spilled, moved in and out, restored, dropped and refused, and the front ends export them as
  `g6:kv_store_*{store="i"}`.
- A hit on an evicted run takes it out of its store, and a hit on a prefix leaves it there. A stored hit wins only when
  it reuses more rows than the pool can.

### The policies on agent traffic (kvman-sim)

`kvman-sim` runs the pool and the stores' own code (src/stash.rs) over the opencode captures with the engine's time
model (prefill 1,300 tok/s, decode 30/51.5/64.4 tok/s for 1/2/4 sessions, SSD 5.7 GB/s write, 7.0 read, every drain on
everyone's critical path), runs growing on demand. Traces: `sim/traces/before.csv` (4 agents, which fit the pool),
`agents8.csv` (the 8-agent replay: both captures at once) and `agents8.csv --repeat 4 --stagger 300` (32 agents in other
folders: each copy shares the 7.4K tokens every span shares). Writes are per rank. `grow` has no store.

| 4 agents | TTFT p50 / p90 s | prefilled tokens | written GB/h | read GB |
|---|---|---|---|---|
| grow | 3.1 / 25.3 | 248,144 | 0 | 0 |
| evicted (#37) | 3.4 / 20.8 | 215,376 | 109.0 | 1.8 |
| all | 3.3 / 20.8 | 215,376 | 93.5 | 1.8 |
| sessions | 3.3 / 20.8 | 215,376 | 90.8 | 1.8 |
| system, N = 1 | 3.1 / 20.6 | 235,856 | 3.7 | 0.7 |
| system, N = 2 | 3.1 / 24.4 | 244,048 | 0.3 | 0.2 |

| 8 agents | TTFT p50 / p90 s | prefilled tokens | written GB/h | read GB |
|---|---|---|---|---|
| grow | 197 / 615 | 2,578,825 | 0 | 0 |
| evicted (#37) | 123 / 508 | 473,481 | 145.1 | 126.6 |
| evicted + prefix bases | 122 / 508 | 473,481 | 126.4 | 120.5 |
| all | 122 / 508 | 473,481 | 126.4 | 120.5 |
| sessions | 122 / 508 | 506,249 | 122.7 | 118.7 |
| system, N = 1 | 152 / 584 | 1,751,433 | 3.6 | 56.5 |
| system, N = 2 or 3 | 197 / 615 | 2,501,001 | 0.1 | 4.3 |
| all, write limit 0.25 | 148 / 615 | 2,025,865 | 28.5 | 31.0 |

| 32 agents, 200 GB store | TTFT p50 / p90 s | prefilled tokens | written GB/h | read GB |
|---|---|---|---|---|
| grow | 1,333 / 2,285 | 11.5M | 0 | 0 |
| evicted (#37) | 889 / 1,751 | 1.81M | 155.4 | 594 |
| all | 896 / 1,749 | 1.84M | 135.2 | 554 |
| sessions | 904 / 1,749 | 2.15M | 130.2 | 536 |
| system, N = 1 | 1,211 / 2,064 | 7.92M | 3.5 | 263 |
| system, N = 2 | 1,373 / 2,268 | 11.1M | 0.0 | 21 |
| sessions, write limit 0.25 | 1,350 / 2,236 | 10.5M | 24.8 | 56 |

With a 29 GB store (what the testers ran) the 8-agent numbers are the same; at 32 agents `all` falls to 1,148 / 2,194 s
and `sessions` to 1,174 / 2,194 s, since the store turns over before the agents come back.

- Inside the pool, every policy that stores evictions writes 90-110 GB/h per rank for one restore: what the testers saw
  (~22 MiB/s a node). `system` writes 0.3-3.7 GB/h and loses nothing measurable.
- With more agents than the pool holds, storing sessions pays: `all` and `sessions` cut TTFT 33-38% and prefill 80-85%
  against no store. `sessions` writes about what `all` does on agent traffic, since nearly every agent turn continues
  its conversation; it saves the one-shots.
- `system` with N = 2 stores, for these agents, only the 4,096 rows every agent shares: opencode puts each agent's
  working folder 29% into the system prompt, so agents in different folders share only the cells before it, and each
  agent's own span starts one conversation. N = 1 stores every conversation's span the first time (3.6 GB/h here) and
  gets about half of what `sessions` gets. For many conversations on one system prompt (a chat front end, agents in
  one folder) N = 2 stores it once.
- Prefix bases save 13% of `evicted`'s writes; under `all` and `sessions` the stored system prefix already does it.
- An earlier version of this table counted a conversation's turn as a new conversation whenever the pool had let its
  last run go unstored, so under `system` each agent's own span reached N and was stored (9 prefixes, 9.4 GB/h on the
  cluster). The pool now remembers the heads of the runs it lets go (the last 1,024), and a turn that continues one
  counts toward nothing and counts as continued for `sessions`. The sim's traces had also cut each continuation's
  overlap short (characters scaled to tokens); a turn that holds the whole previous request now overlaps it exactly.

So `system` stays the default: cheap and safe, but on agent fleets in separate folders it does little. `sessions`
(with a big store) is for agent fleets larger than the KV pool.

### A run's tail past R

Past R each rank holds every nr-th logical page of the attention caches and their indexer keys (its shard, at physical
pages R/64 + j), and every logical page of the bindings that are not sharded (the MTP layer). #37 stored only a run's
pages below R, so a 94K-token session restored 65,536 rows and prefilled the rest (~22 of 26.2 s TTFT, reported by
testers). Now each rank's own store keeps the tail too: ceil(T / nr) blocks of its shard (attention pages and indexer
keys, in the page block's layout), and the T logical pages' replicated parts packed into blocks (83 a block for the full
GLM 5.3: 41,984 B of the MTP layer each). Before the indexer keys were sharded with the latent (#44) every rank held
every page's keys, and the replicated part carried them too (15 a block); that is block format 2. Every rank queues the same copies and the same slots; the bytes in them are each rank's own. A tail
goes back only into a run past R, whole, before the prefill writes the rows past the hit. It is on when the pool's R is
the CP map's (`G6_S2_BELOW` = `G6_CP_R`, the default). The blocks of a tail differ by rank by design, so a store shared
by the ranks would have to leave tails out.

### Lock-step

Every decision is the pool's, made from calls every rank makes in the same order. The head counts and matches at the
event. The grant carries `restored` (1 + the tier) and `counted`, and each worker repeats `count`, `match` and `take` at
its grant and requires the same values and the same pages. The pool digest folds in every tier (free slots, each entry's
id, key, valid rows and slots) and the count table, so a rank whose tiers differ fails at the next event. I/O results
and timing never feed back into a decision.

The same copies take a different time on each rank, and a GPU wait for a peer's rows gives up after ~10 s
(`G6_SPIN_TIMEOUT_CYC`). So after any drain that copied something, every rank passes a barrier before the next round
(`EventEndpoint::sync`, a Sync/Go round on the control links, run from the Driver's `settle` hook after growth). The copy
lists are the same on every rank, so every rank passes the same barriers. With no tiers there is no hook and no frame.

### The copies

Each pool call that can queue copies (reserve, take, grow, retain) leaves a list, in order: spills (page to block) and
moves (block to block) as placement made them, then restores (block to page). A restore may write a page that a spill of
the same call still has to read, and a move reads a slot before a later copy in the list reuses it. The engine runs the
list before anything writes those pages, in `bind` (a grant), `grown` (growth) and `retain` (a promoted prefix), then
calls `io_done`. Until then the pool refuses reserve, take and grow (`PendingIo`), so no later call can reuse a slot a
restore still reads.

On the device a page block is one pool page of every layer binding, main model then draft. One kernel copies a batch of
32 pages between the bindings and mapped pinned host memory, and each block then goes to or from its store
(`engine/runtime/s2/kv_tier.cuh`). The staging memory is registered with the RDMA device, so a KV node reads and writes
it directly.

### Persistence (SSD stores: prefixes and counts)

Off by default: at boot each rank removes its folder (`PATH/r<r>`: blocks, index and the count table) and starts empty,
and logs that it did. `persist=on` keeps it. The reason is a boot that serves wrong output. On 2026-10-12 some cluster
boots of this branch failed the gate (first token and smoke) while others passed, and for a while a store that the
failing boots had written looked like a cause: such a boot could spill wrong blocks and push counts, and a persisted
store would hand both to every later boot. The failures turned out not to involve the store: one came on a boot whose
store had just been wiped, all four came within half an hour (18:37-19:03 UTC), and the 41 boots that followed in a
gate-only A/B, stores on and off alternating, all passed, with the same prefill output on every boot probed. But the gap is
real: nothing checks that a stored block holds what a cold prefill would compute. So a store keeps blocks only within
the boot that wrote them unless asked otherwise. Persist only a store you trust, and clear it (remove the folder) after
a boot that served wrong output.

A `persist=on` SSD store with `--set-id` (the launcher always passes the model id) keeps an index beside its blocks.
The header names what makes blocks compatible: the model id, TP, R, the pool's placement pages, the grid, the page
block's size and layout, the store's size, the block format (`KV_BLOCK_FORMAT` in `engine/runtime/s2/kv_tier.cuh`,
bumped whenever what a block holds changes) and the engine build (an FNV-1a hash of the running engine binary, so a
rebuild that could compute other KV bits never reads the old build's blocks). Another header starts the index empty,
with a log line naming both headers: the old blocks are then unreferenced and their slots are reused. Records are add (an entry: id, key, tokens, valid
rows, slots), drop (an id) and count (a prefix's count; the first SSD store's index holds the table), each with a
length and a checksum, synced as they are written (the counts of an admission at its drain). The engine writes an
entry's blocks before `io_done` appends its add record. A drop record is synced before the entry's slots can be reused.
A record that fails its checksum ends the log, and the file is cut there, so a torn write never names blocks that hold
anything else. Entries load in commit order. After boot, every worker sends the head its loaded entries. The head keeps
those every rank has, in its own order, and every rank keeps exactly that list, so the ranks start in lock-step. The log
is rewritten from the live entries when it holds four times as many records. Evicted runs are not persisted, and neither
are KV node leases.

### Tested on a single GB10 build box

| test | what it checks |
|---|---|
| `cargo test` (`tests/stash.rs`) | two ranks on 60 random traces with two tiers, growth and runs past R: each grant's reused rows hold its prompt's tokens after the copies (spills, moves, restores), the ranks queue the same copies and agree on the digest, no slot is held twice or lost. A five-turn chat counts once, and the third conversation stores the shared prefix and a fourth reads it back. A full tier moves its oldest entry on. A committed prefix survives a reopen. One whose copies were never reported done does not come back, and a torn record and another model's index are ignored |
| `pool_stash_test` (host_tests.sh) | the same through the C interface, with the copies run through `mem:`, an `ssd:` file and a loopback KV node over TCP |
| `control_protocol_test` | the grant's `restored` and `counted` fields round trip in `G6S2E5` |
| m4sim (`G6_KV_TIER_CHECK`), layers 0 and 3 + MTP, four simulated ranks | a reuse hit's pages sent through `mem:`, `ssd:` and a loopback KV node, poisoned on the device, and read back: the printed lines equal the plain run's |
| m4sim (`G6_KV_RANKS_CMP=1`) | below R (G6_CP_R covering the context) every simulated rank's page blocks equal rank 0's, byte for byte. Without that setting each rank holds other pages (the CP map), as expected |

TP=4, TP=3 and TP=6 build. The served path (Service with stored hits, counting in the grant, the boot agreement) has not
run on the cluster: it needs four boxes.

### On the cluster (TP=4, 2026-10-12)

The four GLM boxes, TP=4 home, the release from Hugging Face with the DFlash2 drafter, `--g6env-x G6_POOL_LOG=1`.
Requests went one at a time at temperature 0 with logprobs, unless stated. The replays are the opencode captures in the
agent capture set. Boot-to-boot byte equality is not a check under the default config (the DFlash2 verify shape varies),
so the check of output identity is the first token and its logprob: prefill is deterministic for the same reused rows.

RigMark at `reasoning_effort` low, this branch without `--kv-tier`: code 49.7 tok/s, prose 30.4, 8K prefill cold 1,321
tok/s, aggregate C1 36.3 / C2 51.2 / C4 64.0. Smoke 11/11.

Evict run: turn 3 of four agents one after another, then turn 4 of each. The pool keeps none of the turn-3 runs. A new
agent's first hit takes the previous agent's run for the 4,096 rows of system prompt they share, and the pool dropped
the rest of that run.

| turn 4 TTFT, s | no tier | SSD tier (first build) | KV node + SSD | KV node + SSD, shared-prefix dedupe |
|---|---|---|---|---|
| agent 0 | 13.17 | 4.47 | 4.67 | 4.56 |
| agent 1 | 17.12 | 16.78 (miss) | 2.19 | 2.15 |
| agent 2 | 14.53 | 14.48 (miss) | 2.82 | 2.94 |
| agent 3 | 12.85 | 4.10 | 4.67 | 5.12 |

The first build missed two of four. A hit on another conversation's run took that run without evicting it, so the tier
never saw it. Since ef55a86 that run's rows past the hit go to the tier, and every turn 4 restores. The first token and
its logprob of each turn 4 are the same in both later builds.

Shared prefix: eight conversations whose prompts differ only in the last user message (~12,600 tokens shared). With no
tier each hit on a sibling's run takes 0.85 s. With the KV node tier first and no dedupe, each hit spilled the sibling's
whole run (~200 pages) and took 1.2-3.8 s. With dedupe (32f7bfd) a sibling's run past the stored prefix is 7-9 pages,
and the hits take 0.92-1.06 s once the prefix is stored. The variants' first tokens and logprobs agree across all
builds, including a variant read from the SSD after the pool was flushed and one read after a restart.

Restart (#37 kept an SSD tier across restarts; now that takes `persist=on`): its index brought back 10 (first build) and 2 (later build) shared prefixes. The first request after
boot read its 12,288 rows from the SSD: TTFT 1.32 s and 1.41 s, against 9.6-10.3 s cold.

Copies, rank 0, from the drain log: a 200-page spill to the KV node 725-898 ms (device copy 6 ms), the same to the SSD
158 ms. Restores of 256-384 pages took 162-238 ms, and a spill that also moved 400-500 pages on to the SSD 0.8-1.4 s.
Remote writes run at about 1 GB/s per rank because all four ranks write through the KV node box's single 100G port without PFC: the
clients count 90,000-150,000 packet sequence errors and the KV node box 247,000 out-of-sequence packets and 166,000 CNPs, with no
ack timeouts. The four-box bench through that port: 2.6-3.2 GB/s write and 3.1-3.8 GB/s read per box (about 12 GB/s
total), rows at 13.7 us p50 (6.0 us alone), a 2,048-row gather 1.77 ms (0.85 alone).

Ranks byte-identical: every block of the SSD tier was the same on all four ranks, 512 slots (shared prefixes), 1,895
slots (whole evicted runs, decode rows included, moved on from the KV node) and 1,164 slots in three runs.

The 8-agent replay (both captures, six turns each, all at once) with the KV node + SSD tiers lost the cluster twice
before the barrier above, once after 38 s and once after 35 requests: every rank stopped with "S2 prefill
fabric/attention error". The cause was one rank's drain running seconds longer than the others'. Logging every rank's
drains showed it: the same drain took 5.0 s on one rank and 1.3 s on the other three, and in three later clean runs the
ranks' drain ends spread by up to 2.6, 3.8 and 4.4 s, with the slow rank changing from drain to drain. The KV node's
writes are what vary: four ranks write into the KV node box's one 100G port with no PFC, and the switch dropped 23-27 million
packets on that port in each 8-agent run (90-190 thousand on each GLM box's port).

`G6_KV_DRAIN_DELAY=<rank>:<ms>[:<every>]` (test only: that rank sleeps inside every `every`th drain that copies
anything) makes it deterministic. Without the barrier, 12 s on rank 2's first drain failed the epoch at the next prefill
with the crashes' text on every rank. Ranks 0, 1 and 3 printed `extra 63000c02`, a CP staging wait (`k_cp_pf_wait`,
`0x63000c00 | flag << 8 | peer`) for peer 2 that timed out 25 s into the quantum; rank 2 was still asleep. With the
barrier, ten such 12 s drains on rank 2 in the evict run passed: the ranks started every prefill quantum within 45 ms of
each other, each affected TTFT grew by the 12 s, and the eight turns' first tokens and logprobs were those of the runs
without the delay.

With the barrier and no delay, the 8-agent replay ran clean twice: all 47 turns in 1,469 s and 1,609 s (TTFT median
137 and 167 s, p90 370 and 407 s), every turn's first token and logprob those of `main` without a tier (2,354 s, median
198 s, p90 598 s). The ranks' drain ends spread by up to 5.0 and 7.3 s in those two runs, the second close to the GPU
wait timeout; the barrier took both.

### On the cluster: store policies (TP=4, 2026-10-12)

The same boxes and release, against `main` at #37. First token and logprob as above.

- Stores off: RigMark code 51.0 tok/s, prose 30.6, 8K prefill cold 1,333 tok/s, against `main` 51.1 / 30.5 / 1,277;
  smoke 11/11 and the same outputs. The front end's system span costs 3.1-4.3 ms a request (one more template render).
- `--kv-keep system`, SSD store: the shared-prefix conversations above stored their 12,288-row system prefix. After a
  70K-token request flushed the pool, the first of them again took 1.07 s (9.6 s cold), first token and logprob equal
  to the cold run's.
- A run past R, 93,908 tokens, asked again after a 70K-token request flushed the pool: 74 s with no store, 31.3 s on
  `main` with `--kv-tier keep=evicted` (65,536 rows restored, the rest prefilled), 8.41 s with `--kv-keep all` and the
  tail. First token and logprob the same in all three.
- `system` on the 8-agent replay: all 47 turns' first tokens and logprobs equal to `main`'s without a store; 1,899 s,
  TTFT median 194 s, 9 prefixes stored at 9.4 GB/h a rank. That was the miscount described above (each agent's own
  span reached N); fixed since, and the sim now puts the same trace at 0.1 GB/h.
- The 8-agent replay after the count fix (2026-10-13, main at #43): both captures, six turns each, all at once, one
  50 GiB SSD store a rank, `--kv-system-after 2`. Written and restored bytes are rank 0's store counters.

| `--kv-keep` | wall | TTFT p50 / p90 | tokens generated | written | restored, hits | first tokens = no store |
|---|---|---|---|---|---|---|
| no store | 2,045 s | 234 / 480 s | 63,974 | 0 | - | - |
| `sessions` | 1,576 s | 126 / 407 s | 59,173 | 98 GB/h (43.0 GB) | 29.8 GB, 28 hits | 47/47 |
| `all` | 1,729 s | 131 / 436 s | 69,969 | 97 GB/h (46.7 GB) | 32.3 GB, 36 hits | 47/47 |
| `system` | 2,589 s | 213 / 750 s | 86,690 | 1.2 GB/h (0.9 GB) | 4.3 GB, 9 hits | 47/47 |

  Wall time follows the tokens generated, which vary from run to run with the same inputs (the long turns differ), so
  TTFT p50 and GB/h are the numbers to compare. `sessions` cuts TTFT p50 by 46% for about 98 GB/h a rank; `all` writes
  about the same; `system` writes 1.2 GB/h (9.4 before the count fix) for a smaller gain on agents in different
  folders, as the simulator predicted.
- Rebased onto #41, with persistence opt-in: stores off, smoke 12/12, RigMark code 50.6 tok/s, prose 30.5, 8K prefill
  cold 1,325. `--kv-keep system` without `persist`: a planted index was removed at boot, no index was written, and
  after the 70K flush the system prefix came back from the store in 1.06 s with the cold first token and logprob.
  With `persist=on` the index header ended `format 2 engine <hash>`; after a restart the index brought back the
  prefix and the gate's first request read its 192 pages from the store, first token and logprob as cold.

### Byte-identical ranks, and one copy for all

m4sim's ranks hold identical page blocks below R. The cluster check will say whether the real ranks do: the latent comes
from the all-reduced hidden state, and every rank runs the same kernels on it. If they do, one copy could serve every
rank: 4x the capacity at TP=4, and a quarter of the writes. It changes one rule. Today a restored slot is free as soon
as the call that restored it ends, which is safe because each rank reads only its own store. With a shared store, a fast
rank could reuse a slot (a spill at its next growth) that a slow rank has yet to read in the same event. So freed slots
would wait one event fence, when the head has every rank's ACK, before reuse. That needs a pool call at each event,
deterministic like the others. The writes could be split across ranks (page i by rank i mod TP), and every rank would
read the whole run. Until the cluster check, each rank keeps its own store.

### Still to do

- A check that a stored block holds what a cold prefill computes (a sampled block verified at boot or on restore), so
  `persist=on` can trust a store across boots.

- The DFlash2 drafter's prompt-context taps come from the prefill a hit skips, so a restored run drafts like a pool
  hit does today.
- `status.s2` carries each tier's counters. The front ends do not export them yet.
- One store for all ranks (above): the cluster showed the blocks are byte-identical.
- A KV node shared by four writers needs a port that does not drop: PFC or ECN on the switch port to the node, or a
  rate limit in the client. The barrier keeps the ranks in step, but each slow drain still costs every rank its time.
