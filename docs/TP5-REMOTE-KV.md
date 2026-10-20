# TP5 boot option: T5XR (T5X + remote KV store on a 6th Spark)

T5X stays the default TP5 serve (docs/TP5-SERVE.md). T5XR is a boot option: the same T5X config, plus one KV store tier
in the RAM of a sixth DGX Spark (the KV node, dgx2). When the GPU KV pool evicts a chat, the ranks write its KV
blocks to the node over RDMA. When the chat comes back, they read the blocks back instead of computing the prompt
again. No KV goes to SSD.

- KV node: `./gr8 kv up --gb 106` on dgx2 locks 106 GiB of node RAM (the most that locks with 10 GB left free).
  Each of the 5 ranks leases 21G of it (`KV_STORE=remote:10.98.0.2:7350,size=21G`).
- `KV_KEEP=sessions`: the store keeps a run only after it has continued at least once. A one-turn prompt (a needle,
  a single long document) is never stored, so it cannot push the chats out of the store.
- Persist is off. The store is empty after a KV node or serve restart.
- Build: branch `rkv` (on top of 0574fe5, the T5X source). The T5X build cannot use it: with `STEERING_FORCE=1`, no
  store held any block before the steering fix (below).

## Measured (2026-10-20, 2000 MHz cap, all on TP5 = dgx1 dgx3 dgx5 dgx6 dgx4, KV node dgx2)

Full tables and evidence: `~/legions/tuning/reports/remote-kv-tp5/RESULTS.md` (dev box).

KV node link (step 1): one client writes/reads 13.98 GB/s. With all 5 ranks at once, they share the node's one
200G port: about 14.5 GB/s total, 2.7-3.3 GB/s per rank. Local NVMe gives 11.5 / 13.8 GB/s per box.

Evict/restore A/B (short design: 10 chats x ~22K tokens, about 3.2x a reduced pool, 3 rounds, `KV_KEEP=all`,
steering off unless noted; 30 requests per arm):

| arm | store per rank | restore TTFT p50 / max | vs no store | fast restores | SSD written per box | outputs |
|---|---|---|---|---|---|---|
| A | none | 14.24 / 14.89 s | - | 0/20 | 9-19 MB (OS only) | reference |
| B | SSD 5G | 1.87 / 2.62 s | -87% | 20/20 | ~12.4 GB (KV) | = A 30/30 |
| C | remote 1G + SSD 1G | 14.69 / 15.47 s | none | 4/20 | ~11.5 GB | = A 30/30 |
| D | SSD 2G | 14.33 / 15.05 s | none | 0/20 | ~12.4 GB | = A 30/30 |
| E | remote 21G + SSD 1G | 2.02 / 2.80 s | -86% | 20/20 | 4-11 MB (OS only) | = A 30/30 |
| F | remote 21G | 2.03 / 2.84 s | -86% | 20/20 | 5-16 MB (OS only) | = A 30/30 |
| AS | none, forced steering | 14.30 / 14.89 s | - | 0/20 | 9-17 MB (OS only) | steered reference |
| FS | remote 21G, forced steering (build 864c387) | 2.06 / 2.94 s | -86% | 20/20 | 5-12 MB (OS only) | = AS 30/30 |

F vs B: node RAM restores 0.16 s slower than local NVMe (drain 366 vs 162 ms, the shared node port), and writes no
KV to SSD; B writes ~12.4 GB per box per 3.5-minute run. Small tiers (1-2G per rank) gave no gain and still wrote to SSD.

Tier 2 on the full T5X config (T5XR, forced steering, `KV_KEEP=sessions`, CAP 617,536):

| check | T5X | T5XR | result |
|---|---|---|---|
| prefill 8K / 32K, prose decode 1 chat (tok/s) | 1,572 / 1,569 / 39.8 | 1,570 / 1,545 / 39.5 | same, inside the +-2% rerun spread |
| needle 500K | - | answer found (500,460 tokens, first token 477 s) | pass |
| lowest free memory, any box | 2.66-2.67 GB | 2.45-2.48 GB | pass (floor 1.5 GB); the store costs ~0.2 GB per rank |
| tier 2 (build 864c387): 4 chats x ~97.5K, turn 3 after eviction | cached | 0 GB stored: every steered run refused | FAIL, fixed in 5c9fdc4 |

Retest (build 2c1a4a7): 4 chats x ~97.5K tokens, 3 turns; before turn 3 a 355K-token filler prompt pushes chats out of
the GPU pool. The twin is T5X with no store.

| pair | turn 1 | turn 2 | turn 3 |
|---|---|---|---|
| T5X vs T5X, 4 chats at once (noise floor) | 4/4 | 2/4 | 0/4 |
| T5XR vs T5X, 4 chats at once | 4/4 | 4/4 | 0/4 |
| T5XR vs T5X, one chat at a time (T5X chats cached) | 4/4 | 4/4 | 4/4 |

- Turn 3, one chat at a time: all 4 chats restored from the KV node. First token 1.26-5.51 s, against 4.52-4.67 s
  for the cached T5X chats; a full re-prefill takes about 67 s.
- Rank 0: 13.4 GB written to the node, 25.6 GB read back, 0 dropped, never full. The store refused only one-turn prompts.
- Store use: 5 stored chats of ~97.5K tokens took 1,898 blocks of 3.5 MB per rank (~6.7 GB), so 21G per rank holds
  about 15 such chats (estimate from the rank-0 log).

## Steering fix (branch rkv, local commits)

With `STEERING_FORCE=1` every request is in a non-stock cache namespace, and the stores skipped non-stock runs.

- `864c387` A store entry carries its 70-byte cache namespace. An evicted steered run is stored whole under it and
  matches only the same namespace. Prefix kinds and the persist index stay stock-only. The rank digest covers the
  namespace.
- `5c9fdc4` A steered prompt that continues a run of its own namespace counts as continued, so `KV_KEEP=sessions`
  (min_hits 1) stores it. Before, every steered run had 0 hits and was refused.
- `2c1a4a7` The write limit earns credit on a steered prompt's length, as for a stock prompt, not on its whole grant.
- Also on the branch: `5f5b862` (copy batch fixes: a batch of 1 advances; one block is never written twice in a batch),
  `c5f674d` (drain trace in the rank logs).

Tests: `cargo test` in engine/kvman-rs and `scripts/host_tests.sh` pass (pool_namespace_test, pool_stash_test).

## Boot and stop

Files: `docs/tp5-remote-kv/T5XR.env` (the T5X env with `G6_HOME` = the rkv tree and two KV lines) and
`docs/tp5-remote-kv/serve-t5xr.sh` (a draft of serve-t5x.sh for T5XR; not installed on dgx1).

1. Build the rkv branch (TP5 widths: `G6_WIDTHS="4 5" F12_WIDTHS="4 5"`, Rust 1.94.1) on a box that is not serving,
   and copy the tree to `/root/rkv/gr8ns` on the five boxes. Copy `g6front-rs` too, with `scp -p`: the files in `bin/`
   must be mode 755, or the boot fails.
2. Stop T5X: `ssh root@100.75.70.62 bash /root/tp5/serve-t5x.sh down`.
3. KV node on dgx2 (a tree with `bin/kvman-remote` at `/root/rkv/gr8`; tested with the c5f674d node build):
   ```
   bash /root/ablit/memclean.sh
   cd /root/rkv/gr8 && systemd-run --quiet --collect -p LimitMEMLOCK=infinity --unit=rkv-node --setenv=PATH=$PATH \
     /bin/bash -c "cd /root/rkv/gr8 && exec ./gr8 kv up --gb 106 --mentat none --node-ip 10.98.0.2 --foreground > /root/rkv/node-up.log 2>&1"
   ./gr8 kv status     # ready: "106.00 GiB locked, 106.00 GiB free"
   ```
4. On all five: memclean and `nvidia-smi -lgc 0,2000`, then start each rank as for T5X, with `-p LimitMEMLOCK=infinity`
   and the T5XR env: `cd /root/rkv/gr8ns && ./gr8 glm53 up @/root/rkv/T5XR.env`. Ready: `/health` 200 on dgx1 and
   max_model_len 617,532 (about 50 s). The rank-0 log shows `s2 store 0: KV node 10.98.0.2:7350 ... RDMA`.
5. Stop: `./gr8 glm53 down @/root/rkv/T5XR.env` on dgx1, wait until no rank process is left, then on dgx2
   `cd /root/rkv/gr8 && ./gr8 kv down`. Then `serve-t5x.sh up` brings T5X back.

The draft script does steps 2-5: `serve-t5xr.sh up` (refuses while T5X or any g6m4 serve runs), `down`
(`KEEP_NODE=1` leaves the node up), `status`.

## Limits

- The store helps only when it holds the working set. 21G per rank held all chats in every test. 1-2G per rank gave
  0-4 of 20 restores and still wrote KV to disk (SSD tiers).
- Restores share the node's one 200G port: a 5-rank drain is ~366 ms against ~162 ms from local NVMe.
- With several requests at once, greedy output on this build changes with batch order, with or without a store
  (T5X vs T5X: 0/4 equal at turn 3). Restored KV is exact: one request at a time, T5XR output equals the cached T5X
  output byte for byte.
- SSD: T5XR has no SSD tier. The KV goes only to node RAM; measured SSD writes were OS-level only (4-16 MB per box per run).
- The KV node is a sixth box. While T5XR serves, dgx2 cannot do other work. Its RAM is locked.
- Persist is off. A restart of the node or the serve empties the store.
