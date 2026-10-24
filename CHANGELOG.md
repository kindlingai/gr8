# Changelog

## 2026-10-24: KV-node leases that outlive a restart (persist=lease) and ./gr8 <model> resize (D-331 M4)

- `--kv-store remote:HOST:PORT,size=N,persist=lease`: the KV node, run with `--holdover-s S`, parks the lease for S
  seconds when the engine closes it, and the next engine takes it back with its blocks. The engine's index of that
  store (sessions too) lives in `/dev/shm`, and an orderly quit stores the pool's runs first. Off by default; not with
  `group=` yet.
- `./gr8 <model> resize --restart idle [OPTIONS]`: waits for no running request, then down and up with OPTIONS (a new
  `--cap`), so the next turn of a session restores from the node. `RESTART=off` (the default) refuses.
- Tested on loopback only (engine/kvman-rs/REMOTE-TIER.md, Persistence (KV-node leases)). Not measured on a cluster.

## 2026-10-24: a KV node shared by clusters (group leases, D-331 M3)

- `--kv-store remote:...,group=NAME[,cap=N][,weight=N]`: the stores of one cluster form a group lease on a KV node,
  and capacity moves between clusters by whole segments (B3, ported onto main). `engine/kvman-rs/REMOTE-TIER.md`,
  Sharing a node.
- Grants and gives are transactions with an id and an absolute target: a stale or repeated GROW / SHRINK is refused or
  a no-op, a grant one rank took is held for the others, and given segments go free only once every rank gave.
- A group opening on a full node asks the others down toward their floors and waits for them; `kvman-remote
  --group-plan FILE` reserves planned floors. Empty segments are given at once. A give to a group of another rank
  count strands nothing. A tier asks for room for an entry too big for it.
- `KV_SHARE_WHEN=idle` (default) gives only while the engine is idle; `always` whenever asked.
- Wire change: REPORT replies carry the transaction id, GROW / SHRINK take it (`E_STALE` = 8). Both sides of a node
  must run this build.

## 2026-10-23: G6_TR3_W2ALL, the window-2 fc1 split on every MoE layer, default on at TP=4

- `G6_TR3_W2ALL=1` applies the window-2 fc1 split (KS1x2, WG1) to every MoE layer at M > 1, not only the layers the k2
  gate picks. Bit-exact (`row_budget_test --only k1`: the same FNV on all 54 K1 lines off and on). Launch knob
  `TR3_W2ALL`, 1 at TP=4, unset at TP=3 / TP=6; 0 turns it off. Measured on four GB10 at TP=4 (2000 MHz cap): prose
  decode +1.3%, code c4 +3.4%.

## 2026-10-23: GLM-5.3-Flash defaults to a 262,144-token context (CAP 262,208)

- `./gr8 glm53f` now defaults `CAP` to 262,208 (was 33,280) at TP=4 and TP=2: a 262,144-token request plus a page for
  the guard rows, so `max_model_len` is 262,206. Only Flash's 11 DSA layers and the MTP layer keep KV, 19,968 B a token
  a rank, and without CP every rank holds the whole context: about 5.2 GB a rank, computed from `engine/attn/kvfmt.h`
  (docs/FLASH.md, Context).
- Not measured yet: the boot's memory headroom at this `CAP`, and the time of a 262K prefill. A prefill that runs past
  the shim's 900 s `--stall-timeout` fails. Set `CAP=33280` for the old pool.

## 2026-10-21: the TP=4 fp8 KV pool grows to 2,280 pages a rank (CAP 485,376)

- The default TP=4 `CAP` is now 485,376 (was 408,832): 2,280 physical pages a rank, up from 1,981. The memory comes
  from the MTP draft prefill sharing the target prefill's scratch (`G6_PF_DRAFT_SHARE`, on since 2026-10-20). With
  `--share-limit 65536` it is 387,072. `KV_FORMAT=int4` keeps its 2,390 pages (CAP 513,536) until it is re-measured.
- Measured on the TP=4 cluster, same build, three 4-agent replays each: 1,011 / 1,109 / 1,174 s and 39.6-41.8 completion
  tok/s, against 1,084 / 1,100 / 1,171 s and 38.1-38.9 at 1,981 pages; TTFT median 16.8-20.2 s against 14.5-18.7 s.
  MemAvailable at serving is 1.1-1.5 GiB a box (2.8-3.1 GiB before), with no swap or OOM kill in any run.
- Set `CAP=408832` for the old pool.

## 2026-10-21: G6_DFL_REUSE defaults on when serving DFlash2 with MTP

- `G6_DFL_REUSE` now defaults to 2 when the engine serves DFlash2 with MTP (off elsewhere, and any explicit value still
  wins). A request whose prefix-reuse hit skipped into the drafter's window, or that runs alone on MTP, keeps the
  DFlash2 drafter instead of running MTP for its whole life. That is most later turns of a conversation.
- Why: on a warm cache (RigMark run 2, every prompt a prefix hit) code held at 51.1 tok/s with it on against 44.3 off,
  +15%, with prose, multi-agent and cold prefill unchanged. Drafts only, so output is bit-identical (KL mean 0.0,
  top-1 1.0). This closes the former "which requests get the DFlash2 drafter depends on timing" issue.
- Re-checked on 2026-10-21 on the TP=4 build of the remote-rows release: warm code 58.5 tok/s with it on against 48.9 off
  (+19.6%), cold code 58.7 against 56.0.
- `G6_DFL_REUSE=0` restores the old behaviour.

## 2026-10-20: remote attention rows

- `--kv-rows STORE,size=N` (experimental, off by default) puts each rank's attention rows past `SHARE_LIMIT` in a rows
  store, a KV node (`remote:auto`) or memory, as their pages fill. After each indexer selection the rows it picked are
  fetched into resident slots, under the candidate merge, so decode reads them as if they were local.
- `local=PAGES` releases a page once its rows are in the store: a session's tail cycles through that many local pages a
  rank, and the S2 pool gives the MTP drafter its own tail pages so none is ever released under it. A prefill chunk
  reads the released pages it needs from the store. A finished session with released pages is retained with its store
  records, so its next turn adopts them and prefills only the new rows; block stores (`--kv-store`) copy at most its
  first `SHARE_LIMIT` tokens, and an evicted one's records are freed.
- `place=PAGES` caps the main model's KV pages a rank, so `--cap` may pass the context they would hold. `size=` is the
  per-rank lease, and a session waits until its tail fits it. `fmt=int4` stores 416 B rows for 656.
- A long prefill no longer trips the shim's stall timeout: rank 0 writes an `f` line after every prefill quantum, which
  the shim takes as life. When the shim does give up on a request (engine silent, unreadable resp), it now CANCELs it
  instead of leaving the engine to prefill and decode it to the end.
- A prefill reads only the staged layer's rows of each released page (`G6_RT_PF_SLICE=0`: the group's whole block).
- The MTP draft prefill borrows the target prefill's tensor-core scratch by default (`G6_PF_DRAFT_SHARE`; 0 = its own
  root), about 2 GB a rank at a 655,360-token pool, output unchanged. With it, `place=1600` at `--cap 655360` leaves
  stock serving's headroom on TP=4.
- On the TP=4 cluster: a 41,399-token request with 40 local pages a rank finished in 39 s with the same output as
  without release; 35 pages were released and at most 32 of the 40 were in use. A 445,855-token prompt prefills in
  613 s, and its next turn answers in 11 s from the retained records.

## 2026-10-20: each KV store can keep its own kind (`keep=`)

- `--kv-store STORE,size=N,keep=system|sessions|all` sets what that store keeps; a store without `keep=` takes
  `--kv-keep`. Placement already went to the first store that takes an entry's kind, so
  `ssd:...,persist=on,keep=system;remote:auto,size=4G,keep=all` keeps system prompts on the SSD across restarts and
  every other evicted run on the KV node. Before, every store took `--kv-keep`, and the deprecated `--kv-tier`'s
  per-tier `keep=` was parsed and then ignored.
- On the TP=4 cluster with that config: a 153,580-token session evicted to the KV node came back in 4.1 s (136 s
  cold), each rank restoring 2,872 pages (its first 32,768 tokens and its shard of the tail) in 0.55 s.
- `docs/kv-tiers.svg` draws the tiers: replicated below `SHARE_LIMIT`, sharded past it, and the stores below the pool.
## 2026-10-18: S2 sizes three things by the DFlash2 drafter's verify width, not by a block of 8

- A DFlash2 round's MTP first form runs on the accepted prefix + 1 rows (1 .. `dfl_k` + 1); the boot now captures every
  one of those widths explicitly instead of relying on the prompt chunk and the verify widths to cover them.
- `sc.max_rows` (one step's bound) is the largest of a round (sessions x (`--mtp` + 1)), a prompt chunk and the DFlash2
  verify width (`dfl_k` + 1).
- The per-request acceptance histogram may have up to 16 bins (was 8).
- With the block-8 DFlash2 drafter nothing changes: the captured widths, the row bound and the histogram are those of
  before. A drafter with a wider block (a block-12 one was served with these fixes) boots and serves; serving one also
  needs drafter-side changes that are not part of this.
## 2026-10-18: CP decode: the remote head groups as one attention call (G6_CP_ONECHAIN, off by default)

- Past R each rank runs the other ranks' head groups over its own keys (`launch_attn_cp`, phase 1): one `k_vget`, one
  `g6_attn_cp_group` (scores, softmax, context parts, merge) and one `k_vput` per remote group, 18 launches a layer at
  TP=4. `G6_CP_ONECHAIN=1` (launch knob `CP_ONECHAIN`) gathers the three groups' q / rope side by side
  (`k_vget_heads`), runs one 48-head `g6_attn_cp_groups` call whose merge writes each group's partial into its own
  block, and publishes the three partials in the S1 order: 8 launches a layer.
- Every kernel indexes a (row, head) as `row * heads + h` and batches heads in groups that divide 16, so each head runs
  the 16-head kernel's code on the same operands in the same order: the published partials are byte-identical
  (`engine/attn/tests/cpgroups.cu`, one GPU, built by hand). TP=4 only and not with `G6_ATTN_FLIP` (the boot stops
  with a message otherwise); about 1.3 MB per CP fabric row of device memory, printed by rank 0.
- Measured on four GB10 at TP=4 (R 65536): decode +3.6 / +2.5 / +1.7 % at 70K / 80K / 90K-token prompts and +9.5 % at
  ~120K, the same output SHA as off; below R nothing changes.

## 2026-10-18: decode buckets: nine slots, so a SHARE_LIMIT off the doubling keeps the 2048 bucket

- The decode graphs are captured per context bucket (2048, 4096, ... below the capacity, the capacity, and CP's R when it
  is not already a boundary). With eight slots a limit such as 98304 filled the list and `context_buckets` merged the
  smallest bucket away, so every short-context step ran on the 4096 graph. `NBK` is now 9 and the doubling stops at
  seven buckets, as it did when the list was full: at every TP default (R a power of two among the doublings) the list,
  the graphs and the outputs are those of before; an R off the doubling takes the ninth slot (one more bucket of step
  graphs). `test_context_buckets` checks both.

## 2026-10-18: --share-limit takes any multiple of 64 from 4096

- The launcher refused a `SHARE_LIMIT` that is not a power of two. The engine does not need one: `context_buckets`
  makes any `G6_CP_R` that is a multiple of a KV page a decode bucket boundary, so a graph never mixes the replicated and
  CP paths. The rule is now a whole multiple of 64 from 4096; every limit the old rule took gives the same engine line.
- A limit off the doubling trades longest context for decode speed between the two limits: at TP=4, 98304 gives CAP
  212,224 (at the default 1,981 pages a rank), and on four GB10 it decoded +8.3 .. +13.6 % faster than 65536 at 70K-90K
  token prompts (those sessions keep their KV replicated and skip the CP attention path); short prompts level.
## 2026-10-19: per-request step counters in usage (`return_step_stats`, opt-in)

- A chat or completions request with `"return_step_stats": true` gets `usage.step_stats`: its verify rounds, draft and
  accepted tokens (and per draft position), the tokens the engine generated, and its server-side prefill and decode
  time. These are the numbers the request added to `/metrics` (`vllm:spec_decode_*`, `vllm:request_prefill_time_seconds`,
  `vllm:request_decode_time_seconds`), so a benchmark client can split decode speed into steps and acceptance per
  request at any concurrency instead of differencing `/metrics` around one request at a time. Streaming: in the last
  chunk (`stream_options.include_usage`). Without the field the responses are unchanged.
- The S2 shim adds each request's counters to its `finish` frame (`spec`, `timing`: optional, PROTOCOL.md); the engine
  is unchanged.
## 2026-10-18: the drafter choice: a frozen cost table and K-stop-aware MTP pricing (G6_S2_DSEL_MODEL, off by default)

- Since the confidence stop (`G6_MTP_KSTOP`) a request's solo MTP rounds verify 1..3 drafts chosen per round, but the
  drafter choice (DSel) priced every MTP round as `--mtp` drafts with fixed drafting constants. `G6_S2_DSEL_MODEL=1`
  (launch knobs `DSEL_MODEL`, `DSEL_TABLE`) prices both drafters from one frozen cost table, `G6_S2_DSEL_TABLE` (verify
  ms `v1`..`v8`, drafting ms `d`, `m`, `e`, a probe's extra `p`, and optionally the MTP rounds' prior `g`, `k`, `n`, the
  estimates' prior weight `w` and `r`), and an MTP round as the stop runs it: 1 + c_1 (1 + g) tokens over the mean table
  cost of the request's MTP rounds, both shrinking toward the prior until the request has MTP rounds.
- `r`: DFlash2 is priced from the request's DFlash2 rounds and scored probes as they ran, that evidence aged by `r` per MTP
  round, so a structured request that a dip moved to MTP comes back.
- Only the choice reads the table: `G6_SPEC_TABLE` still prices the DFlash2 rounds' own draft count. The table's canonical
  form is in the S2 runtime signature. Off, the steps are unchanged (`batch_dselmodel_test`).
- `G6_S2_TIMELINE` also prints the mean GPU span per step kind and row count, and `tools/s2_dsel_table.py` turns one
  boot's timeline and `# kstop` lines into the table; with `G6_MTP_LPLOG` and the model on, rank 0 prints a `# dsel round`
  line per solo round.
- Measured on four GB10 at TP=4 (with the solo confidence stop on, `G6_S2_DSEL_PROBE=32`): RigMark c1 prose +1.8 %, code
  +3.0 %, structured -3.6 % against DSel with the MTP round priced at 2.9 ms.
## 2026-10-18: a per-session MTP confidence stop in shared rounds (G6_MTP_KSTOPB, off by default)

- When several sessions decode together, a shared round drafts `--mtp` drafts for every session and verifies all of
  them, however unlikely a session's later drafts are; the solo stop (`G6_MTP_KSTOP`) does not act there. With
  `G6_MTP_KSTOPB=1` (launch knobs `MTP_KSTOPB`, `MTP_KSTOPB_KMIN`, `MTP_KSTOPB_TAU`, `MTP_KSTOPB_TAU_S`) the chain still
  drafts K for every session, and each session q verifies its own k_q: the most j with p_1 x .. x p_i >= t_i for every
  i <= j (the MTP head's log-probabilities, the same bytes on every rank), at least `G6_MTP_KSTOPB_KMIN`. Thresholds
  `G6_MTP_KSTOPB_TAU` (default 0.6, 0.88; up to seven) at every session count, or `G6_MTP_KSTOPB_TAU_S<n>` for rounds
  of n sessions.
- The main step's width is the sum of the sessions' k_q + 1; the boot captures a main graph for every width the rule
  reaches (about 115 MiB each over 8 buckets), `G6_MTP_KSTOPB_WIDTHS` limits them (a round then verifies the next most
  likely drafts up to a captured width). `G6_MTP_LPLOG=1` adds `# kstopb` and `# kstopb_ms` lines per shared round.
- Drafts only: the target decides every token; the verify's row count varies, as with the solo stop. The policy is in the
  S2 runtime signature. Off, and on with zero thresholds, the planner's steps are unchanged (`batch_kstopb_test`).
- Measured on four GB10 at TP=4 (`--mtp 2`, thresholds per session count): ABBA over six blocks, c2 +1.5 %, c4 +3.7 %
  decode; c1 unchanged.
  and fc2 n-split (NS2) per step row count from a table, `M:KS1xWG/NS2` entries (`a-b` ranges). An fc1 shape is taken
  only where KS1 x WG equals the row class's own, and NS2 may be any divisor of 384, so the partials, their k ranges and
  fc2's summation order are those of the built-in split: same bytes at any accepted table. A malformed table fails the
  bind with a message.
  `prefetch.global.L2` per KB granule over its slot's fc1 and fc2 ranges at the start of the launch, so the address
  translations of the later CTA waves and of fc2 are walked while fc1 runs. A decode step's routed experts touch about
  5,300 2 MB pages a rank at TP=4, more than the GPU's translation reach. No output change.
- `engine/kernels/tests/k1_audit.cu` (built by `scripts/build.sh`, run by hand on an idle GB10): the routed MoE phase on
  synthetic exlr8 v2 layers in the TP=4 rank shape, all 75 MoE layers resident, against a plain read of the same bytes;
  log-probabilities identical on 75 prompts; decode +2.0 % prose, +1.7 % code, +1.5 % structured at c1.
## 2026-10-18: decode: a paced L2 prefetch of the next weights in the step's idle-DRAM windows (G6_L2PF, off by default)
- A decode step leaves DRAM idle in its collective waits (the two all-reduces per layer, the `G6_QA_SPLIT` /
  `G6_IDXQ_SPLIT` gathers) and through most of the latency-bound attention, while the GEMVs around them start cold.
  `G6_L2PF` (launch knob `L2PF`, a window mask) has one warp on a high-priority side stream prefetch the layer's next
  weights into L2 in those windows: 8 KiB `cp.async.bulk.prefetch.L2` chunks from a token bucket (`G6_L2PF_GBPS`, 175
  GB/s, in waves of `G6_L2PF_WAVE`), not a burst, so dependent loads of the attention are left alone; a second warp
  polls the collective's tails and ends the window once every peer has arrived.
- One queue per layer in read order (x_in, idx_q, q_b, UK, UV, o_proj up to `G6_L2PF_OPMB`, router, shared fc1);
  windows: 1 = the MLP all-reduce wait, 2 = the gather waits, 4 = the attention, 8 = the attention all-reduce wait. The
  side stream is forked per window and joined at the end of the step part, so captured graphs hold the forks as edges.
  Unset: no stream, no launch, the graphs are unchanged.
- A prefetch writes nothing: same bytes out. `engine/kernels/tests/l2pf_bench.cu` (built by `scripts/build.sh`, run by
  hand) byte-compares the decode layer sequence's outputs per arm and times it per layer.
- Measured on four GB10 at TP=4 (`L2PF=15`): T=0 top-20 log-probabilities identical on 75 prompts; decode c1 +2.7 %
  prose, +2.3 % code, +2.7 % structured; `l2pf_bench` -10.1 us a layer at 3 rows, -38.6 us at 12.
## 2026-10-19: S2: a row cap for steps past SHARE_LIMIT that also sizes the CP fabric (G6_CP_STEP_ROWS, off by default)

- Since #94 sessions past `SHARE_LIMIT` (R) share batched S2 steps, so the CP fabric and the CP graphs at every bucket
  past R are sized for the widest S2 step: `SESS` x (`MTP` + 1) = 24 rows at 8 sessions / MTP 2, 1.358 GB pinned and
  registered per rank at TP=4 (8 rows: 0.453 GB). `G6_CP_MMAX` can only raise that.
- `G6_CP_STEP_ROWS=n` (launch knob `CP_STEP_ROWS`) sizes the fabric for n rows, and `Driver::step` forms rounds so
  that no step past R carries more than n: while a session past R decodes and the decoding sessions' rows do not fit,
  a round takes at most n / (`MTP` + 1) sessions, those past R first, and the others sit it out the way a grow stall
  does (their state does not change); every decoding session gets one round per cycle. Rounds without a session past
  R keep the full batch. n is at least the rows of a solo or prompt step past R (the boot names the minimum) and is
  clamped to the widest S2 step. The boot does not capture the main widths above n at the buckets past R.
- `G6_CP_STEP_SEG=1` (launch knob `CP_STEP_SEG`): a round past R takes only sessions past R, so a session below R
  never runs the CP attention because a long session is resident.
- Sessions past R give the same tokens as without the knob; a session below R left out of a past-R round runs that
  round on the replicated path instead of the CP path (the LSE-merge difference #94 introduced, ~1e-5). Set, the
  knob joins the scheduler digest, the bootstrap agreement and the S2 runtime signature; unset, nothing changes.
- Host test `engine/runtime/s2/tests/cp_rows_test.cpp` (in `scripts/host_tests.sh`). Measured on four GB10 at TP=4
  and the default CAP: with the 24-row fabric the head was down to ~0.5 GiB MemAvailable while serving and hit
  NV_ERR_NO_MEMORY under load; with `CP_STEP_ROWS=8` it had 3.33 GiB (the other boxes 2.48-2.77). T=0 output
  unchanged at short context; a ~71K-token request plus three short ones ran concurrently with the cap engaged,
  without errors.

## 2026-10-20: tool calls stream as they are written

- The front end streams a tool call's name and arguments while the model writes them, instead of sending the whole call
  once the reply ends. This applies to a call in the answer, outside a code fence, that starts the answer or follows text
  with no space between, which is how GLM-5.3 places a call. String arguments (file contents, commands) stream as
  generated; typed ones (integers, booleans, objects) go out once each value ends. Other calls (after a space or
  newline, in reasoning, in a fence) are held until the reply ends, as before, so a call quoted in prose stays prose.
  A streamed call is a call even if prose follows it.
## 2026-10-18: SHARE_LIMIT default 32768 at TP=3

- TP=3 now defaults `SHARE_LIMIT` to 32,768 (was 65,536), the mirror of the TP=4 change. Two or three long (32K+)
  chats stay resident and decode together instead of the second queueing; longest context rises 241,600 -> 307,136 at
  the same 1,941 pages a rank. Measured on three GB10 at TP=3: two 38K-token chats run concurrently (both finish
  within 1 s of each other), smoke 14/14, decode/prefill headline unchanged. `--share-limit 65536` restores the old
  behaviour.
## 2026-10-18: SHARE_LIMIT default 32768 at TP=4

- The TP=4 serving default `SHARE_LIMIT` is now 32,768 (was 65,536). Each session replicates its first 32,768 tokens on
  every rank instead of 65,536, so two or three long (32K+) chats stay resident and decode together instead of the second
  waiting for the first to finish (CP batch views, landed in #94). The longest context rises to 408,832 tokens (was
  310,528) at the same per-rank KV memory. Trade: decode from 32K to 64K of context is context-parallel now, a few percent
  slower per step for a single such session; short-context decode and prefill (the RigMark/pbench headline numbers) are
  unchanged. `--share-limit 65536` restores the old behaviour. TP=3 and TP=6 defaults are unchanged.
## 2026-10-17: a fabric pool at boot and TP groups at runtime (EL6), Fabric::close

- `engine/comms/fabric_pool.h` (`FabricPool`): every box that may join a layout opens one control link to the pool head at
  boot and keeps it; the head checks that every box was started with the same pool. A group is an ordered list of pool
  indices (its first member is rank 0), announced by the head and acknowledged by every box, each learning its rank or
  that it is idle, plus the group's fabric port base, S2 port and S2 head. Group ids alternate between two port ranges so
  the next layout's group can be built while the current one runs. `group_fabric()` builds a member's `FabricConfig`
  (mesh only; rings are refused). The engine does not call the pool yet: nothing listens on the new ports until the
  engine runs on a group.
- `Fabric::close()` frees a mesh fabric (every QP, MR, CQ, PD, device context, the pinned region and the trace buffer, in
  that order) after draining the proxy; the fabric can then be init'ed again. `fabric_live()` counts the live objects for
  the leak gate. The destructor still only stops the proxy.
- Each fabric rotates its own barrier port (before, one static counter was shared by every fabric in the process; a
  barrier still uses one of its own fabric's `port+1..port+3`), and each mesh QP draws a fresh start PSN instead of a
  fixed function of (rank, peer), so a late packet of a dropped group cannot match a recycled QP number at its expected PSN.
- Tests: `fabric_pool_test` (host, loopback; in `scripts/host_tests.sh`) and `fabric_group_test` (built by
  `scripts/build.sh`, run by hand on the cluster: today's fabric against the all-boxes group, bit-identical digests, then
  groups created and dropped in a loop with the verbs objects, pinned bytes and descriptors checked back to baseline).
## 2026-10-17: several sessions past R stay resident together (CP batch views); G6_S2_DECODE_PF, G6_S2_ADMIT_HOLD, G6_S2_DECODE_ROUNDS (default off)

- A session past `SHARE_LIMIT` (R) no longer needs the KV pool to itself. On the pool's page classes a split run past R
  shares the pool with other split runs and with runs below R (`engine/kvman-rs` `pool.rs`), and the rows of several such
  sessions run in one batched step through the layers' CP batch views (`s2.cuh`, `rank.cuh`): a step whose bucket is past
  R takes the CP attention block on those views, with each row's own block table. Before, a second long agent chat queued
  until the first ended. Not behind a knob: it applies whenever more than one session is past R. Batched CP attention
  equals the per-session calls bitwise per rank (`s2_attn_test` CP cases, `cp_batch_cpu_test`); steps below R are unchanged.
  Each past-R session still replicates its first R tokens on every rank, so at TP=4's default `SHARE_LIMIT` of 65,536
  (1,024 of a rank's 1,981 pages) a second one does not fit and waits as before; with `--share-limit 32768` two 72K
  sessions sent at once were both resident on the four-box cluster (one queued behind the other on the previous build).
- `G6_S2_DECODE_PF=<rows>` (default off): while any session decodes, a prefill quantum is at most this many rows on one
  lane, so decoding sessions get a round sooner. 0 or a multiple of 64, at most `--prefill`.
- `G6_S2_ADMIT_HOLD=1` (default off, head only): queued admissions are granted one after another at one held boundary
  instead of one per notice (`Commit::hold`, `event_gate.h`).
- `G6_S2_DECODE_ROUNDS=<K>` (default off): while a prefill waits and sessions decode, K decode rounds run per prefill
  quantum instead of one.
- Control version 6: the commit frame gains the hold field, so the frame magic is `G6S2E6` and the bootstrap agreement
  carries ` ctl 6`. Ranks on different builds refuse at bootstrap, not at the first event; deploy the same build on every rank.

## 2026-10-17: the front end exports KV, TTFT and per-request prefill and decode timing

- `/metrics` on the S2 serve had no `vllm:kv_*`, `request_*_time_seconds`, `time_to_first_token` or
  `cache_config_info` series, so dashboards that read them showed no decode or prefill rate, KV use or slot count. The
  shim now measures them from the requests' resp lines and `status.s2` (`kv`, `latency` in the `metrics` frame) and the
  front end renders them (see `frontend/README.md`, Metrics mapping).
- A series is rendered only when the engine sent its field. The S2 admission line now includes the reused prompt-row
  count. The shim uses that count for measured prefix-cache hits and computed prefill tokens, instead of reporting a
  false zero or counting reused rows as compute.
- `g6s2shim.py --sessions N` (the launcher passes `SESS`) is the `kv_cache_max_concurrency` the shim reports.

## 2026-10-17: the decode-perf flags default on in the serving config

- `G6_DRAFT_PAR`, `G6_QA_SPLIT` / `G6_IDXQ_SPLIT` / `G6_MTP_SPLIT`, and `G6_MTP_KSTOP` now default on where their
  preconditions hold (DRAFT_PAR and KSTOP: `--serve` + DFlash2 + `--mtp` [2+ for KSTOP]; the split group: TP>1, and
  MTP_SPLIT also `--mtp`). A bare or TP=1 boot is unchanged, and any knob still overrides to 0.
- Measured on four GB10 at TP=4 (pbench, single stream) against all off: all four on gave +7.0% decode (all_decode
  30.85 -> 33.0 tok/s), no prefill regression, acceptance 0.513 -> 0.565. Alone: DRAFT_PAR +3.7%, QA_SPLIT +4.7%,
  KSTOP +2.3%. The split group and DRAFT_PAR are bit-exact by construction; KSTOP changes only draft count (the target
  decides every token). `G6_DFL_REUSE` stays off (its gain is multi-turn only and did not show here).
## 2026-10-16: a decode step timeline for S2 (G6_S2_TIMELINE, a measurement)

- `G6_S2_TIMELINE=N` (every rank, off by default) splits the time between decode steps into GPU work and host round
  trip. Around each decode step launch (main, first-form draft, chain, DFlash2 context-only and query steps) the host
  stamps its clock and records a CUDA event before and after the graph launch; after the fence it adds the step's GPU
  span, input prep, launch call and fence wait to per-kind totals, and the GPU idle since the previous step's end plus
  the host time since the previous fence to per-transition totals. Every N main steps each rank prints them.
- Prefill quanta, mode switches and the drafter's prompt fill cut the chain; gaps over 20 ms count as breaks. With
  `G6_DRAFT_PAR` the drafter step on its own stream is timed apart (`dfl_ctx_par` / `dfl_query_par`).
- It sizes the host round trip that a device-side accept and launch could remove. Unset, it records nothing.

## 2026-10-16: the DFlash2 drafter step under MTP's first form, on its own stream (G6_DRAFT_PAR, off by default)

- In a round of a request that decodes alone with DFlash2, the drafter step ran after MTP's first form (and after the
  MTP chain on the drafter choice's MTP rounds). Neither reads what the other writes: they are fixed graphs on separate
  weights, scratch and fabrics, and the drafter choice reads nothing the first form writes. `G6_DRAFT_PAR=1` makes the
  choice first, launches the drafter step on its own stream, runs the first form (and the chain) on the main stream,
  and waits for the drafter at the end of the round. 2 puts a drafting step on a high-priority stream.
- Every kernel that waits for its peers runs at most 8 CTAs, so the other stream always finds SMs and neither can
  starve the other's fabric wait.
- The same graphs on the same inputs as serial: every step's bytes are unchanged. `batch_draftpar_test` (host) runs
  serial and parallel batches through the drafter choice, the confidence stop, the insert rule and `G6_DFL_REUSE`, and
  checks the same tokens, requests, steps and statistics. `G6_S2_ROUNDLOG` adds a `# s2_dfl` line: the drafter step's
  launch-to-wait time.

## 2026-10-16: an MTP confidence stop for a request's solo MTP rounds (G6_MTP_KSTOP, off by default)

- A request that decodes alone and drafts with MTP (the drafter choice's MTP rounds, and with `G6_DFL_REUSE` its
  warm-up and promoted rounds) chained and verified `--mtp` drafts every round. `G6_MTP_KSTOP=1` chains and verifies
  1..`G6_MTP_KSTOP_KMAX` (at most 3) drafts instead, chosen from the MTP head's own log-probabilities, which S2 now
  returns with the draft steps: draft j is verified while p_1 x .. x p_j >= `G6_MTP_KSTOP_TAU` t_j, and the chain stops
  before a draft that could not pass. `G6_MTP_KSTOP=2` takes the count from the drafter choice's MTP estimates.
- The log-probabilities are the same bytes on every rank, so every rank takes the same k; the policy is in the S2
  runtime signature. A session that another request joins has its chain completed to K before its first shared round.
- Drafts only: the target decides every token, but the verify's row count varies (2..kmax + 1) as a DFlash2 round's
  does. Off, the planner's steps are unchanged with or without the log-probability hook (`batch_kstop_test`).
- `G6_MTP_LPLOG=1` prints one `# kstop` line per solo round on rank 0 (the drafts' log-probabilities, the verified and
  accepted counts), for tuning the thresholds offline. It changes nothing else.

## 2026-10-16: DFlash2 after a prefix-reuse hit, and for a request left alone on MTP (G6_DFL_REUSE, off by default)

- At TP=4 a request whose prefix-reuse hit skipped into the drafter's 2,048-position window, or that
  was not alone at its prompt tail, ran MTP for its whole life. That is most later turns of a conversation.
  `G6_DFL_REUSE=1` lets the hit request keep the drafter: the drafter's context starts past the hit's rows, or at the
  window's start when its KV ring still holds the previous turn of the same conversation (S2 tracks which tokens the
  ring's rows belong to). With fewer than `G6_DFL_REUSE_MIN` (256) context rows the request runs MTP rounds that feed
  the drafter until it holds that many, then drafts with DFlash2. `G6_DFL_REUSE=2` also gives the drafter to a request
  that runs alone on MTP, from its last token, the same way.
- Drafts only: the target decides every token. Unset, no step changes. A session that starts its warm-up on MTP has
  its chain drafted at admit, so its first MTP round verifies K drafts. Host test: `batch_dflreuse_test`.
- With it on, rank 0 prints a `# s2 dflash reuse slot` line for every request: rounds on each drafter, where the
  context started, the warm-up and the promotion, so a request that never had DFlash2 shows too.
## 2026-10-16: the fabric's queue pairs take the ports' active MTU, not 4096
- Every RDMA queue pair was moved to RTR with `path_mtu = IBV_MTU_4096`, whatever the port could carry. On a switched
  RoCE fabric whose interfaces run a 1500-byte MTU the ports' active MTU is 1024, and the first boot failed with an S2
  prefill fabric error. With this change the same cluster boots and serves at 1500.
- Each rail now reads its port's active MTU at init (`ibv_query_port`) and sends it with its QP in the bootstrap; each
  connection takes the lower of the two ends' values, as the KV node's verbs code already does
  (`engine/kvman-rs/src/remote/verbs.rs`). After RTS the QP is read back and its path MTU checked, and every rail and QP
  logs its MTU.
- Ports at 4096 behave as before. The bootstrap's `QPInfo` grows one field, so every rank must run the same build (the
  launcher's handshake compares engine builds already). `fabric_mtu_test` (host) checks the choice.
## 2026-10-16: the replicated decode GEMVs split across the ranks (G6_QA_SPLIT, G6_IDXQ_SPLIT, G6_MTP_SPLIT, off by default)
- Every rank runs the GEMVs on the layer input in full (q_a, kv_a, the indexer's k and weights; the indexer's wq_b;
  the MTP draft layer's eh_proj and its own q_a / kv_a / indexer), because their weights are replicated. With the flags
  on, each rank runs only its own 16-output tiles (`bind::narrow` offsets the record pointer and the output columns; the
  weights stay replicated) and the columns are all-gathered on a fabric of their own: one collective per layer, port +
  64 for the main model and port + 72 for the MTP draft, proxy on `--proxy-cpu` - 9.
- `G6_QA_SPLIT=1` splits q_a, kv_a and the indexer's k and weights and gathers before the q_a norm; `=2` splits q_a only
  and runs the others in full while the gather is on the wire. `G6_IDXQ_SPLIT=1` splits the indexer's wq_b, publishes
  it, runs q_b, then gathers. `G6_MTP_SPLIT=1` does both for the MTP draft layer, and its eh_proj.
- Bit-exact: an output tile's arithmetic does not depend on which tiles a launch covers. Every rank must set the same
  values: they are in the S2 runtime signature, but the gather fabrics are made at engine start, before the engines
  compare their settings, so ranks started with different split flags hang at boot instead of failing.
- m4sim reads the three flags too and runs each model's gathers on a loopback, with the attention block (and the
  draft's eh_proj) launched in gather phases between which it delivers (`QsPh`; phase -1, what g6m4 passes, launches the
  block in its old order). `scripts/m4sim_split_check.sh` (one GB10 and a model) runs flags off against every
  combination on the plain, graph, MTP, S2 and CP paths and compares stdout byte for byte. m4sim also gains row lines
  (`G6_ROW_LINES=1`) and a row-count sweep (`--minv`) as probes.
## 2026-10-16: the draft heads and the DFlash2 drafter body on INT4 g32 copies (G6_DRAFT_HEAD_I4, G6_DFL_HEAD_I4, G6_DFL_BODY_I4, off by default)
- A draft step reads the same MXFP8 weights as the target: the MTP draft's lm_head (246 MB a rank at TP4), and the
  DFlash2 drafter's head and body (701 MB of body records a rank). Drafts are only proposals, so their weights can be
  smaller without touching what is committed. At boot each rank converts its own records to INT4 g32 copies on the GPU
  (`kernels/head_i4.cuh`, `k_mx_to_i4`; `DEC_I4PAIR` builds repack them for the GEMV's k parts) and rebinds the draft
  GEMVs to them, with the same splits: the MTP draft head reads 134 MB instead of 246 MB (`G6_DRAFT_HEAD_I4=1`), the
  DFlash2 drafter's head the same records (`G6_DFL_HEAD_I4=1`, one copy for both), and its body 382 MB instead of 701 MB
  (`G6_DFL_BODY_I4=1`). The MXFP8 records stay in the image (the target's head reads them), so each copy is extra memory.
- Drafts only: the target's head and every verify step are unchanged, so every committed token is the target's. The
  drafts and their log-probabilities can differ from the MXFP8 ones. Every rank converts its own slice by the same
  arithmetic, so the drafts are the same on every rank; the flags are in the S2 runtime signature.
- Tests: `dhead_i4_test` (`scripts/c1_tests.sh`, one GB10) checks the conversion byte for byte against a host
  derivation and the INT4 GEMV against a host dot on the head and drafter shapes, and times them. `dflreplay`
  (`build_m4.sh dfl`) replays a `G6_DFL_DUMP` through the MXFP8 and the INT4 drafter on one GPU and reports how often
  their drafts agree and how many each would have had accepted.
## 2026-10-16: the trellis shared expert on a forked stream during the router (G6_ROUTE_OVL, off by default)
- On a decode MoE layer with separate fc1 / fc2 launches (2+ rows), the shared expert waited for the router GEMV (8 of
  48 SMs) and the single-CTA plan kernel, though it reads only x2 and not the plan, and the DRAM sat mostly idle while
  those ran. `G6_ROUTE_OVL=1` forks a side stream before the router GEMV and runs the shared expert's fc1 and fc2 units
  there, joined before the finalize; `=2` runs its fc1 only there, joined before fc2. In a capture the fork and the join
  are graph edges. Fused layers (1 row) and the MTP layer run as before.
- `k1_tr3_fc1_ovl` / `k1_tr3_fc2_ovl` run the shared (or the routed) CTAs of `k1_tr3_fc1` / `_fc2` with the same unit
  numbering, bodies and scratch, so the output bytes are the default's. Registers equal the default kernels' (64, no
  new spills).
- `k1_ovl_test` (built by `scripts/build.sh`, run by hand on an idle GB10) checks mode 1 and mode 2 byte for byte against
  the default MoE layer, as plain launches and as one CUDA graph, M = 1..12 on a synthetic TP4 layer with the real router
  GEMV, and benches them.
## 2026-10-16: exact decode attention kernels, off by default
- Three rounds of decode attention kernels, each behind an environment variable that defaults to 0
  ([docs/SETTINGS.md](docs/SETTINGS.md)). Each is written to be bit-identical to the kernel it replaces (the same
  products, the same fmaf order and the same skips per row, head and slot); the measured set below is the one checked.
  - Round 1: `G6_ATTN_SC2` (scores with S key slots per warp, the query loaded once for all S) and `G6_ATTN_CTX2` (the
    context kernel with more gathered pairs per batch, or packed and branch-free with two batches in flight).
  - Round 2: `G6_ATTN_SC2_TR` (the heads' score sums reduced by one transpose across the warp), `G6_ATTN_CTX3` (context
    with one test per batch of slots, `G6_ATTN_CTX3_HG` for the head group), `G6_ATTN_SEL2` (the selector's final sort
    as a rank by count over several CTAs per row) and `G6_IDX_GRP` (in S2 batched steps the indexer reads each key tile
    once for a group of rows of one session).
  - Round 3: `G6_ATTN_QT` (absorb stores q and rope lane-transposed, and the scores kernel reads them with 16 B loads)
    and `G6_ATTN_FRONT` (a step's KV append, index append and index query prep, or index reuse, in one launch).
- `engine/attn/tests/attnk.cu` checks them on one GB10. It runs a full-index and a shared layer as the engine replays
  them (`g6_attn_replay_run` in a CUDA graph), first with every flag 0, then with each arm, and compares byte for byte
  against that stock chain over several runs with the outputs and scratch poisoned first: both layers' outputs, the
  indices, the indexer's scores and ids, the appended KV and index rows, per layer the raw scores and context parts,
  and the shared layer's probabilities and context. Any difference fails the arm and the bench exits 3. It also times
  each arm against a base (the rounds 1 + 2 set) per kernel and per call, hot and behind an L2 flush. Checked: the
  rounds 1 + 2 set (`SC2=4 SC2_TR=2 CTX2=4 CTX3=2 SEL2=16 IDX_GRP=16`) and it with `QT=1 FRONT=1` were byte-identical
  at contexts 2K, 8K and 32K with 1, 3, 8 and 12 rows. The other accepted values were not swept.
- Measured on 4x DGX Spark at TP4, RigMark c1 decode screens on base 6448a16:
  - round 1 (`SC2=4 CTX2=4`): prose +3.4% to +4.7%, code and structured +4% to +7%, and 8.6 ms less per step at c4
    (12 rows);
  - round 2 (`SC2_TR=2 CTX3=2 SEL2=16 IDX_GRP=16`): prose +0.7% at 1K context; at 32K context up to 3.1 ms less per
    step at c1 and 10.3 ms less at c4;
  - round 3 (`QT=1 FRONT=1`): prose +0.6%, code +0.7%, structured +1.2%; in the attnk bench 0.8 ms less per step at 2K
    context and 3 rows, 1.5-1.9 ms less at 12 rows.
- With all eight set (`G6_ATTN_SC2=4 G6_ATTN_SC2_TR=2 G6_ATTN_CTX2=4 G6_ATTN_CTX3=2 G6_ATTN_SEL2=16 G6_IDX_GRP=16
  G6_ATTN_QT=1 G6_ATTN_FRONT=1`) the four-box cluster's output at T=0 was identical to the stock kernels'.

## 2026-10-16: a CI runner for each KV format, and fewer widths on a pull request

- The engine job was the whole critical path of a 22-minute run: 1,187 s, with the fp8 and the int4 stage competing for
  one runner's four cores for most of it (int4 finished at 551 s, fp8 at about 1,000 s). They share no compilation, so
  each has a runner now. `dist` splits into `dist-fp8` and `dist-int4`, which are what the two jobs export, and stays
  the union of them, so `docker build --target dist -o out .` is unchanged.
- A pull request compiles three TP widths instead of six: 1, the `PF_F12` width 4, and the largest. A binary that lacks
  a width names the ones it has when asked for another (`engine/runtime/widths.cpp`), so an artifact built this way
  cannot quietly fail a TP=3 run. `main` and tags still compile all six.
- One leg exports the build cache for both, rather than a scope each. This repo's Actions cache was at 10.29 GiB
  against a 10 GB cap and evicting, which is why the cargo and loader stages had been rebuilding on runs that should
  have hit them.

## 2026-10-16: g6m4 and m4sim parse their shared flags in one place

- The real engine and the reference driver take 29 of the same flags, and m4sim's output is the oracle g6m4 is held to,
  but each main held its own copy of the parse lines: 171 identical lines between them, 103 of them substantial. They
  now come from `engine/runtime/engine_opts.h`, which is host-only, so `engine_opts_test` checks every shared flag, the
  bounds both mains checked (`--chunk`, `--mtp`) and the two bad values (`--v2-dense-form`, `--spec-accept`) under g++
  (`scripts/host_tests.sh`). Nothing about the flags changed: `--layers` still defaults to the same four layers, the
  flags that land in `Plan` are applied only when the command line names them so `Plan` keeps its own defaults, and the
  flags each main owns alone are untouched.

## 2026-10-15: one engine binary for every TP width

- `scripts/build.sh` builds one `g6m4` and one `m4sim` that run TP = 1, 2, 3, 4, 6 and 8 (and with `KV_FORMAT=int4`
  one `g6m4-int4`). `g6m4` takes the width from `--ips`; `m4sim` from `--tp N`, else from the rank packs in
  `--packs DIR`. Each width is compiled as its one-width build was (`DEC_S2` at every width, `PF_F12` at 4) into its
  own instance in the binary (`engine/runtime/build_m4.sh`), so its kernels are byte for byte the ones that build had.
- `G6_WIDTHS="3 4"` builds only those widths, and `G6_NR=4` only TP=4. `./gr8 glm53 build` no longer needs the TP. The
  image and the tester zip ship `bin/` alone (the image keeps `bin3/` and `bin6/` as links to it, which
  `docker/entrypoint.sh` still picks from), and the handshake's engine hash is the same at every width.
- `dtr_test` is `dtr_test_tp<N>`, one per width.

## 2026-10-15: the tool-call parser stops reading prose as calls, and the Python front end is gone

- The GLM parser searched the whole reply for the tool-call tags, so a reply that *wrote about* a call was taken for one:
  an example in the answer became a real `tool_calls` entry (and in reasoning it ended the reasoning, moving the text
  around it), an opener with no closer swallowed everything after it, a `</tool_call>` inside an `<arg_value>` cut the
  call short and leaked the rest of the format into the content, and a call naming a tool the request did not offer
  vanished from the text instead of staying prose. Those are one bug: the tags meant something wherever they appeared.
- A run of the reply from a `<tool_call>` on is now held and decided as a whole. It is a call only when the reply ends
  with it — a call the model makes is its last segment, and it stops right after `</tool_call>` — and text again when
  anything else follows: prose, a code fence's closing backticks, a `</think>`. As text its raw bytes go back where they
  came from, so nothing is lost and reasoning that quoted an example stays reasoning. An opener right after a backtick is
  inline code; a closer inside an `<arg_value>` is part of the value (the format has no escape); an unterminated call, or
  one naming a tool the request did not offer, is prose.
- The visible cost is the streaming cadence for calls: their deltas go out when the run is decided rather than as it is
  written, because a call that turns out to be prose cannot be taken back from a client. The final text and arguments are
  identical, and the README says so where it described the old per-key streaming.
- The Python front end (`frontend/g6front/`) is removed. `./gr8 glm53 up` runs `g6front-rs`, and the Python one was a
  second implementation of this same parser that a change like this would have had to keep in step byte for byte; its
  only consumers were `tests/diff_fronts.py`, `tests/bench_fronts.py` and `tests/test_parser.py`, removed with it.
  `wire.py` and `tokenizer.py` move up to `frontend/` for the shim, the mock engine and the shim tests, which stay, and
  `requirements.txt` is the tokenizer alone. The parser's tests are the Rust front end's now (11 new, 48 in all), and
  `frontend/tests/test_api.py` covers the HTTP path against the mock engine.

## 2026-10-14: a notice targets the work in flight, not a constant eight boundaries

- The first hardware trace of the event log says what #61's remaining delay is: the cancel anchored at boundary 8 landed
  at 16, while the 31,984-token prompt's prefill ran boundaries 9-13. The lookahead bound — `LOOKAHEAD = 8`
  (`s2/service.h`) — and at the cluster's line it could hardly be otherwise: a quantum at `--prefill 4096 --pf-lanes 2`
  is 8,192 rows and R is 65,536, exactly 8 quanta, so no below-R prompt spans more than 8 prefill boundaries and a
  notice anchored when one starts can never land inside it. The abort took 20.096 s, and the follow-up — whose admit
  cannot anchor until the cancel commits, because events serialize — saw 20.527 s TTFT. The reading rule this record
  carried was backwards: `N >= L` is what a lookahead larger than the remaining prefill produces (that run had `L` = 13,
  `N` = 16), not an acquittal of the lookahead.
- The target is now sized by the work in flight: the first boundary whose work after the one in flight covers 2,048
  prefill rows, within `boundary + 2` and `boundary + 8` (`Scheduler::horizon`, `s2/scheduler.h`; `s2/service.h` reads
  and clamps it). During a prefill that is the floor — the trace's cancel targets boundary 10, landing one to two
  quanta after the abort instead of past the whole prefill, and the follow-up's admit then anchors on an engine whose
  long prefill is gone, commits at once, and its TTFT is its own short prefill. In pure decode the target is the cap,
  exactly the old constant's behaviour, because a decode round adds no coverage (stalls can shrink one to almost
  nothing). The boundary in flight is not counted, since its remaining time is unknowable; tails count one row; the
  alternation with a decoding session and a partial last quantum are accounted; and a remainder under 2,048 rows opens
  the target to the cap, because that work is fast.
- The margin is sound for the same reason the constant's was: each step is one TP collective, so no rank departs a
  boundary before every rank completed it, and 2,048 prefill rows is about 1.5 s at the ~1,300 rows/s the gate measures
  for a cold 8K prefill — an order of magnitude past the milliseconds a notice needs to reach every rank. `horizon()`
  is a pure function of the scheduler state: it projects the same state machine `next()`/`completed()` drive, on a copy,
  so every rank names the same boundary and the notice's digest agreement holds; the compute thread publishes it beside
  the progress boundary, horizon first, so a torn read of the pair can only widen the target.
- `next()` is unchanged: its selection moved verbatim to `Scheduler::plan`, which `horizon()` also drives, and
  `scheduler_fair_test` still passes bit-for-bit, so m4sim's batch and every quantum the fair prefill serves are
  untouched. New `scheduler_horizon_test` (host, `scripts/host_tests.sh`) pins the rule: pure decode keeps the cap,
  #61's 31,990-token prompt targets the floor and opens to the cap when only its tail remains, the alternation and the
  partial quantum are counted, a 2,048-row quantum covers exactly, and a randomized run checks the clamp, re-derives the
  target through `plan()` at every boundary, and confirms two schedulers driven by the same events name the same one.
- `docs/EVENT-LATENCY.md` is rewritten to the rule and carries two corrections: the reading rule above, and the
  discrimination run — do not grow the prompt to 64K (that crosses R, where the run is exclusive and the follow-up is
  refused, so abort and TTFT stop measuring the same thing); shrink the quantum instead (`--prefill 2048 --pf-lanes 1`,
  where every notice lands mid-prefill).

## 2026-10-15: a forced tool_choice makes the reply a call, not text

- `tool_choice: "required"` and a function-naming object were a 400 ("named / required tool_choice is not supported").
  They now force the reply to be a tool call with the engine's insert rule (PROTOCOL.md `insert`): the engine puts the
  thinking close and the tool-call opener in place of the reply's first token, and the name too for a named choice, so
  the model cannot answer in text. The close comes first because the template leaves `<think>` open at the end of a
  chat prompt, and a call written inside the reasoning span is reasoning text, not a call. That is the mechanism the thinking budget already uses to close a `<think>` block, per request and from the
  committed tokens alone, so no engine or protocol change was needed — the front end only had to stop refusing them.
- It is a strong nudge, not a grammar. gr8's engine samples, so the front end cannot mask logits: a call to a name the
  request did not offer still comes back as text (the parser drops an unknown name), and the arguments are the model's
  own, coerced by the tool's schema as before. A forced reply carries no reasoning — its first token is the opener, not
  `<think>` — and the opener counts toward `completion_tokens` and toward `max_tokens` like any inserted token.
- Refused, with the field named: `required` with no tools, a name that is not in `tools`, an unknown string (it used to
  reach the parser and be treated as `auto`), `tool_choice` beyond a string or a function object, `logprobs` (an
  inserted token has no logprob), an explicit `thinking_budget` (one insert rule to a request; a server default budget
  steps aside instead), and an engine whose hello does not say `insert` — the capability `thinking_budget` needs too.
- New `tool_choice.rs` unit tests for the parsing and the forced text (the pure parts; the ids need the tokenizer);
  `tests/test_api.py` cases against the mock, which now continues inside a call an insert opened (the name if the opener
  does not hold one, then the arguments, then the close, with no reasoning); and a smoke case where the message asks for
  prose in French and still gets `get_weather`. The Python reference front end stays a 400 — it speaks hello version 1
  and has no `insert` field — so the two `err_tool_choice` cases are gone from `diff_fronts.py` rather than diverging
  there, and `frontend/README.md` says which front end has it.

## 2026-10-14: the event boundaries are logged, so a cancel's delay can be attributed

- #61's ~24 s has two candidate mechanisms that need different fixes: the scheduler served one prefill at a time and
  always the lowest request id, and a cancel is anchored at `boundary + LOOKAHEAD` with `LOOKAHEAD = 8` (`s2/service.h`).
  Eight boundaries is fine slack for decode rounds of tens of milliseconds; during a prefill it is up to eight 8,192-row
  quanta, about 50 s. Nothing printed the boundary a notice was anchored to, so a run could show the delay but not say
  which mechanism produced it — and the two disagree only when a prompt outruns the lookahead.
- `G6_S2_EVENTLOG=1` prints three lines to stderr on a serving build, with no state, no digest and no behaviour change:
  `notice` (rank 0's control thread: the boundary it read and the future boundary it chose, i.e. `A` and `A + 8`),
  `commit` (every rank: the boundary the notice landed at, so a rank that never prints it failed the epoch) and `step`
  (every rank: one line per prefill quantum and prompt tail — request, row range, boundary — and not per decode round,
  which would be far too much). Off by default.
- `docs/EVENT-LATENCY.md` is the run: a long prompt whose client aborts 1.5 ms after admission (the shim already turns
  the disconnect into the CANCEL), then a short follow-up, at 16K/32K/64K and `SESS` 2 and 4. 64K is the discriminator,
  because 8 prefill quanta plus the tail outrun the 8-boundary lookahead. If the notice lands before the long request's
  last prefill boundary the lookahead binds and the event-latency work is needed; if it lands after, the scheduler's
  admission was the whole story.

## 2026-10-14: the prefill quantum goes to the shortest prompt, not the lowest request id

- `Scheduler::next()` took at most one prefilling session per quantum and always the lowest request id. Native ordinals
  only increase, so the session that started prefilling first held every quantum until its prompt was done, whatever
  sizes were waiting: a 900-token prompt sharing the engine with a 32K one got no step until the 32K prompt finished.
  Found while reading #61.
- It does not change #61's number, and the cluster run says why: the follow-up there is never admitted. An admit is an
  event anchored at `boundary + LOOKAHEAD` with `LOOKAHEAD = 8` (`s2/service.h`), a quantum at `--prefill 4096
  --pf-lanes 2` is 8,192 rows, and R is 65,536 — exactly 8 quanta. A below-R prompt therefore never occupies more than
  8 prefill boundaries, so the admit lands at or after its last one, and above R the run is exclusive and the admit is
  refused outright. Two sessions prefilling at once is unreachable from a client in that configuration, which is the one
  #61 reports. Measured on four GB10s at TP=4, a short request submitted as a 31,990-token prompt is admitted: 23.5 s
  against 22.6 s on `main`, i.e. unchanged. At `--prefill 2048 --pf-lanes 1`, where that prompt is 16 quanta and the
  second admit lands mid-prefill, 14.2 s against 28.5 s, with the long prompt's own time unchanged. #61's remaining
  delay is the lookahead, counted in boundaries rather than time.
- The prefill choice is now the session with the fewest prompt rows left, `req.id` breaking ties. A session passed over
  `PREFILL_DEFER_MAX` (3) times is owed a turn and outranks every session that is not; once owed it is passed over at
  most once more per competing prefilling session, so a stream of short requests cannot defer a long prefill without
  bound. The choice reads only replicated slot state — `phase`, `prefilled`, `req.id` and the new per-slot `deferred`
  counter, which is in `Scheduler::digest` — never wall clock or arrival order, so every rank picks the same session and
  the event digest confirms it. Decode still gets its round between prefill quanta; only which prefill runs changed.
- `scheduler_fair_test` (`scripts/host_tests.sh`, no GPU) covers it: a short request admitted behind a 31,984-token
  prefill gets the first quantum and reaches decode in two boundaries while the long one is still prefilling; a long
  prefill under seven short ones is never passed over more than `PREFILL_DEFER_MAX + MAX_SESSIONS` quanta; a single
  session's quanta are unchanged (8192/8192/8192/7407/1 rows); and two schedulers driven by the same events choose the
  same sessions and agree on every digest over 300 boundaries.
- A prefill can now be interrupted by another session's quantum and resumed, which no earlier build could reach. On the
  cluster, at a 2,048-row quantum, the interrupted 32K prompt still answers from the row its quanta cover: incident
  000948, item 13, shard 12, checksum 07887, all correct.

## 2026-10-13: a session past R takes its own pages from the shared pool

- A request whose worst case exceeds R (`SHARE_LIMIT`, default `G6_CP_R`) runs alone, and its run took logical pages
  0..w-1 of the whole pool: every retained conversation was evicted, and the requests behind it waited for the pool
  even though their own pages would fit. The pool now grants such a run its own pages in the two classes the models
  already have: its prefix (R/64 pages) and its shard of the tail below `place` (the main model's physical pages),
  the drafter's copy of the rest above it. Retained short runs keep their pages through a long turn and give them
  back only as the classes need them; a continuation of the long conversation adopts the retained run's start as
  today. Scheduling is unchanged: the run still needs the pool to itself while it lives.
- The engine builds the run's tables from its list (`s2_solo_cp_pages`: its CP map, the MTP layer and the drafter
  take the list as built) and clears its pages only at bind (`s2_reset_pages`), on both models. A run's output is
  byte-identical to the tables-as-built run: the new m4sim gate `G6_S2_XLIST` runs every sequence of a `--prefill`
  CP build past R on a shuffled two-class page list and compares stdout (as `G6_S2_PFBASE` does below R).
- The store's tail blocks name the run's list pages (its shard below `place`, the drafter's pages above), and a
  restore into a shorter new request drops the blocks past its grant, whose rows the prefill rewrites. A tail's
  blocks were the binding's sharded pages; `KV_BLOCK_FORMAT` goes to 4, so a persisted store of format 3 starts
  empty.

## 2026-10-13: the engine checks its weights at boot, and `/health?deep` checks them again while it serves

- The Rust loader sums each 32 MB chunk of an image on the CPU while it sits in the pinned staging buffer. After the load
  the GPU sums the same chunks from device memory. A chunk that differs stops the boot with exit 3, naming the rank, the
  image, the chunk's bytes and the tensor it starts in ("g6: weight check: ..."), and the supervisor exits 3 as well. This is the load-time check #49
  asked for: it catches a bad copy to the GPU before any request is served.
- Once the graphs are captured, each rank sums its images again (binding rewrites some tensors in place) and logs the
  digest. `curl localhost:8100/health?deep` on the head then has every rank sum its weights on a low-priority stream
  while it serves (about 0.3 s at TP=4) and compare them with those sums. The answer is one JSON object, 200 when every
  rank passes, else 503. It answers localhost only.
- `G6_WEIGHT_CHECK=0` turns both off. The check adds about 1 s of CPU across the loader threads and two GPU passes at
  boot. `weight_check_test` (`TEST=weight_check_test scripts/c1_tests.sh`) checks the GPU sums against the CPU's.

## 2026-10-13: a restart keeps the earlier boots' engine logs, and an engine's exit names its reason

- Each boot moves the previous `g6m4.r<r>.txt` and `.err` to `out/prev1` and keeps three earlier boots (`prev1` to
  `prev3`). A restart after a bad boot used to overwrite its logs (#49).
- When an engine exits, the supervisor log names the line that says why (`g6: ...`, an error word, a CUDA or fabric
  check, a Rust panic), found anywhere in the last 4 MB of the rank's stderr or stdout. Before, it showed only the last
  20 lines, and threads still writing could push the reason out of them. A process killed by a signal is named (SIGKILL
  points at the OOM killer).
- The low-memory warning before boot names this TP and the variant's expert size, not "a TP=4 rank needs ~90 GB".
- The checks warn when `vm.compaction_proactiveness` is not 0, which the README requires.

## 2026-10-13: zero new device memory before the engine's streams write to it

- `dmalloc` zeroed with `cudaMemset`, which runs on the legacy default stream and returns before it finishes. The
  engine's streams are non-blocking, so a write issued right after on them could land first and then be zeroed. The
  prefill identity page table (`cp_pf_alloc`) is filled that way at boot: if the zeros won for entry k >= 1, every
  prefill row from 64k on went to page 0, with no error word, while decode stayed exact. That fits #49's bad boots
  (wrong for the whole boot, a cutoff at 64 or 128 prefilled rows, decode exact, weights exact); a bad boot with this
  build would confirm it. `dmalloc` now waits
  for its zeroing, the TP=4 pair-weight slots do the same before their rolls, and the boot reads the identity table
  back and stops on a wrong entry.

## 2026-10-13: the TP=4 fp8 KV pool grows to 1,981 pages a rank (CAP 310,528)

- The default TP=4 pool was 1,920 pages a rank (CAP 294,912), sized when every rank held every token's indexer keys.
  Since #44 the keys sit on the latent's pages, and the corrected memory model (docs/KV4.md, Capacity: it counts the
  buffers that grow with CAP) puts the same per-rank memory at 1,981 pages. The default `CAP` is now 310,528, and
  `--share-limit 32768` gives 408,832. TP=3 and TP=6 are unchanged.
- Boxes keep about 0.3 GiB less free memory than with #44 at 1,920 pages: what they had before #44.
- `--kv-format int4` keeps its measured 2,390 pages (CAP 415,232): 1.34x the fp8 default now.

## 2026-10-13: `--kv-format int4`: a 4-bit KV latent, chosen at launch (experimental, off by default)

- `--kv-format int4` (`KV_FORMAT=int4`) caches the MLA latent as INT4 (one FP16 scale per 32 dims) after a 512-point
  Hadamard rotation: 416 B a row instead of 656 B. The rope key stays BF16 and the indexer keys stay FP8.
  [docs/KV4.md](docs/KV4.md) has the row, the kernels and the numbers.
- At TP=4 it holds about 1.4x the tokens in the same memory: its default CAP is 415,232 against 294,912. Decode is about
  3% slower, 8K prefill about 6% faster and speculative acceptance about 2% lower. Needles at 32K-200K are found as with
  fp8, 12/12. TP=3 and TP=6 have estimated defaults (345,088 and 857,600) until they are measured.
- The row is fixed at build time (`-DG6_KV4=1`), so the image ships two engines per TP: `bin/g6m4` (fp8, the default,
  unchanged) and `bin/g6m4-int4`, about 15 MB each. The launcher picks one from `KV_FORMAT`, and each refuses the other's
  format. `KV_FORMAT=int4 ./gr8 glm53 build` builds `g6m4-int4` next to the fp8 engine.
- A KV store's index key names the row and the engine build, so a `persist=on` store written in one format is cleared
  by the other. KV node leases are sized by the engine's block and named `-int4` for int4.
- `engine/attn/kvfmt.h` is the one definition of the row: kernels, CP staging, S2 page copies, store page blocks, dumps
  and m4sim read it. `kv4_test` (`TEST=kv4_test scripts/c1_tests.sh`) checks the append round trip and the decode,
  prefill and row-owner attention against a host reference, for either row. `kvman-sim --mla-row 416` models the
  smaller page blocks.
- The capacity model now counts the five buffers that grow with CAP (QO tables, tensor-core prefill states, decode
  scratch, the MTP layer's scratch and prefill state, about 9.6 KB a token). They are why int4's gain is 1.4x, not
  656/416.

## 2026-10-12: the engine inserts tokens (`generate` `insert`), and the thinking budget uses it

- A request may carry one insert rule: once the reply holds N tokens and none of them is X, the engine puts the rule's
  tokens (1 to 64) in place of the reply's next token, even when that token is X. S2 requests carry it as
  `ins <N>:<X or -1>:<ids>`, and ready.s2 is version 3. The shim passes `generate`'s `insert` and says `"insert": true`
  in its hello. It refuses `insert` with `logprobs` (an inserted token has none) and on a version-2 engine.
- The rule is applied in s2_batch.h from the committed tokens alone, so every rank cuts at the same token. Tokens the
  verify sampled past it are dropped, as rejected drafts are. The inserted tokens are emitted at once. At the session's
  next round, after its run grew to the new length, they run as teacher-forced main steps (as prompt steps, each with
  its first-form draft and the DFlash2 drafter's context), the last row sampling the token after them, then the chain.
  The scheduler digest counts inserted tokens, so a rank that missed a rule fails at the next event. A request without
  a rule runs as before: m4sim prints the same lines.
- The front end sends `thinking_budget` as the rule (after N - 1 tokens, unless `</think>`, insert `\n</think>\n\n`)
  when the engine's hello says `insert`. The cut then costs one more main step, where the continuation cost 0.8 to
  4.2 s of prefill. With an older engine the front end still cancels and continues.

## 2026-10-12: CP: indexer keys are spread across ranks with the latent

- Under CP every rank held every token's indexer keys (22 layers x 132 B per token, all CAP tokens), though a rank
  scores only the keys of its own pages past R. They now sit on the latent's pages: the replicated first `SHARE_LIMIT`
  tokens, then the rank's own pages. At TP=4 that is 0.32 GiB per rank instead of 0.76 GiB; at TP=6, 0.33 instead
  of 1.59 (SHARE_LIMIT 32768).
- Prefill stages them with the latent: each staging pool gains one layer's index pages (2 x 8.4 KB per page of CAP),
  filled, gathered from the owners (two more RDMA ranges per page), hashed (`G6_CP_PF_VERIFY`) and copied back with
  the latent rows. An S2 run below R reads them in place, as before.
- Output is unchanged: m4sim prints the same lines as before with CP pages spread across ranks, two prefill lanes,
  row-owner index (`G6_PF_OWN=2`), MTP, prefix reuse and S2 sessions. Tier page blocks below R keep their layout and contents. A
  stored run's tail past R now carries its indexer keys in each rank's shard instead of the replicated part
  (`KV_BLOCK_FORMAT` 3, so a `persist=on` store of format 2 starts empty).
- The defaults (`CAP`, pages per rank) are unchanged; the memory this frees is listed in the PR for a later default.

## 2026-10-12: what the KV stores keep (`--kv-store`, `--kv-keep`), system prompts by default

- Experimental, off by default. `--kv-store STORE,size=N` (repeatable, tried in order; `KV_STORE`, stores separated by `;`) says where, and
  `--kv-keep system|sessions|all` (`KV_KEEP`) what. `system`, the default: each system prompt with its tools, once,
  after `--kv-system-after` conversations (default 2) started with it. `sessions`: those, plus runs the pool evicts that
  continued their conversation at least once. `all`: every eviction (#37's `keep=evicted`). `--kv-tier` and
  `--kv-shared-after` remain one release as deprecated aliases: a tier that kept evicted runs maps to `all`.
- Testers ran `keep=evicted` on live agent traffic: ~30,400 pages a rank in 37 minutes, nothing restored, ~2 TB a day.
  kvman-sim on the opencode captures agrees (109 GB/h a rank for four agents that fit the pool, one restore); `system`
  writes under 4 GB/h. `sessions` and `all` pay off only with more agents than the KV pool holds.
- A turn that continues a conversation whose last run the pool let go unstored counts toward no prefix (the pool
  remembers the heads of the last 1,024 runs it let go), and counts as continued for `sessions`.
  `engine/kvman-rs/REMOTE-TIER.md` has the numbers. `system` is the recommended setting; `sessions` for agent fleets
  larger than the KV pool.
- The front ends send each chat request's system span (`system_tokens`, PROTOCOL.md): the system messages and tools as
  templated. A new conversation counts only the prefix cells inside it, so an agent's own history never passes for a
  system prompt.
- An SSD store starts empty at every boot: blocks, index and counts. `persist=on` keeps it across restarts, keyed on
  the model, layout, block format (`KV_BLOCK_FORMAT`) and a hash of the engine binary; another key clears it with a log
  line. A boot that served wrong output must not hand its blocks or counts to the next (REMOTE-TIER.md, Persistence).
- A run's prefix that another run in the pool holds is stored once and referenced; a restore takes it from a retained
  run in the pool when one has it.
- A run past `SHARE_LIMIT` keeps its tail under `sessions` and `all`: each rank stores its own shard and the
  replicated parts, and a hit restores the rows past R instead of prefilling them again (checked in m4sim; not yet on
  the cluster).
- `--kv-write-limit F` (`KV_WRITE_LIMIT`): at most F blocks written per page of KV computed. Deterministic, off by
  default.
- Rank 0 logs, at most once a minute, what a store did once it fills, passes entries on, drops or refuses them.
  `status.s2` has per-store bytes and pages spilled, moved, restored, dropped and refused, exported as
  `g6:kv_store_*{store="i"}`. The shim now sends the engine status under `scheduler`, so the `g6:s2_*` metrics, never
  exported before, are too.

## 2026-10-12: thinking controls and a `response_format` guard in the front end

- GLM 5.3's chat template cannot turn thinking off, and the front end passed `enable_thinking` to it, which it never
  read. Now `chat_template_kwargs.enable_thinking` false and `reasoning_effort` `none` are a 400 that points to
  `reasoning_effort` `low` or a `thinking_budget`. The front end stops adding an `enable_thinking` of its own.
- `reasoning_effort` (top level, or in `chat_template_kwargs`) must be none, minimal, low, medium, high, xhigh or max,
  and the template hears the nearest level it names, ties going higher (TensorFold 0.6.1's rule): on GLM 5.3 minimal is
  low, medium is high and xhigh is max. Before, every value but low and high rendered as max.
- `thinking_budget` (an integer or null), as TensorFold reads it: while the think block is open, the reply's Nth token
  is replaced by `\n</think>\n\n` and the answer follows. 0 and null take the server default, and a negative budget is
  no limit. The front end ends the engine run there and continues from the
  prompt, the reply and the close, so the client sees one reply. The continuation prefills the prompt's last partial
  4,096-row cell and the reasoning again (2.5-4.2 s at TP=4's cold prefill rate on the mock, frontend/README.md).
- `--default-reasoning-effort` and `--default-thinking-budget` (`DEFAULT_REASONING_EFFORT`, `DEFAULT_THINKING_BUDGET`),
  off by default.
- `response_format` other than `{"type": "text"}` is a 400 until JSON mode lands. It was ignored before.
- Chat usage reports `completion_tokens_details.reasoning_tokens`.
- The smoke test checks the refusal and a budget where it checked thinking off. The gate's client, `pbench --effort
  nothink` and the smoke test leave out `enable_thinking`. Their prompts render the same as before.

## 2026-10-11: block stores below the KV pool (`--kv-tier`), off by default

- `--kv-tier STORE,size=N(G|T)[,keep=evicted|shared|all]`, given once per tier (`KV_TIER`, tiers separated by `;`).
  STORE is `ssd:PATH` (a folder per box), `remote:HOST:PORT` or `remote:auto` (a KV node from the mentat group). A
  `keep=evicted` tier holds the retained runs the pool evicts. A `keep=shared` tier holds the prefixes that started
  `--kv-shared-after` conversations (default 3), written once. A request that continues a stored entry reads its pages
  back and skips their prefill. Tiers are tried in flag order, and a full tier passes its least recently used entries
  on to the next tier that keeps them.
- A request that continues a live, retained or stored run counts toward no prefix, so a chat of five turns counts once.
- An SSD tier's shared prefixes survive restarts: an index beside the blocks, keyed by the model, TP, R and the page
  block's layout. A record is committed only after its blocks are written, and the ranks keep only the entries every
  rank loaded.
- The head counts and matches, and the grant carries the restored tier and the count. The control frames are version
  `G6S2E4`, so every rank needs this build. Without `--kv-tier` the pool does no extra work and its digest is unchanged.
- On the TP=4 cluster an evicted agent turn came back in 2.2-5.1 s (12.9-17.1 s without a tier), a stored system
  prompt after a restart in 1.3-1.4 s (~10 s cold), and the restored blocks were the same on all four ranks.
  `engine/kvman-rs/REMOTE-TIER.md` has the numbers.
- After a tier's copies the ranks pass a barrier (a Sync/Go round on the control links, frames `Sync` and `Go`) before
  the next round. One rank's copies to a KV node ran seconds longer than another's, and a skew near 10 s failed the next
  prefill's GPU waits with "S2 prefill fabric/attention error": an 8-agent replay lost the cluster twice that way. The
  prefill error now prints each error word, the layer, the request and the time into the quantum, and every rank logs
  its drains with the wall clock. `G6_KV_DRAIN_DELAY` and `G6_S2_BIND_DELAY` are test-only knobs that stall one rank.

## 2026-10-11: a remote KV page store, and `./gr8 kv` to run its node

- `./gr8 kv up --gb N` locks N GiB on a box with spare memory and serves it as KV page blocks (`kvman-remote`). It
  joins the mentat group as an agent without GPUs, with the service `gr8-kv`. `down`, `status` (`status all` for every
  node in the group) and `logs` work as for the model. KNOBS.md lists its options.
- Clients reach a node over RDMA (one-sided reads and writes) where both ends have a RoCE v2 address, over TCP
  otherwise. Each client holds a named lease, and the NIC refuses its QPs any other lease's memory.
- The engine does not use it yet. `engine/kvman-rs/REMOTE-TIER.md` has the numbers: 14.0 GB/s per PCIe root over a
  direct cable and 3.8 us for one row, against 5.7-7.0 GB/s and 43 us on the box's NVMe, and 1.18 GB/s over a 10G LAN.
  The same file has the plan for spilling evicted retained runs to either store.
- `kvman-remote-bench` reports what a lease moves from a client box, and how fast.

## 2026-10-11: the engine is called gr8 in the docs

- The README opens with "gr8: a super-fast local AI engine". Docs, comments, error messages and the API's `owned_by`
  call the engine gr8. Binary names (`g6m4`, `g6load`, `g6front-rs`), `G6_*` settings, `--g6env-x`, the `g6:` metrics
  and the protocol tags keep their names.
- The Docker instructions pull the image first, and docker/README.md says how to update every box to a new image.

## 2026-10-11: CI builds everything on GitHub's runners

- The engine job builds on GitHub's hosted arm64 runner, so `ghcr.io/kindlingai/gr8` is published from every push to
  main and every tag again. The self-hosted runner, its fork-refusal hook and the `GR8_ARM_RUNNER` / `GR8_NATIVE`
  switches are gone. Pull requests from forks now build the engines and the image too (never pushed).
- `docker/.env.example` and `--variant`'s help name the current defaults (`bit225` at TP=3) and every variant.

## 2026-10-11: S2 sessions take KV pages as they need them

- A session below R starts with the pages for its prompt, the token after it and the guard, and takes one more page
  each time its output reaches the end of the last one. Agent harnesses send `max_tokens` 32,000, so a 13,000-45,000
  token prompt used to hold 45,000-77,000 rows from admission, and one or two sessions filled the 1,920-page pool at
  TP=4.
- No session ever has to give pages back: the pool admits a request, or lets a session grow, only when every live
  session can still reach prompt + `max_tokens` + guard (a safe state). A session whose page would break that sits out
  rounds until another session finishes, then runs the same steps it would have run. Requests past R, by prompt +
  `max_tokens`, still take the whole pool at once.
- A finished session's pages go back to the pool when it finishes, not at the next admission.
- `status.s2` adds `held_pages`, `worst_pages`, `grown`, `stalled_rounds` and each run's pages held and worst case.
  The front ends export the first four as `g6:s2_held_pages`, `g6:s2_worst_pages`, `g6:s2_grown_total` and
  `g6:s2_stalled_rounds_total`. `G6_S2_GROW=0` gives every run its worst case at admission, as before.
- The control frames are version `G6S2E3`, so every rank needs this build.
- Predicted by `kvman-sim` on two recorded opencode traces (four sessions each, TP=4 time model, not yet measured on
  the cluster): median TTFT 42.6 s -> 4.1 s and wall time -15% on the first. The second gains 6-9% wall time and
  little TTFT, since 25 of its 54 requests are past R with their `max_tokens`. `engine/kvman-rs/SSD-TIER.md` has the
  tables.
- m4sim: `G6_S2_GROW=P` runs the `--s2` sessions on a pool of P pages that grows on demand.

## 2026-10-11: README: what concurrent requests produce

- The README and `kindling.json` said each request's output is byte for byte what it would be alone. That does not
  hold: when requests share decode steps, a request's tokens and logprobs can differ from its solo run (seen in m4sim
  on main). The README now says so, and AGENTS.md asks a change to keep output identical to the
  build before it, not to the solo run.
- The README's section on rank packs goes: it covers only the release on Hugging Face.

## 2026-10-11: S2 is the only serving path

- `g6m4` needs a DEC_S2 build and refuses to compile without one. The loop that served one request at a time (`--serve`
  without S2, `DIR/req.<n>` and `DIR/ready` on every rank) is gone. Without `--serve`, `g6m4` still runs its `--prompt`
  sequences once and exits: `docs/REUSE-GPU-TEST.md` uses that.
- `frontend/g6shim.py` is gone, with `tests/test_g6shim.py`, `tests/test_shim.py` and `tests/fake_serve_engine.py`.
  `g6s2shim.py` now holds `EOS_IDS` and `VOCAB`.
- The launcher always starts the Rust front end and `g6s2shim.py`. The `FRONT` option is gone, and a `FRONT` line in an
  old config file is ignored. The Python front end (`frontend/g6front`) stays as the reference the Rust one is tested
  against.

## 2026-10-11: S2 sessions hold page lists

- A session's KV is a list of pool pages, the lowest free ones first. A request is admitted whenever enough pages are
  free anywhere in the pool: fragmentation never stalls admission. `status.s2` drops `fragmentation_stalls`, and with it
  the `g6:s2_fragmentation_stalls_total` metric. `largest_free_run` counts the pages one request can take.
- A prefix-reuse hit keeps the retained run's first pages, as many as its cached rows fill, and takes the rest from the
  free pages. A miss evicts whole retained runs, the least recently retired first.
- Every rank runs the same allocator. A grant carries its list's hash, and each event's state hash covers every rank's
  pool. The control frames are version `G6S2E2`, so every rank needs this build.
- m4sim: `G6_S2_PAGES=seed[,pages]` gives every `--s2` session a shuffled, non-contiguous page list.

## 2026-10-11: requests up to 65,536 tokens share decode steps at TP=4 and TP=3

- A request shares the pool only while its prompt + `max_tokens` fits in R, the KV every rank holds in full. R was
  32,768, and agent harnesses send longer prompts, often without `max_tokens`, so each of their requests ran alone and
  the others waited. `SHARE_LIMIT` (`--share-limit`) sets R: 65,536 at TP=4 and TP=3, 32,768 at TP=6. At TP=4 two
  49,000-token requests now decode together at about 21 tok/s each (40.1 aggregate), where they ran one after the
  other at 13.8 aggregate; three 33,000-token ones reach 50.3 aggregate.
- An unset `CAP` follows `SHARE_LIMIT` so each rank keeps the same KV memory. The longest context drops to 294,912
  tokens at TP=4 (was 393,216) and 241,600 at TP=3 (was 307,200). `--share-limit 32768` gives back the old pool.
  Past 32,768 tokens a session that fits the limit decodes without the cross-rank gather, so it is faster alone too
  (29.5-31.3 vs 26.5-27.6 tok/s at 49,000 tokens of context).
- A chat request without `max_tokens` gets what keeps it within the limit, but at least 8,192 tokens
  (`--share-tokens` in the front end). It used to get all the room the context leaves, which always ran alone.

## 2026-10-11: TP=3 boots bit225 by default; a new release commit reuses the cached files

- The exlr8 release (v0.5) adds five width profiles: `bit225`, `bit243`, `bit260`, `bit325`, `bit338`. They spend
  bits by tensor type, then by layer, in the order that lowered full-vocabulary KL against bf16 the most per byte.
  The model card has their sizes and KL.
- TP=3 boots `bit225` unless `--variant` says otherwise: `down1`'s size with 5-10% lower KL. On three GB10s, two
  boots each: code 38.6 vs 38.3 tok/s, prose 29.0 vs 29.6, the same memory. A release without `bit225` boots `down1`.
- TP=4 keeps `home`. `bit325` has its size and 1.5-7% lower KL, but decoded about 2.5% slower in three boots each
  (code 48.7 vs 49.9 tok/s). `--variant bit325` picks it.
- `prepare` for a new commit of a release hard-links each file an older commit left in the cache when the hub gives it
  the same hash and size at both, and copies an expert file's fetch journal with it. A commit that only adds manifests
  (v0.5) fetches nothing else. Every expert half is still checked against its index file.

## 2026-10-10: TP=6 serves through S2

- `scripts/launch_tp6.sh` runs `scripts/launch_s2.sh` at TP=6: up to `SESS=8` requests at once, prefix reuse, MTP only
  (`DFLASH=0`: the DFlash2 drafter's 8 KV heads do not split six ways; `SPEC_ADAPT=0`). `PF_OWN` defaults to 3 (full-head
  attention weights from `packs/qo.pack`; `PF_OWN=1` needs no QO pack), `READY_TMO` to 1800.
- `scripts/build.sh` builds `-DDEC_S2` at `G6_NR=6` (the default), 4 and 3; `-DPF_F12` stays TP=4 only.
- The rank-health probe runs before the boot at TP=6; the shim and the front end run on CPUs `0..PROXY_CPU-8`; the
  shim gets `--stall-timeout 2400`.
- `PLAIN=1` (the one-request loop) is refused at TP=6.

## 2026-10-10: Docker: one compose file for every box

- `docker/compose.yaml` runs the launcher in the image on every box (`docker compose -f docker/compose.yaml up -d`),
  with its options from `docker/.env`. The image's entrypoint is `./gr8 glm53`, so `prepare`, `status`, `smoke` and
  `down` are `docker compose run` or `exec` away. The launcher in the image owns every default.
- The container runs as uid 1000 on a read-only image with a private `/tmp`, every capability dropped but `IPC_LOCK`,
  no new privileges, no core dumps and `oom_score_adj 1000`. Its state is two volumes: the model cache and `G6_HOME`.
  With no ssh at TP=4 and TP=3, it needs no ssh keys and no host paths.
- `restart: on-failure`: a cluster-wide `down` exits 0 everywhere and stays down. When a rank fails, every box now
  exits 1 (the head's `down` says it is a failure), so every container restarts and the boxes boot again together.
  Before, the workers the head stopped exited 0.
- The runtime image has `jq`, which `./gr8 glm53 smoke` needs. Based on OsakaTX's `docker/gr8` (#10).

## 2026-10-10: TP=4 and TP=3 need no ssh between the boxes

- The head engine sends each request's bytes to the other ranks over its S2 control links, in a `Request` frame just
  before every ADMIT notice for it, and the workers take them from there instead of their own `req.<id>`. The hash in the
  notice still checks them. The head also passes its orderly quit on (`Quit` frame), and each worker quits through its
  own `DIR/quit` as before. Control frames may be 16 MiB (was 1 MiB), so a 393K-token prompt fits.
- `g6s2shim.py --dir DIR` (the head's own directory) replaces `--rank-dirs`; it no longer uses ssh. The one-request shim
  (`g6shim.py`) still reaches the other ranks over ssh.

## 2026-10-10: `--shape ring`: TP=3 and TP=4 without a switch

- Each box's f0 port is cabled to the f1 port of the next box in `--peers` order. Every port shows up as two RDMA
  devices, one per PCIe root. Neighbours are written directly on their own devices. At TP=4 the diagonal is reached
  through the next rank: the origin writes the payload into a 32 MB relay ring on that rank and sends its ring position
  and destination offset, and a relay thread there forwards the same bytes to the same offset, then returns a credit.
  Writes of 40 bytes or less (the bulk flags) travel inside the SEND. The rings pin about 0.25 GB per rank at TP=4
  (the first version mirrored each fabric's receive side, 5.2 GB per rank, which left too little memory for the
  default CAP).
- The launcher passes `--ring` and the LAN addresses as `--ips`: the fabric and S2 bootstraps go over the LAN.
  `--ring-devs` names the four devices when they differ from the GB10 defaults. The relay thread takes the core 8
  below `--proxy-cpu` (11 by default), below the seven other proxies.
- Measured on four GB10s on a cable ring at 200G, beta 1 with the patch, MTP k=2, against a switch: c1 decode
  38.5 vs 42.5 tok/s, c4 aggregate 69.2 vs 72.0, prefill 1,236-1,264 vs 1,243-1,318 tok/s.

## 2026-10-10: serving is S2 in every build that has it; no `--s2` flag

- A DEC_S2 build (TP=4, TP=3) always serves through S2: `g6m4 --serve DIR --s2-port N ...`. `--s2 DIR` is gone, so a
  served TP=4 or TP=3 rank cannot fall back to the one-request loop. The TP=6 build has no S2 yet, and its `--serve`
  stays the one-request loop. Without `--serve`, any build runs its offline test prompts as before.
- The launcher passes `--serve $G6_HOME/s2`; the cluster options digest changes with it, so every box upgrades together.

## 2026-10-10: `G6_PF_GEMMF32=1`: the QO o_proj uses the full-tile GEMM (opt-in)

- The row-owner QO o_proj (`G6_PF_OWN=3`) writes FP32, which kept it on the guarded `k_gemm_mx8b` while the BF16 / FP16
  output GEMMs take the full-tile kernel: the per-mma guards make nvcc wrap each QMMA in WARPSYNC / BSSY / BSYNC.
  `G6_PF_GEMMF32=1` gives `k_gemm_mx8f` (the unguarded full-tile kernel) an FP32 epilogue and launches it for big
  o_proj launches. Same mma calls, operands and k order per output, so the bits are the same. Default 0; needs
  `-DPF_GEMMF`.
- One GPU, TP=6 shape (M = 682 and 683 rows, N = 6144, K = 16384): 0 differing output bytes over two runs of each M,
  median 1.546 -> 1.377 ms per layer. TP=6 prefill, 8K prompt: 1953 -> 2004 tok/s (three runs: 2014 / 1979 / 2019),
  32K prompt: 1952 -> 1980; the canary output is byte-identical with the switch off.

## 2026-10-10: TP=3 boots down1 by default; the engine stops short of the OOM killer

- `--tp 3` with the exlr8 release picks `--variant down1` unless one is given. `home` puts 108.7 GB of weights on a
  TP=3 rank, which leaves no room for the KV pool. `down1` at TP=3: 78.4 GB per rank, 15.6-17 GiB free at the default
  CAP of 307200.
- The engine reads MemAvailable after each layer of the KV pool and stops under `G6_MEM_MIN` (8 GiB) with a message
  naming `--cap` and `--variant`. Before, the kernel's OOM killer ended the boot and took sshd and agents on the box
  with it.

## 2026-10-10: o_proj prefill reads its trellis tiles

- `bindp::k_gemm_tr` loads the raw K4 tiles of a dense trellis matrix (half the bytes of its FP8 copy) and decodes each
  stage into the bytes the copy would hold, so its output equals the copy path's to the bit. The decode of the next
  stage overlaps the current stage's mma.
- o_proj prefill uses it, including the pair-owner path (`G6_PF_OWN=4`), whose o_proj half now travels as raw tiles.
  The o_proj FP8 copies are gone: about 2.0 GB less per rank at TP4. `G6_DTR_PF_COPY=1` brings them back (A/B).
- Prefill time, tiles / copy, TP4 rank-0 shapes at 2200 MHz: o_proj 0.97-1.00 at 512 rows, 0.72 at 2048, 0.73-1.00 at
  4096 (the copy path's time varies between runs); the pair o_proj 0.69-0.95. q_b and the dense MLP keep their copies:
  on the tiles they measured 1.07-1.33.

## 2026-10-10: boot from the exlr8 release alone (step 1, part 2)

- `g6m4 --v2 DIR --v2-slices DIR --v2-widths FILE --v2-dense DIR`, with no `--manifest` / `--pack`: the Rust loader
  plans each rank's whole image from the release (`engine/loader-rs/src/release.rs`). The experts come from the width
  files as `--v2` reads them. Every other tensor comes from `dense/`, copied as stored or computed at load (`Op::Gen`),
  under the names a converted pack uses. No TR3 checkpoint, GPTQ codes or packs are needed. Rank 0 prints which form
  each matrix took.
- `--v2-dense-form trellis` (the default) keeps q_b, o_proj and the dense MLP of layers 0-2 as the release's trellis
  entries for the dense trellis kernels, so every weight is the release's own: the MXFP8 matrices as stored, the int4
  o_proj / q_b records from the release's codes, kv_b and the embedding in BF16. Only lm_head is rounded to MXFP8.
  `--v2-dense-form rounded` is the fallback: it rounds those matrices to the MXFP8 and int4 forms of a converted pack.
- Under `G6_PF_OWN=3` the image also holds the full-head q_b / o_proj / UK / UV, so TP=6 needs no QO pack.
- The launcher: `--model` takes the release (https:// or a local copy) as the whole model, and `V2_DENSE` follows from
  it. `--base-packs` stays optional. The set id of a release is its weight set id with the tokenizer files.
- Non-expert weights per rank: TP=4 7.91 GB in the trellis form (11.21 rounded, about 11.6 from packs), TP=3 8.70
  (13.10, 13.80), TP=6 7.13 (9.33). The prefill FP8 copies of the trellis matrices add 2.81 GB per rank at TP=4.
- Checks: `tests/release_check.py` compares every planned non-expert tensor with an independent derivation
  (`loader/exlr8.py`, `g6ref.py` records, `plan.py` ownership), byte for byte: TP=4 trellis on all four ranks (876
  tensors, 11.85 GB), TP=4 rounded (302, 6.00 GB), TP=3 trellis on all three ranks (348, 8.30 GB), TP=6 rounded with the
  full-head tensors (72, 2.64 GB). On one GB10, m4sim from the release (layers 0-3, all ranks agree) matches an fp32
  PyTorch forward of the same weights (`engine/ref/release_forward.py`): per sublayer 1.2e-3 to 9.3e-3 rel L2 in the
  trellis form, logits KL 1.5e-5 nats per position, where a BF16 copy of that forward gives 9.3e-5. TP=3 and TP=4
  generate the same tokens. Prefill and decode-only runs generate the same tokens.

## 2026-10-10: the engine serves the dense trellis matrices

- Decode: `bind::make_layer` binds q_b, o_proj and the dense MLP of layers 0-2 from the release's trellis tensors
  (`LLLL.NAME.trellis / .suh / .svh`, `dtr::layer_mats`) when a layer has them: one launch each for q_b and o_proj,
  gate + up and down for the MLP. o_proj and down write the fabric send slot under `DEC_SLOT`. The MTP layer takes the
  same path. `q_b.gptq` (int4 g32, the layers where int4 won the encode) has a binding too. `G6_DTR=0` binds the int4 /
  MXFP8 records instead. Each rank logs how many layers run each form.
- Prefill: the matrices run as exact FP8 copies of their tiles on today's MXFP8 GEMMs. The quantize pass applies the
  input rotation (`Xq::suh`) and the GEMM epilogue the output rotation (`Gemm::osvh`). Prefill time stays at today's. `G6_PF_OWN=4` carries the copies and the pair's vectors over the wire. OWN=3 needs `--qo-pack`.
- `dtr_test --bind` runs one dense layer through both binders on one GPU and `engine/ref/dtr_ref.py` checks the results
  against `loader/exlr8.py` at TP=3, 4 and 6.

## 2026-10-10: `bin/g6load dflash2` converts the drafter without numpy

- `g6load dflash2 --src DIR --out DIR [--nr 4] [--ranks R] [--bf16 CLASSES]` (`engine/loader-rs/src/dflash2.rs`) writes
  the DFlash2 drafter's `dflash.r<r>.pack` and `.manifest.json`, byte for byte what `tools/convert/dflash2.py` writes:
  the same row and column slices, MXFP8 quantization (E8M0 scales, E4M3 round to nearest even) and decode record
  layouts, a sha256 per tensor and the manifest JSON as Python prints it. On `incoai/GLM-5.3-DFlash2` all four TP=4
  ranks match (pack sha256 and manifest bytes), in 7-9 s a rank against 14 s, and so does rank 1 with every class
  in BF16.
- `./gr8 glm53 up --drafter` converts this box's rank with it, so nothing on the boot path needs numpy now.
  `tools/convert/dflash2.py` stays for `./gr8 glm53 packs` and its `--head` test packs.
- `cargo test` compares both on a small synthetic drafter (`tests/data/dflash2`, from `tests/make_dflash2_fixture.py`)
  with zero groups, -0.0, BF16 subnormals and rounding ties.

## 2026-10-10: dense trellis kernels for the exlr8 dense weights (not wired)

- `engine/kernels/dtr.cuh`: decode GEMV for the release's trellis K4 dense matrices (q_b, o_proj, the dense MLP of
  layers 0-2) on the release's entries as stored. One launch per matrix rotates its input in a prologue and reduces
  in fp32. The MLP's gate and up run as one launch whose epilogue writes down's rotated input. `dtr_prefill.cuh` runs
  prefill on an FP8 copy of the tiles (exact: the codebook is E4M3) with today's MXFP8 GEMMs inside the rotations.
- `engine/kernels/tests/dtr_test.cu` + `engine/ref/dtr_ref.py`: checks against `loader/exlr8.py` on the release's
  files and times decode against today's GEMVs. On one GB10 at 2200 MHz, TP4 rank shapes, M = 1/3/8: q_b 0.53-0.57x
  of today's MXFP8 time, o_proj 0.90-0.95x of int4 g32, the dense MLP 0.62-0.68x of the int4 K1 path.
- The engine does not call them yet: the loader's tensors (`LLLL.NAME.trellis/.suh/.svh`, `dtr::layer_mats`) come next.

## 2026-10-10: a decoder for the exlr8 dense and vocabulary files (step 1, part 1)

- `engine/loader-rs/src/exlr8.rs` decodes the exlr8 v2 release's `dense/exlr8-dense-LLL` and `dense/exlr8-vocab`
  matrices in every format they use (trellis K4, int4 g32, MXFP8, FP8 block, FP8 row, BF16, FP32). The values equal the
  output of `loader/exlr8.py` bit for bit in fp32. The decoder takes any block of a matrix, so a rank reads only its own
  trellis entries. The two Hadamard products sum in input order with one fused multiply-add per term. NVPL and cuBLAS
  use that order for the reference's fp32 matmuls, and any other order changes the last bit of some results.
- `exlr8dump` writes a layer's decoded tensors under the reference's names. `tests/exlr8_compare.py` compares the dump
  with `loader/exlr8.py` on the release's files: all 80 files (549 matrices, 16.0 billion elements) are bit-identical in
  fp32 and in BF16, and so is every rank's block at TP=3, 4 and 6 for layers 0, 2, 3, 4, 6, 78 and the vocabulary.
- `cargo test` checks the decoder against the reference's output on a synthetic fixture of every format and side
  (`tests/data/exlr8-fixture.safetensors`, 194 KB), so CI needs neither torch nor the release.
- Nothing in the engine uses it yet: the boot from the release alone builds on it.

## 2026-10-09: DFlash2 under `--s2` (the S2 launcher default, `DFLASH=1`)

- `--s2 --dflash-manifest .. --spec-adaptive` (with `--mtp`): a session alone at its prompt tail drafts with DFlash2
  as the plain build's `dflash_sequence` does (prompt context from the prefill's taps, adaptive 1..7, one SpecLen per
  process). Each DFlash2 round also runs the MTP first-form pass over the accepted rows, so when a second session
  arrives the K-1 chain steps hand the session to the shared MTP rounds (`Batch::demote`). A demoted session stays on
  MTP. The batched graphs are captured without the drafter's residual taps; the solo graphs keep them. Guard rows are
  14 with the drafter loaded (a verify writes 7 rows past the last token).
- A prefix-reuse hit drafts with DFlash2 only when its re-prefill covers the drafter's 2,048-row window.
- Gate (TP4): 10 requests at temp 0 with logprobs, adaptive k, byte-identical to the plain build's DFlash2; 8 concurrent
  requests (the first starts on DFlash2 and demotes) byte-identical to MTP-only S2. RigMark 1.0.0 (knapcio's
  comparison id, `reasoning_effort` low): code 49.5, prose 28.4, structured 68.4 tok/s (plain DFlash2 49.2 / 29.1 /
  67.0; S2 MTP 40.9 / 31.3 / 43.2), C2 50.6, C4 63.6. Free memory at 8 sessions: 2.7-3.6 GB per node.

## 2026-10-09: prefix reuse under `--s2` (`REUSE=1 scripts/launch_s2.sh`)

- `--s2 --reuse-prefix`: a retired session's run stays in the pool with its prompt + emitted ids (`s2/pool.h`). At
  admission the head matches the new prompt against every retained run with the plain build's rules
  (`prefix_reuse.h`); a hit takes that run's base, extended in place, and its prefill starts at the retained row
  (`Grant::reuse`, sent to every rank, which re-derives it before claiming). Retained pages count as free: a claim
  that overlaps one evicts it, and a past-R run evicts all of them.
- On by default under `--s2`: the grid rule (`G6_REUSE_ALIGN`, whole 4,096-row prefill cells, so a hit is exact) and
  `G6_REUSE_NOEND` (the `prompt >= written_end` rule refused replaying a prompt as it was; attention reads only
  positions at or below a row's own).
- New runs go in the middle of the largest gap free of live and retained runs, so the conversation before it and this
  one both have room to grow on their next turn.
- Gate (TP4, 20,001-token prompts, temp 0, logprobs 5): every output identical to a cold boot of the same binary; a
  continued conversation 7.1 s vs 17.3 s cold, the same prompt again 4.6 s vs 17.3 s, two conversations continued at
  once next to a decoding session 16.1 / 15.4 s vs 28.8 / 28.1 s.

## 2026-10-09: fused K1 schedules only the plan's slots

- The fused K1 kernel handed out items for M x 8 slots (the capacity) and ran the empty ones (fence, barrier, counter)
  for slots past the plan's experts; it now hands out the plan's own slots (`n_slots_dev`). The last CTA zeroes the
  counters with all its threads instead of one. Same work per real item. Experts per step at one session: 14.49 ->
  14.21 ms at 1 row, 29.55 -> 29.22 ms at 3 rows; isolated K1 at 4-8 rows 220 -> 227 GB/s.


## 2026-10-09: multi-tile GEMVs at up to 128 registers

- The dense GEMVs above 8 rows (S2 batches) ran a 1024-thread-bound kernel, capped at 64 registers: the INT4 o_proj and
  q_a fell from ~190 to ~120 GB/s at 24 rows. Bindings with ks <= 16 now take a 512-thread-bound copy (fewer tiles per
  CTA, same bits). S2 at 24 rows: o_proj 9.2 -> 7.5 ms, q_a + kv_a 6.75 -> 6.3 ms per step; 8 sessions 75.5 -> 77.0 tok/s.


## 2026-10-09: separate K1 launches at 16+ rows

- `row_budget_test --time-v2`: K1 on an exlr8 v2 layer per row count, with the bytes read and GB/s (fresh routing and
  six layer copies per run, so every byte comes from DRAM). On GB10 at 2200 MHz the fused launch reaches 220-225 GB/s
  at 3-12 rows and falls to 203 at 24 and 193 at 32, where the separate fc1 / fc2 launches are 2-3.5% faster.
- v2 layers now take the separate launches when the step has 16+ rows (S2 batches; the fused / separate choice keeps
  every partial and its order). S2 at 8 sessions: 73.5 -> 76.5 tok/s (code); 4 sessions unchanged.


## 2026-10-09: short prompts skip OWN=4 (`G6_PF_QP_MIN`, default 768)

- OWN=4 pays a fixed ~0.13-0.35 s per request (the pair weights rolled through every layer), so it lost on short prompts:
  a ~50-token prompt's TTFT was ~0.69 s against ~0.34 s with OWN=2 (TP4 prefill tok/s OWN=4 / OWN=2: 256 tokens
  401 / 506, 1024 tokens 846 / 809, 8192 tokens 1,254 / 809). A prompt shorter than `G6_PF_QP_MIN` now runs OWN=2 in
  both the plain build and S2; 0 restores OWN=4 for every request that fits `G6_PF_QP_CTX`. Short prompts' output then
  equals an OWN=2 (and OWN=1) boot's. S2 with OWN=4 at 4 / 8 sessions back to OWN=1's decode aggregate (code 61.6 / 73.8).

## 2026-10-09: `G6_PF_OWN=4` prefill under `--s2`

- S2 prefill can run pair-owner QO attention (`G6_PF_OWN=4` with `G6_PF_QP_ROLL` / `G6_PF_QP_CTX`, as the TP4 launcher
  sets them); S2 prefill runs on the CP staging pool, indexed by position, so a session's base page never reaches the
  QO prepass. Each request takes OWN=4 or OWN=2 by the plain build's rule (prompt + gen against `G6_PF_QP_CTX`). Solo
  output equals the plain build's OWN=4 output byte for byte; 8K prefill 818 -> 1,260 tok/s.

## 2026-10-09: adaptive k verify costs re-measured

- `spec_len.h`'s verify-time table (ms by rows 1..8) now holds the current DFlash2 step times
  (43.0 / 52.4 / 60.4 / 68.4 / 76.1 / 83.7 / 91.1 / 97.4, was 44 ... 111): steps of 4-8 rows got 9-12% cheaper with
  Adaptive k now picks longer drafts. One session, DFlash2: after a 57.6K prompt 30.3 -> 34.1 tok/s (two
  runs each); short prompts code / prose / structured 38.9 / 35.0 / 41.3 -> 38.2 / 36.8 / 40.2.

## 2026-10-09: one-launch absorb / value project at any row count

- The plain build ran absorb and value project per row (weights reloaded per row) above 3 rows, so DFlash2 verify
  steps of 4-8 rows paid it. They now take the one-launch kernels at any row count. One session, DFlash2 (default):
  short prompts +2% (38.1 / 33.6 / 40.5 -> 38.7 / 34.5 / 41.3 code / prose / structured), 31.9 vs 28.0 tok/s after a
  57.6K prompt. MTP k=2 and prefill unchanged.

## 2026-10-09: head-batched attention for every decode call

- The head-batched scores and context kernels now serve every decode attention call, not only the fixed-K ones: verify
  rows of a fresh index and contexts past 2048 take them too (the generic `mla_scores` / `context_parts` do the same
  arithmetic; `dec_heads_test` checks both). One session, TP4: MTP k=2 34.2 -> 36.3 tok/s after a 13.5K prompt and
  28.0 -> 32.2 after 57.6K; DFlash2 32.8 -> 35.1 and 23.7 -> 28.0. Short prompts and prefill unchanged.

## 2026-10-09: decode step profile and attention kernels

- `G6_DEC_EVPROF=N`: per-stage decode step times from event-record nodes in the captured main-step graphs (attention
  split into its kernels), printed every N main steps in each rank's log.
- Decode attention scores and context load each key once for a group of heads; absorb and value project run all rows of
  an S2 step in one launch with the weights loaded once; `k_ar_add_norm` loads all peers before adding. Bit-identical
  (`dec_heads_test` in `scripts/c1_tests.sh`). S2 at 24 rows: attention 51 -> 28 ms per step.
- Boots without QO attention (every `--s2` boot) now build the decode RoPE table; absorb computed RoPE in FP64 before.
- S2 aggregate decode (code, TP4, 2200 MHz): S4 55.6 -> 60.3 tok/s, S8 67.8 -> 74.0. One session and prefill
  unchanged.
- `--s2` + `G6_S2_ADAPT=1` (opt-in): each round's MTP draft count from the sessions' accept counts and the S2 step costs.

## 2026-10-08: prompt-prefix reuse, `--reuse-prefix` (ported from the K1/CP serve line)

- `g6m4 --reuse-prefix` keeps the attention caches of the longest common prefix of the new prompt and the previous
  sequence (its prompt + the ids it actually emitted) instead of resetting them, so an agentic conversation re-prefills
  only its suffix. `engine/runtime/prefix_reuse.h` holds the CPU state; the engine's cache path is unchanged.
- `G6_REUSE_ALIGN=1` makes a hit exact: it retains only whole cold prefill grid cells,
  `[0, floor(min(lcp, pf_end) / grid) * grid)` with `grid = --prefill MP`, and re-prefills from that grid point. Every
  retained cell was then computed with the cold chunk shape, so a hit's cache rows and output tokens are the cold run's
  bit for bit. A retained prefix below one grid cell is a miss (`why` = `grid`).
- Off unless given. Env arms: `G6_REUSE_STRICT=1` (strict append only), `G6_REUSE_NOEND=1` (drop the
  `prompt >= written_end` rule), and the gate/diag arms `G6_REUSE_POISON=1` (`reuse_poison.cuh`: a hit NaNs the stale
  rows first, so a read of one shows up), `G6_REUSE_HDUMP=<dir>` + `G6_REUSE_HDUMP_SEQ=<q,..>` and
  `G6_REUSE_ZSCRATCH=1` (`reuse_hdump.cuh`).
- Refused with `--dflash-manifest` (the drafter's prompt-context taps
  are written by the prefill rows a hit skips).
- `engine/tests/test_prefix_reuse.cpp` (CPU, in `scripts/build.sh`) and `engine/tests/test_prefix_reuse.py` (the same
  under ASan/UBSan, plus the `g6m4.cu` hook order) cover the state machine. Measured on an earlier build at `--prefill 4096
  --pf-lanes 2`: hit TTFT 53.8 s vs cold 155.6 s over 14 requests (-65%), hit output identical to cold. Not yet measured
  here.

## 2026-10-05: Rust front end, g6front-rs

- New `frontend/g6front-rs/`: the OpenAI/vLLM front end in Rust (hyper, HF `tokenizers` 0.23.2 as the Python
  front end uses, minijinja). It takes the same flags and gives byte-identical responses to `g6front`.
  `frontend/tests/diff_fronts.py` checks 127 cases, then replays the successful ones concurrently. Opt-in with
  `FRONT=rs` in `launch_tp6.sh`/`launch_tp4.sh` after `scripts/build_front.sh`; Python stays the default until the
  cluster gates run.
- In a linux/aarch64 container, against the Python front end: 64 vs 211 MB resident idle and 95 vs 362 MB under 16
  streams. TTFT added at a 225K-token prompt is 38 vs 316 ms (parallel encode at safe pre-tokenizer cut points). It
  carries about 3.4x the stream chunks per second at less than half the CPU per chunk. Details in
  `frontend/README.md`.
- `mock_engine.py --plain`: no per-request work that grows with the prompt, for front-end benchmarks.

## 2026-10-05: adaptive draft length; TP4 drafts with DFlash2 by default

- `scripts/launch_tp4.sh` now boots the DFlash2 drafter with an adaptive draft count of 1..7 by default. Against MTP k=2
  at 2200 MHz: on answer text code +21%, prose level, structured +46%; on reasoning text about level (code -2.6%,
  within boot noise). It uses less memory, because the MTP layer is not built. `DFLASH=0` brings back MTP k=2 with a
  fixed draft count (setting `MTP=k` does the same). The launcher stops with instructions when a node has no drafter
  pack. TP6 (`launch_tp6.sh`) and multi-session (`--s2`) stay on MTP.
- `--spec-adaptive` chooses each round's draft count for MTP and DFlash2 (`engine/runtime/spec_len.h`, one host
  function per round). Expected tokens come from per-position acceptance, kept per bucket of the drafter's own
  probability of the draft, and shrunk toward the position's recent rate. Cost comes from the verify step time by row
  count at 2200 MHz. The decision reads only inputs that are the same bytes on every rank. MTP also stops its chain when
  one more draft does not pay. The engine flag is off unless given, and fixed k is unchanged. The TP4 launcher gives it
  with DFlash2.
- The DFlash2 selector walk reports each draft's probability among its 16 candidates (`DflOut::lp`, softmax of the walk
  scores). The drafts do not change.
- `engine/tests/test_spec_len.cpp` (CPU) and `engine/ref/dflash2_ab.py` (two feature sets for one token sequence
  through the reference drafter: feature distance and teacher-forced acceptance).

## 2026-10-05: exlr8-e4m3 expert packs, chosen per layer

- The converter reads TR3 and exlr8-e4m3 checkpoints. Both have the same tensor names and layout. The layer file's
  safetensors metadata `exlr8` tells the two formats apart, and the converter checks it against exlr8's shared input
  signs. An exlr8-e4m3 layer's `tr3.*` manifest entries carry `"expert_format": "exlr8-e4m3"`, and the manifest's top
  level maps those layers to their format (`expert_formats`). The converter writes TR3 packs and manifests byte-identical to before, so existing TP4 and
  TP6 packs stay valid. TP6 works from the same checkpoint: the per-128-block vectors take exlr8's 512-channel scale
  groups as they are.
- Decode rounds each codebook value to FP8 E4M3 (RN-even, saturating) on exlr8-e4m3 layers, as the encoder did and as
  prefill already does: `k1_tr3_fc1/fc2/fused<true>`, picked per layer at bind time. The TR3 kernels' SASS is unchanged
  (`G6_NR=6` and `G6_NR=4`). Unit test: `engine/kernels/tests/tr3_e4m3_test.cu`, built by `scripts/build.sh` and run
  there when a GPU is visible.
- exlr8-e4m3 packs need an engine at or after this change. An older engine ignores `expert_format` and runs them
  without the rounding, with no warning. At TP4 that raised KL against the TR3 reference from 0.049 to 0.053.
- At bind time each exlr8-e4m3 layer must have one hidden-side sign vector across its block rows (`tr3.v1`,
  `tr3.v3`), or the rank stops. Rank 0 logs which layers are exlr8-e4m3.
- TP4 at 2200 MHz, exlr8-e4m3 against TR3 on the same build: prefill, TTFT and decode within boot noise, 87K needle
  found, KL against the TR3 reference 0.049 (0.053 without the rounding), top-1 agreement 93.9%.
- `tools/convert/experr.py`: one layer's routed-expert error against BF16 for TR3 and exlr8 checkpoints (CPU).

## 2026-10-05: rejection sampling for drafts (opt-in, `SPEC_ACCEPT=rejection`)

- `--spec-accept rejection` makes verify steps accept drafts by rejection sampling at temperature > 0, for MTP and the
  DFlash2 drafter. Drafts are greedy. The target keeps a draft d with probability p(d) under the request's temperature
  and top_p (its own uniform draw, keyed by seed and position), else it samples p without d. The output distribution is
  still the target's, but a seed no longer gives the same text as without speculation. The default stays `coupled`.
- New head kernels `k_head_local_rs`, `k_head_merge_rs` and `k_head_merge_tp_rs` (`engine/kernels/glue.cuh`), chosen by
  `Model::rs`. With the flag off the head runs the same kernels as before. Temperature 0 gives the same bits in both modes.
  `k_head_local`'s source and PTX are unchanged, but its SASS is not: the shared arrays of `head_local_topp` and
  `head_local_topk`, which now have two calling kernels, move to module scope, so its shared-memory offsets and schedule
  differ from before. A SASS diff against an older build shows this function even with the flag off.
- `engine/runtime/rstest.cu` (build_m4.sh `rs`): the head kernels of all ranks on one GPU, token frequencies against the
  target distribution by chi-square for both modes.

## 2026-10-05: DFlash2 block drafter (opt-in, `DFLASH=1`, TP4)

- `--dflash-manifest/--dflash-pack` (`engine/runtime/dflash.cuh`) replace MTP with the DFlash2 drafter of GLM 5.3. It
  drafts 7 tokens per round in one forward of its 6 layers, from the target's residual after layers 5, 19, 33, 47, 61
  and 75. Each TP4 rank holds 16 query heads, 2 KV heads and 3,072 MLP channels as MXFP8 records (0.70 GB). BF16
  records gave the same acceptance. The drafter runs on its own fabric region (14 collectives) as one captured graph,
  5.6 ms per step at 2200 MHz. Its KV is a 4,096-position ring per layer (25 MB per rank), and taps in the prefill give
  it the prompt's last 2,048 positions.
- `--dflash-sel 1` (default) runs the DFlash2 selector walk over each position's top 16. `--dflash-k K` verifies only
  the first K drafts. Committed tokens are always the target's own samples. At `--dflash-k 2` the output equals MTP
  k=2's byte for byte.
- Target taps: `Model::dfl_tap` and `Prefill::dfl_tap` hooks in host code, null without `--dflash`. The MTP path and the
  TP6 build stay as they were.
- Tools: `tools/convert/dflash2.py` (drafter safetensors -> rank packs, `--head` for lm_head test packs),
  `engine/ref/dflash2_ref.py` (NumPy reference), `engine/ref/dflash2_check.py` with `engine/runtime/dfltest.cu` (the
  step of all ranks on one GPU against the reference), `engine/ref/dflash2_replay.py` (replays a `G6_DFL_DUMP`).

## 2026-10-04: S2 round counters and the S2 shim

- **Round log.** `G6_S2_ROUNDLOG=1` (DEC_S2 build, rank 0) prints one `# s2_round` line per decode step: kind, round id, active sessions of the round, sessions and rows of the step, launch+fence ms, and on main steps the distinct experts per MoE layer. Off by default, no cost. `tools/s2_costfit.py` fits `step_ms = base + per_expert * experts(d*rows) + per_token * rows` per step kind.
- **`frontend/g6s2shim.py`.** PROTOCOL.md v1 on the engine socket over `g6m4 --s2 DIR`, so several requests run at once, with a real `CANCEL`. `tests/fake_s2_engine.py` and `tests/test_s2shim.py` run it without a GPU. See `frontend/README.md`.

## 2026-10-04: logprobs and prompt_logprobs

- The head returns the top 20 tokens of a row on request (`SampleRow::topk`): each rank selects its slice's best 20 and
  sends them after the normal head record; rank 0's merge turns them into ids, logprobs (log-sum-exp over the full
  vocabulary) and the ranks of the sampled and teacher-forced tokens. `engine/kernels/topk.cuh` holds the order, merge
  and rank rule. A step with no asking row sends the same fabric payload as before (boot gate: pbench with logprobs off vs main).
- `--serve` requests take `lp <k>` / `plp <k>`; `resp.<n>` gets `l` lines (before the `t` line) and `p` lines.
- `prompt_logprobs` runs the prompt through decode steps (teacher forced, 8 rows per step), not the prefill kernels, so
  the numbers are decode numerics and the prompt is limited to 4096 tokens by default. S2 sessions reject both keys.
- `frontend/g6shim.py` accepts `logprobs` / `prompt_logprobs` (max 20) and reports `max_logprobs: 20`.
- `engine/tests/test_topk.cpp`: CPU checks of the log-sum-exp, the merge and the ranks.

## 2026-10-04: rank health probe, boot admission, progress-thread isolation

- New `tools/health/`: `rank_probe.py` probes every rank before a boot (RDMA write rate per rail around a ring of the ranks, GPU clock and throttle reasons under load, a driver wedge check, device-memory read and host-to-device copy rates compared between ranks, proxy CPU state) and names the reason for each refusal. `gpuprobe.cu` is its GPU part (built by `scripts/build.sh`, copied by `scripts/stage.sh`). The probe only reads and measures, and refuses a node where a model is booted.
- `scripts/launch_tp6.sh` runs the probe first and refuses to boot (exit 3) when a rank fails; `RANK_HEALTH=0` skips it. The shim and front end now run off the proxy CPUs.
- Engine: `G6_PROGRESS_ISOLATE=1` (default) moves non-proxy threads off the proxy CPUs before the first barrier. The thresholds were checked on six healthy nodes (`tools/health/README.md`).
- `bench/gate/gate.py run` needs `--health FILE` (or `--no-health`) and stores the probe result in every run record; `baseline` and `check` refuse records without an admitted probe result. A missing or failed RDMA write result refuses the rank. Isolation exempts only the registered proxy thread ids.

## 2026-10-04: pair-owner QO prefill at TP4 (`G6_PF_OWN=4`)

- **`G6_PF_OWN=4`** (TP4 builds only; [docs/DESIGN-QO-PAIR.md](docs/DESIGN-QO-PAIR.md)). OWN=3's attention block split
  over the head pairs {0,1} and {2,3}: each rank runs its pair's 32 heads on its own owner rows and its counterpart's
  (rank ^ 2), sends the counterpart's o_proj partial to that rank alone, and sums the two partials in a fixed order.
  The pair weights roll through two slots per layer, built from the rank pack and the pair partner's half over RDMA
  (`G6_PF_QP_ROLL`, a new bulk fabric on port + 56), so the cost is about 1.3-1.4 GB of device memory plus 84 MB pinned per rank instead of the 13 GB QO pack (about
  1.4 KB per position of it scales with `G6_PF_QP_CTX`).
  `tools/convert/pairpack.py` builds resident pair packs (`--qp-pack`) for smaller pools. Ported from adapt's TP4 tree.
- **`launch_tp4.sh` runs `G6_PF_OWN=4` by default** (`QP_ROLL=2`, `QP_CTX=$CAP`, `QP_EARLY=1`). At 2200 MHz: 8K/32K
  prefill 1,318/1,288 tok/s (OWN=1: 816/654), 87K TTFT 73.3 s (156.9 s), decode unchanged within noise, lowest
  MemAvailable 5.2 GiB at a 393,216 pool with a 250K prompt.
- Fabric: reduce-scatter collectives can go to one peer (`Fabric::Rs::p2p`), and bulk jobs can address one peer
  (`BulkJob::peer`). Defaults keep the old behaviour. The TP6 g6m4 and m4sim SASS is identical to main.

## 2026-10-04: TP4 (four Sparks) end to end, and pinned host-to-device copies

- **TP4.** The engine builds and runs at `G6_NR=4` (`G6_NR=4 scripts/build.sh`). Every per-rank width that was a TP6
  literal (attention head groups, vocab slice, routed-expert blocks and scratch, dense channels, buffer sizes, rank
  count checks) now comes from `engine/g6nr.h`; at NR=6 the values are the old ones and the TP6 GPU code is unchanged.
  The converter takes four GPTQ dirs (`tp4/r0..r3`) and writes four rank packs (`plan.set_nr`, `nr: 4` in the
  manifests); `qopack.py` takes four rank packs; `build_packs.sh` picks TP4 from the GPTQ dir. New `launch_tp4.sh` and
  `config/cluster.env.tp4.example`; `build.sh BUNDLE_LIBS=1` and `stage.sh SHARED=1` for nodes without the CUDA
  toolkit and packs on a shared mount. TP4 runs `G6_PF_OWN=1`: the 13 GB QO pack does not fit next to a 90 GB TP4
  rank pack.
- **Pinned host-to-device copies** (`engine/h2d.h`, `g6h2d::copy`). Every copy from pageable host memory to the GPU
  goes through a per-thread pinned buffer, so a 64 KiB-page kernel cannot hang on it. Host code only.

## 2026-10-04: gate harness (`bench/gate/`)

- New: `bench/gate/gate.py` measures a build (KL drift, needle, long prompt, pbench, workloads, drafter counters), appends one JSON line per run, builds a baseline per topology and checks a build against it. See `bench/gate/README.md`. No engine change.

## 2026-10-04: S2 multi-session snapshot (since b85a7a6, the K1H2 snapshot)

The default build (`scripts/build.sh`) is still K1H2. The new multi-session code is compiled in only with
`EXTRA=-DDEC_S2 scripts/build.sh`, and it is turned on at run time with `g6m4 --s2`. See KNOWN-ISSUES.md: S2 still has an open crash on long multi-chat decode.

- **S2 batched multi-session decode (opt-in, `-DDEC_S2`).** New files: `engine/runtime/s2.cuh`, `s2_batch.h`,
  `draft_shape.h`, `engine/runtime/s2/*` (scheduler, sessions, run pool, control channel, native dispatch).
  - Up to 8 sessions decode in one step. Sessions are grouped into MTP row classes (S = 3..8 at mtp 2 use ceil(S/2) groups per round).
  - Each session's KV is one contiguous run in a shared pool. A slot can move to a new base, and the 2->1 / S=1 transitions are handled.
  - Prefill can start at a base page. Below R the loop runs on the CP binding (`g6m4 --s2`).
  - The rank-0 shim talks to the native adapter over persistent control sockets.
- S2 slot views are sized for MX decode rows, not for the prefill chunk. This saves about 8.9 GB per rank at the 614K boot.
- A batched S2 chain draft step now has one row per session.
- If a step fails, S2 prints the fabric / attention / CP error words before the fail-closed check.
- New per-step timer `G6_STEP_TMR=1`, off by default. It prints a per-session request/event label for each step (TAIL / DECODE1 / ROUND / MIXED).
- Control-path disconnect logging: on a recv/send failure the engine prints ret, errno, remaining bytes and frame counts before it throws the epoch-fatal error.
- SELX (`engine/attn/selx.cuh`): our own exact top-2048 select for prefill index tiles. It runs on tiles with n >= 32 rows and context >= `G6_PF_SELX_CTX`. Its output is identical to SEL32's. Unset = SEL32 (unchanged behaviour).
- CP decode `-DCP_F1A`: the phase-2 per-partial wait + get is fused into one launch (off by default).
- The front end metrics have a hook for S2.

## Earlier: multi-session (S2) fixes on six Sparks

### Multi-session decode fault ("FABRIC" error word)

- **Symptom:** the engine stopped (all ranks rc=3) with `fabric 1627389953` (or `...57`), `attn 0 cp 0`, on a main step
  with 6 rows, after minutes of steady decode with two or more chats. Ranks 1-5 logged
  `control recv failed: ret=-1 errno=11 ... remaining=4 of 4 bytes`.
- **Cause:** each worker's control thread waited for the next ADMIT/CANCEL frame with the socket receive timeout
  (`G6_S2_TIMEOUT`, default 120 s). In steady decode no frames are sent, so after 120 s the first worker to time out
  left the epoch before its next main step. The other ranks then hit the fabric wait timeout (~9.3 s) and raised the
  error word naming that worker (p in the low byte). Rank 0 is the leader and never waits idle, so it logged nothing.
- **Fix:** the event reader waits without a timeout until a frame starts (a partial frame still times out); TCP
  keepalive (10 s + 4 x 5 s) and `TCP_USER_TIMEOUT` 30 s detect a lost peer instead.
- **Checked:** main with `G6_S2_TIMEOUT=20` failed the same way at main step 161; the fix ran 9 waves of S = 2/3/4
  chats (8K prompts, gen 4096) for 50 min, 29,997 main steps, at the default timeout with no fault.

### S=3/4 descriptor race

- A batched KV-append descriptor was read while it was still zero, before its upload finished (the memset and the upload ran on different streams).
- After the fix: no fault, and exact outputs at S=2/3/4 in 4 of 4 passes, all six ranks rc=0.
