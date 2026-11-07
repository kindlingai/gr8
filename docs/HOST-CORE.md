# One host core, per-model GPU modules

Status: R0 (#343), R1 (#344) and R2 (#345, #346, #351, #353) done; R3a in progress. Inventory taken at 35a8f1c; reviewed
against the code by a second pass.

## Why

The engine binary is a compile of `engine/runtime/g6m4.cu` per model (GLM-5.3, GLM-5.3-Flash, DeepSeek-V4.1-Flash), per
TP width and per build variant (int4 KV, `DS41_MX`, the DS41 trunk). Every compile carries the whole program: option
parsing, bootstrap, the S2 service, the scheduler, the KV pool and the loader, as well as the kernels.
`build_m4.sh` then hides each width's copy from the others with `ld -r` and `objcopy`, and `scripts/build.sh` makes one
binary per model.

Only the kernels and the code that sizes and launches them need the model and the width at compile time. The rest
reads them as constants because it happens to be compiled in the same translation unit. The result:

- The host logic exists once per model, width and variant. The host tests compile it with a model flag chosen per
  test: most with none (the GLM shape), five with `-DG6_MODEL_DS41=1` (`scripts/host_tests.sh`), none
  of the S2 layer with `-DG6_MODEL_FLASH=1`, so the Flash cost table at `s2_batch.h:196` is never compiled on the host.
- What a model must provide is not written down anywhere. It is spread over about 72 model `#if`s in `g6m4.cu` and 85
  in `engine/runtime/*.cuh` (52 of them in `rank.cuh`), and three serve adapters of different shapes.
- Two tests (`engine/tests/test_flash_serve_check.py`, `test_flash_serve_prefill_check.py`) cut code out of
  `flash_native.cuh` as text, because that code only exists inside nvcc. The GLM pool and bind wiring
  (`g6m4.cu:3057-3323`) has no unit test.
- Settings that must agree across ranks live in hand-built signature strings (`g6m4.cu:1919`, `2186-2191`, `3459`).

## Target

```
g6m4 (one binary)
├── host core        Rust, compiled once, plus a small C++ main and the fabric
│   options and knobs, bootstrap, fabric plan, the S2 service, scheduler, sessions, planner, driver,
│   KV pool (kvman-rs), tier and checkpoint indexes, loader planning (loader-rs), logging, signatures
│   reads the model and width as data: ModuleInfo
└── GPU modules      nvcc, one per (model, width, variant)
    kernels, attention, MoE, device buffers, step graphs and their capture, prefill, drafters
    behind one C table: ModuleOps
```

The core picks a module by `--model` and the width (`--ips` count), as `widths.cpp` picks a width today. Inside a module
`NR`, `MX` and the geometry stay `constexpr`, so kernels are unchanged.

### ModuleInfo

What the core needs to know about a module, filled from the module's constants. R1 carries the part host code branches on
today; the rest joins as the steps that need it arrive:

- model id and name, architecture, layer count, hidden size, vocabulary, top-k
- `nr`, `mx` (rows a step), `s2n` (session slots in `S2In`), session slot limit
- KV: row bytes and name (`g6kv::ROW`, `g6kv::NAME`), page size, Flash KDA state size
- drafting: MTP depth, DFlash2 (block, window, which widths), DSpark
- capabilities: prefix park, Flash checkpoints, grammars, `GR8_PF_OWN=4`, relaxed accept, Engram
- the fabric plan: collective count and slot bytes per fabric, port offset and proxy CPU, in bring-up order. It is a
  function of the module and the knobs (`GR8_DS_PF_LANES`, `GR8_PF_OWN`, `GR8_GLM53F_PF_POOL_OWNER` change it), because
  each `Fabric::init` is its own TCP rendezvous on `port + offset` (`comms/fabric.cpp:100-110`) and the order of those
  calls is part of the contract across ranks
- the runtime-signature fragment for every compile-time choice the module was built with

### ModuleOps

One C table per module, null for unsupported, as `G6AttnOps` (`attn/ops.h:15-39`) does today. There are four hook
layers now: `NativeHooks` (`s2/service.h:62-99`), `DispatchHooks` (`s2/dispatch.h:28-52`), `DriverHooks`
(`s2/driver.h:21-60`) and `s2::Hooks` (`s2_batch.h`). Only `DispatchHooks` and `bind`, `grown`, `retire`,
`fence`, `shrink`, `resize`, `before_work`, `drained` and `flash_room` are module calls. `emit`, `prompt`, `round`,
`terminal` and `settle` are callbacks between host layers and stay out of the table.

| Call | Contract |
|---|---|
| `abi_version()` | the core refuses a module built against another version |
| `info()` | the ModuleInfo above |
| `boot(args, &m)` | device image from the loader's binding table, fabrics from the core, graph capture |
| `pool_geometry(m)` | pages, `place`, `below`, slot pages, tier pieces |
| `bind`, `grown`, `shrink`, `retire`, `fence` | write a slot's run into the main and draft tables; no pool policy |
| `mode(old, new)`, `bucket(draft, ctx)` | solo/batched switch; graph bucket for a context |
| `step_sizes(kind, M)` | per-step collective sizes (computed in four places in `g6m4.cu` today) |
| `launch(view)` | one MAIN, DRAFT or CHAIN step: `StepIn` in, tokens and the error word out |
| `prefill(view)` | one quantum, single or packed, with its sequence number |
| `logprob`, `draft_lp`, `grammar_mask` | per-row read-backs |
| `dfl_start`, `dfl_ok`, `dfl` | DFlash2, optional |
| `dspark_round` | DSpark, optional |
| `snapshot`, `restore`, `tier_copy` | byte movers for prefix park, Flash checkpoints and KV tiers; the indexes stay in the core |
| `resize`, `flash_room`, `errors`, `shutdown` | |

### Boundary types

One common header holds what crosses the table:

- `StepIn`, `StepOut`, `glue::SampleRow`: sized to the largest `MX` (48), every field present. The Flash and DS41 fields
  add about 0.6 KB.
- `S2In` stays per module. It is a device struct that kernels copy into shared memory (`s2_in.h:34-41`), so fixing it
  at 32 slots would change GLM's kernels. The core needs only the host session limit, from `ModuleInfo`, and treats
  `S2In` as an opaque buffer of `2 + 5 * s2n` words.
- `DflIn`/`DflOut`, `Grant`/`Run`, the prefill views.
- `g6rt::Model` (`rank.cuh:417-589`) splits: the host-set fields become a `ModelConfig` passed to `boot`, and the rest
  stays inside the module as an opaque handle. `StepCtl`, `Layer`, `Draft`, `Prefill`, `S2Model` never cross.

## Inventory

Classes: H = host behaviour that becomes data or a trait, L = host-side sizing, graph or buffer planning that moves into
the module, D = device code, S = already shared and model-free.

### g6m4.cu

All of it after line 156 is one `main()`; sections share locals through lambdas.

| Lines | Section | Class |
|---|---|---|
| 156-577 | flags, option checks, DS41 checks, `--check-args` | H (model flags, spec defaults, session defaults) |
| 579-675 | row counts to capture, `NC`, collective order, grammar and QP checks | L, H |
| 677-1160 | fabric bring-up (main, DS41 W runs and lane B, prefill, Flash pool-owner and router-split, draft, DSpark, DFlash2, QP at TP4) | L; the drafter knobs (990-1106) are H |
| 1191-1311 | weight load and build | load S, build L |
| 1312-1619 | steering, drafts, prefill and DFlash2 build, S2 init, `prepare` | L |
| 1726-1906 | graph tables, `launch_main`, capture | L |
| 1908-2121 | Flash serve | H + L |
| 2122-2644 | DS41 serve, W-row prefill hook, lane B | H + L + D |
| 2645-3550 | GLM serve: `RunPool`, `KvTiers`, `NativeDispatch`, `Service` | H (inside the GLM branch) + L |
| 3552-4001 | offline prompt loop | L |

Of 59 grouped blocks: H 25, L 28, S 2, and 6 that mix L with device calls. The file launches no kernel directly, but
it launches captured graphs (`g6m4.cu:1781`) from graph tables that are `main()` locals (1726-1751). Besides `NR` (46
uses) it reads `MP` 83 times, `HID` 38 and `NL` 33 (`model_geom.h:25-33`).

### engine/runtime headers

- Stay as they are: about 40 pure host headers, among them `serve.h`, `grammar.h`, `insert_rule.h`, `ngram_lookup.h`,
  `ctx_lookup.h`, the steering headers, `flash_kv.h` and the small DS41 headers. `ds41_lanes.h` already takes `NR` and
  `MX` as arguments and is the model to copy.
- Need geometry or width as data: `s2_batch.h` (`S2N_SLOTS`, the Flash `S2Cost` reading `G6_NR`), `spec_len.h`,
  `ds41_ops.h`, `ds41_dspark.h`, `ds41_pf_split.h`, `ds41_slots.h`, `ds41_rowshard.h`, `qp_wire.h`.
- Split into a host half and a module half:
  - `rank.cuh` (2,833 lines; 31 Flash, 22 DS41, 48 `NR`). Its speculation loop (`Plan`, `step_inputs`, `plan_rows`,
    `mtp_sequence`, `rank.cuh:2485-2800`) serves only the offline `--prompt` loop and `m4sim`; the serving planner is
    `s2::Batch`. `build`, `launch_step`, graph capture and the kernels (`rank.cuh:747-2160`) stay in the module.
  - `s2.cuh`: page-list arithmetic (`s2_run_list`, `s2_check_pages`, `s2_list_place`, `s2_cp_map`) to the core;
    `s2_set_pages` and the page kernels behind `bind`/`tier_copy`.
  - `prefill.cuh`: `pf_slot_bytes`, `pf_perm_order` and `pf_env_check` already take `nr` and go to the core; the TP4-only
    stubs become a capability.
  - `dflash.cuh`: meta parsing and checks to the core; the replicated-versus-split body (`OK_NR`, `REP`) stays
    compile-time in the module.
  - `rt_rows.cuh`: lease and eviction policy to the core; gathers and spills in the module.
  - `packfind.h`: the planning is already Rust (`g6l_ds41_plan`, `g6l_release_plan`, called at `packfind.h:225-233`).
    What is left in C++ is `RsSink` (pinned staging and device copies, `packfind.h:98-173`) and the `Find` map, which are
    module material.
  - `steering_boot.cuh`, `reuse_hdump.cuh`, `weight_check.cuh`: small, same pattern.
- Move whole into modules: `flash_step.cuh`, `ds41_step.cuh` (already included under `#if` at `rank.cuh:2821-2828`),
  `flash_prefill.cuh`, the `flash_dsa*`, `qo_weights.cuh`, `qp.cuh`, the DS41 ops headers.

### engine/runtime/s2

- Already model-free host code: bootstrap, control, ingress, trace, resources, `pool.h`, `kv_plan.h`, `kv_share.h`,
  `cp_rows.h`, `flash_lend.h`, `flash_ring.h`, `flash_checkpoint.h`, `driver.h`, `service.h`, and the DS41 step-layout
  and lane logic.
- Compile-time model switches in host code: `scheduler.h` (`MAX_SESSIONS` 32 or 8; DS41-only rules in
  `Config::validate`), `sessions.h` (`ds41_decode_prompt`), `dispatch.h` (DSpark row keys), `s2_in.h` (`S2N`).
- Host code with CUDA calls, needing a module call each: `kv_tier.cuh`, `flash_checkpoint.cuh`,
  `ds41_prefix_cache.cuh`, `ds41_dspark_round.cuh`.
- The three adapters differ in shape:

  | | GLM `native.cuh` | Flash `flash_native.cuh` | DS41 `ds41_native.cuh` |
  |---|---|---|---|
  | entry | `NativeDispatch::hooks()`, dispatch hooks only; the pool and bind wiring is in `g6m4.cu` | `serve_flash()` builds every hook and runs `Service` | `serve_ds41()` builds every hook and runs `Service` |
  | prefill | `begin_prefill`/`end_prefill` on two `Prefill`s | `dispatch.prefill`, checkpoint variant | `prefill_seq`, `prefill_pack_seq` |
  | drafter | MTP, DFlash2 | MTP, DFlash2, borrow-idle | DSpark |
  | KV | paged pool, tiers, CP rows | paged pool, KDA slots, checkpoints | a slab per slot, prefix snapshots, park |

### kernels, attention, build, launcher

- The attention triples (`attn.cu`/`bind.cu`/`ops_glm.cu` and the Flash and DS41 equivalents) are separate compile
  units already and are module material as they stand. Everything in `engine/kernels/*.cuh` is header-only and compiles
  inside `g6m4.cu`.
- Compile-time choices that stay inside modules: `G6_NR`, `G6_MX`/`DS41_MX`, `G6_KV4`, `G6_FLASH_KV_FP8`,
  `DS_MHC_TRUNK_BF16`, `PF_F12`, `PF_FC2GL`, `G6_CP`, the `DEC_*` flags. TP4-only types (`QpW`, `QpRoll`, `QpWire`,
  `Prefill::qp*`) change struct layouts by width, which is fine inside a module.
- The launcher picks a binary per model (`tools/launch/models.py`, `supervisor.py` `engine_bin`), finds engines by
  process name (`ENGINES`), and hashes `bin/<engine>` and `bin/lib` into `engine_build`, which ranks compare at the
  handshake. Modules must join that hash.
- `m4sim.cu` (2,529 lines, 32 model `#if`s) repeats much of `g6m4.cu` and drives `s2::Batch` directly, not `Service`.

## Sequence

The host core is written in Rust, one piece at a time, each piece chosen so that moving it removes a coupling (a model
`#if`, a compile-time width or geometry use in host code, a host and device mix, a duplicated block) rather than only
translating code. The pattern is the one `knobs-rs` uses: a static library with a generated C header, `#[repr(C)]`
structs mirrored by a C++ `static_assert` on size (as `s2_in.h:30` does). Anything that takes a `cudaStream_t` stays C++.

Each step is one PR. Each keeps serving output the same, checked three ways: host tests (ported to Rust where the code
moved), a replay of logged rounds (R0), and on the cluster one fresh boot per model with T=0 replies compared for the
first request of the boot only (later requests can differ under `--spec-adaptive`; see `tools/ngram/README.md`), plus a
`bench/bench.py` gate.

### R0. Guard rails

1. Whole-session T=0 comparisons: `GR8_S2_FRESH_SPEC=1` restarts the adaptive verify-width estimates with every request.
   With `GR8_DFL_REUSE=0`, prompts that share no prefix (a prefix hit gives the drafter only the suffix), the drafter
   choice on, and DS41's park and prefix snapshots off, every request's T=0 reply is independent of the ones before it,
   and one boot per side compares a whole prompt set (`docs/TESTING.md`, "A cluster"). The reference replies per model
   are recorded on a cluster boot.
2. The model-generic S2 host tests also built with `-DG6_MODEL_FLASH=1` (`scripts/host_tests.sh`).
3. A round-replay log: `GR8_S2_REPLAY_LOG=PATH` makes each rank's planner write its settings, every driver call and
   every hook's request and answer, and after each outermost call the per-slot state the driver reads back, to
   `PATH.<pid>` (`s2_log.h`; the field lists sit beside
   each struct in `s2_batch.h`, with a size check for most fields left out). `engine/runtime/s2/tests/round_replay`
   reruns the planner from a log and checks every step it asks for and every state it leaves (`s2/round_replay.h`). This is the guard that sees
   planner changes after the first round, which a first-request T=0 check cannot.
4. A floating-point check: `scripts/host_tests.sh` replays a log with `round_replay` built with `-ffp-contract=off` and
   with `-ffp-contract=fast`. Since R3a's learn records (below) the `off` build must keep every learned estimate bit
   for bit and the `fast` build only every step (`--steps-only`): on the test's workload (the drafter choice and the
   confidence stop on) contracted multiply-adds change the estimates' bits from the first round on, without changing
   a step. On the GLM cluster's serving log both builds keep every bit (R3a's boot).
5. The in-tree tests nothing built (`s2/tests/ds41_prefill_test.cpp`, `ds41_prefill_lanes_test.cpp`,
   `ds41_s32_step_test.cpp`, `s32_64k_mtp8_test.cpp`, `batch_ds41_dspark_test.cpp`) are built; two needed fixes for
   fields added since they were written (steering slots, a loaded steering profile). The dead `s2/v1_state.h` is gone.

### R1. ModuleInfo

`engine/module_info.h`: a C struct `G6ModuleInfo` filled at compile time from the build's constants (model, `nr`, `mx`,
session slots, hidden size, layers, MTP layers, vocabulary, experts, top-k, KV row bytes and format, page) and
capability bits for the model behaviours host code branches on (MTP 0, prefill packs, parked slots, prompts as decode
steps, DSpark, DFlash2, the Flash cost table, KDA state, grammars). `g6mod::INFO` is a constant expression, so the
scheduler's and planner's slot counts are array sizes taken from it. It comes first so that later steps read traits from
one place instead of growing a second mechanism. Still one build per model.

The S2 layer's model `#if`s that encode behaviour read it instead: the scheduler's rules (`Config::validate`),
`Sessions::payload` (the prompt rule moves to `s2/prompt_rule.h`, which every build includes), and Dispatch's DSpark keys.
The slot capacities stay compile-time (`S2N` in `s2_in.h` is shared with device code; a check ties it to `ModuleInfo`)
until R2 makes them runtime in Rust. `g6m4` prints the description at boot. `engine/runtime/tests/module_info_test.cpp`
checks it against the constants in each model's shape.

### R2. Scheduler and sessions to Rust (`engine/scheduler-rs`)

Moves `s2/scheduler.h`, `sessions.h`, `kv_plan.h`, `kv_share.h`, `cp_rows.h`, `flash_lend.h` and `flash_ring.h` (about
1,300 lines, integer logic). The boundary is `#[repr(C)]` `Key`, `Slot`, `Run`, `Grant`, `Shrink`, `Resize`, `Quantum`,
`Config` and `g6sch_*` calls (`g6sched.h`). C++ `Config` is the shared `g6sch_config` plus the steering receipt, so its
settings are listed once. At startup the C++ compares a digest of every shared struct's layout (each field's offset,
size and kind) from `g6sched.h` with the same digest from the Rust types (`src/layout.rs`); it covers the copy of
kvman-rs's tier counters too, while kvman-rs's own header stays hand-kept until the crates link as one library (R10).
The Rust takes the slot count at run time; the C++ around it keeps
`MAX_SESSIONS` arrays until R3 removes `S2N_SLOTS`, since 85 sites in the driver and planner size by it. Proof: the
scheduler, driver and pool host tests, the seeded command-stream transcripts recorded from the C++ (below), and
confirmation boots whose round logs (R0) match byte for byte.

R2 lands in four parts:

1. R2a: the crate (`engine/scheduler-rs`, `g6sched.h`) and the small policies first: KV copy batches
   (`kv_plan.h`), Flash chunk lending (`flash_lend.h`), Flash KDA rings and capture sets (`flash_ring.h`), rounds past R
   (`cp_rows.h`). The C++ headers keep their types and call Rust; the state stays in C++ structs with the same layout,
   so C++ copies and owns it as before. Until R2d the C++ they replaced stayed in `s2/tests/ref/` for a differential
   test that ran seeded random inputs through both and required the same results and refusals. Every build, test and image
   links the library (`scripts/build.sh`, `build_m4.sh`, `host_tests.sh`, `c1_tests.sh`, the Dockerfile, CI).
2. R2b: the `Scheduler` itself, against the retained C++ on random command streams.
3. R2c: the request file's parser (`sessions.h` `parse_request`, read the way libstdc++'s `istringstream` read it, so
   the same requests pass and fail), `kv_share.h`'s `share_want` and grant/give rule, and the admission FIFO, now a
   Rust-owned queue. `Sessions` itself stays C++ glue over the `Emitter` (`serve.h`) and the grammar state, which the
   non-S2 path shares. DS41 `img` fields go back to C++ (`ds41_image.h`) through a callback, as steering resolution
   does, so their refusals keep their place in the order of checks.
4. R2d: the reference copies go. The differential test became `scheduler_rs_vectors_test`: the same seeded streams
   (a portable generator, `s2/tests/scheduler_rs_vectors.h`) write every result, refusal and state to a transcript,
   and each transcript's hash was recorded from the C++ (with libstdc++) after the two matched in every model shape.
   `s2/pool.h` stays: it is the FFI its C++ callers (the service, the driver, `g6m4`) use, and it goes when they move
   (R4, R5). `share_apply` goes with it, since its tier resize must hold `RunPool`'s lock.

### R3. The planner to Rust

Moves `s2_batch.h` (about 1,500 lines), `spec_len.h`, `insert_rule.h`, `ngram_lookup.h` and `ctx_lookup.h`. The boundary is
`g6s2_batch_round(order)` with the module's `run`, `lp` and `dfl` as C callbacks. It removes `S2N_SLOTS` from the planner
(`dispatch.h` and `g6m4.cu` keep it until R5 and R8), the Flash `S2Cost` and `SpecLen::flash_cost` `#if`s and the static
`SpecLen`. This is the one step that can change T=0 output, so it lands in two parts, the guard first:

1. R3a, no behaviour change: the round log (version 3) carries everything the port must reproduce. A `learn` record
   after each state record hashes everything else the planner keeps between calls (the per-session and shared
   `SpecLen` estimates, the drafter choice, the confidence stop, the drafter's ring, the per-slot state and counters);
   the `round` and `prompt` hooks are logged; the header carries the step costs (`S2Cost`), which the replay checks
   against its build, the C library and the model. `round_replay --dump-learn LINE` writes the learned state out at a
   learn record, to find the field a mismatch is in. `ModuleInfo` gains the drafter's block rows and the pack size,
   which the host headers now read. Confirmed on the cluster: each rank's log, less the version 3 records, is the
   earlier boots' log, and replays with its learn records.
2. R3b, the port, behind a thin C++ `s2::Batch` over the Rust, so `round_replay`, `batch_replay_test` and the version 3
   format run unchanged against it. The gate is behaviour and speed, not bit-exact estimates: the replay must ask for
   every step of the test logs and of a serving log (`--steps-only`), T=0 must match on the cluster, and the planning
   time per round must not grow (measured before and after, one C call a round, no heap allocation per round in the
   Rust). The learn records stay as the diagnostic: when a step differs, `--dump-learn` shows which estimate drifted
   first. g++ contracts multiply-adds in C++ under every `-std`, so the GB10 planner fuses them and Rust does not unless
   it calls `mul_add`; the last bits of an estimate may then differ, which is acceptable while every step matches. The
   planner's floating point is a few hundred operations a round, so its speed is the per-round overhead, not FMA or
   vectorization. In the port: the Rust exports its state through C getters and the C++ facade hashes it with
   `s2log::Hash`, so the learn record has one implementation; `exp` (`kprob`) is `f64::exp` from the same libm (the header
   names it); `DSelTable::parse`'s `strtod` and `str`'s `%.10g` (it feeds the runtime signature) stay C++ or are tested
   against it.

   Landed as `engine/scheduler-rs` `spec_len.rs`, `ngram.rs`, `planner.rs` and `planner_ffi.rs` (`g6s2_*` in
   `g6sched.h`, in the layout digest). The C++ parses the n-gram table and passes its entries. Each `Batch` call is one
   C call. Each hook is a C function pointer over the facade's `std::function`. The facade catches a hook's exception
   and rethrows it after the Rust returns. `SpecLen`, `DSelTable::parse`/`str`, the knob reads and the Flash cost table stay C++ (the plain
   build's `--spec-adaptive` keeps `SpecLen`). Checked on macOS: the GLM serving logs v2 and v3 replay in full (learn
   records included, so no estimate drifted), as do the R3b baseline logs: `batch_replay_test`'s log in each model shape
   and every log the planner tests wrote under `GR8_S2_REPLAY_LOG` that the C++ itself replays (11,086). The `round`
   call over the v3 log: mean 1.2 us, p50 1.1, p99 2.7 (C++: 1.9, 1.7, 3.8, ten runs each). A round allocates nothing in
   the Rust after warm-up (a counting-allocator test). One C++ bug surfaced and is kept: when an insert rule fires
   exactly at `n_gen`, the verify's draft step reads the token one past the committed ones. The C++ read past its
   vector, and the port reads 0. The session is over and the row's output unused. `s2::Batch` is move-only.

### R4. The driver to Rust

`s2/driver.h` (289 lines) is glue between the scheduler (R2), the planner (R3) and the engine hooks: it applies commits,
grows runs before a round, picks capped rounds past R, runs one scheduler quantum on the planner and reports
completions. After R3 every piece it calls is Rust, so the driver moves whole and owns what it drives:

- State: the driver owns the scheduler's slots and state and the planner (both Rust-owned handles, not C++ mirrors),
  the per-slot payloads, emitted counts and initialised flags, and the capped-round cycle. `Scheduler`'s C++ mirror of
  the slots goes; the service, still C++ until R5, reads the slots through one exported view (`slots()` today).
- Hooks: `DriverHooks` becomes one C table of engine callbacks (`bind`, `retire`, `prefill`, `prefill_pack`, `run`,
  `emit`, `prompt`, `round`, `dfl_ok`, `dfl_start`, `dfl`, `draft_lp`, `terminal`, `grow`, `finished`, `resize`,
  `shrink`, `settle`), each a function pointer and a context, as the planner's hooks are since R3b. This table is the
  first half of R5's `ModuleOps`.
- The pool: `grow`, `finished`, `bind`, `retire`, `shrink` and `resize` reach the pool through g6m4's lambdas today.
  Linking kvman-rs into the same library as scheduler-rs (one Cargo workspace, one staticlib; R10 needs it anyway)
  lets the driver call the pool Rust to Rust, under the lock the C++ `RunPool` takes today (the head's control thread
  reads the pool). `s2/pool.h` then serves only the C++ still left (g6m4's tier and checkpoint wiring, until R8), and
  `kv_share.h`'s `share_apply`, which orders pool calls, moves beside the driver.
- The pool's lock. `RunPool` (`pool.h`) takes one `std::mutex` for each call and holds it only for that call. The
  compute thread changes the pool on every rank (bind, grow, retire, park, the resize hook's `share_apply`). On the
  head, the control thread also changes it at an event (a cancel's retire, `make_room`'s shrink, `count`, `match` then
  `take`, `reserve`) while the compute thread waits in the event boundary. Outside an event it runs at the same time as
  the compute thread: `publish_status` reads `free_pages`, `room` and `stats`, and the share poll's `coldest` updates
  the pool's ages. Workers' control threads never touch the pool. The sequences of several calls (`match` then `take`,
  `fits` then `reserve`, `share_apply`) hold because the other thread is waiting at the boundary, not because of the
  lock, and that stays true when the lock moves. R5 must keep that pause when the control loop moves to Rust,
  and must not read the pool back to back: Rust's `Mutex` is not fair, and a reader that locks again at once starves
  the compute thread.
  The lock moves into kvman-rs at the same granularity:
  - `Pool` sits behind a `Mutex` in its handle. Every C entry point and every Rust caller locks it for one call.
    `RunPool` loses its own mutex and stays a thin face. A decode round makes one `grow` per decoding slot, and an
    uncontended Rust `Mutex` costs what `std::mutex` does.
  - The pool never calls out while it holds the lock. That keeps the order acyclic: Flash takes `ck_mu` and then calls
    the pool, and kvman-rs's own locks (the RAM stripes, tier buffers) are taken inside pool calls. When `share_apply`
    moves to Rust (R4c), it locks for `resize_tier` and drops the lock before `run_io`, which calls back into C++.
  - `g6kv_list` is the one call that returns a pointer into the pool. It copies into the caller's buffer under the
    lock, as `RunPool` does today. Callers use it per bind, grow and retire, not per round.
  - Some calls block under the lock today and still will: `io_done` and `add_tier` sync the tier index, and pool logging
    writes to stderr. On the head they delay admission and status during a drain. Moving that I/O outside the lock is
    a possible later change.
  - Not chosen: holding the lock for a whole quantum. Admission and status would wait out a round, which takes
    milliseconds. Also not chosen: a status snapshot that the compute thread publishes, with no lock. `match` and `fits`
    read the whole prefix index, so admission would have to move to the compute thread. That is an R5 question.
  - R4 in PRs, each with a boot. R4a: the driver logic, behind a facade. R4b: the lock moves into kvman-rs's handle
    and `RunPool` drops its mutex, with today's build. R4b2: kvman-rs and scheduler-rs in one Cargo workspace and one
    staticlib. `panic = "abort"` for release goes in the workspace root, since member profiles are ignored. The
    scripts that link two libraries (`build_m4.sh`, `scripts/build.sh`, `c1_tests.sh`, `host_tests.sh`) link one. R4c:
    the driver calls the pool Rust to Rust, the `Scheduler` mirror goes, and `share_apply` moves beside the driver.
    R4a and R4b do not depend on each other.
  - R4b landed. `ffi::Handle` holds the `Mutex<Pool>`, and each `g6kv_*` call on a `g6kv_pool` locks it once, reads
    included. `g6kv_list` takes the caller's buffer and its `cap`, and reports the run's length. `RunPool::list` sizes
    the buffer from `Run::pages`, one allocation as before. kvman-rs still unwinds in dev (a `#[should_panic]` test in
    `remote/server.rs`), so a poisoned lock hands the pool on as it is. No pool call calls a C function pointer.
    Checked: `tests/ffi_threads.rs` (a compute and a control thread on one handle through the C API), kvman-rs's
    `cargo test` and `host_tests.sh` on Linux, and the pool host tests on macOS. Boot pending.
  - R4b2 landed. `engine/Cargo.toml` is the workspace (`hostcore-rs`, `kvman-rs`, `scheduler-rs`; one `Cargo.lock`,
    the same dependency versions). `hostcore-rs` is the one staticlib, `engine/target/release/libg6host.a`; the other
    two are rlibs. Release keeps `opt-level = 3` and `panic = "abort"`; dev unwinds. knobs-rs and loader-rs stay
    separate builds. `cargo test --release` no longer fails at random: a crate with a `cdylib` has no hash in its file
    names, and the tests (unwind) and the binaries (abort) each built it to the same files. kvman-rs and knobs-rs drop
    their `cdylib`s, and `libgr8knobs.so` comes from `cargo rustc --crate-type cdylib`. The ledger's crash test runs
    in dev only, since release `kvman-ledger` has no crash points. Checked: `nm` finds all 97 `g6kvman.h` and all 108
    `g6sched.h` functions in `libg6host.a`; on Linux, `host_tests.sh` passes and `cargo test --release --workspace`
    passes from clean six times with no collision (the remote arena test needs `CAP_IPC_LOCK` in a container); the
    pool and driver host tests pass on macOS. Boot pending.
- Proof: the round log replays unchanged (the driver's calls into the planner are the log's calls), the scheduler and
  driver host tests (`driver_grow_test`, the `scheduler_*` tests, `cp_rows_test`) run against the Rust driver behind a
  thin C++ facade as R3b did, and a confirmation boot's logs equal the earlier boots'. One C call per quantum.

R4a landed as `engine/scheduler-rs` `driver.rs` (`g6s2_driver_*` in `g6sched.h`, its tables in the layout digest). Rust
owns apply, step, the completions, the emit hook's accounting and the capped-round cycle (`cp_rows.rs` called directly),
with the per-slot emitted counts, initialised flags and what the driver reads of a payload (prompt length,
`decode_prompt`). `DriverHooks` crosses as one C table of function pointers and a context. The planner's calls go
through a second table to the C++ `s2::Batch`, which writes the round log, and the scheduler's state is still the C++
`Scheduler`'s, passed per call. A quantum is one call into the Rust; its planner calls come back out through the table
(a decode round: `active` and the completion reads per session, `demote` or `promote`, `round`), as the C++ driver made
them. `Payload` stays C++: the facade copies it into and out of the slot when the Rust says and builds each planner
`Session` from it, so a new `Payload` field does not cross the boundary. Hooks may read the driver back, as the
service's retire does, so the Rust state is `Cell`s behind `&self` and apply and step refuse to be entered from a hook.
`Driver` keeps its API (service.h and the tests build unchanged) and is movable. Checked on macOS: every round log the
driver tests write (5,531 in the default, Flash and DS41 shapes) is byte-identical to the C++ driver's;
`scheduler_rs_vectors_test` gains a driver section (seeded grants, cancels, resizes, growth, capped rounds, early stops,
miscounts and throwing hooks) whose hashes were recorded from the C++ driver and match in every shape; `round_replay
--steps-only` replays the GLM serving logs v2 and v3 and the same 11,092 earlier test logs as the build before it. A
decode round allocates nothing in the Rust after warm-up (a counting-allocator test; `scheduler.rs` `plan` and the
capped-round pick moved to stack arrays for it). Linux: `scripts/host_tests.sh` passes with Rust 1.94.1.
`driver_facade_test` covers the exception, refusal, call-back and move paths.

R4c landed. The Rust driver owns the scheduler's config, slots and counters, and the C++ reads them through one view
(`g6s2_driver_view`): `SchedulerView`, the read half of `Scheduler`'s API, which `Driver::state()` returns and the
service, `Dispatch`, `Progress` and the tests take. `Scheduler` stays for the tests and tools that run one without a
driver. The service hands the driver the rank's pool (`PoolHooks::pool`, the adapter's `RunPool`), and the driver makes
the pure pool calls Rust to Rust (`DriverPool`): growth before a round, then a `grown` hook for the engine's tables and
`Sessions`; a worker's `shrink_run`, checked against the head's; and the release of a leaving session's run when
nothing is retained. The service requires the pool, and the driver refuses `DriverPool::grow` without the `grown` hook.
The head skips the cancel and shrink calls it made while building the commit. Each call locks the pool once, and no
hook runs under the lock. The worker's bind (`count`, `match`, `take`, `reserve`, which Flash and DS41 wrap), `retain`
and `park`, and the resize hooks stay C++. Retain and park stay model work: in R5 they become one `ModuleOps` op,
`leave(slot) -> retained | released`, not driver pool calls. `share_apply` is `kv_share.rs` `apply`: it locks the pool
for `resize_tier` alone and calls `run_io` unlocked, and `kv_share.h` keeps its signature over it. Flash's checkpoint
lease keeps its own C++ sequence, since it resizes the checkpoint index, not the pool. Checked on macOS: the 10,693
round logs that the driver, scheduler and DS41 host tests write in the default, Flash and DS41 shapes are
byte-identical to the build before, with `driver_grow_test` on the driver's pool path; the vectors keep their hashes,
and a new section, `driver_pool` (a head and a worker over real pools: grants, cancels, shrinks, growth, finishes), was
recorded from the service's old pool hooks in each shape; `kv_share_test` against a loopback node prints what the build
before did; `round_replay --steps-only` replays the GLM serving logs v2 and v3 and the same 11,092 test logs; a decode
round with pool growth allocates nothing in the Rust. Linux (Rust 1.99.1): `scripts/host_tests.sh` and `cargo test
--release --workspace` pass. Boot pending.

### R5. Service, control, bootstrap and ingress to Rust; ModuleOps

Moves `service.h`, `bootstrap.h`, `control_*.h`, `ingress.h`, `trace.h`, `event_gate.h` and `resources.h`;
`frontend/pseudo-engine/src/s2.rs` already implements the S2 file protocol in Rust and can be lifted. `ModuleOps` is
born here as the module-facing subset of `NativeHooks` and `DispatchHooks`. Each adapter's `std::function`s become
`extern "C"` trampolines over the same lambdas, so no device code moves. Most of the work is building the context
struct those lambdas need: today they capture `main()` locals by reference (`g6m4.cu:3057-3323` captures the pool, the
tiers, the model, the draft, the fabric and the step buffers). Two PRs:

1. R5a, bootstrap and control: `bootstrap.h`, `control_io.h`, `control_protocol.h`, `control_channel.h` and
   `event_gate.h`. The gate goes with them: the channels drive it, and the event-boundary pause lives in it.
2. R5b, the service and compute side: `service.h` (its control, watcher and compute loops), `ingress.h`, `trace.h`,
   `resources.h`, and `ModuleOps`.

R5a landed as `engine/scheduler-rs` `src/control/` (`g6ctl_*` in `g6sched.h`; its structs in the layout digest). Rust owns
each link's socket (`io.rs`, closed when the C++ `ControlIO` that owns the handle goes), the frames (`wire.rs`), the
gate and the ACK set (`gate.rs`), the head's and workers' channels (`channel.rs`) and the bootstrap's sockets and
exchanges (`bootstrap.rs`: listen, accept, connect with its retry, the hello, release and ACK, `agree_lines`,
`capture_barrier`, `agreement_difference`). The C++ classes keep their APIs (the service, `g6m4.cu`, the Flash and DS41
adapters and the tests build unchanged) and hold one Rust handle each. Callbacks (the digest with the room,
`make_commit`, `on_quit`) cross as a function pointer and a context per call. The facade holds a callback's exception
and rethrows it once the Rust returns, after the Rust failed the epoch as the C++ did. A refusal's message is per thread,
since the head's control and compute threads call one channel at once. No thread is started in Rust and the threads
touch what they touched: on the head the control thread runs `event` (announce, ACKs, commit) and the compute thread
`boundary`, `applied` and `sync`, the watcher `quit` and `fail`; sends to the workers share one lock, and whichever of
the control and compute threads waits reads a worker's link. A worker's control thread is its link's only reader. The
channels hold the links as borrowed pointers, as the C++ did: the links outlive them. The gate's no-event path is
atomics only, and `make_commit` runs with no Rust lock held while every rank waits in the gate, so the pause that makes
the head's multi-call pool sequences safe is unchanged. The socket calls are declared in `sys.rs` (std has no keepalive
or poll), with Linux's and macOS's constants; `control_io.h` compares every constant and the `sockaddr_in`, `timeval`
and `pollfd` layouts with the system headers once (`g6ctl_sys`). `BootstrapConfig`, its `agreement()` string (the
capability, which moves with `ingress.h`), `atomic_file` and `publish_ready` stay C++ until R5b. `ControlIO`'s frame
counters are now methods. Checked: `control_vectors_test` (every frame kind, each decoder on each frame and on cut,
flipped and extended copies, the Encoder and Decoder pieces, `match`, seeded gate and ACK-set sequences, the
agreement diff) matches hashes recorded from the C++ (git 5381433c) in every model shape, and each build decodes the
other's 2,696 frames, which are byte-identical; `control_wire_test` records both directions of each link of a TP3 boot
through a relay (bootstrap, store agreement, capture barrier, held and released admissions with a shrink, a cancel, a
resize, barriers, quit) and matches the C++'s bytes. The control, bootstrap, agreement, hold, drain and loopback TP1/2
tests pass unchanged in each shape on macOS, and `scripts/host_tests.sh` and `cargo test --release --workspace` pass on
Linux. Boot pending.

R5b goes in three parts, in this order. R5b1, the S2 files: `ingress.h`, `trace.h`, and from `bootstrap.h` and
`service.h` the agreement string, `atomic_file`, `publish_ready` and status.s2. R5c: `ModuleOps` and the adapters'
trampolines and context struct, which changes no behaviour, so the loops then port onto a C hook table rather than
inventing trampolines while their logic moves. R5b2: `service.h`'s intake, admission and its control, watcher and
compute loops. Its decisions depend on timing in a running service, so they are first pulled out, in the C++, as pure
functions of explicit inputs and frozen with seeded vectors: the notice target from (boundary, horizon, held,
notice_max), the `queue_.check` sequence from (revision, live, limit, fits, room), and the ADMIT_HOLD drain from (queue
front, command stream). The loop wiring rests on the round logs, the hold, drain and loopback tests and a boot.

R5b1 landed as `engine/scheduler-rs` `src/service/` (`g6svc_*` in `g6sched.h`; its structs in the layout digest). Rust
owns the command and request files (`files.rs` `Ingress`: next, peek, the request and its hash; `parse_command`,
`number`), the capability line and the agreement, ready.s2, `atomic_file`, status.s2's JSON (`status.rs`) and the trace
file (`trace.rs`, one write per event from a reused buffer under its lock). Config::validate's steering half moved with
the capability, and `check_abi` compares `PROFILES_MAX` and `SCALE_MAX`. The C++ keeps its APIs; `publish_status` still
calls the pool hooks, in the same order, and hands their values to the Rust. `resources.h` had no caller and no test
and is deleted rather than ported. `frontend/pseudo-engine/src/s2.rs` was not lifted: its messages differ from the
engine's. Checked on macOS: `service_vectors_test` (commands, numbers, the ingress files with gaps and abort.s2,
capability and agreement over seeded configs and steering receipts, `atomic_file`, `publish_ready`, the trace with its
instants' times masked, status.s2 through the service's own `publish_status`) matches hashes recorded from the C++ (git
05425dab) in every model shape; the 10,936 round logs the driver, scheduler, service and DS41 host tests write in the
three shapes are byte-identical to the build before; the control, hold, drain, steering, image, grammar, pool and Flash
checkpoint tests pass unchanged. Linux: `scripts/host_tests.sh` passes. Boot: TP4 GLM on the stock pool, main then
R5b1, the round logs byte-identical on all four ranks and the 24 T=0 replies matching. The stock-pool logs equal the
earlier boots' with the efcd row store too, so a gate needs no KV node.

R5c landed. `g6s2_module_ops` (g6sched.h) is a module's table: `bind`, `bind_request`, `grown`, `shrink`, `retire`,
`retain`, `park`, `fence`, `before_work`, `drained`, `resize`, `flash_room`, the step calls `bucket`, `mode`, `launch`,
`prefill`, `prefill_seq`, `prefill_pack`, `logprob`, `draft_lp`, `dfl_ok`, `dfl_start`, `dfl`, and the grammar arena and
the penalty tables as data. `scheduler-rs` `module.rs` fixes its layout, and the layout digest covers it with
`g6s2_lp_row`. A null member is a call the module does not have. Each call returns 0, or 1 when the module threw. The
module's `ModuleCtx` holds the exception, and `Module` (`s2/module.h`), the caller's side, rethrows it. The core passes
what it does not read as opaque pointers to the C++ objects (launch, payload, sampling, prefill parts, the output vector, request,
reuse state). R10's boundary types replace them.

The service and `Dispatch` call only the table. `NativeHooks::module` carries it. When it is unset, `FnModule` builds
one over `NativeHooks`' std::function members, so the tests build unchanged. Each adapter's hooks are member functions
of a context struct (`GlmModule` in g6m4.cu, `FlashModule`, `Ds41Module`). The struct's members are references to the locals
the lambdas captured, under the same names, so each body is the lambda's text. One template trampoline per call
(`mod::bind<C, &C::on_bind>`) fills the table. g6m4.cu's Flash additions (DFlash2, borrow-idle) keep their state in
g6m4.cu, so they reach serve_flash as `FlashExtras`, std::functions that its members call.

Still std::function: `PoolHooks` other than retain and park (pool policy over `RunPool`, R8), `share`, `FlashExtras`,
and `DriverHooks`. `DriverHooks` are the driver's calls into the service, and R5b2 folds them into the Rust service.
Their module calls go through the table. retain and park stay two calls, and R4c's `leave(slot) -> retained | released`
waits. DS41 alone parks, and the service refuses `Config::park` without that call. No hook knows whether the run was
kept: GLM's hands it to the pool's `retain` and the pool's keep policy decides, and DS41's and Flash's release it after their
snapshot. An outcome means moving those releases out of the bodies.

Checked on macOS. `module_calls_test` runs two ranks over the real Service, Driver, scheduler, pools and gate, with a
fake module that logs every call with its arguments. Its scenarios cover burst admissions, prefill quanta and packs,
growth, prefix reuse, cancels, a shrink, a resize, DFlash2 and parks. A command meant for mid-run is written inside a
step, and the step waits for its notice in the head's trace. Its transcripts match hashes recorded from the service at
2ecb4b43 in every shape, as std::function hooks and as a table, under clang and GCC 15, and its round logs equal the
old build's. The 10,936 round logs the driver, scheduler, service and DS41 host tests write in the three shapes are
byte-identical to the build before. The control, hold, drain, steering, image, grammar, pool and Flash checkpoint tests
pass unchanged. The CUDA files are not compiled here: a script compared every moved body with its lambda. Linux:
`scripts/host_tests.sh` passes, and GLM, Flash and DS41 build. Boot: TP4 GLM on the stock pool, main then R5c, the
round logs byte-identical on all four ranks and the 24 T=0 replies matching.

R5b2's decisions are frozen. `s2/admit.h` holds them as functions of their inputs: the notice target, the retry gate,
the shrink, the grant, the recall and refusal lines, the ADMIT_HOLD drain and hold, and the idle tests. `service.h`
calls them where the expressions were. `admit_vectors_test` records each over seeded inputs, and records the control
loop's admission path over the Rust `AdmissionQueue` with scripted pool answers. That path covers grants, both stalls,
shrinks, recalls, held chains, commands landing during a chain and resizes in flight. Its hashes match under GCC on
x86_64 and clang on arm64. `scripts/host_tests.sh` passes on Linux, and GLM, Flash and DS41 build. Boot: TP4 GLM, R5c
then this, the round logs byte-identical on all four ranks and the 24 T=0 replies matching.

The loops call four kinds of C++ that stay C++ past R5: the pool policy (`PoolHooks`, until R8), `Sessions` (the
emitters, the resp files, the terminal lines and the counters), the request checks that compile a grammar or read
images, and what `retire_run` reads before it retains or parks a run (`Dispatch::rows`, the driver facade's emitted
tokens and prompt). So R5b2 lands in three PRs:

1. R5b2a, `PoolOps`: the pool policy becomes one C table, `g6s2_pool_ops`, as R5c did for the module. It holds the
   calls that are policy: `match`, `take`, `digest`, `tiers`, `refusal`, `r_rows` and `make_room`, with `pool`, `grow`
   and DS41's run clamp as data. The other members (`free_pages`, `room`, `reserve`, `fits`, `runs`, `stats`, `count`)
   are one-line forwards to `RunPool` in every adapter, and the service calls the pool for them directly. Each
   adapter's policy lambdas become member functions of a context struct. `module_calls_test` logs the forwards through
   a test observer on the pool, against hashes recorded first.
2. R5b2b, the head's control thread (workers keep `receive_events`): `controller`, `event` with its commit, `intake`,
   `drain_admits`, `monitor` and the request maps. They call `admit.rs` (`admit.h` ported, against
   `admit_vectors_test`), the pool, the pool table, the Rust queue, ingress, trace, status and channels, and a host
   table for the request checks and `retire_run`. `Sessions::request` is pure and moves. `FlashLend` is already Rust
   (`g6sch_lend_*`) and becomes the service's state. The state both threads touch (`progress_`, `resizing_`,
   `stopping_`) moves into the Rust service, with C calls for the compute thread. The commit keeps its pool-call order,
   which the workers replay: the cancel's retire, then `r_rows`, `match`, `fits`, an optional `make_room` and a second
   fit, the queue's check, and in a grant `count`, `match`, then `take` or `reserve`. It reads the driver's state only
   inside the event-boundary pause, and the Rust says so where it reads it.
3. R5b2c, the compute thread: `compute`, the staged request, the commit's application and the step loop. `DriverHooks`
   fold into the Rust service. `Sessions` and `Dispatch` stay C++ behind the host table until R9 and R8.

Each one is checked by the round logs, the hold, drain and loopback tests, `module_calls_test` and a TP4 boot. From
R5b2b on, `module_calls_test` also has a tight-pool scenario (a cancel of a live reused run, then an admission that
fits only on the pages it freed, with a worker), and it and `admit_hold_drain_test` run under ASan and TSan.

R5b2a landed. `g6s2_pool_ops` (g6sched.h, `s2/pool_ops.h`) is a module's pool policy: `match`, `take`, `digest`,
`tiers`, `refusal`, `r_rows` and `make_room`, with the pool's g6kv handle, `max_run_pages`, `grow`, `count` and `status`
as data. `scheduler-rs` `pool_ops.rs` fixes its layout and `g6s2_pool_hit`'s (`PoolHit`), and the layout digest covers
both. The calls return 0, or 1 when the module threw, and the context is a `ModuleCtx`, as in ModuleOps. `PoolPolicy`
is the caller's side. When `PoolHooks::ops` is unset, `FnPool` builds the table over `PoolHooks`' std::functions, so the
tests keep their hooks. Each adapter's policy lambdas are member functions of `GlmPool` (g6m4.cu), `FlashPool` and
`Ds41Pool`, over references to the same locals, filled in with `pol::` trampolines. The service calls the pool itself
for free pages, room, reserve, fits, count, stats and runs. Each was a one-line forward wherever it was set. What
differed between adapters is now data: DS41 held room, reserve and fits to one session's slab (`max_run_pages`), only
GLM with block stores counted prompts (`count`), and only GLM showed growth totals and runs in status.s2 (`status`).
`publish_status` reads the pool and `write_status` writes the file, so `service_vectors_test` seeds the second.

Checked on Linux (GCC 12, x86_64). `module_calls_test` logs the service's plain pool calls through a test RunPool that
hides the real one's methods, so production has no hook for it. Its hashes were recorded from the service before the
change (51569016) and hold after it, in every shape, with the pool policy as std::functions and as a table. The 13,731
round logs the host tests write are byte-identical before and after. `scripts/host_tests.sh` and `cargo test --release
--workspace` pass. The GLM, Flash and DS41 engines build on a GB10. Boot: TP4 GLM, the freeze then R5b2a, the round
logs byte-identical on all four ranks and the 24 T=0 replies matching.

R5b2b landed. The head's control thread is `scheduler-rs` `src/service/control.rs` (`g6svc_control_*` in g6sched.h;
`g6svc_control_cfg`, `g6svc_host`, `g6svc_checked` and `g6svc_pool_call` in the layout digest): `controller`, `event`
with its commit, `intake`, `drain_admits` and the watcher's `monitor`, with the request maps, the attempt, event, wait,
refusal and recall state and the borrow-idle `Lend`. The decisions are `admit.rs`, and `admit.h` is a facade over it.
`Sessions::request` and the cache namespace moved too. The C++ `Service::run` still starts the threads: the head's
control thread calls `g6svc_control_run`, every rank's watcher `g6svc_control_monitor`, and workers keep
`receive_events`. The progress the compute thread publishes, a resize in flight and the stop flag are Rust atomics, which
the compute thread reaches through C calls. The Rust borrows the Service's endpoint, ingress, queue and trace, the
driver's slots and counters, and the pool. It makes the plain pool calls itself and the policy calls through
`g6s2_pool_ops`, in the order the workers replay, and reads the driver's state only inside the commit, while every
compute thread waits at the event boundary. The host table holds what stays C++, with PoolOps' conventions: `check`
(the parser's ServeReq, `grammar_ok`, `images_ok`), `retire_run`, `share`, and the policy's `refusal` and `tiers`, which
return C++ objects. An ADMIT's bytes go out before the commit erases its entry, so the event holds its own reference.
The head's plain pool calls are now Rust's, so `module_calls_test` reads them from a test observer on the control
(`g6svc_control_observe`) rather than from its RunPool.

Checked on Linux (GCC 12, x86_64, Rust 1.99.1). `module_calls_test` gained the tight-pool scenario, recorded from the
C++ first (f8b86e29): a follow-up takes a retained run and decodes, a long prompt waits for room, and it fits and
reserves from page 0 once the follow-up's cancel retains its run. Every scenario keeps its hashes in every shape, as
hooks and as tables. `admit_vectors_test`, `service_vectors_test`, `steering_state_test` (which now compares the two
cache namespaces), the hold, drain and loopback TP1/TP2 tests pass unchanged, and the 13,743 round logs the host tests
write are byte-identical before and after. `module_calls_test` and `admit_hold_drain_test` pass under ASan and UBSan,
now in `scripts/host_tests.sh`. Under TSan both report the same races before and after the port: accesses ordered by the
Rust channel's, pool's and trace's locks, which TSan cannot see in uninstrumented Rust. `scripts/host_tests.sh` and
`cargo test --release --workspace` pass. The GLM, Flash and DS41 engines build on a GB10. Boot: TP4 GLM, R5b2a then R5b2b, the round logs byte-identical on all four ranks, the 24 T=0
replies matching and the head's s2 log lines equal apart from times.

R5b2c landed. Every rank's compute thread is `scheduler-rs` `src/service/compute.rs` (`g6svc_compute_*` in g6sched.h;
`g6svc_compute_cfg` and `g6svc_staged` in the layout digest): the idle wait, the event boundary (the module's fence,
the staged request checked against the notice's hash, the pool's and the scheduler's digests, once per event), the
commit's application (the cancel's terminal lines, the payload, the driver's apply, the trace and eventlog commit
lines, a worker's `admitted`, the resize flag, `synchronize_mode`, the progress) and the step loop. `Service::run` still
starts the threads and calls `g6svc_compute_run`. The `DriverHooks` with service logic call into it: `bind` (a worker
makes the head's `count`, `match`, `take` or `reserve` on its own pool and checks the run), `retire`, `finished`,
`settle` (the drain barrier), `terminal`'s drafter lines and `emit`'s first-token trace. The hooks that only forward to
`Dispatch`, `Sessions` or the module stay C++ lambdas. `Sessions` and `Dispatch` stay C++ behind `g6svc_host`, which
gains the compute thread's calls: the driver facade (`idle_boundary`, `digest`, `apply`, `step`), the staged request
(`stage`, `unstage`: the parser's ServeReq), `abort`, `mode`, `counters`, `begin`, `bound` and `release`. Each thread
has its own host context, so their exceptions stay apart. The Driver facade, which keeps the payloads and the planner,
is not ported: the Rust reaches it only through the host. A worker's plain pool calls go through the control, so the
test observer now sees them there.

Checked on Linux (GCC 12, x86_64, Rust 1.99.1). `module_calls_test` gained a `cancel_prefill` scenario, recorded from
the C++ first (e0ee311f): a cancel lands while its session is 384 rows into a 1,500-row prefill, beside a session that
decodes on, and the head's last resp lines join its transcript. Every scenario keeps its hashes in every shape, as
hooks and as tables. `admit_vectors_test`, `service_vectors_test`, the hold, drain and loopback TP1/TP2 tests and the
driver, scheduler and DS41 host tests pass unchanged, and the 13,777 round logs the host tests write are byte-identical
before and after. In the default and DS41 shapes, each scenario's eventlog lines and both ranks' traces, times
masked, equal the C++'s. `module_calls_test`
and `admit_hold_drain_test` pass under ASan and UBSan. `scripts/host_tests.sh` and `cargo test --release --workspace`
pass. The GLM, Flash and DS41 engines build on a GB10. TSan on `module_calls_test` and `admit_hold_drain_test` reports the same places as before the port, under the
Rust names they moved to: each handoff is ordered by a Rust lock TSan does not see. Boot: TP4 GLM, R5b2b then R5b2c, the
round logs byte-identical on all four ranks, the 24 T=0 replies matching and the head's s2 log lines equal.

### R6. Options, signatures and the fabric plan to Rust

Four PRs: each move to Rust follows a C++-only PR that freezes what moves.

- R6a, options and signatures frozen. `main()`'s option parse (about 87 knob reads, 20 model `#if`s, the spec and
  session defaults) becomes `engine_options.h` `parse_engine_options`, a host function that fills `EngineOptions` and
  returns a refusal as its message. The three runtime signature builders become `s2/signature.h`
  `runtime_signature(SigInputs)`. `--check-args` works for every model. Checked by `options_vectors_test` (the launcher's
  argument lists per model and variant and one mutation per refusal, from `scripts/check_args.py`),
  `signature_vectors_test` (a seeded sweep recorded from the C++ before the move) and `--check-args` transcripts of the
  old and new engines.
- R6b, options and signatures to Rust. The same two vectors tests hold unchanged, and the `--check-args` transcripts of
  R6a's engines and R6b's are byte-identical.
- R6c, the fabric plan frozen. The planning half of the fabric bring-up becomes a pure function of `EngineOptions` and
  `ModuleInfo` that returns a `FabricPlan` (per fabric: port offset, proxy CPU, collective count, slot bytes, runs,
  pool); `fabric_bring_up(plan)` creates the `Fabric`s in plan order. A dump of the plan per model and launcher list,
  recorded from the C++ first.
- R6d, the plan and its sizing headers to Rust: `FabricPlan` as `#[repr(C)]`, and the DS41 plan headers (`ds41_ops.h`,
  `ds41_pf_split.h`, `ds41_slots.h`, `ds41_rowshard.h`, `ds41_lanes.h`) and `qp_wire.h` as pure functions of
  `(nr, mx, hid, nl)`. The plan dumps are byte-identical.

Boots: one TP4 GLM boot per PR, the round logs byte-identical and the 24 T=0 replies matching. Flash and DS41 get no
boot. What each PR moves runs before the first CUDA call and reads only argv, the knob table and the build's constants
(or, for a signature, a few values main already has), so a dump of every field it produces, byte-identical for every
launcher list on the real engines, shows the boot starts from the same state.

R6a landed. `engine_options.h` holds the parse that was `g6m4.cu:157-694`: `EngineOptions` has every value `main()`
reads afterwards, `parse_engine_options` returns `""` or the message `main()` dies with (same text, same exit code), and
`describe()` dumps every field. The host pieces it needs moved out of CUDA headers: `rank_host.h` (`die`, `Plan`,
`V2Src`, `parse_ints`), `ds41_engram_boot.h` (the Engram boot reads) and `flash_route_split_mode`; the four device
constants it reads have copies that `g6m4.cu` asserts. `steering::prepare_boot` and `grammar_init` read files, so they
stay in `main()` after the parse: a steering or grammar refusal now comes after the parse's checks, and no successful
boot changes. `--check-args` exits once, after them and before the first CUDA call, in all three builds (it was DS41's
alone, before the prompt and QP checks). `s2/signature.h` builds the three signatures from one `SigInputs`; Flash's
checkpoint suffix joins it after the store is made, and the strings the other builders make (`knob::signature()`,
`ngram_sig`, `kstopb_sig`, the drafter choice's table) are inputs. The DS41 agreement tests use it instead of reading
`g6m4.cu`.

Checked on Linux (GCC 12, x86_64). `signature_vectors_test` was recorded from the three expressions as they were
(bd1a1201): 20,000 strings a model, the hashes the same after the move. `options_vectors_test` runs `options_cases.txt`
(19 GLM, 16 Flash and 9 DS41 launcher lists, then 102, 100 and 106 mutations), each case in its own process: 121 GLM
cases in the default (TP6) shape and at TP4, 116 Flash, 115 DS41. The TP6 shape's GLM mutations start from the TP4 list
with `GR8_PF_OWN=3`, which TP6 takes. `scripts/check_options_coverage.py` reads every shape's transcripts: each of the
93 reachable refusals is reached, and the 7 no case can reach are listed with the reason. `scripts/check_args.py` ran
the same cases through the base commit's engines (built with `scripts/check_args_base.patch`, which adds `--check-args`
to GLM and Flash at the new exit point) and through R6a's on a GB10: of 352 transcripts, 341 are byte-identical. The
other 11 are the changes above: a steering refusal after the parse's lines, or the parse's refusal before main's (three
GLM cases, one Flash, one DS41), and the six DS41 cases (an empty or unreadable prompt, a prompt past `--cap`, a QP
flag) that DS41's old exit passed. `scripts/host_tests.sh` and `cargo test --release --workspace` pass. The GLM, Flash
and DS41 engines build on a GB10. Boot: TP4 GLM, R5b2c then R6a, the round logs byte-identical on all four ranks, the 24 T=0
replies matching and the head's log lines equal apart from memory, load speed and the epoch.

R6b landed. The parse is `engine/scheduler-rs` `src/options.rs` and the signatures `src/signature.rs`; `engine_options.h`
and `s2/signature.h` are facades over `g6opt_*` and `g6sig_runtime` (g6sched.h; `g6opt_knobs`, `g6opt_host`,
`g6opt_strs`, `g6opt_plan`, `g6opt_options`, `g6sig_inputs` and `G6ModuleInfo` in the layout digest, which the facade
checks before it parses). The Rust returns the refusal, or `g6opt_options`: scalars, with strings and lists in its own
storage until `g6opt_free`. The facade rebuilds `EngineOptions` from it (`Plan`, `SpecLen`, `KStop`, `CommonOpts`,
`FabricConfig`), and `describe()` stays C++ over the result, so `options_vectors_test` checks the whole path. Knobs come
through gr8_knobs.h's C API, passed in as function pointers so `libg6host.a` needs nothing from `libgr8knobs.a`'s place
on the link line; `knobs gen` writes their ids to `gr8_knob_ids.rs`, and `env_lint.py` counts those reads. The model
`#if`s are a match on the module's model. The constants the parse read from device headers joined `G6ModuleInfo`
(the DFlash2 round's drafts and its replicated body, `TP_SORT`, the int4 KV build, XGrammar, DS41's markov tile and
draft block), which went to ABI 3; `g6m4.cu` checks them against the headers. atoi, atof, strtoul, the @file reader and
`std::to_string(double)` (printf's %f, which Rust's formatting does not match for every double) call the C library.
Three pieces stay C++: the DSpark knob parse (`ds41_dspark.h`, shared with the DS41 serve path and tests; it reads
knobs only, so it runs first and its result is passed in), `ds41_engram_colls` (it dies on a bad Engram config, so the
Rust calls back into it where the C++ called it) and `coll_exec_order`, which the facade applies to the parsed layers.
`flash_route_split_mode` stays for its other callers; the parse reads its knob itself. `qp_wire.h`'s
`roll_env_refusal` had no other caller and moved. m4sim still parses the shared flags with `engine_opts.h`.

Checked on Linux (GCC 12, x86_64, Rust 1.99.1). The ABI bump changes one number in the parse's rank 0 module line, so
the `options_vectors_test` hashes were rerecorded in that commit alone, from the C++ parse: every shape's dump equals
R6a's with "abi 2" read as "abi 3". The Rust parse keeps those hashes in all four shapes, and the coverage check reaches
93 of 93 refusals (7 unreachable). `signature_vectors_test` keeps its three hashes. `scripts/check_args.py` on a GB10:
the 352 transcripts of R6a's engines and R6b's (121 GLM, 116 Flash, 115 DS41) are byte-identical apart from the ABI
number. `scripts/host_tests.sh` and `cargo test --release --workspace` pass, and the 13,777 round logs the host tests
write are byte-identical to R6a's. The GLM, Flash and DS41 engines build on a GB10. Boot: TP4 GLM, R6a then R6b, the round logs byte-identical on all four ranks, the 24 T=0
replies matching, and the head's log lines equal apart from the module line's ABI, times and memory.

### R7. Step sizes

One Rust `step_sizes(kind, M)` per model replaces the four copies of the collective-size code (`g6m4.cu:2214, 2878,
3839, 3950`) and the draft and DSpark ones, calling the C++ `enqueue_step`. The DS41 sizes include reduce-scatter and
row-shard math over `Fabric::Rs` and `CollMask` (`fabric.h:99-107, 276`, `g6m4.cu:2304-2324, 2458-2476, 2589-2603`),
which need a C layout. Per step, so measure.

### R8. Adapter host halves, one PR per model (GLM, Flash, DS41)

GLM first: `NativeDispatch::hooks()` (`native.cuh:468`) is already the device side, and its pool and tier wiring
(`g6m4.cu:2645-3550` without `NativeDispatch`) is host code over a Rust pool. Then the `KvTiers` index
(`kv_tier.cuh:153-225`; the byte movers stay), the Flash checkpoint index (`flash_checkpoint.h`; `pages`, `state` and
the IO stay in the `.cuh`), and the DS41 prefix-cache policy (`ds41_prefix_cache.cuh:52-330`; its nine CUDA calls become
`snapshot` and `restore`). The text-scraping Flash tests retire here.

### R9. Leftovers

The loader hands each module a binding table of (name, offset, bytes); `RsSink` and `devimage.cpp` stay C++. The offline
prompt loop (`g6m4.cu:3552-4001` with `rank.cuh:2485-2800`) is deleted in favour of `m4sim` or ported last. `m4sim`
becomes the core over a simulated fabric, so it runs the same service as `g6m4`.

### R10. One compile

The core is a Rust static library and a C++ `main` of about 200 lines. Modules compile per (model, width, variant) with
today's `ld -r` and `objcopy` localization; the core selects one by `--model` and width. `scripts/build.sh` makes one
`g6m4`, the launcher passes `--model` instead of choosing a binary, and `engine_build` hashes the one binary. CI builds
modules on separate jobs and links once. Modules as shared objects loaded at boot (`RTLD_LOCAL`, hidden visibility)
can follow; `engine_build` then hashes every module file.

### What stays C++ and CUDA

`rank.cuh:747-2160` (build, bind, launch, capture), the `s2.cuh` page kernels and `s2_set_pages`, `prefill.cuh`, the
`dflash.cuh` bodies, the `rt_rows.cuh` gathers, `flash_step.cuh`, `ds41_step.cuh`, `attn/`, `kernels/`,
`NativeDispatch` (`native.cuh:152-486`, which owns streams and graphs), the fabric and its proxy (`comms/fabric.cpp`;
`kvman-rs/src/remote/verbs.rs` shows Rust verbs are possible later), `devimage.cpp`, `grammar.cc` (XGrammar), and the
byte movers in `kv_tier.cuh` and `flash_checkpoint.cuh`.

### R11. DS41's KV on the shared paged pool (with Chuck)

A DS41 slot reserves a whole slab (`--capacity-slots`) whatever its request's length, and its prefix snapshots and park
duplicate what the pool already has (paged stores, the spare-RAM cache, prefix reuse and stash, checkpoints). GLM and
Flash reserve `prompt + max_tokens` pages, so short requests leave room for long ones, under one admission rule and one
meaning of the per-request limit. This step moves DS41 onto that pool, which removes a model-specific KV path and leaves
R8 one KV model to describe. DS41's rows are already defined per 64-token kvman page (`engine/attn/kvfmt.h`, the DS41
section: ca/ica rows, ratio-2 and ratio-1 layers). The work:

- the DS41 attention and indexer kernels address a slot's slab directly; they take a block table as GLM's and Flash's do;
- prefix snapshots and park become pool stash entries or checkpoints (`ds41_prefix_cache.cuh`, `Config::park`, the
  scheduler's parked phase);
- per-session state that is not KV stays per session (the 128-token windows, Engram, DSpark), as Flash keeps its KDA
  slots (`FlashSlots`).

It is DS41's owner's area, so it waits for him. Nothing in R4 to R10 depends on it: until it lands, those steps move
DS41's slab path beside the pool's callers.

### How much moves

An estimate from line counts: about 8,000-9,000 lines of C++ host code move to Rust (the `s2/` host headers about
4,000, `s2_batch.h` and `spec_len.h` 1,630, about 2,500 of `g6m4.cu`'s 4,106, the plan headers about 600). About 55-60 of
`g6m4.cu`'s model `#if`s go (flags, serve wiring, signatures), and all 10 in `s2/`, `s2_batch.h`, `spec_len.h` and
`s2_in.h`. The 85 in `engine/runtime/*.cuh` stay and become internal to modules. Left as host-side C++: about 1,600
lines of `g6m4.cu` (weight load, build, capture, the device halves of the adapters) and `m4sim.cu` until R9.

## Open questions

- Whether the planner's floating-point decisions survive the port unchanged (R0.4, R3).
- Which variants stay: int4 KV, `DS41_MX` 40/48, the FP32 DS41 trunk. Each one kept is another module per width.
- Static state that must be per module or per process: `s2_tl()` (`native.cuh:114`, read from the GLM hooks at
  `g6m4.cu:3414-3424`), `EvProf` (`rank.cuh:680`), `made` (`rank.cuh:1822`), `Ds41PrefixCache::live()` (read at
  `g6m4.cu:4091-4100`), the static `SpecLen` in `S2Cost`, function-static knob caches (the C++ knob header becomes
  module-only), about 370 `__device__` globals.
  The device globals are fine per module; the host singletons must move to the core or into the module handle.
- The fabric plan must reproduce today's order of `Fabric::init` calls and port offsets exactly, for every knob
  setting that changes it, and keep `capture_barrier` (`s2/bootstrap.h`) after capture.
