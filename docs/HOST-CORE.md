# One host core, per-model GPU modules

Status: design, not started. Inventory taken at 35a8f1c.

## Why

The engine binary is a compile of `engine/runtime/g6m4.cu` per model (GLM-5.3, GLM-5.3-Flash, DeepSeek-V4.1-Flash), per
TP width and per build variant (int4 KV, `DS41_MX`, the DS41 trunk). Every compile carries the whole program: option
parsing, bootstrap, the S2 service, the scheduler, the KV pool and the loader, as well as the kernels.
`build_m4.sh` then hides each width's copy from the others with `ld -r` and `objcopy`, and `scripts/build.sh` makes one
binary per model.

Only the kernels and the code that sizes and launches them need the model and the width at compile time. The rest
reads them as constants because it happens to be compiled in the same translation unit. The result:

- The host logic exists 7 × 3 times, and the host tests compile it with no model flag, so they test the GLM shape only
  (8 session slots; the Flash cost table at `s2_batch.h:196` is never compiled on the host).
- What a model must provide is not written down anywhere. It is spread over about 80 model `#if`s in `g6m4.cu`,
  about 90 in `engine/runtime/*.cuh`, 53 in `rank.cuh` alone, and three serve adapters of different shapes.
- Two tests (`engine/tests/test_flash_serve_check.py`, `test_flash_serve_prefill_check.py`) cut code out of
  `flash_native.cuh` as text, because that code only exists inside nvcc. The GLM pool and bind wiring
  (`g6m4.cu:3057-3323`) has no unit test.
- Settings that must agree across ranks live in hand-built signature strings (`g6m4.cu:1919`, `2186-2191`, `3459`).

## Target

```
g6m4 (one binary)
├── host core        g++, compiled once
│   CLI and knobs, bootstrap and fabric bring-up, the S2 service, scheduler, sessions, planner (s2_batch.h),
│   KV pool (kvman-rs), tiers and checkpoint indexes, loader planning, logging, signatures
│   reads the model and width as data: ModuleInfo
└── GPU modules      nvcc, one per (model, width, variant)
    kernels, attention, MoE, device buffers, step graphs and their capture, prefill, drafters
    behind one C table: ModuleOps
```

The core picks a module by `--model` and the width (`--ips` count), as `widths.cpp` picks a width today. Inside a module
`NR`, `MX` and the geometry stay `constexpr`, so kernels are unchanged.

### ModuleInfo

What the core needs to know about a module, filled from the module's constants:

- model id and name, architecture, layer count, hidden size, vocabulary, top-k
- `nr`, `mx` (rows a step), `s2n` (session slots in `S2In`), session slot limit
- KV: row bytes and name (`g6kv::ROW`, `g6kv::NAME`), page size, Flash KDA state size
- drafting: MTP depth, DFlash2 (block, window, which widths), DSpark
- capabilities: prefix park, Flash checkpoints, grammars, `GR8_PF_OWN=4`, relaxed accept, Engram
- the fabric plan: collective count and slot bytes per fabric, ports, proxy CPUs (needed before bring-up)
- the runtime-signature fragment for every compile-time choice the module was built with

### ModuleOps

One C table per module. `NativeHooks` (`s2/service.h:62-99`) already is this table in `std::function` form, and
`G6AttnOps` (`attn/ops.h`, a null slot means unsupported) is the precedent for the C form. The core's hooks become thin
lambdas over it.

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
- `S2In`: sized to 32 slots for every model (162 words; GLM carries 42 today). The copy is already strided.
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

Of 59 grouped blocks: H 25, L 28, S 2, and 6 that mix L with device calls. The file launches no kernel itself.

### engine/runtime headers

- Stay as they are: about 40 pure host headers, among them `serve.h`, `grammar.h`, `insert_rule.h`, `ngram_lookup.h`,
  `ctx_lookup.h`, the steering headers, `flash_kv.h` and the small DS41 headers. `ds41_lanes.h` already takes `NR` and
  `MX` as arguments and is the model to copy.
- Need geometry or width as data: `s2_batch.h` (`S2N_SLOTS`, the Flash `S2Cost` reading `G6_NR`), `spec_len.h`,
  `ds41_ops.h`, `ds41_dspark.h`, `ds41_pf_split.h`, `ds41_slots.h`, `ds41_rowshard.h`, `qp_wire.h`.
- Split into a host half and a module half:
  - `rank.cuh` (2,833 lines; 31 Flash, 22 DS41, 48 `NR`). The speculation loop (`Plan`, `step_inputs`, `plan_rows`,
    `mtp_rows`, `mtp_sequence`), profiling tables and print helpers go to the core; `build`, `launch_step` and the
    kernels stay. `mtp_sequence`'s Flash blocks become a `needs_kda_ring` capability.
  - `s2.cuh`: page-list arithmetic (`s2_run_list`, `s2_check_pages`, `s2_list_place`, `s2_cp_map`) to the core;
    `s2_set_pages` and the page kernels behind `bind`/`tier_copy`.
  - `prefill.cuh`: `pf_slot_bytes`, `pf_perm_order` and `pf_env_check` already take `nr` and go to the core; the TP4-only
    stubs become a capability.
  - `dflash.cuh`: meta parsing and checks to the core; the replicated-versus-split body (`OK_NR`, `REP`) stays
    compile-time in the module.
  - `rt_rows.cuh`: lease and eviction policy to the core; gathers and spills in the module.
  - `packfind.h`: loader planning to the core with `nr` and the model as arguments; it hands the module a binding table
    of (name, device pointer, bytes).
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

Each step is one PR. Each keeps serving output the same: host tests, the CI build of every model, and on the cluster one
fresh boot per model with T=0 replies compared for the first request of the boot only (under `--spec-adaptive` later
requests can differ for reasons unrelated to the change; see `tools/ngram/README.md`), plus a `bench/bench.py` gate.

### 0. Guard rails

1. A T=0 reference per model: a fixed prompt list where each prompt is the first request of a fresh boot, kept with
   the gate scripts, so later steps compare against it.
2. Build the unbuilt in-tree tests (`s2/tests/ds41_prefill_test.cpp`, `ds41_prefill_lanes_test.cpp`,
   `ds41_s32_step_test.cpp`, `s32_64k_mtp8_test.cpp`, `batch_ds41_dspark_test.cpp`) or delete them; delete the dead
   `s2/v1_state.h`.
3. Compile the host tests once per model flag, so the Flash and DS41 shapes of `s2_batch.h`, `scheduler.h` and
   `sessions.h` are covered before anything moves.

### 1. Give main a shape (no behaviour change, one TU)

1. Move the GLM serve wiring (`g6m4.cu:2645-3550`, with the pool and bind hooks at 3057-3323) into `s2/serve_glm.cuh`
   with a `GlmContext`, the same shape as `serve_flash` and `serve_ds41`. Add a host test against a fake context, as
   `serve_flash` deserves too, and retire the two text-scraping tests.
2. Cut `main()` into functions with explicit arguments: parse, fabric plan, fabric bring-up, build, capture, serve,
   offline. The shared locals become a struct.
3. One `step_sizes(kind, M)` per model instead of the four copies (`g6m4.cu:2214, 2878, 3839, 3950`) and the draft and
   DSpark ones.
4. One function per model for the runtime signature fragment.

### 2. Host capacities as data

1. `S2N` and `MAX_SESSIONS` fixed at 32 for every model, with the runtime session limit doing the work. Measure the GLM
   step cost of the larger `S2In` first (162 words instead of 42).
2. `S2Cost` takes the width as an argument; `spec_len.h` drops its Flash `#if`.
3. The scheduler's and sessions' model rules (`Config::validate`, `ds41_decode_prompt`, the DSpark row keys) become
   model traits read at run time.

After this, `service.h`, `scheduler.h`, `sessions.h`, `dispatch.h` and `s2_batch.h` compile once and their tests cover
all three models from one build.

### 3. ModuleInfo, still per-model builds

1. Add `ModuleInfo`, filled from each model's constants.
2. Host headers read geometry from it instead of `geom::` and `g6kv::`: the DS41 plan headers (`ds41_ops.h`,
   `ds41_dspark.h`, `ds41_pf_split.h`, `ds41_slots.h`, `ds41_rowshard.h`), `qp_wire.h`, and the host halves of
   `kv_tier.cuh`, `flash_checkpoint.cuh` and `ds41_prefix_cache.cuh`.
3. Width becomes a runtime `nr` in host code (`g6m4.cu`'s 46 `g6nr::NR` uses, `packfind.h:225-233`).
4. The hardcoded GLM name and layer count in the steering check (`g6m4.cu:400-401`) come from `ModuleInfo`.

### 4. Cut the mixed headers

In this order, smallest to largest: `packfind.h`, `prefill.cuh` sizing, `dflash.cuh` meta, `rt_rows.cuh` policy,
`s2.cuh` page lists, `rank.cuh`'s speculation loop. Each host half moves to a header with no CUDA include, gets a host
test, and the device half keeps its current callers.

### 5. ModuleOps for DS41

DS41 goes first: its host plan already feeds device ops through narrow calls, and it has the fewest switches in `s2/`.

1. Define `ModuleOps` and the common boundary header.
2. Implement it for DS41 over `Ds41Context`; `serve_ds41` builds `NativeHooks` from the table.
3. The module reports its fabric plan before bring-up; the core brings fabrics up and passes them to `boot`.
4. Graph tables become module-owned behind `launch` (`exec[NBK][MX+1]` and the DS41 session and decode tables).

### 6. ModuleOps for Flash, then GLM

The same steps. GLM goes last because it serves production and has the largest adapter. The Flash checkpoint and
borrow-idle paths and GLM's tiers and CP rows are the parts with the most host and device interleaving.

### 7. The host core as its own compile

1. `main`, the service, scheduler, planner, pool, tiers, loader planning and fabric bring-up compile with g++ into the
   core. `devimage.cpp` moves into the modules (it calls the CUDA runtime).
2. Modules compile per (model, width, variant), localized with today's `ld -r` and `objcopy` steps; the core selects
   one by `--model` and width. `scripts/build.sh` makes one `g6m4`; the per-model binaries go away.
3. The launcher passes `--model` instead of choosing a binary; `engine_build` hashes the one binary.
4. `m4sim` becomes the core plus a simulated-fabric build of each module, so it runs `Service` like `g6m4`.
5. CI builds modules on separate jobs and links once.

### 8. Later, if wanted

Modules as shared objects loaded at boot (`RTLD_LOCAL`, hidden visibility) instead of `objcopy`, so a module can be
rebuilt alone. `engine_build` then hashes every module file.

## Open questions

- Whether a larger `S2In` costs GLM anything measurable per step (step 2.1).
- Which variants stay: int4 KV, `DS41_MX` 40/48, the FP32 DS41 trunk. Each one kept is another module per width.
- Static state that must be per module or per process: `s2_tl()`, `Ds41PrefixCache::live()` (read at
  `g6m4.cu:4091-4100`), the static `SpecLen` in `S2Cost`, function-static knob caches, about 370 `__device__` globals.
  The device globals are fine per module; the host singletons must move to the core or into the module handle.
- The bootstrap TCP barriers make the order of fabric bring-up part of the contract across ranks. The fabric plan
  from `info()` must reproduce today's order exactly.
