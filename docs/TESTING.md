# Testing

The tests come in five tiers, by what they need: a CPU, the pseudo engine, one GPU, the release's files, or a booted
cluster. CI runs the first tier on every PR, the halves its files touch (.github/workflows/ci.yml, the changes job).
The rest run by hand, and a PR's gates table says which ones it ran ([AGENTS.md](../AGENTS.md), "How changes land" and "The gate").

## CPU: `./gr8 test`

    ./gr8 test              # everything CI's CPU jobs run (.github/ci/cpu-tests.sh)
    ./gr8 test frontend     # the front end, the shim, the launcher and the pseudo engine (the cpu-frontend job)
    ./gr8 test engine       # the host engine tests, steering, the converter, the gate harness (the cpu-engine job)

Both halves need g++, Python 3 with NumPy, and cargo (rustup reads `frontend/g6front-rs/rust-toolchain.toml`). The
engine half needs Linux. The front-end half also runs on macOS with Apple's clang, and CI runs it there too. On macOS
the launch end to end test skips unless the loopback has the boxes' addresses:

    for a in 127.0.0.{2,3,4} 127.0.{1,2}.{1,2,3,4}; do sudo ifconfig lo0 alias $a up; done

With `GR8_PSEUDO_TOKENIZER` set to a directory of GLM-5.3's tokenizer files, the front-end half also runs the pseudo
engine's tests below. CI sets it. `.github/ci/fetch-tokenizer.sh DIR` fetches the files.

What the two halves cover, and how to run one part alone:

| test | half | command |
|---|---|---|
| S2 batch planner (with insert rules), presence / frequency penalties, whole-reply and thinking-only (`penalty_test`: the head's row logic, a drafted think close, and the per-step tables), control barrier, pool reuse, runs that grow on demand, the pool's block stores (`--kv-store`) in lock-step through memory, a file and a loopback KV node; the planner's round log and its replay; the model-generic S2 tests again in the Flash build's shape; the DS41 prefill, lane and 32-session tests | engine | `scripts/host_tests.sh` |
| the S2 page allocator's properties (no page lost or held twice, growth only from a safe state, no deadlock, ranks agree, across shrinks and grows of the pool), and the SSD and remote page stores (the remote one over loopback TCP; `kvman-remote-bench --isolation` checks RDMA on the hardware) | engine | `cargo test --manifest-path engine/kvman-rs/Cargo.toml` |
| the exlr8 dense decoder against `loader/exlr8.py` (a synthetic fixture), the layout planner (`engine/loader-rs/src/layout.rs`): the counts the engine reads, balance, the fewest pieces moved, and the trellis codebooks (`src/codebook.rs`): every state against the GPU test's hashes, and each layer's codebook from its width files, with the refusals | engine | `cargo test --manifest-path engine/loader-rs/Cargo.toml` |
| the S2 host logic moved to Rust (`engine/scheduler-rs`: KV copy batches, Flash chunk lending and KDA rings, rounds past R, the scheduler, the admission queue, the KV share rule, the request parser); through its C++ headers against transcripts recorded from the C++ it replaced: `scheduler_rs_vectors_test` in `scripts/host_tests.sh` | engine | `cargo test --manifest-path engine/scheduler-rs/Cargo.toml` |
| the Python reference's trellis codebooks (`engine/ref/g6ref.py`) against the same hashes | engine | `python3 -m unittest discover -s engine/ref -p 'test_*.py'` |
| the v2 expert converter (`tools/convert/v2.py`), with each codebook and its refusals | engine | `python3 -m unittest tools.convert.test_v2` |
| the benchmark | engine | `python3 -m unittest discover -s bench/tests` |
| the n-gram draft source replay and A/B tool (#56: table, chain and tree proposals, acceptance, held-out folds; `ab.py` against a stub server) | engine | `python3 -m unittest discover -s tools/ngram/tests` |
| the rank health probe | engine | `python3 -m unittest discover tools/health` |
| the S2 shim | front end | `cd frontend && python3 -m unittest tests.test_s2shim -v` |
| the front end | front end | `cd frontend/g6front-rs && cargo test --release` |
| the pseudo engine's rules (`model/responses.toml`) | front end | `cd frontend/pseudo-engine && cargo test` |
| the launch: options, model sets, handshake, mentat, the KV node (`./gr8 kv`), and four supervisors with stub engines on 127.0.0.1-4 | front end | `python3 -m unittest discover -s tools/launch` |

`cpu-tests.sh` also runs the steering and converter tests and a shell syntax check, and `scripts/host_tests.sh` runs
the host C++ tests. Read them for the full list.

## The pseudo engine: no GPU

`frontend/pseudo-engine` answers the engine protocol from rules in a TOML file, so the front end and the launcher run
end to end without a model. Both tests need GLM-5.3's tokenizer files (the release's top level, or a box's model cache
`tokenizer/` directory).

| test | command |
|---|---|
| the front end and the smoke test against the pseudo engine | `frontend/pseudo-engine/smoke.sh TOKENIZER_DIR` |
| the front end's HTTP API: streaming, tools, thinking budgets, JSON modes, logprobs, `return_step_stats`, deep health (`frontend/tests/test_api.py`, with the weights in `frontend/tests/api_responses.toml`) | `frontend/tests/api.sh TOKENIZER_DIR` |
| `./gr8 pseudo up` end to end: supervisor, handshake, g6pseudo under `--serve`, shim, front end (needs the release builds) | `GR8_PSEUDO_TOKENIZER=DIR python3 -m unittest discover -s tools/launch -p 'test_pseudo_e2e.py'` |

`./gr8 test pseudo TOKENIZER_DIR` runs the first two.

## One GPU

On a GB10 with CUDA 13 (`ARCH=120a` for an RTX 50xx).

| test | command |
|---|---|
| row budget, S2 attention, the KV pool shrunk and grown under live sessions, and the rest of `scripts/c1_tests.sh`'s header | `./gr8 test gpu bin`, and `GR8_NR=4 EXTRA=-DDEC_S2 ./gr8 test gpu bin_s2` for the S2 build |
| m4sim from the release against a PyTorch forward of the same weights (transformers with `glm_moe_dsa`) | `engine/ref/release_forward.py` |

### Every GPU test

Each `.cu` test, by what builds and runs it. "GLM flags" means `scripts/build.sh`'s `K1H2_FLAGS` with
`-DG6_NR=4 -DDEC_S2`, and "Flash flags" means `-DG6_MODEL_FLASH=1 -DG6_NR=2`. Unless a row says otherwise, a test needs
one GPU and makes its own synthetic weights.

`scripts/c1_tests.sh` builds and runs these (`./gr8 test gpu DIR`, or `TEST=name` for one):

| test | checks |
|---|---|
| `row_budget_test` | dense GEMVs and the K1 expert stream give the same bits for M rows as in chunks of 8 |
| `s2_attn_test` | one batched attention call per layer against the per-session calls, bit for bit |
| `kv4_test` | the FP8 and INT4 (`G6_KV4`) KV latent rows: round trip and attention |
| `kvb8_test` | `kv_b` in FP8 against its BF16 form (#50, skipped under `G6_KV4`) |
| `canon_test` | a session's MLA and indexer rows are the same bytes whichever rank and layout wrote them (#152) |
| `kv_resize_test` | the KV pool shrunk and grown under live sessions (#153) |
| `flash_kv_chunks_test` | `GR8_GLM53F_KV_CHUNKS`: Flash's caches in kvchunks regions under one chunk map |
| `weight_check_test` | the weight check's sums, GPU against CPU |
| `steering_boot_cuda_test`, `steering_consume_test`, `steering_cuda_test`, `steering_shared_test` | the steering kernels against the stock ones |
| `s2_steer_test` | S2 batched decode steers each row as solo decode does (S2 builds) |
| `headgroup_test` | sparse prefill attention for more than 16 heads, against a CPU reference |
| `kda_g9_det` | the G9 ON kernels give the same bits run to run |
| `test_flash_cp` | Flash context-parallel decode at TP=2: the caches each rank holds, the merged index lists and the outputs against the replicated path (the lists and caches bit for bit, the outputs within the LSE merge's rounding) |
| `ds41_v2_attn_test`, `ds41_v2_index_test`, `ds41_v2_moe_test`, `ds41_v2_engram_test`, `ds41_v2_ffn_test`, `ds41_v2_mhc_test`, `ds41_v2_timing_test`, `test_ds41_v2_width`, `test_ds41_v2_wire` | the DS41 v2 prefill (`GR8_DS_PF_V2`): its attention, indexer, MoE, Engram, FFN and mHC kernels against the stock ones, its event timing, and its wire (pack, publish, wait, reduce) over mapped slots, all on synthetic data. `TEST=ds41_v2` runs these alone, each in the one configuration its header names |
| `ds41_v2_l20_tail_test` | Synthetic hoisted layer-20 projection, BF16 gather, compressed/index rows and six-segment tail handover against full-row reference; explicit device plus mapped-host allocation budget |
| `test_ds41_v2_send_ffn_opt_gpu` | Synthetic direct remote attention send, split local projection, dual-layout FFN unpack; byte equality, guards and negative controls |
| `test_ds41_v2_s8_combo_gpu` | Synthetic four-rank public State API sequence: Range A then hoisted layer 20 on the same fabric/epochs, all send/FFN modes, ragged shapes, skip-late and tracing; bounded test-only proxy, not RDMA or speed evidence |

`scripts/build.sh` runs `tr3_e4m3_test` (the trellis codebook decode, when a GPU is visible: every state of the mcg,
mul1 and pmcgx codebooks against a host reference and the hashes the Rust and Python references check, and every lane
decode path of the kernels on random tiles; `--cb --k --tiles --values` also checks tiles the exlr8 encoder wrote
against its own decoded values), `l2pf_queue_test`, `k1_split_test` and
`sampler_u01_test` (all three host only; the last checks the sampler's uniform draw for every 24-bit value), the four steering tests and `s2_steer_test`. It builds these without running them:

| test | checks |
|---|---|
| `l2pf_bench` | `GR8_L2PF` prefetch: speed, and exactness against no prefetch |
| `gpuprobe` | the GPU half of the rank health probe (`tools/health/rank_probe.py` runs it) |
| `dtr_test_tp4`, `dtr_test_flash_tp4` | dense trellis matrices, at TP4: dumps for `engine/ref/dtr_ref.py`, and decode timing (needs release entries) |
| the DS41 v2 tests above, and `ds41_v2_fc2_profile` | the v2 down kernels' occupancy on this GPU (a profile, not a test) |

`engine/runtime/build_m4.sh` builds these, by mode:

| mode | test | checks | needs |
|---|---|---|---|
| `rs` | `rstest` | the vocab-split head kernels for rejection sampling, all ranks on one GPU | |
| `dfl` | `dfltest` | the DFlash2 drafter step against `engine/ref/dflash2_check.py` | drafter packs |
| `dfl` | `dflreplay` | `GR8_DFL_BODY_I4` against MXFP8, replaying a `GR8_DFL_DUMP` | drafter packs, a dump |

No script builds the rest. Each file's header has its build line and usage. Each compiles with the flags shown.
`pf_v2_test` and `k1_v2_test` check against `engine/ref/k1_v2_ref.py`, so they
need a pack and the NumPy reference.

| test | flags | checks | needs |
|---|---|---|---|
| `engine/kda/tests/kda_g9_test` | Flash | the G9 ON kernels against the OFF kernels and host math | |
| `engine/kernels/tests/k1_v2_test` | `-DG6_NR=4` | exlr8 v2 decode MoE: dumps for `engine/ref/k1_v2_ref.py`, and timing | a rank's expert pack |
| `engine/kernels/tests/pf_v2_test` | `-DG6_NR=4` | exlr8 v2 prefill MoE: dumps for `engine/ref/k1_v2_ref.py`, and timing | a rank's expert pack |

## The release's files

On a CPU with torch and NumPy:

| test | command |
|---|---|
| the exlr8 dense decoder against `loader/exlr8.py` on the release's files | `engine/loader-rs/tests/exlr8_compare.py` |
| a rank image planned from the release (`--v2-dense`) against an independent derivation, from the formulas or a layout file | `engine/loader-rs/tests/release_check.py`, with `--layout FILE` for a layout |

## A cluster

On a booted cluster, from the head:

- `./gr8 glm53 smoke` passes every case.
- `./gr8 glm53 bench` checks correctness and measures decode, concurrency and prefill; with `--baseline` it gates them
  against `main` ([bench/README.md](../bench/README.md), [AGENTS.md](../AGENTS.md), "The gate").
- T=0 replies between two builds or settings: under `--spec-adaptive` the planner's verify-width estimates learn across
  requests, and the verify's tokens are not bit-identical across widths, so a request's T=0 reply can depend on the
  requests before it. Compare replies of the first request of a fresh boot only, or boot both sides with
  `GR8_S2_FRESH_SPEC=1` (the estimates restart with every request) and `GR8_DFL_REUSE=0` (no drafter context carried
  from the request before), use prompts that share no prefix (or `--reuse-prefix` off: a prefix hit gives the drafter
  only the suffix's context), keep the drafter choice on (`GR8_S2_DSEL`; with it off and `GR8_MTP_KSTOP=2` its MTP
  estimates are not restarted), and on DS41 leave `GR8_DS_PARK` and the prefix snapshot cache off. Then every request
  can be compared.
- The planner's decisions over a whole session: boot with `GR8_S2_REPLAY_LOG=PATH` in `EXTRA_ENV`, serve the requests,
  and replay each rank's `PATH.<pid>` on a build of the change:
  `g++ -std=c++17 -O2 [-DG6_MODEL_DS41=1 | -DG6_MODEL_FLASH=1] -I engine/runtime engine/runtime/s2/tests/round_replay.cpp
  engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm -o round_replay && ./round_replay PATH.<pid>`.
  It passes when the change's planner asks for every step the logged one did and leaves the same sessions active, as
  long and with the same committed tokens after every call; it names the first difference. A Flash log also needs
  `GR8_GLM53F_S2_COST` set as the engine had it and, when that is 1, `-DG6_NR=` the engine's width.

`ds41_v2_w16k_test` exercises wide planner routing/up/down (both relayout modes), capacity admission and Engram wire exactness under an explicit <384 MiB allocation budget. `test_ds41_v2_s8_combo_gpu` also covers compact-width M=8257/12288, all four send/FFN modes, skip-late and tracing. Both are synthetic correctness checks, not speed evidence.
