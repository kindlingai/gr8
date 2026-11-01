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
| S2 batch planner (with insert rules), control barrier, pool reuse, runs that grow on demand, the pool's block stores (`--kv-store`) in lock-step through memory, a file and a loopback KV node | engine | `scripts/host_tests.sh` |
| the S2 page allocator's properties (no page lost or held twice, growth only from a safe state, no deadlock, ranks agree, across shrinks and grows of the pool), and the SSD and remote page stores (the remote one over loopback TCP; `kvman-remote-bench --isolation` checks RDMA on the hardware) | engine | `cargo test --manifest-path engine/kvman-rs/Cargo.toml` |
| the exlr8 dense decoder against `loader/exlr8.py` (a synthetic fixture), the layout planner (`engine/loader-rs/src/layout.rs`): the counts the engine reads, balance, the fewest pieces moved, and the trellis codebooks (`src/codebook.rs`): every state against the GPU test's hashes, and each layer's codebook from its width files, with the refusals | engine | `cargo test --manifest-path engine/loader-rs/Cargo.toml` |
| the Python reference's trellis codebooks (`engine/ref/g6ref.py`) against the same hashes | engine | `python3 -m unittest discover -s engine/ref -p 'test_*.py'` |
| the v2 expert converter (`tools/convert/v2.py`), with each codebook and its refusals | engine | `python3 -m unittest tools.convert.test_v2` |
| the gate harness | engine | `python3 -m unittest discover bench/gate/tests` |
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
- `bench/gate/gate.py` checks KL, the needle, and prefill and decode against `main` ([AGENTS.md](../AGENTS.md), "The
  gate").
- `./gr8 glm53 bench` runs pbench alone. The README's numbers come from RigMark.
