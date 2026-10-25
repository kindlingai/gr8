# Testing

The tests come in five tiers, by what they need: a CPU, the pseudo engine, one GPU, the release's files, or a booted
cluster. CI runs the first tier on every PR once a maintainer comments `/ci` or `/ready`. The rest run by hand, and a
PR's gates table says which ones it ran ([AGENTS.md](../AGENTS.md), "How changes land" and "The gate").

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
| S2 batch planner (with insert rules), control barrier, pool reuse, runs that grow on demand, the Rust pool against the C++ one it replaced, the pool's block stores (`--kv-store`) in lock-step through memory, a file and a loopback KV node | engine | `scripts/host_tests.sh` |
| the S2 page allocator's properties (no page lost or held twice, growth only from a safe state, no deadlock, ranks agree, across shrinks and grows of the pool), and the SSD and remote page stores (the remote one over loopback TCP; `kvman-remote-bench --isolation` checks RDMA on the hardware) | engine | `cargo test --manifest-path engine/kvman-rs/Cargo.toml` |
| the exlr8 dense decoder against `loader/exlr8.py` (a synthetic fixture), and the layout planner (`engine/loader-rs/src/layout.rs`): the counts the engine reads, balance, the fewest pieces moved | engine | `cargo test --manifest-path engine/loader-rs/Cargo.toml` |
| the gate harness | engine | `python3 -m unittest discover bench/gate/tests` |
| the n-gram draft source replay (#56: table, chain and tree proposals, acceptance, held-out folds) | engine | `python3 -m unittest discover -s tools/ngram/tests` |
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

On a GB10 with CUDA 13 (`ARCH=120a` for an RTX 50xx). `scripts/build.sh` also builds and runs the host tests
`test_spec_len` and `test_prefix_reuse`.

| test | command |
|---|---|
| row budget, S2 attention, decode heads, the KV pool shrunk and grown under live sessions, and the rest of `scripts/c1_tests.sh`'s header | `./gr8 test gpu bin`, and `G6_NR=4 EXTRA=-DDEC_S2 ./gr8 test gpu bin_s2` for the S2 build |
| Flash session checkpoints (`G6_FLASH_CACHE`): poison and restore bit for bit, overwrite, least-recently-used drops, torn or changed blocks refused, through memory, a file and a loopback KV node | `nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -I engine/runtime engine/tests/test_flash_ckpt.cu engine/kvman-rs/target/release/libg6kvman.a -lpthread -ldl -lm -o test_flash_ckpt`, then `./test_flash_ckpt mem: ssd:FILE remote:HOST:PORT` (a loopback node: `kvman-remote --gb 0.5 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --ready node.json`) |
| Flash checkpoints on a group lease (`--kv-store ...,group=`, D-331 M8 (c)): ask before a drop, grant and grow, give with ages, least-recently-used drop and packing, restore bit for bit after a move, back to the floor, a refused transaction, two ranks agreeing | `TEST=flash_ckpt_group_test ARCH=120a ./gr8 test gpu DIR` (starts a loopback node with 2 MiB segments) |
| the replicated-GEMV split (`G6_QA_SPLIT`, `G6_IDXQ_SPLIT`, `G6_MTP_SPLIT`) against the replicated path, m4sim stdout byte for byte (needs a model) | `MODEL_ARGS="--packs DIR" scripts/m4sim_split_check.sh` |
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
| `dec_heads_test` | the head-batched decode attention kernels against the per-head ones, byte for byte (skipped under `G6_KV4`) |
| `kv4_test` | the FP8 and INT4 (`G6_KV4`) KV latent rows: round trip and attention |
| `kvb8_test` | `kv_b` in FP8 against its BF16 form (#50, skipped under `G6_KV4`) |
| `canon_test` | a session's MLA and indexer rows are the same bytes whichever rank and layout wrote them (#152) |
| `kv_resize_test` | the KV pool shrunk and grown under live sessions (#153) |
| `flash_kv_chunks_test` | `G6_FLASH_KV_CHUNKS`: Flash's caches in kvchunks regions under one chunk map |
| `flash_ckpt_group_test` | Flash checkpoints on a group lease, against a loopback KV node |
| `weight_check_test` | the weight check's sums, GPU against CPU |
| `steering_boot_cuda_test`, `steering_consume_test`, `steering_cuda_test`, `steering_shared_test` | the steering kernels against the stock ones |
| `s2_steer_test` | S2 batched decode steers each row as solo decode does (S2 builds) |
| `dhead_i4_test` | `G6_DRAFT_HEAD_I4`: the INT4 draft head against a host derivation |
| `headgroup_test` | sparse prefill attention for more than 16 heads, against a CPU reference |
| `cpgroups` | `G6_CP_ONECHAIN` against the stock CP groups: bytes and time |
| `flash_kpool_stream_test` | the streaming pool scorer against the hi/lo one: bytes and time |
| `flash_sparse_mla_test` | fused sparse MLA prefill against QK, softmax and PV |
| `flash_sparse_mla_v2_test` | `sparse_mla_pf_v2` against `sparse_mla_pf_512`, byte for byte (#10) |
| `kda_g9_det` | the G9 ON kernels give the same bits run to run |
| `flash_dsa_f16_test` | `G6_PF_DSA_F16GEMM` against the full-GV kernel and fp64 |
| `mhc_decode_stage_test` | the decode mHC pinned-partial stage: bytes and time |

`scripts/build.sh` runs `tr3_e4m3_test` (the trellis codebook decode, when a GPU is visible), `l2pf_queue_test` and
`k1_split_test` (both host only), the four steering tests and `s2_steer_test`. It builds these without running them:

| test | checks |
|---|---|
| `l2pf_bench` | `G6_L2PF` prefetch: speed, and exactness against no prefetch |
| `k1_ovl_test` | `G6_ROUTE_OVL` against the default MoE layer |
| `gpuprobe` | the GPU half of the rank health probe (`tools/health/rank_probe.py` runs it) |
| `dtr_test_tp<N>`, `dtr_test_flash_tp<N>` | dense trellis matrices, at each width: dumps for `engine/ref/dtr_ref.py`, and decode timing (needs release entries) |

`engine/runtime/build_m4.sh` builds these, by mode:

| mode | test | checks | needs |
|---|---|---|---|
| `rs` | `rstest` | the vocab-split head kernels for rejection sampling, all ranks on one GPU | |
| `dfl` | `dfltest` | the DFlash2 drafter step against `engine/ref/dflash2_check.py` | drafter packs |
| `dfl` | `dflreplay` | `G6_DFL_BODY_I4` against MXFP8, replaying a `G6_DFL_DUMP` | drafter packs, a dump |

No script builds the rest. Each file's header has its build line and usage. Each compiles with the flags shown.
`flash_dsa_pf_test` fails today: in its W8d case the KV and index caches match, but the block output does not
(rel_l2 156 against the P1 math). `pf_v2_test` and `k1_v2_test` check against `engine/ref/k1_v2_ref.py`, so they
need a pack and the NumPy reference.

| test | flags | checks | needs |
|---|---|---|---|
| `engine/attn/tests/attnk` | GLM, `-DPF_F12` | decode attention kernel arms against the stock chain (`attnk_r10a.sh`, `docs/SETTINGS.md`) | |
| `engine/attn/tests/idxbench` | GLM | times the decode indexer stages | |
| `engine/kda/tests/kda_g9_test` | Flash | the G9 ON kernels against the OFF kernels and host math | |
| `engine/kda/tests/kda_ring_test` | Flash | the KDA carry mode (the MTP verify ring) against stock runs, bit for bit | |
| `engine/kernels/tests/k1_audit` | GLM | the routed-expert MoE reads its weights at the DRAM floor (`docs/SETTINGS.md`) | about 74 GB of memory |
| `engine/kernels/tests/mtp_lastrow_test` | GLM | `G6_MTP_LASTROW` against the full MoE layer (`scripts/mtp_lastrow_check.sh`) | |
| `engine/kernels/tests/k1_v2_test` | `-DG6_NR=4` | exlr8 v2 decode MoE: dumps for `engine/ref/k1_v2_ref.py`, and timing | a rank's expert pack |
| `engine/kernels/tests/pf_v2_test` | `-DG6_NR=4` | exlr8 v2 prefill MoE: dumps for `engine/ref/k1_v2_ref.py`, and timing | a rank's expert pack |
| `engine/runtime/tests/dhead2_test` | GLM | `G6_DRAFT_HEAD2` against a host derivation, and timing | |
| `engine/runtime/tests/flash_dsa_pf_test` | Flash | Flash's dense MLP and DSA prefill against the P1 math | |
| `engine/tests/test_flash_ckpt` | `-DG6_NR=2` | Flash session checkpoints (the table above) | |

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
