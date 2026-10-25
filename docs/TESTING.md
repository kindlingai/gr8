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

`cpu-tests.sh` also runs the other host C++ tests, the steering and converter tests and a shell syntax check. Read it
for the full list.

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

On a GB10 with CUDA 13 (`ARCH=120a` for an RTX 50xx). `scripts/build.sh` also builds and runs `test_spec_len` and
`test_prefix_reuse`, and `tr3_e4m3_test` when a GPU is visible.

| test | command |
|---|---|
| row budget, S2 attention, decode heads, the KV pool shrunk and grown under live sessions, and the rest of `scripts/c1_tests.sh`'s header | `./gr8 test gpu bin`, and `G6_NR=4 EXTRA=-DDEC_S2 ./gr8 test gpu bin_s2` for the S2 build |
| Flash session checkpoints (`G6_FLASH_CACHE`): poison and restore bit for bit, overwrite, least-recently-used drops, torn or changed blocks refused, through memory, a file and a loopback KV node | `nvcc -O2 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -I engine/runtime engine/tests/test_flash_ckpt.cu engine/kvman-rs/target/release/libg6kvman.a -lpthread -ldl -lm -o test_flash_ckpt`, then `./test_flash_ckpt mem: ssd:FILE remote:HOST:PORT` (a loopback node: `kvman-remote --gb 0.5 --port 0 --bind 127.0.0.1 --min-free-gb 0 --no-lock --ready node.json`) |
| Flash checkpoints on a group lease (`--kv-store ...,group=`, D-331 M8 (c)): ask before a drop, grant and grow, give with ages, least-recently-used drop and packing, restore bit for bit after a move, back to the floor, a refused transaction, two ranks agreeing | `TEST=flash_ckpt_group_test ARCH=120a ./gr8 test gpu DIR` (starts a loopback node with 2 MiB segments) |
| the replicated-GEMV split (`G6_QA_SPLIT`, `G6_IDXQ_SPLIT`, `G6_MTP_SPLIT`) against the replicated path, m4sim stdout byte for byte (needs a model) | `MODEL_ARGS="--packs DIR" scripts/m4sim_split_check.sh` |
| m4sim from the release against a PyTorch forward of the same weights (transformers with `glm_moe_dsa`) | `engine/ref/release_forward.py` |

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
