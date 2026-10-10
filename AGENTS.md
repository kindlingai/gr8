# Working on this repo

g6 serves the full GLM 5.3 on three, four or six GB10 boxes, and every change has to keep its numbers. Most of what follows is
about proving a change does, before it lands.

If you're an agent working here, read this whole file first. If you're a person, the same rules apply to you.

## How changes land

One change per PR, with a gates table (before on `main`, after, same boot conditions), a review, then a squash merge.
The rules, short:

1. No lost wins: prefill, decode, TTFT, context and memory headroom stay within noise of `main` (about 3% on one boot's
   decode; past that, a second boot).
2. A change that cannot keep the numbers, or changes a design choice, stops and goes to the maintainers first.
3. Keep each change to what its issue or PR describes.
4. Every topology builds (`G6_NR=4`, `6` and `3`), and the others' behaviour does not change unless the issue says so.

Before anything leaves your machine, scrub it: no IP addresses, hostnames, box names, usernames, keys or paths from
your own setup. The config examples use placeholders for a reason.

## Build

On a GB10 with CUDA 13:

    ./gr8 glm53 build               # this cluster's TP, and the Rust front end, into bin/
    G6_NR=4 scripts/build.sh bin4   # the TP=4 engine on its own
    G6_NR=3 scripts/build.sh bin3   # the TP=3 engine
    scripts/build.sh bin6           # the TP=6 engine

`scripts/build.sh` also builds and runs `test_spec_len` and `test_prefix_reuse`, and `tr3_e4m3_test` when a GPU is
visible. A PR builds every topology (CI's engine build does).

## Test

| test | where | command |
|---|---|---|
| S2 batch planner, control barrier, pool reuse | any Linux box with g++ | `scripts/host_tests.sh` |
| row budget, S2 attention, decode heads | one GB10 | `scripts/c1_tests.sh bin`, and `G6_NR=4 EXTRA=-DDEC_S2 scripts/c1_tests.sh bin_s2` for the S2 build |
| the S2 shim | anywhere, stdlib only | `cd frontend && python3 -m unittest tests.test_s2shim -v` |
| the front end | anywhere | `cd frontend/g6front-rs && cargo test --release`; `frontend/README.md` has the API tests |
| the gate harness | anywhere | `python3 -m unittest discover bench/gate/tests` |
| the rank health probe | anywhere | `python3 -m unittest discover tools/health` |
| the exlr8 dense decoder against `loader/exlr8.py` | anywhere (a synthetic fixture), and on the release's files with torch | `cargo test --manifest-path engine/loader-rs/Cargo.toml`; `engine/loader-rs/tests/exlr8_compare.py` |
| a rank image planned from the release (`--v2-dense`) against an independent derivation | the release's files, torch and NumPy | `engine/loader-rs/tests/release_check.py` |
| m4sim from the release against a PyTorch forward of the same weights | one GB10, transformers with `glm_moe_dsa` | `engine/ref/release_forward.py` |
| the launch: options, model sets, handshake, mentat | anywhere, stdlib only | `python3 -m unittest discover -s tools/launch` |
| the launch end to end (four supervisors, stub engines) | Linux (it needs 127.0.0.2-4) | the line above runs it, and skips it elsewhere |

A kernel or anything else that changes numerics gets an exactness or tolerance test against the path it replaces, and
the PR says which and why a tolerance is safe.

## Launch

`./gr8 glm53 up` is how the cluster starts: the same command on every box, which starts that box's supervisor
(`tools/launch/`). The supervisors find each other (`--peers`, or the mentat group), compare their options, model set
and engine build, and each starts its own rank; `status`, `logs` and `down` work from any box (README, "Running it").
The engine command line lives in `scripts/engine_args.sh`. The supervisor and the deprecated ssh launchers
(`./gr8 glm53 --ssh up`) both build it from there, so change it there. Every knob in [KNOBS.md](KNOBS.md) is a flag and a config key.

S2 is the served path. At TP=4 and TP=3 the engine always serves through S2 (`g6m4 --serve`, a DEC_S2 build): up to `SESS` requests at once, and at TP=4 the
DFlash2 drafter for a request that runs alone (TP=3 is MTP only). A change to decode, prefill or the scheduler is
measured under S2, and multi-session output must stay byte-identical to each request run alone. TP=6 still serves one
request at a time, until it moves to S2.

A change to the packs, the loader or the converter needs a boot with the pack hash check on (no `NOHASH=1`).

A change to `./gr8 glm53`, the supervisor or a launcher keeps every knob working, and keeps the checks to failures a
boot cannot survive. A new knob that reaches the engine goes in `scripts/engine_args.sh` and `tools/launch/options.py`
(scope `cluster`), so the handshake compares it.

## The gate

The light gates, on the target topology:

1. both builds compile, and the change's own test passes;
2. one boot serves: `./gr8 glm53 smoke` passes every case;
3. KL against main's reference on fixed sequences is within the boot-to-boot range, an 87K-token needle is found, and
   pbench 8K/32K prefill and prose decode are within noise of `main` or better.

`bench/gate/gate.py` runs all of 3 the same way on every build ([bench/gate/README.md](bench/gate/README.md)):

    python3 bench/gate/gate.py run --url http://127.0.0.1:8100 --topology tp4 --commit <sha> --boot b1 \
        --kl-ref bench/gate/refs/tp4.json --health $G6_HOME/out/rank_health.json
    python3 bench/gate/gate.py check --baseline bench/gate/baselines/tp4.json

`--health` wants the rank health probe's result for this boot. `./gr8 glm53 --ssh up` at TP=6 writes it. Otherwise
run `scripts/rank_health.sh` before `./gr8 glm53 up` (the probe refuses a box with a model booted).

`./gr8 glm53 bench --tag <name>` runs pbench alone. Numbers for the README come from RigMark 1.0.0 at `reasoning_effort`
low, GPU clocks capped at 2200 MHz.

## kindling.json

`kindling.json` is the README's results table in a form a program can read. When a PR changes the table, it changes
`kindling.json` in the same PR, with the same numbers and rounding. Leave out a number that wasn't measured; don't
write 0 or an estimate. Check it parses: `python3 -m json.tool kindling.json`.

## Things worth knowing

- If a rank dies, the whole engine stops. `./gr8 glm53 status` names the rank and its last log line.
