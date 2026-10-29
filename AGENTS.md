# Working on this repo

gr8 serves the full GLM 5.3 on three, four or six GB10 boxes, and every change has to keep its numbers. Most of what follows is
about proving a change does, before it lands.

If you're an agent working here, read this whole file first. If you're a person, the same rules apply to you.

## How changes land

One change per PR, with a gates table (before on `main`, after, same boot conditions), a review, then a squash merge.
The rules, short:

1. No lost wins: prefill, decode, TTFT, context and memory headroom stay within noise of `main` (about 3% on one boot's
   decode; past that, a second boot).
2. A change that cannot keep the numbers, or changes a design choice, stops and goes to the maintainers first.
3. Keep each change to what its issue or PR describes.
4. Every topology builds (`GR8_NR=4`, `6` and `3`), and the others' behaviour does not change unless the issue says so.

Before anything leaves your machine, scrub it: no IP addresses, hostnames, box names, usernames, keys or paths from
your own setup. The config examples use placeholders for a reason.

## Contributor credit

External contributors should add their GitHub username to [CREDITS.md](CREDITS.md) in their contribution PR, with a
short description of their work and links to relevant PRs or commits when available. Extend an existing entry when
adding more work. Keep the full contributor list in the README's Credits section in sync with CREDITS.md.

When preparing a PR on someone else's behalf, ensure their work is credited and preserve their Git author/coauthor
attribution. Describe proposed work accurately without presenting it as a merged or measured improvement.

## Build

On a GB10 with CUDA 13:

    ./gr8 glm53 build               # the engine and the Rust front end, into bin/
    scripts/build.sh bin            # the engine: one g6m4 and one m4sim for TP = 1, 2, 3, 4, 6 and 8
    GR8_NR=4 scripts/build.sh bin4   # TP=4 only, a shorter build (GR8_WIDTHS="3 4" for several)

g6m4 takes its width from `--ips`, m4sim from `--tp N` or the rank packs in `--packs DIR`. Each width is compiled as
its own build was (`engine/runtime/build_m4.sh`), so one build covers every topology. `scripts/build.sh` also builds
and runs `test_spec_len` and `test_prefix_reuse`, and `tr3_e4m3_test` when a GPU is visible. g6m4 links the S2 page allocator from
`engine/kvman-rs`, which build.sh builds with cargo. Where cargo is missing, build that crate (and `engine/loader-rs`)
first with `cargo build --release --manifest-path <crate>/Cargo.toml`.

## Test

`./gr8 test` runs the CPU tests CI runs (`frontend` or `engine` for one half). [docs/TESTING.md](docs/TESTING.md) lists
every test, by what it needs (a CPU, the pseudo engine, one GPU, the release's files, a cluster), with its command.

A kernel or anything else that changes numerics gets an exactness or tolerance test against the path it replaces, and
the PR says which and why a tolerance is safe.

## Launch

`./gr8 glm53 up` is how the cluster starts: the same command on every box, which starts that box's supervisor
(`tools/launch/`). The supervisors find each other (`--peers`, or the mentat group), compare their options, model set
and engine build, and each starts its own rank; `status`, `logs` and `down` work from any box (README, "Running it").
The engine command line lives in `scripts/engine_args.sh`, which the supervisor reads, so change it there. Every option `./gr8 knobs` lists is a flag and a config key, and its flag, scope, default and meaning live in `knobs/launch.yaml` (`knobs/kv.yaml` for `./gr8 kv`).

S2 is the only served path. `scripts/build.sh` builds every TP with DEC_S2, which g6m4 needs. `g6m4 --serve` serves up
to `SESS` requests at once, and at TP=4 (and TP=6 with `DFLASH=1`, the drafter's body replicated) the DFlash2 drafter for a request that runs alone (TP=3 is MTP only).
A change to decode, prefill or the scheduler is measured under S2. Its output must stay byte-identical to the build
before it for the same requests in the same order, alone and sharing steps (m4sim prints every session's lines, so
compare its stdout between the two builds). Sharing steps can change a request's output against its solo run, so
that is not the comparison.

A change to the loader or the converter needs a boot with the hash check on (no `NOHASH=1`).

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
        --kl-ref bench/gate/refs/tp4.json --health $GR8_WORK_DIR/out/rank_health.json
    python3 bench/gate/gate.py check --baseline bench/gate/baselines/tp4.json

`--health` wants the rank health probe's result for this boot: run `scripts/rank_health.sh` with `./gr8`'s options
before `./gr8 glm53 up` (the probe refuses a box with a model booted). It needs `PEERS`, `IPS` and `IPS2` set, and
passwordless ssh from the head to every box. Without them, `gate.py run --no-health` runs, but its
records do not count for `baseline` or `check`.

`./gr8 glm53 bench --tag <name>` runs pbench alone. Numbers for the README come from RigMark 1.0.0 at `reasoning_effort`
low, GPU clocks capped at 2200 MHz.

## kindling.json

`kindling.json` is the README's results table in a form a program can read. When a PR changes the table, it changes
`kindling.json` in the same PR, with the same numbers and rounding. Leave out a number that wasn't measured; don't
write 0 or an estimate. Check it parses: `python3 -m json.tool kindling.json`.

## Things worth knowing

- If a rank dies, the whole engine stops. `./gr8 glm53 status` names the rank and its last log line.
- The runtime calls attention through `engine/attn/ops.h`, one op table per model (`ops_glm.cu`, `ops_flash.cu`). An op
  only one model has stays null in the other's table, so it needs no stub; the file says how to add one.
