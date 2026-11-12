# Releasing gr8

Community recipes, dashboards, and deployments build directly on `gr8`. After `v1.0`, the ecosystem provides two
complementary sources of stability:

- **Numbered `v*` tags on `gr8` (cut from `stable`):** Multi-model engine releases verified across supported models and
  `TP=N` topologies for integrators who want consistent, rock-solid `gr8` versions across models (or a reliable base to
  tune from).
- **Standalone tuned recipes:** Model- and box-count-specific repositories tuned for turn-key out-of-the-box performance;
  each recipe pins an exact `v*` tag or merged `gr8` commit and bumps to newer engine releases when its numbers hold.

## 1. `main`, `stable`, and the three release gates

- **`main` is the single integration branch:** Every contributor change lands on `main` through a reviewed,
  squash-merged pull request. Do not maintain long-lived model, topology, or feature branches in this repository, and do
  not publish recipes that apply out-of-tree engine patch stacks (`git am`) onto older commits. Land model fixes and
  validated speed paths in `gr8` via normal pull requests on `main` behind model/topology dispatch or documented knobs,
  and delete topic branches once their pull request merges.
- **`stable` is a protected fast-forward-only branch:** `stable` is never committed to or merged into by hand. Only the
  automated recipe gate fast-forwards `stable` to a `main` commit that passed hardware testing, keeping a single linear
  history with no cherry-picks or branch drift.
- **Three increasing gate levels (`PR -> main -> stable -> v*` tag):**
  1. **Level 1 (`PR -> main`, every pull request):** GitHub Actions CI green plus the submitter's before/after
     correctness and speed receipts ([AGENTS.md](AGENTS.md#how-changes-land) and [The gate](AGENTS.md#the-gate)).
  2. **Level 2 (`main -> stable`, nightly recipe gate):** Runs `gr8`'s default `GLM-5.3` `TP=4` setup and the
     `kindlingai` base-recipe gate set on GB10 clusters against `main`'s newest CI-green commit:
     - **What blocks `stable`:** Any failed boot or wrong result (smoke, counting, corruption, prompt cache, 128K needle,
       or referenced `KL`) **anywhere** blocks `stable`, and a speed regression beyond noise on **`gr8`'s own default
       setup (`GLM-5.3` `TP=4`)** blocks `stable`.
     - **Tuned-recipe speed does not block `stable`:** If a tuned standalone recipe boots and passes every correctness
       check on the candidate commit, a speed dip on that tuned recipe does **not** block advancing `stable`; instead,
       that recipe keeps its prior faster engine pin (is not bumped) until re-tuned.
     - **Advancing vs. fixing forward:** When all blocking checks pass, `stable` fast-forwards to that commit (and the
       gate runner can tag `main`'s already-built `ghcr.io/kindlingai/gr8:sha-<commit>` image as `:stable` without
       recompiling). If a nightly run fails a blocking check, `stable` stays parked on the last passing commit while
       contributors roll forward a fix on `main` — bisecting and reverting the offending commit only when a forward fix is
       not straightforward. If `stable` fails two consecutive nights, only pull requests that fix the failure or revert
       the offending commit merge to `main` until `stable` is green again.
  3. **Level 3 (`stable -> v*` tag, per release):** Full benchmark and harness gate across the supported `TP=N` model
     configurations on the `stable` commit (cold prefill, single-stream and concurrent decode, greedy reproducibility,
     long-context passkey needles, teacher-forced `KL`, and quality evaluation). Wrong results anywhere or a speed
     regression on `gr8`'s default `GLM-5.3` `TP=4` setup block the tag; a tuned standalone recipe that measures slower on
     the candidate tag does not block cutting the `v*` release and simply remains on its prior pin until updated.

## 2. Numbered releases

Following `v1.0`, releases use `vMAJOR.MINOR.PATCH` (`v1.0.1`, `v1.1.0`, `v2.0.0`) as annotated Git tags cut **only
from `stable` commits** that have passed the Level-3 release gate:

- **`PATCH` (`v1.0.1`):** Bug fixes, test or documentation updates, and internal improvements that keep all existing CLI
  flags, config files, HTTP endpoints, weight/steering formats, and default numerical behavior compatible.
- **`MINOR` (`v1.1.0`):** Backward-compatible additions: new models or TP topologies, new optional flags or knobs in
  `knobs/*.yaml`, additive HTTP fields, or validated default speedups that keep existing launch commands working (with
  any numerical trade documented and reversible by flags).
- **`MAJOR` (`v2.0.0`):** Any breaking change (section 3) that requires users, clients, or recipes to update their
  commands, config files, container setup, or stored artifacts.
- **Pre-releases (`vX.Y.Z-rc.N` or `beta-*`):** Optional test cuts on `stable` before a release; recipes for general use
  avoid pre-release tags. (Note that in `.github/workflows/ci.yml`, `docker/metadata-action` uses its default
  `latest=auto` flavor on tag pushes, so pushing any `v*` or `beta-*` tag — including a pre-release or an older-series
  patch tag — also moves `ghcr.io/kindlingai/gr8:latest`.)

Published release tags are immutable: never delete or force-move a `v*` tag. Ship fixes via `main -> stable` in the next
`PATCH` release.

### What tagging a release builds

Pushing an annotated `v*` tag on `stable` runs the full release pipeline in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

1. CPU tests (`frontend` and `engine`) and `tools/env_lint.py`.
2. Arm64 Rust binaries and all six engine build legs at each leg's supported TP widths (`1, 2, 3, 4, 5, 6, 8` for
   GLM-5.3 `fp8` and `int4`, `1, 2, 3, 4` for Flash `fp8` and `int4`, `4` for `ds41`, plus the GPU test binaries).
3. Reproducible tester archive (`gr8-<short-sha>.zip` + `.sha256` via `scripts/make_zip.py`), uploaded as an Actions
   workflow artifact on the tag run.
4. Runtime container image `ghcr.io/kindlingai/gr8:<tag>` (and `sha-<commit>`), smoke-tested and pushed with build
   provenance attestation.

Note that `ghcr.io/kindlingai/gr8:main` moves on every non-Markdown merge to `main`, `:latest` moves on non-Markdown
merges to `main` as well as on `v*` / `beta-*` tag pushes (`docker/metadata-action` default `latest=auto`), and any
`:stable` image tag moves with the `stable` branch; none of them is an immutable release pin.

## 3. Breaking changes and release notes

A change counts as **breaking** (requiring `MAJOR`, plus migration steps) if it breaks an existing working setup:

1. **CLI and configuration (`./gr8`, `cluster.env`, `switch.env`, `knobs/*.yaml`):** Removing or renaming a `./gr8`
   command (`glm53`, `glm53f`, `ds41`, `switch`, `kv`, `knobs`, `models`, `query`, `perftest`, `test`), removing or
   renaming a documented `--<knob>` flag or `GR8_*` / `G6_*` knob, or tightening/changing accepted values so an existing
   valid config file or command is rejected or changes meaning.
2. **HTTP API and wire protocol (`g6front-rs` on port `8100`, [PROTOCOL.md](PROTOCOL.md)):** Removing or renaming a
   documented endpoint (such as `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`,
   `/health`, or `/metrics`), or removing/changing request or response fields relied on by OpenAI-compatible clients,
   Kindling dash (`tools/console`), or custom front ends.
3. **Weights and steering profiles (`engine/loader-rs`, `tools/steering`):** Dropping support for a currently documented
   Hugging Face `exlr8` model release or a published steering profile schema
   ([docs/STEERING-PROFILE-SCHEMA.md](docs/STEERING-PROFILE-SCHEMA.md)).
4. **Cluster, runtime, or numerical contract:** Raising stock memory or disk requirements so a documented model and TP
   topology with default `--cap` no longer boots on 128 GB GB10 boxes, changing required ports (`7340`, `7341`, `8100`)
   or container entrypoint behavior, raising minimum CUDA/driver/OS requirements, or changing an exact default launch to
   lossy numerics without explicit disclosure and one-line rollback flags.

*(Rotating an internal cache key so a persistent generation or prompt cache rebuilds automatically on first boot after
upgrade is not a breaking change, provided no manual cleanup is required, though release notes must call out the
one-time rebuild time and disk use.)*

### Required GitHub Release sections

Every tagged release has a matching GitHub Release with four short sections:

1. **Highlights and measured numbers** — what improved or was added, with model, TP width, and boot conditions
   ([AGENTS.md](AGENTS.md#the-gate)).
2. **`Breaking changes`** — always present. Write `None.` when the release is backward-compatible; otherwise list each
   changed flag, file, endpoint, or requirement with the exact before/after migration command or config line.
3. **Default or cache-key changes** — any changed default knob, any numerical trade with its exact rollback flags, and
   any cache key that triggers a one-time rebuild on upgrade.
4. **Toolchain / Dockerfile sync** — any change to `CUDA_VERSION`, `rust-toolchain.toml` (`RUST_VERSION`), or runtime
   apt packages in `Dockerfile` that downstream recipe Dockerfiles mirror.

### Cutting a release

1. Confirm the candidate commit is on `stable` (having already passed CI on `main` and the Level-2 nightly gate) and has
   passed the Level-3 full harness gate across the supported `TP=N` model configurations.
2. Walk the keep-current checklist in [AGENTS.md](AGENTS.md#keeping-docs-and-recipes-current) so `README.md`, model
   pages (`docs/GLM-5.3.md`, `docs/FLASH.md`, `docs/DEEPSEEK.md`, `docs/SWITCHING.md`), `KNOBS.md`, `KNOWN-ISSUES.md`,
   `CREDITS.md`, `recipes.json`, and `docs/cards/` graphics match on `stable`. (Pushes to `main` that touch only
   `**.md` skip `ci` via `paths-ignore`, whereas pushing the `v*` tag always runs the full `ci` build.)
3. Create and push an annotated tag on the verified `stable` commit:
   ```bash
   git checkout stable && git pull --ff-only origin stable
   git tag -a v1.1.0 -m "gr8 v1.1.0"
   git push origin v1.1.0
   ```
4. Wait for the tag's `ci` workflow run to publish `ghcr.io/kindlingai/gr8:<tag>`, publish the GitHub Release on `<tag>`
   with the four sections above and the image's `sha256` digest, and update the current release and pin example in
   [README.md](README.md#building-on-gr8) on `main`.
5. Check the blessed standalone recipes against `<tag>`: open pin-bump pull requests for the recipes whose correctness
   and speed hold, and leave any recipe that measures slower on its prior pin (reflected in `README.md`'s recipe
   freshness column) until re-tuned.

## 4. How recipes pin `gr8`

Deployable recipes package one model and box count (container setup, tuned `.env` or `knobs.sh`, and `kindling.json`) in
**their own standalone repositories** — such as `glm-5.3-flash-exlr8-1x-spark`, `glm-5.3-flash-exlr8-2x-spark`,
`deepseek-v4.1-flash-exlr8-4x-spark`, `glm-5.3-exlr8-4x-spark`, `glm-5.3-exlr8-5x-spark`, `glm-5.3-exlr8-6x-spark`, and
community recipe repositories — rather than inside `gr8`. `gr8` stays engine-focused and links to recipe repositories
from its generated `README.md` table, including a **freshness column** comparing each recipe's pinned `gr8` tag or commit
against the latest `v*` release and `stable` so users and contributors can see which recipes are current and which are
candidates to update. Recipes hold configuration and packaging only, never engine patch stacks.

### What to pin

- **Preferred when numbers hold:** Pin an immutable numbered `v*` release tag (`v1.0`, `v1.1.0`, ...).
- **Exact merged `stable` or `main` commit (between tags or for model-specific tuning):** Because `v*` tags land less
  often than model-specific tuning, a recipe — especially for a new model or extra tuning — may land its fixes in `gr8`
  through normal pull requests on `main` and pin that **exact merged `main` or `stable` 40-character commit SHA**
  (`sha-<short-sha>@sha256:<digest>` or submodule at `<commit>`), moving to a later `v*` tag when its numbers hold there.
- **Never pin moving refs or engine patches:** Never pin a floating branch or moving image tag (`HEAD` of `main` or
  `stable`, `:latest`, `:main`, or `:stable`), never pin unmerged pull request or fork commits, and never carry engine
  patch stacks in a recipe.

### Supported pin styles

1. **Image-pinned recipe (`compose.yaml` + `.env`, or thin wrapper `Dockerfile`; no submodule, no engine compile):**
   - Pin `ghcr.io/kindlingai/gr8:<v*-tag-or-sha-<short>>@sha256:<digest>` directly in `compose.yaml` with a `.env` file
     of launcher options, or base a thin wrapper `Dockerfile` on that image layering `knobs.sh`, `entrypoint.sh`, and
     `bin/gr8` on top of `/opt/gr8/bin/*`:
     ```yaml
     # compose.yaml
     image: ${GR8_IMAGE:-ghcr.io/kindlingai/gr8:v1.0@sha256:<digest>}
     ```
     ```dockerfile
     # or thin wrapper Dockerfile
     ARG GR8_IMAGE=ghcr.io/kindlingai/gr8:v1.0@sha256:<digest>
     FROM ${GR8_IMAGE}
     ```
2. **Submodule recipe (`gr8/` submodule, optional binary copy from `ghcr.io/kindlingai/gr8`):**
   - Point the `gr8` Git submodule at the exact commit of a numbered `v*` release tag or a merged `stable`/`main` commit
     on `https://github.com/kindlingai/gr8`:
     ```bash
     git -C gr8 fetch --tags origin
     git -C gr8 checkout <v*-tag-or-commit-sha>
     git add gr8
     ```
   - Keep the recipe `Dockerfile`'s `CUDA_VERSION`, `RUST_VERSION` (matching `gr8/rust-toolchain.toml`), and runtime
     libraries in sync with `gr8/Dockerfile` at that commit, and prefer copying prebuilt `/opt/gr8/bin/*` binaries from
     `ghcr.io/kindlingai/gr8` by digest (`ENGINE_SRC=image`) whenever no custom engine compile flags are needed.

### Knobs, `kindling.json`, and upgrades

- **Explicit recipe tuning, no hidden drift:** Recipes are tuned for their target model and box count; keep recipe-level
  options documented explicitly in `.env` or `knobs.sh` files (such as `TUNE=speed`, `TUNE=context`, or `TUNE=quality`,
  plus container volume and artifact paths) rather than undocumented overrides, and re-verify or drop superseded
  overrides when bumping `gr8`.
- **Record the engine pin in `kindling.json`:** Record an `"engine"` object in the recipe's `kindling.json` with `repo`
  (`"kindlingai/gr8"`), `ref` (`v*` tag, `"stable"`, or `"main"`), 40-character `commit`, and `pin` (`"image"` or
  `"submodule"`, plus `image` and `image_digest` when pinning an image), mirrored in the recipe's `README.md` and release
  notes so `gr8`'s `README.md` freshness column and the recipe gate know the exact engine build each recipe runs and was
  measured on.
- **Updating a recipe to a newer `gr8` build:** Bump the `gr8` pin in a dedicated pull request (`Bump gr8 to vX.Y.Z` or
  `sha-<short>`), review any **`Breaking changes`**, **Default or cache-key changes**, and **Toolchain / Dockerfile
  sync** notes across the intervening commits, smoke- and bench-test on target hardware, and merge the bump only when
  correctness and speed hold — otherwise leave the recipe on its faster pin until tuned, and update `kindling.json` /
  README speed cards only with numbers measured on the pinned build.
