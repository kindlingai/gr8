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

Following `v1.0`, releases use `vMAJOR.MINOR.PATCH` (`v1.0.1`, `v1.1.0`, `v2.0.0`) and are automated with
`release-please` (`release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/release.yml`).
Pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/) (checked on every PR by
`.github/workflows/pr-title.yml`), because `gr8` squash-merges PRs to `main` and `release-please` derives version bumps
and [`CHANGELOG.md`](CHANGELOG.md) entries from commit subjects on `main`:

- **`PATCH` (`fix:`, `perf:`, `revert:` — e.g. `v1.2.1`):** Bug fixes, compatible performance improvements, and internal
  changes that keep all existing CLI flags, config files, HTTP endpoints, weight/steering formats, and default numerical
  behavior compatible.
- **`MINOR` (`feat:` — e.g. `v1.3.0`):** Backward-compatible additions: new models or TP topologies, new optional flags
  or knobs in `knobs/*.yaml`, additive HTTP fields, or validated default speedups that keep existing launch commands
  working (with any numerical trade documented and reversible by flags).
- **`MAJOR` (`!:` or `BREAKING CHANGE:` footer — e.g. `v2.0.0`):** Any breaking change (section 3) that requires users,
  clients, or recipes to update their commands, config files, container setup, or stored artifacts.
- **Explicit version override:** Include a `Release-As: X.Y.Z` footer in a commit message to request a specific version
  in the next Release PR.
- **Pre-releases (`vX.Y.Z-rc.N` or `beta-*`):** Optional test cuts before a release; recipes for general use avoid
  pre-release tags. Pre-release tags and older-series tag backfills never move `ghcr.io/kindlingai/gr8:latest` or `:main`.

Published release tags are immutable: never delete or force-move a `v*` tag. Ship fixes via `main` in the next `PATCH`
release.

### What cutting a release builds

Every push to `main` runs [`.github/workflows/release.yml`](.github/workflows/release.yml), which runs `release-please`
first (updating the open Release PR, or creating the `vX.Y.Z` tag and GitHub Release when a Release PR merges) and then
invokes [`.github/workflows/ci.yml`](.github/workflows/ci.yml) once via `workflow_call` (passing `tag` when a release
was cut, and skipping non-release pushes that touch only `**.md`). For a tagged release, `ci.yml` runs the full
pipeline once without duplicating builds:

1. CPU tests (`frontend` and `engine`) and `tools/env_lint.py`.
2. Arm64 Rust binaries and all engine build legs at each leg's supported TP widths (`1, 2, 3, 4, 5, 6, 8` for GLM-5.3
   `fp8` and `int4`, `1, 2, 3, 4` for Flash `fp8` and `int4` and `qw38`, `4` for `ds41`, `1, 4, 6` for `ink2`, plus
   the GPU test binaries).
3. Reproducible tester archive (`gr8-<short-sha>.zip` + `.sha256` via `scripts/make_zip.py`), uploaded as an Actions
   workflow artifact and attached to the GitHub Release with build-provenance attestation.
4. Runtime container image `ghcr.io/kindlingai/gr8:<tag>` (and `sha-<commit>`, plus `:latest` and `:main` when the
   commit is still `main`'s head and not a pre-release), smoke-tested and pushed with build-provenance attestation.
5. Updates the GitHub Release body with the immutable `ghcr.io/kindlingai/gr8:<tag>@sha256:<digest>` reference,
   attached tester archive filenames, and `gh attestation verify` commands.

To republish or backfill an existing `v*` tag manually (for example after a transient CI failure), trigger
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) via `workflow_dispatch` with the `tag` input set to that tag.

Note that `ghcr.io/kindlingai/gr8:main` and `:latest` move on non-Markdown merges to `main` (never on pre-releases or
backfills of older tags), and any `:stable` image tag moves with the `stable` branch; none of them is an immutable
release pin.

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

### Automated release records and optional version specs

Every `v*` release cut by `release-please` automatically produces:

- a **Release PR** (`chore(main): release X.Y.Z`) updating [`.release-please-manifest.json`](.release-please-manifest.json)
  and [`CHANGELOG.md`](CHANGELOG.md) from merged Conventional Commits (`feat`, `fix`, `perf`, `revert`, and breaking
  changes),
- a **Git tag and GitHub Release** on `<tag>` when that Release PR is squash-merged to `main`, and
- **attached build artifacts and verification notes** added by `.github/workflows/ci.yml` once the release build
  finishes (`gr8-<short-sha>.zip`, `.zip.sha256`, build-provenance attestations, and the immutable
  `ghcr.io/kindlingai/gr8:<tag>@sha256:<digest>` reference).

Machine-readable [`releases/<tag>.json`](releases/README.md) version specs (`schema_version: 1`) are optional for
automated `release-please` releases (preserved for `v1.0`–`v1.2.0` and available for curated major/milestone records).
Whenever a `releases/v*.json` file is present, `python3 scripts/make_zip.py --check` validates its schema and date
parity with [`CHANGELOG.md`](CHANGELOG.md) on every pull request (and keeps strict 4-subsection + commit/digest sync
for historical manual `## [vX.Y.Z] — YYYY-MM-DD` entries).

For significant releases (new defaults, numerical trade-offs, cache-key resets, or `Dockerfile` / toolchain changes),
include those details in the PR commit messages (`BREAKING CHANGE:` footer when breaking) or edit the Release PR's
`CHANGELOG.md` entry before merging.

### Cutting a release

1. Land changes on `main` via normal pull requests with Conventional Commit titles (verified by `pr-title.yml`).
   `release-please` automatically opens and updates the pending Release PR (`chore(main): release X.Y.Z`) on every push
   to `main`.
2. Before merging the Release PR, verify the Level-3 hardware gate across the supported `TP=N` model configurations and
   walk the keep-current checklist in [AGENTS.md](AGENTS.md#keeping-docs-and-recipes-current) (`README.md`, model pages,
   `KNOBS.md`, `KNOWN-ISSUES.md`, `CREDITS.md`, `recipes.json`, and `docs/cards/`).
3. Review the Release PR (optionally adding measured benchmark highlights or upgrade notes to its `CHANGELOG.md` diff).
   If `RELEASE_PLEASE_TOKEN` is configured as a repository secret, `ci.yml` and `pr-title.yml` trigger automatically on
   the Release PR; otherwise (using the default `GITHUB_TOKEN`), close and reopen the Release PR once so GitHub triggers
   the required PR checks. Comment `/ready` and squash-merge the Release PR.
4. Merging the Release PR runs `.github/workflows/release.yml`, which creates `<tag>` and its GitHub Release, then runs
   `.github/workflows/ci.yml` once to test, build all engine legs, push `ghcr.io/kindlingai/gr8:<tag>`, upload
   `gr8-<short-sha>.zip` + `.sha256` with provenance attestations, and append the immutable `@sha256:<digest>` block to
   the GitHub Release notes.
5. Optionally update [`README.md`](README.md#building-on-gr8) with the new `<tag>@sha256:<digest>` pin example (and add
   a curated [`releases/<tag>.json`](releases/README.md) if desired), then check standalone recipes against `<tag>` and
   open pin-bump pull requests where correctness and speed hold.

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
  `sha-<short>`), review [`CHANGELOG.md`](CHANGELOG.md) and [`releases/<tag>.json`](releases/README.md) (specifically
  **`Breaking changes`**, **Default or cache-key changes**, and **Toolchain / Dockerfile sync**) across the intervening
  commits, smoke- and bench-test on target hardware, and merge the bump only when correctness and speed hold — otherwise
  leave the recipe on its faster pin until tuned, and update `kindling.json` / README speed cards only with numbers
  measured on the pinned build.
