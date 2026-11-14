# Version specs (`releases/<tag>.json`)

Each numbered `v*` release of `gr8` has a machine-readable JSON specification in this directory named `releases/<tag>.json`
(for example, [`v1.0.json`](v1.0.json) and [`v1.0.1.json`](v1.0.1.json)), paired with its human-readable section in
[`../CHANGELOG.md`](../CHANGELOG.md) and its GitHub Release ([`../RELEASING.md`](../RELEASING.md)).

Recipes, deployment tools, dashboards, and agents can read `releases/<tag>.json` with standard-library JSON parsers or
`jq` to inspect a release's immutable commit SHA, GHCR image digest, toolchain pins, shipped binaries, supported models
and `TP` topologies, default launch settings, compatibility flags, and verified benchmark numbers.

## Schema (`schema_version: 1`)

Every `releases/<tag>.json` file carries these top-level fields:

- `schema_version` (`int`): `1`.
- `version` (`string`): Release number without the leading `v` (such as `"1.0"` or `"1.0.1"`).
- `tag` (`string`): Exact Git tag (`"v" + version`), matching the `<tag>.json` filename.
- `previous_tag` (`string | null`): Prior numbered `v*` release tag, or `null` for `v1.0`.
- `date` (`string`): Release date in `YYYY-MM-DD` (UTC).
- `commit` (`string`): Exact 40-character lowercase hex Git commit SHA of `<tag>`.
- `release_url` (`string`): GitHub Release URL (`https://github.com/kindlingai/gr8/releases/tag/<tag>`).
- `validation_gate` (`string`): Short factual description of the release gate run for `<tag>`.
- `container` (`object`): Published runtime image metadata (`image`, `digest`, `pinned`, `platforms`, `target_arch`).
- `toolchain` (`object`): `Dockerfile` and `rust-toolchain.toml` pins (`cuda`, `base_os`, `rust`, `runtime_apt`).
- `binaries` (`array<string>`): Paths under `/opt/gr8/` shipped by `Dockerfile` `dist`.
- `api` (`object`): Front-end and cluster network contract (`protocol`, `http_port`, `control_port`, `s2_port`, `endpoints`).
- `models` (`array<object>`): Served GPU models in `./gr8` at `<tag>` (`id`, `title`, `architecture`, `served_names`,
  `checkpoint`, `drafter`, `engine_binaries`, `tp_supported`, `variants`, `default_variant_by_tp`, `int4_kv`, and
  `defaults_by_tp`).
- `breaking` (`bool`): `true` only when `breaking_changes` is non-empty (`MAJOR` release).
- `breaking_changes` (`array<object>`): Empty `[]` when backward-compatible; otherwise migration entries.
- `default_or_cache_changes` (`array<object>`): Changed launch/engine defaults, numerical trade-offs with `rollback`
  commands, or cache-key rebuild notes relative to `previous_tag`.
- `toolchain_changes` (`array<string>`): `Dockerfile` or `rust-toolchain.toml` changes relative to `previous_tag`.
- `operational_notes` (`array<string>`): Non-breaking operational notes for operators upgrading to `<tag>`.
- `prs` (`array<object>`): Pull requests included in `<tag>` (`number`, `commit`, `summary`).
- `benchmarks` (`array<object>`): Headline measured results verified for `<tag>`.

## Validation

`python3 scripts/make_zip.py --check` (run by CI on every pull request) and
`python3 -m unittest bench/tests/test_releases.py` check that every `releases/v*.json` file conforms to `schema_version: 1`
and stays in 1-to-1 sync with [`../CHANGELOG.md`](../CHANGELOG.md).
