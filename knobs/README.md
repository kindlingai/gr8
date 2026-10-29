# Knobs

Every setting gr8 reads from the environment, one YAML file per area. `engine/knobs-rs` reads these files when it
builds (`build.rs`), so a file that breaks a rule below stops the build. The engine, its tests and the Rust crates read
a knob through the typed accessors this generates (`knob::name()` in C++, `gr8knobs::NAME.get()` in Rust).
`tools/env_lint.py` refuses a read that goes around them.

After a change, `cargo run --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen` writes
`engine/knobs-rs/gr8_knobs.h` and [docs/ENV.md](../docs/ENV.md), and CI checks that both are current.

## A file

```yaml
prefix: "GR8_"        # every name in the file starts with this
scope: test           # optional: the scope of every entry, which then names none
knobs:
  GR8_ATTN_CTX2:
    scope: agree
    kind: int
    default: "0"      # or unset: "what the code does when it is unset"
    values: "0..4"
    legacy: ["G6_ATTN_CTX2"]
    doc: >-
      What it does.
```

- `prefix`: every name in the file starts with it. Old names in `legacy` need not.
- `scope`:
  - `agree`: every rank of one engine runs the same value. A value that differs from the default joins the S2
    runtime signature, so a rank with another value refuses to start.
  - `local`: may differ between ranks (paths, per-box resources, logging).
  - `test`: read by test programs, audits and benchmarks. A test knob lives only in a file whose `scope` is `test`.
  - `outside`: read by the build, the launcher or the image, never by the engine.
- `kind`: `flag` (0 or 1), `int` (decimal or 0x hex), `float`, `text`, `path`, `ints` or `floats` (comma lists).
- `default`, or `unset`: give one. `default` is the value an unset variable has, and the accessor returns it.
  `unset` says what the code does when the variable is unset, and the accessor returns an optional.
- `values`: `"lo..hi"` (either end may be left out), `"a|b|c"`, or nothing for any value. A list checks each element.
- `legacy`: old names still read, with a warning at boot. Without it a `GR8_NAME` knob keeps `G6_NAME`. `[]` keeps
  none.
- `doc`: one paragraph.

A name ending in `<n>` is a family: the name followed by any decimal number (`GR8_MTP_KSTOPB_TAU_S<n>`). A name
outside the `GR8_` space (`TMPDIR`, in `system.yaml`) is read under exactly that name.

At boot an empty value counts as unset, a value outside `values` stops the engine, and so does a `GR8_` or `G6_` name
that no file lists.
