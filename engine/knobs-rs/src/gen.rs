//! The files `knobs gen` writes from the table: the C++ header and docs/ENV.md.

use crate::{table_hash, Kind, Scope, Spec, TABLE};

/// Where `knobs gen` writes each file, relative to the repository, and its text.
pub fn files() -> [(&'static str, String); 2] {
    [("engine/knobs-rs/gr8_knobs.h", header()), ("docs/ENV.md", env_md())]
}

fn doc(s: &Spec) -> String {
    s.doc.lines().map(str::trim).collect::<Vec<_>>().join(" ")
}

fn cpp_type(k: Kind) -> &'static str {
    match k {
        Kind::Flag => "bool",
        Kind::Int => "int64_t",
        Kind::Float => "double",
        Kind::Text | Kind::Path => "std::string",
        Kind::Ints => "std::vector<int64_t>",
        Kind::Floats => "std::vector<double>",
    }
}

fn reader(k: Kind) -> &'static str {
    match k {
        Kind::Flag => "flag",
        Kind::Int => "int_",
        Kind::Float => "float_",
        Kind::Text | Kind::Path => "text",
        Kind::Ints => "ints",
        Kind::Floats => "floats",
    }
}

/// gr8_knobs.h: the C API and one typed accessor per knob in namespace knob.
pub fn header() -> String {
    let mut h = format!(
        r#"#pragma once
// Written by `cargo run --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen` from engine/knobs-rs/src/table.rs:
// change the table, not this file. docs/ENV.md describes every knob.
//
// knob::name() returns the value of GR8_NAME: its default when unset, or an optional for a knob without one. The value
// is read once per process. A value the table refuses stops the process at its first read. knob::init() at the start
// of main prints every problem of the environment and returns how many are errors. The C API is in libg6kvman.a.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

extern "C" {{
int gr8_knobs_init(uint64_t table_hash);
const char* gr8_knobs_signature(void);
int gr8_knob_flag(uint32_t id, uint32_t n, int* out);
int gr8_knob_int(uint32_t id, uint32_t n, int64_t* out);
int gr8_knob_float(uint32_t id, uint32_t n, double* out);
const char* gr8_knob_text(uint32_t id, uint32_t n);
ptrdiff_t gr8_knob_ints(uint32_t id, uint32_t n, const int64_t** out);
ptrdiff_t gr8_knob_floats(uint32_t id, uint32_t n, const double** out);
}}

namespace knob
{{

/// Prints the environment's problems (bad values, unknown GR8_ names, old G6_ names) to stderr and returns the number
/// of errors. The engine stops when it is not 0.
inline int init() {{ return gr8_knobs_init(0x{:016x}ull); }}

/// The set agree-scope knobs that differ from their defaults, as " env NAME=value ...", or "".
inline std::string signature() {{ return gr8_knobs_signature(); }}

namespace detail
{{
inline std::optional<bool> flag(uint32_t id, uint32_t n)
{{
    int v;
    if (!gr8_knob_flag(id, n, &v)) return std::nullopt;
    return v != 0;
}}
inline std::optional<int64_t> int_(uint32_t id, uint32_t n)
{{
    int64_t v;
    if (!gr8_knob_int(id, n, &v)) return std::nullopt;
    return v;
}}
inline std::optional<double> float_(uint32_t id, uint32_t n)
{{
    double v;
    if (!gr8_knob_float(id, n, &v)) return std::nullopt;
    return v;
}}
inline std::optional<std::string> text(uint32_t id, uint32_t n)
{{
    const char* v = gr8_knob_text(id, n);
    if (!v) return std::nullopt;
    return std::string(v);
}}
inline std::optional<std::vector<int64_t>> ints(uint32_t id, uint32_t n)
{{
    const int64_t* v;
    const ptrdiff_t len = gr8_knob_ints(id, n, &v);
    if (len < 0) return std::nullopt;
    return std::vector<int64_t>(v, v + len);
}}
inline std::optional<std::vector<double>> floats(uint32_t id, uint32_t n)
{{
    const double* v;
    const ptrdiff_t len = gr8_knob_floats(id, n, &v);
    if (len < 0) return std::nullopt;
    return std::vector<double>(v, v + len);
}}
}} // namespace detail
"#,
        table_hash()
    );
    for (i, s) in TABLE.iter().enumerate() {
        let (t, r, f) = (cpp_type(s.kind), reader(s.kind), s.name.to_lowercase());
        let by_ref = matches!(s.kind, Kind::Text | Kind::Path | Kind::Ints | Kind::Floats);
        h += &format!("\n/// {}{}\n", s.var(None), if s.family { "<n>" } else { "" });
        h += &if s.family {
            format!("inline std::optional<{t}> {f}(uint32_t n) {{ return detail::{r}({i}, n); }}\n")
        } else if s.default.is_some() {
            let ret = if by_ref { format!("const {t}&") } else { t.to_string() };
            format!("inline {ret} {f}()\n{{\n    static const {t} v = *detail::{r}({i}, 0);\n    return v;\n}}\n")
        } else {
            let ret = if by_ref { format!("const std::optional<{t}>&") } else { format!("std::optional<{t}>") };
            format!("inline {ret} {f}()\n{{\n    static const std::optional<{t}> v = detail::{r}({i}, 0);\n    return v;\n}}\n")
        };
    }
    h + "\n} // namespace knob\n"
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|")
}

/// docs/ENV.md: every knob by scope.
pub fn env_md() -> String {
    let mut out = String::from(
        "# Engine environment

Every setting the engine, its tests and its build read from the environment, from `engine/knobs-rs/src/table.rs`.
`knobs gen` writes this page: change the table, not this page. [SETTINGS.md](SETTINGS.md) explains how the main ones
work together, and [KNOBS.md](../KNOBS.md) lists the launcher options that set them.

The knob `NAME` is the variable `GR8_NAME`. Its old name `G6_NAME` still works and prints a warning at boot. A name
ending in `<n>` stands for the name followed by any number. An empty value counts as unset. A value outside the
listed ones stops the engine at boot, and so does a `GR8_` or `G6_` name the engine does not know about.

Types: flag is 0 or 1, int is decimal or 0x hex, ints and floats are comma lists. Values: `lo..hi` is a range
(either end may be left out), `a|b` the only accepted values.
",
    );
    for (scope, title, text) in [
        (
            Scope::Agree,
            "Every rank the same",
            "Every rank of one engine must run the same value. The set ones that differ from the default join the S2 \
             runtime signature, so a rank whose value differs refuses to start.",
        ),
        (Scope::Local, "Per rank", "These may differ between ranks: paths, per-box resources, logging and timing."),
        (
            Scope::Test,
            "Tests and benchmarks",
            "Read by test programs, audits and benchmarks, never by a serving engine.",
        ),
        (Scope::Outside, "Build and launcher", "Read by the build or the launcher, never by the engine."),
    ] {
        out += &format!(
            "\n## {title}\n\n{text}\n\n| variable | type | unset | values | what it does |\n|---|---|---|---|---|\n"
        );
        for s in TABLE.iter().filter(|s| s.scope == scope) {
            let var = s.var(None) + if s.family { "<n>" } else { "" };
            let unset = s.default.map(|d| format!("`{d}`")).unwrap_or_else(|| cell(s.unset));
            out += &format!("| `{var}` | {} | {unset} | {} | {} |\n", s.kind.word(), cell(s.values), cell(&doc(s)));
        }
    }
    out
}
