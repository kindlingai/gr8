//! Writes `$OUT_DIR/table.rs`, the `table!` call that src/table.rs includes, from the YAML files in knobs/ at the
//! repository root (knobs/README.md). A file that breaks its own rules stops the build.

use std::fmt::Write as _;
use std::path::Path;

#[path = "src/yaml.rs"]
mod yaml;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../knobs");
    println!("cargo:rerun-if-changed={}", dir.display());
    for p in yaml::files(&dir).unwrap_or_default() {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    let table = yaml::load(&dir).unwrap_or_else(|e| panic!("knobs: {e}"));
    let q = |s: &str| format!("{s:?}");
    let mut out = String::from("table! {\n");
    for (name, e) in &table {
        if !e.doc.is_empty() {
            writeln!(out, "    #[doc = {}]", q(&e.doc)).unwrap();
        }
        let value = match &e.default {
            Some(d) => format!("= {}", q(d)),
            None => format!("? {}", q(&e.unset)),
        };
        let env = e.env.as_deref().map(|v| format!(", env {}", q(v))).unwrap_or_default();
        let legacy = e.legacy.iter().map(|l| q(l)).collect::<Vec<_>>().join(", ");
        let family = if e.family { "[n]" } else { "" };
        let values = q(&e.values);
        writeln!(out, "    {name}{family}: {} {} {value}, {values}{env}, legacy [{legacy}];", e.scope, e.kind).unwrap();
    }
    out += "}\n";
    std::fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("table.rs"), out).unwrap();
}
