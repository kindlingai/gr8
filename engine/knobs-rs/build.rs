//! Writes two files from the YAML files in knobs/ at the repository root (knobs/README.md): `$OUT_DIR/table.rs`, the
//! `table!` call that src/table.rs includes, and `$OUT_DIR/table.json`, every knob and launcher option for the C API's
//! gr8_knobs_table_json (the launcher reads it). A file that breaks its own rules stops the build.

use std::fmt::Write as _;
use std::path::Path;

#[path = "src/yaml.rs"]
mod yaml;

/// `s` as a JSON string.
fn js(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out += "\\\"",
            '\\' => out += "\\\\",
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out + "\""
}

fn js_opt(s: &Option<String>) -> String {
    s.as_deref().map(js).unwrap_or_else(|| "null".into())
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../knobs");
    println!("cargo:rerun-if-changed={}", dir.display());
    for p in yaml::files(&dir).unwrap_or_default() {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    let (table, options) = yaml::load(&dir).unwrap_or_else(|e| panic!("knobs: {e}"));
    let q = |s: &str| format!("{s:?}");
    let mut out = String::from("table! {\n");
    let mut json = Vec::new();
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
        let var = e.env.clone().unwrap_or_else(|| format!("GR8_{name}"));
        let legacy = e.legacy.iter().map(|l| js(l)).collect::<Vec<_>>().join(",");
        json.push(format!(
            "{{\"name\":{},\"var\":{},\"family\":{},\"scope\":{},\"kind\":{},\"default\":{},\"unset\":{},\"values\":{},\
             \"legacy\":[{legacy}],\"doc\":{},\"file\":{}}}",
            js(name),
            js(&var),
            e.family,
            js(&e.scope),
            js(&e.kind),
            js_opt(&e.default),
            js(&e.unset),
            js(&e.values),
            js(&e.doc),
            js(&e.file)
        ));
    }
    out += "}\n";
    let opts: Vec<String> = options
        .iter()
        .map(|o| {
            format!(
                "{{\"for\":{},\"key\":{},\"scope\":{},\"default\":{},\"flag\":{},\"doc\":{}}}",
                js(&o.command),
                js(&o.key),
                js_opt(&o.scope),
                js_opt(&o.default),
                js_opt(&o.flag),
                js(&o.doc)
            )
        })
        .collect();
    let json = format!("{{\"knobs\":[{}],\"options\":[{}]}}", json.join(","), opts.join(","));
    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(Path::new(&out_dir).join("table.rs"), out).unwrap();
    std::fs::write(Path::new(&out_dir).join("table.json"), json).unwrap();
}
