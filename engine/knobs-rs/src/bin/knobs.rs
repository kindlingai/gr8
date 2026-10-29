//! knobs gen                    write gr8_knobs.h and docs/ENV.md from the table
//! knobs check                  fail when either file is not what gen writes
//! knobs validate NAME=VALUE..  fail on an assignment the engine would refuse: an unknown name or a bad value
//! knobs list                   one line per knob: NAME, scope, kind, and `family` for NAME<n>

use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gen") if args.len() == 1 => {
            for (path, text) in gr8knobs::gen::files() {
                std::fs::write(repo.join(path), text).unwrap_or_else(|e| panic!("{path}: {e}"));
                println!("wrote {path}");
            }
            ExitCode::SUCCESS
        }
        Some("check") if args.len() == 1 => {
            let stale: Vec<_> = gr8knobs::gen::files()
                .into_iter()
                .filter(|(path, text)| std::fs::read_to_string(repo.join(path)).ok().as_ref() != Some(text))
                .map(|(path, _)| path)
                .collect();
            for path in &stale {
                eprintln!("knobs: {path} is out of date: run `knobs gen`");
            }
            if stale.is_empty() {
                println!("knobs check: {} knobs, generated files current", gr8knobs::TABLE.len());
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some("list") if args.len() == 1 => {
            for s in gr8knobs::TABLE {
                println!("{} {:?} {}{}", s.name, s.scope, s.kind.word(), if s.family { " family" } else { "" });
            }
            ExitCode::SUCCESS
        }
        Some("validate") => {
            let pairs = args[1..].iter().map(|a| {
                let (k, v) = a.split_once('=').unwrap_or((a, ""));
                (OsString::from(k), OsString::from(v))
            });
            let env = gr8knobs::Env::read(pairs);
            for p in env.problems() {
                eprintln!("knobs: {}: {}", if p.error { "error" } else { "warning" }, p.text);
            }
            if env.problems().iter().any(|p| p.error) {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        _ => {
            eprintln!("usage: knobs gen | check | list | validate NAME=VALUE..");
            ExitCode::from(2)
        }
    }
}
