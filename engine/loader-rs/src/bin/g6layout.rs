//! Writes the layout files g6m4 --layout reads (src/layout.rs): which rank holds each piece of GLM 5.3.
//!
//!   g6layout formula --widths FILE --members a,b,c,d --out FILE
//!       today's layout (the formulas) over the members, rank r being the r-th
//!   g6layout plan --widths FILE --from FILE --members a,b,d --out FILE
//!       a layout for the members planned from --from's: each member keeps what it can and the fewest pieces move
//!
//! --widths is the variant's width manifest (exlr8-v2-manifest): it sizes the expert halves. Both print what moved from
//! --from (plan) as JSON: pieces by kind, and per rank the bytes of expert halves it gains (what a box fetches) and
//! holds.

use g6load::json;
use g6load::layout::{self, Layout, Model};
use std::collections::BTreeMap;

fn main() {
    if let Err(e) = run() {
        eprintln!("g6layout: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().ok_or("usage: g6layout formula|plan --widths FILE --members a,b,.. [--from FILE] --out FILE")?;
    let mut opt: BTreeMap<&str, &str> = BTreeMap::new();
    let mut it = argv[1..].iter();
    while let Some(k) = it.next() {
        opt.insert(k.as_str(), it.next().ok_or(format!("{k} needs a value"))?.as_str());
    }
    let need = |k: &str| opt.get(k).copied().ok_or(format!("{cmd}: {k} is required"));
    let widths = need("--widths")?;
    let model = Model::release(&json::parse(&std::fs::read_to_string(widths).map_err(|e| format!("{widths}: {e}"))?)?)?;
    let members: Vec<String> = need("--members")?.split(',').map(str::to_string).collect();
    let (old, new) = match cmd.as_str() {
        "formula" => {
            let l = Layout::formula(&model, members)?;
            (l.clone(), l)
        }
        "plan" => {
            let old = Layout::read(need("--from")?)?;
            old.check(&model)?;
            let new = layout::plan(&model, &old, members)?;
            (old, new)
        }
        c => return Err(format!("unknown command {c}: formula or plan")),
    };
    new.check(&model)?;
    let out = need("--out")?;
    std::fs::write(out, new.to_json()).map_err(|e| format!("{out}: {e}"))?;
    let m = layout::moves(&model, &old, &new)?;
    let list = |v: &[u64]| v.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
    let pieces: Vec<String> = m.pieces.iter().map(|(k, n)| format!("\"{k}\": {n}")).collect();
    println!("{{\"members\": {}, \"moved\": {{{}}}, \"gained_bytes\": [{}], \"holds_bytes\": [{}]}}",
             new.nr(), pieces.join(", "), list(&m.gained), list(&m.holds));
    Ok(())
}
