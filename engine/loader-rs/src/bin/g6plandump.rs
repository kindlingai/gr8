//! Plans a rank image from the exlr8 release (src/release.rs) and writes some of its tensors to a safetensors file,
//! byte for byte as the engine would load them. tests/release_check.py checks the planner with it, and
//! engine/ref/release_forward.py's weights follow the same forms.
//!
//!   g6plandump DENSE V2 SLICES WIDTHS NR RANK OUT.safetensors [--layers 0,3,78] [--globals] [--experts] [--qo]
//!              [--rounded] [--sizes]
//!
//! --layers picks the L<LLL>.* tensors to write (default none), --globals the G.* ones. Routed-expert tensors (tr3.*)
//! are written only with --experts. --qo plans the G6_PF_OWN=3 full-head tensors too. --rounded gives the trellis
//! matrices in today's forms (MXFP8 / int4), and without it they are the release's entries. The table of forms goes to stderr, and with --sizes
//! the image's bytes by tensor kind to stdout.

use std::io::Write;

fn main() {
    if let Err(e) = run() {
        eprintln!("g6plandump: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        return Err("usage: g6plandump DENSE V2 SLICES WIDTHS NR RANK OUT [--layers L,..] [--globals] [--experts] [--qo]".into());
    }
    let num = |i: usize| a[i].parse::<usize>().map_err(|_| format!("{}: not a number", a[i]));
    let (nr, rank, out) = (num(5)?, num(6)?, &a[7]);
    let (mut layers, mut globals, mut experts, mut qo, mut rounded, mut sizes) = (Vec::new(), false, false, false, false, false);
    let mut i = 8;
    while i < a.len() {
        match a[i].as_str() {
            "--layers" => {
                let v = a.get(i + 1).ok_or("--layers L,..")?;
                layers = v.split(',').map(|x| x.parse::<usize>().map_err(|_| format!("--layers {v}"))).collect::<Result<_, _>>()?;
                i += 2;
            }
            "--globals" => { globals = true; i += 1; }
            "--experts" => { experts = true; i += 1; }
            "--qo" => { qo = true; i += 1; }
            "--rounded" => { rounded = true; i += 1; }
            "--sizes" => { sizes = true; i += 1; }
            x => return Err(format!("unknown option {x}")),
        }
    }
    let t0 = std::time::Instant::now();
    let opts = g6load::release::Opts { qo, trellis: !rounded };
    let (img, table) = g6load::release::plan(&a[1], &a[2], &a[3], &a[4], nr, rank, opts)?;
    eprint!("{table}");
    if sizes {
        // bytes by tensor kind over the whole image: the name without its layer, or tr3.* for every expert tensor
        let mut by: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        for t in &img.tensors {
            let k = if t.name.starts_with('G') { t.name.clone() } else if t.name[5..].starts_with("tr3.") { "experts (tr3.*)".into() } else { t.name[5..].to_string() };
            *by.entry(k).or_default() += t.nbytes;
        }
        println!("| tensor | GB |\n|---|---:|");
        for (k, v) in &by {
            println!("| {k} | {:.3} |", *v as f64 / 1e9);
        }
        println!("| image | {:.3} |", img.bytes as f64 / 1e9);
    }
    let want = |n: &str| -> bool {
        if let Some(rest) = n.strip_prefix('G') {
            return globals && rest.starts_with('.');
        }
        let l = n.get(1..4).and_then(|d| d.parse::<usize>().ok());
        l.is_some_and(|l| layers.contains(&l)) && (experts || !n[5..].starts_with("tr3."))
    };
    let st = |d: &str| match d { "u8" => "U8", "bf16" => "BF16", "f16" => "F16", "f32" => "F32", "i64" => "I64", x => x }.to_string();
    let pick: Vec<_> = img.tensors.iter().filter(|t| want(&t.name)).collect();
    let mut head = String::from("{");
    let mut off = 0u64;
    for t in &pick {
        let s: Vec<String> = t.shape.iter().map(i64::to_string).collect();
        head += &format!("\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}},", t.name, st(&t.dtype), s.join(","), off + t.nbytes);
        off += t.nbytes;
    }
    head = head.trim_end_matches(',').to_string() + "}";
    while head.len() % 8 != 0 {
        head.push(' ');
    }
    let io = |e: std::io::Error| format!("{out}: {e}");
    let mut w = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(out).map_err(io)?);
    w.write_all(&(head.len() as u64).to_le_bytes()).map_err(io)?;
    w.write_all(head.as_bytes()).map_err(io)?;
    for t in &pick {
        let mut buf = vec![0u8; t.nbytes as usize];
        g6load::read_range(&img.plan, t.offset, &mut buf)?;
        w.write_all(&buf).map_err(io)?;
    }
    w.flush().map_err(io)?;
    eprintln!("g6plandump: rank {rank} of TP {nr}: image {:.2} GB; wrote {} tensors ({:.2} GB) in {:.1} s",
              img.bytes as f64 / 1e9, pick.len(), off as f64 / 1e9, t0.elapsed().as_secs_f64());
    Ok(())
}
