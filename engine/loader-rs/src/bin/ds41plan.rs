//! Plans, sizes, hashes and fetch-lists a rank of the DeepSeek-V4.1-Flash-1017 image from the exlr8 native release
//! (src/ds41.rs).
//!
//!   ds41plan RELEASE NR RANK [--layers A,B-C,..] [--experts N] [--no-globals] [--workers N] ACTION...
//!
//! RELEASE is a local copy of kindlingai/deepseek-v4.1-flash-exlr8-native, or a rank's share of it. Actions, any number:
//!   --sizes              the rank's bytes by tensor kind, the Engram rows, the weights + Engram total
//!   --list               one line per tensor: name dtype shape bytes offset
//!   --hash [--tensors a,b]  sha256 of every tensor of the planned image (or of the named ones), nothing is written
//!   --reads              the release ranges the rank reads, as JSON (what a box fetches)
//!   --engram-json        the Engram tables the rank serves (g6engram-v2), as the engine reads them
//!   --engram-sample N    sha256 of the first and last N valid rows of each Engram slice (weight, scale)
//!   --load               the engine's load path: g6l_ds41_plan + g6l_plan_load into a sink that counts and drops the bytes
//! --layers 0,2,20-22 / --experts 8 plan a sample (the coverage check is then off).

use g6load::ds41::{self, Opts};
use g6load::sha256::Sha256;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("ds41plan: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: ds41plan RELEASE NR RANK [--layers A,B-C,..] [--experts N] [--no-globals] [--workers N] [--sizes] [--list] [--hash [--tensors a,b]] [--reads] [--engram-json] [--engram-sample N] [--load]";
    let mut workers = 4usize;
    let mut o = Opts::default();
    let (mut sizes, mut list, mut hash, mut load, mut reads, mut ejson) = (false, false, false, false, false, false);
    let mut eg_sample = 0usize;
    let mut tensors: Option<Vec<String>> = None;
    let mut pos = Vec::new();
    let mut i = 0;
    let val = |i: usize| a.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value\n{usage}", a[i]));
    while i < a.len() {
        match a[i].as_str() {
            "--sizes" => sizes = true,
            "--list" => list = true,
            "--hash" => hash = true,
            "--load" => load = true,
            "--reads" => reads = true,
            "--engram-json" => ejson = true,
            "--no-globals" => o.no_globals = true,
            "--engram-sample" => { eg_sample = val(i)?.parse().map_err(|_| "--engram-sample N")?; i += 1; }
            "--workers" => { workers = val(i)?.parse().map_err(|_| "--workers N")?; i += 1; }
            "--experts" => { o.experts = Some(val(i)?.parse().map_err(|_| "--experts N")?); i += 1; }
            "--layers" => {
                // a comma list of layers and A-B ranges: 0,2,20-22,39
                let mut ls = Vec::new();
                for part in val(i)?.split(',') {
                    let (x, y) = part.split_once('-').unwrap_or((part, part));
                    let (x, y): (usize, usize) = (x.parse().map_err(|_| "--layers A,B-C,...")?, y.parse().map_err(|_| "--layers A,B-C,...")?);
                    ls.extend(x..y + 1);
                }
                o.layers = Some(ls);
                i += 1;
            }
            "--tensors" => { tensors = Some(val(i)?.split(',').map(str::to_string).collect()); i += 1; }
            s if s.starts_with("--") => return Err(format!("{s}: unknown option\n{usage}")),
            s => pos.push(s.to_string()),
        }
        i += 1;
    }
    if pos.len() != 3 {
        return Err(usage.into());
    }
    let num = |s: &String| s.parse::<usize>().map_err(|_| format!("{s}: not a number"));
    let (dir, nr, rank) = (&pos[0], num(&pos[1])?, num(&pos[2])?);
    let t0 = Instant::now();
    let p = ds41::plan(dir, nr, rank, &o)?;
    let (img, eg) = (&p.img, &p.engram);
    eprintln!("planned rank {rank} of {nr}: {} tensors, {} bytes, {:.2} s (headers only)", img.tensors.len(), img.bytes, t0.elapsed().as_secs_f64());
    if sizes {
        print!("{}", ds41::size_report(img, eg));
    }
    if list {
        for t in &img.tensors {
            let sh: Vec<String> = t.shape.iter().map(i64::to_string).collect();
            println!("{} {} [{}] {} {}", t.name, t.dtype, sh.join(","), t.nbytes, t.offset);
        }
    }
    if reads {
        print!("{}", ds41::reads_json(&p.reads));
    }
    if ejson {
        print!("{}", ds41::engram_json(eg, nr, rank));
    }
    // the engine's load path (posix_fadvise) is Linux-only
    #[cfg(target_os = "linux")]
    if load {
        use std::os::raw::c_void;
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        struct Count {
            chunk: AtomicUsize,
            bytes: AtomicU64,
            nonzero: AtomicU64,
        }
        extern "C" fn alloc(ctx: *mut c_void, n: usize) -> *mut c_void {
            let c = unsafe { &*(ctx as *const Count) };
            c.chunk.store(n, Ordering::Relaxed);
            unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(n.max(1), 4096).unwrap()) as *mut c_void }
        }
        extern "C" fn free(ctx: *mut c_void, p: *mut c_void) {
            let c = unsafe { &*(ctx as *const Count) };
            unsafe { std::alloc::dealloc(p as *mut u8, std::alloc::Layout::from_size_align(c.chunk.load(Ordering::Relaxed).max(1), 4096).unwrap()) }
        }
        extern "C" fn copy(ctx: *mut c_void, _w: i32, _dst: u64, src: *const c_void, n: usize) -> i32 {
            let c = unsafe { &*(ctx as *const Count) };
            let b = unsafe { std::slice::from_raw_parts(src as *const u8, n) };
            c.bytes.fetch_add(n as u64, Ordering::Relaxed);
            c.nonzero.fetch_add(b.iter().filter(|&&v| v != 0).count() as u64, Ordering::Relaxed);
            0
        }
        let cnt = Count { chunk: AtomicUsize::new(0), bytes: AtomicU64::new(0), nonzero: AtomicU64::new(0) };
        let sink = g6load::G6lSink { ctx: &cnt as *const Count as *mut c_void, alloc_staging: alloc, free_staging: free, copy };
        let cdir = std::ffi::CString::new(dir.as_str()).map_err(|e| e.to_string())?;
        let tmp = std::env::temp_dir();
        let out = std::ffi::CString::new(format!("{}/ds41plan-load-{}.manifest.json", tmp.display(), std::process::id())).unwrap();
        let eout = std::ffi::CString::new(format!("{}/ds41plan-load-{}.engram.json", tmp.display(), std::process::id())).unwrap();
        let mut err = vec![0 as std::ffi::c_char; 1024];
        let t1 = Instant::now();
        let h = unsafe { ds41::g6l_ds41_plan(cdir.as_ptr(), nr as u32, rank as u32, out.as_ptr(), eout.as_ptr(), err.as_mut_ptr(), err.len()) };
        let msg = |e: &Vec<std::ffi::c_char>| unsafe { std::ffi::CStr::from_ptr(e.as_ptr()) }.to_string_lossy().into_owned();
        if h.is_null() {
            return Err(format!("g6l_ds41_plan: {}", msg(&err)));
        }
        let mut st = g6load::G6lStats::default();
        let rc = unsafe { g6load::g6l_plan_load(h, &sink, &mut st, err.as_mut_ptr(), err.len()) };
        let load_err = msg(&err);
        unsafe { g6load::g6l_plan_free(h) };
        let _ = std::fs::remove_file(out.to_str().unwrap());
        let _ = std::fs::remove_file(eout.to_str().unwrap());
        if rc != 0 {
            return Err(format!("g6l_plan_load rc {rc}: {load_err}"));
        }
        let (b, nz) = (cnt.bytes.load(Ordering::Relaxed), cnt.nonzero.load(Ordering::Relaxed));
        println!("load rank {rank} of {nr}: image {} bytes, sink got {b} bytes ({nz} nonzero), release read {} bytes, {} threads, {:.1} s (plan + load {:.1} s, {:.2} GB/s into the sink)",
                 st.image_bytes, st.read_bytes, st.threads, st.seconds, t1.elapsed().as_secs_f64(), b as f64 / 1e9 / st.seconds.max(1e-9));
    }    #[cfg(not(target_os = "linux"))]
    if load {
        return Err("--load runs the engine's load path, which needs Linux".into());
    }

    if hash {
        let t1 = Instant::now();
        match &tensors {
            Some(names) => {
                for n in names {
                    let mut h = Sha256::new();
                    h.update(&ds41::read_tensor(img, n)?);
                    println!("{} {n}", h.hex());
                }
            }
            None => {
                let shas = ds41::hash_image(img, workers)?;
                for (t, s) in img.tensors.iter().zip(&shas) {
                    println!("{s} {}", t.name);
                }
            }
        }
        eprintln!("hash: {:.1} s", t1.elapsed().as_secs_f64());
    }
    for e in eg.iter().filter(|_| eg_sample > 0) {
        use std::os::unix::fs::FileExt;
        let n = eg_sample.min(e.valid);
        // rows [r0, r0 + n) of the slice, gathered from its split-file runs
        let rows = |r0: usize, w: bool| -> Result<Vec<u8>, String> {
            let width = if w { e.dim } else { e.sdim };
            let mut out = Vec::with_capacity(n * width);
            for s in &e.segs {
                let (a, b) = ((e.first + r0).max(s.row0), (e.first + r0 + n).min(s.row0 + s.rows));
                if a >= b {
                    continue;
                }
                let f = std::fs::File::open(&s.path).map_err(|x| format!("{}: {x}", s.path))?;
                let mut buf = vec![0u8; (b - a) * width];
                let at = if w { s.w_off } else { s.s_off } + ((a - s.row0) * width) as u64;
                f.read_exact_at(&mut buf, at).map_err(|x| format!("{}: {x}", s.path))?;
                out.extend_from_slice(&buf);
            }
            Ok(out)
        };
        for (what, w) in [("w", true), ("s", false)] {
            for (tag, r0) in [("first", 0), ("last", e.valid - n)] {
                let mut h = Sha256::new();
                h.update(&rows(r0, w)?);
                println!("engram L{:03} {what} {tag} {n} rows from slice row {r0} (table row {}): {}", e.layer, e.first + r0, h.hex());
            }
        }
    }
    Ok(())
}
