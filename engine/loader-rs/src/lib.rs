//! gr8 pack loader. Builds a rank's device image (the pack's bytes at the manifest's offsets) from the pack file and,
//! for a sparse pack (tools/convert/convert.py --v2-refs, tools/convert/v2pack.py), the ranges its rank<r>.sources
//! names. Worker threads fill fixed-size chunks of the image in parallel, each into its own pinned staging buffer, and
//! hand every chunk to the caller's copy callback (cudaMemcpy in g6m4). The image is byte for byte the one
//! engine/loader/devimage.cpp builds. Nothing is hashed: checking the files is an offline step.
//!
//! Environment:
//!   G6_LOAD_MODE=read   (default) pread into the staging buffers, O_DIRECT where aligned; nothing is mapped
//!   G6_LOAD_MODE=mmap   map every file at start; workers prefault a range, copy it out and drop its pages
//!                       (the base for serving from resident pages; slower: 2.2-2.5 GB/s against 3.3 at TP4)
//!   G6_LOAD_THREADS=8   worker threads (8 and 16 read equally fast at TP4: the NVMe is the limit)
//!   G6_LOAD_CHUNK_MB=32 bytes of image per chunk (each worker holds one staging buffer of this size)
//!
//! g6l_v2_plan plans an image without a v2 pack: the base pack's tensors, with the v2 experts read from the raw width
//! files at the offsets of the index slices (src/v2plan.rs). g6l_release_plan plans one from the exlr8 release alone,
//! no pack at all (src/release.rs): the image's other tensors are copied from the dense files or computed from them
//! at load (Op::Gen). g6l_plan_load then loads either like a pack.

pub mod dflash2;
pub mod exlr8;
pub mod fetch;
pub mod json;
mod plan;
pub mod records;
pub mod release;
pub mod sha256;
mod sys;
pub mod v2plan;
pub mod xxh64;

use plan::{Op, Plan};
use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const ALIGN: u64 = 4096;

#[repr(C)]
pub struct G6lExtent {
    pub offset: u64,
    pub nbytes: u64,
}

/// What the loader writes into. Staging buffers come from `alloc_staging` (pinned host memory in g6m4); `copy` moves
/// `n` bytes at `src` to image offset `dst_off` and returns 0 on success. Each worker calls `copy` with its own index.
#[repr(C)]
pub struct G6lSink {
    pub ctx: *mut c_void,
    pub alloc_staging: extern "C" fn(ctx: *mut c_void, n: usize) -> *mut c_void,
    pub free_staging: extern "C" fn(ctx: *mut c_void, p: *mut c_void),
    pub copy: extern "C" fn(ctx: *mut c_void, worker: i32, dst_off: u64, src: *const c_void, n: usize) -> i32,
}

#[repr(C)]
#[derive(Default)]
pub struct G6lStats {
    pub seconds: f64,
    pub image_bytes: u64,
    pub read_bytes: u64,
    pub ranges: u64,
    pub threads: u32,
    pub mode: u32, // 0 mmap, 1 read
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Mmap,
    Read,
}

struct Src {
    buffered: File,
    direct: Option<File>,
    map: Option<sys::Map>,
}

struct Loader<'a> {
    plan: Plan,
    files: Vec<Src>,
    mode: Mode,
    cache: bool,
    sink: &'a G6lSink,
    direct_read_ns: AtomicU64,
    direct_read_bytes: AtomicU64,
    buf_read_ns: AtomicU64,
    buf_read_bytes: AtomicU64,
    mmap_copy_ns: AtomicU64,
    gen_get_ns: AtomicU64,
    gen_memcpy_ns: AtomicU64,
    gen_bytes: AtomicU64,
    item128_read_ns: AtomicU64,
    item128_perm_ns: AtomicU64,
    item128_bytes: AtomicU64,
    mem_memcpy_ns: AtomicU64,
    mem_bytes: AtomicU64,
    sink_copy_ns: AtomicU64,
    alloc_staging_ns: AtomicU64,
    free_staging_ns: AtomicU64,
}

unsafe impl Sync for Loader<'_> {}

impl Loader<'_> {
    /// Copies file bytes `off..off + dst.len()` of file `f` into `dst`.
    fn read(&self, f: usize, off: u64, dst: &mut [u8]) -> Result<(), String> {
        let s = &self.files[f];
        let name = &self.plan.files[f];
        if let Some(m) = &s.map {
            let t0 = Instant::now();
            let res = m.copy_out(off, dst).map_err(|e| format!("{name}: {e}"));
            self.mmap_copy_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return res;
        }
        let mut done = 0usize;
        // O_DIRECT for the 4096-aligned head of the range when the destination is aligned too
        if let Some(d) = &s.direct {
            if off.is_multiple_of(ALIGN) && (dst.as_ptr() as u64).is_multiple_of(ALIGN) {
                let head = dst.len() / ALIGN as usize * ALIGN as usize;
                if head > 0 {
                    let t0 = Instant::now();
                    read_exact_at(d, &mut dst[..head], off).map_err(|e| format!("{name}: {e} (O_DIRECT)"))?;
                    self.direct_read_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    self.direct_read_bytes.fetch_add(head as u64, Ordering::Relaxed);
                    done = head;
                }
            }
        }
        if done < dst.len() {
            let t0 = Instant::now();
            let rem = dst.len() - done;
            read_exact_at(&s.buffered, &mut dst[done..], off + done as u64).map_err(|e| format!("{name}: {e}"))?;
            if !self.cache {
                sys::fadvise_dontneed(&s.buffered, off + done as u64, rem as u64);
            }
            self.buf_read_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.buf_read_bytes.fetch_add(rem as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Fills image bytes `a..a + buf.len()`: pack-file data first, then the source ranges over it.
    fn fill(&self, a: u64, buf: &mut [u8], half: &mut sys::AlignedBuf, perm: &mut sys::AlignedBuf) -> Result<u64, String> {
        let b = a + buf.len() as u64;
        buf.fill(0);
        let mut read = 0u64;
        for &(x, y) in self.plan.pack_data(a, b) {
            let (x, y) = (x.max(a), y.min(b));
            self.read(0, x, &mut buf[(x - a) as usize..(y - a) as usize])?;
            read += y - x;
        }
        for s in self.plan.sources(a, b) {
            let (x, y) = (s.pack_off.max(a), (s.pack_off + s.nbytes).min(b));
            match s.op {
                Op::Copy => {
                    let off = s.file_off + (x - s.pack_off);
                    self.read(s.file, off, &mut buf[(x - a) as usize..(y - a) as usize])?;
                    read += y - x;
                }
                Op::Mem => {
                    let t0 = Instant::now();
                    let m = &self.plan.mems[s.file];
                    buf[(x - a) as usize..(y - a) as usize]
                        .copy_from_slice(&m[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
                    self.mem_memcpy_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    self.mem_bytes.fetch_add(y - x, Ordering::Relaxed);
                }
                Op::Gen => {
                    let tg = Instant::now();
                    let g = &self.plan.gens[s.file];
                    let m = g.get()?;
                    self.gen_get_ns.fetch_add(tg.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let tc = Instant::now();
                    buf[(x - a) as usize..(y - a) as usize]
                        .copy_from_slice(&m[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
                    g.used(y - x);
                    self.gen_memcpy_ns.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    self.gen_bytes.fetch_add(y - x, Ordering::Relaxed);
                }
                Op::Item128 { hidden } => {
                    let n = s.nbytes as usize;
                    let tr = Instant::now();
                    self.read(s.file, s.file_off, half.slice(n))?;
                    self.item128_read_ns.fetch_add(tr.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let tp = Instant::now();
                    plan::item128_to_g6_geometry(half.slice(n), perm.slice(n), hidden);
                    buf[(x - a) as usize..(y - a) as usize]
                        .copy_from_slice(&perm.slice(n)[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
                    self.item128_perm_ns.fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    self.item128_bytes.fetch_add(y - x, Ordering::Relaxed);
                    read += s.nbytes;
                }
            }
        }
        Ok(read)
    }
}

fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        match f.read_at(buf, off) {
            Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, format!("short read at {off}"))),
            Ok(n) => {
                buf = &mut buf[n..];
                off += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(default)
}

fn open_src(path: &str, mode: Mode, cache: bool) -> Result<Src, String> {
    let buffered = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let (direct, map) = match mode {
        Mode::Read if !cache => (std::fs::OpenOptions::new().read(true).custom_flags(sys::O_DIRECT).open(path).ok(), None),
        Mode::Read => (None, None),
        Mode::Mmap => (None, Some(sys::Map::new(&buffered).map_err(|e| format!("cannot map {path}: {e}"))?)),
    };
    Ok(Src { buffered, direct, map })
}

/// Loads the image. `tensors` are the manifest's (offset, nbytes); `sources` is None for a full pack.
pub fn load(pack: &str, pack_bytes: u64, tensors: &[G6lExtent], sources: Option<&str>, root: Option<&str>,
            sink: &G6lSink) -> Result<G6lStats, String> {
    let t0 = Instant::now();
    run(Plan::new(pack, pack_bytes, tensors, sources, root)?, pack_bytes, sink, t0)
}

/// Fills image bytes 0..pack_bytes from `plan` through `sink`. The stats count time from `t0`.
fn run(plan: Plan, pack_bytes: u64, sink: &G6lSink, t0: Instant) -> Result<G6lStats, String> {
    let t_run = Instant::now();
    let mode = match std::env::var("G6_LOAD_MODE").as_deref() {
        Ok("read") | Err(_) => Mode::Read,
        Ok("mmap") => Mode::Mmap,
        Ok(v) => return Err(format!("G6_LOAD_MODE={v}: use mmap or read")),
    };
    let cache = std::env::var("G6_LOAD_CACHE").ok().as_deref() == Some("1");
    let threads = env_u64("G6_LOAD_THREADS", 16) as usize;
    let chunk = env_u64("G6_LOAD_CHUNK_MB", 32) << 20;
    let files = plan.files.iter().map(|p| open_src(p, mode, cache)).collect::<Result<Vec<_>, _>>()?;
    let open_files_s = t_run.elapsed().as_secs_f64();
    let l = Loader {
        plan,
        files,
        mode,
        cache,
        sink,
        direct_read_ns: AtomicU64::new(0),
        direct_read_bytes: AtomicU64::new(0),
        buf_read_ns: AtomicU64::new(0),
        buf_read_bytes: AtomicU64::new(0),
        mmap_copy_ns: AtomicU64::new(0),
        gen_get_ns: AtomicU64::new(0),
        gen_memcpy_ns: AtomicU64::new(0),
        gen_bytes: AtomicU64::new(0),
        item128_read_ns: AtomicU64::new(0),
        item128_perm_ns: AtomicU64::new(0),
        item128_bytes: AtomicU64::new(0),
        mem_memcpy_ns: AtomicU64::new(0),
        mem_bytes: AtomicU64::new(0),
        sink_copy_ns: AtomicU64::new(0),
        alloc_staging_ns: AtomicU64::new(0),
        free_staging_ns: AtomicU64::new(0),
    };
    let nchunks = pack_bytes.div_ceil(chunk);
    let next = AtomicU64::new(0);
    let read_total = AtomicU64::new(0);
    let failed = AtomicBool::new(false);
    let error: Mutex<Option<String>> = Mutex::new(None);
    let max_half = l.plan.max_item128() as usize;
    let started = AtomicUsize::new(0);
    let t_workers = Instant::now();
    std::thread::scope(|sc| {
        for w in 0..threads {
            let (l, next, read_total, failed, error, started) = (&l, &next, &read_total, &failed, &error, &started);
            sc.spawn(move || {
                started.fetch_add(1, Ordering::Relaxed);
                let ta = Instant::now();
                let p0 = (l.sink.alloc_staging)(l.sink.ctx, chunk as usize) as *mut u8;
                let p1 = if p0.is_null() {
                    std::ptr::null_mut()
                } else {
                    (l.sink.alloc_staging)(l.sink.ctx, chunk as usize) as *mut u8
                };
                l.alloc_staging_ns.fetch_add(ta.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if p1.is_null() {
                    if !p0.is_null() {
                        (l.sink.free_staging)(l.sink.ctx, p0 as *mut c_void);
                    }
                    failed.store(true, Ordering::Relaxed);
                    error.lock().unwrap().get_or_insert(format!("worker {w}: no staging buffer of {chunk} bytes"));
                    return;
                }
                let bufs = [p0, p1];
                let mut slot = 0usize;
                let mut half = sys::AlignedBuf::new(max_half.max(1));
                let mut perm = sys::AlignedBuf::new(max_half.max(1));
                while !failed.load(Ordering::Relaxed) {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let p = bufs[slot];
                    slot ^= 1;
                    let a = c * chunk;
                    let n = chunk.min(pack_bytes - a) as usize;
                    let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
                    let r = l.fill(a, buf, &mut half, &mut perm).and_then(|read| {
                        read_total.fetch_add(read, Ordering::Relaxed);
                        let ts = Instant::now();
                        let rc = (l.sink.copy)(l.sink.ctx, w as i32, a, p as *const c_void, n);
                        l.sink_copy_ns.fetch_add(ts.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        match rc {
                            0 => Ok(()),
                            e => Err(format!("copy of image bytes {a}..{} failed ({e})", a + n as u64)),
                        }
                    });
                    if let Err(e) = r {
                        failed.store(true, Ordering::Relaxed);
                        error.lock().unwrap().get_or_insert(e);
                    }
                }
                let tf = Instant::now();
                (l.sink.free_staging)(l.sink.ctx, p0 as *mut c_void);
                (l.sink.free_staging)(l.sink.ctx, p1 as *mut c_void);
                l.free_staging_ns.fetch_add(tf.elapsed().as_nanos() as u64, Ordering::Relaxed);
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let workers_wall_s = t_workers.elapsed().as_secs_f64();
    let total_wall_s = t0.elapsed().as_secs_f64();
    let mut gen_comp_total_ns = 0u64;
    let mut gen_wait_total_ns = 0u64;
    let mut gen_embed_ns = 0u64;
    let mut gen_lm_head_ns = 0u64;
    let mut gen_kv_b_ns = 0u64;
    let mut gen_mxgv_ns = 0u64;
    let mut gen_tr3gv_ns = 0u64;
    let mut gen_other_ns = 0u64;
    let mut by_what: std::collections::BTreeMap<&str, (u64, u64, u64, usize)> = std::collections::BTreeMap::new();
    for g in &l.plan.gens {
        let c = g.compute_ns.load(Ordering::Relaxed);
        let w = g.wait_ns.load(Ordering::Relaxed);
        gen_comp_total_ns += c;
        gen_wait_total_ns += w;
        if g.what.contains("G.embed") {
            gen_embed_ns += c;
        } else if g.what.contains("G.lm_head") {
            gen_lm_head_ns += c;
        } else if g.what.contains("kv_b") {
            gen_kv_b_ns += c;
        } else if g.what.contains(".mx.gv") {
            gen_mxgv_ns += c;
        } else if g.what.contains(".tr3.gv") {
            gen_tr3gv_ns += c;
        } else {
            gen_other_ns += c;
        }
        let e = by_what.entry(g.what.as_str()).or_insert((0, 0, 0, 0));
        e.0 += g.nbytes;
        e.1 += c;
        e.2 += w;
        e.3 += 1;
    }
    for (what, (nb, c, w, cnt)) in by_what {
        if c >= 20_000_000 || w >= 20_000_000 {
            eprintln!(
                "# boot-prof gen [{what}] ({cnt} segs): {:.2} MB, compute {:.3} s, mutex_wait {:.3} thread-s",
                nb as f64 / 1e6,
                c as f64 / 1e9,
                w as f64 / 1e9
            );
        }
    }
    let d_ns = l.direct_read_ns.load(Ordering::Relaxed);
    let d_bytes = l.direct_read_bytes.load(Ordering::Relaxed);
    let b_ns = l.buf_read_ns.load(Ordering::Relaxed);
    let b_bytes = l.buf_read_bytes.load(Ordering::Relaxed);
    let i_read_ns = l.item128_read_ns.load(Ordering::Relaxed);
    let i_perm_ns = l.item128_perm_ns.load(Ordering::Relaxed);
    let i_bytes = l.item128_bytes.load(Ordering::Relaxed);
    let g_get_ns = l.gen_get_ns.load(Ordering::Relaxed);
    let g_cpy_ns = l.gen_memcpy_ns.load(Ordering::Relaxed);
    let g_bytes = l.gen_bytes.load(Ordering::Relaxed);
    let m_cpy_ns = l.mem_memcpy_ns.load(Ordering::Relaxed);
    let m_bytes = l.mem_bytes.load(Ordering::Relaxed);
    let s_cpy_ns = l.sink_copy_ns.load(Ordering::Relaxed);
    let a_stg_ns = l.alloc_staging_ns.load(Ordering::Relaxed);
    let f_stg_ns = l.free_staging_ns.load(Ordering::Relaxed);
    eprintln!(
        "# boot-prof loader-rs wall: total {:.3} s (plan_to_run {:.3} s, open_files {:.3} s, workers_wall {:.3} s across {} threads)",
        total_wall_s,
        (total_wall_s - workers_wall_s - open_files_s).max(0.0),
        open_files_s,
        workers_wall_s,
        threads
    );
    eprintln!(
        "# boot-prof loader-rs thread-sums ({} threads, {:.3} thread-s cap): direct_read {:.3} s ({:.2} GB), buf_read {:.3} s ({:.2} GB), item128(read {:.3} s + perm {:.3} s, {:.2} GB), gen(get {:.3} s = compute {:.3} s + mutex_wait {:.3} s, memcpy {:.3} s, {:.2} GB), mem_memcpy {:.3} s ({:.2} GB), sink_copy {:.3} s, staging(alloc {:.3} s, free {:.3} s)",
        threads,
        workers_wall_s * threads as f64,
        d_ns as f64 / 1e9,
        d_bytes as f64 / 1e9,
        b_ns as f64 / 1e9,
        b_bytes as f64 / 1e9,
        i_read_ns as f64 / 1e9,
        i_perm_ns as f64 / 1e9,
        i_bytes as f64 / 1e9,
        g_get_ns as f64 / 1e9,
        gen_comp_total_ns as f64 / 1e9,
        gen_wait_total_ns as f64 / 1e9,
        g_cpy_ns as f64 / 1e9,
        g_bytes as f64 / 1e9,
        m_cpy_ns as f64 / 1e9,
        m_bytes as f64 / 1e9,
        s_cpy_ns as f64 / 1e9,
        a_stg_ns as f64 / 1e9,
        f_stg_ns as f64 / 1e9
    );
    eprintln!(
        "# boot-prof loader-rs gen-by-kind (compute thread-s): G.embed {:.3} s, G.lm_head {:.3} s, kv_b {:.3} s, mx.gv {:.3} s, tr3.gv {:.3} s, other {:.3} s (total {:.3} s compute, {:.3} s mutex_wait)",
        gen_embed_ns as f64 / 1e9,
        gen_lm_head_ns as f64 / 1e9,
        gen_kv_b_ns as f64 / 1e9,
        gen_mxgv_ns as f64 / 1e9,
        gen_tr3gv_ns as f64 / 1e9,
        gen_other_ns as f64 / 1e9,
        gen_comp_total_ns as f64 / 1e9,
        gen_wait_total_ns as f64 / 1e9
    );
    Ok(G6lStats {
        seconds: total_wall_s,
        image_bytes: pack_bytes,
        read_bytes: read_total.load(Ordering::Relaxed),
        ranges: l.plan.ranges() as u64,
        threads: threads as u32,
        mode: (l.mode == Mode::Read) as u32,
    })
}

fn cstr(p: *const c_char) -> Option<String> {
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()).filter(|s| !s.is_empty())
}

/// C entry point (engine/loader-rs/g6load.h). Returns 0 on success; otherwise writes a message into `err`.
///
/// # Safety
/// `tensors` must point at `ntensors` extents, `sink` and `stats` must be valid, `err` must hold `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_load(pack: *const c_char, pack_bytes: u64, tensors: *const G6lExtent, ntensors: usize,
                                  sources: *const c_char, root: *const c_char, sink: *const G6lSink,
                                  stats: *mut G6lStats, err: *mut c_char, errlen: usize) -> i32 {
    let ext = if ntensors == 0 { &[][..] } else { std::slice::from_raw_parts(tensors, ntensors) };
    let Some(pack) = cstr(pack) else { return write_err(err, errlen, "no pack path") };
    match load(&pack, pack_bytes, ext, cstr(sources).as_deref(), cstr(root).as_deref(), &*sink) {
        Ok(s) => {
            *stats = s;
            0
        }
        Err(e) => write_err(err, errlen, &e),
    }
}

/// Plans a rank's image from a base pack and the exlr8 v2 files (src/v2plan.rs) and writes its manifest to
/// `out_manifest`. Returns a handle for g6l_plan_load and g6l_plan_free, or NULL with a message in `err`.
///
/// # Safety
/// The strings must be NUL-terminated, `err` must hold `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_v2_plan(base_manifest: *const c_char, base_pack: *const c_char, v2_dir: *const c_char,
                                     slices_dir: *const c_char, widths: *const c_char, out_manifest: *const c_char,
                                     err: *mut c_char, errlen: usize) -> *mut c_void {
    let t0 = Instant::now();
    let arg = |p, what| cstr(p).ok_or(format!("g6l_v2_plan: no {what}"));
    let r = (|| {
        let img = v2plan::plan(&arg(base_manifest, "base manifest")?, &arg(base_pack, "base pack")?,
                               &arg(v2_dir, "v2 dir")?, &arg(slices_dir, "slices dir")?, &arg(widths, "width manifest")?)?;
        let out = arg(out_manifest, "output manifest")?;
        std::fs::write(&out, img.manifest()).map_err(|e| format!("{out}: {e}"))?;
        Ok::<_, String>(img)
    })();
    match r {
        Ok(img) => Box::into_raw(Box::new((img, t0))) as *mut c_void,
        Err(e) => {
            write_err(err, errlen, &e);
            std::ptr::null_mut()
        }
    }
}

/// Plans rank `rank`'s image at TP `nr` from the exlr8 v2 release alone (src/release.rs): the dense and vocabulary
/// files under `dense_dir`, the experts from the width files under `v2_dir` at the offsets of the index slices under
/// `slices_dir`, at the widths of the width manifest. Writes the image's manifest to `out_manifest`. On rank 0 it also
/// prints the form of each matrix to stderr. Returns a handle for g6l_plan_load and g6l_plan_free, or
/// NULL with a message in `err`.
///
/// # Safety
/// The strings must be NUL-terminated, `err` must hold `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_release_plan(dense_dir: *const c_char, v2_dir: *const c_char, slices_dir: *const c_char,
                                          widths: *const c_char, nr: u32, rank: u32, qo: i32, trellis: i32,
                                          out_manifest: *const c_char,
                                          err: *mut c_char, errlen: usize) -> *mut c_void {
    release_plan_inner(dense_dir, v2_dir, slices_dir, widths, nr, rank, qo, trellis, None, out_manifest, err, errlen)
}

/// Test-only subset planner for m4sim. The normal g6l_release_plan retains the full-model check.
///
/// # Safety
/// Strings must be NUL-terminated; layers must point to n_layers u32 values; err holds errlen bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_release_plan_selected(dense_dir: *const c_char, v2_dir: *const c_char,
    slices_dir: *const c_char, widths: *const c_char, nr: u32, rank: u32, qo: i32, trellis: i32,
    layers: *const u32, n_layers: usize, out_manifest: *const c_char,
    err: *mut c_char, errlen: usize) -> *mut c_void {
    if layers.is_null() || n_layers == 0 {
        write_err(err, errlen, "selected release plan needs a nonempty layer list");
        return std::ptr::null_mut();
    }
    let selected: Vec<usize> = std::slice::from_raw_parts(layers, n_layers).iter().map(|&l| l as usize).collect();
    release_plan_inner(dense_dir, v2_dir, slices_dir, widths, nr, rank, qo, trellis,
                       Some(&selected), out_manifest, err, errlen)
}

unsafe fn release_plan_inner(dense_dir: *const c_char, v2_dir: *const c_char, slices_dir: *const c_char,
    widths: *const c_char, nr: u32, rank: u32, qo: i32, trellis: i32, selected: Option<&[usize]>,
    out_manifest: *const c_char, err: *mut c_char, errlen: usize) -> *mut c_void {
    let t0 = Instant::now();
    let arg = |p, what| cstr(p).ok_or(format!("g6l_release_plan: no {what}"));
    let r = (|| {
        let (img, table) = release::plan_selected(&arg(dense_dir, "dense dir")?, &arg(v2_dir, "v2 dir")?, &arg(slices_dir, "slices dir")?,
                                         &arg(widths, "width manifest")?, nr as usize, rank as usize,
                                         release::Opts { qo: qo != 0, trellis: trellis != 0 }, selected, true, true)?;
        let out = arg(out_manifest, "output manifest")?;
        std::fs::write(&out, img.manifest()).map_err(|e| format!("{out}: {e}"))?;
        if rank == 0 {
            eprintln!("# rank 0: weights from the exlr8 release, by matrix (rounded = not the release's weights):");
            for line in table.lines() {
                eprintln!("#   {line}");
            }
        }
        Ok::<_, String>(img)
    })();
    match r {
        Ok(img) => Box::into_raw(Box::new((img, t0))) as *mut c_void,
        Err(e) => {
            write_err(err, errlen, &e);
            std::ptr::null_mut()
        }
    }
}

/// Loads an image g6l_v2_plan planned. The stats count time from the plan call. Returns 0, or 1 with a message in
/// `err`.
///
/// # Safety
/// `h` must come from g6l_v2_plan or g6l_release_plan and be loaded once; `sink`, `stats` and `err` as for g6l_load.
#[no_mangle]
pub unsafe extern "C" fn g6l_plan_load(h: *mut c_void, sink: *const G6lSink, stats: *mut G6lStats, err: *mut c_char,
                                       errlen: usize) -> i32 {
    let (img, t0) = &mut *(h as *mut (v2plan::V2Image, Instant));
    let plan = std::mem::replace(&mut img.plan, Plan::empty());
    match run(plan, img.bytes, &*sink, *t0) {
        Ok(s) => {
            *stats = s;
            0
        }
        Err(e) => write_err(err, errlen, &e),
    }
}

/// Image bytes `a..a + buf.len()` of a segment plan (g6l_v2_plan, g6l_release_plan), one range at a time: files opened
/// per call, computed ranges computed and dropped.
pub fn read_range(plan: &Plan, a: u64, buf: &mut [u8]) -> Result<(), String> {
    let b = a + buf.len() as u64;
    buf.fill(0);
    for s in plan.sources(a, b) {
        let (x, y) = (s.pack_off.max(a), (s.pack_off + s.nbytes).min(b));
        let dst = &mut buf[(x - a) as usize..(y - a) as usize];
        let (lo, hi) = ((x - s.pack_off) as usize, (y - s.pack_off) as usize);
        let file = || File::open(&plan.files[s.file]).map_err(|e| format!("{}: {e}", plan.files[s.file]));
        match s.op {
            Op::Copy => read_exact_at(&file()?, dst, s.file_off + lo as u64).map_err(|e| format!("{}: {e}", plan.files[s.file]))?,
            Op::Mem => dst.copy_from_slice(&plan.mems[s.file][lo..hi]),
            Op::Gen => dst.copy_from_slice(&plan.gens[s.file].compute()?[lo..hi]),
            Op::Item128 { hidden } => {
                let n = s.nbytes as usize;
                let (mut half, mut perm) = (vec![0u8; n], vec![0u8; n]);
                read_exact_at(&file()?, &mut half, s.file_off).map_err(|e| format!("{}: {e}", plan.files[s.file]))?;
                plan::item128_to_g6_geometry(&half, &mut perm, hidden);
                dst.copy_from_slice(&perm[lo..hi]);
            }
        }
    }
    Ok(())
}

/// Reads image bytes `off..off + n` of a planned image into `dst` (m4sim's per-tensor loads). Returns 0, or 1 with a
/// message in `err`.
///
/// # Safety
/// `h` must come from g6l_v2_plan or g6l_release_plan, before g6l_plan_load. `dst` must hold `n` bytes, `err` `errlen`.
#[no_mangle]
pub unsafe extern "C" fn g6l_plan_read(h: *mut c_void, off: u64, n: u64, dst: *mut c_void, err: *mut c_char,
                                       errlen: usize) -> i32 {
    let (img, _) = &*(h as *const (v2plan::V2Image, Instant));
    let buf = std::slice::from_raw_parts_mut(dst as *mut u8, n as usize);
    match read_range(&img.plan, off, buf) {
        Ok(()) => 0,
        Err(e) => write_err(err, errlen, &e),
    }
}

/// Frees a g6l_v2_plan handle.
///
/// # Safety
/// `h` must come from g6l_v2_plan or g6l_release_plan and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn g6l_plan_free(h: *mut c_void) {
    if !h.is_null() {
        drop(Box::from_raw(h as *mut (v2plan::V2Image, Instant)));
    }
}

unsafe fn write_err(err: *mut c_char, errlen: usize, msg: &str) -> i32 {
    if !err.is_null() && errlen > 0 {
        let n = msg.len().min(errlen - 1);
        std::ptr::copy_nonoverlapping(msg.as_ptr(), err as *mut u8, n);
        *err.add(n) = 0;
    }
    1
}
