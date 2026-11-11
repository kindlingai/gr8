//! gr8 pack loader. Builds a rank's device image (the pack's bytes at the manifest's offsets) from the pack file and,
//! for a sparse pack (--v2-refs), the ranges its rank<r>.sources
//! names. Worker threads fill fixed-size chunks of the image in parallel, each into its own pinned staging buffer, and
//! hand every chunk to the caller's copy callback (cudaMemcpy in g6m4). The image is byte for byte the one
//! engine/loader/devimage.cpp builds. Nothing is hashed: checking the files is an offline step.
//!
//! Environment:
//!   GR8_LOAD_MODE=read   (default) pread into the staging buffers, O_DIRECT where aligned; nothing is mapped
//!   GR8_LOAD_MODE=mmap   map every file at start; workers prefault a range, copy it out and drop its pages
//!                       (the base for serving from resident pages; slower: 2.2-2.5 GB/s against 3.3 at TP4)
//!   GR8_LOAD_THREADS=8   worker threads (8 and 16 read equally fast at TP4: the NVMe is the limit)
//!   GR8_LOAD_CHUNK_MB=32 bytes of image per chunk (each worker holds one staging buffer of this size)
//!   GR8_LOAD_MERGE_KB=0  0 = one read per range. N > 0 (read mode): each chunk's file ranges sorted by file offset,
//!                       ranges of one file at most N KB apart read with one pread into a scratch buffer and copied
//!                       out (fill_merged). Same bytes; the gaps are read too.
//!   GR8_LOAD_COMPACT=DIR reads the ranges of a source file from its compacted copy under DIR where DIR holds a map
//!                       for it (Plan::remap_compact). Same bytes.
//!
//! g6l_v2_plan plans an image without a v2 pack: the base pack's tensors, with the v2 experts read from the raw width
//! files at the offsets of the index slices (src/v2plan.rs). g6l_release_plan plans one from the exlr8 release alone,
//! no pack at all (src/release.rs): the image's other tensors are copied from the dense files or computed from them
//! at load (Op::Gen). g6l_plan_load then loads either like a pack.

pub mod codebook;
pub mod compact;
pub mod dflash2;
pub mod ds41;
pub mod exlr8;
pub mod fetch;
pub mod gencache;
pub mod json;
pub mod layout;
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
    sink: &'a G6lSink,
    cache: Option<&'a gencache::GenCache>,
    /// GR8_LOAD_MERGE_KB in bytes; None = one read per range
    merge: Option<u64>,
}

/// One worker's buffers: Item128 halves (`half`, `perm`) and the merged reads (`scratch`, empty when merging is off).
struct Bufs {
    half: sys::AlignedBuf,
    perm: sys::AlignedBuf,
    scratch: Option<sys::AlignedBuf>,
    scratch_len: usize,
}

unsafe impl Sync for Loader<'_> {}

// Per-op profiling of the load, GR8_LOAD_PROF=1 (cu-fastboot arm A, ported to DS41): ns/bytes/count triplets printed at
// the end of run(). Off = one relaxed atomic load per op, no accumulation.
// bases: 0 zero | 3 pack | 6 copy | 9 O_DIRECT head | 12 buffered tail | 15 mem | 18 gen
//        21 i128_read | 24 i128_geom | 27 segs iterated | 28 segs <= 64KB | 29 merged preads (bytes read)
fn prof_on() -> bool {
    gr8knobs::LOAD_PROF.get()
}
#[allow(clippy::declare_interior_mutable_const)]
const PROF0: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF: [std::sync::atomic::AtomicU64; 32] = [PROF0; 32];
#[inline]
fn prof_op<F: FnOnce() -> R, R>(base: usize, bytes: u64, f: F) -> R {
    use std::sync::atomic::Ordering::Relaxed;
    if !prof_on() {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    PROF[base].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    PROF[base + 1].fetch_add(bytes, Relaxed);
    PROF[base + 2].fetch_add(1, Relaxed);
    r
}

impl Loader<'_> {
    /// Copies file bytes `off..off + dst.len()` of file `f` into `dst`.
    fn read(&self, f: usize, off: u64, dst: &mut [u8]) -> Result<(), String> {
        let s = &self.files[f];
        let name = &self.plan.files[f];
        if let Some(m) = &s.map {
            return m.copy_out(off, dst).map_err(|e| format!("{name}: {e}"));
        }
        let mut done = 0usize;
        // O_DIRECT for the 4096-aligned head of the range when the destination is aligned too
        if let Some(d) = &s.direct {
            if off.is_multiple_of(ALIGN) && (dst.as_ptr() as u64).is_multiple_of(ALIGN) {
                let head = dst.len() / ALIGN as usize * ALIGN as usize;
                if head > 0 {
                    prof_op(9, head as u64, ||
                        read_exact_at(d, &mut dst[..head], off).map_err(|e| format!("{name}: {e} (O_DIRECT)")))?;
                    done = head;
                }
            }
        }
        if done < dst.len() {
            prof_op(12, (dst.len() - done) as u64, ||
                read_exact_at(&s.buffered, &mut dst[done..], off + done as u64).map_err(|e| format!("{name}: {e}")))?;
            sys::fadvise_dontneed(&s.buffered, off + done as u64, (dst.len() - done) as u64);
        }
        Ok(())
    }

    /// Fills image bytes `a..a + buf.len()`: pack-file data first, then the source ranges over it.
    fn fill(&self, a: u64, buf: &mut [u8], bufs: &mut Bufs) -> Result<u64, String> {
        let b = a + buf.len() as u64;
        prof_op(0, buf.len() as u64, || buf.fill(0));
        let mut read = 0u64;
        for &(x, y) in self.plan.pack_data(a, b) {
            let (x, y) = (x.max(a), y.min(b));
            prof_op(3, y - x, || self.read(0, x, &mut buf[(x - a) as usize..(y - a) as usize]))?;
            read += y - x;
        }
        let merged = self.merge.is_some() && bufs.scratch.is_some();
        if merged {
            read += self.fill_merged(a, buf, bufs)?;
        }
        let Bufs { half, perm, .. } = bufs;
        for s in self.plan.sources(a, b) {
            if merged && matches!(s.op, Op::Copy | Op::Item128 { .. }) {
                continue;
            }
            if prof_on() {
                use std::sync::atomic::Ordering::Relaxed;
                PROF[27].fetch_add(1, Relaxed);
                if s.nbytes <= 65536 {
                    PROF[28].fetch_add(1, Relaxed);
                }
            }
            let (x, y) = (s.pack_off.max(a), (s.pack_off + s.nbytes).min(b));
            match s.op {
                Op::Copy => {
                    let off = s.file_off + (x - s.pack_off);
                    prof_op(6, y - x, || self.read(s.file, off, &mut buf[(x - a) as usize..(y - a) as usize]))?;
                    read += y - x;
                }
                Op::Mem => {
                    let m = &self.plan.mems[s.file];
                    prof_op(15, y - x, || {
                        buf[(x - a) as usize..(y - a) as usize]
                            .copy_from_slice(&m[(x - s.pack_off) as usize..(y - s.pack_off) as usize])
                    });
                }
                Op::Gen => {
                    let g = &self.plan.gens[s.file];
                    let m = prof_op(18, y - x, || g.get(self.cache, s.pack_off))?;
                    prof_op(18, 0, || {
                        buf[(x - a) as usize..(y - a) as usize]
                            .copy_from_slice(&m[(x - s.pack_off) as usize..(y - s.pack_off) as usize])
                    });
                    g.used(y - x);
                }
                Op::Item128 { hidden } => {
                    let n = s.nbytes as usize;
                    prof_op(21, n as u64, || self.read(s.file, s.file_off, half.slice(n)))?;
                    prof_op(24, n as u64, || plan::item128_to_g6_geometry(half.slice(n), perm.slice(n), hidden));
                    buf[(x - a) as usize..(y - a) as usize]
                        .copy_from_slice(&perm.slice(n)[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
                    read += s.nbytes;
                }
            }
        }
        Ok(read)
    }

    /// The Copy and Item128 ranges of image bytes `a..a + buf.len()`, read in file order: ranges of one file whose gap
    /// is at most `self.merge` bytes share one pread into `bufs.scratch` (up to its size), then each is copied to its
    /// place in `buf` (an Item128 range through its permutation). Returns the bytes read, gaps included.
    fn fill_merged(&self, a: u64, buf: &mut [u8], bufs: &mut Bufs) -> Result<u64, String> {
        let b = a + buf.len() as u64;
        let gap = self.merge.unwrap_or(0);
        let cap = bufs.scratch_len as u64;
        // (file, first file byte, end file byte, seg): an Item128 range is read whole, a Copy range only inside a..b
        let mut v: Vec<(usize, u64, u64, &plan::Seg)> = Vec::new();
        for s in self.plan.sources(a, b) {
            match s.op {
                Op::Copy => {
                    let (x, y) = (s.pack_off.max(a), (s.pack_off + s.nbytes).min(b));
                    let off = s.file_off + (x - s.pack_off);
                    v.push((s.file, off, off + (y - x), s));
                }
                Op::Item128 { .. } => v.push((s.file, s.file_off, s.file_off + s.nbytes, s)),
                _ => {}
            }
        }
        v.sort_unstable_by_key(|e| (e.0, e.1));
        let scratch = bufs.scratch.as_mut().unwrap();
        let mut read = 0u64;
        let mut i = 0;
        while i < v.len() {
            let (f, lo) = (v[i].0, v[i].1 / ALIGN * ALIGN);
            let mut hi = v[i].2;
            let mut j = i + 1;
            while j < v.len() && v[j].0 == f && v[j].1 <= hi.saturating_add(gap) && v[j].2.max(hi) - lo <= cap {
                hi = hi.max(v[j].2);
                j += 1;
            }
            if hi - lo > cap {
                return Err(format!("{}: range of {} bytes past the merge buffer of {cap}", self.plan.files[f], hi - lo));
            }
            let run = scratch.slice((hi - lo) as usize);
            prof_op(29, hi - lo, || self.read(f, lo, run))?;
            read += hi - lo;
            for &(_, x0, x1, s) in &v[i..j] {
                let src = &run[(x0 - lo) as usize..(x1 - lo) as usize];
                match s.op {
                    Op::Copy => {
                        let x = s.pack_off.max(a);
                        buf[(x - a) as usize..(x - a) as usize + src.len()].copy_from_slice(src);
                    }
                    Op::Item128 { hidden } => {
                        let n = s.nbytes as usize;
                        let perm = bufs.perm.slice(n);
                        prof_op(24, n as u64, || plan::item128_to_g6_geometry(src, perm, hidden));
                        let (x, y) = (s.pack_off.max(a), (s.pack_off + s.nbytes).min(b));
                        buf[(x - a) as usize..(y - a) as usize]
                            .copy_from_slice(&perm[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
                    }
                    _ => unreachable!(),
                }
            }
            i = j;
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

/// GR8_KVB (#50): `fp8` plans kv_b (UK / UV) as the release's block FP8 bytes and scales, `bf16` as BF16.
fn kvb_rot() -> bool { gr8knobs::KV4_FOLD.get() }

fn kvb_fp8() -> Result<bool, String> {
    Ok(gr8knobs::KVB.get() == "fp8")
}

fn open_src(path: &str, mode: Mode) -> Result<Src, String> {
    let buffered = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let (direct, map) = match mode {
        Mode::Read => (std::fs::OpenOptions::new().read(true).custom_flags(sys::O_DIRECT).open(path).ok(), None),
        Mode::Mmap => (None, Some(sys::Map::new(&buffered).map_err(|e| format!("cannot map {path}: {e}"))?)),
    };
    Ok(Src { buffered, direct, map })
}

/// Loads the image. `tensors` are the manifest's (offset, nbytes); `sources` is None for a full pack.
pub fn load(pack: &str, pack_bytes: u64, tensors: &[G6lExtent], sources: Option<&str>, root: Option<&str>,
            sink: &G6lSink) -> Result<G6lStats, String> {
    let t0 = Instant::now();
    run(Plan::new(pack, pack_bytes, tensors, sources, root)?, pack_bytes, sink, t0, None)
}

/// Fills image bytes 0..pack_bytes from `plan` through `sink`. The stats count time from `t0`.
fn run(plan: Plan, pack_bytes: u64, sink: &G6lSink, t0: Instant, cache: Option<&gencache::GenCache>)
    -> Result<G6lStats, String> {
    let mode = if gr8knobs::LOAD_MODE.get() == "mmap" { Mode::Mmap } else { Mode::Read };
    let threads = gr8knobs::LOAD_THREADS.get() as usize;
    let chunk = (gr8knobs::LOAD_CHUNK_MB.get() as u64) << 20;
    let merge = match gr8knobs::LOAD_MERGE_KB.get() {
        0 => None,
        kb => Some(kb as u64 * 1024),
    };
    run_cfg(plan, pack_bytes, sink, t0, cache, mode, threads, chunk, merge)
}

#[allow(clippy::too_many_arguments)]
fn run_cfg(mut plan: Plan, pack_bytes: u64, sink: &G6lSink, t0: Instant, cache: Option<&gencache::GenCache>, mode: Mode,
           threads: usize, chunk: u64, merge: Option<u64>) -> Result<G6lStats, String> {
    let merge = merge.filter(|_| mode == Mode::Read);
    if let Some(dir) = gr8knobs::LOAD_COMPACT.get() {
        sys::raise_nofile();
        let (nf, moved, left) = plan.remap_compact(dir)?;
        eprintln!("# load-compact: {dir}: {nf} compacted files, {moved} ranges read from them, {left} from the sources");
    }
    let files = plan.files.iter().map(|p| open_src(p, mode)).collect::<Result<Vec<_>, _>>()?;
    let l = Loader { plan, files, mode, sink, cache, merge };
    let nchunks = pack_bytes.div_ceil(chunk);
    let next = AtomicU64::new(0);
    let read_total = AtomicU64::new(0);
    let failed = AtomicBool::new(false);
    let error: Mutex<Option<String>> = Mutex::new(None);
    let max_half = l.plan.max_item128() as usize;
    let started = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for w in 0..threads {
            let (l, next, read_total, failed, error, started) = (&l, &next, &read_total, &failed, &error, &started);
            sc.spawn(move || {
                started.fetch_add(1, Ordering::Relaxed);
                let p = (l.sink.alloc_staging)(l.sink.ctx, chunk as usize) as *mut u8;
                if p.is_null() {
                    failed.store(true, Ordering::Relaxed);
                    error.lock().unwrap().get_or_insert(format!("worker {w}: no staging buffer of {chunk} bytes"));
                    return;
                }
                // a merged pread holds a chunk's ranges and the gaps between them: twice the chunk, plus one
                // Item128 range that starts before the chunk
                let scratch_len = (2 * chunk as usize + max_half).div_ceil(ALIGN as usize) * ALIGN as usize;
                let scratch = match l.merge.map(|_| sys::AlignedBuf::try_new(scratch_len)) {
                    Some(None) => {
                        failed.store(true, Ordering::Relaxed);
                        error.lock().unwrap().get_or_insert(format!(
                            "worker {w}: no merge scratch buffer of {scratch_len} bytes (GR8_LOAD_MERGE_KB; a smaller \
                             GR8_LOAD_CHUNK_MB or GR8_LOAD_THREADS needs less)"));
                        (l.sink.free_staging)(l.sink.ctx, p as *mut c_void);
                        return;
                    }
                    s => s.flatten(),
                };
                let mut bufs = Bufs {
                    half: sys::AlignedBuf::new(max_half.max(1)),
                    perm: sys::AlignedBuf::new(max_half.max(1)),
                    scratch,
                    scratch_len,
                };
                while !failed.load(Ordering::Relaxed) {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let a = c * chunk;
                    let n = chunk.min(pack_bytes - a) as usize;
                    let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
                    let r = l.fill(a, buf, &mut bufs).and_then(|read| {
                        read_total.fetch_add(read, Ordering::Relaxed);
                        match (l.sink.copy)(l.sink.ctx, w as i32, a, p as *const c_void, n) {
                            0 => Ok(()),
                            e => Err(format!("copy of image bytes {a}..{} failed ({e})", a + n as u64)),
                        }
                    });
                    if let Err(e) = r {
                        failed.store(true, Ordering::Relaxed);
                        error.lock().unwrap().get_or_insert(e);
                    }
                }
                (l.sink.free_staging)(l.sink.ctx, p as *mut c_void);
            });
        }
    });
    if prof_on() {
        use std::sync::atomic::Ordering::Relaxed;
        let g = |i: usize| PROF[i].load(Relaxed);
        let trip = |b: usize| format!("{:.2}s/{:.2}GB/{}", g(b) as f64 / 1e9, g(b + 1) as f64 / 1e9, g(b + 2));
        eprintln!("# load-prof zero={} pack={} copy={} direct={} buffered={} mem={} gen={} i128read={} i128geom={} segs={} small={} merged={}",
                  trip(0), trip(3), trip(6), trip(9), trip(12), trip(15), trip(18), trip(21), trip(24), g(27), g(28), trip(29));
    }
    if let Some(c) = cache {
        eprintln!("# gen-cache: {}", c.summary());
    }
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(G6lStats {
        seconds: t0.elapsed().as_secs_f64(),
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
        // GR8_GEN_CACHE (gencache.rs): the key binds the engine binary, the planned manifest and the plan inputs
        let mut img = img;
        let (bm, bp, v2, sl, wd) = (arg(base_manifest, "base manifest")?, arg(base_pack, "base pack")?, arg(v2_dir, "v2 dir")?,
                                    arg(slices_dir, "slices dir")?, arg(widths, "width manifest")?);
        let key_text = format!("v2 base={bm} pack={bp} v2={v2} slices={sl} widths={wd}\n{}\n{}", img.manifest(),
                               gencache::source_identity(&[&bm, &bp, &v2, &sl, &wd]));
        img.cache = gencache::GenCache::from_env(&key_text);
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
/// `slices_dir`, at the widths of the width manifest. The rank holds the pieces the layout file `layout` gives it
/// (src/layout.rs), or with NULL or "" the formulas'. Writes the image's manifest to `out_manifest`. On rank 0 it also
/// prints the form of each matrix to stderr. Returns a handle for g6l_plan_load and g6l_plan_free, or
/// NULL with a message in `err`.
///
/// # Safety
/// The strings must be NUL-terminated, `err` must hold `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn g6l_release_plan(dense_dir: *const c_char, v2_dir: *const c_char, slices_dir: *const c_char,
                                          widths: *const c_char, nr: u32, rank: u32, qo: i32, trellis: i32,
                                          layout: *const c_char, out_manifest: *const c_char,
                                          err: *mut c_char, errlen: usize) -> *mut c_void {
    let t0 = Instant::now();
    let arg = |p, what| cstr(p).ok_or(format!("g6l_release_plan: no {what}"));
    let r = (|| {
        let kvb_fp8 = kvb_fp8()?;
        let kvb_rot = kvb_rot();
        if kvb_fp8 && kvb_rot { return Err("GR8_KV4_FOLD=1 cannot combine with GR8_KVB=fp8".into()); }
        let layout_path: Option<String> = cstr(layout);
        let layout = cstr(layout).map(|p| layout::Layout::read(&p)).transpose()?;
        let (img, table) = release::plan(&arg(dense_dir, "dense dir")?, &arg(v2_dir, "v2 dir")?, &arg(slices_dir, "slices dir")?,
                                         &arg(widths, "width manifest")?, nr as usize, rank as usize, layout.as_ref(),
                                         release::Opts { qo: qo != 0, trellis: trellis != 0, kvb_fp8, kvb_rot })?;
        let out = arg(out_manifest, "output manifest")?;
        std::fs::write(&out, img.manifest()).map_err(|e| format!("{out}: {e}"))?;
        if rank == 0 {
            eprintln!("# rank 0: weights from the exlr8 release, by matrix (rounded = not the release's weights):");
            for line in table.lines() {
                eprintln!("#   {line}");
            }
        }
        // GR8_GEN_CACHE (gencache.rs): the key binds the engine binary, the planned manifest and the plan inputs
        // (the release directories, TP, rank, layout and the transform options)
        let mut img = img;
        let (de, v2, sl, wd) = (arg(dense_dir, "dense dir")?, arg(v2_dir, "v2 dir")?, arg(slices_dir, "slices dir")?,
                                arg(widths, "width manifest")?);
        let lp = layout_path.clone().unwrap_or_default();
        let key_text = format!("release dense={de} v2={v2} slices={sl} widths={wd} nr={nr} rank={rank} qo={qo} trellis={trellis} \
                                kvb_fp8={kvb_fp8} kvb_rot={kvb_rot} layout={lp:?}\n{}\n{}", img.manifest(),
                               gencache::source_identity(&[&de, &v2, &sl, &wd, &lp]));
        img.cache = gencache::GenCache::from_env(&key_text);
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
    let cache = img.cache.as_ref();
    match run(plan, img.bytes, &*sink, *t0, cache) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use plan::Seg;

    struct Img {
        bytes: Mutex<Vec<u8>>,
        staging: usize,
    }
    extern "C" fn alloc(ctx: *mut c_void, n: usize) -> *mut c_void {
        let img = unsafe { &*(ctx as *const Img) };
        assert_eq!(n, img.staging);
        unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(n, 4096).unwrap()) as *mut c_void }
    }
    extern "C" fn free(ctx: *mut c_void, p: *mut c_void) {
        let img = unsafe { &*(ctx as *const Img) };
        unsafe { std::alloc::dealloc(p as *mut u8, std::alloc::Layout::from_size_align(img.staging, 4096).unwrap()) }
    }
    extern "C" fn copy(ctx: *mut c_void, _w: i32, dst: u64, src: *const c_void, n: usize) -> i32 {
        let img = unsafe { &*(ctx as *const Img) };
        let s = unsafe { std::slice::from_raw_parts(src as *const u8, n) };
        img.bytes.lock().unwrap()[dst as usize..dst as usize + n].copy_from_slice(s);
        0
    }

    /// Two source files; Copy and Item128 ranges at shuffled file offsets with gaps, of aligned and odd sizes, laid
    /// out in the image across chunk boundaries. Returns the files and the segments (image order).
    fn fixture(dir: &std::path::Path) -> (Vec<String>, Vec<(u64, u64, Op, usize, u64)>, u64) {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut rnd = |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        let mut files = Vec::new();
        for f in 0..2 {
            let p = dir.join(format!("src{f}"));
            let data: Vec<u8> = (0..4u64 << 20).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 ^ f as u8).collect();
            std::fs::write(&p, data).unwrap();
            files.push(p.to_str().unwrap().to_string());
        }
        // file pieces: slots of 64 KB in each file, handed out in a shuffled order
        let mut slots: Vec<(usize, u64)> = (0..2).flat_map(|f| (0..64u64).map(move |k| (f, k * 65536))).collect();
        for i in (1..slots.len()).rev() {
            slots.swap(i, rnd(i as u64 + 1) as usize);
        }
        let mut segs = Vec::new();
        let mut at = 0u64;
        for (f, slot) in slots {
            at += [0, 4096, 1000, 0][rnd(4) as usize];
            let (n, op) = if rnd(3) == 0 { (3072 * (1 + rnd(4)), Op::Item128 { hidden: 128 }) } else {
                ([4096, 65536, 12345, 40000][rnd(4) as usize], Op::Copy) };
            let off = slot + [0, 512, 4096][rnd(3) as usize];
            segs.push((at, n, op, f, off));
            at += n;
        }
        (files, segs, at + 777)
    }

    fn plan_of(files: &[String], segs: &[(u64, u64, Op, usize, u64)], bytes: u64) -> Plan {
        let s = segs.iter().map(|&(pack_off, nbytes, op, file, file_off)| Seg { pack_off, nbytes, op, file_off, file }).collect();
        Plan::from_segs(files.to_vec(), s, Vec::new(), Vec::new(), bytes).unwrap()
    }

    #[test]
    fn aligned_buf_try_new_reports_a_failed_allocation() {
        assert!(sys::AlignedBuf::try_new(isize::MAX as usize).is_none());
        assert!(sys::AlignedBuf::try_new(usize::MAX).is_none());
        let mut b = sys::AlignedBuf::try_new(5000).unwrap();
        assert_eq!(b.slice(8192).len(), 8192);
    }

    #[test]
    fn merged_reads_load_the_same_image() {
        let dir = std::env::temp_dir().join(format!("g6load-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (files, segs, bytes) = fixture(&dir);
        let mut want = vec![0u8; bytes as usize];
        read_range(&plan_of(&files, &segs, bytes), 0, &mut want).unwrap();
        for (merge, threads, chunk) in [(None, 3, 65536), (Some(0), 3, 65536), (Some(4096), 2, 65536),
                                        (Some(1 << 20), 4, 65536), (Some(1 << 20), 1, 1 << 20), (Some(0), 3, 4096)] {
            let img = Img { bytes: Mutex::new(vec![0xAA; bytes as usize]), staging: chunk as usize };
            let sink = G6lSink { ctx: &img as *const Img as *mut c_void, alloc_staging: alloc, free_staging: free, copy };
            let st = run_cfg(plan_of(&files, &segs, bytes), bytes, &sink, Instant::now(), None, Mode::Read, threads,
                             chunk, merge).unwrap();
            let got = img.bytes.into_inner().unwrap();
            assert!(got == want, "merge {merge:?} threads {threads} chunk {chunk}: image differs");
            let need: u64 = segs.iter().map(|s| s.1).sum();
            // an Item128 range is read whole in each chunk it crosses, so both ways read at least the ranges' bytes
            assert!(st.read_bytes >= need, "merge {merge:?}: read {} of {need}", st.read_bytes);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn compacted_copies_load_the_same_image() {
        let dir = std::env::temp_dir().join(format!("g6load-compact-{}", std::process::id()));
        let cdir = dir.join("compact");
        std::fs::create_dir_all(&cdir).unwrap();
        let (files, segs, bytes) = fixture(&dir);
        let mut want = vec![0u8; bytes as usize];
        read_range(&plan_of(&files, &segs, bytes), 0, &mut want).unwrap();
        // compact src0 only, and only its ranges in image order 0, 2, 4, ...: the others stay on the source
        let src = std::fs::read(&files[0]).unwrap();
        let mut runs: Vec<(u64, u64)> = segs.iter().enumerate().filter(|(i, s)| s.3 == 0 && i % 2 == 0)
            .map(|(_, s)| (s.4, s.1)).collect();
        runs.sort_unstable();
        // overlapping ranges share one run, as a real compaction writes them
        let mut union: Vec<(u64, u64)> = Vec::new();
        for (o, n) in runs.iter().copied() {
            match union.last_mut() {
                Some(u) if o <= u.0 + u.1 => u.1 = u.1.max(o + n - u.0),
                _ => union.push((o, n)),
            }
        }
        let nmoved = runs.len();
        let runs = union;
        let (mut packed, mut map) = (Vec::new(), format!("g6map-v1 {}\n", src.len()));
        for (o, n) in &runs {
            map += &format!("{o} {n} {}\n", packed.len());
            packed.extend_from_slice(&src[*o as usize..(*o + *n) as usize]);
        }
        std::fs::write(cdir.join("src0"), &packed).unwrap();
        std::fs::write(cdir.join("src0.g6map"), &map).unwrap();
        for (mode, merge) in [(Mode::Read, None), (Mode::Read, Some(0)), (Mode::Read, Some(1 << 20)),
                              (Mode::Mmap, None)] {
            let mut p = plan_of(&files, &segs, bytes);
            let (nf, moved, left) = p.remap_compact(cdir.to_str().unwrap()).unwrap();
            assert_eq!((nf, moved, moved + left), (1, nmoved, segs.iter().filter(|s| s.3 == 0).count()));
            assert!(left > 0);
            let img = Img { bytes: Mutex::new(vec![0xAA; bytes as usize]), staging: 65536 };
            let sink = G6lSink { ctx: &img as *const Img as *mut c_void, alloc_staging: alloc, free_staging: free, copy };
            run_cfg(p, bytes, &sink, Instant::now(), None, mode, 3, 65536, merge).unwrap();
            assert!(img.bytes.into_inner().unwrap() == want, "merge {merge:?}: image differs");
        }
        // Every run is checked, not just the last: overflow, source bounds, copy bounds and overlap.
        for bad in ["no numbers", "0 1", "0 0 0", "18446744073709551615 2 0",
                    "0 2 18446744073709551615", "0 1 18446744073709551614\n2 1 0",
                    "2 2 0\n1 1 2", "0 3 0\n2 1 3"] {
            std::fs::write(cdir.join("src0.g6map"), format!("g6map-v1 {}\n{bad}\n", src.len())).unwrap();
            assert!(plan_of(&files, &segs, bytes).remap_compact(cdir.to_str().unwrap()).is_err(), "{bad}");
        }
        // a map written for another size of the source is refused
        std::fs::write(cdir.join("src0.g6map"), format!("g6map-v1 {}\n", src.len() + 1)).unwrap();
        assert!(plan_of(&files, &segs, bytes).remap_compact(cdir.to_str().unwrap()).unwrap_err().contains("stale"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn compact_knob_subprocesses() {
        let dir = std::env::temp_dir().join(format!("g6load-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for value in [None, Some(""), Some(dir.to_str().unwrap())] {
            // The knob table is cached per process. Test the real boot path in fresh processes, including the
            // usual soft descriptor limit: 300 sources plus their copies need more than 1024 descriptors.
            let mut cmd = std::process::Command::new("sh");
            cmd.args(["-c", "ulimit -Sn 1024; exec \"$@\"", "sh"])
                .arg(std::env::current_exe().unwrap())
                .args(["--exact", "tests::compact_knob_child", "--ignored", "--nocapture"])
                .env_clear();
            if let Some(v) = value { cmd.env("GR8_LOAD_COMPACT", v); }
            assert!(cmd.status().unwrap().success(), "compact knob {value:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "run by compact_knob_subprocesses with a fresh knob table"]
    fn compact_knob_child() {
        let dir = std::env::temp_dir().join(format!("g6load-env-child-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let compact = gr8knobs::LOAD_COMPACT.get();
        let mut files = Vec::new();
        let mut segs = Vec::new();
        for i in 0..300 {
            let name = format!("src{i}");
            let path = dir.join(&name);
            std::fs::write(&path, [0u8; 8]).unwrap();
            files.push(path.to_str().unwrap().to_owned());
            segs.push(Seg { pack_off: i as u64, nbytes: 1, op: Op::Copy, file: i, file_off: 3 });
            if let Some(c) = compact {
                // Deliberately distinguish the copy from the source to prove which descriptor was read.
                std::fs::write(std::path::Path::new(c).join(&name), [42u8]).unwrap();
                std::fs::write(std::path::Path::new(c).join(format!("{name}.g6map")), "g6map-v1 8\n3 1 0\n").unwrap();
            }
        }
        let p = Plan::from_segs(files, segs, Vec::new(), Vec::new(), 300).unwrap();
        let img = Img { bytes: Mutex::new(vec![0xAA; 300]), staging: 4096 };
        let sink = G6lSink { ctx: &img as *const Img as *mut c_void, alloc_staging: alloc, free_staging: free, copy };
        run_cfg(p, 300, &sink, Instant::now(), None, Mode::Read, 1, 4096, None).unwrap();
        assert_eq!(img.bytes.into_inner().unwrap(), vec![if compact.is_some() { 42 } else { 0 }; 300]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

}
