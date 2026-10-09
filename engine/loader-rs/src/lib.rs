//! g6 pack loader. Builds a rank's device image (the pack's bytes at the manifest's offsets) from the pack file and,
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
//! files at the offsets of the index slices (src/v2plan.rs). g6l_plan_load then loads it like a pack.

mod json;
mod plan;
mod sys;
mod v2plan;

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
}

unsafe impl Sync for Loader<'_> {}

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
            if off % ALIGN == 0 && (dst.as_ptr() as u64) % ALIGN == 0 {
                let head = dst.len() / ALIGN as usize * ALIGN as usize;
                if head > 0 {
                    read_exact_at(d, &mut dst[..head], off).map_err(|e| format!("{name}: {e} (O_DIRECT)"))?;
                    done = head;
                }
            }
        }
        if done < dst.len() {
            read_exact_at(&s.buffered, &mut dst[done..], off + done as u64).map_err(|e| format!("{name}: {e}"))?;
            sys::fadvise_dontneed(&s.buffered, off + done as u64, (dst.len() - done) as u64);
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
                    let m = &self.plan.mems[s.file];
                    buf[(x - a) as usize..(y - a) as usize]
                        .copy_from_slice(&m[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
                }
                Op::Item128 => {
                    let n = s.nbytes as usize;
                    self.read(s.file, s.file_off, half.slice(n))?;
                    plan::item128_to_g6(half.slice(n), perm.slice(n));
                    buf[(x - a) as usize..(y - a) as usize]
                        .copy_from_slice(&perm.slice(n)[(x - s.pack_off) as usize..(y - s.pack_off) as usize]);
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
    run(Plan::new(pack, pack_bytes, tensors, sources, root)?, pack_bytes, sink, t0)
}

/// Fills image bytes 0..pack_bytes from `plan` through `sink`. The stats count time from `t0`.
fn run(plan: Plan, pack_bytes: u64, sink: &G6lSink, t0: Instant) -> Result<G6lStats, String> {
    let mode = match std::env::var("G6_LOAD_MODE").as_deref() {
        Ok("read") | Err(_) => Mode::Read,
        Ok("mmap") => Mode::Mmap,
        Ok(v) => return Err(format!("G6_LOAD_MODE={v}: use mmap or read")),
    };
    let threads = env_u64("G6_LOAD_THREADS", 8) as usize;
    let chunk = env_u64("G6_LOAD_CHUNK_MB", 32) << 20;
    let files = plan.files.iter().map(|p| open_src(p, mode)).collect::<Result<Vec<_>, _>>()?;
    let l = Loader { plan, files, mode, sink };
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
                let mut half = sys::AlignedBuf::new(max_half.max(1));
                let mut perm = sys::AlignedBuf::new(max_half.max(1));
                while !failed.load(Ordering::Relaxed) {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= nchunks {
                        break;
                    }
                    let a = c * chunk;
                    let n = chunk.min(pack_bytes - a) as usize;
                    let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
                    let r = l.fill(a, buf, &mut half, &mut perm).and_then(|read| {
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
/// `h` must come from g6l_v2_plan and be loaded once; `sink`, `stats` and `err` as for g6l_load.
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

/// Frees a g6l_v2_plan handle.
///
/// # Safety
/// `h` must come from g6l_v2_plan and must not be used again.
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
