//! DeepSeek-V4.1 image input: the release's `inference/image_processor.py` (`load_image`) in Rust, up to the uint8 RGB
//! the vision tower reads, so the engine gets exactly the pixels the official code would build.
//!
//! An image is decoded (first frame), converted as PIL's `convert("RGB")` does (alpha dropped, not composited; grey
//! replicated; no EXIF rotation), planned (`plan_image_grid`: the min_pixels upscale, the patch rounding and the token
//! cap) and padded (`ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))`: `ImageOps.contain` with Pillow's
//! BICUBIC resample, pasted centred on grey). The resample is a port of Pillow's `libImaging/Resample.c` (8 bits per
//! channel: coefficients in fixed point with 22 fraction bits, a horizontal pass over the source rows the vertical pass
//! needs, then the vertical pass), so on a lossless input the bytes match Pillow's. Lossy formats (JPEG, lossy WebP)
//! go through other decoders than PIL's (libjpeg-turbo, libwebp) and can differ from them by a few levels.
//!
//! In the prompt an image is `n_llm_h * (n_llm_w + 1) + 2` copies of the image token (`num_image_tokens`); the engine
//! gets the padded RGB and the patch grid (`vit_h x vit_w`, 14 px patches) for each span.

use base64::Engine as _;
use sha2::{Digest, Sha256};

/// `vision_config` of the release's config.json.
pub const PATCH: usize = 14;
pub const DOWNSAMPLE: usize = 3;
pub const MAX_IMAGE_TOKENS: usize = 1024;
pub const MIN_PIXELS: usize = 295_936;
/// `image_token_id`: the placeholder `<｜deepseek_image｜>` and every position of an image span.
pub const IMAGE_TOKEN_ID: u32 = 129_264;
/// The grey `ImageOps.pad` fills with.
pub const PAD_GREY: u8 = 127;

/// Per request: at most this many images ...
pub const MAX_IMAGES_PER_REQUEST: usize = 16;
/// ... and at most this many bytes of base64 among them.
pub const MAX_IMAGE_B64_BYTES_PER_REQUEST: usize = 48 << 20;
/// PIL's decompression-bomb limit (`2 * Image.MAX_IMAGE_PIXELS`): PIL refuses a larger image, so this does too.
pub const MAX_IMAGE_PIXELS: u64 = 178_956_970;

#[derive(Debug, PartialEq)]
pub struct VisionError(pub String);

fn verr<T>(s: impl Into<String>) -> Result<T, VisionError> {
    Err(VisionError(s.into()))
}

// ---------------------------------------------------------------------------------------------------------------
// plan_image_grid (image_processor.py), in f64 as Python computes it
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub best_h: usize,
    pub best_w: usize,
}

impl Plan {
    pub fn n_span(&self) -> usize {
        num_image_tokens(self.n_llm_h, self.n_llm_w)
    }
    pub fn vit_h(&self) -> usize {
        self.best_h / PATCH
    }
    pub fn vit_w(&self) -> usize {
        self.best_w / PATCH
    }
}

pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

fn llm_grid(best_h: i64, best_w: i64) -> (i64, i64) {
    let (p, d) = (PATCH as i64, DOWNSAMPLE as f64);
    (((best_h.div_euclid(p)) as f64 / d).ceil() as i64, ((best_w.div_euclid(p)) as f64 / d).ceil() as i64)
}

fn solve_resize_ratio(height: i64, width: i64) -> (i64, i64) {
    let (p, max_n) = (PATCH as i64, MAX_IMAGE_TOKENS as i64);
    let r = height as f64 / width as f64;
    let max_w_float = (((max_n - 2) as f64) / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = p * DOWNSAMPLE as i64;
    if max_w_float < 1.0 {
        return ((max_n - 2).div_euclid(2) * cell, cell);
    }
    if max_h_float < 1.0 {
        return (cell, (max_n - 3) * cell);
    }
    let bw = (max_w_float.floor() as i64 * cell) as f64 / width as f64;
    let bh = (max_h_float.floor() as i64 * cell) as f64 / height as f64;
    let beta = bw.min(bh);
    (
        (height as f64 * beta / p as f64).floor() as i64 * p,
        (width as f64 * beta / p as f64).floor() as i64 * p,
    )
}

/// `plan_image_grid(width, height, args)` with the release's vision_config (max_wh_ratio null).
pub fn plan_image_grid(width: usize, height: usize) -> Plan {
    let p = PATCH as i64;
    let (mut w, mut h) = (width as i64, height as i64);
    let area = w * h;
    if 0 < area && area < MIN_PIXELS as i64 {
        let ratio = (MIN_PIXELS as f64 / area as f64).powf(0.5);
        w = (w as f64 * ratio) as i64;
        h = (h as f64 * ratio) as i64;
    }
    let mut best_w = (w as f64 / p as f64).ceil() as i64 * p;
    let mut best_h = (h as f64 / p as f64).ceil() as i64 * p;
    let (mut lh, mut lw) = llm_grid(best_h, best_w);
    if num_image_tokens(lh as usize, lw as usize) > MAX_IMAGE_TOKENS {
        (best_h, best_w) = solve_resize_ratio(h, w);
        (lh, lw) = llm_grid(best_h, best_w);
        debug_assert!(num_image_tokens(lh as usize, lw as usize) <= MAX_IMAGE_TOKENS);
    }
    Plan { n_llm_h: lh as usize, n_llm_w: lw as usize, best_h: best_h as usize, best_w: best_w as usize }
}

// ---------------------------------------------------------------------------------------------------------------
// Pillow's BICUBIC resample (libImaging/Resample.c, 8 bpc) and ImageOps.contain / pad
// ---------------------------------------------------------------------------------------------------------------

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc` for the whole input range (box = (0, in_size)): per output pixel the
/// first input index and the fixed-point taps.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
}

fn coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let in0 = 0.0f32;
    let in1 = in_size as f32;
    let scale = (in1 - in0) as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = in0 as f64 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C's (int) truncates toward zero
        let xmin = ((center - support + 0.5) as i32).max(0);
        let xmax = ((center + support + 0.5) as i32).min(in_size as i32) - xmin;
        let mut ww = 0.0;
        for x in 0..xmax {
            let w = bicubic(((x + xmin) as f64 - center + 0.5) * ss);
            k[x as usize] = w;
            ww += w;
        }
        let row = &mut kk[xx * ksize..(xx + 1) * ksize];
        for x in 0..xmax as usize {
            if ww != 0.0 {
                k[x] /= ww;
            }
            let v = k[x] * (1u32 << PRECISION_BITS) as f64;
            row[x] = if k[x] < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 };
        }
        bounds.push((xmin as usize, xmax.max(0) as usize));
    }
    Coeffs { ksize, bounds, kk }
}

#[inline]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((ow, oh), BICUBIC)` of an RGB image (packed rows, 3 bytes a pixel).
pub fn resize_bicubic(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    use rayon::prelude::*;
    assert_eq!(src.len(), w * h * 3);
    if (w, h) == (ow, oh) {
        return src.to_vec(); // Image.resize returns a copy
    }
    let need_h = ow != w;
    let need_v = oh != h;
    let ch = coeffs(w, ow);
    let mut cv = coeffs(h, oh);
    // the source rows the vertical pass reads
    let y_first = cv.bounds[0].0;
    let y_last = cv.bounds[oh - 1].0 + cv.bounds[oh - 1].1;
    let half = 1i32 << (PRECISION_BITS - 1);
    let (tmp, tw, _th): (std::borrow::Cow<[u8]>, usize, usize) = if need_h {
        for b in &mut cv.bounds {
            b.0 -= y_first;
        }
        let rows = y_last - y_first;
        let mut out = vec![0u8; ow * rows * 3];
        out.par_chunks_mut(ow * 3).enumerate().for_each(|(yy, orow)| {
            let irow = &src[(yy + y_first) * w * 3..(yy + y_first + 1) * w * 3];
            for xx in 0..ow {
                let (xmin, xmax) = ch.bounds[xx];
                let k = &ch.kk[xx * ch.ksize..xx * ch.ksize + xmax];
                let (mut s0, mut s1, mut s2) = (half, half, half);
                for (x, &kx) in k.iter().enumerate() {
                    let p = &irow[(x + xmin) * 3..(x + xmin) * 3 + 3];
                    s0 = s0.wrapping_add(p[0] as i32 * kx);
                    s1 = s1.wrapping_add(p[1] as i32 * kx);
                    s2 = s2.wrapping_add(p[2] as i32 * kx);
                }
                orow[xx * 3] = clip8(s0);
                orow[xx * 3 + 1] = clip8(s1);
                orow[xx * 3 + 2] = clip8(s2);
            }
        });
        (std::borrow::Cow::Owned(out), ow, rows)
    } else {
        (std::borrow::Cow::Borrowed(src), w, h)
    };
    if !need_v {
        return tmp.into_owned();
    }
    let mut out = vec![0u8; tw * oh * 3];
    out.par_chunks_mut(tw * 3).enumerate().for_each(|(yy, orow)| {
        let (ymin, ymax) = cv.bounds[yy];
        let k = &cv.kk[yy * cv.ksize..yy * cv.ksize + ymax];
        for xx in 0..tw {
            let (mut s0, mut s1, mut s2) = (half, half, half);
            for (y, &ky) in k.iter().enumerate() {
                let o = ((y + ymin) * tw + xx) * 3;
                s0 = s0.wrapping_add(tmp[o] as i32 * ky);
                s1 = s1.wrapping_add(tmp[o + 1] as i32 * ky);
                s2 = s2.wrapping_add(tmp[o + 2] as i32 * ky);
            }
            orow[xx * 3] = clip8(s0);
            orow[xx * 3 + 1] = clip8(s1);
            orow[xx * 3 + 2] = clip8(s2);
        }
    });
    out
}

/// `ImageOps.contain`'s target size: the largest size of the image's aspect within (tw, th), Python's round().
pub fn contain_size(w: usize, h: usize, tw: usize, th: usize) -> (usize, usize) {
    let im_ratio = w as f64 / h as f64;
    let dest_ratio = tw as f64 / th as f64;
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let nh = (h as f64 / w as f64 * tw as f64).round_ties_even() as usize;
            return (tw, nh);
        }
        let nw = (w as f64 / h as f64 * th as f64).round_ties_even() as usize;
        return (nw, th);
    }
    (tw, th)
}

/// `ImageOps.pad(image, (tw, th), color=(127, 127, 127))` (BICUBIC, centred).
pub fn pad(src: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Result<Vec<u8>, VisionError> {
    let (rw, rh) = contain_size(w, h, tw, th);
    if rw == 0 || rh == 0 {
        return verr(format!("a {w}x{h} image is too thin to resize to {tw}x{th}"));
    }
    let resized = resize_bicubic(src, w, h, rw, rh);
    if (rw, rh) == (tw, th) {
        return Ok(resized);
    }
    let mut out = vec![PAD_GREY; tw * th * 3];
    let (x0, y0) = if rw != tw {
        (((tw - rw) as f64 * 0.5).round_ties_even() as usize, 0)
    } else {
        (0, ((th - rh) as f64 * 0.5).round_ties_even() as usize)
    };
    for y in 0..rh.min(th - y0) {
        let n = rw.min(tw - x0) * 3;
        let o = ((y + y0) * tw + x0) * 3;
        out[o..o + n].copy_from_slice(&resized[y * rw * 3..y * rw * 3 + n]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// Decoding: PIL's Image.open(...).convert("RGB")
// ---------------------------------------------------------------------------------------------------------------

/// An RGB image, rows packed, 3 bytes a pixel.
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

fn check_pixels(w: u64, h: u64) -> Result<(), VisionError> {
    if w == 0 || h == 0 {
        return verr("the image is empty");
    }
    if w * h > MAX_IMAGE_PIXELS {
        return verr(format!("the image is {w}x{h} pixels, more than {MAX_IMAGE_PIXELS}"));
    }
    Ok(())
}

/// The first frame of a GIF as PIL builds it: the logical screen filled with the frame's transparent index (index 0
/// without one), the frame's indices pasted at its offset, then the palette (the frame's own, else the global one).
fn decode_gif(bytes: &[u8]) -> Result<Rgb, VisionError> {
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::Indexed);
    let e = |e: gif::DecodingError| VisionError(format!("cannot decode the GIF image: {e}"));
    let mut d = opts.read_info(bytes).map_err(e)?;
    let (sw, sh) = (d.width() as usize, d.height() as usize);
    check_pixels(sw as u64, sh as u64)?;
    let global = d.global_palette().map(<[u8]>::to_vec);
    let Some(f) = d.read_next_frame().map_err(e)? else { return verr("the GIF image has no frame") };
    let fill = f.transparent.unwrap_or(0);
    let mut idx = vec![fill; sw * sh];
    let (fl, ft, fw, fh) = (f.left as usize, f.top as usize, f.width as usize, f.height as usize);
    for y in 0..fh {
        if ft + y >= sh {
            break;
        }
        for x in 0..fw {
            if fl + x < sw {
                idx[(ft + y) * sw + fl + x] = f.buffer[y * fw + x];
            }
        }
    }
    let pal = f.palette.clone().or(global).unwrap_or_default();
    let mut px = vec![0u8; sw * sh * 3];
    for (i, &c) in idx.iter().enumerate() {
        let c = c as usize * 3;
        if c + 3 <= pal.len() {
            px[i * 3..i * 3 + 3].copy_from_slice(&pal[c..c + 3]);
        }
    }
    Ok(Rgb { w: sw, h: sh, px })
}

/// Decode PNG / JPEG / WebP / GIF bytes to RGB as PIL's `Image.open(...).convert("RGB")` does.
pub fn decode_rgb(bytes: &[u8]) -> Result<Rgb, VisionError> {
    use image::{DynamicImage as D, ImageFormat as F};
    let fmt = image::guess_format(bytes).map_err(|_| VisionError("unrecognised image data (PNG, JPEG, WebP or GIF)".into()))?;
    match fmt {
        F::Gif => return decode_gif(bytes),
        F::Png | F::Jpeg | F::WebP => {}
        other => return verr(format!("unsupported image format {other:?} (PNG, JPEG, WebP or GIF)")),
    }
    let (w, h) = image::ImageReader::with_format(std::io::Cursor::new(bytes), fmt)
        .into_dimensions()
        .map_err(|e| VisionError(format!("cannot read the image: {e}")))?;
    check_pixels(w as u64, h as u64)?;
    let mut r = image::ImageReader::with_format(std::io::Cursor::new(bytes), fmt);
    let mut lim = image::Limits::default();
    lim.max_alloc = Some(w as u64 * h as u64 * 8 + (64 << 20));
    r.limits(lim);
    let im = r.decode().map_err(|e| VisionError(format!("cannot decode the image: {e}")))?;
    let (w, h) = (im.width() as usize, im.height() as usize);
    let n = w * h;
    let mut px = Vec::with_capacity(n * 3);
    match im {
        D::ImageRgb8(b) => px = b.into_raw(),
        D::ImageRgba8(b) => b.chunks_exact(4).for_each(|p| px.extend_from_slice(&p[..3])),
        D::ImageLuma8(b) => b.iter().for_each(|&v| px.extend_from_slice(&[v, v, v])),
        D::ImageLumaA8(b) => b.chunks_exact(2).for_each(|p| px.extend_from_slice(&[p[0], p[0], p[0]])),
        // PIL opens 16-bit grey as I;16, and its I;16 -> RGB conversion clips at 255
        D::ImageLuma16(b) => b.iter().for_each(|&v| {
            let v = v.min(255) as u8;
            px.extend_from_slice(&[v, v, v])
        }),
        // 16-bit colour and grey+alpha load as 8-bit RGB(A) in PIL: the high byte
        D::ImageLumaA16(b) => b.chunks_exact(2).for_each(|p| {
            let v = (p[0] >> 8) as u8;
            px.extend_from_slice(&[v, v, v])
        }),
        D::ImageRgb16(b) => b.iter().for_each(|&v| px.push((v >> 8) as u8)),
        D::ImageRgba16(b) => b.chunks_exact(4).for_each(|p| px.extend_from_slice(&[(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8])),
        other => px = other.to_rgb8().into_raw(),
    }
    debug_assert_eq!(px.len(), n * 3);
    Ok(Rgb { w, h, px })
}

// ---------------------------------------------------------------------------------------------------------------
// One image, end to end
// ---------------------------------------------------------------------------------------------------------------

/// An image ready for the engine.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub plan: Plan,
    /// sha256 over (vit_h u32 LE, vit_w u32 LE, rgb), first 16 hex digits
    pub key: String,
    /// best_h x best_w x 3
    pub rgb: Vec<u8>,
}

impl Prepared {
    pub fn vit_h(&self) -> usize {
        self.plan.vit_h()
    }
    pub fn vit_w(&self) -> usize {
        self.plan.vit_w()
    }
}

pub fn content_key(vit_h: usize, vit_w: usize, rgb: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update((vit_h as u32).to_le_bytes());
    h.update((vit_w as u32).to_le_bytes());
    h.update(rgb);
    let d = h.finalize();
    d[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// RGB -> the padded RGB load_image builds (before it normalises).
pub fn prepare_rgb(im: &Rgb) -> Result<Prepared, VisionError> {
    let plan = plan_image_grid(im.w, im.h);
    let rgb = pad(&im.px, im.w, im.h, plan.best_w, plan.best_h)?;
    let key = content_key(plan.vit_h(), plan.vit_w(), &rgb);
    Ok(Prepared { plan, key, rgb })
}

/// Base64 as Python's `base64.b64decode` reads a request's: whitespace is skipped, the padding may be left off.
pub fn b64_decode(s: &str) -> Result<Vec<u8>, VisionError> {
    use base64::engine::{general_purpose::GeneralPurpose, DecodePaddingMode, GeneralPurposeConfig};
    const E: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
    );
    let r = if s.bytes().any(|b| b.is_ascii_whitespace()) {
        let clean: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        E.decode(clean)
    } else {
        E.decode(s)
    };
    r.map_err(|e| VisionError(format!("bad base64 image data: {e}")))
}

pub fn b64_encode(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

/// Base64 image data -> what the engine gets.
pub fn prepare_b64(b64: &str) -> Result<Prepared, VisionError> {
    let bytes = b64_decode(b64)?;
    prepare_rgb(&decode_rgb(&bytes)?)
}

/// The plan alone, from the image's header (for /tokenize, which counts the prompt without building the pixels).
pub fn plan_b64(b64: &str) -> Result<Plan, VisionError> {
    let bytes = b64_decode(b64)?;
    let (w, h) = match image::guess_format(&bytes) {
        Ok(image::ImageFormat::Gif) => {
            let d = gif::DecodeOptions::new()
                .read_info(&bytes[..])
                .map_err(|e| VisionError(format!("cannot decode the GIF image: {e}")))?;
            (d.width() as u32, d.height() as u32)
        }
        Ok(f @ (image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP)) => {
            image::ImageReader::with_format(std::io::Cursor::new(&bytes), f)
                .into_dimensions()
                .map_err(|e| VisionError(format!("cannot read the image: {e}")))?
        }
        _ => return verr("unrecognised image data (PNG, JPEG, WebP or GIF)"),
    };
    check_pixels(w as u64, h as u64)?;
    Ok(plan_image_grid(w as usize, h as usize))
}

/// The per-request limits on the images' base64.
pub fn check_budget<S: AsRef<str>>(images: &[S]) -> Result<(), VisionError> {
    if images.len() > MAX_IMAGES_PER_REQUEST {
        return verr(format!("{} images in one request; at most {MAX_IMAGES_PER_REQUEST}", images.len()));
    }
    let total: usize = images.iter().map(|s| s.as_ref().len()).sum();
    if total > MAX_IMAGE_B64_BYTES_PER_REQUEST {
        return verr(format!(
            "the request's images are {total} bytes of base64; at most {MAX_IMAGE_B64_BYTES_PER_REQUEST}"
        ));
    }
    Ok(())
}

/// One image span in the final prompt, for the engine (PROTOCOL.md `images`).
#[derive(Clone, Debug)]
pub struct Span {
    /// index of the span's first token in the prompt
    pub pos: usize,
    pub vit_h: usize,
    pub vit_w: usize,
    pub key: String,
    /// the padded RGB, base64 (standard alphabet, padded, no newlines)
    pub rgb_b64: String,
}

/// Replace the i-th image token of `ids` (the tokenized prompt, one token a placeholder) with image i's span.
/// `n_spans[i]` is image i's span length. Returns the new ids and each span's first position. Errors when the number
/// of image tokens is not the number of images.
pub fn expand(ids: &[u32], n_spans: &[usize]) -> Result<(Vec<u32>, Vec<usize>), VisionError> {
    let found = ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    if found != n_spans.len() {
        return verr(format!("the prompt has {found} image tokens but the request has {} images", n_spans.len()));
    }
    let extra: usize = n_spans.iter().map(|n| n - 1).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut pos = Vec::with_capacity(n_spans.len());
    let mut it = n_spans.iter();
    for &t in ids {
        if t == IMAGE_TOKEN_ID {
            let n = *it.next().unwrap();
            pos.push(out.len());
            out.resize(out.len() + n, IMAGE_TOKEN_ID);
        } else {
            out.push(t);
        }
    }
    Ok((out, pos))
}

/// A token count in the unexpanded prompt -> the same point in the expanded one (spans that start before it count in
/// full).
pub fn expanded_index(unexpanded_ids: &[u32], n: usize, n_spans: &[usize]) -> usize {
    let mut k = 0;
    let mut out = 0;
    for &t in unexpanded_ids.iter().take(n) {
        if t == IMAGE_TOKEN_ID {
            out += n_spans.get(k).copied().unwrap_or(1);
            k += 1;
        } else {
            out += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/ref/ds41/fixtures/vision")
    }

    fn read_ppm(p: &Path) -> (usize, usize, Vec<u8>) {
        let b = std::fs::read(p).unwrap();
        // "P6\n<w> <h>\n255\n" + bytes
        let mut fields = Vec::new();
        let mut i = 0;
        while fields.len() < 4 {
            while b[i].is_ascii_whitespace() {
                i += 1;
            }
            let s = i;
            while !b[i].is_ascii_whitespace() {
                i += 1;
            }
            fields.push(String::from_utf8(b[s..i].to_vec()).unwrap());
        }
        i += 1;
        assert_eq!(fields[0], "P6");
        let (w, h) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
        (w, h, b[i..].to_vec())
    }

    #[test]
    fn plan_matches_the_official_table() {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("plan.json")).unwrap()).unwrap();
        let cases = v.as_array().unwrap();
        assert!(cases.len() >= 20);
        for c in cases {
            let g = |k: &str| c[k].as_u64().unwrap() as usize;
            let p = plan_image_grid(g("w"), g("h"));
            assert_eq!(
                (p.best_w, p.best_h, p.n_llm_h, p.n_llm_w, p.n_span()),
                (g("best_w"), g("best_h"), g("n_llm_h"), g("n_llm_w"), g("n_span_tokens")),
                "{c}"
            );
        }
    }

    /// The fixtures' padded RGB was made by the official load_image calls (engine/ref/ds41/vision_golden.py).
    #[test]
    fn pad_matches_the_official_pixels() {
        for name in ["square_300x300", "wide_1100x280", "tall_320x1000"] {
            let png = std::fs::read(fixtures().join(format!("{name}.png"))).unwrap();
            let meta: Value =
                serde_json::from_str(&std::fs::read_to_string(fixtures().join(format!("{name}.json"))).unwrap()).unwrap();
            let (w, h, want) = read_ppm(&fixtures().join(format!("{name}.ppm")));
            let p = prepare_rgb(&decode_rgb(&png).unwrap()).unwrap();
            assert_eq!((p.plan.best_w, p.plan.best_h), (w, h), "{name}");
            assert_eq!(p.vit_h() as u64, meta["n_vit_h"].as_u64().unwrap());
            assert_eq!(p.vit_w() as u64, meta["n_vit_w"].as_u64().unwrap());
            assert_eq!(p.plan.n_span() as u64, meta["n_span_tokens"].as_u64().unwrap());
            assert!(p.rgb == want, "{name}: padded RGB differs from the official bytes");
        }
    }

    /// Random images against PIL (frontend/tests/vision_pil_cases.py wrote them): sizes that upscale and downscale,
    /// both aspects, random content, the PNG colour types, GIF and lossless WebP. GR8_VISION_CASES=DIR runs a set made
    /// with `--n`; its lossy cases (JPEG, lossy WebP) only report how far they are off.
    #[test]
    fn pad_matches_pil_on_random_images() {
        let dir = match std::env::var("GR8_VISION_CASES") {
            Ok(d) => PathBuf::from(d),
            Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("src/vision_cases"),
        };
        let idx: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
        let cases = idx.as_array().unwrap();
        assert!(cases.len() >= 10);
        for c in cases {
            let name = c["file"].as_str().unwrap();
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let p = prepare_rgb(&decode_rgb(&bytes).unwrap()).unwrap();
            assert_eq!(
                (p.plan.best_w, p.plan.best_h),
                (c["best_w"].as_u64().unwrap() as usize, c["best_h"].as_u64().unwrap() as usize),
                "{name}"
            );
            let same = content_key(p.vit_h(), p.vit_w(), &p.rgb) == c["key"].as_str().unwrap();
            if let Ok(want) = std::fs::read(dir.join(format!("{name}.rgb"))) {
                let d: Vec<u8> = p.rgb.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).collect();
                let n = d.iter().filter(|&&x| x != 0).count();
                eprintln!(
                    "{name}: {}x{} exact={same} differing bytes {n}/{} max |diff| {}",
                    p.plan.best_w,
                    p.plan.best_h,
                    d.len(),
                    d.iter().max().unwrap_or(&0)
                );
            }
            if c["exact"].as_bool().unwrap_or(true) {
                assert!(same, "{name}: padded RGB differs from PIL's");
            }
        }
    }

    #[test]
    fn expand_and_positions() {
        let t = IMAGE_TOKEN_ID;
        let (ids, pos) = expand(&[1, t, 2, t], &[3, 2]).unwrap();
        assert_eq!(ids, vec![1, t, t, t, 2, t, t]);
        assert_eq!(pos, vec![1, 5]);
        assert!(expand(&[1, t], &[]).is_err());
        assert!(expand(&[1], &[3]).is_err());
        assert_eq!(expanded_index(&[1, t, 2, t], 2, &[3, 2]), 4);
        assert_eq!(expanded_index(&[1, t, 2, t], 1, &[3, 2]), 1);
    }

    #[test]
    fn budget_and_base64() {
        assert!(check_budget(&vec!["a"; MAX_IMAGES_PER_REQUEST]).is_ok());
        assert!(check_budget(&vec!["a"; MAX_IMAGES_PER_REQUEST + 1]).is_err());
        let big = "A".repeat(MAX_IMAGE_B64_BYTES_PER_REQUEST / 2 + 1);
        assert!(check_budget(&[big.as_str(), big.as_str()]).is_err());
        assert_eq!(b64_decode("aGk=").unwrap(), b"hi");
        assert_eq!(b64_decode("aGk").unwrap(), b"hi");
        assert_eq!(b64_decode("aG\nk=").unwrap(), b"hi");
        assert!(b64_decode("a*Gk").is_err());
        assert!(prepare_b64("aGk=").is_err());
        assert_eq!(content_key(1, 2, b"x").len(), 16);
    }
}
