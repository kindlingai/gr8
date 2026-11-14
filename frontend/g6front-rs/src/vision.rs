//! Image input: each model's image processor in Rust, up to the uint8 RGB its vision tower reads, so the engine gets
//! exactly the pixels the reference code would build. `Model` picks the model: DeepSeek-V4.1 (below), GLM-5.3-Flash
//! (`glm5v`) or Inkling-2-Small (`ink2`).
//!
//! DeepSeek-V4.1: the release's `inference/image_processor.py` (`load_image`).
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
//!
//! GLM-5.3-Flash: transformers' `Glm5NextImageProcessor` (PIL backend), in `glm5v` (docs/FLASH-VISION.md). An image is
//! `vit_h * vit_w / 4` copies of `<|image|>` between the template's `<|begin_of_image|>` and `<|end_of_image|>`.
//!
//! Inkling-2-Small: the checkpoint's image processor, in `ink2`. No resize: transparency is composited on a grey
//! checkerboard, the image is padded on the right and the bottom to 40 px multiples, and it is `ceil(h / 40) *
//! ceil(w / 40)` copies of `<|unused_200054|>` after the template's `<|content_image|>`.

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
    ((height as f64 * beta / p as f64).floor() as i64 * p, (width as f64 * beta / p as f64).floor() as i64 * p)
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

/// The first frame of a GIF: the logical screen as palette indices, the palette, and the frame's transparent index.
struct GifFrame {
    w: usize,
    h: usize,
    idx: Vec<u8>,
    pal: Vec<u8>,
    transparent: Option<u8>,
}

/// The first frame of a GIF as PIL builds it: the logical screen filled with the frame's transparent index (index 0
/// without one), the frame's indices pasted at its offset; the palette is the frame's own, else the global one.
fn gif_frame(bytes: &[u8]) -> Result<GifFrame, VisionError> {
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
    Ok(GifFrame { w: sw, h: sh, idx, pal, transparent: f.transparent })
}

/// The first frame of a GIF through its palette, transparency dropped.
fn decode_gif(bytes: &[u8]) -> Result<Rgb, VisionError> {
    let g = gif_frame(bytes)?;
    let mut px = vec![0u8; g.w * g.h * 3];
    for (i, &c) in g.idx.iter().enumerate() {
        let c = c as usize * 3;
        if c + 3 <= g.pal.len() {
            px[i * 3..i * 3 + 3].copy_from_slice(&g.pal[c..c + 3]);
        }
    }
    Ok(Rgb { w: g.w, h: g.h, px })
}

/// A PNG, JPEG or WebP image decoded by the `image` crate, under the pixel limit.
fn decode_dynamic(bytes: &[u8], fmt: image::ImageFormat) -> Result<image::DynamicImage, VisionError> {
    let (w, h) = image::ImageReader::with_format(std::io::Cursor::new(bytes), fmt)
        .into_dimensions()
        .map_err(|e| VisionError(format!("cannot read the image: {e}")))?;
    check_pixels(w as u64, h as u64)?;
    let mut r = image::ImageReader::with_format(std::io::Cursor::new(bytes), fmt);
    let mut lim = image::Limits::default();
    lim.max_alloc = Some(w as u64 * h as u64 * 8 + (64 << 20));
    r.limits(lim);
    r.decode().map_err(|e| VisionError(format!("cannot decode the image: {e}")))
}

/// The format of image bytes: PNG, JPEG, WebP or GIF, else an error.
fn image_format(bytes: &[u8]) -> Result<image::ImageFormat, VisionError> {
    use image::ImageFormat as F;
    let fmt = image::guess_format(bytes)
        .map_err(|_| VisionError("unrecognised image data (PNG, JPEG, WebP or GIF)".into()))?;
    match fmt {
        F::Gif | F::Png | F::Jpeg | F::WebP => Ok(fmt),
        other => verr(format!("unsupported image format {other:?} (PNG, JPEG, WebP or GIF)")),
    }
}

/// Decode PNG / JPEG / WebP / GIF bytes to RGB as PIL's `Image.open(...).convert("RGB")` does.
pub fn decode_rgb(bytes: &[u8]) -> Result<Rgb, VisionError> {
    use image::DynamicImage as D;
    let fmt = image_format(bytes)?;
    if fmt == image::ImageFormat::Gif {
        return decode_gif(bytes);
    }
    let im = decode_dynamic(bytes, fmt)?;
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
        D::ImageRgba16(b) => b
            .chunks_exact(4)
            .for_each(|p| px.extend_from_slice(&[(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8])),
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
    /// the patch grid (14 px patches; Inkling-2-Small: 20 px)
    pub vit_h: usize,
    pub vit_w: usize,
    /// the image's span in the prompt, in image tokens
    pub n_span: usize,
    /// sha256 over (vit_h u32 LE, vit_w u32 LE, rgb), first 16 hex digits
    pub key: String,
    /// (vit_h * patch) x (vit_w * patch) x 3
    pub rgb: Vec<u8>,
}

/// The model whose image processor and image token a request's images take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Ds41,
    Glm5v,
    Ink2,
}

impl Model {
    /// the image token: the prompt's placeholder, and every token of an image span
    pub fn token(self) -> u32 {
        match self {
            Model::Ds41 => IMAGE_TOKEN_ID,
            Model::Glm5v => glm5v::IMAGE_TOKEN_ID,
            Model::Ink2 => ink2::IMAGE_TOKEN_ID,
        }
    }
    pub fn prepare_rgb(self, im: &Rgb) -> Result<Prepared, VisionError> {
        match self {
            Model::Ds41 => prepare_rgb(im),
            Model::Glm5v => glm5v::prepare_rgb(im),
            Model::Ink2 => ink2::prepare_rgb(im),
        }
    }
    /// Image bytes -> RGB as the model's processor reads them: Inkling-2-Small composites transparency on its
    /// checkerboard, the others drop it.
    pub fn decode(self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        match self {
            Model::Ink2 => ink2::decode_ink2_rgb(bytes),
            Model::Ds41 | Model::Glm5v => decode_rgb(bytes),
        }
    }
    /// Base64 image data -> what the engine gets.
    pub fn prepare_b64(self, b64: &str) -> Result<Prepared, VisionError> {
        self.prepare_rgb(&self.decode(&b64_decode(b64)?)?)
    }
    /// The span alone, from the image's header (for /tokenize, which counts the prompt without building the pixels).
    pub fn n_span_b64(self, b64: &str) -> Result<usize, VisionError> {
        let (w, h) = dims_b64(b64)?;
        Ok(match self {
            Model::Ds41 => plan_image_grid(w, h).n_span(),
            Model::Glm5v => glm5v::plan(w, h).tokens(),
            Model::Ink2 => ink2::plan(w, h)?.tokens,
        })
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
    Ok(Prepared { vit_h: plan.vit_h(), vit_w: plan.vit_w(), n_span: plan.n_span(), key, rgb })
}

/// GLM-5.3-Flash: transformers' `Glm5NextImageProcessor` (PIL backend) up to the padded uint8 RGB, with the release's
/// processor_config.json (patch 14, merge 2, temporal patch 2, 16 to 8000 image tokens). docs/FLASH-VISION.md has the
/// steps.
pub mod glm5v {
    use super::{content_key, resize_bicubic, Prepared, Rgb, VisionError, PATCH};

    /// `image_token_id`: `<|image|>`, the template's placeholder and every token of an image span
    pub const IMAGE_TOKEN_ID: u32 = 154_854;
    pub const MERGE: usize = 2;
    pub const TEMPORAL: usize = 2;
    pub const MIN_TOKENS: usize = 16;
    pub const MAX_TOKENS: usize = 8000;
    /// the canvas is a multiple of this (patch * merge * patch_expand_factor)
    pub const FACTOR: usize = PATCH * MERGE;

    /// The canvas (target) and the size the image is resized to inside it (content); the rest is zero padding on the
    /// right and the bottom.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Plan {
        pub target_h: usize,
        pub target_w: usize,
        pub content_h: usize,
        pub content_w: usize,
    }

    impl Plan {
        pub fn vit_h(&self) -> usize {
            self.target_h / PATCH
        }
        pub fn vit_w(&self) -> usize {
            self.target_w / PATCH
        }
        pub fn tokens(&self) -> usize {
            (self.vit_h() / MERGE) * (self.vit_w() / MERGE)
        }
    }

    fn align(v: usize, f: usize) -> usize {
        v.div_ceil(f) * f
    }

    /// `smart_resize(num_frames=2, height, width, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000)`, in
    /// f64 where Python computes in floats.
    pub fn smart_resize(height: usize, width: usize) -> (usize, usize) {
        let (f, frames) = (FACTOR, TEMPORAL);
        let ppt = TEMPORAL * f * f;
        let (min_pixels, max_pixels) = (MIN_TOKENS * ppt, MAX_TOKENS * ppt);
        let (mut ah, mut aw) = (align(height, f), align(width, f));
        let mut budget = frames * ah * aw;
        if budget < min_pixels {
            let scale = (min_pixels as f64 / (frames * height * width) as f64).sqrt();
            ah = align(((height as f64 * scale).ceil() as usize).max(1), f);
            aw = align(((width as f64 * scale).ceil() as usize).max(1), f);
            budget = frames * ah * aw;
        }
        if budget > max_pixels {
            // fit_within_budget: the tallest content height whose aligned canvas fits, by bisection
            let (mut low, mut high) = (1i64, height as i64);
            let (mut bh, mut bw) = (f, f);
            while low <= high {
                let ch = ((low + high) / 2) as usize;
                let cw = ((width as f64 * ch as f64 / height as f64).floor() as usize).max(1);
                let (th, tw) = (align(ch, f), align(cw, f));
                if frames * th * tw <= max_pixels {
                    (bh, bw) = (th, tw);
                    low = ch as i64 + 1;
                } else {
                    high = ch as i64 - 1;
                }
            }
            (ah, aw) = (bh, bw);
        }
        (ah, aw)
    }

    /// The processor's `resize`: the canvas, and the content size (the image is not scaled up unless it is below the
    /// token floor).
    pub fn plan(width: usize, height: usize) -> Plan {
        let (th, tw) = smart_resize(height, width);
        let mut scale = (th as f64 / height as f64).min(tw as f64 / width as f64);
        if TEMPORAL * height * width >= TEMPORAL * FACTOR * FACTOR * MIN_TOKENS {
            scale = scale.min(1.0);
        }
        let ch = ((height as f64 * scale).floor() as usize).min(th).max(1);
        let cw = ((width as f64 * scale).floor() as usize).min(tw).max(1);
        Plan { target_h: th, target_w: tw, content_h: ch, content_w: cw }
    }

    /// RGB -> the padded RGB the processor builds before it rescales: Pillow's BICUBIC to the content size when that
    /// differs, then zeros (black) to the canvas on the right and the bottom.
    pub fn prepare_rgb(im: &Rgb) -> Result<Prepared, VisionError> {
        let p = plan(im.w, im.h);
        let content = resize_bicubic(&im.px, im.w, im.h, p.content_w, p.content_h);
        let mut rgb = vec![0u8; p.target_h * p.target_w * 3];
        for y in 0..p.content_h {
            let o = y * p.target_w * 3;
            rgb[o..o + p.content_w * 3].copy_from_slice(&content[y * p.content_w * 3..(y + 1) * p.content_w * 3]);
        }
        let key = content_key(p.vit_h(), p.vit_w(), &rgb);
        Ok(Prepared { vit_h: p.vit_h(), vit_w: p.vit_w(), n_span: p.tokens(), key, rgb })
    }
}

/// Inkling-2-Small: the checkpoint's image processor up to the padded uint8 RGB (engine/runtime/ink2_vision_plan.h has
/// the same geometry). The image is not resized. Transparency is composited on an 8 px grey checkerboard, then the
/// image is padded with zeros on the right and the bottom to multiples of 40 px: one image token per 40 x 40 px, each a
/// 2 x 2 block of 20 px patches.
pub mod ink2 {
    use super::{content_key, decode_dynamic, decode_rgb, gif_frame, image_format, verr, Prepared, Rgb, VisionError};

    /// `image_token_id`: `<|unused_200054|>`, the template's placeholder and every token of an image span. The
    /// `<|content_image|>` before it is a text token.
    pub const IMAGE_TOKEN_ID: u32 = 200_054;
    pub const PATCH: usize = 20;
    pub const MERGE: usize = 2;
    pub const FACTOR: usize = PATCH * MERGE;
    pub const MIN_TOKENS: usize = 1;
    /// The engine's budget: 3200 x 3200 px, or 3840 x 2160 (5184 tokens).
    pub const MAX_TOKENS: usize = 6400;
    /// The checkerboard under transparency: squares of this many pixels, the two greys.
    pub const CHECKER: usize = 8;
    pub const CHECKER_LIGHT: u8 = 204;
    pub const CHECKER_DARK: u8 = 153;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Plan {
        /// image tokens down and across
        pub rows: usize,
        pub cols: usize,
        /// patches down and across
        pub vit_h: usize,
        pub vit_w: usize,
        /// the padded canvas, pixels
        pub pad_h: usize,
        pub pad_w: usize,
        pub tokens: usize,
    }

    /// The grid of a width x height image; an error outside the token budget.
    pub fn plan(width: usize, height: usize) -> Result<Plan, VisionError> {
        if width == 0 || height == 0 {
            return verr("the image is empty");
        }
        let (rows, cols) = (height.div_ceil(FACTOR), width.div_ceil(FACTOR));
        let tokens = rows * cols;
        if !(MIN_TOKENS..=MAX_TOKENS).contains(&tokens) {
            return verr(format!(
                "the image is {width}x{height} pixels, {tokens} image tokens; at most {MAX_TOKENS} (one per {FACTOR}x{FACTOR} pixels)"
            ));
        }
        let (vit_h, vit_w) = (rows * MERGE, cols * MERGE);
        Ok(Plan { rows, cols, vit_h, vit_w, pad_h: vit_h * PATCH, pad_w: vit_w * PATCH, tokens })
    }

    /// The checkerboard's grey at (y, x).
    pub fn checker(y: usize, x: usize) -> u8 {
        if (y / CHECKER + x / CHECKER) % 2 == 0 {
            CHECKER_LIGHT
        } else {
            CHECKER_DARK
        }
    }

    /// One channel over the background at alpha a, rounded to nearest in integers.
    pub fn over(c: u8, a: u8, bg: u8) -> u8 {
        ((c as u32 * a as u32 + bg as u32 * (255 - a as u32) + 127) / 255) as u8
    }

    /// RGBA rows -> RGB over the checkerboard.
    pub fn composite(rgba: &[u8], w: usize, h: usize) -> Vec<u8> {
        let mut px = Vec::with_capacity(w * h * 3);
        for (i, p) in rgba.chunks_exact(4).enumerate() {
            let bg = checker(i / w, i % w);
            px.extend_from_slice(&[over(p[0], p[3], bg), over(p[1], p[3], bg), over(p[2], p[3], bg)]);
        }
        debug_assert_eq!(px.len(), w * h * 3);
        px
    }

    /// Decode PNG / JPEG / WebP / GIF bytes (first frame) to RGB. An image with an alpha channel or a transparent
    /// palette index is brought to 8-bit RGBA (16-bit samples: the high byte) and composited on the checkerboard; an
    /// opaque image decodes as `decode_rgb` does.
    pub fn decode_ink2_rgb(bytes: &[u8]) -> Result<Rgb, VisionError> {
        use image::DynamicImage as D;
        let fmt = image_format(bytes)?;
        if fmt == image::ImageFormat::Gif {
            let g = gif_frame(bytes)?;
            let Some(t) = g.transparent else { return decode_rgb(bytes) };
            let mut rgba = vec![0u8; g.w * g.h * 4];
            for (i, &c) in g.idx.iter().enumerate() {
                let at = c as usize * 3;
                if at + 3 <= g.pal.len() {
                    rgba[i * 4..i * 4 + 3].copy_from_slice(&g.pal[at..at + 3]);
                }
                rgba[i * 4 + 3] = if c == t { 0 } else { 255 };
            }
            return Ok(Rgb { w: g.w, h: g.h, px: composite(&rgba, g.w, g.h) });
        }
        let im = decode_dynamic(bytes, fmt)?;
        let (w, h) = (im.width() as usize, im.height() as usize);
        let hi = |v: u16| (v >> 8) as u8;
        let rgba: Vec<u8> = match im {
            D::ImageRgba8(b) => b.into_raw(),
            D::ImageLumaA8(b) => b.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
            D::ImageRgba16(b) => b.iter().map(|&v| hi(v)).collect(),
            D::ImageLumaA16(b) => b.chunks_exact(2).flat_map(|p| [hi(p[0]), hi(p[0]), hi(p[0]), hi(p[1])]).collect(),
            D::ImageRgba32F(_) => im.to_rgba8().into_raw(),
            _ => return decode_rgb(bytes),
        };
        Ok(Rgb { w, h, px: composite(&rgba, w, h) })
    }

    /// RGB -> the padded RGB for the engine: the image at the top left, zeros to the canvas on the right and the bottom.
    pub fn prepare_rgb(im: &Rgb) -> Result<Prepared, VisionError> {
        let p = plan(im.w, im.h)?;
        let mut rgb = vec![0u8; p.pad_h * p.pad_w * 3];
        for y in 0..im.h {
            let o = y * p.pad_w * 3;
            rgb[o..o + im.w * 3].copy_from_slice(&im.px[y * im.w * 3..(y + 1) * im.w * 3]);
        }
        let key = content_key(p.vit_h, p.vit_w, &rgb);
        Ok(Prepared { vit_h: p.vit_h, vit_w: p.vit_w, n_span: p.tokens, key, rgb })
    }
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

/// An image block's base64 data. Inline data only: a data URL (`image_url.url`, `image_url` as a string, `url`), an
/// Anthropic `source` of type base64, or `data`. A URL the server would have to fetch is refused.
pub fn block_b64(b: &serde_json::Value) -> Result<&str, VisionError> {
    use serde_json::Value;
    fn data_url(u: &str) -> Result<&str, VisionError> {
        let Some(rest) = u.strip_prefix("data:") else {
            return verr("image URLs are not fetched; send the image inline as a data: URL or base64");
        };
        match rest.split_once(',') {
            Some((meta, data)) if meta.ends_with(";base64") => Ok(data),
            _ => verr("an image data: URL must be base64 (data:<type>;base64,<data>)"),
        }
    }
    match b.get("image_url") {
        Some(Value::String(u)) => return data_url(u),
        Some(Value::Object(o)) => {
            return data_url(o.get("url").and_then(Value::as_str).unwrap_or(""));
        }
        _ => {}
    }
    if let Some(src) = b.get("source").filter(|v| !v.is_null()) {
        return match src {
            Value::String(u) => data_url(u),
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("base64") => {
                o.get("data").and_then(Value::as_str).ok_or_else(|| VisionError("image source has no data".into()))
            }
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("url") => {
                data_url(o.get("url").and_then(Value::as_str).unwrap_or(""))
            }
            _ => verr("image source must be {\"type\": \"base64\", \"data\": ...}"),
        };
    }
    if let Some(Value::String(u)) = b.get("url") {
        return data_url(u);
    }
    match b.get("data") {
        Some(Value::String(d)) => Ok(d),
        _ => verr("image block does not contain a valid source"),
    }
}

/// The image's width and height, from its header.
fn dims_b64(b64: &str) -> Result<(usize, usize), VisionError> {
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
    Ok((w as usize, h as usize))
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
pub fn expand(ids: &[u32], n_spans: &[usize], token: u32) -> Result<(Vec<u32>, Vec<usize>), VisionError> {
    let found = ids.iter().filter(|&&t| t == token).count();
    if found != n_spans.len() {
        return verr(format!("the prompt has {found} image tokens but the request has {} images", n_spans.len()));
    }
    let extra: usize = n_spans.iter().map(|n| n - 1).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut pos = Vec::with_capacity(n_spans.len());
    let mut it = n_spans.iter();
    for &t in ids {
        if t == token {
            let n = *it.next().unwrap();
            pos.push(out.len());
            out.resize(out.len() + n, token);
        } else {
            out.push(t);
        }
    }
    Ok((out, pos))
}

/// A token count in the unexpanded prompt -> the same point in the expanded one (spans that start before it count in
/// full).
pub fn expanded_index(unexpanded_ids: &[u32], n: usize, n_spans: &[usize], token: u32) -> usize {
    let mut k = 0;
    let mut out = 0;
    for &t in unexpanded_ids.iter().take(n) {
        if t == token {
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
                serde_json::from_str(&std::fs::read_to_string(fixtures().join(format!("{name}.json"))).unwrap())
                    .unwrap();
            let (w, h, want) = read_ppm(&fixtures().join(format!("{name}.ppm")));
            let p = prepare_rgb(&decode_rgb(&png).unwrap()).unwrap();
            assert_eq!((p.vit_w * PATCH, p.vit_h * PATCH), (w, h), "{name}");
            assert_eq!(p.vit_h as u64, meta["n_vit_h"].as_u64().unwrap());
            assert_eq!(p.vit_w as u64, meta["n_vit_w"].as_u64().unwrap());
            assert_eq!(p.n_span as u64, meta["n_span_tokens"].as_u64().unwrap());
            assert!(p.rgb == want, "{name}: padded RGB differs from the official bytes");
        }
    }

    /// Random images against PIL (frontend/tests/vision_pil_cases.py wrote them): sizes that upscale and downscale,
    /// both aspects, random content, the PNG colour types, GIF and lossless WebP. GR8_TEST_VISION_CASES=DIR runs a set made
    /// with `--n`; its lossy cases (JPEG, lossy WebP) only report how far they are off.
    #[test]
    fn pad_matches_pil_on_random_images() {
        let dir = match gr8knobs::TEST_VISION_CASES.get() {
            Some(d) => PathBuf::from(d),
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("src/vision_cases"),
        };
        let idx: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
        let cases = idx.as_array().unwrap();
        assert!(cases.len() >= 10);
        for c in cases {
            let name = c["file"].as_str().unwrap();
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let p = prepare_rgb(&decode_rgb(&bytes).unwrap()).unwrap();
            assert_eq!(
                (p.vit_w * PATCH, p.vit_h * PATCH),
                (c["best_w"].as_u64().unwrap() as usize, c["best_h"].as_u64().unwrap() as usize),
                "{name}"
            );
            let same = content_key(p.vit_h, p.vit_w, &p.rgb) == c["key"].as_str().unwrap();
            if let Ok(want) = std::fs::read(dir.join(format!("{name}.rgb"))) {
                let d: Vec<u8> = p.rgb.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).collect();
                let n = d.iter().filter(|&&x| x != 0).count();
                eprintln!(
                    "{name}: {}x{} exact={same} differing bytes {n}/{} max |diff| {}",
                    p.vit_w * PATCH,
                    p.vit_h * PATCH,
                    d.len(),
                    d.iter().max().unwrap_or(&0)
                );
            }
            if c["exact"].as_bool().unwrap_or(true) {
                assert!(same, "{name}: padded RGB differs from PIL's");
            }
        }
    }

    fn glm_fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/ref/flash/fixtures/vision")
    }

    /// The processor's grid for 24 sizes (engine/ref/flash/vision_golden.py: transformers' smart_resize and resize).
    #[test]
    fn glm5v_plan_matches_the_processor() {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(glm_fixtures().join("plan.json")).unwrap()).unwrap();
        let cases = v.as_array().unwrap();
        assert!(cases.len() >= 20);
        for c in cases {
            let g = |k: &str| c[k].as_u64().unwrap() as usize;
            let p = glm5v::plan(g("w"), g("h"));
            assert_eq!(
                (p.target_h, p.target_w, p.content_h, p.content_w, p.tokens()),
                (g("target_h"), g("target_w"), g("content_h"), g("content_w"), g("tokens")),
                "{c}"
            );
        }
    }

    /// The padded RGB of the tower's fixtures (no resize: the pad alone) and of 6 images that the processor resizes
    /// (below the token floor, above the cap), against the PIL backend's bytes.
    #[test]
    fn glm5v_pixels_match_the_processor() {
        use sha2::{Digest, Sha256};
        let hex = |b: &[u8]| Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect::<String>();
        for name in ["red_disc_150x150", "bar_chart_280x200", "photo_scene_250x180"] {
            let png = std::fs::read(glm_fixtures().join(format!("{name}.png"))).unwrap();
            let meta: Value =
                serde_json::from_str(&std::fs::read_to_string(glm_fixtures().join(format!("{name}.json"))).unwrap())
                    .unwrap();
            let p = Model::Glm5v.prepare_rgb(&decode_rgb(&png).unwrap()).unwrap();
            let (w, h, want) = read_ppm(&glm_fixtures().join(format!("{name}.ppm")));
            assert_eq!((p.vit_w * PATCH, p.vit_h * PATCH), (w, h), "{name}");
            assert_eq!(p.n_span as u64, meta["tokens"].as_u64().unwrap(), "{name}");
            assert!(p.rgb == want, "{name}: padded RGB differs from the processor's");
            assert_eq!(hex(&p.rgb), meta["rgb_sha256"].as_str().unwrap());
        }
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(glm_fixtures().join("pixels.json")).unwrap()).unwrap();
        for c in v.as_array().unwrap() {
            let name = c["image"].as_str().unwrap();
            let png = std::fs::read(glm_fixtures().join(name)).unwrap();
            let p = Model::Glm5v.prepare_rgb(&decode_rgb(&png).unwrap()).unwrap();
            assert_eq!(
                (p.vit_h * PATCH, p.vit_w * PATCH),
                (c["target_h"].as_u64().unwrap() as usize, c["target_w"].as_u64().unwrap() as usize),
                "{name}"
            );
            assert_eq!(hex(&p.rgb), c["rgb_sha256"].as_str().unwrap(), "{name}: padded RGB differs from PIL's");
        }
    }

    fn png(w: u32, h: u32, color: image::ExtendedColorType, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut out),
            data,
            w,
            h,
            color,
            image::ImageFormat::Png,
        )
        .unwrap();
        out
    }

    #[test]
    fn ink2_plan_is_the_processor_grid() {
        let p = ink2::plan(1, 1).unwrap();
        assert_eq!((p.rows, p.cols, p.vit_h, p.vit_w, p.pad_h, p.pad_w, p.tokens), (1, 1, 2, 2, 40, 40, 1));
        let p = ink2::plan(120, 80).unwrap(); // width 120, height 80
        assert_eq!((p.rows, p.cols, p.vit_h, p.vit_w, p.pad_h, p.pad_w, p.tokens), (2, 3, 4, 6, 80, 120, 6));
        let p = ink2::plan(41, 40).unwrap();
        assert_eq!((p.rows, p.cols, p.pad_w, p.tokens), (1, 2, 80, 2));
        assert_eq!(ink2::plan(3840, 2160).unwrap().tokens, 5184);
        assert_eq!(ink2::plan(3200, 3200).unwrap().tokens, ink2::MAX_TOKENS);
        assert_eq!(ink2::plan(40 * 6400, 40).unwrap().tokens, ink2::MAX_TOKENS);
        assert!(ink2::plan(3201, 3200).is_err());
        assert!(ink2::plan(3200, 3201).is_err());
        assert!(ink2::plan(40 * 6400 + 1, 1).is_err());
        assert!(ink2::plan(0, 5).is_err());
        assert!(ink2::plan(5, 0).is_err());
        // the padded bytes of the largest image fit the request's base64 budget
        assert!(
            (ink2::MAX_TOKENS * ink2::FACTOR * ink2::FACTOR * 3).div_ceil(3) * 4 <= MAX_IMAGE_B64_BYTES_PER_REQUEST
        );
        assert_eq!((ink2::PATCH, ink2::MERGE, ink2::FACTOR, ink2::MIN_TOKENS), (20, 2, 40, 1));
        assert_eq!(Model::Ink2.token(), 200_054);
    }

    #[test]
    fn ink2_checkerboard_compositing_is_exact() {
        // the board: 8 px squares, 204 where the square row and column sum is even
        for (y, x, want) in
            [(0, 0, 204), (7, 7, 204), (0, 8, 153), (8, 0, 153), (8, 8, 204), (15, 16, 153), (16, 16, 204)]
        {
            assert_eq!(ink2::checker(y, x), want, "({y}, {x})");
        }
        // rounding to nearest in integers, every channel value at a few alphas against the formula in f64
        for a in [0u8, 1, 127, 128, 254, 255] {
            for c in 0..=255u8 {
                for bg in [204u8, 153] {
                    let exact = (c as f64 * a as f64 + bg as f64 * (255.0 - a as f64)) / 255.0;
                    let got = ink2::over(c, a, bg);
                    assert!((got as f64 - exact).abs() <= 0.5 + 1e-9, "c {c} a {a} bg {bg}: {got} vs {exact}");
                }
            }
        }
        assert_eq!((ink2::over(10, 255, 204), ink2::over(10, 0, 204), ink2::over(10, 0, 153)), (10, 204, 153));
        assert_eq!(ink2::over(255, 128, 153), ((255 * 128 + 153 * 127 + 127) / 255) as u8);

        // an RGBA PNG, 20 x 12: clear on the left half, half red in the middle, opaque blue on the right
        let (w, h) = (20usize, 12usize);
        let mut rgba = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(match x {
                    0..=9 => &[90, 60, 30, 0],
                    10..=14 => &[255, 0, 0, 128],
                    _ => &[0, 0, 255, 255],
                });
            }
        }
        let im = ink2::decode_ink2_rgb(&png(w as u32, h as u32, image::ExtendedColorType::Rgba8, &rgba)).unwrap();
        assert_eq!((im.w, im.h), (w, h));
        for y in 0..h {
            for x in 0..w {
                let bg = if (y / 8 + x / 8) % 2 == 0 { 204u32 } else { 153 };
                let want: [u8; 3] = match x {
                    0..=9 => [bg as u8; 3],
                    10..=14 => [
                        ((255 * 128 + bg * 127 + 127) / 255) as u8,
                        ((bg * 127 + 127) / 255) as u8,
                        ((bg * 127 + 127) / 255) as u8,
                    ],
                    _ => [0, 0, 255],
                };
                assert_eq!(im.px[(y * w + x) * 3..(y * w + x) * 3 + 3], want, "({y}, {x})");
            }
        }
        // the other models drop the alpha instead
        let dropped = decode_rgb(&png(w as u32, h as u32, image::ExtendedColorType::Rgba8, &rgba)).unwrap();
        assert_eq!(dropped.px[..3], [90, 60, 30]);
        assert_eq!(
            Model::Glm5v.decode(&png(w as u32, h as u32, image::ExtendedColorType::Rgba8, &rgba)).unwrap().px,
            dropped.px
        );

        // grey + alpha, and 16-bit RGBA (the high byte), composite the same way
        let la: Vec<u8> = (0..w * h).flat_map(|i| [200u8, if i % 2 == 0 { 0 } else { 255 }]).collect();
        let im = ink2::decode_ink2_rgb(&png(w as u32, h as u32, image::ExtendedColorType::La8, &la)).unwrap();
        assert_eq!(im.px[..6], [204, 204, 204, 200, 200, 200]);
        assert_eq!(im.px[(9 * w) * 3..(9 * w) * 3 + 3], [153, 153, 153]);
        let mut wide = Vec::new();
        for _ in 0..w * h {
            for v in [0x4000u16, 0x80ff, 0xc000, 0x8000] {
                wide.extend_from_slice(&v.to_ne_bytes()); // the encoder takes native-endian samples
            }
        }
        let im = ink2::decode_ink2_rgb(&png(w as u32, h as u32, image::ExtendedColorType::Rgba16, &wide)).unwrap();
        assert_eq!(im.px[..3], [ink2::over(0x40, 0x80, 204), ink2::over(0x80, 0x80, 204), ink2::over(0xc0, 0x80, 204)]);
    }

    #[test]
    fn ink2_opaque_images_decode_as_rgb() {
        let (w, h) = (9usize, 5usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let bytes = png(w as u32, h as u32, image::ExtendedColorType::Rgb8, &rgb);
        let a = ink2::decode_ink2_rgb(&bytes).unwrap();
        assert_eq!((a.w, a.h, &a.px), (w, h, &rgb));
        assert_eq!(a.px, decode_rgb(&bytes).unwrap().px);
        let grey: Vec<u8> = (0..w * h).map(|i| (i * 5) as u8).collect();
        let bytes = png(w as u32, h as u32, image::ExtendedColorType::L8, &grey);
        let a = ink2::decode_ink2_rgb(&bytes).unwrap();
        assert_eq!(a.px, decode_rgb(&bytes).unwrap().px);
        assert_eq!(a.px[3..6], [5, 5, 5]);
        // an RGBA PNG whose alpha is 255 everywhere is its RGB
        let rgba: Vec<u8> = rgb.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
        let a = ink2::decode_ink2_rgb(&png(w as u32, h as u32, image::ExtendedColorType::Rgba8, &rgba)).unwrap();
        assert_eq!(a.px, rgb);
        assert!(ink2::decode_ink2_rgb(b"not an image").is_err());
    }

    #[test]
    fn ink2_gif_transparency_shows_the_checkerboard() {
        // 2 colours, index 1 transparent, a 12 x 10 screen with a 4 x 4 frame of index 0 at (2, 1)
        let (sw, sh) = (12u16, 10u16);
        let mut bytes = Vec::new();
        {
            let mut e = gif::Encoder::new(&mut bytes, sw, sh, &[10, 20, 30, 250, 250, 250]).unwrap();
            let mut f = gif::Frame::default();
            (f.left, f.top, f.width, f.height) = (2, 1, 4, 4);
            f.buffer = std::borrow::Cow::Owned(vec![0u8; 16]);
            f.transparent = Some(1);
            e.write_frame(&f).unwrap();
        }
        let im = ink2::decode_ink2_rgb(&bytes).unwrap();
        assert_eq!((im.w, im.h), (12, 10));
        for y in 0..10usize {
            for x in 0..12usize {
                let inside = (2..6).contains(&x) && (1..5).contains(&y);
                let want = if inside { [10, 20, 30] } else { [ink2::checker(y, x); 3] };
                assert_eq!(im.px[(y * 12 + x) * 3..(y * 12 + x) * 3 + 3], want, "({y}, {x})");
            }
        }
        // the other models show the transparent index's palette colour
        assert_eq!(decode_rgb(&bytes).unwrap().px[..3], [250, 250, 250]);
    }

    #[test]
    fn ink2_pads_right_and_bottom_with_zeros() {
        let (w, h) = (50usize, 41usize);
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i % 200 + 1) as u8).collect();
        let p = Model::Ink2.prepare_rgb(&Rgb { w, h, px: px.clone() }).unwrap();
        assert_eq!((p.vit_h, p.vit_w, p.n_span), (4, 4, 4));
        assert_eq!(p.rgb.len(), 80 * 80 * 3);
        for y in 0..80 {
            for x in 0..80 {
                let got = &p.rgb[(y * 80 + x) * 3..(y * 80 + x) * 3 + 3];
                if y < h && x < w {
                    assert_eq!(got, &px[(y * w + x) * 3..(y * w + x) * 3 + 3], "({y}, {x})");
                } else {
                    assert_eq!(got, [0, 0, 0], "({y}, {x})");
                }
            }
        }
        assert_eq!(p.key, content_key(4, 4, &p.rgb));
        // an aligned image is passed through
        let px: Vec<u8> = (0..40 * 80 * 3).map(|i| (i % 251) as u8).collect();
        let p = ink2::prepare_rgb(&Rgb { w: 80, h: 40, px: px.clone() }).unwrap();
        assert_eq!((p.vit_h, p.vit_w, p.n_span, &p.rgb), (2, 4, 2, &px));
        // through base64: the span from the header alone is the prepared span
        let bytes = png(50, 41, image::ExtendedColorType::Rgb8, &vec![7u8; 50 * 41 * 3]);
        let b64 = b64_encode(&bytes);
        assert_eq!(Model::Ink2.n_span_b64(&b64).unwrap(), 4);
        let p = Model::Ink2.prepare_b64(&b64).unwrap();
        assert_eq!((p.vit_h, p.vit_w, p.n_span, p.rgb[0], p.rgb[50 * 3]), (4, 4, 4, 7, 0));
        assert!(Model::Ink2.prepare_b64("aGk=").is_err());
    }

    #[test]
    fn ink2_expand_keeps_the_content_marker() {
        // <|content_image|> (200005) is text; only <|unused_200054|> grows to the span
        let t = ink2::IMAGE_TOKEN_ID;
        let (ids, pos) = expand(&[7, 200_005, t, 9, 200_005, t, 8], &[6, 1], t).unwrap();
        assert_eq!(ids, vec![7, 200_005, t, t, t, t, t, t, 9, 200_005, t, 8]);
        assert_eq!(pos, vec![2, 10]);
        assert_eq!(ids.iter().filter(|&&i| i == 200_005).count(), 2);
        assert_eq!(expanded_index(&[7, 200_005, t, 9, 200_005, t, 8], 4, &[6, 1], t), 9);
        assert!(expand(&[7, 200_005, 9], &[6], t).is_err());
        assert!(expand(&[1, t], &[3], glm5v::IMAGE_TOKEN_ID).is_err());
    }

    #[test]
    fn expand_and_positions() {
        for t in [IMAGE_TOKEN_ID, glm5v::IMAGE_TOKEN_ID] {
            let (ids, pos) = expand(&[1, t, 2, t], &[3, 2], t).unwrap();
            assert_eq!(ids, vec![1, t, t, t, 2, t, t]);
            assert_eq!(pos, vec![1, 5]);
            assert!(expand(&[1, t], &[], t).is_err());
            assert!(expand(&[1], &[3], t).is_err());
            assert_eq!(expanded_index(&[1, t, 2, t], 2, &[3, 2], t), 4);
            assert_eq!(expanded_index(&[1, t, 2, t], 1, &[3, 2], t), 1);
        }
        // the other model's image token is text
        assert!(expand(&[1, IMAGE_TOKEN_ID], &[3], glm5v::IMAGE_TOKEN_ID).is_err());
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
        assert!(Model::Ds41.prepare_b64("aGk=").is_err());
        assert!(Model::Glm5v.prepare_b64("aGk=").is_err());
        assert_eq!(content_key(1, 2, b"x").len(), 16);
    }

    #[test]
    fn image_sources() {
        use serde_json::json;
        let ok = [
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}),
            json!({"type": "input_image", "image_url": "data:image/png;base64,QUJD"}),
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}),
            json!({"type": "image", "url": "data:image/jpeg;base64,QUJD"}),
            json!({"type": "image", "data": "QUJD"}),
        ];
        for b in ok {
            assert_eq!(block_b64(&b).unwrap(), "QUJD", "{b}");
        }
        for b in [
            json!({"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}),
            json!({"type": "image", "source": {"type": "url", "url": "file:///etc/passwd"}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png,raw"}}),
            json!({"type": "image"}),
        ] {
            assert!(block_b64(&b).is_err(), "{b}");
        }
    }
}
