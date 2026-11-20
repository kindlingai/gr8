#!/usr/bin/env python3
"""Random images and PIL's padded RGB for them, the reference the Rust image preprocessing (g6front-rs src/vision.rs)
is checked against: the same calls as the release's inference/image_processor.py load_image (Image.open, convert("RGB"),
plan_image_grid, ImageOps.pad with grey 127).

    python3 frontend/tests/vision_pil_cases.py --small --out frontend/g6front-rs/src/vision_cases
    python3 frontend/tests/vision_pil_cases.py --n 30 --out DIR      # a larger random set, not committed
    GR8_TEST_VISION_CASES=DIR cargo test --release vision::tests::pad_matches_pil

--small writes the committed set (under 200 KB): every PNG colour type PIL and the decoder read differently, a
transparent GIF, and sizes that upscale (min_pixels), downscale (the token cap) and pad either way. Each case is listed
in index.json with its plan and the content key (sha256 over vit_h, vit_w as u32 LE and the padded bytes); a random set
also writes the padded bytes (<file>.rgb) so the test can report how far a lossy decode is off.
"""
import argparse
import hashlib
import io
import json
import math
import os
import random
import struct
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ARGS = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=1024,
                       vision_min_pixels=295936, vision_max_wh_ratio=None)


# plan_image_grid and its helpers, verbatim from the release's inference/image_processor.py
def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio)


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width, height, args):
    p = args.vision_patch_size
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)


def padded(data: bytes):
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    _, _, bh, bw = plan_image_grid(image.width, image.height, ARGS)
    out = np.asarray(ImageOps.pad(image, (bw, bh), color=(127, 127, 127)), dtype=np.uint8)
    p = ARGS.vision_patch_size
    key = hashlib.sha256(struct.pack('<II', bh // p, bw // p) + out.tobytes()).hexdigest()[:16]
    return bw, bh, key, out


def content(rng, w, h, block):
    """Low-frequency random colour blocks plus a few shapes and lines (compresses well at large block sizes)."""
    gw, gh = max(1, -(-w // block)), max(1, -(-h // block))
    a = rng.integers(0, 256, size=(gh, gw, 3), dtype=np.uint8)
    a = np.repeat(np.repeat(a, block, 0), block, 1)[:h, :w]
    im = Image.fromarray(np.ascontiguousarray(a), "RGB")
    d = ImageDraw.Draw(im)
    for _ in range(4):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(x0, w + 1)), int(rng.integers(y0, h + 1))
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        d.rectangle([x0, y0, x1, y1], outline=c, width=1)
        d.line([0, int(rng.integers(0, h)), w - 1, int(rng.integers(0, h))], fill=c, width=1)
    return im


def encode(im, fmt, **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


def small_cases(rng):
    """(name, bytes) of the committed set."""
    out = []
    # upscale (area < min_pixels), both aspects, odd sizes
    for w, h in [(37, 23), (23, 61), (120, 200), (301, 157), (500, 31)]:
        out.append((f"up_{w}x{h}.png", encode(content(rng, w, h, 3), "PNG", optimize=True)))
    # pad only (no resample) and tiny downscale-by-aspect
    out.append(("pad_700x500.png", encode(content(rng, 700, 500, 50), "PNG", optimize=True)))
    # the token cap: real downscales
    for w, h in [(1600, 1203), (900, 2500), (3000, 200)]:
        out.append((f"down_{w}x{h}.png", encode(content(rng, w, h, 100), "PNG", optimize=True)))
    # colour types
    base = content(rng, 83, 47, 2)
    out.append(("mode_L.png", encode(base.convert("L"), "PNG")))
    out.append(("mode_LA.png", encode(base.convert("LA"), "PNG")))
    rgba = base.convert("RGBA")
    rgba.putalpha(Image.fromarray(rng.integers(0, 256, (47, 83), dtype=np.uint8), "L"))
    out.append(("mode_RGBA.png", encode(rgba, "PNG")))
    pal = base.convert("P", palette=Image.Palette.ADAPTIVE, colors=37)
    out.append(("mode_P.png", encode(pal, "PNG")))
    out.append(("mode_P_trns.png", encode(pal, "PNG", transparency=5)))
    out.append(("mode_1.png", encode(base.convert("1"), "PNG")))
    g16 = Image.fromarray(rng.integers(0, 600, (47, 83), dtype=np.uint16))
    out.append(("mode_I16.png", encode(g16, "PNG")))
    out.append(("mode_P.gif", encode(pal, "GIF")))
    out.append(("mode_P_trns.gif", encode(pal, "GIF", transparency=3)))
    out.append(("lossless.webp", encode(base, "WEBP", lossless=True)))
    return out


def random_cases(rng, n):
    out = []
    for i in range(n):
        if i % 3 == 0:  # upscale
            w, h = int(rng.integers(1, 540)), int(rng.integers(1, 540))
        elif i % 3 == 1:  # around min_pixels / plain pad
            w, h = int(rng.integers(300, 1300)), int(rng.integers(300, 1300))
        else:  # the token cap, either aspect
            w, h = int(rng.integers(200, 5000)), int(rng.integers(200, 5000))
        im = content(rng, w, h, int(rng.choice([1, 2, 7, 40])))
        fmt = ["PNG", "PNG", "PNG", "JPEG", "WEBP", "WEBPL", "GIF"][i % 7]
        if fmt == "JPEG":
            out.append((f"r{i}_{w}x{h}.jpg", encode(im, "JPEG", quality=int(rng.integers(50, 96))), False))
        elif fmt == "WEBP":
            out.append((f"r{i}_{w}x{h}.webp", encode(im, "WEBP", quality=80), False))
        elif fmt == "WEBPL":
            out.append((f"r{i}_{w}x{h}_l.webp", encode(im, "WEBP", lossless=True), True))
        elif fmt == "GIF":
            out.append((f"r{i}_{w}x{h}.gif", encode(im.convert("P", palette=Image.Palette.ADAPTIVE), "GIF"), True))
        else:
            out.append((f"r{i}_{w}x{h}.png", encode(im, "PNG"), True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--small", action="store_true", help="the committed set")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    cases = [(n, b, True) for n, b in small_cases(rng)] if a.small else random_cases(rng, a.n)
    index = []
    for name, data, exact in cases:
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
        bw, bh, key, px = padded(data)
        e = {"file": name, "best_w": bw, "best_h": bh, "key": key, "exact": exact}
        if not a.small:
            with open(os.path.join(a.out, name + ".rgb"), "wb") as f:
                f.write(px.tobytes())
        index.append(e)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=0)
    print(len(index), "cases,", sum(len(d) for _, d, _ in cases), "bytes of images")


if __name__ == "__main__":
    main()
