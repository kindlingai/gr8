#!/usr/bin/env python3
"""Golden data for the GLM-5.3-Flash vision tower (`glm5_next_vision`), from transformers' reference code, on the CPU.
    PYTHONPATH=<transformers >= 5.16> python3 -I engine/ref/flash/vision_golden.py --release DIR --out engine/ref/flash/fixtures/vision
DIR holds the release's config.json, processor_config.json and dense/vision-bf16.safetensors. The script writes three
synthetic PNG images (a red disc, a small bar chart, a non-square photo-like scene), then for each one:
  <name>.png            the image (lossless, so a decoder gives the same pixels everywhere)
  <name>.ppm            the padded RGB the tower reads (what the front end sends the engine), binary PPM
  <name>.json           the grid, sha256 of the padded RGB and of the bf16 patch tensor, error statistics
  <name>.ref.bin        float32 rows of the tower output that the test compares in full (rows listed in the json)
  <name>.proj.bin       float32 [n_tokens, NPROJ]: every output row projected on NPROJ fixed directions
  <name>.proj_bf16.bin  the same projections of the bf16 run
plus plan.json: the processor's grid (smart_resize + the content resize) for many image sizes, the token floor and
cap among them, and pixels.json: the sha256 of the padded RGB of images that need a resize, from the PIL backend.
The tower output is n_tokens x 4096 floats; the sampled rows and the projections keep the files small and still catch
any error that moves an output row (an error has to be orthogonal to all NPROJ directions to hide).
Two runs of the reference: in float32 (bf16 weights and bf16 patches, upcast) -- the golden data -- and in bf16 as a
serving stack runs it. The json records the bf16 run's relative error against float32, the scale a serving
kernel's bf16 rounding is judged on. The tower grows a few massive-activation patches (norms near 800 against 35)
in flat regions, and which of several near-identical patches takes that role depends on the rounding, so a few
output rows differ a lot between any two runs that round differently: the median row error is the stable measure.
"""
import argparse
import hashlib
import json
import math
import os

import numpy as np

NPROJ = 64


def proj_matrix(dim, nproj):
    """[dim, nproj] float32 of +-1/sqrt(dim) from xorshift32 (seed 0x9E3779B9), row-major, one bit per entry:
    the C++ test regenerates it the same way (engine/ref/ds41/vision_golden.py uses the same generator)."""
    s = 0x9E3779B9
    out = np.empty(dim * nproj, dtype=np.float32)
    v = 1.0 / np.sqrt(dim)
    for i in range(dim * nproj):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out[i] = v if (s & 1) else -v
    return out.reshape(dim, nproj)


def make_images(out):
    """Three small synthetic images, none a multiple of 28 px, so each one is padded."""
    from PIL import Image, ImageDraw
    paths = []
    # a red disc on white
    im = Image.new("RGB", (150, 150), (255, 255, 255))
    ImageDraw.Draw(im).ellipse([25, 25, 125, 125], fill=(220, 20, 20))
    paths.append(("red_disc_150x150", im))
    # a bar chart: four bars of heights 3, 7, 5, 9 with their labels and values
    w, h = 280, 200
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.line([30, 170, 270, 170], fill=(0, 0, 0), width=2)
    d.line([30, 10, 30, 170], fill=(0, 0, 0), width=2)
    for i, (lab, v, col) in enumerate([("A", 3, (40, 90, 200)), ("B", 7, (220, 120, 30)), ("C", 5, (40, 160, 60)),
                                       ("D", 9, (190, 40, 40))]):
        x0 = 45 + i * 58
        d.rectangle([x0, 170 - v * 16, x0 + 36, 169], fill=col)
        d.text((x0 + 14, 176), lab, fill=(0, 0, 0))
        d.text((x0 + 14, 170 - v * 16 - 12), str(v), fill=(0, 0, 0))
    paths.append(("bar_chart_280x200", im))
    # a photo-like scene: sky gradient, sun, hills, a house
    w, h = 250, 180
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.stack([90 + 80 * yy / h, 150 + 60 * yy / h, 235 - 20 * yy / h], -1)
    hill = (yy > 120 + 18 * np.sin(xx / 23.0))[..., None]
    grass = np.stack([60 + 30 * np.sin(xx / 7.0) * np.cos(yy / 5.0), 140 + 25 * np.cos(xx / 11.0), 50 + 0 * xx], -1)
    px = np.where(hill, grass, sky).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(px, "RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([190, 18, 226, 54], fill=(255, 220, 60))
    d.rectangle([60, 95, 120, 140], fill=(200, 180, 150))
    d.polygon([(52, 95), (90, 65), (128, 95)], fill=(150, 40, 40))
    d.rectangle([82, 115, 98, 140], fill=(90, 60, 30))
    paths.append(("photo_scene_250x180", im))
    out_paths = []
    for name, im in paths:
        p = os.path.join(out, name + ".png")
        im.save(p, optimize=True)
        out_paths.append((name, p))
    return out_paths


def plan(proc, w, h):
    """The processor's geometry for a w x h image: (target_h, target_w, content_h, content_w), its resize() steps."""
    from transformers.models.glm5_next.image_processing_glm5_next import smart_resize
    factor = proc.patch_size * proc.merge_size * proc.patch_expand_factor
    tf = proc.temporal_patch_size
    th, tw = smart_resize(num_frames=tf, height=h, width=w, temporal_factor=tf, factor=factor,
                          min_pixels=proc.min_image_tokens, max_pixels=proc.max_image_tokens)
    scale = min(th / h, tw / w)
    if tf * h * w >= tf * factor * factor * proc.min_image_tokens:
        scale = min(1.0, scale)
    ch = max(1, min(th, math.floor(h * scale)))
    cw = max(1, min(tw, math.floor(w * scale)))
    return th, tw, ch, cw


def padded_rgb(proc_pil, im):
    """The uint8 RGB the PIL backend builds before it rescales: the content resize (BICUBIC) and the zero pad."""
    from PIL import Image
    th, tw, ch, cw = plan(proc_pil, im.width, im.height)
    if (cw, ch) != (im.width, im.height):
        im = im.resize((cw, ch), resample=Image.BICUBIC)
    a = np.zeros((th, tw, 3), dtype=np.uint8)
    a[:ch, :cw] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return a


def patches_from_rgb(a, mean, std, patch=14, merge=2, tps=2):
    """pixel_values from padded uint8 RGB as the torchvision backend computes them: x = (u - fl(mean * 255)) /
    fl(std * 255) in float32, then the patch order of patchify (2 x 2 merge blocks row-major, the 4 patches inside
    row-major, each patch (c, t, py, px) with the frame repeated over t)."""
    m = (np.asarray(mean, np.float32) * np.float32(255.0)).astype(np.float32)
    s = (np.asarray(std, np.float32) * np.float32(255.0)).astype(np.float32)
    x = (a.astype(np.float32) - m) / s                      # [H][W][3]
    H, W, _ = a.shape
    gh, gw = H // patch, W // patch
    x = x.transpose(2, 0, 1).reshape(3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)                     # (bh, bw, mh, mw, c, py, px)
    x = np.broadcast_to(x[:, :, :, :, :, None], x.shape[:5] + (tps,) + x.shape[5:])
    return np.ascontiguousarray(x).reshape(gh * gw, 3 * tps * patch * patch), gh, gw


def plan_cases(proc):
    sizes = [(1, 1), (14, 14), (27, 29), (28, 28), (50, 30), (112, 112), (113, 111), (150, 150), (280, 200),
             (250, 180), (300, 300), (640, 480), (1024, 701), (1280, 720), (1920, 1080), (2500, 2500), (2504, 2504),
             (3000, 2000), (4000, 3000), (8000, 100), (100, 8000), (30000, 20), (20, 30000), (3, 5000)]
    out = []
    for w, h in sizes:
        th, tw, ch, cw = plan(proc, w, h)
        out.append({"w": w, "h": h, "target_h": th, "target_w": tw, "content_h": ch, "content_w": cw,
                    "vit_h": th // proc.patch_size, "vit_w": tw // proc.patch_size,
                    "tokens": (th // proc.patch_size) * (tw // proc.patch_size) // proc.merge_size ** 2})
    return out


def pixel_cases(proc_pil, proc_tv, out):
    """The padded RGB of images that need the content resize (below the token floor, above the cap), as hashes."""
    from PIL import Image, ImageDraw
    res = []
    for w, h in [(50, 30), (23, 61), (90, 40), (2700, 2600), (4000, 3000), (12000, 900)]:
        if w * h < 1_000_000:   # small: a texture in every pixel
            yy, xx = np.mgrid[0:h, 0:w]
            a = np.stack([(xx * 7 + yy * 3) % 256, (xx * 2 + yy * 11) % 256, ((xx ^ yy) * 5) % 256], -1).astype(np.uint8)
            im = Image.fromarray(a, "RGB")
        else:                   # large: flat colour, shapes and thin lines, so the PNG stays small
            im = Image.new("RGB", (w, h), (30, 60, 90))
            d = ImageDraw.Draw(im)
            for k in range(12):
                d.rectangle([w * k // 13, h * (k % 5) // 6, w * k // 13 + w // 20, h * (k % 5) // 6 + h // 7],
                            fill=(20 * k % 256, 255 - 17 * k % 256, 90 + 11 * k))
                d.line([0, h * k // 12, w - 1, h - 1 - h * k // 12], fill=(255, 255, 255), width=1 + k % 3)
        ImageDraw.Draw(im).ellipse([w // 5, h // 5, w // 2, h // 2], fill=(255, 0, 0))
        name = f"resize_{w}x{h}.png"
        im.save(os.path.join(out, name), optimize=True)
        pad = padded_rgb(proc_pil, im)
        th, tw, ch, cw = plan(proc_pil, w, h)
        # the torchvision backend resizes with its own bicubic: how far its pixels are from PIL's, in 8-bit levels
        pv = proc_tv(images=[im], return_tensors="pt")["pixel_values"].float().numpy()
        mine, _, _ = patches_from_rgb(pad, proc_tv.image_mean, proc_tv.image_std)
        lv = float((np.abs(pv - mine) * np.float32(255.0) * np.asarray(proc_tv.image_std, np.float32).min()).max())
        # the PIL backend's own pixel values: the same uint8 pixels as `pad` (a level off would be 0.0145 away)
        pp = proc_pil(images=[im], return_tensors="np")["pixel_values"].astype(np.float32)
        pil_diff = float(np.abs(pp - mine).max())
        assert pil_diff < 1e-5, (name, pil_diff)
        res.append({"image": name, "target_h": th, "target_w": tw, "content_h": ch, "content_w": cw,
                    "rgb_sha256": hashlib.sha256(pad.tobytes()).hexdigest(), "pil_backend_maxdiff": pil_diff,
                    "torchvision_max_levels_off": round(lv, 2),
                    "torchvision_values_differing": int((pv != mine).sum()), "values": int(pv.size)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--resize-cases", action="store_true", help="also write pixels.json and its resize_*.png inputs")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.release, "config.json")))
    vc = Glm5NextVisionConfig(**{k: v for k, v in cfg["vision_config"].items() if k != "model_type"})
    proc_tv = AutoImageProcessor.from_pretrained(a.release)                  # the default (torchvision) backend
    proc_pil = AutoImageProcessor.from_pretrained(a.release, backend="pil")
    w = load_file(os.path.join(a.release, "dense", "vision-bf16.safetensors"))
    sd = {k[len("model.visual."):]: v for k, v in w.items() if k.startswith("model.visual.")}
    assert len(sd) == len(w) == 347, (len(sd), len(w))
    models = {}
    for dt in (torch.float32, torch.bfloat16):
        vc._attn_implementation = "eager"
        m = Glm5NextVisionModel(vc)
        missing, unexpected = m.load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=False)
        assert not unexpected and all("inv_freq" in k for k in missing), (missing, unexpected)
        models[dt] = m.to(dt).eval()
    os.makedirs(a.out, exist_ok=True)
    dim = cfg["vision_config"]["out_hidden_size"]
    R = proj_matrix(dim, NPROJ)
    summary = []
    for name, path in make_images(a.out):
        im = Image.open(path).convert("RGB")
        pad = padded_rgb(proc_pil, im)
        with open(os.path.join(a.out, name + ".ppm"), "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (pad.shape[1], pad.shape[0]))
            f.write(pad.tobytes())
        bt = proc_tv(images=[im], return_tensors="pt")
        bp = proc_pil(images=[im], return_tensors="np")
        pv = bt["pixel_values"].float().numpy()
        grid = [int(x) for x in bt["image_grid_thw"][0]]
        assert [int(x) for x in bp["image_grid_thw"][0]] == grid
        mine, gh, gw = patches_from_rgb(pad, proc_tv.image_mean, proc_tv.image_std)
        assert [1, gh, gw] == grid, (grid, gh, gw)
        tv_exact = bool(np.array_equal(mine, pv))
        pil_maxdiff = float(np.abs(bp["pixel_values"].astype(np.float32) - pv).max())
        assert tv_exact, "the patch formula differs from the torchvision backend"
        x = torch.from_numpy(pv).to(torch.bfloat16)
        thw = torch.tensor([grid])
        outs = {}
        with torch.inference_mode():
            for dt, m in models.items():
                outs[dt] = m(x.to(dt), grid_thw=thw).pooler_output.float().numpy()
        y, yb = outs[torch.float32], outs[torch.bfloat16]
        n = y.shape[0]
        assert y.shape == (gh * gw // 4, dim), y.shape
        row_rel = np.linalg.norm(yb - y, axis=1) / np.linalg.norm(y, axis=1)
        mw = gw // 2
        rows = sorted({0, 1, mw - 1, mw, n // 2, n - mw, n - 2, n - 1})
        y[rows].astype(np.float32).tofile(os.path.join(a.out, name + ".ref.bin"))
        (y @ R).astype(np.float32).tofile(os.path.join(a.out, name + ".proj.bin"))
        (yb @ R).astype(np.float32).tofile(os.path.join(a.out, name + ".proj_bf16.bin"))
        meta = {
            "image": name + ".png", "ppm": name + ".ppm", "w": im.width, "h": im.height,
            "vit_h": gh, "vit_w": gw, "tokens": n, "dim": dim, "nproj": NPROJ, "rows": rows,
            "rgb_sha256": hashlib.sha256(pad.tobytes()).hexdigest(),
            "patches_bf16_sha256": hashlib.sha256(x.view(torch.int16).numpy().tobytes()).hexdigest(),
            "torchvision_patches_match_formula": tv_exact, "pil_backend_patch_maxdiff": pil_maxdiff,
            "bf16_vs_f32": {"rel_fro": float(np.linalg.norm(yb - y) / np.linalg.norm(y)),
                            "row_rel_max": float(row_rel.max()), "row_rel_mean": float(row_rel.mean()),
                            "row_rel_median": float(np.median(row_rel))},
            "rms": float(np.sqrt((y.astype(np.float64) ** 2).mean())),
        }
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        summary.append(meta)
        print(name, json.dumps(meta["bf16_vs_f32"]), "pil maxdiff", pil_maxdiff, flush=True)
    with open(os.path.join(a.out, "plan.json"), "w") as f:
        json.dump(plan_cases(proc_tv), f, indent=1)
    if a.resize_cases:
        with open(os.path.join(a.out, "pixels.json"), "w") as f:
            json.dump(pixel_cases(proc_pil, proc_tv, a.out), f, indent=1)


if __name__ == "__main__":
    main()
