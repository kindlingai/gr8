#!/usr/bin/env python3
"""Golden data for the DS41 vision tower + aligner, from DeepSeek's own reference code, on the CPU.

    python3 -I engine/ref/ds41/vision_golden.py --official DIR --weights FILE --out engine/ref/ds41/fixtures/vision

DIR is the official release's `inference/` directory (vision.py, image_processor.py); FILE is the release's
`dense/exlr8-extra-vision.safetensors`. The script writes three synthetic PNG images of different sizes and shapes,
then for each one:

  <name>.png            the image (lossless, so a decoder gives the same pixels everywhere)
  <name>.ppm            the resized + padded RGB the tower reads (what the front end sends the engine), binary PPM
  <name>.json           grid sizes, the sha256 of the bf16 patch tensor, error statistics
  <name>.ref.bin        float32 rows of the aligner output that the test compares in full (rows listed in the json)
  <name>.proj.bin       float32 [n_tokens, NPROJ]: every aligner row projected on NPROJ fixed directions

plus plan.json: plan_image_grid for many (width, height) pairs, the token cap cases among them.

The full output is n_tokens x 5120 floats; the sampled rows and the projections keep the files small and still catch
any error that moves an output row (an error has to be orthogonal to all NPROJ directions to hide).

Two runs of the official code: in float32 (bf16 weights and patches, upcast) -- the golden data -- and as the
official generate.py runs it (bf16 parameters and activations). The json records the bf16 run's relative error
against float32, which is the scale a serving kernel's bf16 rounding is judged on.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

NPROJ = 64


def proj_matrix(dim, nproj):
    """[dim, nproj] float32 of +-1/sqrt(dim) from xorshift32 (seed 0x9E3779B9), row-major, one bit per entry:
    the C++ test regenerates it the same way."""
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
    """Three synthetic images: small landscape (upscaled to min_pixels), wide, and large (capped at max tokens)."""
    from PIL import Image, ImageDraw

    # each at or just above min_pixels (544 x 544), so the raw pixel files stay under 1 MB; the token cap is covered by
    # the plan table (plan_cases), which needs no pixels
    specs = [("square_300x300", 300, 300), ("wide_1100x280", 1100, 280), ("tall_320x1000", 320, 1000)]
    paths = []
    for k, (name, w, h) in enumerate(specs):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = (127 + 127 * np.sin(xx / (17 + 5 * k))).astype(np.uint8)
        g = (255 * yy / max(h - 1, 1)).astype(np.uint8)
        b = ((xx + yy) * (3 + k) % 256).astype(np.uint8)
        im = Image.fromarray(np.stack([r, g, b], -1), "RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([w // 8, h // 6, w // 3, h // 2], fill=(250, 30, 30))
        d.ellipse([w // 2, h // 3, w // 2 + h // 3, 2 * h // 3], fill=(20, 200, 40))
        d.line([0, h - 1, w - 1, 0], fill=(255, 255, 255), width=max(2, h // 60))
        p = os.path.join(out, name + ".png")
        im.save(p, optimize=True)
        paths.append((name, p))
    return paths


def padded_rgb(path, args, ip):
    """The uint8 image load_image builds before it normalizes (same calls, same order)."""
    import io
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(ip.load_image_bytes({"url": path}))) as source:
        image = source.convert("RGB")
    _, _, best_height, best_width = ip.plan_image_grid(image.width, image.height, args)
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    return np.asarray(image, dtype=np.uint8)


def plan_cases(args, ip):
    sizes = [(1, 1), (14, 14), (300, 300), (544, 544), (545, 543), (1100, 280), (320, 1000), (640, 480), (1024, 701),
             (1280, 720), (1920, 1080), (1600, 1200), (4000, 3000), (8000, 100), (100, 8000), (30000, 20), (20, 30000),
             (3, 5000), (999, 1001), (4096, 4096), (1303, 1302), (2000, 50)]
    out = []
    for w, h in sizes:
        n_llm_h, n_llm_w, bh, bw = ip.plan_image_grid(w, h, args)
        out.append({"w": w, "h": h, "best_w": bw, "best_h": bh, "n_llm_h": n_llm_h, "n_llm_w": n_llm_w,
                    "n_span_tokens": ip.num_image_tokens(n_llm_h, n_llm_w)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.official))
    import torch
    from safetensors.torch import load_file

    import image_processor as ip
    from vision import Aligner, ViT

    torch.set_num_threads(a.threads)
    cfg = json.load(open(a.config))
    vc = cfg["vision_config"]
    args = SimpleNamespace(
        dim=cfg["text_config"]["hidden_size"], vision_n_layers=vc["num_hidden_layers"], vision_dim=vc["hidden_size"],
        vision_n_heads=vc["num_attention_heads"], vision_inter_dim=vc["intermediate_size"],
        vision_patch_size=vc["patch_size"], vision_rope_theta=vc["rope_theta"],
        vision_downsample_ratio=vc["downsample_ratio"], vision_max_n_token=vc["max_image_tokens"],
        vision_min_pixels=vc["min_pixels"], vision_max_wh_ratio=vc.get("max_wh_ratio"))

    w = load_file(a.weights)
    models = {}
    for dt in (torch.float32, torch.bfloat16):
        torch.set_default_dtype(dt)
        vit, al = ViT(args), Aligner(args)
        for mod, pre in ((vit, "vision."), (al, "aligner.")):
            sd = {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}
            mod.load_state_dict(sd, strict=True)
            mod.to(dt)
            # the official RMSNorm keeps its weight in float32 whatever the default dtype
            for m in mod.modules():
                if type(m).__name__ == "RMSNorm":
                    m.weight.data = m.weight.data.float()
        models[dt] = (vit.eval(), al.eval())
    torch.set_default_dtype(torch.float32)

    os.makedirs(a.out, exist_ok=True)
    R = proj_matrix(args.dim, NPROJ)
    summary = []
    for name, path in make_images(a.out):
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"url": path}, args)
        pad = padded_rgb(path, args, ip)
        assert pad.shape == (n_vit_h * args.vision_patch_size, n_vit_w * args.vision_patch_size, 3), pad.shape
        with open(os.path.join(a.out, name + ".ppm"), "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (pad.shape[1], pad.shape[0]))
            f.write(pad.tobytes())
        # the pixels -> patches step of load_image, redone from the PPM: the engine's patchify reads exactly these bytes
        x = torch.from_numpy(pad.astype(np.float32)).permute(2, 0, 1) / 255
        x = ((x - 0.5) / 0.5).to(torch.bfloat16)
        p_ = args.vision_patch_size
        x = x.reshape(3, n_vit_h, p_, n_vit_w, p_).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p_, p_)
        assert torch.equal(x.view(torch.int16), patches.view(torch.int16))
        assert patches.dtype == torch.bfloat16
        outs = {}
        with torch.inference_mode():
            for dt, (vit, al) in models.items():
                outs[dt] = al(vit(patches.to(dt), n_vit_h, n_vit_w), n_vit_h, n_vit_w).float()
        y, yb = outs[torch.float32].numpy(), outs[torch.bfloat16].numpy()
        assert y.shape == (n_llm_h * n_llm_w, args.dim), y.shape
        row_rel = np.linalg.norm(yb - y, axis=1) / np.linalg.norm(y, axis=1)
        n = y.shape[0]
        rows = sorted({0, 1, n_llm_w - 1, n_llm_w, n // 2, n - n_llm_w, n - 2, n - 1})
        y[rows].astype(np.float32).tofile(os.path.join(a.out, name + ".ref.bin"))
        (y @ R).astype(np.float32).tofile(os.path.join(a.out, name + ".proj.bin"))
        types = ip.image_token_types(n_llm_h, n_llm_w)
        meta = {
            "image": name + ".png", "n_vit_h": n_vit_h, "n_vit_w": n_vit_w, "n_llm_h": n_llm_h, "n_llm_w": n_llm_w,
            "n_span_tokens": int(types.numel()), "dim": args.dim, "nproj": NPROJ, "rows": rows,
            "patches_sha256": hashlib.sha256(patches.contiguous().view(torch.int16).numpy().tobytes()).hexdigest(),
            "patches_shape": list(patches.shape),
            "bf16_vs_f32": {"rel_fro": float(np.linalg.norm(yb - y) / np.linalg.norm(y)),
                            "row_rel_max": float(row_rel.max()), "row_rel_mean": float(row_rel.mean())},
            "rms": float(np.sqrt((y.astype(np.float64) ** 2).mean())),
        }
        json.dump(meta, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        summary.append(meta)
        print(name, n_vit_h, n_vit_w, n_llm_h, n_llm_w, meta["bf16_vs_f32"], flush=True)
    json.dump(plan_cases(args, ip), open(os.path.join(a.out, "plan.json"), "w"), indent=0)
    print(json.dumps({"images": [m["image"] for m in summary]}))


if __name__ == "__main__":
    main()
