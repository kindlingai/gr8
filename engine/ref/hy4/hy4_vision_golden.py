#!/usr/bin/env python3
"""Generate exact PyTorch golden embeddings for Hy4-flash vision tower (vit.* in vision-bf16.safetensors).

Runs the exact reference math from HF PR 50903 modeling_hy_v4_flash.py:
  - HYV4FlashVisionPatchEmbed (Conv2d 3->1152 k=16 s=16 + bilinear pos_embed scale=(size+0.1)/128 align_corners=False)
  - 27 x HYV4FlashVisionBlock (LayerNorm 1e-5, Q/K/V/O 16 heads x 72 dim, GELU MLP 1152->4304->1152)
  - HYV4FlashVisionPatchMerger (RMSNorm 1e-5, Conv2d 1152->4608 k=2 s=2, GELU, Conv2d 4608->3072 k=1, append image_newline)

Zero transformers install required (PyTorch + safetensors/struct only).
"""
import argparse
import json
import math
import struct
import numpy as np
import torch
import torch.nn.functional as F


def load_safetensors(path: str, device: str):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        data_off = 8 + hlen
        f.seek(data_off)
        raw = f.read()
    mv = memoryview(raw)
    tensors = {}
    for k, meta in hdr.items():
        if k == "__metadata__":
            continue
        o0, o1 = meta["data_offsets"]
        shape = meta["shape"]
        dt = meta["dtype"]
        if dt == "BF16":
            arr = np.frombuffer(mv[o0:o1], dtype=np.uint16).reshape(shape).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16).to(device)
        elif dt == "F32":
            arr = np.frombuffer(mv[o0:o1], dtype=np.float32).reshape(shape).copy()
            t = torch.from_numpy(arr).to(device)
        else:
            raise ValueError(f"Unsupported dtype {dt} for {k}")
        tensors[k] = t
    return tensors


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return weight * xf.to(dt)


@torch.no_grad()
def run_hy4_vision(w: dict, rgb: np.ndarray, vit_h: int, vit_w: int, device: str) -> torch.Tensor:
    torch.backends.cudnn.enabled = False
    # rgb: [vit_h*16, vit_w*16, 3] uint8
    H, W = vit_h * 16, vit_w * 16
    assert rgb.shape == (H, W, 3)
    img = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)  # [H, W, 3]
    img = (img / 255.0 - 0.5) / 0.5
    # [1, 3, H, W] -> patches [vit_h*vit_w, 3, 16, 16] row-major
    img_chw = img.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)  # [1, 3, H, W]
    x = F.conv2d(img_chw, w["vit.embeddings.patch_embedding.weight"], w["vit.embeddings.patch_embedding.bias"], stride=16)
    x = x.flatten(2).transpose(1, 2)  # [1, N, 1152]

    # Interpolate learned pos embedding: rows 1..16384 -> [1, 1152, 128, 128]
    pos = w["vit.embeddings.position_embedding.weight"][1:].reshape(1, 128, 128, 1152).permute(0, 3, 1, 2).float()
    pos = F.interpolate(
        pos,
        scale_factor=((vit_h + 0.1) / 128.0, (vit_w + 0.1) / 128.0),
        mode="bilinear",
        align_corners=False,
    )
    pos = pos.permute(0, 2, 3, 1).reshape(1, -1, 1152).to(torch.bfloat16)
    assert pos.shape == x.shape, f"pos {pos.shape} vs x {x.shape}"
    x = x + pos

    # 27 ViT layers
    N = vit_h * vit_w
    for i in range(27):
        p = f"vit.layers.{i}."
        h1 = F.layer_norm(x, (1152,), w[p + "input_layernorm.weight"], w[p + "input_layernorm.bias"], eps=1e-5)
        q = F.linear(h1, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"]).view(1, N, 16, 72).transpose(1, 2)
        k = F.linear(h1, w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"]).view(1, N, 16, 72).transpose(1, 2)
        v = F.linear(h1, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"]).view(1, N, 16, 72).transpose(1, 2)
        ao = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / math.sqrt(72))
        ao = ao.transpose(1, 2).reshape(1, N, 1152)
        x = x + F.linear(ao, w[p + "self_attn.o_proj.weight"], w[p + "self_attn.o_proj.bias"])

        h2 = F.layer_norm(x, (1152,), w[p + "post_attention_layernorm.weight"], w[p + "post_attention_layernorm.bias"], eps=1e-5)
        m = F.gelu(F.linear(h2, w[p + "mlp.dense_h_to_4h.weight"], w[p + "mlp.dense_h_to_4h.bias"]), approximate="none")
        x = x + F.linear(m, w[p + "mlp.dense_4h_to_h.weight"], w[p + "mlp.dense_4h_to_h.bias"])

    # Patch merger (vit.perceive)
    y = rms_norm(x, w["vit.perceive.norm.weight"], eps=1e-5)
    y = y.reshape(1, vit_h, vit_w, 1152).permute(0, 3, 1, 2)
    y = F.conv2d(y, w["vit.perceive.proj.0.weight"], w["vit.perceive.proj.0.bias"], stride=2)
    y = F.gelu(y, approximate="none")
    y = F.conv2d(y, w["vit.perceive.proj.2.weight"], w["vit.perceive.proj.2.bias"], stride=1)
    # y: [1, 3072, mh, mw]; append image_newline at dim=-1
    B, C, mh, mw = y.shape
    nl = w["vit.perceive.image_newline"].reshape(1, C, 1, 1).expand(B, C, mh, 1).to(y.dtype)
    y = torch.cat([y, nl], dim=-1)  # [1, 3072, mh, mw+1]
    y = y.reshape(1, C, -1).permute(0, 2, 1).squeeze(0)  # [mh*(mw+1), 3072]
    return y.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--vit-h", type=int, required=True)
    ap.add_argument("--vit-w", type=int, required=True)
    ap.add_argument("--rgb-out", required=True)
    ap.add_argument("--ref-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rgb = rng.integers(0, 256, size=(args.vit_h * 16, args.vit_w * 16, 3), dtype=np.uint8)
    rgb.tofile(args.rgb_out)

    w = load_safetensors(args.weights, args.device)
    out = run_hy4_vision(w, rgb, args.vit_h, args.vit_w, args.device)
    out.numpy().tofile(args.ref_out)
    print(f"GOLDEN_OK grid={args.vit_h}x{args.vit_w} out={tuple(out.shape)} std={out.std().item():.4f} max={out.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
