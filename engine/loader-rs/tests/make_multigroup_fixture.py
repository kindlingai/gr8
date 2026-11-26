#!/usr/bin/env python3
"""Cross-check gr8's multi-group exlr8 dense loader against `exlr8/loader/exlr8.py` on synthetic EG2 and Qwen3-VL
multi-group release directories (including 8B non-128 FP8_ROW side=OUT and copied BF16 pos_embed).

    python3 make_multigroup_fixture.py EXLR8_REPO EXLR8DUMP_BIN [--work /tmp/mg-check]

Generates deterministic EG2 and Qwen3-VL multi-group releases, loads them with `exlr8.load_dense_model` (if present)
or `exlr8.load_matrices`, runs `exlr8dump` on every group:layer and vocab file, and asserts bit-exact FP32 and BF16
equality for every tensor.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch


def fnv64(t: torch.Tensor) -> int:
    h = 0xCBF29CE484222325
    for b in t.contiguous().cpu().view(torch.uint8).reshape(-1).tolist():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def write_st(path: str, tensors: dict, meta: dict):
    import json
    import struct

    st_names = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int64: "I64",
        torch.uint64: "U64",
        torch.uint8: "U8",
    }
    header, off = {}, 0
    for name, t in tensors.items():
        if isinstance(t, np.ndarray):
            raw = np.ascontiguousarray(t).tobytes()
            dt = "U64" if t.dtype == np.uint64 else "I64"
            shp = list(t.shape)
        else:
            raw = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
            dt = st_names[t.dtype]
            shp = list(t.shape)
        header[name] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    if meta:
        header["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    raw_h = json.dumps(header, separators=(",", ":")).encode()
    raw_h += b" " * (-len(raw_h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw_h)) + raw_h)
        for t in tensors.values():
            if isinstance(t, np.ndarray):
                f.write(np.ascontiguousarray(t).tobytes())
            else:
                f.write(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())


def add_matrix(X, out_dict, name, rec, data):
    out_dict[f"{name}.record"] = rec
    out_dict[f"{name}.checksum"] = np.array(X.dense_checksums(rec, data), dtype=np.uint64)
    for k, v in data.items():
        out_dict[f"{name}.{k}"] = v


def build_qwen3vl_release(X, root: str):
    os.makedirs(root, exist_ok=True)
    g = torch.Generator().manual_seed(42)

    def u8(*shp):
        return torch.randint(0, 256, shp, generator=g, dtype=torch.int32).to(torch.uint8)

    def fp8(*shp):
        b = u8(*shp)
        return torch.where((b & 0x7F) == 0x7F, b ^ 1, b)

    # text:0
    text = {}
    add_matrix(
        X,
        text,
        "self_attn.q_proj",
        torch.tensor([X.TRELLIS, 128, 128, 4, 1, 128, 1, 1], dtype=torch.int64),
        {
            "trellis": u8(1, 128 * 128 // 256 * 32 * 4),
            "suh": (torch.randn(128, generator=g) * 0.2).half(),
            "svh": (torch.randn(128, generator=g) * 0.05).half(),
        },
    )
    text["input_layernorm.weight"] = torch.randn(128, generator=g).to(torch.bfloat16)
    write_st(
        os.path.join(root, "exlr8-dense-text-000.safetensors"),
        text,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "qwen3_vl", "group": "text", "layer": "0"},
    )

    # vision:0 (8B-style non-128 FP8_ROW side=OUT)
    vis = {}
    add_matrix(
        X,
        vis,
        "mlp.linear_fc1",
        torch.tensor([X.FP8_ROW, 144, 128, 0, 1, 128, 0, 0], dtype=torch.int64),
        {"fp8": fp8(144, 128), "scale": torch.rand(144, generator=g) * 0.01},
    )
    add_matrix(
        X,
        vis,
        "mlp.linear_fc2",
        torch.tensor([X.FP8_ROW, 128, 144, 0, 1, 128, 0, 0], dtype=torch.int64),
        {"fp8": fp8(128, 144), "scale": torch.rand(128, generator=g) * 0.01},
    )
    vis["attn.qkv.bias"] = torch.randn(128, generator=g).to(torch.bfloat16)
    write_st(
        os.path.join(root, "exlr8-dense-vision-000.safetensors"),
        vis,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "qwen3_vl", "group": "vision", "layer": "0"},
    )

    # global:0 (copied 2D BF16 pos_embed + 5D patch_embed.proj.weight)
    glob = {}
    add_matrix(
        X,
        glob,
        "model.visual.pos_embed",
        torch.tensor([X.BF16, 160, 64, 0, 0, 0, 0, 0], dtype=torch.int64),
        {"bf16": torch.randn(160, 64, generator=g).to(torch.bfloat16)},
    )
    glob["model.visual.patch_embed.proj.weight"] = torch.randn(4, 3, 2, 4, 4, generator=g).to(torch.bfloat16)
    write_st(
        os.path.join(root, "exlr8-dense-global-000.safetensors"),
        glob,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "qwen3_vl", "group": "global", "layer": "0"},
    )

    # vocab
    voc = {}
    add_matrix(
        X,
        voc,
        "embed_tokens",
        torch.tensor([X.FP8_ROW, 64, 128, 0, 1, 128, 0, 0], dtype=torch.int64),
        {"fp8": fp8(64, 128), "scale": torch.rand(64, generator=g) * 0.01},
    )
    voc["norm.weight"] = torch.randn(128, generator=g).to(torch.bfloat16)
    write_st(
        os.path.join(root, "exlr8-vocab.safetensors"),
        voc,
        {"format": "exlr8-v2", "contents": "vocab", "model_profile": "qwen3_vl"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exlr8_repo")
    ap.add_argument("dump_bin")
    ap.add_argument("--work", default="/tmp/exlr8-multigroup-check")
    a = ap.parse_args()
    sys.path.insert(0, a.exlr8_repo)
    from loader import exlr8 as X

    rel = os.path.join(a.work, "qwen3vl")
    build_qwen3vl_release(X, rel)
    ref_all = {}
    if hasattr(X, "load_dense_model"):
        ref_all = X.load_dense_model(rel, dtype=torch.float32, device="cpu", verify=True)
    else:
        for which, pfx in [
            ("exlr8-dense-text-000.safetensors", lambda n: f"model.language_model.layers.0.{n}"),
            ("exlr8-dense-vision-000.safetensors", lambda n: f"model.visual.blocks.0.{n}"),
            ("exlr8-dense-global-000.safetensors", lambda n: n),
            ("exlr8-vocab.safetensors", lambda n: {"embed_tokens": "model.language_model.embed_tokens",
                                                   "norm.weight": "model.language_model.norm.weight"}.get(n, n)),
        ]:
            ref_all.update(X.load_matrices(X.SafeFile(os.path.join(rel, which)), pfx, torch.float32, "cpu", True))

    got_all = {}
    for target in ("text:0", "vision:0", "global:0", "vocab"):
        out = os.path.join(a.work, f"{target.replace(':', '_')}.safetensors")
        subprocess.run([a.dump_bin, rel, target, out, "--dtype", "f32"], check=True)
        sf = X.SafeFile(out)
        for n in sf.names():
            got_all[n] = sf.get(n)
    assert set(got_all) == set(ref_all), f"key diff: {set(got_all) ^ set(ref_all)}"
    for k in sorted(ref_all):
        assert ref_all[k].shape == got_all[k].shape and ref_all[k].dtype == got_all[k].dtype
        assert fnv64(ref_all[k]) == fnv64(got_all[k]), f"bit mismatch on {k}"
    print(f"OK: {len(ref_all)} multi-group tensors match bit-for-bit")


if __name__ == "__main__":
    main()
