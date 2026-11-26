#!/usr/bin/env python3
"""Cross-check gr8's multi-group exlr8 dense loader against `exlr8/loader/exlr8.py` on synthetic EG2 and Qwen3-VL
multi-group release directories (including 8B non-128 FP8_ROW side=OUT and copied BF16 pos_embed).

    python3 make_multigroup_fixture.py EXLR8_REPO EXLR8DUMP_BIN [--work /tmp/mg-check]

Generates deterministic EG2 and Qwen3-VL multi-group releases, loads them with `exlr8.load_dense_model` (if present)
or `exlr8.load_matrices`, runs `exlr8dump --verify` on every group:layer and vocab file, and asserts bit-exact FP32
and BF16 equality for every tensor.
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


def _rand_helpers(seed: int):
    g = torch.Generator().manual_seed(seed)

    def u8(*shp):
        return torch.randint(0, 256, shp, generator=g, dtype=torch.int32).to(torch.uint8)

    def fp8(*shp):
        b = u8(*shp)
        return torch.where((b & 0x7F) == 0x7F, b ^ 1, b)

    return g, u8, fp8


def build_qwen3vl_release(X, root: str):
    os.makedirs(root, exist_ok=True)
    g, u8, fp8 = _rand_helpers(42)

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


def build_eg2_release(X, root: str):
    dense = os.path.join(root, "dense")
    os.makedirs(dense, exist_ok=True)
    g, u8, fp8 = _rand_helpers(1337)

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
        os.path.join(dense, "exlr8-dense-text-000.safetensors"),
        text,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "embedding_gemma2", "group": "text", "layer": "0"},
    )

    # vision:0
    vis = {}
    add_matrix(
        X,
        vis,
        "self_attn.q_proj.linear",
        torch.tensor([X.INT4, 128, 128, 4, 1, 128, 0, 0], dtype=torch.int64),
        {"q4": u8(128, 64), "s4": (torch.rand(128, 4, generator=g) * 0.05).half()},
    )
    write_st(
        os.path.join(dense, "exlr8-dense-vision-000.safetensors"),
        vis,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "embedding_gemma2", "group": "vision", "layer": "0"},
    )

    # audio:0
    aud = {}
    add_matrix(
        X,
        aud,
        "self_attn.post.linear",
        torch.tensor([X.MXFP8, 128, 128, 0, 1, 128, 0, 0], dtype=torch.int64),
        {"fp8": fp8(128, 128), "e8m0": torch.randint(120, 134, (128, 4), generator=g, dtype=torch.int32).to(torch.uint8)},
    )
    write_st(
        os.path.join(dense, "exlr8-dense-audio-000.safetensors"),
        aud,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "embedding_gemma2", "group": "audio", "layer": "0"},
    )

    # global:0
    glob = {}
    add_matrix(
        X,
        glob,
        "language_model.embedding_projection",
        torch.tensor([X.FP8_BLOCK, 144, 144, 0, 1, 128, 0, 0], dtype=torch.int64),
        {"fp8": fp8(144, 144), "scale": torch.rand(2, 2, generator=g) * 0.02},
    )
    write_st(
        os.path.join(dense, "exlr8-dense-global-000.safetensors"),
        glob,
        {"format": "exlr8-v2", "contents": "dense", "model_profile": "embedding_gemma2", "group": "global", "layer": "0"},
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
    write_st(
        os.path.join(dense, "exlr8-vocab.safetensors"),
        voc,
        {"format": "exlr8-v2", "contents": "vocab", "model_profile": "embedding_gemma2"},
    )


def verify_release(X, dump_bin: str, work: str, name: str, rel: str, targets: tuple, fallback_specs: list):
    for cli_dtype, torch_dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        if hasattr(X, "load_dense_model"):
            ref_all = X.load_dense_model(rel, dtype=torch_dtype, device="cpu", verify=True)
        else:
            ref_all = {}
            for sub, pfx in fallback_specs:
                p = os.path.join(rel, sub)
                ref_all.update(X.load_matrices(X.SafeFile(p), pfx, torch_dtype, "cpu", True))

        got_all = {}
        for target in targets:
            out = os.path.join(work, f"{name}_{cli_dtype}_{target.replace(':', '_')}.safetensors")
            subprocess.run([dump_bin, rel, target, out, "--dtype", cli_dtype, "--verify"], check=True)
            sf = X.SafeFile(out)
            for n in sf.names():
                got_all[n] = sf.get(n)
        assert set(got_all) == set(ref_all), f"{name}/{cli_dtype} key diff: {set(got_all) ^ set(ref_all)}"
        for k in sorted(ref_all):
            assert ref_all[k].shape == got_all[k].shape and ref_all[k].dtype == got_all[k].dtype, (
                f"{name}/{cli_dtype} shape/dtype mismatch on {k}: "
                f"{ref_all[k].shape}/{ref_all[k].dtype} vs {got_all[k].shape}/{got_all[k].dtype}"
            )
            assert fnv64(ref_all[k]) == fnv64(got_all[k]), f"{name}/{cli_dtype} bit mismatch on {k}"
        print(f"OK: {name} ({cli_dtype}): {len(ref_all)} multi-group tensors match bit-for-bit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exlr8_repo")
    ap.add_argument("dump_bin")
    ap.add_argument("--work", default="/tmp/exlr8-multigroup-check")
    a = ap.parse_args()
    sys.path.insert(0, a.exlr8_repo)
    from loader import exlr8 as X

    q_rel = os.path.join(a.work, "qwen3vl")
    build_qwen3vl_release(X, q_rel)
    verify_release(
        X,
        a.dump_bin,
        a.work,
        "qwen3_vl",
        q_rel,
        ("text:0", "vision:0", "global:0", "vocab"),
        [
            ("exlr8-dense-text-000.safetensors", lambda n: f"model.language_model.layers.0.{n}"),
            ("exlr8-dense-vision-000.safetensors", lambda n: f"model.visual.blocks.0.{n}"),
            ("exlr8-dense-global-000.safetensors", lambda n: n),
            (
                "exlr8-vocab.safetensors",
                lambda n: {"embed_tokens": "model.language_model.embed_tokens",
                           "norm.weight": "model.language_model.norm.weight"}.get(n, n),
            ),
        ],
    )

    e_rel = os.path.join(a.work, "embedding_gemma2")
    build_eg2_release(X, e_rel)
    verify_release(
        X,
        a.dump_bin,
        a.work,
        "embedding_gemma2",
        e_rel,
        ("text:0", "vision:0", "audio:0", "global:0", "vocab"),
        [
            ("dense/exlr8-dense-text-000.safetensors", lambda n: f"language_model.layers.0.{n}"),
            ("dense/exlr8-dense-vision-000.safetensors", lambda n: f"vision_tower.encoder.layers.0.{n}"),
            ("dense/exlr8-dense-audio-000.safetensors", lambda n: f"audio_tower.layers.0.{n}"),
            ("dense/exlr8-dense-global-000.safetensors", lambda n: n),
            ("dense/exlr8-vocab.safetensors", lambda n: {"embed_tokens": "language_model.embed_tokens"}.get(n, n)),
        ],
    )


if __name__ == "__main__":
    main()
