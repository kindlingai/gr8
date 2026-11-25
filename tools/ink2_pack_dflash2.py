#!/usr/bin/env python3
"""Pack a BF16 Inkling-2-Mini DFlash2 drafter (`incoai/Inkling-2-Mini-DFlash2`, Apache-2.0)
into a self-contained Melded Drafter-in-Quant safetensors artifact (`dflash2-melded.safetensors`)
for zero-config auto-discovery inside an Inkling-2-Mini exlr8 model directory.

Format (`gr8-ink2-dflash2-melded-v1`):
  - Body 2D linear matrices (`fc.weight`, `candidate_selector.hidden_projection.weight`, and
    5x `layers.{l}.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj,{attention,mlp}_conv.kernel_projection}.weight`):
      Pre-quantized SoA Q4_0 (`<name>.qs` U8 [R, C/2] + `<name>.sc` F16 [R, C/32]) or Q8_0 (`<name>.q8` I8 + `<name>.sc` F16).
      Exact bit-for-bit match to `engine/kernels/ink2_dflash2.cuh::k_df2_bf16_to_q4_soa`.
  - Selector codebooks (`candidate_selector.{predecessor,successor}_codebook`):
      Pre-quantized SoA Q4_0 (`<name>.qs` U8 [V, 128] + `<name>.sc` F16 [V, 8]), Q8_0, or BF16.
      Optionally trimmed to `valid_vocab` (200058) to drop unused padded vocabulary rows.
  - Norms and conv base kernels: stored in F32 (or BF16).
  - Target `lm_head` is shared directly with the trunk at zero extra disk or VRAM bytes.

Total disk size (default Q4_0 body + Q4_0 codebooks): ~278 MB (3.4x smaller than 945 MB BF16).
Apache-2.0 attribution to `incoai/Inkling-2-Mini-DFlash2` is preserved in `__metadata__`.
"""
from __future__ import annotations

import argparse
import json
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

BODY_2D_TENSORS: list[str] = [
    "fc.weight",
    "candidate_selector.hidden_projection.weight",
]
for _l in range(5):
    _p = f"layers.{_l}."
    BODY_2D_TENSORS.extend([
        _p + "self_attn.q_proj.weight",
        _p + "self_attn.k_proj.weight",
        _p + "self_attn.v_proj.weight",
        _p + "self_attn.o_proj.weight",
        _p + "mlp.gate_proj.weight",
        _p + "mlp.up_proj.weight",
        _p + "mlp.down_proj.weight",
        _p + "attention_conv.kernel_projection.weight",
        _p + "mlp_conv.kernel_projection.weight",
    ])

NORM_TENSORS: list[str] = [
    "hidden_norm.weight",
    "norm.weight",
]
for _l in range(5):
    _p = f"layers.{_l}."
    NORM_TENSORS.extend([
        _p + "input_layernorm.weight",
        _p + "post_attention_layernorm.weight",
        _p + "self_attn.q_norm.weight",
        _p + "self_attn.k_norm.weight",
        _p + "attention_conv.base_kernel",
        _p + "mlp_conv.base_kernel",
    ])

CODEBOOK_TENSORS: list[str] = [
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
]


def bf16_bytes_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    # Round-to-nearest-even to BF16
    lsb = (u32 >> 16) & np.uint32(1)
    rounded = u32 + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16).tobytes()


def quantize_q4_soa(w_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize 2D float32 [R, C] (C % 32 == 0) to SoA Q4_0 matching k_df2_bf16_to_q4_soa."""
    assert w_f32.ndim == 2 and w_f32.shape[1] % 32 == 0
    R, C = w_f32.shape
    nb = C // 32
    blks = w_f32.reshape(R, nb, 32)
    amax = np.max(np.abs(blks), axis=-1)  # [R, nb]
    d = np.where(amax > 1e-12, amax / np.float32(7.5), np.float32(0.0)).astype(np.float32)
    sc_f16 = d.astype(np.float16)
    df = sc_f16.astype(np.float32)
    inv = np.where(df > 0.0, np.float32(1.0) / np.where(df > 0.0, df, np.float32(1.0)), np.float32(0.0))
    scaled = blks * inv[..., None] + np.float32(8.0)
    q = np.where(df[..., None] > 0.0, np.clip(np.rint(scaled), 0.0, 15.0), 8.0).astype(np.uint8)
    lo = q[:, :, :16]
    hi = q[:, :, 16:]
    packed = (lo | (hi << np.uint8(4))).reshape(R, C // 2)
    return np.ascontiguousarray(packed, dtype=np.uint8), np.ascontiguousarray(sc_f16, dtype=np.float16)


def dequantize_q4_soa(qs: np.ndarray, sc_f16: np.ndarray) -> np.ndarray:
    R, half_c = qs.shape
    C = half_c * 2
    nb = C // 32
    p = qs.reshape(R, nb, 16)
    lo = (p & np.uint8(15)).astype(np.int16) - 8
    hi = (p >> np.uint8(4)).astype(np.int16) - 8
    q = np.concatenate([lo, hi], axis=-1).astype(np.float32)
    return (q * sc_f16.astype(np.float32)[..., None]).reshape(R, C)


def quantize_q8_soa(w_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize 2D float32 [R, C] (C % 32 == 0) to SoA Q8_0 matching k_df2_bf16_to_q8_soa."""
    assert w_f32.ndim == 2 and w_f32.shape[1] % 32 == 0
    R, C = w_f32.shape
    nb = C // 32
    blks = w_f32.reshape(R, nb, 32)
    amax = np.max(np.abs(blks), axis=-1)
    d = np.where(amax > 1e-12, amax / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    sc_f16 = d.astype(np.float16)
    df = sc_f16.astype(np.float32)
    inv = np.where(df > 0.0, np.float32(1.0) / np.where(df > 0.0, df, np.float32(1.0)), np.float32(0.0))
    q = np.where(df[..., None] > 0.0, np.clip(np.rint(blks * inv[..., None]), -127.0, 127.0), 0.0).astype(np.int8)
    return np.ascontiguousarray(q.reshape(R, C), dtype=np.int8), np.ascontiguousarray(sc_f16, dtype=np.float16)


def read_safetensors(path: Path) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
        payload = f.read()
    return hdr, payload


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]], metadata: dict[str, str]) -> int:
    hdr: dict = {"__metadata__": metadata}
    off = 0
    chunks: list[bytes] = []
    for name, dtype, shape, raw in tensors:
        # 16-byte align every tensor start for clean vectorized GPU loads
        pad = (-off) & 15
        if pad:
            chunks.append(b"\x00" * pad)
            off += pad
        end = off + len(raw)
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, end]}
        chunks.append(raw)
        off = end
    hbytes = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    # Pad header JSON with spaces so payload starts on a 16-byte boundary
    hpad = (-(8 + len(hbytes))) & 15
    if hpad:
        hbytes += b" " * hpad
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for c in chunks:
            f.write(c)
    return path.stat().st_size


def load_tensor_f32(hdr: dict, payload: bytes, name: str) -> tuple[np.ndarray, list[int]]:
    if name not in hdr:
        raise KeyError(f"missing tensor {name}")
    info = hdr[name]
    a, b = info["data_offsets"]
    raw = payload[a:b]
    shape = list(info["shape"])
    dt = info["dtype"]
    if dt == "BF16":
        return bf16_bytes_to_f32(raw, tuple(shape)), shape
    if dt == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(tuple(shape)).copy(), shape
    if dt == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(tuple(shape)).astype(np.float32), shape
    raise ValueError(f"unsupported dtype {dt} for {name}")


def pack_dflash2(
    src_path: Path,
    dst_path: Path,
    body_fmt: str = "q4_0",
    cb_fmt: str = "q4_0",
    valid_vocab: int = 200058,
    source_repo: str = "incoai/Inkling-2-Mini-DFlash2",
) -> dict:
    t0 = time.perf_counter()
    hdr, payload = read_safetensors(src_path)
    out_tensors: list[tuple[str, str, list[int], bytes]] = []
    body_nmse: list[float] = []
    cb_nmse: list[float] = []

    for name in BODY_2D_TENSORS:
        w, shape = load_tensor_f32(hdr, payload, name)
        if body_fmt == "q4_0":
            qs, sc = quantize_q4_soa(w)
            rec = dequantize_q4_soa(qs, sc)
            nmse = float(np.mean((w - rec) ** 2) / max(float(np.mean(w ** 2)), 1e-12))
            body_nmse.append(nmse)
            out_tensors.append((name + ".qs", "U8", list(qs.shape), qs.tobytes()))
            out_tensors.append((name + ".sc", "F16", list(sc.shape), sc.tobytes()))
        elif body_fmt == "q8_0":
            q8, sc = quantize_q8_soa(w)
            out_tensors.append((name + ".q8", "I8", list(q8.shape), q8.tobytes()))
            out_tensors.append((name + ".sc", "F16", list(sc.shape), sc.tobytes()))
        else:
            raise ValueError(f"unknown body_fmt {body_fmt}")

    for name in NORM_TENSORS:
        w, shape = load_tensor_f32(hdr, payload, name)
        out_tensors.append((name, "F32", shape, np.ascontiguousarray(w, dtype=np.float32).tobytes()))

    for name in CODEBOOK_TENSORS:
        w, shape = load_tensor_f32(hdr, payload, name)
        if valid_vocab > 0 and w.shape[0] > valid_vocab:
            w = np.ascontiguousarray(w[:valid_vocab])
            shape = list(w.shape)
        if cb_fmt == "q4_0":
            qs, sc = quantize_q4_soa(w)
            rec = dequantize_q4_soa(qs, sc)
            nmse = float(np.mean((w - rec) ** 2) / max(float(np.mean(w ** 2)), 1e-12))
            cb_nmse.append(nmse)
            out_tensors.append((name + ".qs", "U8", list(qs.shape), qs.tobytes()))
            out_tensors.append((name + ".sc", "F16", list(sc.shape), sc.tobytes()))
        elif cb_fmt == "q8_0":
            q8, sc = quantize_q8_soa(w)
            out_tensors.append((name + ".q8", "I8", list(q8.shape), q8.tobytes()))
            out_tensors.append((name + ".sc", "F16", list(sc.shape), sc.tobytes()))
        elif cb_fmt == "bf16":
            out_tensors.append((name, "BF16", shape, f32_to_bf16_bytes(w)))
        else:
            raise ValueError(f"unknown cb_fmt {cb_fmt}")

    metadata = {
        "format": "gr8-ink2-dflash2-melded-v1",
        "source_model": source_repo,
        "source_license": "apache-2.0",
        "base_model": "thinkingmachines/Inkling-2-Mini",
        "body_quant": body_fmt,
        "codebook_quant": cb_fmt,
        "target_layer_ids": "[5,23,41,48]",
        "valid_vocab": str(valid_vocab),
        "attribution": f"Derived from {source_repo} (Apache-2.0); pre-quantized for gr8.",
    }
    out_bytes = write_safetensors(dst_path, out_tensors, metadata)
    in_bytes = src_path.stat().st_size
    stats = {
        "input_bytes": in_bytes,
        "output_bytes": out_bytes,
        "compression_ratio": round(in_bytes / max(out_bytes, 1), 3),
        "body_fmt": body_fmt,
        "cb_fmt": cb_fmt,
        "valid_vocab": valid_vocab,
        "mean_body_nmse": round(float(np.mean(body_nmse)), 6) if body_nmse else 0.0,
        "mean_cb_nmse": round(float(np.mean(cb_nmse)), 6) if cb_nmse else 0.0,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return stats


def self_test() -> int:
    rng = np.random.default_rng(1234)
    w = rng.normal(0.0, 0.5, size=(64, 256)).astype(np.float32)
    qs, sc = quantize_q4_soa(w)
    rec = dequantize_q4_soa(qs, sc)
    nmse = float(np.mean((w - rec) ** 2) / np.mean(w ** 2))
    assert qs.shape == (64, 128) and sc.shape == (64, 8)
    assert nmse < 0.015, f"Q4_0 NMSE too high: {nmse}"

    # Zero-row edge case
    z = np.zeros((4, 64), dtype=np.float32)
    zqs, zsc = quantize_q4_soa(z)
    zrec = dequantize_q4_soa(zqs, zsc)
    assert np.all(zrec == 0.0) and np.all(zqs == 0x88)

    # Full safetensors round-trip on synthetic mini DFlash2
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.safetensors"
        dst = Path(td) / "dflash2-melded.safetensors"
        synth = []
        for name in BODY_2D_TENSORS:
            a = rng.normal(0.0, 0.2, size=(16, 64)).astype(np.float32)
            synth.append((name, "BF16", [16, 64], f32_to_bf16_bytes(a)))
        for name in NORM_TENSORS:
            a = rng.normal(1.0, 0.1, size=(64,)).astype(np.float32)
            synth.append((name, "BF16", [64], f32_to_bf16_bytes(a)))
        for name in CODEBOOK_TENSORS:
            a = rng.normal(0.0, 0.3, size=(128, 256)).astype(np.float32)
            synth.append((name, "BF16", [128, 256], f32_to_bf16_bytes(a)))
        write_safetensors(src, synth, {"format": "pt"})
        stats = pack_dflash2(src, dst, body_fmt="q4_0", cb_fmt="q4_0", valid_vocab=100)
        hdr2, _ = read_safetensors(dst)
        assert hdr2["__metadata__"]["source_license"] == "apache-2.0"
        assert hdr2["fc.weight.qs"]["shape"] == [16, 32]
        assert hdr2["candidate_selector.predecessor_codebook.qs"]["shape"] == [100, 128]
        assert stats["compression_ratio"] > 3.0
    print(f"[SELF-TEST PASS] q4_nmse={nmse:.6f}, synth_ratio={stats['compression_ratio']}x")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack Inkling-2-Mini DFlash2 into melded safetensors.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--src", type=Path, help="Path to BF16 incoai/Inkling-2-Mini-DFlash2 model.safetensors")
    ap.add_argument("--dst", type=Path, help="Output path (e.g., <model_dir>/dflash2-melded.safetensors)")
    ap.add_argument("--body-fmt", default="q4_0", choices=["q4_0", "q8_0"])
    ap.add_argument("--cb-fmt", default="q4_0", choices=["q4_0", "q8_0", "bf16"])
    ap.add_argument("--valid-vocab", type=int, default=200058, help="Trim codebooks to valid_vocab (0 = keep full)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.src or not args.dst:
        ap.error("--src and --dst are required unless --self-test is given")
    stats = pack_dflash2(args.src, args.dst, body_fmt=args.body_fmt, cb_fmt=args.cb_fmt, valid_vocab=args.valid_vocab)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
