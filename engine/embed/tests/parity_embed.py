#!/usr/bin/env python3
"""Parity verifier: compares g6embed (Unix socket PROTOCOL.md v1 and/or g6front-rs /v1/embeddings)
against reference PyTorch forward (exlr8.load_dense_model + transformers Qwen3VLModel) on the exact same
quantized release weights.

Asserts cosine similarity >= 0.9999 across:
  - Single short & medium text prompts
  - Variable-length packed batch (B=8)
  - Matryoshka dimensions (2048, 1536, 1024, 512, 256) and normalize=False
  - Multimodal prompt with synthetic 512x512 RGB image (ViT + DeepStack + 3D M-RoPE)
  - Optional HTTP /v1/embeddings end-to-end verification (text, MRL, base64, multimodal messages)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import socket
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


class SocketClient:
    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.buf = b""
        self.hello = self.recv_frame()
        if self.hello.get("type") != "hello" or not self.hello.get("embeddings"):
            raise RuntimeError(f"unexpected hello: {self.hello}")

    def close(self) -> None:
        self.sock.close()

    def send_frame(self, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.sock.sendall(struct.pack(">I", len(data)) + data)

    def _recv_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self.buf)))
            if not chunk:
                raise RuntimeError("EOF on engine socket")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv_frame(self) -> dict[str, Any]:
        (n,) = struct.unpack(">I", self._recv_exact(4))
        return json.loads(self._recv_exact(n).decode("utf-8"))

    def embed_batch(self, reqs: list[dict[str, Any]]) -> dict[int, np.ndarray]:
        for r in reqs:
            self.send_frame(r)
        out: dict[int, np.ndarray] = {}
        while len(out) < len(reqs):
            f = self.recv_frame()
            t = f.get("type")
            if t == "embedding":
                out[int(f["id"])] = np.asarray(f["embedding"], dtype=np.float32)
            elif t == "finish":
                if f.get("reason") not in ("stop", None):
                    raise RuntimeError(f"embed failed: {f}")
        return out


def make_test_image(w: int = 512, h: int = 512) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((xx * 255) // max(1, w - 1)).astype(np.uint8)
    g = ((yy * 255) // max(1, h - 1)).astype(np.uint8)
    b = (((xx ^ yy) + 37) & 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def load_reference_model(model_dir: Path, device: str, exlr8_repo: str | None) -> tuple[Any, Any]:
    if exlr8_repo:
        sys.path.insert(0, exlr8_repo)
    from exlr8.loader.exlr8 import load_dense_model
    from transformers import AutoConfig, AutoProcessor, Qwen3VLModel

    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False)
    model = Qwen3VLModel(cfg).to(dtype=torch.bfloat16)
    sd = load_dense_model(str(model_dir), dtype=torch.bfloat16)
    # Strip "model." prefix if Qwen3VLModel expects language_model.* / visual.*
    model_keys = set(model.state_dict().keys())
    mapped = {}
    for k, v in sd.items():
        if k in model_keys:
            mapped[k] = v.clone()
        elif k.startswith("model.") and k[6:] in model_keys:
            mapped[k[6:]] = v.clone()
        else:
            mapped[k] = v.clone()
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"warning: missing ref keys ({len(missing)}): {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"warning: unexpected ref keys ({len(unexpected)}): {unexpected[:5]}", file=sys.stderr)
    model = model.to(device).eval()
    proc = AutoProcessor.from_pretrained(str(model_dir), min_pixels=4096, max_pixels=1310720)
    return model, proc


@torch.no_grad()
def ref_embed_tokens(
    model: Any,
    input_ids: list[int],
    device: str,
    dimensions: int | None = None,
    normalize: bool = True,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> np.ndarray:
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    kwargs: dict[str, Any] = {"input_ids": ids, "attention_mask": attn}
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.to(device=device, dtype=torch.bfloat16)
        kwargs["image_grid_thw"] = image_grid_thw.to(device=device)
    out = model(**kwargs)
    last = out.last_hidden_state[0, -1, :].float()
    if dimensions is not None:
        last = last[:dimensions]
    if normalize:
        last = F.normalize(last, p=2, dim=-1)
    return last.cpu().numpy().astype(np.float32)


def http_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--socket", help="g6embed Unix domain socket path")
    ap.add_argument("--http-url", help="optional g6front-rs base URL (e.g. http://127.0.0.1:8080)")
    ap.add_argument("--exlr8-repo", help="path to exlr8 repo containing exlr8/loader/exlr8.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.9999)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model, proc = load_reference_model(model_dir, args.device, args.exlr8_repo)
    tok = proc.tokenizer
    full_dim = int(model.config.text_config.hidden_size)

    prompts = [
        "Explain how Hadamard-rotated trellis quantization preserves cosine similarity in dense embeddings.",
        "What is the capital of France?",
        "def quicksort(xs): return xs if len(xs) <= 1 else quicksort([x for x in xs[1:] if x < xs[0]]) + [xs[0]] + quicksort([x for x in xs[1:] if x >= xs[0]])",
        "Distributed memory systems store episodic representations as normalized dense vectors retrieved via approximate nearest neighbor search. " * 6,
        "Short query.",
        "Multi-lingual test: Bonjour le monde, Hallo Welt,こんにちは世界.",
        "Numbers and symbols: 3.141592653589793, 2^16 = 65536, O(N log N).",
        "Final sentence in an eight-way packed variable-length prefill batch.",
    ]
    tok_lists = [tok.encode(p, add_special_tokens=True) for p in prompts]

    # Build multimodal prompt via transformers AutoProcessor so pixel grid & tokens match exactly
    img = make_test_image(512, 512)
    mm_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the color gradient in this image."}]}]
    mm_text = proc.apply_chat_template(mm_messages, tokenize=False, add_generation_prompt=False)
    mm_inputs = proc(text=[mm_text], images=[img], return_tensors="pt")
    mm_ids = mm_inputs["input_ids"][0].tolist()
    mm_grid = mm_inputs["image_grid_thw"]  # [[1, vit_h, vit_w]]
    vit_h, vit_w = int(mm_grid[0, 1].item()), int(mm_grid[0, 2].item())
    img_tok_id = int(model.config.image_token_id)
    img_pos = mm_ids.index(img_tok_id)
    rgb_resized = img.resize((vit_w * 16, vit_h * 16), Image.Resampling.BICUBIC)
    rgb_b64 = base64.b64encode(rgb_resized.tobytes()).decode("ascii")

    ref_text = [ref_embed_tokens(model, ids, args.device) for ids in tok_lists]
    ref_mrl = {
        d: ref_embed_tokens(model, tok_lists[0], args.device, dimensions=d, normalize=True)
        for d in [full_dim, 1536, 1024, 512, 256]
        if d <= full_dim
    }
    ref_unnorm = ref_embed_tokens(model, tok_lists[0], args.device, normalize=False)
    ref_mm = ref_embed_tokens(
        model,
        mm_ids,
        args.device,
        pixel_values=mm_inputs["pixel_values"],
        image_grid_thw=mm_grid,
    )

    ok = True
    results: list[dict[str, Any]] = []

    def check(name: str, got: np.ndarray, ref: np.ndarray) -> None:
        nonlocal ok
        c = cos_sim(got, ref)
        mad = float(np.max(np.abs(got - ref)))
        passed = c >= args.threshold
        if not passed:
            ok = False
        rec = {"case": name, "dim": int(got.shape[0]), "cos": round(c, 8), "max_abs_diff": round(mad, 6), "pass": passed}
        results.append(rec)
        print(json.dumps(rec))

    if args.socket:
        cli = SocketClient(args.socket)
        # 1. Single & batched text (B=8 packed)
        reqs = [{"type": "embed", "id": i + 1, "prompt": ids} for i, ids in enumerate(tok_lists)]
        got_batch = cli.embed_batch(reqs)
        for i in range(len(tok_lists)):
            check(f"sock_batch_{i}_len{len(tok_lists[i])}", got_batch[i + 1], ref_text[i])

        # 2. MRL dimensions + normalize=False
        mrl_reqs = []
        rid = 100
        for d in ref_mrl:
            mrl_reqs.append({"type": "embed", "id": rid, "prompt": tok_lists[0], "dimensions": d, "normalize": True})
            rid += 1
        mrl_reqs.append({"type": "embed", "id": 200, "prompt": tok_lists[0], "normalize": False})
        got_mrl = cli.embed_batch(mrl_reqs)
        for idx, d in enumerate(ref_mrl):
            check(f"sock_mrl_{d}", got_mrl[100 + idx], ref_mrl[d])
        check("sock_unnorm", got_mrl[200], ref_unnorm)

        # 3. Multimodal 512x512
        mm_req = {
            "type": "embed",
            "id": 300,
            "prompt": mm_ids,
            "images": [{"pos": img_pos, "vit_h": vit_h, "vit_w": vit_w, "rgb_b64": rgb_b64}],
        }
        got_mm = cli.embed_batch([mm_req])[300]
        check(f"sock_mm_{vit_h}x{vit_w}", got_mm, ref_mm)
        cli.close()

    if args.http_url:
        base = args.http_url.rstrip("/")
        # Discover model id from /v1/models
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as resp:
            model_id = json.loads(resp.read().decode("utf-8"))["data"][0]["id"]

        # 1. Text batch via /v1/embeddings
        r = http_post(f"{base}/v1/embeddings", {"model": model_id, "input": prompts[:4]})
        for i in range(4):
            got = np.asarray(r["data"][i]["embedding"], dtype=np.float32)
            check(f"http_text_{i}", got, ref_text[i])

        # 2. MRL + base64 encoding
        r_b64 = http_post(
            f"{base}/v1/embeddings",
            {"model": model_id, "input": prompts[0], "dimensions": 512, "encoding_format": "base64"},
        )
        raw = base64.b64decode(r_b64["data"][0]["embedding"])
        got_512 = np.asarray(struct.unpack(f"<{len(raw)//4}f", raw), dtype=np.float32)
        check("http_mrl_512_b64", got_512, ref_mrl[512])

        # 3. Multimodal messages with inline base64 PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        r_mm = http_post(
            f"{base}/v1/embeddings",
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                            {"type": "text", "text": "Describe the color gradient in this image."},
                        ],
                    }
                ],
            },
        )
        got_http_mm = np.asarray(r_mm["data"][0]["embedding"], dtype=np.float32)
        check("http_mm_512", got_http_mm, ref_mm)

    if not results:
        print("no --socket or --http-url provided", file=sys.stderr)
        return 2
    min_cos = min(r["cos"] for r in results)
    print(f"SUMMARY: {len(results)} cases, min_cos={min_cos:.8f}, pass={ok}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
