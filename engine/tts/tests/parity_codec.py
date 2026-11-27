#!/usr/bin/env python3
"""Gate E2 codec parity verifier: native CUDA g6tts_bench codec vs PyTorch fish-speech DAC (fp32).

Verifies all three Gate E2 thresholds across 20 clips (synthetic multi-harmonic speech-like clips
or user-supplied WAV/PCM clips via --clips-dir):
  1. Decode parity:  SNR(torch_fp32_decode, gr8_decode) >= 45.0 dB on every clip (and mean).
  2. Encode parity:  code exact-match >= 0.98 over all 10 codebooks x all frames, AND
                     round-trip SNR(torch_decode(torch_codes), torch_decode(gr8_codes)) >= 25.0 dB.
  3. Streaming parity: streaming decode (strides 1,1,2,4,...) matches whole-sequence decode within
                     <= 1 LSB of s16 across every sample of every clip.

Usage:
  python3 engine/tts/tests/parity_codec.py \
      --bench bin/g6tts_bench \
      --codec-safetensors /work-data/out/dev-ref/codec/exlr8-codec.safetensors \
      --codec-pth /work-data/src/drama-3/codec.pth \
      --fish-speech /work-data/src/fish-speech \
      [--clips-dir DIR] [--n-clips 20] [--device cuda:0] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

SAMPLE_RATE = 44100
FRAME_SAMPLES = 2048
NUM_CODEBOOKS = 10

MIN_DECODE_SNR_DB = 45.0
MIN_ENCODE_EXACT_MATCH = 0.98
MIN_ROUNDTRIP_SNR_DB = 25.0
MAX_STREAMING_LSB_DIFF = 1


def compute_snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-noise ratio in dB: 10 * log10(sum(ref^2) / sum((ref - test)^2))."""
    r = ref.astype(np.float64).reshape(-1)
    t = test.astype(np.float64).reshape(-1)
    n = min(r.size, t.size)
    r, t = r[:n], t[:n]
    sig = float(np.sum(r * r))
    err = float(np.sum((r - t) ** 2))
    if sig <= 0.0:
        return float("inf") if err == 0.0 else 0.0
    if err <= 1e-24:
        return 120.0
    return 10.0 * math.log10(sig / err)


def f32_to_s16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def s16_to_f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32767.0


def synthesize_test_clip(idx: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a deterministic diverse speech-like test clip (0.6s to 3.0s, mono fp32 in [-0.9, 0.9])."""
    rng = np.random.default_rng(20261127 + idx * 7919)
    # Vary duration across clips, including non-multiples of 2048 to exercise right-padding in encode
    dur_samples = int(rng.integers(int(0.6 * sample_rate), int(3.0 * sample_rate)))
    t = np.arange(dur_samples, dtype=np.float64) / float(sample_rate)

    # Fundamental frequency with gentle vibrato + formant-like harmonics + syllabic envelope + subtle fricative noise
    f0 = float(rng.uniform(95.0, 260.0))
    vib = 1.0 + 0.015 * np.sin(2.0 * np.pi * float(rng.uniform(3.5, 6.5)) * t)
    phase = 2.0 * np.pi * np.cumsum(f0 * vib) / float(sample_rate)

    y = np.zeros(dur_samples, dtype=np.float64)
    for h in range(1, 10):
        amp = float(rng.uniform(0.05, 0.45)) / (h ** 0.75)
        y += amp * np.sin(h * phase + float(rng.uniform(0.0, 2.0 * np.pi)))

    # Syllabic amplitude modulation (2-6 Hz) with brief pauses
    mod_freq = float(rng.uniform(2.0, 6.0))
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * mod_freq * t + float(rng.uniform(0.0, 2.0 * np.pi))))
    env = np.power(env, 1.5)
    # Smooth fade-in/fade-out
    fade = min(1024, dur_samples // 10)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        env[:fade] *= ramp
        env[-fade:] *= ramp[::-1]

    noise = rng.normal(0.0, 0.015, size=dur_samples)
    y = y * env + noise * (0.2 + 0.8 * env)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y * (float(rng.uniform(0.4, 0.85)) / peak)
    return y.astype(np.float32)


def load_clips(clips_dir: Path | None, n_clips: int) -> list[tuple[str, np.ndarray]]:
    clips: list[tuple[str, np.ndarray]] = []
    if clips_dir is not None and clips_dir.is_dir():
        wavs = sorted(clips_dir.glob("*.wav"))[:n_clips]
        if wavs:
            import soundfile as sf

            for wp in wavs:
                data, sr = sf.read(str(wp), dtype="float32", always_2d=True)
                mono = data.mean(axis=1)
                if sr != SAMPLE_RATE:
                    import torchaudio

                    mono = (
                        torchaudio.functional.resample(torch.from_numpy(mono).unsqueeze(0), sr, SAMPLE_RATE)
                        .squeeze(0)
                        .numpy()
                        .astype(np.float32)
                    )
                clips.append((wp.name, mono))
    while len(clips) < n_clips:
        idx = len(clips)
        clips.append((f"synth_{idx:02d}", synthesize_test_clip(idx)))
    return clips[:n_clips]


def load_fish_dac(fish_speech_dir: Path, codec_pth: Path, device: str) -> Any:
    sys.path.insert(0, str(fish_speech_dir))
    import hydra
    from omegaconf import OmegaConf

    cfg_path = fish_speech_dir / "fish_speech" / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    model = hydra.utils.instantiate(cfg)
    try:
        sd = torch.load(str(codec_pth), map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(str(codec_pth), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Clone tensors to stay 64K-page safe before moving to CUDA
    sd = {k: v.clone() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(sd, strict=False)
    model = model.float().to(device).eval()
    return model


def run_bench_decode(
    bench_bin: str,
    codec_st: str,
    codes_10xF: np.ndarray,
    tmp_dir: Path,
    strides: str | None = None,
) -> np.ndarray:
    in_path = tmp_dir / "codes_in.bin"
    out_path = tmp_dir / ("pcm_stream.bin" if strides else "pcm_whole.bin")
    codes_10xF.astype("<i4", copy=False).tofile(in_path)
    cmd = [
        bench_bin,
        "--codec-path",
        codec_st,
        "--codec-decode",
        str(in_path),
        "--out",
        str(out_path),
    ]
    if strides:
        cmd += ["--codec-strides", strides]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"g6tts_bench decode failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}")
    return np.fromfile(out_path, dtype="<i2")


def run_bench_encode(bench_bin: str, codec_st: str, pcm_f32: np.ndarray, tmp_dir: Path) -> np.ndarray:
    in_path = tmp_dir / "pcm_in.bin"
    out_path = tmp_dir / "codes_out.bin"
    pcm_f32.astype("<f4", copy=False).tofile(in_path)
    cmd = [
        bench_bin,
        "--codec-path",
        codec_st,
        "--codec-encode",
        str(in_path),
        "--out",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"g6tts_bench encode failed (rc={res.returncode}):\n{res.stderr}\n{res.stdout}")
    flat = np.fromfile(out_path, dtype="<i4")
    if flat.size % NUM_CODEBOOKS != 0:
        raise RuntimeError(f"unexpected codes size {flat.size}")
    F = flat.size // NUM_CODEBOOKS
    return flat.reshape(NUM_CODEBOOKS, F)


@torch.no_grad()
def evaluate_clip(
    name: str,
    pcm_f32: np.ndarray,
    model: Any,
    bench_bin: str,
    codec_st: str,
    device: str,
    strides: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="parity_codec_") as td:
        tmp_dir = Path(td)

        # 1. PyTorch fp32 encode -> reference codes [10, F]
        wav_t = torch.from_numpy(pcm_f32).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        wav_lens = torch.tensor([pcm_f32.shape[0]], device=device, dtype=torch.long)
        ref_codes_t, _ = model.encode(wav_t, wav_lens)  # [1, 10, F]
        ref_codes = ref_codes_t[0].cpu().numpy().astype(np.int32)
        F = int(ref_codes.shape[1])

        # 2. PyTorch fp32 decode of reference codes -> reference audio [F * 2048]
        ref_dec_f32 = model.from_indices(ref_codes_t)[0, 0].float().cpu().numpy()

        # 3. Native CUDA whole-sequence decode + streamed decode (strides 1,1,2,4)
        gr8_whole_s16 = run_bench_decode(bench_bin, codec_st, ref_codes, tmp_dir, strides=None)
        gr8_stream_s16 = run_bench_decode(bench_bin, codec_st, ref_codes, tmp_dir, strides=strides)

        decode_snr_db = compute_snr_db(ref_dec_f32, s16_to_f32(gr8_whole_s16))
        stream_snr_db = compute_snr_db(ref_dec_f32, s16_to_f32(gr8_stream_s16))
        max_lsb_diff = int(np.max(np.abs(gr8_whole_s16.astype(np.int32) - gr8_stream_s16.astype(np.int32))))

        # 4. Native CUDA encode -> compare codes + round-trip decode SNR
        gr8_codes = run_bench_encode(bench_bin, codec_st, pcm_f32, tmp_dir)
        if gr8_codes.shape != ref_codes.shape:
            raise AssertionError(f"{name}: encode shape mismatch gr8={gr8_codes.shape} vs torch={ref_codes.shape}")
        exact_match = float(np.mean(gr8_codes == ref_codes))
        per_cb_match = [float(np.mean(gr8_codes[q] == ref_codes[q])) for q in range(NUM_CODEBOOKS)]

        gr8_codes_t = torch.from_numpy(gr8_codes).to(device=device, dtype=torch.long).unsqueeze(0)
        gr8_enc_dec_f32 = model.from_indices(gr8_codes_t)[0, 0].float().cpu().numpy()
        roundtrip_snr_db = compute_snr_db(ref_dec_f32, gr8_enc_dec_f32)

        return {
            "clip": name,
            "samples": int(pcm_f32.shape[0]),
            "frames": F,
            "decode_snr_db": decode_snr_db,
            "stream_snr_db": stream_snr_db,
            "stream_max_lsb_diff": max_lsb_diff,
            "encode_exact_match": exact_match,
            "encode_per_cb_match": per_cb_match,
            "roundtrip_snr_db": roundtrip_snr_db,
        }


def check_gate_e2(results: list[dict[str, Any]]) -> dict[str, Any]:
    min_decode_snr = min(r["decode_snr_db"] for r in results)
    mean_decode_snr = float(np.mean([r["decode_snr_db"] for r in results]))
    mean_stream_snr = float(np.mean([r["stream_snr_db"] for r in results]))
    max_lsb_diff = max(r["stream_max_lsb_diff"] for r in results)
    total_frames = sum(r["frames"] for r in results)
    weighted_exact = sum(r["encode_exact_match"] * r["frames"] for r in results) / max(1, total_frames)
    min_roundtrip_snr = min(r["roundtrip_snr_db"] for r in results)
    mean_roundtrip_snr = float(np.mean([r["roundtrip_snr_db"] for r in results]))

    checks = {
        "decode_snr_ge_45db": min_decode_snr >= MIN_DECODE_SNR_DB,
        "encode_exact_match_ge_0_98": weighted_exact >= MIN_ENCODE_EXACT_MATCH,
        "roundtrip_snr_ge_25db": min_roundtrip_snr >= MIN_ROUNDTRIP_SNR_DB,
        "streaming_within_1_lsb": max_lsb_diff <= MAX_STREAMING_LSB_DIFF,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": {
            "n_clips": len(results),
            "total_frames": total_frames,
            "min_decode_snr_db": min_decode_snr,
            "mean_decode_snr_db": mean_decode_snr,
            "mean_stream_snr_db": mean_stream_snr,
            "max_stream_lsb_diff": max_lsb_diff,
            "encode_exact_match": weighted_exact,
            "min_roundtrip_snr_db": min_roundtrip_snr,
            "mean_roundtrip_snr_db": mean_roundtrip_snr,
        },
        "clips": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate E2 Drama 3 native CUDA codec parity check.")
    ap.add_argument("--bench", required=True, help="Path to g6tts_bench binary")
    ap.add_argument("--codec-safetensors", required=True, help="Path to exlr8-codec.safetensors")
    ap.add_argument("--codec-pth", required=True, help="Path to upstream codec.pth")
    ap.add_argument("--fish-speech", required=True, help="Path to fish-speech checkout @ 0f8a12e8")
    ap.add_argument("--clips-dir", default=None, help="Optional directory of .wav clips (fills to --n-clips with synthetic)")
    ap.add_argument("--n-clips", type=int, default=20, help="Number of clips to verify (default: 20)")
    ap.add_argument("--strides", default="1,1,2,4", help="Streaming decode stride schedule (default: 1,1,2,4)")
    ap.add_argument("--device", default="cuda:0", help="PyTorch reference device (default: cuda:0)")
    ap.add_argument("--json", dest="json_out", default=None, help="Optional path to write JSON report")
    args = ap.parse_args(argv)

    clips = load_clips(Path(args.clips_dir) if args.clips_dir else None, args.n_clips)
    model = load_fish_dac(Path(args.fish_speech), Path(args.codec_pth), args.device)

    results: list[dict[str, Any]] = []
    for name, pcm_f32 in clips:
        r = evaluate_clip(
            name=name,
            pcm_f32=pcm_f32,
            model=model,
            bench_bin=args.bench,
            codec_st=args.codec_safetensors,
            device=args.device,
            strides=args.strides,
        )
        results.append(r)
        print(
            f"[{name}] F={r['frames']:3d}  dec_snr={r['decode_snr_db']:6.2f} dB  "
            f"stream_lsb={r['stream_max_lsb_diff']}  "
            f"enc_match={r['encode_exact_match']:.4f}  rt_snr={r['roundtrip_snr_db']:6.2f} dB",
            flush=True,
        )

    report = check_gate_e2(results)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n")

    s = report["summary"]
    print(
        f"\nGate E2 summary ({s['n_clips']} clips, {s['total_frames']} frames):\n"
        f"  decode SNR:      min={s['min_decode_snr_db']:.2f} dB, mean={s['mean_decode_snr_db']:.2f} dB (req >= {MIN_DECODE_SNR_DB:.1f})\n"
        f"  streaming LSB:   max={s['max_stream_lsb_diff']} (req <= {MAX_STREAMING_LSB_DIFF})\n"
        f"  encode match:    {s['encode_exact_match']:.4f} (req >= {MIN_ENCODE_EXACT_MATCH:.2f})\n"
        f"  round-trip SNR:  min={s['min_roundtrip_snr_db']:.2f} dB, mean={s['mean_roundtrip_snr_db']:.2f} dB (req >= {MIN_ROUNDTRIP_SNR_DB:.1f})\n"
        f"  verdict:         {'PASS' if report['passed'] else 'FAIL'}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
