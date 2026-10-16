#!/usr/bin/env python3
"""Parse G6_DEC_EVPROF stderr lines (# rank R decode stage profile ...) from an 8-layer stepA TP2 run
and emit the per-stage 8-layer table + 45-layer scaled table for Lane KB and Lane KA."""
import argparse
import json
import re
from pathlib import Path

# Scale factors from 8-layer stepA (L0..L7: 6 KDA, 2 DSA, 3 Dense MLP, 5 MoE MLP) to 45-layer full model
# (34 KDA, 11 DSA, 3 Dense MLP, 42 MoE MLP, 45 mHC/all-reduce layers, 1 head/embed).
STAGE_META = {
    # KDA stages (6 layers in stepA -> 34 in full model)
    "kda proj":           ("KB", "KDA", 6, 34, "KDA q/k/v + f_a/g_a/b projections"),
    "kda prep":           ("KB", "KDA", 6, 34, "KDA conv1d + silu + f_b/g_b gates"),
    "kda step":           ("KB", "KDA", 6, 34, "KDA FP32 state recurrence"),
    "kda out":            ("KB", "KDA", 6, 34, "KDA o_norm + sigmoid gate + frag/fp16"),
    "kda oproj":          ("KB", "KDA", 6, 34, "KDA o_proj shard GEMV"),
    # DSA stages (2 layers in stepA -> 11 in full model)
    "q_a+kv_a":           ("KB", "DSA", 2, 11, "DSA q_a + kv_a + idx_k + idx_w + idx_g + q_a_norm"),
    "idx_k+idx_w":        ("KB", "DSA", 2, 11, "DSA idx_k + idx_w"),
    "q_b+idx_q":          ("KB", "DSA", 2, 11, "DSA xf_qa + q_b + idx.wq_b"),
    "attn append":        ("KB", "DSA", 2, 11, "DSA KV cache + kpool append"),
    "attn index scores":  ("KB", "DSA", 2, 11, "DSA indexer kpool + top-k scores"),
    "attn select/reuse":  ("KB", "DSA", 2, 11, "DSA index page top-k select"),
    "attn absorb":        ("KB", "DSA", 2, 11, "DSA W_UK absorb into q_nope"),
    "attn mla scores":    ("KB", "DSA", 2, 11, "DSA MLA attention scores"),
    "attn normalize":     ("KB", "DSA", 2, 11, "DSA softmax normalize"),
    "attn context":       ("KB", "DSA", 2, 11, "DSA MLA context accumulate"),
    "attn merge":         ("KB", "DSA", 2, 11, "DSA split-K merge"),
    "attn value project": ("KB", "DSA", 2, 11, "DSA W_UV value projection"),
    "attention":          ("KB", "DSA", 2, 11, "DSA attention tail"),
    "dsa oproj":          ("KB", "DSA", 2, 11, "DSA xf_attn + o_proj shard GEMV"),
    # Dense MLP stages (3 layers in stepA -> 3 in full model)
    "dense gate":         ("KB", "DenseMLP", 3, 3, "Dense MLP L0-2 gate + up + SwiGLU"),
    "dense down":         ("KB", "DenseMLP", 3, 3, "Dense MLP L0-2 down projection"),
    "dense out":          ("KB", "DenseMLP", 3, 3, "Dense MLP L0-2 BF16 cast"),
    # MoE MLP stages (5 layers in stepA -> 42 in full model)
    "moe router":         ("KA/KB", "MoE-Router", 5, 42, "MoE router GEMV (FP32 x @ BF16 router.w)"),
    "moe plan":           ("KA/KB", "MoE-Router", 5, 42, "MoE k_router_plan top-8 + slot packing"),
    "moe xhalf":          ("KA", "MoE", 5, 42, "MoE k_to_half (FP32 -> FP16)"),
    "moe xfrag":          ("KA", "MoE", 5, 42, "MoE k_xfrag B-operand fragment table"),
    "k1 fc1 (fused)":     ("KA", "MoE", 5, 42, "MoE K1 gate+up (routed + shared trellis)"),
    "k1 fc2":             ("KA", "MoE", 5, 42, "MoE K1 down (routed + shared trellis)"),
    "k1 fin":             ("KA", "MoE", 5, 42, "MoE K1 finalize weighted sum -> BF16 partial"),
    "experts":            ("KA", "MoE", 5, 42, "MoE block tail"),
    # Fabric + mHC stages (8 layers in stepA -> 45 in full model)
    "o_proj+put":         ("KB", "Fabric", 8, 45, "Publish attention partial to fabric slot (2i)"),
    "wait(attn)":         ("KB", "Fabric", 8, 45, "Wait for peer attention partials (2i)"),
    "put(mlp)":           ("KB", "Fabric", 8, 45, "Publish MLP partial to fabric slot (2i+1)"),
    "wait(mlp)":          ("KB", "Fabric", 8, 45, "Wait for peer MLP partials (2i+1)"),
    "mhc out_inv":        ("KB", "mHC", 8, 45, "mHC peer reduce + post/comb stream update + flat RMS inv (2x/layer)"),
    "reduce+norm(attn)":  ("KB", "mHC", 8, 45, "mHC FFN input mix + coeffs + collapse_norm + fragments"),
    "reduce+norm(mlp)":   ("KB", "mHC", 8, 45, "mHC next-attn input mix + coeffs + collapse_norm + fragments (or final norm)"),
    "other":              ("KB", "Other", 8, 45, "Inter-stage launch / mark overhead"),
    # Head + Step (1 in stepA -> 1 in full model)
    "head local":         ("KB", "Head", 1, 1, "Final norm fragments + lm_head GEMV + local top-1/max"),
    "head merge":         ("KB", "Head", 1, 1, "TP head merge across ranks"),
    "step in/out+embed":  ("KB", "Head", 1, 1, "k_step_in + k_mhc_embed + L0 input mHC + k_step_out"),
}

LINE_RE = re.compile(
    r"# rank (\d+) decode stage profile \(([^)]+)\), M=(\d+), (\d+) steps, ms per step:(.*) step ([0-9.]+)"
)
ITEM_RE = re.compile(r"\s*([^,]+?) ([0-9.]+) \(([0-9.]+)%\),")


def parse_log(text):
    records = []
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        rank, how, M, steps, body, step_ms = (
            int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5), float(m.group(6))
        )
        stages = {}
        for im in ITEM_RE.finditer(body):
            stages[im.group(1).strip()] = float(im.group(2))
        records.append({
            "rank": rank, "how": how, "M": M, "steps": steps, "step_ms": step_ms, "stages": stages, "raw": line
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", required=True, help="rank 0 stderr log path")
    ap.add_argument("--r1", help="rank 1 stderr log path")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    r0_recs = [r for r in parse_log(Path(args.r0).read_text()) if r["M"] == 1]
    r1_recs = [r for r in parse_log(Path(args.r1).read_text()) if r["M"] == 1] if args.r1 and Path(args.r1).exists() else []
    if not r0_recs:
        raise SystemExit(f"No M=1 profile lines found in {args.r0}")
    rec0 = r0_recs[-1]
    rec1 = r1_recs[-1] if r1_recs else None

    rows = []
    tot_8 = 0.0
    tot_45 = 0.0
    by_group_8 = {}
    by_group_45 = {}
    for name, ms8 in rec0["stages"].items():
        if name == "gap between main steps":
            continue
        lane, grp, n8, n45, desc = STAGE_META.get(name, ("KB", "Other", 8, 45, name))
        scale = n45 / n8
        ms45 = ms8 * scale
        ms8_r1 = rec1["stages"].get(name, 0.0) if rec1 else None
        tot_8 += ms8
        tot_45 += ms45
        by_group_8[grp] = by_group_8.get(grp, 0.0) + ms8
        by_group_45[grp] = by_group_45.get(grp, 0.0) + ms45
        rows.append({
            "stage": name, "lane": lane, "group": grp, "n8": n8, "n45": n45, "scale": scale,
            "ms_8L_r0": ms8, "ms_8L_r1": ms8_r1, "ms_45L_est": ms45,
            "per_layer_us": (ms8 * 1000.0) / n8, "desc": desc
        })

    out_data = {
        "r0_last": rec0,
        "r1_last": rec1,
        "tot_8L_ms": tot_8,
        "tot_45L_est_ms": tot_45,
        "by_group_8L_ms": by_group_8,
        "by_group_45L_est_ms": by_group_45,
        "rows": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out_data, indent=2) + "\n")

    md = []
    md.append("# GLM-5.3-Flash TP2 Decode Stage Profile (`G6_DEC_EVPROF`, `--graph`, `M=1`)\n")
    md.append(f"- **Source run (rank 0)**: `{rec0['steps']}` steps, 8-layer `stepA` total = **{tot_8:.3f} ms/step**")
    if rec1:
        md.append(f"- **Source run (rank 1)**: `{rec1['steps']}` steps, 8-layer `stepA` total = **{rec1['step_ms']:.3f} ms/step**")
    md.append(f"- **Scaled 45-layer full-model total**: **{tot_45:.3f} ms/step** (vs `fs128` 45-layer wall p50 `54.94 ms/step`)\n")

    md.append("## Summary by Functional Group\n")
    md.append("| Group | Lane | 8L `stepA` (ms/step) | 8L Share | 45L Scaled (ms/step) | 45L Share |")
    md.append("|---|---|---:|---:|---:|---:|")
    for grp, ms45 in sorted(by_group_45.items(), key=lambda kv: -kv[1]):
        ms8 = by_group_8[grp]
        lane = "KA" if grp == "MoE" else ("KA/KB" if grp == "MoE-Router" else "KB")
        md.append(f"| **{grp}** | {lane} | {ms8:.3f} | {100*ms8/tot_8:.1f}% | **{ms45:.3f}** | **{100*ms45/tot_45:.1f}%** |")
    md.append(f"| **TOTAL** | **ALL** | **{tot_8:.3f}** | **100.0%** | **{tot_45:.3f}** | **100.0%** |\n")

    md.append("## Per-Stage Breakdown\n")
    md.append("| Stage | Lane | Group | 8L Count | 45L Count | 8L r0 (ms) | 8L r1 (ms) | Per-Layer (us) | 45L Scaled (ms) | 45L % | Description |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda x: -x["ms_45L_est"]):
        r1s = f"{r['ms_8L_r1']:.3f}" if r["ms_8L_r1"] is not None else "-"
        md.append(
            f"| `{r['stage']}` | {r['lane']} | {r['group']} | {r['n8']} | {r['n45']} | "
            f"{r['ms_8L_r0']:.3f} | {r1s} | {r['per_layer_us']:.1f} | **{r['ms_45L_est']:.3f}** | "
            f"{100*r['ms_45L_est']/tot_45:.1f}% | {r['desc']} |"
        )
    md.append("")
    Path(args.out_md).write_text("\n".join(md))
    print(f"Wrote {args.out_json} and {args.out_md} (8L={tot_8:.3f} ms, 45L={tot_45:.3f} ms)")


if __name__ == "__main__":
    main()
