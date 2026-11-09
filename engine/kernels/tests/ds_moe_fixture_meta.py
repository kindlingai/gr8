#!/usr/bin/env python3
"""Write OUT_DIR/summary.json (dtype + shape per tensor) for the dumps `ds_shared_test fixture` made, so ds41_compare.py reads them right.
  ds_moe_fixture_meta.py OUT_DIR S LAYER [LAYER ...]"""
import json
import sys

out, S, layers = sys.argv[1], int(sys.argv[2]), [int(a) for a in sys.argv[3:]]
t = {}
for L in layers:
    t[f"L{L}.router_idx"] = {"dtype": "int32", "shape": [S, 6]}
    t[f"L{L}.router_w"] = {"dtype": "float32", "shape": [S, 6]}
    t[f"L{L}.moe_routed"] = {"dtype": "float32", "shape": [S, 5120]}
    t[f"L{L}.shared_out"] = {"dtype": "float32", "shape": [S, 5120]}
    t[f"L{L}.ffn_out"] = {"dtype": "bfloat16", "shape": [1, S, 5120]}
    t[f"L{L}.ffn_out_bf16partials"] = {"dtype": "bfloat16", "shape": [1, S, 5120]}
json.dump({"tensors": t}, open(f"{out}/summary.json", "w"), indent=1)
