#!/usr/bin/env python3
"""Synthetic operands on source-derived shapes, NOT captured decision rows."""
import argparse
import json


def cases(suite="all", late_final=False):
    result = {}

    def add(n, pos, ratio, floor=0, wl=128, keys=None, weight=None):
        keys = keys or max(32768, pos + n)
        ident = f"n{n}_p{pos}_r{ratio}_f{floor}_w{wl}_k{keys}"
        row = result.setdefault(ident, {"id": ident, "args": [n, pos, ratio, floor, wl, keys],
                                       "weights": {"32K": 0, "128K": 0},
                                       "input_provenance": "synthetic_source_derived_shape",
                                       "topk_pattern": "uniform_without_replacement",
                                       "operands": "synthetic", "decision_rows": False,
                                       "geometry": {"ring_rows": 8319, "slot_count": 1,
                                           "dispatch_phase": "late_replay" if floor else "main" if n == 1 else "prefill_v2",
                                           "key_capacity_source": "synthetic_not_boot_manifest",
                                           "configuration": "single_slot_rowcap8192_not_captured"}})
        if weight:
            row["weights"][weight[0]] += weight[1]

    if suite in ("all", "real"):
        for length, label in [(32768, "32K"), (131072, "128K")]:
            for pos in range(0, length - 1, 8192):
                n = min(8192, length - 1 - pos)
                for ratio, count in [(0, 2), (2, 18), (1, 1)]:
                    add(n, pos, ratio, weight=(label, count))
                if not late_final or length - 1 - (pos + n) < 256:
                    add(128, pos + n - 128, 1, pos + n - 128, weight=(label, 19))
            for ratio, count in [(0, 2), (2, 18), (1, 20)]:
                add(1, length - 1, ratio, weight=(label, count))
    if suite in ("all", "repro"):
        for ratio in (2, 1, 0):
            add(8192, 24576, ratio, keys=32768)
    if suite in ("all", "smoke"):
        add(16, 0, 2)
        add(16, 24576, 1)
        add(16, 24576, 0)
        add(128, 32639, 1, 32639)
        for wl in (37, 100):
            for ratio in (2, 1):
                add(16, 24576, ratio, wl=wl)
    if suite == "locality":
        # Explicit opt-in: adapter must implement the extra pattern argument.
        for pattern in ("uniform", "shared", "local"):
            for ratio in (1, 2):
                args = [128, 24576, ratio, 0, 128, 32768]
                ident = f"locality_{pattern}_r{ratio}"
                result[ident] = dict(id=ident, args=args + ["--topk-pattern", pattern],
                    weights={"32K": 0, "128K": 0},
                    input_provenance="synthetic_source_derived_shape", operands="synthetic",
                    topk_pattern=pattern, decision_rows=False)
    return list(result.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=("all", "real", "repro", "smoke", "locality"), default="all")
    p.add_argument("--late-final", action="store_true")
    a = p.parse_args()
    print(json.dumps(cases(a.suite, a.late_final), indent=2))
