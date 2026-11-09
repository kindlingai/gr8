#!/usr/bin/env python3
"""fc2 case list for the sweep bench. SYNTHETIC routing and values on the served v2 shapes; no prompt weights.

Case args: [M] plus option pairs --experts N (6..384, default 384), --pattern uniform|skew, --holes 0|1.
`repro` is the served full-width call: M=8192 tokens, 384 experts, uniform top-6 (the ncu campaign's fc2 shape).
"""
import argparse
import json

PROVENANCE = {"input_provenance": "synthetic", "decision_rows": False, "weights": {}}


def case(m, pattern="uniform", holes=0, experts=384, note=""):
    ident = f"fc2_m{m}_{pattern}" + ("_holes" if holes else "") + ("" if experts == 384 else f"_e{experts}")
    args = [m]
    if experts != 384:
        args += ["--experts", experts]
    args += ["--pattern", pattern]
    if holes:
        args += ["--holes", 1]
    row = {"id": ident, "args": args, **{k: (dict(v) if isinstance(v, dict) else v) for k, v in PROVENANCE.items()}}
    row.update(M=m, pattern=pattern, holes=bool(holes), experts=experts, note=note)
    return row


def cases(suite="all"):
    repro = [case(8192, note="served full-width call, 4 column-group launches")]
    if suite == "repro":
        return repro
    if suite == "smoke":   # tiny: fits a busy 5090 (about 0.1 GB per fixture)
        return [case(256, experts=64), case(200, "skew", 1, 64), case(8, experts=16)]
    if suite == "all":
        return repro + [case(8192, "skew", note="power-law expert popularity: long expert tails, many tiles per slot"),
                        case(8191, note="odd M: last tile of every expert partly live"),
                        case(8192, holes=1, note="cut routes (idx -1): masked rows inside tiles"),
                        case(2048), case(128)]
    raise ValueError(f"unknown suite {suite}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=("all", "repro", "smoke"), default="all")
    p.add_argument("--late-final", action="store_true", help="accepted for driver compatibility; ignored")
    a = p.parse_args()
    print(json.dumps(cases(a.suite), indent=2))
