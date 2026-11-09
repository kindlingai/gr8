#!/usr/bin/env python3
"""make_prompts.py TOKENIZER_JSON: build prompts/p128.json and prompts/p1200.json (token ids, BOS id 0 first)
from prompts/source.txt. p1200 exceeds the index top-k (512) at both compressed ratios (600 and 1200 positions)."""
import json, os, sys
from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(sys.argv[1])
BOS = 0
def ids_of(fn):
    return tok.encode(open(os.path.join(here, "prompts", fn), encoding="utf-8").read(), add_special_tokens=False).ids


for name, n, src in (("p128", 128, None), ("p1200", 1200, None), ("p128code", 128, "source_code.txt")):
    ids = ids_of(src) if src else ids_of("source.txt")
    assert len(ids) >= n - 1, (len(ids), n)
    out = [BOS] + ids[: n - 1]
    json.dump({"name": name, "n_tokens": len(out), "ids": out}, open(os.path.join(here, "prompts", name + ".json"), "w"))
    print(name, len(out), out[:8])
