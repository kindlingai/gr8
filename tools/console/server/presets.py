"""Three boot presets. Numbers are copied from the named docs, not measured here.

DeepSeek and GLM-5.3-Flash write only TUNE. Their recipe knobs.sh expands it.
GLM-5.3 on gr8 main has no TUNE: Speed clears VARIANT and KV_FORMAT (launcher
defaults), Quality sets VARIANT=bit325, Context sets KV_FORMAT=int4.
"""

MODELS = {"ds41": "DeepSeek-V4.1-Flash", "glm53": "GLM-5.3", "glm53f": "GLM-5.3-Flash"}
KEYS = ("TUNE", "VARIANT", "KV_FORMAT")


def _tune(value):
    return {"TUNE": value, "VARIANT": None, "KV_FORMAT": None}


def _row(preset_id, label, setting, summary, caption, assign):
    return dict(id=preset_id, label=label, setting=setting, summary=summary, caption=caption, assign=assign)


# DS41: recipes' knobs.sh / README "Pick what to prioritize" (SESS/CAP and the 327,670,
# 262,134, and 1,048,566 token request limits). GLM-5.3: docs/GLM-5.3.md TP=4 table,
# the bit325 variant row, and the --kv-format int4 option. Flash: that recipe's
# knobs.sh and the same README section (bit349 / k4, pools, request limits).
CATALOG = {
    "ds41": [
        _row("speed", "Speed", "TUNE=speed",
             "Fast mode. 32 chats, each up to 327,670 tokens. Replies are not bit-identical to the exact model's.",
             "32 slots · ctx 327,670", _tune("speed")),
        _row("quality", "Quality", "TUNE=quality",
             "The exact model's output. 5 chats, each up to 262,134 tokens.",
             "5 slots · ctx 262,134", _tune("quality")),
        _row("context", "Context", "TUNE=context",
             "Fast mode, one chat up to 1,048,566 tokens. 16 chats.",
             "16 slots · ctx 1,048,566", _tune("context")),
    ],
    "glm53": [
        _row("speed", "Speed", "launcher defaults",
             "Experts home 3.25 bpw, DFlash2 + MTP drafts, 8 at once, KV pool 485,376.",
             "8 slots · pool 485,376", {"TUNE": None, "VARIANT": None, "KV_FORMAT": None}),
        _row("quality", "Quality", "VARIANT=bit325",
             "Experts bit325 3.25 bpw: KL 1.5-7% lower than home, decode about 2.5% slower.",
             "8 slots · pool 485,376", {"TUNE": None, "VARIANT": "bit325", "KV_FORMAT": None}),
        _row("context", "Context", "KV_FORMAT=int4",
             "Experimental 4-bit KV: about 1.26x the context, decode about 3% slower.",
             "8 slots · larger pool", {"TUNE": None, "VARIANT": None, "KV_FORMAT": "int4"}),
    ],
    "glm53f": [
        _row("speed", "Speed", "TUNE=speed",
             "bit349 weights, 8 at once, pool 2,097,152, requests up to 262,144 tokens.",
             "8 slots · ctx 262,144", _tune("speed")),
        _row("quality", "Quality", "TUNE=quality",
             "K4 weights, 4 at once, pool 262,208, requests up to 262,144 tokens.",
             "4 slots · ctx 262,144", _tune("quality")),
        _row("context", "Context", "TUNE=context",
             "Speed plus context-parallel KV, requests up to 1,048,576 tokens.",
             "8 slots · ctx 1,048,576", _tune("context")),
    ],
}
SOURCE = {
    "ds41": "recipe TUNE (knobs.sh)",
    "glm53f": "recipe TUNE (knobs.sh)",
    "glm53": "launcher options (docs/GLM-5.3.md)",
}


def _norm(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    return value or None


def classify(model, values):
    got = {key: _norm(values.get(key)) for key in KEYS}
    for item in CATALOG[model]:
        if got == item["assign"]:
            return item["id"]
    return "custom"


def changes_for(model, preset_id):
    for item in CATALOG[model]:
        if item["id"] == preset_id:
            return dict(item["assign"])
    raise ValueError("unknown preset")


def describe(model, values, mixed, current):
    if mixed:
        return "mixed across boxes"
    if current in ("speed", "quality", "context"):
        return next(item["setting"] for item in CATALOG[model] if item["id"] == current)
    parts = [f"{key}={values[key]}" for key in KEYS if _norm(values.get(key)) is not None]
    return " ".join(parts) if parts else "unset"


def public(model):
    return [dict(id=item["id"], label=item["label"], setting=item["setting"], summary=item["summary"], caption=item["caption"])
            for item in CATALOG[model]]
