"""Three boot presets. Numbers are copied from the named docs, not measured here.

Every preset removes explicit SESS, CAP, and MAX_MODEL_LEN. Recipe knobs.sh
applies those, and VARIANT, only when unset (${KEY:-default}); a leftover
would keep the old slot count or pool, so the card text would be false.
DeepSeek and GLM-5.3-Flash also write TUNE and remove VARIANT. Unset TUNE
on those models is Speed, the recipe default. DeepSeek also removes EXACT
and the fast words knobs.sh writes ahead of the user's EXTRA_ENV, because
that file appends EXTRA_ENV last and a leftover wins. GR8_DS_ENGRAM* stays.
GLM-5.3-Flash removes GR8_CP from EXTRA_ENV for the same reason. GLM-5.3
has no TUNE. Speed clears VARIANT and KV_FORMAT. Quality sets
VARIANT=bit325 and clears KV_FORMAT. Context sets KV_FORMAT=int4 and
clears VARIANT. Any explicit SESS, CAP, MAX_MODEL_LEN, or one of those
recipe leftovers reads as custom. A preset that changes VARIANT (unset means
home) is checked against each box's verified weight marker before restart.
"""

MODELS = {"ds41": "DeepSeek-V4.1-Flash", "glm53": "GLM-5.3", "glm53f": "GLM-5.3-Flash"}

# knobs.sh lines 28-40 write these, then append the user's EXTRA_ENV, so the
# user's copy wins. GR8_DS_MHC_Y_MXFP8 is not in that fast= string; a leftover
# is still appended and would change the run, so a preset removes it too.
# GR8_DS_ENGRAM and GR8_DS_ENGRAM_PREFIX are not in this list.
DS41_EXTRA = (
    "GR8_DS_RELAXED_ACCEPT",
    "GR8_DS_MOE_TOPK_DECODE",
    "GR8_DS41_HEAD_MX",
    "GR8_DS_ATTN_TC_MF",
    "GR8_DS_ATTN_TC_TILE",
    "GR8_DS_MHC_ROW_SHARD",
    "GR8_DS_MHC_Y_MXFP8",
    "GR8_DS_CACHE_MIB",
    "GR8_DS_SESS_DRAFTS",
)
FLASH_EXTRA = ("GR8_CP",)


def _tune(value):
    # None removes an explicit override so the recipe default can apply.
    return {"TUNE": value, "VARIANT": None, "KV_FORMAT": None, "SESS": None, "CAP": None, "MAX_MODEL_LEN": None}


def _glm(variant, kv_format):
    return {"TUNE": None, "VARIANT": variant, "KV_FORMAT": kv_format, "SESS": None, "CAP": None, "MAX_MODEL_LEN": None}


def _ds(value):
    row = _tune(value)
    row["EXACT"] = None
    for key in DS41_EXTRA:
        row[key] = None
    return row


def _flash(value):
    row = _tune(value)
    row["GR8_CP"] = None
    return row


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
             "32 slots · ctx 327,670", _ds("speed")),
        _row("quality", "Quality", "TUNE=quality",
             "The exact model's output. 5 chats, each up to 262,134 tokens.",
             "5 slots · ctx 262,134", _ds("quality")),
        _row("context", "Context", "TUNE=context",
             "Fast mode, one chat up to 1,048,566 tokens. 16 chats.",
             "16 slots · ctx 1,048,566", _ds("context")),
    ],
    "glm53": [
        _row("speed", "Speed", "launcher defaults (VARIANT home)",
             "Experts home, 3.25 bpw: the 64 experts per layer with the highest K3 error at K4, the rest K3. DFlash2 or MTP drafts, 8 at once, pool 485,376.",
             "8 slots · pool 485,376", _glm(None, None)),
        _row("quality", "Quality", "VARIANT=bit325",
             "Experts bit325, same 3.25 bpw size, bits placed differently: down projections at K4 in the layers where bits lower KL most, the rest K3. KL 1.5-7% lower than home, decode about 2.5% slower.",
             "8 slots · same size as home", _glm("bit325", None)),
        _row("context", "Context", "KV_FORMAT=int4",
             "Experimental 4-bit KV: about 1.26x the context, decode about 3% slower.",
             "8 slots · pool 513,536 (int4 not re-measured)", _glm(None, "int4")),
    ],
    "glm53f": [
        _row("speed", "Speed", "TUNE=speed",
             "bit349 weights, 8 at once, pool 2,097,152, requests up to 262,144 tokens.",
             "8 slots · ctx 262,144", _flash("speed")),
        _row("quality", "Quality", "TUNE=quality",
             "K4 weights, 4 at once, pool 262,208, requests up to 262,144 tokens.",
             "4 slots · ctx 262,144", _flash("quality")),
        _row("context", "Context", "TUNE=context",
             "Speed plus context-parallel KV, requests up to 1,048,576 tokens.",
             "8 slots · ctx 1,048,576", _flash("context")),
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


def keys_for(model):
    return tuple(CATALOG[model][0]["assign"])


def engine_keys(model):
    if model == "ds41":
        return frozenset(DS41_EXTRA)
    if model == "glm53f":
        return frozenset(FLASH_EXTRA)
    return frozenset()


def classify(model, values):
    got = {key: _norm(values.get(key)) for key in keys_for(model)}
    # knobs.sh defaults an absent TUNE to speed. Overrides still count.
    if model in ("ds41", "glm53f") and got.get("TUNE") is None:
        got["TUNE"] = "speed"
    for item in CATALOG[model]:
        if got == item["assign"]:
            return item["id"]
    return "custom"


def changes_for(model, preset_id):
    for item in CATALOG[model]:
        if item["id"] == preset_id:
            return dict(item["assign"])
    raise ValueError("unknown preset")


def diff_for(model, values, mixed, preset_id):
    rows = []
    for key, after in changes_for(model, preset_id).items():
        before = "mixed" if key in mixed else _norm(values.get(key))
        if before != after:
            rows.append({"key": key, "before": before, "after": after})
    return rows


def describe(model, values, mixed, current):
    if mixed:
        return "mixed across boxes"
    if current in ("speed", "quality", "context"):
        return next(item["setting"] for item in CATALOG[model] if item["id"] == current)
    parts = [f"{key}={values[key]}" for key in keys_for(model) if _norm(values.get(key)) is not None]
    return " ".join(parts) if parts else "unset"


def public(model, values=None, mixed=()):
    values = values or {}
    return [dict(id=item["id"], label=item["label"], setting=item["setting"], summary=item["summary"], caption=item["caption"],
                 diff=diff_for(model, values, mixed, item["id"]))
            for item in CATALOG[model]]
