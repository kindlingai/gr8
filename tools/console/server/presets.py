"""Three boot presets. Numbers are copied from the named docs, not measured here.

Every preset removes explicit SESS and MAX_MODEL_LEN. Recipe knobs.sh
applies those, and VARIANT, only when unset (${KEY:-default}); a leftover
would keep the old slot count or length, so the card text would be false.
No preset removes CAP. CAP is the box's own limit on the KV pool, set to fit
that ring; a preset that removed it let the pool grow past what the ring
holds, and the engine failed to boot. CAP is not a preset key, so a stock
CAP, or any other CAP value, never makes a box read as custom.
On GLM-5.3 only, a preset that moves a box between the fp8 KV row (Speed,
Quality) and the int4 KV row (Context) rescales an explicit CAP to the same
KV memory (cap_for); an unset CAP stays unset. DeepSeek and GLM-5.3-Flash
presets do not set KV_FORMAT (the recipe picks it), so their CAP stays.
DeepSeek and GLM-5.3-Flash also write TUNE and remove VARIANT. Unset TUNE
on those models is Speed, the recipe default. DeepSeek also removes EXACT
and the fast words knobs.sh writes ahead of the user's EXTRA_ENV, because
that file appends EXTRA_ENV last and a leftover wins. GR8_DS_ENGRAM* stays.
GLM-5.3-Flash removes GR8_CP from EXTRA_ENV for the same reason. GLM-5.3
has no TUNE. Speed clears VARIANT and KV_FORMAT. Quality sets
VARIANT=bit325 and clears KV_FORMAT. Context sets KV_FORMAT=int4 and
clears VARIANT. Any explicit SESS, MAX_MODEL_LEN, or one of those
recipe leftovers reads as custom. A preset that changes VARIANT (unset means
home) is checked against each box's verified weight marker before restart.
An unresolved or unreadable cache, or a failed check, is unknown, not a missing download.
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
    return {"TUNE": value, "VARIANT": None, "KV_FORMAT": None, "SESS": None, "MAX_MODEL_LEN": None}


def _glm(variant, kv_format):
    return {"TUNE": None, "VARIANT": variant, "KV_FORMAT": kv_format, "SESS": None, "MAX_MODEL_LEN": None}


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


# DS41: recipes' knobs.sh / README "Pick what to prioritize" (SESS and the 327,670,
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
             "Experts home, 3.25 bpw: the 64 experts per layer with the highest K3 error at K4, the rest K3. DFlash2 or MTP drafts, 8 at once.",
             "8 slots · launcher defaults", _glm(None, None)),
        _row("quality", "Quality", "VARIANT=bit325",
             "Experts bit325, same 3.25 bpw size, bits placed differently: down projections at K4 in the layers where bits lower KL most, the rest K3. KL 1.5-7% lower than home, decode about 2.5% slower.",
             "8 slots · same size as home", _glm("bit325", None)),
        _row("context", "Context", "KV_FORMAT=int4",
             "Experimental 4-bit KV: about 1.26x the pool when no CAP limits it, decode about 3% slower.",
             "8 slots · int4 KV (not re-measured)", _glm(None, "int4")),
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


# GLM-5.3 CAP pair at TP=4, as docs/GLM-5.3.md:107-108 quotes it: int4 CAP 513,536 against
# fp8 CAP 408,832, about 1.26x. A CAP-sized buffer does not shrink with the row (docs/KV4.md:96),
# so the 656/416 page ratio is not used. The KV4.md:101 table gives a larger ratio
# (310,528 -> 415,232, 1.34x); this one is smaller, so it asks for less memory.
SAME_MEMORY_CAP = {"fp8": 408832, "int4": 513536}
# One page is 64 tokens (scripts/engine_args.sh:18-23, share_cap), and a page adds 64 x TP tokens
# to CAP (docs/KV4.md:96-97, 256 at TP4), so CAP moves in steps of 64 x TP.
PAGE_TOKENS = 64


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


def _pool(cap):
    """Dialog and card text for a CAP change, or ("", "") when there is none."""
    if not cap:
        return "", ""
    if cap["before"] == "mixed":
        return " KV pool: CAP is scaled on each box to hold the same KV memory.", ""
    return (f" KV pool: CAP {int(cap['before']):,} to {int(cap['after']):,} tokens, the same KV memory.",
            f" \u00b7 CAP {int(cap['after']):,}")


def public(model, values=None, mixed=(), caps=None):
    """The cards. caps maps a preset id to its cap_diff, when CAP changes."""
    values, caps = values or {}, caps or {}
    cards = []
    for item in CATALOG[model]:
        diff = diff_for(model, values, mixed, item["id"])
        cap = caps.get(item["id"])
        summary, caption = _pool(cap)
        if cap:
            diff.append({"key": "CAP", **cap})
        cards.append(dict(id=item["id"], label=item["label"], setting=item["setting"], summary=item["summary"] + summary,
                          caption=item["caption"] + caption, diff=diff))
    return cards


def _kv(value):
    """fp8 for unset or fp8, int4 for int4, None for any other value."""
    value = _norm(value)
    return "fp8" if value in (None, "fp8") else "int4" if value == "int4" else None


def _block(env, boxes):
    tp = _norm(env.get("TP"))
    tp = int(tp) if tp and tp.isdigit() and int(tp) > 0 else max(boxes, 1)
    return PAGE_TOKENS * tp


def _grow(cap, block):
    """The int4 CAP that holds the memory of this fp8 CAP, aligned to the box's own CAP."""
    return cap + (cap * SAME_MEMORY_CAP["int4"] // SAME_MEMORY_CAP["fp8"] - cap) // block * block


def _shrink(cap, block):
    """The largest aligned fp8 CAP whose int4 image is not above this CAP; 514,304 gives 409,600."""
    low = cap * SAME_MEMORY_CAP["fp8"] // SAME_MEMORY_CAP["int4"] // block * block
    if low <= 0:
        return None
    while _grow(low + block, block) <= cap:
        low += block
    return low


def cap_for(model, preset_id, env, boxes):
    """The CAP text one box needs for this preset, or None when its CAP stays as it is.
    env holds the box's KV_FORMAT, CAP and TP. Only GLM-5.3 changes CAP, and only when the
    preset moves the box between the fp8 and int4 KV rows and CAP is an explicit integer."""
    if model != "glm53":
        return None
    cap = _norm(env.get("CAP"))
    if not cap or not cap.isdigit() or int(cap) <= 0:
        return None
    source, target = _kv(env.get("KV_FORMAT")), _kv(changes_for(model, preset_id).get("KV_FORMAT"))
    if source is None or target is None or source == target:
        return None
    block = _block(env, boxes)
    new = _grow(int(cap), block) if target == "int4" else _shrink(int(cap), block)
    return str(new) if new else None


def cap_diff(model, preset_id, envs):
    """The CAP change a preset makes on all boxes: {before, after}, the same text on every box;
    {before: "mixed", after: "scaled on each box"} when boxes differ; None when no CAP changes."""
    pairs = [(_norm(env.get("CAP")), cap_for(model, preset_id, env, len(envs))) for env in envs]
    if not any(after for _, after in pairs):
        return None
    if len(set(pairs)) == 1 and pairs[0][1]:
        return {"before": pairs[0][0], "after": pairs[0][1]}
    return {"before": "mixed", "after": "scaled on each box"}
