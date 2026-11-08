"""Small, explicit boot-time surface; defaults remain owned by the launcher."""

MODELS = {"ds41": "DeepSeek-V4.1-Flash", "glm53": "GLM-5.3", "glm53f": "GLM-5.3-Flash"}

# None means remove the override, NOT an invented model default.
KNOBS = {
    "MTP": ("Draft tokens", "speed", "int", "Draft count at boot.", (0, 7)),
    "DFLASH": ("Drafter", "speed", "bool", "Requires prepared drafter weights; unavailable on DeepSeek.", None),
    "SPEC_ADAPT": ("Adaptive drafts", "speed", "bool", "Choose draft count from cost and acceptance.", None),
    "REUSE": ("Prefix reuse", "speed", "bool", "Reuse an earlier prompt prefix.", None),
    "SESS": ("Chat slots", "capacity", "int", "Concurrent decoding sessions; graph memory grows with slots.", (1, 32)),
    "CAP": ("KV pool", "capacity", "int", "Shared cache capacity in tokens; must fit available memory.", (1, 2**31 - 1)),
    "MAX_MODEL_LEN": ("Longest request", "capacity", "int", "Prompt + answer limit; the engine also limits it to available KV.", (1, 1048576)),
    "ALLOW_THINKING_OFF": ("Thinking off", "capacity", "bool", "Experimental: let requests disable reasoning. Output may degrade.", None),
    "GR8_GEN_CACHE": ("Generated tensor cache", "fast switching", "path", "SCREEN: GLM and DeepSeek only. First boot writes a build-specific disk cache.", None),
    "KV_STORE": ("Persistent KV store", "fast switching", "store", "SCREEN: GLM only. SSD persist=on,keep=sessions; needs stop grace and disk space.", None),
    "GR8_STOP_GRACE": ("Stop grace", "fast switching", "duration", "Compose stop allowance; persistent KV uses at least 150s.", None),
    "GR8_LOAD_COMPACT": ("Compact expert copies", "fast switching", "path", "SCREEN: GLM TP4 / Flash TP2. Prepare copies separately before enabling.", None),
    "GR8_LOAD_MERGE_KB": ("Compact read merge", "fast switching", "merge", "SCREEN: set 1 with prepared compact copies.", None),
    "GR8_QUIT_FAST": ("Fast quit", "fast switching", "bool", "SCREEN: GLM TP4 only. Exit after orderly KV save and quit barrier.", None),
}
ENGINE_KEYS = {key for key in KNOBS if key.startswith("GR8_")} - {"GR8_STOP_GRACE"}
FAST_KEYS = {
    "ds41": {"GR8_GEN_CACHE"},
    "glm53": {"GR8_GEN_CACHE", "KV_STORE", "GR8_STOP_GRACE", "GR8_LOAD_COMPACT", "GR8_LOAD_MERGE_KB", "GR8_QUIT_FAST"},
    "glm53f": {"GR8_LOAD_COMPACT", "GR8_LOAD_MERGE_KB"},
}


def supported(key, model, width):
    if key in ("DFLASH", "SPEC_ADAPT") and model == "ds41":
        return False  # engine_args.sh fixes these to zero
    if key == "DFLASH" and model == "glm53" and width not in (4, 6):
        return False
    if KNOBS[key][1] != "fast switching":
        return True
    if key not in FAST_KEYS[model]:
        return False
    if key in ("GR8_LOAD_COMPACT", "GR8_LOAD_MERGE_KB", "GR8_QUIT_FAST", "KV_STORE", "GR8_STOP_GRACE"):
        return width == (2 if model == "glm53f" else 4)
    return True


def validate(changes, model, width):
    import re
    if not isinstance(changes, dict) or len(changes) > len(KNOBS):
        raise ValueError("changes must be a knob map")
    for key, value in changes.items():
        if key not in KNOBS or not supported(key, model, width):
            raise ValueError(f"unsupported knob: {key}")
        if value is None:
            continue
        if not isinstance(value, str) or len(value) > 1024 or any(c in value for c in "\n\r\x00$`\"'\\"):
            raise ValueError(f"invalid literal value: {key}")
        kind, bounds = KNOBS[key][2], KNOBS[key][4]
        good = False
        if kind in ("bool", "merge"):
            good = value in ("0", "1")
        elif kind == "int":
            good = value.isascii() and value.isdecimal() and bounds[0] <= int(value) <= bounds[1]
            if key == "MTP" and model == "ds41":
                good = good and int(value) <= 5
            if key == "SESS" and model != "ds41":
                good = good and int(value) <= 8
        elif kind == "path":
            good = value.startswith("/") and not re.search(r"\s", value)
        elif kind == "duration":
            good = bool(re.fullmatch(r"[1-9][0-9]*s", value))
        elif kind == "store":
            good = value.startswith("ssd:/") and "persist=on" in value.split(",")[1:] and "keep=sessions" in value.split(",")[1:] and not re.search(r"\s", value)
        if not good:
            raise ValueError(f"invalid value for {key}")


def catalog(model, width):
    return [dict(key=k, label=v[0], group=v[1], kind=v[2], description=v[3],
                 bounds=([1, 8] if k == "SESS" and model != "ds41" else [0, 5] if k == "MTP" and model == "ds41" else v[4]))
            for k, v in KNOBS.items() if supported(k, model, width)]
