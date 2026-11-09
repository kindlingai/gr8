"""Whether a preset's variant weights are already on each box.

tools/launch/hfsource.py writes complete/tp<N>.rank<R>.<variant>-<sha>.json
last, after that rank's files are verified. A manifest alone is not weights
on disk. This module is also the script sent over SSH: it uses the standard
library only.
"""
import json
import math
import os
import re
import subprocess
import sys

# Wait ceiling when no fetch log has a rate. Not a measured rate and not shown.
UNMEASURED_RATE_MB_S = 1
RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*MB/s")
MODEL_URL = re.compile(
    r"https://huggingface\.co/([\w.-]+/[\w.-]+?)(?:@([\w.-]+)|/tree/([\w.-]+))?/?"
)


def effective_variant(value):
    if not isinstance(value, str):
        return "home"
    value = value.strip()
    return value or "home"


def changed_variant(values, changes, mixed=()):
    """Variant a preset will boot, when that differs from the one running now.

    Unset VARIANT is home. Mixed boxes are a change: at least one box differs.
    """
    if "VARIANT" not in changes:
        return None
    target = effective_variant(changes.get("VARIANT"))
    if "VARIANT" in mixed or effective_variant(values.get("VARIANT")) != target:
        return target
    return None


def download_text(size_gb, rate_mb_s):
    """Dialog sentence. Minutes only when a fetch log supplied a rate."""
    if size_gb is None or size_gb <= 0:
        return "needs a download of this variant on every box that does not have it"
    gb = max(1, int(round(size_gb)))
    if rate_mb_s is None or rate_mb_s <= 0:
        return f"needs a ~{gb} GB download per box"
    minutes = max(1, int(round(size_gb * 1000.0 / rate_mb_s / 60.0)))
    return f"needs a ~{gb} GB download per box (~{minutes} min)"


def extra_seconds(size_gb, rate_mb_s):
    """Seconds to add to the stock health wait so a download can finish."""
    if size_gb is None or size_gb <= 0:
        return 3600
    rate = rate_mb_s if rate_mb_s and rate_mb_s > 0 else UNMEASURED_RATE_MB_S
    return max(1, math.ceil(size_gb * 1e9 / (rate * 1e6)))


def last_rate(text):
    found = RATE.findall(text or "")
    return float(found[-1]) if found else None


def expert_gb(manifest, tp, hidden):
    """GB of expert weights on the busiest rank. Same split as modelset.expert_gb."""
    layers = manifest["layers"]
    half_bytes = {k: hidden * 16 * k for k in (2, 3, 4)}
    per = [0] * tp
    for key, expert in layers.items():
        layer = int(key)
        shared = expert.get("shared") or 0
        if isinstance(shared, int):
            shared = {"gate": [shared] * 16, "up": [shared] * 16, "down": shared}
        rows = [(index, expert["gate"][index], expert["up"][index], expert["down"][index])
                for index in range(len(expert["down"]))]
        if shared["down"]:
            rows.append((len(expert["down"]), shared["gate"], shared["up"], shared["down"]))
        for index, gate, up, down in rows:
            for half in range(16):
                per[(16 * index + half + layer) % tp] += half_bytes[gate[half]] + half_bytes[up[half]] + half_bytes[down]
    return max(per) / 1e9


def _hidden(manifest, release):
    if manifest.get("hidden_size"):
        return int(manifest["hidden_size"])
    for path in (os.path.join(release, "config.json"), os.path.join(release, "tokenizer", "config.json")):
        try:
            with open(path) as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            continue
        hidden = config.get("hidden_size") or (config.get("text_config") or {}).get("hidden_size")
        if hidden:
            return int(hidden)
    return 6144


def manifest_gb(path, tp):
    try:
        with open(path) as handle:
            manifest = json.load(handle)
        if tp < 1:
            return None
        return expert_gb(manifest, tp, _hidden(manifest, os.path.dirname(os.path.dirname(path))))
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


def release_dir(cache, model):
    match = MODEL_URL.fullmatch(model or "")
    if not match or not cache:
        return None
    repo, rev = match.group(1), match.group(2) or match.group(3) or "main"
    root = os.path.join(cache, "hf", repo.replace("/", "--"))
    if re.fullmatch(r"[0-9a-f]{40}", rev):
        commit = rev
    else:
        try:
            with open(os.path.join(root, "refs", rev)) as handle:
                commit = handle.read().strip()
        except OSError:
            return None
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    return os.path.join(root, commit)


def weights_ready(release, tp, rank, variant):
    pattern = re.compile(
        rf"^tp{int(tp)}\.rank{int(rank)}\.{re.escape(variant)}-[0-9a-f]{{12}}(?:\.layout-[0-9a-f]{{12}})?\.json$"
    )
    try:
        names = os.listdir(os.path.join(release, "complete"))
    except OSError:
        return False
    for name in names:
        if not pattern.fullmatch(name):
            continue
        try:
            with open(os.path.join(release, "complete", name)) as handle:
                listed = json.load(handle).get("files") or {}
            if listed and all(
                os.path.isfile(os.path.join(release, rel)) and os.path.getsize(os.path.join(release, rel)) == size
                for rel, size in listed.items()
            ):
                return True
        except (OSError, ValueError):
            continue
    return False


def box_rank(peers, node_ip, fallback):
    peers = list(peers or [])
    if node_ip and node_ip in peers:
        return peers.index(node_ip)
    if peers:
        ips = set()
        try:
            out = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
            ips.update(out.split())
        except (OSError, subprocess.SubprocessError):
            pass
        for index, peer in enumerate(peers):
            if peer in ips:
                return index
    return int(fallback)


def measured_rate(home):
    if not home or os.path.abspath(home) in ("/", ""):
        return None
    root = os.path.join(home, "out")
    if not os.path.isdir(root):
        root = home
        if not os.path.isdir(root):
            return None
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - root.count(os.sep) > 2:
            dirnames.clear()
            continue
        dirnames[:] = dirnames[:10]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size <= 0 or size > 20_000_000:
                continue
            try:
                found.append((os.path.getmtime(path), path, size))
            except OSError:
                continue
            if len(found) >= 40:
                break
        if len(found) >= 40:
            break
    rate = None
    for _, path, size in sorted(found)[-8:]:
        try:
            with open(path, "rb") as handle:
                if size > 65536:
                    handle.seek(size - 65536)
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            continue
        parsed = last_rate(text)
        if parsed:
            rate = parsed
    return rate


def probe(request):
    variant = request.get("variant") or ""
    if not re.fullmatch(r"[A-Za-z0-9_]+", variant):
        raise ValueError("unsupported variant")
    tp = int(request.get("tp") or 0)
    rank = box_rank(request.get("peers") or [], request.get("node_ip") or "", int(request.get("rank") or 0))
    release = release_dir(request.get("cache") or "", request.get("model") or "")
    present = bool(release and tp > 0 and rank >= 0 and weights_ready(release, tp, rank, variant))
    size = None
    if release and tp > 0:
        size = manifest_gb(os.path.join(release, "manifests", f"exlr8-manifest-{variant}.json"), tp)
    return {"present": present, "size_gb": size, "rate_mb_s": measured_rate(request.get("home") or "")}


def aggregate(rows):
    sizes = [row["size_gb"] for row in rows if row.get("size_gb")]
    rates = [row["rate_mb_s"] for row in rows if row.get("rate_mb_s")]
    return {
        "present": bool(rows) and all(row.get("present") for row in rows),
        "size_gb": max(sizes) if sizes else None,
        "rate_mb_s": min(rates) if rates else None,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(probe(json.load(sys.stdin))))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
