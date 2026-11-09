"""Whether a preset's variant weights are already on each box.

tools/launch/hfsource.py writes complete/tp<N>.rank<R>.<variant>-<sha>.json
last, after that rank's files are verified. A manifest alone is not weights
on disk. Unset GR8_CACHE and GR8_HOME are the Docker named volumes from
docker/compose.yaml (project name gr8). A path that cannot be resolved, a
probe that cannot run, or a cache that exists but this user cannot read is
unknown, not missing. This module is also the script sent over SSH: it uses the
standard library only.
"""
import json
import math
import os
import re
import stat
import subprocess
import sys

# Wait ceiling when no fetch log has a rate. Not a measured rate and not shown.
UNMEASURED_RATE_MB_S = 1
# docker/compose.yaml pins `name: gr8`. Unset GR8_CACHE/GR8_HOME select volumes gr8_cache and gr8_home.
COMPOSE_PROJECT = "gr8"
RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*MB/s")
VOLUME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
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


def _span(count, unit):
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def health_sentence(seconds):
    """Dialog clause for the health wait the restart will use."""
    total = max(0, int(math.ceil(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(_span(hours, "hour"))
    if minutes:
        parts.append(_span(minutes, "minute"))
    if secs or not parts:
        parts.append(_span(secs if parts else total, "second"))
    return "The health wait is extended to " + " ".join(parts) + "."


def download_text(size_gb, rate_mb_s, health_seconds=None):
    """Dialog sentences. Minutes only when a fetch log supplied a rate.

    The size counts every expert half on the busiest rank, so it is an upper
    bound: halves the current variant already fetched are not fetched again.
    """
    if size_gb is None or size_gb <= 0:
        sentence = "This preset downloads the variant weights to each box that does not have them."
    else:
        gb = max(1, int(round(size_gb)))
        sentence = f"This preset downloads up to about {gb} GB to each box."
        if rate_mb_s is not None and rate_mb_s > 0:
            minutes = max(1, int(round(size_gb * 1000.0 / rate_mb_s / 60.0)))
            sentence += f" At the last measured rate this takes about {minutes} minutes."
    if health_seconds is not None:
        sentence += " " + health_sentence(health_seconds)
    return sentence


def unchecked_text(boxes):
    """Name the boxes whose weight check did not run."""
    numbers = []
    for box in boxes or []:
        try:
            number = int(box)
        except (TypeError, ValueError):
            continue
        if number > 0:
            numbers.append(number)
    if not numbers:
        return "Cannot check the weights on every box."
    labels = [f"box {number}" for number in numbers]
    if len(labels) == 1:
        return f"Cannot check the weights on {labels[0]}."
    if len(labels) == 2:
        return f"Cannot check the weights on {labels[0]} and {labels[1]}."
    return "Cannot check the weights on " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def extra_seconds(size_gb, rate_mb_s):
    """Seconds to add to the stock health wait so a download can finish."""
    if size_gb is None or size_gb <= 0:
        return 3600
    rate = rate_mb_s if rate_mb_s and rate_mb_s > 0 else UNMEASURED_RATE_MB_S
    return max(1, math.ceil(size_gb * 1e9 / (rate * 1e6)))


def preset_notice(status, health_budget):
    """Dialog text and the extended health wait, or (None, None) when every box has the marker.

    Unknown is not a download: no size claim and no extended wait. A confirmed
    miss includes the wait, which can be hours when no fetch rate was measured.
    """
    state = status.get("state")
    if state not in ("present", "missing", "unknown"):
        state = "present" if status.get("present") is True else "missing"
    if state == "present":
        return None, None
    if state == "unknown":
        return unchecked_text(status.get("unchecked")), None
    health = float(health_budget) + extra_seconds(status.get("size_gb"), status.get("rate_mb_s"))
    return download_text(status.get("size_gb"), status.get("rate_mb_s"), health), health


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


def volume_mountpoint(name):
    """Mountpoint of one Docker named volume, or None when it cannot be inspected."""
    if not VOLUME.fullmatch(name or ""):
        return None
    try:
        out = subprocess.check_output(
            ["docker", "volume", "inspect", name],
            text=True, timeout=10, stderr=subprocess.DEVNULL)
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    point = data[0].get("Mountpoint") or ""
    if os.path.isabs(point):
        return point
    return None


def resolve_dir(value, short_name):
    """Absolute host path, or the named volume's mountpoint.

    docker/compose.yaml binds ${GR8_CACHE:-cache} and ${GR8_HOME:-home}. An
    empty value or a volume short name is <project>_<name>. None means the
    directory could not be resolved, which the caller reports as unknown.
    """
    value = (value or "").strip()
    if os.path.isabs(value):
        return value
    short = value or short_name
    if not VOLUME.fullmatch(short):
        return None
    return volume_mountpoint(COMPOSE_PROJECT + "_" + short)


class Unreadable(Exception):
    """A path exists but this user cannot read it, so what it holds is not known."""


def _absent(exc):
    return isinstance(exc, (FileNotFoundError, NotADirectoryError))


def check_readable(path):
    """True when path can be read and traversed, False when it is absent.

    Raises Unreadable on a permission or other read error: a Docker volume mountpoint
    under /var/lib/docker (mode 710 root) resolves for a docker-group user but does not read.
    """
    try:
        os.stat(path)
    except OSError as exc:
        if _absent(exc):
            return False
        raise Unreadable(path) from exc
    if not os.access(path, os.R_OK | os.X_OK):
        raise Unreadable(path)
    return True


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
        except OSError as exc:
            if _absent(exc):
                return None
            raise Unreadable(root) from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    return os.path.join(root, commit)


def _listed_files_match(release, listed):
    for rel, size in listed.items():
        try:
            info = os.stat(os.path.join(release, rel))
        except OSError as exc:
            if _absent(exc):
                return False
            raise Unreadable(rel) from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            return False
    return True


def weights_ready(release, tp, rank, variant):
    pattern = re.compile(
        rf"^tp{int(tp)}\.rank{int(rank)}\.{re.escape(variant)}-[0-9a-f]{{12}}(?:\.layout-[0-9a-f]{{12}})?\.json$"
    )
    try:
        names = os.listdir(os.path.join(release, "complete"))
    except OSError as exc:
        if _absent(exc):
            return False
        raise Unreadable(release) from exc
    denied = None
    for name in names:
        if not pattern.fullmatch(name):
            continue
        try:
            with open(os.path.join(release, "complete", name)) as handle:
                listed = json.load(handle).get("files") or {}
            if listed and _listed_files_match(release, listed):
                return True
        except Unreadable as exc:
            denied = exc
        except ValueError:
            continue
        except OSError as exc:
            if not _absent(exc):
                denied = Unreadable(name)
                denied.__cause__ = exc
    if denied:
        raise denied
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


def _state(row):
    state = row.get("state")
    if state in ("present", "missing", "unknown"):
        return state
    if row.get("present") is True:
        return "present"
    return "unknown" if row.get("present") is None else "missing"


def probe(request):
    variant = request.get("variant") or ""
    if not re.fullmatch(r"[A-Za-z0-9_]+", variant):
        raise ValueError("unsupported variant")
    tp = int(request.get("tp") or 0)
    rank = box_rank(request.get("peers") or [], request.get("node_ip") or "", int(request.get("rank") or 0))
    cache = resolve_dir(request.get("cache") or "", "cache")
    home = resolve_dir(request.get("home") or "", "home")
    unknown = {"state": "unknown", "present": False, "size_gb": None, "rate_mb_s": None}
    if not cache or not MODEL_URL.fullmatch(request.get("model") or ""):
        return unknown
    try:
        # Absent is missing. Present but unreadable is unknown, never a download claim.
        check_readable(cache)
        release = release_dir(cache, request.get("model") or "")
        present = bool(release and tp > 0 and rank >= 0 and weights_ready(release, tp, rank, variant))
    except Unreadable:
        return unknown
    size = None
    if release and tp > 0:
        size = manifest_gb(os.path.join(release, "manifests", f"exlr8-manifest-{variant}.json"), tp)
    return {
        "state": "present" if present else "missing",
        "present": present,
        "size_gb": size,
        "rate_mb_s": measured_rate(home) if home else None,
    }


def aggregate(rows):
    states = [_state(row) for row in rows]
    if not rows or any(state == "unknown" for state in states):
        overall = "unknown"
    elif all(state == "present" for state in states):
        overall = "present"
    else:
        overall = "missing"
    sizes = [row["size_gb"] for row, state in zip(rows, states) if state != "unknown" and row.get("size_gb")]
    rates = [row["rate_mb_s"] for row, state in zip(rows, states) if state != "unknown" and row.get("rate_mb_s")]
    unchecked = []
    for row, state in zip(rows, states):
        if state != "unknown" or not row.get("box"):
            continue
        try:
            unchecked.append(int(row["box"]))
        except (TypeError, ValueError):
            continue
    return {
        "state": overall,
        "present": overall == "present",
        "size_gb": max(sizes) if sizes else None,
        "rate_mb_s": min(rates) if rates else None,
        "unchecked": unchecked,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(probe(json.load(sys.stdin))))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
