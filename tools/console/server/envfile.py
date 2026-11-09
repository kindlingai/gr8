"""Literal dotenv edits. This module also runs over SSH with a JSON stdin request."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import tempfile


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse(text):
    values = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=(.*)$", line)
        if not match:
            continue
        key, raw = match.groups()
        if key in values:
            raise ValueError(f"duplicate dotenv key: {key}")
        words = shlex.split(raw, comments=True)
        # Accept unquoted multi-word EXTRA_ENV used by the launcher and Compose.
        values[key] = " ".join(words)
    return values


def engine_env(values):
    result = {}
    for token in shlex.split(values.get("EXTRA_ENV", "")):
        key, sep, value = token.partition("=")
        if not sep or key in result:
            raise ValueError("ambiguous EXTRA_ENV; resolve it before editing")
        result[key] = value
    return result


def read_values(text, keys, engine_keys):
    direct = parse(text)
    extra = engine_env(direct)
    return {key: (extra if key in engine_keys else direct).get(key) for key in keys}


def patch(text, changes, engine_keys):
    direct = parse(text)
    extra = engine_env(direct)
    updates = {k: v for k, v in changes.items() if k not in engine_keys}
    extra_changed = False
    for key, value in changes.items():
        if key not in engine_keys:
            continue
        if value is None:
            if key in extra:
                extra.pop(key)
                extra_changed = True
        elif extra.get(key) != value:
            extra[key] = value
            extra_changed = True
    # Leave EXTRA_ENV byte-for-byte alone when none of its watched words changed.
    if extra_changed:
        updates["EXTRA_ENV"] = shlex.join(f"{k}={v}" for k, v in extra.items()) or None
    result, seen = [], set()
    for line in text.splitlines(keepends=True):
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=", line)
        key = match[1] if match else None
        if key not in updates:
            result.append(line)
        else:
            seen.add(key)
            if updates[key] is not None:
                result.append(f"{key}={shlex.quote(updates[key])}\n")
    for key, value in updates.items():
        if key not in seen and value is not None:
            if result and not result[-1].endswith("\n"):
                result[-1] += "\n"
            result.append(f"{key}={shlex.quote(value)}\n")
    return "".join(result)


def transact(path, request):
    path = Path(path)
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("dotenv must be a regular file, not a symlink")
    if st.st_size > 262144:
        raise ValueError("dotenv is too large")
    text = path.read_bytes().decode()
    revision = digest(text)
    if request["action"] == "read":
        return {"text": text, "revision": revision}
    if request.get("expected") != revision:
        raise ValueError("dotenv changed since review; refresh before retrying")
    replacement = request["text"]
    if not isinstance(replacement, str) or len(replacement.encode()) > 262144:
        raise ValueError("invalid replacement")
    fd, name = tempfile.mkstemp(prefix=".console-env-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            os.fchmod(file.fileno(), stat.S_IMODE(st.st_mode))
            os.fchown(file.fileno(), st.st_uid, st.st_gid)
            file.write(replacement)
            file.flush()
            os.fsync(file.fileno())
        # Recheck immediately before replace, including changes during the write.
        if digest(path.read_bytes().decode()) != revision:
            raise ValueError("dotenv changed during write")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {"revision": digest(replacement)}


if __name__ == "__main__":
    try:
        req = json.load(sys.stdin)
        print(json.dumps(transact(req["path"], req)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
