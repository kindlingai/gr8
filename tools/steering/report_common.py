"""Strict JSONL input and atomic JSON output for the offline report tools."""
import hashlib
import json
import math
import os
import tempfile


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'duplicate JSON key {key!r}')
        out[key] = value
    return out


def _number(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError('non-finite JSON number')
    return value


def _constant(token):
    raise ValueError(f'invalid JSON constant {token}')


def read_rows(path):
    """Return objects and the SHA-256 of the exact bytes parsed, including blank lines."""
    result, digest = [], hashlib.sha256()
    with open(path, 'rb') as f:
        for line, raw in enumerate(f, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode('utf-8'), object_pairs_hook=_object,
                                 parse_float=_number, parse_constant=_constant)
                if not isinstance(row, dict):
                    raise ValueError('expected an object')
                result.append(row)
            except (ValueError, UnicodeError, RecursionError) as e:
                raise ValueError(f'{path}: line {line}: {e}') from None
    if not result:
        raise ValueError(f'{path}: no records')
    return result, digest.hexdigest()


def write_report(path, report):
    """Publish a complete report without clobbering an existing output on failure."""
    fd, temporary = tempfile.mkstemp(prefix='.report-', dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
