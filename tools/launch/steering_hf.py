"""Explicit steering artifacts from a pinned HF release; no work on stock boots."""
import fcntl
import json
import os
import re
import tempfile

import modelset
import steering_local

release = steering_local.release
INDEX = 'steering/index.json'
MAX_ARTIFACT = 8 + release.MAX_HEADER + release.MAX_PAYLOAD
IDENTITY = {'STEERING_MODEL_REVISION': 'revision', 'STEERING_TOKENIZER_REVISION': 'tokenizer_revision',
            'STEERING_CHAT_TEMPLATE_SHA256': 'chat_template_sha256', 'STEERING_WEIGHT_SET': 'weight_set'}


def request(o):
    """Remember the explicit selector/constraints before resolution replaces them with local paths/identities."""
    if not hasattr(o, '_hf_steering'):
        value = o.get('STEERING_PROFILE') or ''
        o._hf_steering = (value[3:], {k: o.get(k) for k in IDENTITY}) if value.startswith('hf:') else None
    return o._hf_steering


def parse_index(raw):
    index = release.parse_json(raw, 'steering index JSON')
    if not isinstance(index, dict) or set(index) != {'schema_version', 'default_profile', 'profiles'} or \
            type(index['schema_version']) is not int or index['schema_version'] != 1 or \
            not isinstance(index['profiles'], list):
        raise ValueError('want schema_version 1, default_profile and profiles')
    records = {}
    for rec in index['profiles']:
        if not isinstance(rec, dict) or set(rec) != {'name', 'file', 'sha256', 'byte_count'}:
            raise ValueError('invalid profile record')
        name = rec['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', name) or name in records:
            raise ValueError('invalid or duplicate profile name')
        if rec['file'] != f'steering/{name}.safetensors' or not isinstance(rec['sha256'], str) or \
                not re.fullmatch(r'[0-9a-f]{64}', rec['sha256']) or type(rec['byte_count']) is not int or \
                not 8 < rec['byte_count'] <= MAX_ARTIFACT:
            raise ValueError(f'invalid profile file record for {name}')
        records[name] = rec
    default = index['default_profile']
    if not isinstance(default, str) or default not in records:
        raise ValueError('default_profile is not listed')
    return default, records


def _save(path, value):
    fd, tmp = tempfile.mkstemp(prefix='.index-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _index(source):
    path = os.path.join(source.dir, INDEX)
    receipt = os.path.join(source.dir, 'steering', '.index-identity.json')
    if os.path.isfile(path) and os.path.isfile(receipt):
        # These small integrity records are scoped to the resolved commit. Recheck cached bytes on every restart;
        # a complete cache does not need the hub's tree API. Like the model cache, this is not a trust store.
        ident = release.parse_json(release.read_limited(receipt, 4096, 'index identity'), 'index identity')
        if not isinstance(ident, dict) or set(ident) != {'commit', 'sha256', 'byte_count'} or \
                ident['commit'] != source.commit or not isinstance(ident['sha256'], str) or \
                not re.fullmatch(r'[0-9a-f]{64}', ident['sha256']) or type(ident['byte_count']) is not int or \
                not 0 < ident['byte_count'] <= release.MAX_HEADER:
            raise ValueError('invalid cached index identity')
        raw = release.read_limited(path, release.MAX_HEADER, 'steering index')
        if len(raw) != ident['byte_count'] or release.sha256(raw) != ident['sha256']:
            raise ValueError('cached steering index size or digest mismatch')
    else:
        tree = source.repo.tree()
        if INDEX not in tree:
            raise ValueError(f'needs {INDEX}')
        if not 0 < tree[INDEX] <= release.MAX_HEADER:
            raise ValueError('steering index is too large or empty')
        source.repo.fetch(INDEX, path)
        # Repo.fetch reuses existing files; verify those too before recording their identity.
        if not source.repo.matches(INDEX, path):
            raise ValueError('steering index differs from the pinned release')
        raw = release.read_limited(path, release.MAX_HEADER, 'steering index')
        parse_index(raw)
        _save(receipt, {'commit': source.commit, 'sha256': release.sha256(raw), 'byte_count': len(raw)})
    return parse_index(raw)


def fetch(source, requested):
    """Verify and materialize one selected artifact, reusing only identical cached bytes."""
    try:
        directory = os.path.join(source.dir, 'steering')
        os.makedirs(directory, exist_ok=True)
        # Serialize fetch's shared .part files and publication across processes using this commit's cache.
        with open(os.path.join(directory, '.fetch.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            default, records = _index(source)
            name = default if requested == 'default' else requested
            if name not in records:
                raise ValueError(f'no steering profile named {name!r}')
            rec = records[name]
            artifact = os.path.join(source.dir, rec['file'])
            if not os.path.isfile(artifact):
                if source.repo.tree().get(rec['file']) != rec['byte_count']:
                    raise ValueError('profile file is missing or its published size differs from the index')
                source.repo.fetch(rec['file'], artifact)
            if os.path.getsize(artifact) != rec['byte_count'] or modelset.sha256_file(artifact) != rec['sha256']:
                raise ValueError('profile index size or digest mismatch')
            manifest, _ = release.read(artifact)
            if manifest['profile_name'] != name:
                raise ValueError('embedded profile name differs from the index')
            out = os.path.join(directory, 'profiles', name)
            release.unpack(artifact, out)
            return out, manifest
    except (OSError, ValueError, RuntimeError) as e:
        raise modelset.ModelError(f'{source}: --steering-profile hf:{requested}: {e}') from None


def bind(o, source):
    """Derive profile assertions; steering_local.prepare checks them against the actual resolved model next."""
    selected = request(o)
    if not selected:
        return
    if source is None or o.get('BASE_PACKS'):
        raise modelset.ModelError('--steering-profile hf:... requires an HF release without --base-packs')
    directory, manifest = source.steering_profile
    for key, field in IDENTITY.items():
        value = manifest['model'][field]
        if selected[1][key] and selected[1][key] != value:
            raise modelset.ModelError(f'--steering-profile hf:{selected[0]}: explicit {key} differs from the profile')
        o.values[key] = value
    o.values['STEERING_PROFILE'] = directory
    source.log(f'{source}: steering profile {manifest["profile_name"]} explicitly selected')
