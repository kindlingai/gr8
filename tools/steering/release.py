#!/usr/bin/env python3
"""Pack and unpack a GR8 schema-v1 steering profile as one deterministic safetensors file.

The payload has one F32 tensor, ``directions``. The exact UTF-8 profile manifest is stored in
``__metadata__.gr8_manifest``.  This tool intentionally uses only the Python standard library so the launcher can
materialize a downloaded profile before any model dependencies are installed.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile

FORMAT = 'gr8-steering-profile-v1'
MAX_HEADER = 1 << 20
MAX_PAYLOAD = 64 << 20


class ProfileError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ProfileError(f'duplicate JSON key {key}')
        out[key] = value
    return out


def parse_json(raw, where):
    def number(token):
        value = float(token)
        if not math.isfinite(value):
            raise ProfileError(f'invalid {where}: non-finite number')
        return value

    def constant(token):
        raise ProfileError(f'invalid {where}: non-finite number {token}')

    def check(value, depth=0):
        if depth > 64:
            raise ProfileError(f'invalid {where}: nested too deeply')
        if isinstance(value, str):
            value.encode('utf-8')  # reject lone escaped surrogates, as the C++ reader does
        elif isinstance(value, dict):
            for key, item in value.items():
                key.encode('utf-8')
                check(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                check(item, depth + 1)

    try:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')  # JSON here never accepts UTF-16/32 or a BOM
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_float=number, parse_constant=constant)
        check(value)
        return value
    except ProfileError:
        raise
    except (UnicodeError, ValueError, RecursionError) as e:
        raise ProfileError(f'invalid {where}: {e}') from None


def read_limited(path, limit, what):
    with open(path, 'rb') as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ProfileError(f'{what} is too large')
    return raw


def _object(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ProfileError(f'invalid {where}')


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _u32(value):
    return _integer(value) and 0 <= value <= 0xffffffff


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_manifest(manifest, payload, expected=None):
    _object(manifest, ('schema_version', 'profile_name', 'model', 'scope', 'layers', 'alphas', 'directions',
                       'provenance'), 'manifest fields')
    if not _integer(manifest.get('schema_version')) or manifest['schema_version'] != 1:
        raise ProfileError('want a schema-v1 steering manifest')
    name = manifest.get('profile_name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', name):
        raise ProfileError('invalid profile_name')
    layers, alphas = manifest.get('layers'), manifest.get('alphas')
    model, directions = manifest.get('model'), manifest.get('directions')
    if not isinstance(layers, list) or not layers or not isinstance(alphas, list) or len(layers) != len(alphas):
        raise ProfileError('inconsistent layers and alphas')
    _object(model, ('name', 'revision', 'weight_set', 'tokenizer_revision', 'chat_template_sha256', 'hidden_size',
                    'layer_count'), 'model fields')
    for field in ('name', 'revision', 'weight_set', 'tokenizer_revision'):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ProfileError(f'invalid model.{field}')
    if not _digest(model['weight_set']):
        raise ProfileError('invalid model.weight_set')
    if not _u32(model.get('hidden_size')) or model['hidden_size'] == 0 or \
            not _u32(model.get('layer_count')) or model['layer_count'] == 0:
        raise ProfileError('invalid model dimensions')
    if not _digest(model.get('chat_template_sha256')):
        raise ProfileError('invalid model.chat_template_sha256')
    if expected is not None:
        _object(expected, model.keys(), 'expected model identity fields')
        if any(type(expected[key]) is not type(value) or expected[key] != value for key, value in model.items()):
            raise ProfileError('model identity mismatch')
    last = -1
    for layer in layers:
        if not _integer(layer) or layer <= last or layer >= model['layer_count']:
            raise ProfileError('layers must be unique, ordered, and in range')
        last = layer
    for alpha in alphas:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 4.0 or \
                not math.isfinite(alpha):
            raise ProfileError('alpha must be finite and in [0, 4]')
    if manifest.get('scope') not in ('shared', 'full'):
        raise ProfileError('unknown scope')
    want = len(layers) * model['hidden_size'] * 4
    _object(directions, ('file', 'dtype', 'byte_count', 'sha256'), 'directions record')
    if directions.get('file') != 'directions.f32' or \
            directions.get('dtype') != 'float32-le' or not _u32(directions.get('byte_count')) or \
            directions['byte_count'] != want:
        raise ProfileError('invalid directions record')
    if len(payload) != want or len(payload) > MAX_PAYLOAD:
        raise ProfileError('directions payload size mismatch')
    if directions.get('sha256') != sha256(payload):
        raise ProfileError('directions payload digest mismatch')
    provenance = manifest.get('provenance')
    if not isinstance(provenance, dict) or not set(provenance).issubset({
            'extraction_tool', 'extraction_commit', 'dataset_manifests', 'capture_manifest_sha256', 'fit_sha256',
            'family', 'direction_layer'}) or \
            not {'extraction_tool', 'extraction_commit', 'dataset_manifests'} <= set(provenance):
        raise ProfileError('invalid provenance fields')
    if not isinstance(provenance['extraction_tool'], str) or not provenance['extraction_tool'] or \
            not isinstance(provenance['extraction_commit'], str) or not provenance['extraction_commit']:
        raise ProfileError('missing extraction provenance')
    datasets = provenance['dataset_manifests']
    _object(datasets, ('train', 'validation', 'final'), 'provenance.dataset_manifests')
    for field, value in datasets.items():
        if not _digest(value):
            raise ProfileError(f'invalid provenance.dataset_manifests.{field}')
    for field in ('capture_manifest_sha256', 'fit_sha256'):
        if field in provenance and not _digest(provenance[field]):
            raise ProfileError(f'invalid provenance.{field}')
    if provenance.get('family') not in (None, 'global-band', 'per-layer'):
        raise ProfileError('unknown provenance.family')
    direction_layer = provenance.get('direction_layer')
    if direction_layer is not None and \
            (not _integer(direction_layer) or not 0 <= direction_layer < model['layer_count']):
        raise ProfileError('invalid provenance.direction_layer')
    values = struct.iter_unpack('<f', payload)
    width = model['hidden_size']
    norm2 = 0.0
    for i, (value,) in enumerate(values, 1):
        if not math.isfinite(value):
            raise ProfileError('non-finite direction')
        norm2 += value * value
        if i % width == 0:
            norm = math.sqrt(norm2)
            if not 0.999 <= norm <= 1.001:
                raise ProfileError('direction norm outside [0.999, 1.001]')
            norm2 = 0.0
    return name, [len(layers), model['hidden_size']]


def validate_directory(profile_dir, expected=None):
    """Validate the host contract against an optional complete caller-provided model identity."""
    manifest_path = os.path.join(profile_dir, 'manifest.json')
    payload_path = os.path.join(profile_dir, 'directions.f32')
    try:
        manifest = parse_json(read_limited(manifest_path, MAX_HEADER, 'manifest'), 'manifest JSON')
        payload = read_limited(payload_path, MAX_PAYLOAD, 'directions payload')
    except OSError as e:
        raise ProfileError(str(e)) from None
    validate_manifest(manifest, payload, expected)
    return manifest


def _atomic_write(path, chunks):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.steering-', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            for chunk in chunks:
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def pack(profile_dir, output):
    """Pack exact input bytes deterministically, atomically replacing the output artifact."""
    manifest_raw = read_limited(os.path.join(profile_dir, 'manifest.json'), MAX_HEADER, 'manifest')
    manifest = parse_json(manifest_raw, 'manifest JSON')
    payload = read_limited(os.path.join(profile_dir, 'directions.f32'), MAX_PAYLOAD, 'directions payload')
    _, shape = validate_manifest(manifest, payload)
    manifest_json = manifest_raw.decode('utf-8')
    header = {
        '__metadata__': {'format': FORMAT, 'gr8_manifest': manifest_json,
                         'manifest_sha256': sha256(manifest_raw)},
        'directions': {'dtype': 'F32', 'shape': shape, 'data_offsets': [0, len(payload)]},
    }
    raw_header = canonical(header).encode()
    raw_header += b' ' * ((8 - len(raw_header) % 8) % 8)
    if len(raw_header) > MAX_HEADER:
        raise ProfileError('safetensors header is too large')
    _atomic_write(output, (struct.pack('<Q', len(raw_header)), raw_header, payload))


def _read(path):
    with open(path, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        raw = f.read(8)
        if len(raw) != 8:
            raise ProfileError('truncated safetensors header length')
        n, = struct.unpack('<Q', raw)
        if n <= 0 or n % 8 or n > MAX_HEADER or 8 + n > size:
            raise ProfileError('invalid safetensors header length')
        header = parse_json(f.read(n), 'safetensors header')
        _object(header, ('__metadata__', 'directions'), 'safetensors fields; want exactly one directions tensor')
        meta, tensor = header['__metadata__'], header['directions']
        _object(meta, ('format', 'gr8_manifest', 'manifest_sha256'), 'GR8 metadata fields')
        if meta.get('format') != FORMAT or not isinstance(meta.get('gr8_manifest'), str):
            raise ProfileError('missing GR8 steering metadata')
        if not isinstance(tensor, dict) or set(tensor) != {'dtype', 'shape', 'data_offsets'} or tensor['dtype'] != 'F32':
            raise ProfileError('directions must be one F32 tensor')
        offsets = tensor['data_offsets']
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(_integer(x) for x in offsets) or \
                offsets != [0, size - 8 - n] or not 0 <= offsets[1] <= MAX_PAYLOAD:
            raise ProfileError('invalid directions data offsets')
        payload = f.read(MAX_PAYLOAD + 1)
        if len(payload) != offsets[1]:
            raise ProfileError('directions payload size mismatch')
    manifest = parse_json(meta['gr8_manifest'], 'embedded manifest')
    manifest_raw = meta['gr8_manifest'].encode('utf-8')
    if len(manifest_raw) > MAX_HEADER or meta.get('manifest_sha256') != sha256(manifest_raw):
        raise ProfileError('embedded manifest is too large or has the wrong digest')
    _, shape = validate_manifest(manifest, payload)
    if not isinstance(tensor['shape'], list) or not all(_integer(x) for x in tensor['shape']) or tensor['shape'] != shape:
        raise ProfileError('directions tensor shape mismatch')
    return manifest, payload, manifest_raw


def read(path):
    """Return the validated manifest mapping and directions bytes from an artifact."""
    manifest, payload, _ = _read(path)
    return manifest, payload


def unpack(path, output_dir):
    """Publish an exact profile directory; reuse identical contents, reject a conflicting destination."""
    manifest, payload, manifest_raw = _read(path)
    output_dir = os.path.abspath(output_dir)
    if os.path.islink(output_dir):
        raise ProfileError('output directory must not be a symlink')
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if read_limited(os.path.join(output_dir, 'manifest.json'), MAX_HEADER, 'manifest') == manifest_raw and \
                read_limited(os.path.join(output_dir, 'directions.f32'), MAX_PAYLOAD, 'directions payload') == payload:
            return manifest
        raise ProfileError('output directory contains a different profile')
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    stage = tempfile.mkdtemp(prefix='.steering-', dir=parent)
    try:
        _atomic_write(os.path.join(stage, 'manifest.json'), (manifest_raw,))
        _atomic_write(os.path.join(stage, 'directions.f32'), (payload,))
        os.replace(stage, output_dir)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack'); p.add_argument('profile_dir'); p.add_argument('output')
    p = sub.add_parser('unpack'); p.add_argument('input'); p.add_argument('output_dir')
    args = ap.parse_args(argv)
    try:
        if args.command == 'pack':
            pack(args.profile_dir, args.output)
        else:
            unpack(args.input, args.output_dir)
    except (OSError, ProfileError) as e:
        ap.exit(2, f'{ap.prog}: {e}\n')


if __name__ == '__main__':
    main()
