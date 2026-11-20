#!/usr/bin/env python3
"""Build the tester zip: one commit's source tree plus the prebuilt arm64 binaries, with a MANIFEST.txt of sha256s.

    scripts/make_zip.py --bin DIR [--ref HEAD] [--out DIR] [--allow-missing]
    scripts/make_zip.py --check [--ref HEAD]

--check reads the tree and stops before the binaries: it fails where a build would fail on the source tree, and
writes nothing. CI runs it on every PR, where the binaries are not built.

DIR is the `dist` export of the Dockerfile (docker buildx build --target dist -o type=local,dest=DIR .): DIR/bin holds
the engine (every TP width) and the front end. The source files come from
`git archive <ref>`, not the working tree, so untracked or modified files never reach the zip. Writes
OUT/gr8-<short sha>.zip and
OUT/gr8-<short sha>.zip.sha256. The zip is reproducible: the same commit and the same binaries give the same bytes
(entries sorted, timestamps set to the commit time, fixed modes).

Every top-level entry of the tree must be in SHIP or SKIP below; a new one stops the build until someone decides.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shipped. The engine sources and their tests stay in: scripts/host_tests.sh and c1_tests.sh build the s2 and runtime
# tests, and tools/convert imports engine/ref/g6ref.py. A tester can rebuild from the zip with the README's own commands.
SHIP = {'README.md', 'CHANGELOG.md', 'LICENSE', 'KNOWN-ISSUES.md', 'PROTOCOL.md', 'RELEASING.md', 'releases', 'bench',
        'config', 'docs', 'engine', 'frontend', 'scripts', 'tools', 'Dockerfile', '.dockerignore', 'docker', 'deploy',
        'gr8', 'KNOBS.md', 'AGENTS.md', 'model.yaml', 'recipes.json', 'smoketest', 'CREDITS.md', 'knobs',
        'rust-toolchain.toml'}
# Not shipped: repo plumbing.
SKIP = {'.github', '.gitignore'}

TAG_RE = re.compile(r'^v[0-9]+(?:\.[0-9]+){1,2}(?:-[0-9A-Za-z.-]+)?$')
DATE_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
SHA40_RE = re.compile(r'^[0-9a-f]{40}$')
DIGEST_RE = re.compile(r'^sha256:[0-9a-f]{64}$')
CHANGELOG_REL_LINE_RE = re.compile(r'^## \[(v[^\]]+)\] — ([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$')
REQUIRED_CHANGELOG_SUBSECTIONS = (
    '### Highlights and measured numbers',
    '### Breaking changes',
    '### Default or cache-key changes',
    '### Toolchain / Dockerfile sync',
)
REQUIRED_SPEC_KEYS = {
    'schema_version', 'version', 'tag', 'previous_tag', 'date', 'commit', 'release_url', 'validation_gate',
    'container', 'toolchain', 'binaries', 'api', 'models', 'breaking', 'breaking_changes',
    'default_or_cache_changes', 'toolchain_changes', 'operational_notes', 'prs', 'benchmarks',
}
REQUIRED_MODEL_KEYS = {
    'id', 'title', 'architecture', 'served_names', 'checkpoint', 'drafter', 'engine_binaries',
    'tp_supported', 'variants', 'default_variant_by_tp', 'int4_kv', 'defaults_by_tp',
}


def _is_int(val):
    return isinstance(val, int) and not isinstance(val, bool)


def check_release_records(raw_files):
    """Validate CHANGELOG.md and releases/*.json when present in raw_files (path -> bytes)."""
    has_changelog = 'CHANGELOG.md' in raw_files
    rel_paths = sorted(p for p in raw_files if p.startswith('releases/'))
    if not has_changelog and not rel_paths:
        return []
    errors = []
    if not has_changelog:
        errors.append('CHANGELOG.md is missing while releases/ is present')
    if 'releases/README.md' not in raw_files:
        errors.append('releases/README.md is missing')
    spec_paths = [p for p in rel_paths if p != 'releases/README.md']
    if not spec_paths:
        errors.append('no releases/v*.json spec files found')

    changelog_sections = {}
    if has_changelog:
        text = raw_files['CHANGELOG.md'].decode('utf-8')
        lines = text.splitlines(keepends=True)
        h2_sections = []
        in_fence = False
        offset = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                in_fence = not in_fence
            elif not in_fence and line.startswith('## '):
                h2_sections.append((offset, offset + len(line), stripped))
            offset += len(line)
        rel_entries = []
        for idx, (start, body_start, heading) in enumerate(h2_sections):
            end = h2_sections[idx + 1][0] if idx + 1 < len(h2_sections) else len(text)
            if heading == '## [Unreleased]':
                continue
            m = CHANGELOG_REL_LINE_RE.match(heading)
            if not m:
                errors.append(
                    f'CHANGELOG.md: malformed H2 heading "{heading}" '
                    '(expected "## [Unreleased]" or "## [vX.Y.Z] — YYYY-MM-DD")'
                )
                continue
            rel_entries.append((m.group(1), m.group(2), text[body_start:end]))
        if not rel_entries:
            errors.append('CHANGELOG.md has no "## [vX.Y.Z] — YYYY-MM-DD" release headings')
        for tag, date, body in rel_entries:
            if not TAG_RE.match(tag):
                errors.append(f'CHANGELOG.md: invalid release tag "{tag}"')
            if tag in changelog_sections:
                errors.append(f'CHANGELOG.md: duplicate section for {tag}')
            changelog_sections[tag] = (date, body)
            for heading in REQUIRED_CHANGELOG_SUBSECTIONS:
                if heading not in body:
                    errors.append(f'CHANGELOG.md ({tag}): missing "{heading}"')

    specs = {}
    for path in spec_paths:
        base = path[len('releases/'):]
        if '/' in base or not base.endswith('.json'):
            errors.append(f'{path}: unexpected file under releases/ (expected releases/<tag>.json)')
            continue
        file_tag = base[:-5]
        if not TAG_RE.match(file_tag):
            errors.append(f'{path}: filename tag "{file_tag}" does not match vMAJOR.MINOR[.PATCH]')
            continue
        try:
            spec = json.loads(raw_files[path].decode('utf-8'))
        except Exception as exc:
            errors.append(f'{path}: invalid JSON: {exc}')
            continue
        if not isinstance(spec, dict):
            errors.append(f'{path}: top-level JSON value must be an object')
            continue
        missing_k = sorted(REQUIRED_SPEC_KEYS - set(spec))
        extra_k = sorted(set(spec) - REQUIRED_SPEC_KEYS)
        if missing_k:
            errors.append(f'{path}: missing keys: {", ".join(missing_k)}')
            continue
        if extra_k:
            errors.append(f'{path}: unknown keys: {", ".join(extra_k)}')
        if not _is_int(spec['schema_version']) or spec['schema_version'] != 1:
            errors.append(f'{path}: schema_version must be integer 1')
        if not isinstance(spec['version'], str) or not spec['version']:
            errors.append(f'{path}: version must be a non-empty string')
        if not isinstance(spec['tag'], str) or spec['tag'] != file_tag:
            errors.append(f'{path}: tag "{spec["tag"]}" does not match filename "{file_tag}"')
        if isinstance(spec['version'], str) and spec['tag'] != f'v{spec["version"]}':
            errors.append(f'{path}: tag "{spec["tag"]}" must equal "v" + version ("{spec["version"]}")')
        prev_tag = spec['previous_tag']
        if prev_tag is not None and (not isinstance(prev_tag, str) or not TAG_RE.match(prev_tag)):
            errors.append(f'{path}: previous_tag must be null or a valid v* tag string')
        if not isinstance(spec['date'], str) or not DATE_RE.match(spec['date']):
            errors.append(f'{path}: invalid date "{spec["date"]}"')
        if not isinstance(spec['commit'], str) or not SHA40_RE.match(spec['commit']):
            errors.append(f'{path}: commit must be a 40-char lowercase hex SHA')
        want_url = f'https://github.com/kindlingai/gr8/releases/tag/{file_tag}'
        if spec['release_url'] != want_url:
            errors.append(f'{path}: release_url must be "{want_url}"')
        if not isinstance(spec['validation_gate'], str) or not spec['validation_gate'].strip():
            errors.append(f'{path}: validation_gate must be a non-empty string')

        c = spec['container']
        if not isinstance(c, dict):
            errors.append(f'{path}: container must be an object')
        else:
            want_img = f'ghcr.io/kindlingai/gr8:{file_tag}'
            if c.get('image') != want_img:
                errors.append(f'{path}: container.image must be "{want_img}"')
            if not isinstance(c.get('digest'), str) or not DIGEST_RE.match(c['digest']):
                errors.append(f'{path}: container.digest must be sha256:<64 hex>')
            elif c.get('pinned') != f'{want_img}@{c["digest"]}':
                errors.append(f'{path}: container.pinned must equal "{want_img}@{c["digest"]}"')
            if not isinstance(c.get('platforms'), list) or not c['platforms']:
                errors.append(f'{path}: container.platforms must be a non-empty list')
            if not isinstance(c.get('target_arch'), str) or not c['target_arch']:
                errors.append(f'{path}: container.target_arch must be a non-empty string')

        tc = spec['toolchain']
        if not isinstance(tc, dict) or not all(isinstance(tc.get(k), str) and tc[k] for k in ('cuda', 'base_os', 'rust')) \
                or not isinstance(tc.get('runtime_apt'), list) or not tc['runtime_apt']:
            errors.append(f'{path}: toolchain must define cuda, base_os, rust, and non-empty runtime_apt')

        if not isinstance(spec['binaries'], list) or not spec['binaries'] or not all(isinstance(b, str) for b in spec['binaries']):
            errors.append(f'{path}: binaries must be a non-empty list of strings')

        api = spec['api']
        if not isinstance(api, dict) or not isinstance(api.get('protocol'), str) \
                or not all(_is_int(api.get(p)) for p in ('http_port', 'control_port', 's2_port')) \
                or not isinstance(api.get('endpoints'), list) or not api['endpoints']:
            errors.append(f'{path}: api must define protocol, integer ports, and non-empty endpoints')

        models = spec['models']
        if not isinstance(models, list) or not models:
            errors.append(f'{path}: models must be a non-empty list')
        else:
            for m in models:
                if not isinstance(m, dict) or set(m) != REQUIRED_MODEL_KEYS:
                    errors.append(f'{path}: model entry keys must match {sorted(REQUIRED_MODEL_KEYS)}')
                    continue
                if not isinstance(m['tp_supported'], list) or not m['tp_supported'] \
                        or not all(_is_int(tp) for tp in m['tp_supported']):
                    errors.append(f'{path} ({m["id"]}): tp_supported must be a non-empty integer list')
                    continue
                tp_keys = {str(tp) for tp in m['tp_supported']}
                if not isinstance(m['defaults_by_tp'], dict) or set(m['defaults_by_tp']) != tp_keys:
                    errors.append(f'{path} ({m["id"]}): defaults_by_tp keys must match tp_supported {sorted(tp_keys)}')

        if not isinstance(spec['breaking'], bool) or not isinstance(spec['breaking_changes'], list):
            errors.append(f'{path}: breaking (bool) and breaking_changes (list) are required')
        elif spec['breaking'] != bool(spec['breaking_changes']):
            errors.append(f'{path}: breaking={spec["breaking"]} disagrees with breaking_changes={spec["breaking_changes"]}')

        for lst_key in ('default_or_cache_changes', 'toolchain_changes', 'operational_notes', 'prs', 'benchmarks'):
            if not isinstance(spec[lst_key], list):
                errors.append(f'{path}: {lst_key} must be a list')

        specs[file_tag] = spec

    if has_changelog and specs:
        only_cl = sorted(set(changelog_sections) - set(specs))
        only_sp = sorted(set(specs) - set(changelog_sections))
        if only_cl:
            errors.append('CHANGELOG.md entries missing releases/<tag>.json: ' + ', '.join(only_cl))
        if only_sp:
            errors.append('releases/<tag>.json missing "## [<tag>]" in CHANGELOG.md: ' + ', '.join(only_sp))
        roots = [t for t, s in specs.items() if s['previous_tag'] is None]
        if len(roots) != 1:
            errors.append(f'expected exactly 1 root release with previous_tag=null, found {roots}')
        seen_prev = {}
        for tag, spec in specs.items():
            prev = spec['previous_tag']
            if isinstance(prev, str):
                if prev not in specs:
                    errors.append(f'releases/{tag}.json: previous_tag "{prev}" has no releases/{prev}.json')
                elif prev in seen_prev:
                    errors.append(f'releases/{tag}.json: duplicate previous_tag "{prev}" (also used by {seen_prev[prev]})')
                else:
                    seen_prev[prev] = tag
                cur = tag
                visited = set()
                while isinstance(cur, str) and cur in specs:
                    if cur in visited:
                        errors.append(f'releases/{tag}.json: cycle detected in previous_tag chain at {cur}')
                        break
                    visited.add(cur)
                    cur = specs[cur]['previous_tag']
            if tag in changelog_sections:
                cl_date, cl_body = changelog_sections[tag]
                if cl_date != spec['date']:
                    errors.append(f'{tag}: CHANGELOG.md date {cl_date} != releases/{tag}.json date {spec["date"]}')
                if isinstance(spec['commit'], str) and spec['commit'] not in cl_body:
                    errors.append(f'{tag}: CHANGELOG.md section does not mention commit {spec["commit"]}')
                digest = spec['container'].get('digest', '') if isinstance(spec['container'], dict) else ''
                if isinstance(digest, str) and digest and digest not in cl_body:
                    errors.append(f'{tag}: CHANGELOG.md section does not mention digest {digest}')
                if f'releases/{tag}.json' not in cl_body:
                    errors.append(f'{tag}: CHANGELOG.md section does not link releases/{tag}.json')
    return errors


def skipped_file(path):
    """Files inside a shipped directory that still stay out: measurements, logs and git plumbing."""
    base = os.path.basename(path)
    return (base == '.gitignore' or base.endswith('.log')
            or (path.startswith('bench/') and base.startswith('results') and base.endswith('.jsonl')))


# Each binary has the same path under --bin and in the zip.
BINARIES = ['bin/g6m4', 'bin/g6m4-int4', 'bin/m4sim', 'bin/gpuprobe', 'bin/tr3_e4m3_test', 'bin/g6front-rs', 'bin/g6load']

HEADER = """gr8 {short} (commit {commit}, {date})

bin/   the engine for every TP width (scripts/build.sh: g6m4 takes the width from --ips), the Rust front end
       g6front-rs, and g6load, which plans and checks the Hugging Face fetch (engine/loader-rs)
Built for GB10 (aarch64, CUDA 13, sm_121a). The binaries link cuBLAS 13 and libibverbs at run time: a node needs the
CUDA 13 runtime libraries (DGX OS has them) and rdma-core. With the prebuilt bin/, skip the engine and front end builds
in README.md: copy it to bin/ on every box, whatever the TP. g6m4-int4 is the same engine with the 4-bit KV latent,
which --kv-format int4 runs (docs/KV4.md).

sha256 of every file below:
"""


def git(*args):
    return subprocess.run(['git', '-C', REPO, *args], check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bin', help='the dist export: DIR/bin')
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--out', default='.')
    ap.add_argument('--allow-missing', action='store_true', help='leave out missing binaries instead of failing')
    ap.add_argument('--check', action='store_true', help='check the tree only: no binaries, no zip')
    a = ap.parse_args()
    if not a.check and not a.bin:
        ap.error('--bin is required without --check')

    commit = git('rev-parse', '--verify', a.ref + '^{commit}').decode().strip()
    short = git('rev-parse', '--short', commit).decode().strip()
    epoch = int(git('show', '-s', '--format=%ct', commit))
    stamp = time.gmtime(max(epoch, 315532800))[:6]   # zip timestamps start in 1980
    top = f'gr8-{short}'

    files = {}   # zip path -> (bytes, mode)
    tar = tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', commit)))
    unknown = set()
    for m in tar.getmembers():
        if not m.isfile():
            if m.issym():
                sys.exit(f'{m.name}: symlinks are not supported in the zip')
            continue
        head = m.name.split('/', 1)[0]
        if head in SKIP or skipped_file(m.name):
            continue
        if head not in SHIP:
            unknown.add(head)
            continue
        files[m.name] = (tar.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
    if unknown:
        sys.exit('top-level entries in neither SHIP nor SKIP (scripts/make_zip.py): ' + ', '.join(sorted(unknown)))
    rel_errors = check_release_records({k: v[0] for k, v in files.items()})
    if rel_errors:
        sys.exit('release record check failed:\n  ' + '\n  '.join(rel_errors))
    if a.check:
        print(f'{short}: {len(files)} source files, every top-level entry shipped or skipped')
        return

    missing = []
    for path in BINARIES:
        p = os.path.join(a.bin, path)
        if not os.path.isfile(p):
            missing.append(path)
            continue
        with open(p, 'rb') as f:
            files[path] = (f.read(), 0o755)   # artifact downloads drop the exec bit
    if missing and not a.allow_missing:
        sys.exit(f'missing binaries under {a.bin}: ' + ', '.join(missing))

    date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(epoch))
    lines = [HEADER.format(short=short, commit=commit, date=date)]
    if missing:
        lines.append('NOT INCLUDED (missing from this build): ' + ', '.join(missing) + '\n\n')
    for path in sorted(files):
        lines.append(f'{hashlib.sha256(files[path][0]).hexdigest()}  {path}\n')
    files['MANIFEST.txt'] = (''.join(lines).encode(), 0o644)

    os.makedirs(a.out, exist_ok=True)
    zpath = os.path.join(a.out, f'{top}.zip')
    with zipfile.ZipFile(zpath, 'w') as z:
        for path in sorted(files):
            data, mode = files[path]
            info = zipfile.ZipInfo(f'{top}/{path}', date_time=stamp)
            info.create_system = 3   # unix, so external_attr carries the mode
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data, compresslevel=9)
    with open(zpath, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(zpath + '.sha256', 'w') as f:
        f.write(f'{digest}  {top}.zip\n')
    print(f'{zpath}: {len(files)} files, {os.path.getsize(zpath) / 1e6:.1f} MB, sha256 {digest}')
    if missing:
        print('missing binaries: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
