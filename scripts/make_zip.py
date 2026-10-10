#!/usr/bin/env python3
"""Build the tester zip: one commit's source tree plus the prebuilt arm64 binaries, with a MANIFEST.txt of sha256s.

    scripts/make_zip.py --bin DIR [--ref HEAD] [--out DIR] [--allow-missing]

DIR is the `dist` export of the Dockerfile (docker buildx build --target dist -o type=local,dest=DIR .): DIR/bin holds
the TP=4 build and the front end, DIR/bin6 the TP=6 build, DIR/bin3 the TP=3 build. The source files come from
`git archive <ref>`, not the working tree, so untracked or modified files never reach the zip. Writes
OUT/gr8-<short sha>.zip and
OUT/gr8-<short sha>.zip.sha256. The zip is reproducible: the same commit and the same binaries give the same bytes
(entries sorted, timestamps set to the commit time, fixed modes).

Every top-level entry of the tree must be in SHIP or SKIP below; a new one stops the build until someone decides.
"""
import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shipped. The engine sources and their tests stay in: scripts/build.sh compiles engine/tests/test_spec_len.cpp and
# test_prefix_reuse.cpp on every build, scripts/host_tests.sh and c1_tests.sh build the s2 and runtime tests, and
# tools/convert imports engine/ref/g6ref.py. A tester can rebuild from the zip with the README's own commands.
SHIP = {'README.md', 'LICENSE', 'CHANGELOG.md', 'KNOWN-ISSUES.md', 'PROTOCOL.md', 'bench', 'config', 'docs', 'engine',
        'frontend', 'scripts', 'tools', 'Dockerfile', '.dockerignore', 'docker', 'gr8', 'KNOBS.md', 'AGENTS.md',
        'model.yaml', 'kindling.json', 'smoketest'}
# Not shipped: repo plumbing.
SKIP = {'.github', '.gitignore'}


def skipped_file(path):
    """Files inside a shipped directory that still stay out: measurements, logs and git plumbing."""
    base = os.path.basename(path)
    return (base == '.gitignore' or base.endswith('.log')
            or (path.startswith('bench/') and base.startswith('results') and base.endswith('.jsonl')))


# Each binary has the same path under --bin and in the zip.
BINARIES = ['bin/g6m4', 'bin/m4sim', 'bin/gpuprobe', 'bin/tr3_e4m3_test', 'bin/g6front-rs', 'bin/g6load',
            'bin6/g6m4', 'bin6/m4sim', 'bin6/gpuprobe', 'bin6/tr3_e4m3_test', 'bin6/g6load',
            'bin3/g6m4', 'bin3/m4sim', 'bin3/gpuprobe', 'bin3/tr3_e4m3_test', 'bin3/g6load']

HEADER = """gr8 {short} (commit {commit}, {date})

bin/   the TP=4 engine with the multi-session path (G6_NR=4 scripts/build.sh), the Rust front end g6front-rs, and
       g6load, which plans and checks the Hugging Face fetch (engine/loader-rs)
bin6/  the TP=6 engine (scripts/build.sh)
bin3/  the TP=3 engine (G6_NR=3 scripts/build.sh)
Built for GB10 (aarch64, CUDA 13, sm_121a). The binaries link cuBLAS 13 and libibverbs at run time: a node needs the
CUDA 13 runtime libraries (DGX OS has them) and rdma-core. With the prebuilt bin/, bin6/ and bin3/, skip the engine and
front end builds in README.md: copy the one for your TP to bin/ on every box.

sha256 of every file below:
"""


def git(*args):
    return subprocess.run(['git', '-C', REPO, *args], check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bin', required=True, help='the dist export: DIR/bin, DIR/bin6 and DIR/bin3')
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--out', default='.')
    ap.add_argument('--allow-missing', action='store_true', help='leave out missing binaries instead of failing')
    a = ap.parse_args()

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
