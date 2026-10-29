#!/usr/bin/env python3
"""Fail on environment reads that go around the knob table (engine/knobs-rs), and on a table out of step with the tree.

  - C, C++ and CUDA under engine/ and tools/: no getenv, secure_getenv or environ, no setenv, unsetenv or putenv
    outside tests (a test sets a child's environment with them), and no "G6_NAME" string where NAME is a knob and
    G6_NAME no compile-time macro. Rust anywhere: no std::env::var, var_os or vars, and no set_var or remove_var
    outside tests. The knobs crate itself is the one exception. Read a setting through its knob: knob::name() in C++,
    gr8knobs::NAME.get() in Rust.
  - every knob the engine, its tests or the front end should read (scope agree, local or test) is read somewhere.
  - every G6_ or GR8_ assignment in the launcher (scripts/engine_args.sh) and the recipes (recipes/**/*.env) is one the
    engine accepts (`knobs validate`).
  - gr8_knobs.h and docs/ENV.md are what `knobs gen` writes (`knobs check`).

  tools/env_lint.py    from anywhere; it builds the knobs tool with cargo
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE = 'engine/knobs-rs/'
C_EXT = re.compile(r'\.(c|cc|cpp|cu|cuh|h|hpp)$')
C_READ = re.compile(r'\b(?:std::)?(?:secure_getenv|getenv)\s*\(|\benviron\b')
C_SET = re.compile(r'\b(?:std::)?(?:setenv|unsetenv|putenv|clearenv)\s*\(')
RS_READ = re.compile(r'\benv::(?:var|var_os|vars|vars_os)\s*\(')
RS_SET = re.compile(r'\benv::(?:set_var|remove_var)\s*\(')
TEST = re.compile(r'(^|/)tests?/|(^|/)test_|_test\.\w+$')
G6_NAME = re.compile(r'\bG6_[A-Z0-9_]*[A-Z0-9]\b')
# a comment, or a string or character literal (kept, for the G6_ check)
TOKENS = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'', re.S)


def knobs(*args):
    """(status, stdout, stderr) of the knobs tool, built once."""
    if not hasattr(knobs, 'bin'):
        manifest = os.path.join(REPO, CRATE, 'Cargo.toml')
        subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', manifest, '--bin', 'knobs'], check=True)
        knobs.bin = os.path.join(REPO, CRATE, 'target', 'debug', 'knobs')
    r = subprocess.run([knobs.bin, *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def files():
    return subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=REPO,
                                   text=True).split('\n')


def code(text):
    """The text with comments blanked, and the string literals it holds."""
    strings = []

    def blank(m):
        t = m.group(0)
        if t[0] in '"\'':
            strings.append(t)
            return '""'
        return re.sub(r'[^\n]', ' ', t)
    return TOKENS.sub(blank, text), strings


def line_of(text, pos):
    return text.count('\n', 0, pos) + 1


def macros(paths):
    """The G6_ names the build defines: -D flags, #define, and names tested by #if lines."""
    out = set()
    for f in paths:
        text = open(os.path.join(REPO, f), errors='replace').read()
        out |= set(re.findall(r'-D(G6_\w+)', text)) | set(re.findall(r'#\s*define\s+(G6_\w+)', text))
        for line in re.findall(r'#\s*(?:if|ifdef|ifndef|elif)\b[^\n]*', text):
            out |= set(G6_NAME.findall(line))
    return out


def bare_reads(paths, table):
    names = {name for name, _scope, _kind, family in table if not family}
    families = {name for name, _scope, _kind, family in table if family}
    defined = macros(paths)

    def knob(g6):
        return g6 not in defined and (g6[3:] in names or re.sub(r'\d+$', '', g6[3:]) in families)
    errs = []
    for f in paths:
        if f.startswith(CRATE):
            continue
        rust = f.endswith('.rs')
        if not rust and not (C_EXT.search(f) and f.startswith(('engine/', 'tools/'))):
            continue
        text = open(os.path.join(REPO, f), errors='replace').read()
        body, strings = code(text)
        bad = [RS_READ if rust else C_READ] + ([] if TEST.search(f) else [RS_SET if rust else C_SET])
        for m in (m for b in bad for m in b.finditer(body)):
            errs.append(f'{f}:{line_of(body, m.start())}: `{m.group(0).strip()}` uses the environment around the knob '
                        f'table: use the knob (engine/knobs-rs/src/table.rs)')
        if TEST.search(f):
            continue
        for s in strings:
            for g6 in G6_NAME.findall(s):
                if knob(g6):
                    errs.append(f'{f}: {s}: {g6} in code: use the knob, and its GR8_ name in messages')
    return errs


def unread(paths, table):
    text = ''
    for f in paths:
        if f.startswith(CRATE) or not (f.endswith('.rs') or C_EXT.search(f)):
            continue
        text += code(open(os.path.join(REPO, f), errors='replace').read())[0]
    errs = []
    for name, scope, _kind, _family in table:
        if scope == 'Outside':
            continue
        if not re.search(r'\bknob::' + name.lower() + r'\s*\(', text) and not re.search(r'\b' + name + r'\.(?:get|var)\(', text):
            errs.append(f'{name}: scope {scope.lower()}, but nothing reads knob::{name.lower()}() or {name}.get(): '
                        'remove it from the table, or mark it outside')
    return errs


def assignments(paths):
    """(file, NAME=VALUE) for each literal G6_/GR8_ assignment outside comments in the launcher and the recipes."""
    out = []
    for f in paths:
        if f != 'scripts/engine_args.sh' and not (f.startswith('recipes/') and f.endswith('.env')):
            continue
        for line in open(os.path.join(REPO, f)):
            line = line.split('#', 1)[0]
            for name, value in re.findall(r'\b((?:G6|GR8)_[A-Z0-9_]*[A-Z0-9])=([^\s"\';)]*)', line):
                if '$' not in value:
                    out.append((f, f'{name}={value}'))
    return out


def main():
    paths = [f for f in files() if f and os.path.isfile(os.path.join(REPO, f))]
    code_, listing, err = knobs('list')
    if code_:
        sys.exit(f'env_lint: knobs list failed:\n{err}')
    table = [tuple(line.split()) + ('',) * (4 - len(line.split())) for line in listing.splitlines()]
    errs = bare_reads(paths, table) + unread(paths, table)
    for f, a in assignments(paths):
        code_, _, err = knobs('validate', a)
        if code_:
            errs.append(f'{f}: {a}: {err.strip()}')
    code_, _, err = knobs('check')
    if code_:
        errs.append(err.strip())
    for e in errs:
        print('env_lint:', e, file=sys.stderr)
    if errs:
        sys.exit(1)
    print(f'env_lint: {len(table)} knobs, no bare environment reads')


if __name__ == '__main__':
    main()
