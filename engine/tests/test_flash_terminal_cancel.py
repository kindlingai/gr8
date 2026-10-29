"""GR8_GLM53F_CACHE=terminal, a cancel at the boundary the prefill checkpoint waits for: compile
runtime/s2/tests/flash_terminal_cancel_test.cpp with flash_native.cuh's own checkpoint hooks pasted in (take_capture,
h.pool.retain, the prefill's capture block, h.retire, h.before_work) and run it on two ranks through the real Service.
  usage: test_flash_terminal_cancel.py [flash_native.cuh]   (default: this tree's)"""
from pathlib import Path
import subprocess
import sys
import tempfile


def cut(source, start, end):
    i = source.index(start)
    j = source.index(end, i) + len(end)
    return source[i:j]


def main():
    root = Path(__file__).resolve().parents[1]
    native = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'runtime/s2/flash_native.cuh'
    source = native.read_text()
    take = cut(source, '    auto take_capture = [&] {', '\n    };').replace('auto take_capture =', 'take_capture =', 1)
    parts = {
        '@TAKE_CAPTURE@': take,
        '@RETAIN@': cut(source, '        h.pool.retain = [&]', '\n        };'),
        '@PREFILL_CAPTURE@': cut(source, '        if (capture_at && first + rows == capture_at) {', '\n        }'),
        '@RETIRE@': cut(source, '    h.retire = [&]', ';\n'),
        '@BEFORE_WORK@': cut(source, '    if (ck) h.before_work = take_capture;', ';'),
    }
    program = (root / 'runtime/s2/tests/flash_terminal_cancel_test.cpp').read_text()
    for k, v in parts.items():
        program = program.replace(k, v)
    kv = root / 'kvman-rs/target/release/libg6kvman.a'
    knobs = root / 'knobs-rs/target/release/libgr8knobs.a'
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / 't.cpp', Path(d) / 't'
        src.write_text(program)
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-pthread', '-I', str(root / 'runtime'), str(src), str(kv),
                        str(knobs), '-lpthread', '-ldl', '-lm', '-o', str(exe)], check=True)
        out = subprocess.run([str(exe)], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sys.stdout.write(out.stdout)
        if out.returncode:
            sys.stdout.write(''.join(l + '\n' for l in out.stderr.splitlines() if l.startswith('FAIL') or 'rank' in l[:8]))
        sys.exit(out.returncode)


if __name__ == '__main__':
    main()
