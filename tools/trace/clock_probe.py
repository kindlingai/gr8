#!/usr/bin/env python3
"""Measures one host's wall clock against this machine's, for tools/trace/merge.py --clock.

    clock_probe.py HOST OUT.json [N=7]

Each rank's GR8_S2_TRACE file and a client's request times carry their own machine's wall clock, and two machines'
clocks differ by whatever NTP left. This opens ONE ssh session to HOST running `date +%s.%N` once per line it reads,
and takes N + 1 samples: this machine's time t0, the host's time, this machine's time t1 (sample 0 also absorbs the
login, so it is kept but rarely the best). The sample with the lowest round trip t1 - t0 gives the estimate
offset = host - (t0 + t1) / 2, good to half that round trip. Run it once before the run and once after: merge.py takes
the mean of the two and refuses a probe whose round trip or before/after drift exceeds its limits.
OUT.json holds the raw samples in integer ns; merge.py recomputes everything from them."""
import json
import os
import select
import subprocess
import sys
import time


def host_ns(line):
    """integer ns of a `date +%s.%N` line."""
    s, ns = line.strip().split('.')
    return int(s) * 10**9 + int(ns.ljust(9, '0')[:9])


def probe(host, n):
    p = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                          'while IFS= read -r x; do date +%s.%N; done'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    samples, buf = [], b''
    try:
        for _ in range(n + 1):
            t0 = time.time_ns()
            p.stdin.write(b'x\n')
            p.stdin.flush()
            while b'\n' not in buf:
                if not select.select([p.stdout], [], [], 20)[0]:
                    sys.exit(f'clock_probe: {host} did not answer in 20 s')
                chunk = os.read(p.stdout.fileno(), 4096)
                if not chunk:
                    sys.exit(f'clock_probe: the ssh session to {host} ended')
                buf += chunk
            t1 = time.time_ns()
            line, buf = buf.split(b'\n', 1)
            samples.append([t0, host_ns(line.decode()), t1])
    finally:
        p.stdin.close()
        rc = p.wait(timeout=20)
    if rc != 0:
        sys.exit(f'clock_probe: ssh {host} exited {rc}')
    return samples


def main(argv):
    if len(argv) not in (2, 3):
        sys.exit(__doc__.split('\n\n')[1])
    host, out = argv[0], argv[1]
    samples = probe(host, int(argv[2]) if len(argv) == 3 else 7)
    with open(out, 'x') as f:
        json.dump({'host': host, 'samples_unit': 'integer ns [local_t0, host, local_t1]', 'samples': samples}, f)
    t0, h, t1 = min(samples, key=lambda s: s[2] - s[0])
    print(f'{out}: {host} offset {(2 * h - t0 - t1) / 2e6:.3f} ms, round trip {(t1 - t0) / 1e6:.3f} ms')


if __name__ == '__main__':
    main(sys.argv[1:])
