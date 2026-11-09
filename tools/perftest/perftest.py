#!/usr/bin/env python3
"""The fabric's collective health: NCCL all-reduce with the Ring and then the Tree algorithm across the boxes.

    ./gr8 perftest --peers A,B,C,D [--hca rocep1s0f1,roceP2p1s0f1] [--sizes 8,1048576,134217728] [--min-gbps 50]

Run the same command on every box, as with ./gr8 <model> up; each box takes the rank of its own LAN address in
--peers, and rank 0 hands the other ranks NCCL's unique id over TCP (--port), once per algorithm. Every algorithm is a
fresh bin/nccl_probe process (NCCL reads its environment once), on the RDMA devices --hca names (default: the ACTIVE
port-1 devices, the switch links), with NCCL choosing each device's RoCE v2 GID. Rank 0 prints bus bandwidth and time
per all-reduce for each size and algorithm.

Why both: a raw point-to-point bandwidth test can read healthy while collectives crawl. The ConnectX-7s on these boxes
once latched a slow fallback state (after DAC cables were hot-plugged; only a power drain cleared it) in which
ib_write_bw gave 109 Gb/s while a 134 MB Ring all-reduce ran at 12 Gb/s, slower than Tree (17); healthy, Ring ran at
111 and Tree at 44 (four boxes; two healthy boxes on both rails give Ring 178 and Tree 59). Rank 0 exits 1 when Ring is
not faster than Tree at the largest size or when Ring's bus bandwidth there is below --min-gbps.
"""

import argparse
import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def local_ipv4():
    """{address: interface} of this box."""
    out = subprocess.run(['ip', '-o', '-4', 'addr'], capture_output=True, text=True, check=True).stdout
    return {f.split()[3].split('/')[0]: f.split()[1] for f in out.splitlines()}


def default_hcas():
    """The ACTIVE RDMA devices on port 1 of each card (names ending in f1): the switch links on these boxes."""
    root = '/sys/class/infiniband'
    out = []
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        try:
            state = open(f'{root}/{d}/ports/1/state').read()
        except OSError:
            continue
        if d.endswith('f1') and 'ACTIVE' in state:
            out.append(d)
    return out


def rendezvous(rank, n, peers, port, algo, timeout, server):
    """Rank 0: None (it makes the id). Others: the id rank 0 hands out for `algo`."""
    if rank == 0:
        return None
    deadline = time.time() + timeout
    while True:
        try:
            with socket.create_connection((peers[0], port), timeout=5) as s:
                s.sendall(f'{algo}\n'.encode())
                line = s.makefile().readline().split()
                if len(line) == 2 and line[0] == 'uid':
                    return line[1]
        except OSError:
            pass
        if time.time() > deadline:
            sys.exit(f'perftest: no id from rank 0 ({peers[0]}:{port}) for {algo} within {timeout:.0f} s')
        time.sleep(0.5)


def serve_id(server, n, algo, uid, timeout):
    """Rank 0: hand `uid` to the other n - 1 ranks, each of which names the algorithm it is at."""
    server.settimeout(timeout)
    for _ in range(n - 1):
        try:
            c, _ = server.accept()
        except socket.timeout:
            sys.exit(f'perftest: not every rank asked for the {algo} id within {timeout:.0f} s')
        with c:
            got = c.makefile().readline().strip()
            if got != algo:
                sys.exit(f'perftest: a rank asked for the {got!r} id while rank 0 runs {algo}')
            c.sendall(f'uid {uid}\n'.encode())


def run(rank, n, peers, args, algo, server, env):
    """One algorithm's probe on this box; returns its result lines {bytes: (us, busbw)}."""
    cmd = [args.bin, '--rank', str(rank), '--nranks', str(n), '--sizes', args.sizes, '--iters', str(args.iters)]
    uid = rendezvous(rank, n, peers, args.port, algo, args.timeout, server)
    if uid:
        cmd += ['--uid', uid]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=dict(env, NCCL_ALGO=algo))
    res = {}
    for line in p.stdout:
        f = line.split()
        if f[:1] == ['uid'] and rank == 0:
            serve_id(server, n, algo, f[1], args.timeout)
        elif f[:1] == ['size']:
            res[int(f[1])] = (float(f[3]), float(f[7]))
    if p.wait():
        sys.exit(f'perftest: nccl_probe ({algo}) exited {p.returncode}')
    return res


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--peers', required=True, help='the boxes\' LAN addresses, rank 0 first, the same list on every box')
    p.add_argument('--hca', help='RDMA devices, comma separated (default: the ACTIVE port-1 devices)')
    p.add_argument('--algos', default='Ring,Tree')
    p.add_argument('--sizes', default='8,65536,1048576,16777216,134217728', help='bytes per all-reduce, comma separated')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--port', type=int, default=29571, help='rank 0\'s TCP port for the ids')
    p.add_argument('--timeout', type=float, default=120.0)
    p.add_argument('--min-gbps', type=float, default=0.0, help='rank 0 exits 1 when Ring\'s bus bandwidth at the largest size is below this')
    p.add_argument('--bin', default=os.path.join(REPO, 'bin', 'nccl_probe'))
    a = p.parse_args(argv)
    peers = a.peers.split(',')
    mine = local_ipv4()
    ranks = [i for i, h in enumerate(peers) if h in mine]
    if len(ranks) != 1:
        sys.exit(f'perftest: this box must hold exactly one --peers address ({a.peers}); it has {sorted(mine)}')
    rank, n = ranks[0], len(peers)
    hcas = a.hca.split(',') if a.hca else default_hcas()
    if not hcas:
        sys.exit('perftest: no ACTIVE port-1 RDMA device; name them with --hca')
    if not os.access(a.bin, os.X_OK):
        sys.exit(f'perftest: no {a.bin} (tools/perftest/build.sh builds it)')
    env = dict(os.environ, NCCL_IB_HCA='=' + ','.join(hcas), NCCL_SOCKET_IFNAME='=' + mine[peers[rank]],
               NCCL_DEBUG=os.environ.get('NCCL_DEBUG', 'WARN'),
               LD_LIBRARY_PATH=os.path.join(os.path.dirname(a.bin), 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    server = None
    if rank == 0:
        server = socket.create_server(('', a.port), reuse_port=True)
    algos = a.algos.split(',')
    print(f'perftest: rank {rank} of {n}, devices {",".join(hcas)}, algorithms {",".join(algos)}', flush=True)
    results = {algo: run(rank, n, peers, a, algo, server, env) for algo in algos}
    if rank != 0:
        return 0
    sizes = sorted(results[algos[0]])
    print(f'{"bytes":>12}' + ''.join(f' | {algo:>5} Gb/s     us' for algo in algos))
    for s in sizes:
        print(f'{s:>12}' + ''.join(f' | {results[algo][s][1]:>10.2f} {results[algo][s][0]:>6.1f}' for algo in algos))
    big = sizes[-1]
    bad = []
    if 'Ring' in results:
        ring = results['Ring'][big][1]
        if ring < a.min_gbps:
            bad.append(f'Ring bus bandwidth {ring:.1f} Gb/s at {big} bytes is below --min-gbps {a.min_gbps:g}')
        if 'Tree' in results and ring <= results['Tree'][big][1]:
            bad.append(f'Ring ({ring:.1f} Gb/s) is not faster than Tree ({results["Tree"][big][1]:.1f}) at {big} bytes: '
                       'the slow-latch signature')
    for b in bad:
        print('perftest: ' + b)
    print('perftest: ' + ('FAIL' if bad else 'ok'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
