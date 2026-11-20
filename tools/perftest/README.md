# perftest

`./gr8 perftest` measures the fabric the way collectives use it: an NCCL all-reduce with the Ring algorithm and then
the Tree algorithm, across the boxes. gr8 does not use NCCL when it serves; this is a health check of the ConnectX
fabric under it.

    tools/perftest/build.sh                                # once, in the CUDA build image: bin/nccl_probe, bin/lib/libnccl.so.2
    ./gr8 perftest --peers A,B,C,D                         # on every box at once, rank 0 first, engines stopped

Each box takes the rank of its own LAN address in `--peers`. Rank 0 hands NCCL's unique id to the others over TCP
(`--port`, default 29571), once per algorithm, and prints bus bandwidth (Gb/s, as nccl-tests reports it) and the time
per all-reduce for each size, 8 B to 128 MiB:

           bytes |  Ring Gb/s     us |  Tree Gb/s     us
               8 |       0.00   16.5 |       0.00   17.8
       134217728 |     178.09 6029.0 |      58.65 18308.1
    perftest: ok

That is two healthy boxes over the switch on both rails. Rank 0 exits 1 when Ring is not faster than Tree at the largest
size, or when Ring's bus bandwidth there is below `--min-gbps`.

Why Ring against Tree: these ConnectX-7s once latched a slow fallback state after DAC cables were hot-plugged, cleared
only by a power drain. In it, a raw point-to-point test (ib_write_bw) still read 109 Gb/s, while a 134 MB Ring
all-reduce over four boxes ran at 12 Gb/s, slower than Tree at 17. Healthy, the same four boxes gave Ring 111 and Tree
44. A bandwidth test alone passes in that state; a collective with dependent steps does not.

Defaults: the RDMA devices are the ACTIVE port-1 ones (`--hca` names others), NCCL picks each device's RoCE v2 GID,
and NCCL's bootstrap uses the interface of the box's `--peers` address. NCCL reads its environment once per process,
so each algorithm runs as a fresh `bin/nccl_probe`; extra NCCL variables in the environment pass through.
