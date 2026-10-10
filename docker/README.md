# The gr8 container

One image for GB10 boxes (linux/arm64, CUDA 13, sm_121a): the TP=4 engine (`bin/`), the TP=3 engine (`bin3/`), the TP=6
engine (`bin6/`), the Rust front end, the loader, the shims and the launcher, on the CUDA 13 runtime image (which brings
cuBLAS) with libibverbs and the mlx5 provider. Each box runs one container, which holds one rank. The head (rank 0)
also runs the shim and the OpenAI front end. The container's entrypoint is `./gr8 glm53`, so it takes the launcher's
commands and options (KNOBS.md) and runs the same checks and handshake as a bare-metal box.

## Running it

On every box, with the NVIDIA container toolkit (on DGX OS by default):

    git clone https://github.com/kindlingai/gr8 && cd gr8
    cp docker/.env.example docker/.env        # then set MODEL, DRAFTER and PEERS (the same file on every box)
    docker compose -f docker/compose.yaml run --rm gr8 prepare    # this box's part of the model, ~90 GB at TP=4
    docker compose -f docker/compose.yaml up -d

`up -d` returns at once, and the boot goes on in the container: the boxes find each other, load (about a minute from a
warm cache) and serve the OpenAI API on the head, port 8100. `docker compose -f docker/compose.yaml ps` shows the
container healthy once this box's supervisor reports the cluster serving. Then, on any box:

    docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 status     # every box
    docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 down       # stop every box
    docker compose -f docker/compose.yaml logs -f                                 # this box's supervisor

and on the head, `docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 smoke` checks its API.

`down` stops the cluster and every container exits 0 and stays down. When a rank fails, every box's container exits
non-zero and restarts, and the boxes boot again together: a box that restarts first waits for the head's next boot. `docker compose down` on one box stops that box, and the
cluster with it: the other boxes restart and wait for it, retrying every `JOIN_TIMEOUT` (600 s). A watchdog that
needs to stop a wedged rank should end its `g6m4`, not its container: the supervisor then reports the failure, and
every box restarts and boots again together. Killing the container takes the supervisor with it, so no supervisor
reports the failure.

`bench` (bench/pbench.py) appends its results to `out/pbench-results.jsonl` in the home volume.

## Settings

`docker/.env` holds the launcher's options, one `KEY=value` per line: compose passes every line into the container, and
the launcher owns every default. Three more lines are for compose itself:

| variable | default | |
|---|---|---|
| `GR8_IMAGE` | `ghcr.io/kindlingai/gr8:latest` | the image |
| `GR8_CACHE` | the `cache` volume | this box's part of the model: a host directory instead, owned by uid 1000 |
| `GR8_HOME` | the `home` volume | logs and state (`out/`, `s2/`, `run/`): a host directory instead, owned by uid 1000 |

TP=4 and TP=3 need no ssh between the boxes. TP=6's one-request loop still writes requests to the other ranks over ssh,
which the container does not set up.

## Networks

The container uses the host network and sees the host's `/sys/class/infiniband`, so the launcher finds this box's
place in `PEERS` and its two fabric rails as on bare metal. `IPS`, `IPS2`, `NODE_IP`, `PORT` and `CONTROL_PORT` mean
what they mean there, and so do `SHAPE=ring` and `RING_DEVS` for boxes cabled without a switch. Without `PEERS`, the
boxes find each other through the mentat daemon on each box.

## What the container gets

| | why |
|---|---|
| host network, `/dev/infiniband`, `IPC_LOCK` and unlimited memlock | the RoCE fabric |
| the GPU | the engine |
| the host's process list (`pid: host`) | the launcher refuses to start when another `g6m4` runs on the box, and can see one started outside the container only there |
| uid 1000, no capabilities but `IPC_LOCK`, no new privileges, a read-only image with a private `/tmp` | no more rights than the engine needs |
| `oom_score_adj 1000` | under memory pressure the kernel ends the engine first, before sshd or the rest of the box |
| no core dumps | a rank maps ~90 GB, and dumping it holds the NVIDIA driver's lock long enough to leave the box unreachable |
| two volumes: the model cache and the home directory | the only state the container writes |

## Building the image

CI publishes `ghcr.io/kindlingai/gr8`. To build it yourself (about 10 minutes on a GB10, plus the first download of
the CUDA base images), from the repo root:

    docker build -t gr8 .                      # then GR8_IMAGE=gr8 in docker/.env
    docker build --target dist -o out .        # only the binaries, into out/bin, out/bin3 and out/bin6

To run a single command in the image without compose, give `docker run` the launcher command, for example
`docker run --rm gr8 show-config --peers ...`, or `--entrypoint bash` for a shell.
