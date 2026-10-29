# The gr8 container

One image for GB10 boxes (linux/arm64, CUDA 13, sm_121a): the engine for every TP width (`bin/`), the Rust front end,
the loader, the shims and the launcher, on the CUDA 13 runtime image (which brings cuBLAS) with libibverbs and the mlx5
provider. Each box runs one container, which holds one rank. The head (rank 0) also runs the shim and the OpenAI front
end. The container's entrypoint is `./gr8 $GR8_MODEL`, so it takes the launcher's commands and options (./gr8 knobs) and
runs the same checks and handshake as a bare-metal box.

## Running it

[docs/DEPLOY.md](../docs/DEPLOY.md#docker-compose) has the steps, with or without a checkout. The commands here run from
a checkout's root; in a directory that holds only `compose.yaml` and `.env`, run them there without
`-f docker/compose.yaml`. On every box, with the NVIDIA container toolkit (on DGX OS by default):

    cp docker/.env.example docker/.env        # then set GR8_MODEL, MODEL, PEERS and (TP=4) DRAFTER: one file for all
    docker compose -f docker/compose.yaml up -d

`up -d` returns at once, and the boot goes on in the container: the first one downloads this box's part of the model
(~90 GB at TP=4; `docker compose -f docker/compose.yaml run --rm gr8 prepare` does that ahead of time, and is
optional), then the boxes find each other, load (about a minute from a warm cache) and serve the OpenAI API on the
head, port 8100. `docker compose -f docker/compose.yaml ps` shows the
container healthy once this box's supervisor reports the cluster serving: the healthcheck runs
`/opt/gr8/gr8 "$GR8_MODEL" health`, and on the head it also needs the API's `/health`. A Kubernetes
pod can run the same command as an `exec` readiness probe on every rank. A KV node's container (`GR8_MODEL=kv` and
`KV_GB` in `docker/.env`, so the entrypoint runs `./gr8 kv up`) gets the same healthcheck, which there runs
`./gr8 kv health`: healthy once the node serves and its process runs, and a Kubernetes `exec` probe runs that command
too. Then, on any box:

    docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 status     # every box
    docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 down       # stop every box
    docker compose -f docker/compose.yaml logs -f                                 # this box's supervisor

and on the head, `docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 smoke` checks its API.

`down` stops the cluster and every container exits 0 and stays down; to boot again, `up -d` on every box (`restart`
does not work inside the container: its `down` ends the container it runs in). When a rank fails, or the head's
front end or shim exits, every box's container exits non-zero and restarts, and the boxes boot again together: a box that restarts first waits for the head's next boot. `docker compose down` on one box stops that box, and the
cluster with it: the other boxes restart and wait for it, retrying every `JOIN_TIMEOUT` (600 s). A watchdog that
needs to stop a wedged rank should end its `g6m4`, not its container: the supervisor then reports the failure, and
every box restarts and boots again together. Killing the container takes the supervisor with it, so no supervisor
reports the failure.

## With model-loader

`docker/model-loader.yaml` makes the box's model-loader fetch the weights ([docs/DEPLOY.md](../docs/DEPLOY.md#with-model-loader)
has every surface):

    docker compose -f docker/compose.yaml -f docker/model-loader.yaml up -d

In a directory that holds only `compose.yaml` and `.env`, fetch `model-loader.yaml` next to them and drop `docker/`.

Give both `-f` files to every compose command (`pull`, `exec`, `logs`). The override:

- mounts model-loader's store read-only at `/var/lib/model-loader` (`MODEL_LOADER_STORE` on the host, default
  `/var/lib/model-loader`), and sets `MODEL_CACHE` to it. gr8 reads the release in place and writes nothing there.
  Its own state (the expert-half check receipts, a converted drafter, unpacked steering profiles) goes to
  `run/model-loader/` in the home volume;
- sets `FETCH=model-loader` and `MODEL_LOADER` to the socket `model-loader.sock` in the store;
- starts gr8 once that socket answers (`model-loader -check`, in a one-shot container from `MODEL_LOADER_IMAGE`).

model-loader runs from its own deploy tree: `deploy/compose` as a container, or `deploy/spark-os` as a systemd unit on
kindling-spark-os. Either runs in group 1000, gr8's, so gr8 can open the socket (mode 0660). With `deploy/compose`,
set its `MODEL_LOADER_STORE` to the same host directory as here, because its default is a named volume. Its catalog
lists the release and, for `DRAFTER`, the drafter, with their Hugging Face repos as sources. gr8 asks model-loader for
the entries whose repos are `MODEL`'s and `DRAFTER`'s, so the catalog may name them anything and nothing in
`docker/.env` repeats the names.

model-loader's own `deploy/compose/gr8.yml` (gr8 in model-loader's project) and `deploy/spark-os/gr8.override.yml`
work in place of this file: they set `MODEL_CACHE` and `MODEL_LOADER`, and `MODEL_LOADER` alone selects model-loader.

The `cache` volume is not used then. A `run --rm gr8 prepare` asks model-loader for this box's pieces and waits until
it holds them, without booting.

## Updating

CI publishes `ghcr.io/kindlingai/gr8:latest` from every push to main, and `sha-<commit>` and release tags beside it.
Compose pulls an image only when the box has none, so a box keeps the image it first pulled until you pull again. To
update, on every box (with the files fetched by URL, fetch `compose.yaml` again first):

    docker compose -f docker/compose.yaml pull
    docker compose -f docker/compose.yaml exec gr8 /opt/gr8/gr8 glm53 down     # once, on any box: stops every box
    docker compose -f docker/compose.yaml up -d

Every box must run the same image: the boxes compare their engine builds before loading, and refuse to boot with a mix.
To stay on one build, set `GR8_IMAGE=ghcr.io/kindlingai/gr8:sha-<commit>` in `docker/.env` on every box.

`bench` (bench/pbench.py) appends its results to `out/pbench-results.jsonl` in the home volume.

## Settings

`docker/.env` holds the launcher's options, one `KEY=value` per line: compose passes every line into the container, and
the launcher owns every default. `GR8_MODEL` has none: it names the model, `glm53` (GLM-5.3) or `glm53f`
(GLM-5.3-Flash), and `MODEL` must be a release of that model. Without it the container exits 2 at once, naming the
line to add. `docker run --rm --entrypoint /opt/gr8/gr8 ghcr.io/kindlingai/gr8 models` lists each model's release, TP
sizes and variants. Three more lines are for compose itself:

| variable | default | |
|---|---|---|
| `GR8_IMAGE` | `ghcr.io/kindlingai/gr8:latest` | the image |
| `GR8_CACHE` | the `cache` volume | this box's part of the model: a host directory instead, owned by uid 1000 |
| `GR8_HOME` | the `home` volume | logs and state (`out/`, `s2/`, `run/`): a host directory instead, owned by uid 1000, and not a bare-metal `GR8_WORK_DIR` (its `bin/` is a build) |
| `MODEL_LOADER_STORE` | `/var/lib/model-loader` | with `docker/model-loader.yaml`: model-loader's store on this box |
| `MODEL_LOADER_IMAGE` | `ghcr.io/kindlingai/model-loader:latest` | with `docker/model-loader.yaml`: the image whose `model-loader -check` gr8's start waits on |

No topology needs ssh between the boxes: TP=6 serves through S2 like TP=4 and TP=3.

## Networks

The container uses the host network and sees the host's `/sys/class/infiniband`, so the launcher finds this box's
place in `PEERS` and its two fabric rails as on bare metal. `IPS`, `IPS2`, `NODE_IP`, `PORT` and `CONTROL_PORT` mean
what they mean there, and so do `SHAPE=ring` and `RING_DEVS` for boxes cabled without a switch. Without `PEERS`, the
boxes find each other through the mentat daemon on each box.

## What the container gets

| | why |
|---|---|
| host network, `/dev/infiniband`, `IPC_LOCK` and unlimited memlock | the RoCE fabric (below) |
| the GPU | the engine |
| the host's process list (`pid: host`) | the launcher refuses to start when another `g6m4` runs on the box, and can see one started outside the container only there |
| uid 1000, no capabilities but `IPC_LOCK`, no new privileges, a read-only image with a private `/tmp` | no more rights than the engine needs |
| `oom_score_adj 1000` | under memory pressure the kernel ends the engine first, before sshd or the rest of the box |
| no core dumps (the entrypoint sets `ulimit -c 0` itself) | a rank maps ~90 GB, and dumping it holds the NVIDIA driver's lock long enough to leave the box unreachable |
| two volumes: the model cache and the home directory | the only state the container writes |

The entrypoint sets these limits itself, so a runtime without compose's `ulimits:` needs no wrapper. It sets
`ulimit -c 0`, and before `up` raises the soft memlock limit to the hard one. When the hard limit is not unlimited, `up`
exits 2 at once with a message: compose sets it (`ulimits.memlock: -1`), `docker run` needs `--ulimit memlock=-1`, and
Kubernetes, which has no ulimits field, takes the container runtime's default limit. The message goes to `docker logs`
only, not `out/supervisor.log`, since the supervisor never started.

## Building the image

CI publishes `ghcr.io/kindlingai/gr8`. To build it yourself (about 10 minutes on a GB10, plus the first download of
the CUDA base images), from the repo root:

    docker build -t gr8 .                      # then GR8_IMAGE=gr8 in docker/.env
    docker build --target dist -o out .        # only the binaries, into out/bin

To run a single command in the image without compose, give `docker run` the launcher command, for example
`docker run --rm gr8 show-config --peers ...`, or `--entrypoint bash` for a shell.
