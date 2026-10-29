# Deploying gr8

gr8 runs one rank on each GB10 box, with the same command on every box: on bare metal, in Docker Compose or on
Kubernetes. Each surface takes the same two settings, the model and how the boxes find each other, and neither has a
default. The launcher owns every other default: `./gr8 knobs` lists every option, and `show-config` prints
each value and where it came from. Each section below ends with `smoke`, which asks the API questions whose answers
can be checked. The commands name `glm53`; for GLM-5.3-Flash, `glm53f`.

## Before you start

Set up every box as the README's [What you need](../README.md#what-you-need) says: one GB10 box per rank with nothing
else holding much memory, a RoCE v2 fabric with two rails (a switch, or at TP=3 or 4 a ring of direct cables), TCP
ports 7340 and 7341 open between the boxes, and about 90 GB of disk for the model. Turn proactive memory compaction
off on every box, as root, once (it holds across reboots; `vm.compaction_proactiveness=20` undoes it):

    echo 'vm.compaction_proactiveness = 0' > /etc/sysctl.d/90-gr8-no-proactive-compaction.conf
    sysctl -w vm.compaction_proactiveness=0

## The two settings

| | the model | how the boxes find each other |
|---|---|---|
| bare metal | `./gr8 glm53`, `--model URL --model-cache DIR` | `--peers A,B,C,D`, or `--tp 4` with mentat |
| Docker Compose | `GR8_MODEL`, `MODEL` in `.env` | `PEERS`, or `TP` with mentat, in `.env` |
| Kubernetes | `options.GR8_MODEL`, `options.MODEL` | mentat, with `size` as the TP (the chart sets `TP` and `HEAD`) |

### The model

The model by name, and `MODEL`, a release of it. gr8 picks neither, and refuses a release of the other model before it
downloads anything.

| name | model | `MODEL` | TP | `DRAFTER` |
|---|---|---|---|---|
| `glm53` | GLM-5.3 | `https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4` | 3, 4, 5 or 6 | TP=4: `https://huggingface.co/incoai/GLM-5.3-DFlash2` |
| `glm53f` | GLM-5.3-Flash ([FLASH.md](FLASH.md)) | `https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4` | 2, 3 or 4 | none |

`./gr8 models` lists each model's release and the values `--tp` and `--variant` take. Each box fetches only its own part
of the release into its model cache: the first `up` downloads it (about 90 GB at TP=4, 35 minutes at 40 MB/s), and later
boots load from the cache in about a minute. `prepare` with the same options downloads ahead of time, and is never
required. On bare metal a Hugging Face `MODEL` needs `MODEL_CACHE`, a local directory for this box's part; the container
sets its own. `MODEL` may also be a local copy of the release, used in place.

`DRAFTER` is optional: without it, TP=4 drafts with MTP only. TP=3, 5 and 6 leave it out, since its 8 KV heads do not
split three, five or six ways (TP=6 runs its whole body on every box with `DFLASH=1`). Its licence (CC BY-NC-ND 4.0)
forbids sharing a converted copy, so each box downloads the drafter and converts its own rank.

### How the boxes find each other

One of:

- `PEERS`: the LAN address of every box, in rank order, the same list on every box. The first is the head: it runs the
  OpenAI API, on port 8100. The TP is the number of addresses.
- mentat: without `PEERS`, the boxes find each other through the mentat daemon (mentatd) on each box
  (`MENTAT=localhost`, the default), and `TP` must be set. The lowest address is the head, or `HEAD`.

Boxes cabled without a switch (TP=3 or 4) add `SHAPE=ring`. With `PEERS`, list the boxes in cable order: each box's f0
port cabled to the next box's f1. With mentat (mentatd 0.24.0 or later, both addresses of each ConnectX port tagged
`rdma` in `MENTAT_ANNOUNCE_IFACES`), any ring.

Before any engine starts, the boxes compare their options, model and engine build. If one box differs, every box stops
and prints the field and each box's value.

## Bare metal

The build needs CUDA 13 (`nvcc` with `sm_121a`), g++ with C++20, CMake, clang and libclang-dev, rdma-core /
libibverbs-dev, Rust 1.85 or later through [rustup](https://rustup.rs) (Ubuntu 24.04's `cargo` package is 1.75), and
Python 3 with venv. The stock DGX OS has all of them but Rust. On every box:

    git clone https://github.com/kindlingai/gr8 && cd gr8
    ./gr8 glm53 build                  # the engine for every TP width and the front end, into bin/
    python3 -m venv venv               # the python the head runs the shim on
    sudo mkdir -p /opt/gr8/cache && sudo chown $USER /opt/gr8/cache
    ./gr8 glm53 up --peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
        --model https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4 --model-cache /opt/gr8/cache \
        --drafter https://huggingface.co/incoai/GLM-5.3-DFlash2

`up` returns once the cluster serves. For three, five or six boxes, list that many addresses, and leave out `--drafter`.
With mentatd on every box, `--tp 4` takes the place of `--peers`. Every box must run the same build: build on each, or
build on one with `BUNDLE_LIBS=1` (which copies cuBLAS next to the engine, for boxes without the CUDA toolkit) and copy
its `bin/` to the others.

To keep the options in a file, copy [config/cluster.env.example](../config/cluster.env.example) to `cluster.env` in the
checkout on every box: then `./gr8 glm53 up` alone is enough. Then, on the head (the first address in `--peers`):

    ./gr8 glm53 status                 # every box, the front end and /health (from any box)
    ./gr8 glm53 smoke                  # from another box: ./gr8 glm53 smoke http://10.0.0.1:8100

`./gr8 glm53 down` on any box stops every box.

## Docker Compose

Nothing to build or clone: CI publishes the image, `ghcr.io/kindlingai/gr8`, with the engine for every TP width and both
models. On every box, with Docker Compose and the NVIDIA container toolkit (DGX OS has both):

    mkdir -p ~/gr8 && cd ~/gr8
    curl -fsSLO https://raw.githubusercontent.com/kindlingai/gr8/main/docker/compose.yaml
    curl -fsSL https://raw.githubusercontent.com/kindlingai/gr8/main/docker/.env.example -o .env
    nano .env                          # GR8_MODEL, MODEL, PEERS and (glm53 at TP=4) DRAFTER: the same file on every box
    docker compose up -d

`.env` sits next to `compose.yaml`: compose reads it from there and passes every line into the container, where the
launcher reads it as an option. In a checkout the same files are `docker/compose.yaml` and `docker/.env.example` (copy
it to `docker/.env`), and every command takes `-f docker/compose.yaml`. To stay on one build, fetch both files from its
commit (the commit in place of `main` in both URLs) and set `GR8_IMAGE=ghcr.io/kindlingai/gr8:sha-<commit>` in `.env`:
every box must run the same image.

`up -d` returns at once, and the boot goes on in the container: the first one downloads this box's part of the model
(`docker compose run --rm gr8 prepare` does that ahead of time, and is optional), then the boxes find each other, load
and serve. `docker compose logs -f` follows this box's supervisor. `docker compose ps` shows the container healthy once
the cluster serves; a first boot still downloading after the healthcheck's 30-minute start period shows unhealthy until
then. Then, on the head:

    docker compose exec gr8 /opt/gr8/gr8 glm53 status
    docker compose exec gr8 /opt/gr8/gr8 glm53 smoke

`docker compose exec gr8 /opt/gr8/gr8 glm53 down` on any box stops every box, and `docker compose up -d` on every box
boots them again. [docker/README.md](../docker/README.md) has the rest: the other settings, updating, and what the
container gets.

## Kubernetes

[deploy/k8s](../deploy/k8s/README.md) is a Helm chart: a LeaderWorkerSet of one pod per box, each running the image's
`up`. The boxes find each other through mentat, and the leader pod is the head wherever the scheduler puts it. The
cluster needs, as the chart's README says:

- the [LeaderWorkerSet](https://lws.sigs.k8s.io) controller, 0.8.0 or later;
- the NVIDIA device plugin, and an RDMA device plugin that gives a pod `/dev/infiniband`, such as NVIDIA's
  k8s-rdma-shared-dev-plugin;
- mentatd on every box, on its localhost, with the box's node address as its LAN address and both addresses of each
  ConnectX port tagged `rdma` (`MENTAT_ANNOUNCE_IFACES`);
- an unlimited memlock hard limit in the container runtime (containerd's or CRI-O's `LimitMEMLOCK=infinity`);
- a default storage class for the per-pod claims, or `hostPath` directories owned by uid 1000 (the chart's
  [Storage](../deploy/k8s/README.md#storage)).

A values file, `my-values.yaml`:

```yaml
image:
  tag: sha-<commit>                 # the same build on every box: they refuse a mix
size: 4                             # the TP: 3, 4, 5 or 6
options:                            # the launcher's options (./gr8 knobs)
  GR8_MODEL: glm53
  MODEL: https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4
  DRAFTER: https://huggingface.co/incoai/GLM-5.3-DFlash2     # glm53 at TP=4 only
  # SHAPE: ring                     # no switch: the boxes cabled in a ring
resources:
  limits:
    rdma/<resource>: 1              # the name your RDMA device plugin exposes
```

Then, from a checkout:

    git clone https://github.com/kindlingai/gr8 && cd gr8
    kubectl create namespace gr8
    kubectl label namespace gr8 pod-security.kubernetes.io/enforce=privileged
    helm install gr8 deploy/k8s -n gr8 -f my-values.yaml
    kubectl -n gr8 get pods -w                                     # Ready once the cluster serves
    kubectl -n gr8 exec gr8-0 -- /opt/gr8/gr8 glm53 status
    kubectl -n gr8 exec gr8-0 -- /opt/gr8/gr8 glm53 smoke http://gr8-api.gr8.svc:8100

The leader is `gr8-0`, and the Service `gr8-api` reaches its API. A cold boot downloads first: the startup probe allows
4 hours. `kubectl -n gr8 scale leaderworkerset gr8 --replicas 0` stops every box, and `--replicas 1` boots them again.
The chart's [README](../deploy/k8s/README.md) has every value, storage, placement and updating.

## With model-loader

[model-loader](https://github.com/kindlingai/model-loader) can fetch the weights instead of gr8. The release comes
from Hugging Face once, or from a NAS that holds the master copy, and the boxes copy the pieces between them over the
LAN. Each box runs model-loader next to gr8, from model-loader's own deploy tree, with the release (and the drafter) in
its `catalog.yaml`. gr8 then asks the box's model-loader for the pieces its rank needs, waits until they are held,
checks every expert half's xxhash64, and reads the weights in place from model-loader's store, which it mounts
read-only. The legacy fetch stays the default; nothing changes for a box without model-loader.

Nothing about the model changes either: the two settings are the same. gr8 finds the release and the drafter in
model-loader's catalog by their Hugging Face repos, whatever the catalog calls them. What selects model-loader is its
address, `MODEL_LOADER` (with `MODEL_CACHE` its store): every setup below sets both, and that alone is enough
(`FETCH=model-loader` is then implied). The catalog should give the width files small segments, as model-loader's
example catalogs do, or each box holds every width file whole.

| | model-loader on the box | gr8 |
|---|---|---|
| bare metal | model-loader's `deploy/compose`, or `deploy/spark-os` | `--model-loader unix:/var/lib/model-loader/model-loader.sock --model-cache /var/lib/model-loader` |
| Docker Compose | model-loader's `deploy/compose`, or `deploy/spark-os` | add `-f docker/model-loader.yaml` (or model-loader's `deploy/compose/gr8.yml`) |
| Kubernetes | model-loader's `deploy/k8s` DaemonSet | `modelLoader.enabled: true` in the chart's values |

model-loader runs in group 1000, gr8's, in all of its deploy trees: its socket, `model-loader.sock` in the store, is
mode 0660 in that group, so gr8 can open it and nothing else on the box can.

### Docker Compose with model-loader

With model-loader running on the box (its `deploy/compose`, or `deploy/spark-os` on kindling-spark-os), fetch the
override next to `compose.yaml` and add it to every command:

    cd ~/gr8
    curl -fsSLO https://raw.githubusercontent.com/kindlingai/gr8/main/docker/model-loader.yaml
    docker compose -f compose.yaml -f model-loader.yaml up -d

`.env` is the same file as without model-loader. The override mounts model-loader's store read-only
(`MODEL_LOADER_STORE` in `.env`, default `/var/lib/model-loader`), points gr8 at it and its socket, and starts gr8
once `model-loader -check` answers. model-loader's `deploy/compose` keeps its store in a named volume unless
`MODEL_LOADER_STORE` is set there: set it to the same host directory on both sides. No `prepare` is needed:
model-loader keeps the pieces between boots. [docker/README.md](../docker/README.md#with-model-loader) has the rest.

model-loader's own `deploy/compose/gr8.yml` (gr8 and model-loader in one project, `GR8_ENV` naming gr8's `.env`) and
`deploy/spark-os/gr8.override.yml` work as they are: they set the store and the socket, which selects model-loader.

### Bare metal with model-loader

With model-loader on the box (its `deploy/compose` or `deploy/spark-os`, store at `/var/lib/model-loader`), give
`up` its socket and store in place of the model cache:

    ./gr8 glm53 up --peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
        --model https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4 \
        --model-loader unix:/var/lib/model-loader/model-loader.sock --model-cache /var/lib/model-loader \
        --drafter https://huggingface.co/incoai/GLM-5.3-DFlash2

The user running gr8 must be in group 1000 to open the socket. The drafter is converted under `G6_HOME`, never in
the store: its licence forbids sharing a converted copy. [config/cluster.env.example](../config/cluster.env.example)
has the same two lines.

### Kubernetes with model-loader

Run model-loader's `deploy/k8s` peer DaemonSet on the GPU nodes (its `examples/gr8` has the settings gr8 needs: group
1000 and a readiness probe on the socket; leave out its gr8 Deployment, which this chart replaces), then add to
`my-values.yaml`:

```yaml
modelLoader:
  enabled: true
  store: /var/lib/model-loader      # the DaemonSet's store on every node (hostPath)
```

The chart mounts the store read-only, sets `MODEL_LOADER` and `MODEL_CACHE`, makes no `cache` claim, and waits in an
init container until model-loader answers. The chart's [README](../deploy/k8s/README.md#model-loader) has the rest.

## After smoke

On the head, the API is OpenAI's:

    curl http://10.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

The README's [Running it](../README.md#running-it) has every command (`status`, `logs`, `down`, `restart`), the logs,
the model cache, the expert variants and the options worth knowing, and
[Troubleshooting](../README.md#troubleshooting) what to do when a boot stops.
