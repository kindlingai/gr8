# gr8 on Kubernetes

A kustomize base: a [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) of one pod per box, each running the
image's `./gr8 glm53 up` in the foreground, and a Service, `gr8-api`, for the OpenAI API on the leader. It is the
container of [docker/compose.yaml](../../docker/compose.yaml), with the same rights: the host network and process list,
the GPU, the RDMA devices, `IPC_LOCK`, uid 1000 on a read-only image.

Every pod gets the same options. `lws.yaml` sets `TP` to the group's size and `HEAD` to the leader pod's DNS name
(`LWS_LEADER_ADDRESS`), which resolves to the leader's box. The boxes find each other through mentat, and the election
makes that box rank 0, so the head runs on the leader wherever the scheduler put it, and `gr8-api` reaches it. No pod
is pinned to a box.

## What the cluster needs

- The boxes, set up as the [README](../../README.md#what-you-need) says (memory, compaction off, TCP 7340 and 7341
  open between them).
- The [LeaderWorkerSet](https://lws.sigs.k8s.io) controller, 0.8.0 or later (its `volumeClaimTemplates`).
- The NVIDIA device plugin (`nvidia.com/gpu`), and an RDMA device plugin that gives a pod `/dev/infiniband`, such as
  NVIDIA's k8s-rdma-shared-dev-plugin. Its resource name goes in the overlay.
- mentatd on every box, on its localhost. Its LAN address must be the box's node address in Kubernetes (the leader
  pod's IP on the host network), since `HEAD` resolves to that. Tag both addresses of each ConnectX port `rdma`
  (`MENTAT_ANNOUNCE_IFACES`); `SHAPE=ring` needs mentatd 0.24.0 or later.
- An unlimited memlock hard limit in the container runtime (containerd's or CRI-O's `LimitMEMLOCK=infinity`):
  Kubernetes has no ulimits field, and the entrypoint exits 2, saying so, when the limit is lower.
- A namespace that admits host-network pods (Pod Security `privileged`).
- The local-path provisioner, or another storage class (below).

## The overlay

A directory of your own, with a `kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: gr8
resources:
  - https://github.com/kindlingai/gr8//deploy/k8s?ref=<commit>    # or the path to a checkout
images:
  - name: ghcr.io/kindlingai/gr8
    newTag: sha-<commit>            # the same build on every box: they refuse a mix
configMapGenerator:
  - name: gr8
    behavior: merge
    literals:
      - DRAFTER=https://huggingface.co/incoai/GLM-5.3-DFlash2     # TP=4 only
      # - SHAPE=ring                # no switch: the boxes cabled in a ring
patches:
  - target:
      kind: LeaderWorkerSet
      name: gr8
    patch: |
      - op: replace                 # TP: 3, 4 or 6
        path: /spec/leaderWorkerTemplate/size
        value: 4
      - op: remove                  # the base's placeholder RDMA resource...
        path: /spec/leaderWorkerTemplate/workerTemplate/spec/containers/0/resources/limits/rdma~1rdma_shared_device_a
      - op: add                     # ...and the name your device plugin exposes (a / is ~1 here)
        path: /spec/leaderWorkerTemplate/workerTemplate/spec/containers/0/resources/limits/rdma~1<resource>
        value: 1
```

For TP=3, `size` 3 and no `DRAFTER`. Then:

    kubectl create namespace gr8
    kubectl label namespace gr8 pod-security.kubernetes.io/enforce=privileged
    kubectl apply -k .
    kubectl -n gr8 get pods -w                                     # Ready once the cluster serves
    kubectl -n gr8 exec gr8-0 -- /opt/gr8/gr8 glm53 status
    kubectl -n gr8 exec gr8-0 -- /opt/gr8/gr8 glm53 smoke http://gr8-api.gr8.svc:8100

The leader is `gr8-0` and the workers `gr8-0-1` to `gr8-0-<TP-1>`; `kubectl logs` shows a pod's supervisor, and
`./gr8 glm53 logs` the rest, as on bare metal. Inside the cluster, the API is `http://gr8-api.gr8.svc:8100`.

When the cluster has GB10 boxes that are not on the group's fabric (with a ring, only the cabled ones can serve), keep
the pods on the right ones: a node label, and a `nodeSelector` on the worker template, which the leader uses too.

### Options

The ConfigMap `gr8` holds the launcher's options ([KNOBS.md](../../KNOBS.md)), the same on every pod. The base sets
only `MODEL`; the launcher keeps every default, so set only what this cluster needs. Its name carries a hash, so a
change of options rolls the group onto pods that read them. Leave out `TP` and `HEAD` (`lws.yaml` sets them, over the
ConfigMap), the container's paths, and `PEERS`: with a peer list the first address is the head, and the leader need
not be on it. `PORT` moves the API, so the container's `api` port must move with it. `HF_TOKEN`, for a gated repo,
belongs in a Secret: add it to `envFrom` with one more patch op, `add` at
`/spec/leaderWorkerTemplate/workerTemplate/spec/containers/0/envFrom/-`, value `{secretRef: {name: <secret>}}`.

### Storage

Each pod gets two claims from local-path: `cache` (this box's part of the model, about 90 GB at TP=4 and 85 GB at
TP=3, one cache for both) and `home` (logs and state: `out/`, `s2/`, `run/`). A claim binds to the box its pod first
ran on, so each pod comes back to its box and its warm cache. The claims outlive the LeaderWorkerSet: delete them by
hand to free the disk. When a box is gone for good, delete its pod's claims (for `gr8-0-2`: `cache-gr8-0-2`,
`home-gr8-0-2`) and that pod starts on another box, which downloads its part again.

For another storage class, `replace` `/spec/leaderWorkerTemplate/volumeClaimTemplates/0/spec/storageClassName` (and
`/1/`). For host directories, `remove` `/spec/leaderWorkerTemplate/volumeClaimTemplates` and `add` two `hostPath`
volumes named `cache` and `home` to the worker template's `volumes`. Each must be owned by uid 1000: `fsGroup` does not
reach a host directory.

## Running it

The probes run `/opt/gr8/gr8 "$GR8_MODEL" health` (glm53 when unset), the compose healthcheck: a pod is Ready once the
cluster serves, and on the leader once its API's `/health` answers too. The startup probe gives a cold boot 4 hours to
download this box's part (about 35 minutes at 40 MB/s); on a slower link, raise its `failureThreshold`.

To stop, set `replicas` to 0 (in the overlay, or `kubectl -n gr8 scale leaderworkerset gr8 --replicas 0`), and back to
1 to boot. `./gr8 glm53 down` in a pod stops every box, but each container exits 0 and restarts, so the group boots
again: it is a restart.

When a rank fails, every box's supervisor exits non-zero, and the LeaderWorkerSet recreates the whole group
(`RecreateGroupOnPodRestart`), which boots again from the caches. Deleting one pod does the same.

To update, change `newTag`: the LeaderWorkerSet replaces the group with pods on the new image. Pin a `sha-<commit>`
tag. The base's `latest` is pulled at every start, so a pod that starts after CI publishes can run a different build
from the others, and the boxes refuse to boot until they agree.
