# gr8 on Kubernetes

A Helm chart: a [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) of one pod per box, each running the image's
`./gr8 $GR8_MODEL up` in the foreground, and a Service, `<release>-api`, for the OpenAI API on the leader. It is the
container of [docker/compose.yaml](../../docker/compose.yaml), with the same rights: the host network and process list,
the GPU, the RDMA devices, `IPC_LOCK`, uid 1000 on a read-only image.

Every pod gets the same options. The chart sets `TP` to the group's size and `HEAD` to the leader pod's DNS name
(`LWS_LEADER_ADDRESS`), which resolves to the leader's box. The boxes find each other through mentat, and the election
makes that box rank 0, so the head runs on the leader wherever the scheduler put it, and the Service reaches it. No pod
is pinned to a box.

## What the cluster needs

- The boxes, set up as the [README](../../README.md#what-you-need) says (memory, compaction off, TCP 7340 and 7341
  open between them).
- The [LeaderWorkerSet](https://lws.sigs.k8s.io) controller, 0.8.0 or later (its `volumeClaimTemplates`).
- The NVIDIA device plugin (`nvidia.com/gpu`), and an RDMA device plugin that gives a pod `/dev/infiniband`, such as
  NVIDIA's k8s-rdma-shared-dev-plugin.
- mentatd on every box, on its localhost. Its LAN address must be the box's node address in Kubernetes (the leader
  pod's IP on the host network), since `HEAD` resolves to that. Tag both addresses of each ConnectX port `rdma`
  (`MENTAT_ANNOUNCE_IFACES`); `SHAPE=ring` needs mentatd 0.24.0 or later.
- An unlimited memlock hard limit in the container runtime (containerd's or CRI-O's `LimitMEMLOCK=infinity`):
  Kubernetes has no ulimits field, and the entrypoint exits 2, saying so, when the limit is lower.
- A namespace that admits host-network pods (Pod Security `privileged`).

## Installing it

A values file for your cluster ([values.yaml](values.yaml) has every value and its default):

```yaml
image:
  tag: sha-<commit>                 # the same build on every box: they refuse a mix
size: 4                             # TP: 3, 4, 5 or 6
options:                            # the launcher's options (KNOBS.md)
  GR8_MODEL: glm53                  # required: glm53 (GLM-5.3) or glm53f (GLM-5.3-Flash)
  MODEL: https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4    # required: a release of that model
  DRAFTER: https://huggingface.co/incoai/GLM-5.3-DFlash2     # glm53 at TP=4 only
  # SHAPE: ring                     # no switch: the boxes cabled in a ring
resources:
  limits:
    rdma/<resource>: 1              # the name your RDMA device plugin exposes
storage:
  cache:
    storageClass: local-path        # "" = the cluster's default; or hostPath: /srv/gr8/cache
  home:
    storageClass: local-path
```

`GR8_MODEL` and `MODEL` have no default, and `helm install` refuses a values file without them: `./gr8 models` lists
each model's release, TP sizes and variants (for GLM-5.3-Flash, `GR8_MODEL: glm53f`, its release, and no `DRAFTER`).
For TP=3, `size: 3` and no `DRAFTER`. Then, from a checkout:

    kubectl create namespace gr8
    kubectl label namespace gr8 pod-security.kubernetes.io/enforce=privileged
    helm install gr8 deploy/k8s -n gr8 -f my-values.yaml
    kubectl -n gr8 get pods -w                                     # Ready once the cluster serves
    kubectl -n gr8 exec gr8-0 -- /opt/gr8/gr8 glm53 status
    kubectl -n gr8 exec gr8-0 -- /opt/gr8/gr8 glm53 smoke http://gr8-api.gr8.svc:8100

With release `gr8`, the leader is `gr8-0` and the workers `gr8-0-1` to `gr8-0-<TP-1>`; `kubectl logs` shows a pod's
supervisor, and `./gr8 glm53 logs` the rest, as on bare metal.

### Options

`options` becomes the ConfigMap the pods read, the same on every pod. The chart sets only `MODEL`; the launcher keeps
every default, so set only what this cluster needs. A change of options rolls the group onto pods that read them. The
chart refuses `TP` and `HEAD` (it sets them) and `PEERS` (with a peer list the first address is the head, and the leader
need not be on it). `PORT`, `CONTROL_PORT` and `S2PORT` move the pod's ports with them. `HF_TOKEN`, for a gated repo,
belongs in a Secret: `extraEnvFrom: [{secretRef: {name: <secret>}}]`. A second group on the same mentat network is a
second release with its own `GROUP`.

### Storage

Each pod has `cache` (this box's part of the model, about 90 GB at TP=4 and 85 GB at TP=3, one cache for both) and
`home` (logs and state: `out/`, `s2/`, `run/`). Each is a claim per pod from `storageClass` (the cluster's default
when empty), of `size`, or with `hostPath`, a directory on every box, which must exist and be owned by uid 1000
(`fsGroup` does not reach a host directory). Local storage, such as local-path, binds each claim to the box its pod
first ran on, so each pod comes back to its box and its warm cache; when a box is gone for good, delete its pod's
claims (for `gr8-0-2`: `cache-gr8-0-2`, `home-gr8-0-2`) and the pod starts on another box, which downloads its part
again. The claims outlive the release (`persistentVolumeClaimRetentionPolicy`): delete them by hand to free the disk.

`MODEL` may also be a local copy of the release, used in place (KNOBS.md). Mount it with `extraVolumes` and
`extraVolumeMounts`, read-only, at the same path on every box, and set `MODEL` to that path in `options`. The cache then
holds no weights:

```yaml
options:
  MODEL: /models/release
extraVolumes:
  - name: release
    hostPath: {path: /srv/models/<release>, type: Directory}
extraVolumeMounts:
  - name: release
    mountPath: /models/release
    readOnly: true
```

### model-loader

With `modelLoader.enabled: true`, the box's [model-loader](https://github.com/kindlingai/model-loader) fetches the
weights, and the pods share its pieces over the LAN rather than each fetching from Hugging Face:

```yaml
modelLoader:
  enabled: true
  store: /var/lib/model-loader      # model-loader's store on every box
```

The chart mounts the store read-only from the box (`hostPath`), sets `FETCH=model-loader`, `MODEL_CACHE` to the store
and `MODEL_LOADER` to its socket, and makes no `cache` claim. An init container waits until `model-loader -check`
answers. model-loader runs on each box from its own deploy tree: its `deploy/k8s` peer DaemonSet with the settings of
its `examples/gr8` (group 1000, gr8's, so gr8 can open its socket, and a readiness probe on the socket; leave out that
example's gr8 Deployment, which this chart replaces), or `deploy/spark-os`. gr8 finds the release (and `DRAFTER`) in model-loader's catalog by the
repo, so nothing in `options` repeats a catalog name.

### Placement

The pods go wherever the scheduler finds a GPU and the RDMA resource. When the cluster has GB10 boxes that are not on
the group's fabric (with a ring, only the cabled ones can serve), keep the pods on the right ones with `nodeSelector`
or `affinity`, which apply to every pod alike. `tolerations` tolerates the common `nvidia.com/gpu` taint;
`runtimeClassName`, `priorityClassName`, `imagePullSecrets`, `podLabels` and `podAnnotations` are there when the
cluster needs them.

## Running it

The probes run `/opt/gr8/gr8 "$GR8_MODEL" health`, the compose healthcheck: a pod is Ready once the
cluster serves, and on the leader once its API's `/health` answers too. The startup probe gives a cold boot 4 hours to
download this box's part (about 35 minutes at 40 MB/s); on a slower link, raise `startupProbe.failureThreshold`.

To stop, set `replicas: 0` (or `kubectl -n gr8 scale leaderworkerset gr8 --replicas 0`), and back to 1 to boot.
`./gr8 glm53 down` in a pod stops every box, but each container exits 0 and restarts, so the group boots again: it is a
restart.

When a rank fails, every box's supervisor exits non-zero, and the LeaderWorkerSet recreates the whole group
(`RecreateGroupOnPodRestart`), which boots again from the caches. Deleting one pod does the same.

To update, change `image.tag` and `helm upgrade`: the LeaderWorkerSet replaces the group with pods on the new image.
