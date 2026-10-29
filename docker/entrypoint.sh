#!/bin/bash
# The image's entrypoint: ./gr8 $GR8_MODEL COMMAND [OPTIONS], with no command meaning `up`. GR8_MODEL is required
# (glm53 or glm53f): nothing picks the model for the runner. It first sets up this box's working directory (G6_HOME):
# out/ exists, and bin/ links to the image's engine, which runs every TP width.
#
# It sets its own limits, so a runtime without compose's `ulimits:` (Kubernetes has no such field) needs no wrapper: no
# core dumps, and before `up` the soft memlock limit raised to the hard one, which must be unlimited.
#
# `up` runs the supervisor in the foreground, so the container lives exactly as long as this box's part of the cluster.
# Its exit status is the launcher's: 0 after a `down` or a stop, 1 when a rank failed, 2 for bad options, 4 when the
# handshake refused the boot. Its output goes to docker logs and to out/supervisor.log, which `logs supervisor` reads.
set -euo pipefail
home=${G6_HOME:?G6_HOME is not set}
# ./gr8 takes --no-check before the command; take it here as CHECK=0, so the command below is the real one.
while [ "${1:-}" = --no-check ]; do export CHECK=0; shift; done
[ $# -gt 0 ] || set -- up
cmd=$1; shift

# ./gr8 refuses an unknown name, listing the models; this names the line to add.
if [ -z "${GR8_MODEL:-}" ]; then
  echo "gr8: GR8_MODEL is not set: name the model, GR8_MODEL=glm53 (GLM-5.3) or GR8_MODEL=glm53f (GLM-5.3-Flash)," \
       "in docker/.env, with docker run -e, or in the container's env on Kubernetes" >&2
  exit 2
fi
model=$GR8_MODEL

# A crash must not dump core: a rank maps ~90 GB, and the dumping thread holds the NVIDIA driver's lock the whole time
# it writes, which can leave the box unreachable until a power cycle.
ulimit -c 0
# The fabric registers its pinned buffers with the NIC, which uid 1000 (no effective IPC_LOCK) may only do within
# RLIMIT_MEMLOCK. The runtime sets the hard limit, so a limited one stops the boot here rather than later in the
# fabric. This runs before out/supervisor.log is set up: the message goes to docker logs (kubectl logs) only.
if [ "$cmd" = up ]; then
  ulimit -S -l "$(ulimit -H -l)"
  if [ "$(ulimit -l)" != unlimited ]; then
    echo "gr8: memlock is limited (ulimit -l: $(ulimit -l) KiB), and the fabric needs it unlimited: set" \
         "ulimits.memlock: -1 in compose, --ulimit memlock=-1 with docker run, or on Kubernetes raise the" \
         "container runtime's default limit (containerd's or CRI-O's)" >&2
    exit 2
  fi
fi
# A GR8_HOME host directory: docker creates a missing one owned by root, and a bare-metal G6_HOME has a real bin/,
# where the link below would land inside it and the host's engine would run.
if [ ! -w "$home" ]; then
  echo "gr8: $home is not writable by uid $(id -u): a GR8_HOME host directory must be owned by it" \
       "(chown $(id -u):$(id -g) DIR)" >&2
  exit 2
fi
if [ -d "$home/bin" ] && [ ! -L "$home/bin" ]; then
  echo "gr8: $home/bin is a directory (a bare-metal G6_HOME?): point GR8_HOME at a directory of its own" >&2
  exit 2
fi
mkdir -p "$home/out"
ln -sfn /opt/gr8/bin "$home/bin"

if [ "$cmd" = up ]; then
  exec > >(tee -a "$home/out/supervisor.log") 2>&1
  exec /opt/gr8/gr8 "$model" up --foreground "$@"
fi
exec /opt/gr8/gr8 "$model" "$cmd" "$@"
