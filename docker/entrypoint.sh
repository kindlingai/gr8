#!/bin/bash
# The image's entrypoint: ./gr8 glm53 COMMAND [OPTIONS], with no command meaning `up`. It first sets up this box's
# working directory (G6_HOME): out/ exists, and bin/ links to the image's engine for the TP (bin/ at TP=4, bin3/ at
# TP=3, bin6/ at TP=6), read with the launcher's own option parser from the environment and the command line.
#
# `up` runs the supervisor in the foreground, so the container lives exactly as long as this box's part of the cluster.
# Its exit status is the launcher's: 0 after a `down` or a stop, 1 when a rank failed, 2 for bad options, 4 when the
# handshake refused the boot. Its output goes to docker logs and to out/supervisor.log, which `logs supervisor` reads.
set -euo pipefail
home=${G6_HOME:?G6_HOME is not set}
[ $# -gt 0 ] || set -- up
cmd=$1; shift

tp=$(python3 - "$@" <<'EOF'
import sys
sys.path.insert(0, "/opt/gr8/tools/launch")
import options
try:
    o = options.parse(sys.argv[1:])
except options.OptionError:
    sys.exit()          # the launcher reports it
print(o.get("TP") or len(options.split_list(o.get("PEERS"))) or "")
EOF
)
case ${tp:-4} in
  4) bin=bin ;;
  3) bin=bin3 ;;
  6) bin=bin6 ;;
  *) echo "gr8: TP=$tp: want 4, 3 or 6" >&2; exit 2 ;;
esac
mkdir -p "$home/out"
ln -sfn "/opt/gr8/$bin" "$home/bin"

if [ "$cmd" = up ]; then
  exec > >(tee -a "$home/out/supervisor.log") 2>&1
  exec /opt/gr8/gr8 glm53 up --foreground "$@"
fi
exec /opt/gr8/gr8 glm53 "$cmd" "$@"
