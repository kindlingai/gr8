# sourced by the other scripts: loads config/cluster.env and defines helpers
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG=${G6_CLUSTER:-$REPO/config/cluster.env}
[ -f "$CFG" ] || { echo "missing $CFG (copy config/cluster.env.example)" >&2; exit 2; }
# shellcheck disable=SC1090
. "$CFG"
# the scripts run cd $G6_HOME, rm -rf s2, touch $G6_HOME/s2/quit on every box: an empty one means the home dir or /
if [ "${G6_HOME+set}" = set ] && [ -z "$G6_HOME" ]; then echo "$CFG: G6_HOME is empty" >&2; exit 2; fi
read -r -a HOST <<< "$HOSTS"
NRANK=${#HOST[@]}
# one ssh connection per node, reused by every call (the launcher polls each rank every second)
S="ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=$HOME/.ssh/g6-%C -o ControlPersist=120"
on() { local r=$1; shift; $S "${HOST[$r]}" "$@"; }   # on <rank> <cmd>: run a command on that rank's node
