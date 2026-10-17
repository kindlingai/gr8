#!/bin/bash
# tp8-window.sh's stop and down commands against the REAL CLI, no mock: scratch copies of this checkout (the TP8 tree)
# and of MAIN_REV (the C4 serve tree) with nothing running and free ports. Each row comes from `tp8-window.sh cmds` on a
# scratch conf, so the strings are the ones a window sends. Run on any box with python3 and git, not a serving tree:
#   bash scripts/test_tp8_window_cli.sh
# shellcheck disable=SC2015
set -u
REPO=$(readlink -f "$(dirname "$0")/..")
MAIN_REV=${MAIN_REV:-1f8a84f}
pass=0 failn=0
ok() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; failn=$((failn + 1)); }
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
mkdir -p "$S/tp8" "$S/main"
(cd "$REPO" && tar cf - gr8 tools scripts) | tar xf - -C "$S/tp8"
git -C "$REPO" archive "$MAIN_REV" gr8 tools scripts | tar xf - -C "$S/main"
for t in tp8 main; do mkdir -p "$S/$t/out"; printf 'G6_HOME=%s\nPORT=18100\nCONTROL_PORT=17350\nS2PORT=17341\n' "$S/$t" > "$S/$t.env"; done
cat > "$S/w.conf" <<EOF
SSH_USER=u; RECOVERY_HOST=r; RECOVERY_ROOT=$S; SYSTEMD_RUN=true; ALERT_CMD=true
TP8_BOXES=(b0 b1 b2 b3 b4 b5 b6 b7); TP8_HOME=$S/tp8; TP8_ENV=$S/tp8.env; TP8_URL=x; TP8_LIFE=1; TP8_READY=1
SERVES=(M T)
M_BOXES=(b0); M_HEAD=b0; M_HOME=$S/main; M_ENV=$S/main.env; M_ENV_SHA=x; M_URL=x; M_UP=together
M_CANARY_HOST=b0; M_CANARY_CMD=true; M_CANARY_MD5=exit0
T_BOXES=(b0); T_HEAD=b0; T_HOME=$S/tp8; T_ENV=$S/tp8.env; T_ENV_SHA=x; T_URL=x; T_UP=together
T_CANARY_HOST=b0; T_CANARY_CMD=true; T_CANARY_MD5=exit0
MEMCLEAN=true; MEMCLEAN_MD5=x; MEMCLEAN_PATH=x; PREP_CMD=true; VERIFY_WEIGHTS_CMD=true; ERR_RE=x
B_P0=1; B_ARM=1; B_P1=1; B_P2=1; B_P3=1; B_P4=1; B_P5=1; B_P6=1; B_RESTORE=1; B_MARGIN=1
EOF
bash "$REPO/scripts/tp8-window.sh" cmds "$S/w.conf" > "$S/cmds" || { bad "cmds"; exit 1; }
row() { awk -F'\t' -v n="$1" '$1 == n {print $3}' "$S/cmds"; }
run() { (cd / && timeout 120 bash -c "$1") > "$S/out" 2>&1; }
# the real wrapper
(cd "$S/tp8" && ./gr8 glm53 --help > /dev/null) && ok "tp8 ./gr8 glm53 --help" || bad "tp8 ./gr8 glm53 --help"
(cd "$S/main" && ./gr8 glm53 --help > /dev/null) && ok "main ./gr8 glm53 --help" || bad "main ./gr8 glm53 --help"
(cd "$S/tp8" && ./gr8 glm53 verify-down > /dev/null 2>&1); [ $? = 2 ] && ok "tp8 ./gr8 does not route verify-down (rc 2)" \
  || bad "tp8 ./gr8 verify-down"
# the TP8 local stop (restore) and a tp8-ring serve's default local stop: real launch.py, exit 0, verified
for n in tp8-stop T-stop; do
  c=$(row $n); run "$c"; rc=$?
  [ $rc = 0 ] && grep -q "^--local: stopping this box only" "$S/out" && grep -q "down, verified" "$S/out" \
    && ok "$n: rc 0, down --local + verified [$c]" || { bad "$n: rc $rc"; cat "$S/out"; }
done
# the head down of a main serve (P1): the real main wrapper, exit 0
c=$(row M-down); run "$c"; rc=$?
[ $rc = 0 ] && grep -q "nothing was running" "$S/out" && ok "M-down (main head down): rc 0 [$c]" || { bad "M-down: rc $rc"; cat "$S/out"; }
# the default local stop on main fails: a main serve needs a STOP_CMD hook (window2.conf C4_STOP_CMD)
c=$(row M-stop); run "$c"; rc=$?
[ $rc != 0 ] && grep -q "unknown option --local" "$S/out" && ok "M-stop default on main fails (rc $rc): needs a hook" \
  || { bad "M-stop default on main: rc $rc"; cat "$S/out"; }
# a stale handshake (an older boot's API port, now a live serve's) fails the verify; moving out/ aside fixes it
c=$(row tp8-p0-verify); run "$c" && grep -q "down, verified" "$S/out" && ok "tp8-p0-verify: rc 0 [$c]" || { bad "tp8-p0-verify"; cat "$S/out"; }
python3 -c 'import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", 18999)); s.listen(); time.sleep(20)' & L=$!
sleep 0.5
printf '{"port": "18999", "control_port": 17350, "s2port": "17341", "rank": 0, "members": ["127.0.0.1"]}' > "$S/tp8/out/handshake.json"
run "$c"; rc=$?
[ $rc = 1 ] && grep -q "listens on 127.0.0.1:18999" "$S/out" && ok "stale handshake + live port: NOT DOWN (rc 1)" || { bad "stale handshake: rc $rc"; cat "$S/out"; }
mv "$S/tp8/out" "$S/tp8/out.w1"; mkdir "$S/tp8/out"
run "$c" && grep -q "down, verified" "$S/out" && ok "out/ moved aside: down, verified" || { bad "out/ moved aside"; cat "$S/out"; }
kill $L 2>/dev/null; wait 2>/dev/null
echo "passed $pass, failed $failn"
[ $failn = 0 ]
