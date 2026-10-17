#!/bin/bash
# tp8-window.sh restore paths against mocks: ssh runs the command here, gr8/curl/systemctl/memclean are stubs that log.
# No host, no GPU. Run: bash scripts/test_tp8_window.sh
# ok/bad always return 0, so A && ok || bad is safe here; the sed text is literal on purpose
# shellcheck disable=SC2015,SC2016
set -u
W=$(readlink -f "$(dirname "$0")/tp8-window.sh")
pass=0 failn=0
ok() { echo "ok   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL $1"; failn=$((failn + 1)); }

setup() {
  T=$(mktemp -d); M=$T/mock; mkdir -p "$M/bin" "$T/tp8" "$T/a" "$T/rd"
  export MOCK=$M PATH="$M/bin:$PATH"
  cat > "$M/bin/ssh" <<'EOF'
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=${1#*@}; shift
echo "ssh $host $*" >> "$MOCK/calls"
MOCK_HOST=$host exec bash -c "$*"
EOF
  cat > "$M/bin/curl" <<'EOF'
#!/bin/bash
[ -e "$MOCK/up" ] && printf 200 || printf 000
EOF
  cat > "$M/bin/systemctl" <<'EOF'
#!/bin/bash
echo "systemctl $*" >> "$MOCK/calls"
EOF
  for h in tp8 a; do
    cat > "$T/$h/gr8" <<'EOF'
#!/bin/bash
echo "gr8 $MOCK_HOST $(basename "$PWD") $*" >> "$MOCK/calls"
case $2 in up) touch "$MOCK/up" ;; down) [ "$(basename "$PWD")" = a ] && rm -f "$MOCK/up" ;; esac
exit 0
EOF
    mkdir -p "$T/$h/out"
  done
  chmod +x "$M"/bin/* "$T"/tp8/gr8 "$T"/a/gr8
  echo hello > "$M/canary.good"
  cat > "$T/rd/window.conf" <<EOF
SSH_USER=u
RECOVERY_HOST=rec
RECOVERY_ROOT=$T
SYSTEMD_RUN=true
ALERT_CMD='echo "\$1" > $M/alerted'
TP8_BOXES=(b0 b1 b2 b3 b4 b5 b6 b7)
TP8_HOME=$T/tp8
TP8_ENV=$T/tp8.env
TP8_URL=http://b0:8100
TP8_LIFE=60
TP8_READY=60
SERVES=(A)
A_BOXES=(b1 b2 b3 b0)
A_HEAD=b0
A_HOME=$T/a
A_ENV=$T/a.env
A_ENV_SHA=x
A_URL=http://b0:8000
A_UP=together
A_GAP=0
A_CANARY_HOST=b0
A_CANARY_CMD='cp $M/canary.src $M/canary.out'
A_CANARY_OUT=$M/canary.out
A_CANARY_MD5=$(md5sum < "$M/canary.good" | cut -d' ' -f1)
MEMCLEAN='echo memclean >> $M/calls'
MEMCLEAN_MD5=x
MEMCLEAN_PATH=x
PREP_CMD=true
VERIFY_WEIGHTS_CMD=true
ERR_RE=FAIL
SERVE_HEALTH_S=5
B_P0=1; B_ARM=1; B_P1=1; B_P2=1; B_P3=30; B_P4=1; B_P5=1; B_P6=1; B_RESTORE=1; B_MARGIN=1
EOF
  cp "$W" "$T/rd/tp8-window.sh"
  echo run-test > "$T/rd/RUN_ID"
  (cd "$T/rd" && sha256sum tp8-window.sh window.conf > SHA256SUMS)
  : > "$M/calls"
}
restore() { bash "$T/rd/tp8-window.sh" restore "$T/rd" > "$T/out" 2>&1; }

# 1. a serve that already serves with its canary is left alone; the test cluster is stopped box by box
setup; touch "$M/up"; cp "$M/canary.good" "$M/canary.src"
restore
[ -e "$T/rd/RESTORED_OK" ] && ok "healthy serve: RESTORED_OK" || bad "healthy serve: RESTORED_OK"
grep -q "gr8 b. a " "$M/calls" && bad "healthy serve was stopped" || ok "healthy serve not stopped"
[ "$(grep -c 'gr8 b. tp8 glm53 down --local' "$M/calls")" = 8 ] && ok "8 local TP8 downs" || bad "8 local TP8 downs"
grep -q "glm53 down @" "$M/calls" && bad "a cluster-wide down was sent" || ok "no cluster-wide down"
grep -q "systemctl stop tp8-restore-run-test.timer" "$M/calls" && ok "timer stopped" || bad "timer stopped"
[ -e "$T/rd/FENCE" ] && ok "fence set" || bad "fence set"

# 2. a second restore after success does nothing
: > "$M/calls"; restore
[ ! -s "$M/calls" ] && grep -q "already RESTORED_OK" "$T/out" && ok "repeat restore is a no-op" || bad "repeat restore is a no-op"

# 3. a down serve is restarted: local down x4, memclean, up, health, canary
setup; cp "$M/canary.good" "$M/canary.src"
restore
[ -e "$T/rd/RESTORED_OK" ] && ok "down serve: RESTORED_OK" || bad "down serve: RESTORED_OK"
[ "$(grep -c 'gr8 b. a glm53 down --local' "$M/calls")" = 4 ] && ok "serve local down x4" || bad "serve local down x4"
[ "$(grep -c 'gr8 b. a glm53 up' "$M/calls")" = 4 ] && ok "serve up x4" || bad "serve up x4"

# 4. a canary mismatch: 2 attempts, then ALERT and the alert command; no third attempt on a repeat call
setup; echo other > "$M/canary.src"
restore
[ -e "$T/rd/ALERT" ] && [ -e "$M/alerted" ] && ok "bad canary: ALERT raised" || bad "bad canary: ALERT raised"
[ "$(grep -c 'gr8 b0 a glm53 up' "$M/calls")" = 2 ] && ok "max 2 attempts" || bad "max 2 attempts"
: > "$M/calls"; restore
[ ! -s "$M/calls" ] && ok "no retry after ALERT" || bad "no retry after ALERT"

# 5. a concurrent restore (the timer while the primary-started one runs) backs off
setup; touch "$M/up"; cp "$M/canary.good" "$M/canary.src"
exec 8> "$T/rd/restore.lock"; flock 8
restore; exec 8>&-
grep -q "another restore owns" "$T/out" && [ ! -e "$T/rd/RESTORED_OK" ] && ok "lock holder excludes a second restore" \
  || bad "lock holder excludes a second restore"

# 6. a changed copy of the conf is refused with ALERT, before any action
setup; echo "# edit" >> "$T/rd/window.conf"
restore
[ -e "$T/rd/ALERT" ] && [ ! -s "$M/calls" ] && ok "tampered conf: ALERT, nothing run" || bad "tampered conf: ALERT, nothing run"

# 7. a TP8 box that is not verified down blocks every serve start
setup; cp "$M/canary.good" "$M/canary.src"
sed -i 's/^exit 0$/[ "$MOCK_HOST" = b5 ] \&\& [ "$2" = verify-down ] \&\& exit 1; exit 0/' "$T/tp8/gr8"
restore
[ -e "$T/rd/ALERT" ] && ! grep -q "gr8 b. a glm53 up" "$M/calls" && ok "TP8 not down: no serve start, ALERT" \
  || bad "TP8 not down: no serve start, ALERT"

# 8. a hooked (non-gr8) serve: its own stop/prep/check/up/health/smoke commands, up on its head only
hooked() {   # replace serve A with a hooked serve B
  sed -i '/^SERVES=/,/^A_CANARY_MD5=/d' "$T/rd/window.conf"
  cat >> "$T/rd/window.conf" <<HOOK
SERVES=(B)
B_BOXES=(b6 b5 b0 b7)
B_UP_BOXES=(b6)
B_HEAD=b6
B_HOME=$T/a
B_ENV=$T/b.env
B_ENV_SHA=x
B_URL=http://b6:8888
B_HEALTH=/v1/models
B_UP=seq
B_GAP=0
B_STOP_CMD='echo "bstop \$MOCK_HOST" >> $M/calls'
B_PREP='echo "bprep \$MOCK_HOST" >> $M/calls'
B_PRE_UP_CHECK='echo "bpre \$MOCK_HOST" >> $M/calls; [ ! -e $M/prefail ]'
B_POST_CHECK='echo "bpost \$MOCK_HOST" >> $M/calls'
B_UP_CMD='echo "bup \$MOCK_HOST" >> $M/calls; touch $M/up'
B_CANARY_HOST=b6
B_CANARY_CMD='echo bsmoke >> $M/calls; [ -e $M/smoke.ok ]'
B_CANARY_MD5=exit0
HOOK
  (cd "$T/rd" && sha256sum tp8-window.sh window.conf > SHA256SUMS)
}
setup; hooked; touch "$M/smoke.ok"
restore
[ -e "$T/rd/RESTORED_OK" ] && ok "hooked serve: RESTORED_OK" || bad "hooked serve: RESTORED_OK"
[ "$(grep -c '^bstop' "$M/calls")" = 4 ] && [ "$(grep -c '^bprep' "$M/calls")" = 4 ] && ok "hooked stop+prep x4" \
  || bad "hooked stop+prep x4"
[ "$(grep -c '^bup' "$M/calls")" = 1 ] && grep -q '^bup b6$' "$M/calls" && ok "hooked up on UP_BOXES only" || bad "hooked up on UP_BOXES only"
[ "$(grep -c '^bpre ' "$M/calls")" = 4 ] && [ "$(grep -c '^bpost' "$M/calls")" = 4 ] && grep -q '^bsmoke' "$M/calls" \
  && ok "hooked checks x4 + smoke" || bad "hooked checks x4 + smoke"
grep -q "gr8 b. a " "$M/calls" && bad "hooked serve ran gr8" || ok "hooked serve ran no gr8"

# 9. hooked serve: a failed pre-up check never starts it; a failed smoke is an ALERT after 2 attempts
setup; hooked; touch "$M/smoke.ok" "$M/prefail"
restore
[ -e "$T/rd/ALERT" ] && ! grep -q '^bup' "$M/calls" && ok "pre-up check fail: no up, ALERT" || bad "pre-up check fail: no up, ALERT"
setup; hooked
restore
[ -e "$T/rd/ALERT" ] && [ "$(grep -c '^bup' "$M/calls")" = 2 ] && ok "smoke fail: 2 attempts, ALERT" || bad "smoke fail: 2 attempts, ALERT"

echo "passed $pass, failed $failn"
[ $failn = 0 ]
