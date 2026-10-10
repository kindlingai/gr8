#!/bin/bash
# G6_XTRACE stage-0 DRIVER (measurement only; the engine is not changed by this script).
# Runs one command (a g6m4 rank, m2b, ...) with G6_XTRACE=1 and samples, per rail, the NIC counters around it:
#   - ethtool -S: RDMA TX/RX bytes + packets (vport), phy bytes, pause, discards, out_of_buffer, netdev drops
#   - RDMA hw_counters (sysfs ports/1/hw_counters): local_ack_timeout_err, req_transport_retries_exceeded,
#     req_rnr_retries_exceeded, rnr_nak_retry_err, out_of_sequence, packet_seq_err, implied_nak_seq_err,
#     duplicate_request, roce_adp_retrans, roce_adp_retrans_to, roce_slow_restart, req_cqe_error
#   - the expected SGE bytes per WR for the decode collective at M rows: M * 6144 * 2 + 8 (payload + 8-B tail)
# Output dir: one CSV of per-interval deltas per rail (ctr.r<rail>.csv), a summary of whole-run deltas, the command log,
# and the engine's own dump (g6xtrace.r<rank>.f<n>.txt via G6_XTRACE_OUT). Per-step deltas: set --interval to the step
# period, or use --steps N with a known step count to get the mean per step from the whole-run delta.
#
#   tools/xtrace_driver.sh --out DIR [--rails "enp1s0f0np0 enP2p1s0f0np0"] [--interval 0.05] [--steps N] [--rows M] \
#                          [--quit-after S] -- CMD ARGS...
#   --quit-after S: touch $OUT/quit after S seconds (G6_XTRACE_QUIT: the engine dumps once, outside the timed loop)
set -u
OUT=""; RAILS="enp1s0f0np0 enP2p1s0f0np0"; IVL=0.05; STEPS=0; ROWS=12; QA=0
while [ $# -gt 0 ]; do case "$1" in
  --out) OUT=$2; shift 2;; --rails) RAILS=$2; shift 2;; --interval) IVL=$2; shift 2;;
  --steps) STEPS=$2; shift 2;; --rows) ROWS=$2; shift 2;; --quit-after) QA=$2; shift 2;;
  --) shift; break;; *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$OUT" ] && [ $# -gt 0 ] || { sed -n 2,18p "$0"; exit 2; }
mkdir -p "$OUT"; rm -f "$OUT/quit"
HW="local_ack_timeout_err req_transport_retries_exceeded req_rnr_retries_exceeded rnr_nak_retry_err out_of_sequence packet_seq_err implied_nak_seq_err duplicate_request roce_adp_retrans roce_adp_retrans_to roce_slow_restart req_cqe_error"
ET="tx_vport_rdma_unicast_bytes rx_vport_rdma_unicast_bytes tx_vport_rdma_unicast_packets rx_vport_rdma_unicast_packets tx_bytes_phy rx_bytes_phy tx_pause_ctrl_phy rx_pause_ctrl_phy rx_discards_phy tx_discards_phy rx_out_of_buffer tx_queue_dropped"
ibdev_of() { for d in /sys/class/infiniband/*; do [ -e "$d/device/net/$1" ] && { basename "$d"; return; }; done; }
snap() {   # one line: values of ET then HW then netdev rx/tx_dropped for netdev $1
  local n=$1 ib; ib=$(ibdev_of "$n")
  local es; es=$(ethtool -S "$n" 2>/dev/null)
  local v=""
  for k in $ET; do v="$v,$(echo "$es" | awk -v k="$k:" '$1==k{print $2; f=1} END{if(!f) print 0}')"; done
  for k in $HW; do v="$v,$(cat /sys/class/infiniband/$ib/ports/1/hw_counters/$k 2>/dev/null || echo 0)"; done
  v="$v,$(awk -v n="$n:" '$1==n{print $5","$13; f=1} END{if(!f) print "0,0"}' /proc/net/dev)"   # netdev rx/tx drop
  echo "${v#,}"
}
hdr="t_ns,$(echo $ET $HW netdev_rx_dropped netdev_tx_dropped | tr ' ' ',')"
i=0; for n in $RAILS; do echo "$hdr" > "$OUT/ctr.r$i.csv"; echo "rail $i netdev $n ibdev $(ibdev_of $n)" >> "$OUT/rails.txt"; i=$((i+1)); done
declare -A PREV FIRST
i=0; for n in $RAILS; do PREV[$i]=$(snap $n); FIRST[$i]=${PREV[$i]}; i=$((i+1)); done
delta() { awk -F, -v a="$1" -v b="$2" 'BEGIN{n=split(a,x,",");split(b,y,",");s="";for(k=1;k<=n;k++)s=s (k>1?",":"") (y[k]-x[k]);print s}'; }
echo "SGE bytes per WR (decode all-reduce, M=$ROWS rows): $((ROWS*6144*2+8)) (payload $((ROWS*6144*2)) + 8-B tail; one WR per peer per collective)" > "$OUT/summary.txt"
G6_XTRACE=1 G6_XTRACE_OUT="$OUT" G6_XTRACE_QUIT="$OUT/quit" "$@" > "$OUT/cmd.log" 2>&1 &
PID=$!
T0=$(date +%s.%N)
while kill -0 $PID 2>/dev/null; do
  sleep "$IVL"
  t=$(date +%s%N); i=0
  for n in $RAILS; do cur=$(snap $n); echo "$t,$(delta "${PREV[$i]}" "$cur")" >> "$OUT/ctr.r$i.csv"; PREV[$i]=$cur; i=$((i+1)); done
  if [ "$QA" != 0 ] && [ ! -e "$OUT/quit" ] && awk -v a="$T0" -v b="$(date +%s.%N)" -v q="$QA" 'BEGIN{exit !(b-a>=q)}'; then touch "$OUT/quit"; fi
done
wait $PID; RC=$?
i=0; for n in $RAILS; do
  cur=$(snap $n); d=$(delta "${FIRST[$i]}" "$cur")
  echo "rail $i ($n) whole-run delta:" >> "$OUT/summary.txt"
  paste -d= <(echo $ET $HW netdev_rx_dropped netdev_tx_dropped | tr ' ' '\n') <(echo "$d" | tr ',' '\n') | sed 's/^/  /' >> "$OUT/summary.txt"
  if [ "$STEPS" -gt 0 ]; then echo "  per step (/$STEPS): $(echo "$d" | awk -F, -v s=$STEPS '{printf "tx_rdma_B=%.0f rx_rdma_B=%.0f tx_pkts=%.1f rx_pkts=%.1f", $1/s, $2/s, $3/s, $4/s}')" >> "$OUT/summary.txt"; fi
  i=$((i+1)); done
echo "rc=$RC" >> "$OUT/summary.txt"
cat "$OUT/summary.txt"; ls "$OUT"/g6xtrace.* 2>/dev/null
exit $RC
