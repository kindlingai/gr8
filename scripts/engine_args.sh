# The engine command line. The per-box supervisor (tools/launch) sources this file. ./gr8 knobs lists
# the knobs. docs/SETTINGS.md explains every engine variable and flag.
#
# GR8_MODEL picks the model: glm53 (unset), glm53f, GLM-5.3-Flash (bin/g6m4_flash, TP 4 or 2), or pseudo, the pseudo
# engine (bin/g6pseudo, frontend/pseudo-engine).
#
#   engine_defaults          needs TP (4, 6, 3 or 5); sets every knob that is unset to its default for this TP. An unset CAP
#                            follows SHARE_LIMIT, so the KV pool keeps the same physical pages per rank (share_cap below)
#   engine_env               sets ENGINE_ENV, the engine environment for every rank (K=V words), and ENGINE_BIN, the
#                            engine under bin/ for KV_FORMAT (g6m4, or g6m4-int4 for int4)
#   engine_args R RANK_PACKS AUX_PACKS HOME [EPOCH]
#                            prints the arguments that follow `--proxy-cpu N` for rank R. RANK_PACKS holds
#                            rank<R>.{manifest.json,pack} (unused with V2_DENSE: the release is the model), AUX_PACKS
#                            the drafter and QO packs, HOME is GR8_WORK_DIR
#                            (the s2/ directory), EPOCH the S2 epoch. SET_ID, when set, adds --set-id.
#   flash_donor_budget [KIB] GLM-5.3-Flash: the whole GiB this box lends a KV node next to the serve (below)

# share_cap P: the CAP that gives P physical KV pages per rank at this SHARE_LIMIT. Each rank holds the first
# SHARE_LIMIT tokens of a session (64 per page) and 1/TP of the rest.
share_cap() {
  [ $(( SHARE_LIMIT / 64 )) -le "$1" ] || { echo "engine_defaults: SHARE_LIMIT $SHARE_LIMIT is more than the $(( $1 * 64 )) tokens a rank's KV pool holds at TP=$TP" >&2; return 2; }
  echo $(( SHARE_LIMIT + ($1 - SHARE_LIMIT / 64) * 64 * TP ))
}

# int4_pages FP8_PAGES: the pages a rank holds with the int4 KV row (KV_FORMAT=int4) in the memory FP8_PAGES take with the
# fp8 row, everything that grows with CAP counted (docs/KV4.md, Capacity). TP=4 is measured on boots; TP=3 and TP=6 are
# the model's estimates.
int4_pages() {
  case $TP:$1 in
    4:2280) echo 2390 ;;   # measured against fp8 at 1,920 pages, and not re-measured since the drafter's prefill shares
                           # the target's scratch: int4 keeps the memory of 1,981 fp8 pages
    3:1941) echo 2480 ;;
    6:2026) echo 2440 ;;
    6:3307) echo 3940 ;;
    *) echo "engine_defaults: no int4 page count for $1 fp8 pages at TP=$TP: set CAP" >&2; return 2 ;;
  esac
}
# pool_cap P: the default CAP from P fp8 pages a rank, or their int4 equivalent
pool_cap() {
  local p=$1
  if [ "${KV_FORMAT:-fp8}" = int4 ]; then p=$(int4_pages "$1") || return 2; fi
  share_cap "$p"
}

# EXTRA_ENV with each old G6_ name as its GR8_ name, so an override under the old name and a default here are one
# variable to the engine (the knob table refuses both names with different values)
read -ra _extra <<< "${EXTRA_ENV:-}"
EXTRA_ENV=""
for _w in ${_extra[@]+"${_extra[@]}"}; do
  [[ $_w == G6_* ]] && _w=GR8_${_w#G6_}
  EXTRA_ENV="$EXTRA_ENV${EXTRA_ENV:+ }$_w"
done

# tp4_default KNOB VALUE: KNOB=VALUE unless KNOB is set or EXTRA_ENV already carries GR8_KNOB
tp4_default() {
  [ -n "${!1:-}" ] && return 0
  case " ${EXTRA_ENV:-} " in *" GR8_$1="*) return 0 ;; esac
  printf -v "$1" %s "$2"
}

# GLM-5.3-Flash under --serve takes no CP, prefix reuse, DFlash or PF_OWN (engine/runtime/g6m4.cu refuses them).
# MTP=1: --serve needs a draft, and the KDA state does not roll back a rejected draft yet (docs/FLASH.md).
# SESS: 1 (the default) serves one request at a time; 2..8 share decode steps and the CAP pool, each with its own KDA
# state slots (docs/FLASH.md, Several sessions), and a round of every session, SESS x (MTP + 1) rows, must fit the
# engine's 32. CAP is a 262,144-token request plus a page for the guard rows (2 at MTP=1). Only the 11 DSA layers and
# the MTP layer keep KV, 1,664 B a token each, and without CP every rank holds the whole context: about 5.2 GB a rank at
# any TP (docs/FLASH.md, Context). With one session SHARE_LIMIT only sets the front end's default max_tokens.
flash_defaults() {
  case $TP in 4|3|2) ;; *) echo "engine_defaults: TP=$TP; GLM-5.3-Flash wants 4, 3 or 2" >&2; return 2 ;; esac
  local k
  for k in DFLASH REUSE PF_OWN SPEC_ADAPT; do
    [ "${!k:-0}" = 0 ] || { echo "engine_defaults: $k=${!k}: GLM-5.3-Flash has no $k" >&2; return 2; }
  done
  SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-262208}; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-1}; SESS=${SESS:-1}
  case $SESS in [1-8]) ;; *) echo "engine_defaults: SESS=$SESS: GLM-5.3-Flash serves 1 to 8 sessions" >&2; return 2 ;; esac
  [ "$SESS" = 1 ] || [ $((SESS * (MTP + 1))) -le 32 ] ||
    { echo "engine_defaults: SESS=$SESS MTP=$MTP: a round of $((SESS * (MTP + 1))) rows; GLM-5.3-Flash steps hold 32" >&2; return 2; }
  S2PORT=${S2PORT:-7341}; DFLASH=0; SPEC_ADAPT=0; PF_OWN=0; REUSE=0
  flash_donor_defaults
}

# A KV node next to the Flash serve (D-331 M7): FLASH_KV_DONATE_GIB > 0 lends at most that many GiB of this box to a
# ./gr8 kv up --lazy-lock node once the serve has booted. FLASH_KV_GROWTH_GIB is what the serve may still allocate after
# boot; it has no default because it is not measured yet. FLASH_KV_GROUP is the mentat group the node serves (the cluster
# that borrows from it, not this serve's), also without a default. FLASH_CAP_POLICY=protected: the configured CAP's memory is
# never lent. borrow-idle (needs a donation): the KV pool in chunks (GR8_GLM53F_KV_CHUNKS=1), every chunk but one lent to
# the node once the serve has been idle FLASH_KV_IDLE_S (default 60), recalled before a request that needs it is
# admitted (the request waits for the recall; s2/flash_lend.h).
num_ok() { [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]; }
flash_donor_defaults() {
  FLASH_KV_DONATE_GIB=${FLASH_KV_DONATE_GIB:-0}; FLASH_CAP_POLICY=${FLASH_CAP_POLICY:-protected}
  num_ok "$FLASH_KV_DONATE_GIB" || { echo "engine_defaults: FLASH_KV_DONATE_GIB=$FLASH_KV_DONATE_GIB: want GiB" >&2; return 2; }
  case $FLASH_CAP_POLICY in
    protected) ;;
    borrow-idle)
      [ "$FLASH_KV_DONATE_GIB" = 0 ] && { echo "engine_defaults: FLASH_CAP_POLICY=borrow-idle needs FLASH_KV_DONATE_GIB > 0 (a KV node to lend to)" >&2; return 2; }
      FLASH_KV_IDLE_S=${FLASH_KV_IDLE_S:-60}
      [[ $FLASH_KV_IDLE_S =~ ^[0-9]+$ ]] && [ "$FLASH_KV_IDLE_S" -ge 1 ] || { echo "engine_defaults: FLASH_KV_IDLE_S=$FLASH_KV_IDLE_S: want whole seconds >= 1" >&2; return 2; } ;;
    *) echo "engine_defaults: FLASH_CAP_POLICY=$FLASH_CAP_POLICY: want protected or borrow-idle" >&2; return 2 ;;
  esac
  [ "$FLASH_KV_DONATE_GIB" = 0 ] && return 0
  num_ok "${FLASH_KV_GROWTH_GIB:-}" || { echo "engine_defaults: FLASH_KV_DONATE_GIB=$FLASH_KV_DONATE_GIB needs FLASH_KV_GROWTH_GIB, the GiB the serve may still allocate after boot (not measured yet)" >&2; return 2; }
  [ -n "${FLASH_KV_GROUP:-}" ] || { echo "engine_defaults: FLASH_KV_DONATE_GIB=$FLASH_KV_DONATE_GIB needs FLASH_KV_GROUP, the mentat group of the cluster that borrows the node (not this serve's)" >&2; return 2; }
  return 0
}
# flash_donor_budget [KIB]: FLASH_KV_DONATE_GIB, or less when MemAvailable (KIB, else /proc/meminfo, read after the
# serve booted) - FLASH_KV_GROWTH_GIB - KV_MIN_FREE_GB (the node's own floor, 10) is less; whole GiB, 0 = no node.
flash_donor_budget() {
  local kib=${1:-}
  [ "${FLASH_KV_DONATE_GIB:-0}" = 0 ] && { echo 0; return 0; }
  [ -n "$kib" ] || kib=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  num_ok "$kib" || { echo "flash_donor_budget: MemAvailable $kib: want KiB" >&2; return 2; }
  awk -v kib="$kib" -v g="${FLASH_KV_GROWTH_GIB:-0}" -v r="${KV_MIN_FREE_GB:-10}" -v w="$FLASH_KV_DONATE_GIB" \
    'BEGIN { b = kib / 1048576 - g - r; if (b > w) b = w; if (b < 0) b = 0; printf "%d\n", b }'
}

# The pseudo engine: no KV pool to size, MTP, drafter or fabric. CAP bounds a request's prompt plus max_tokens.
pseudo_defaults() {
  SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-202752}; READY_TMO=${READY_TMO:-60}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
  MTP=0; DFLASH=0; SPEC_ADAPT=0; PF_OWN=0; REUSE=0
}

engine_defaults() {
  [ "${GR8_MODEL:-glm53}" = glm53f ] && { flash_defaults; return; }
  [ "${GR8_MODEL:-glm53}" = pseudo ] && { pseudo_defaults; return; }
  case $TP in
    4) SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-$(pool_cap 2280)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       DFLASH=${DFLASH:-1}; SPEC_ADAPT=${SPEC_ADAPT:-$DFLASH}; PF_OWN=${PF_OWN:-4}; REUSE=${REUSE:-1}
       # the window-2 fc1 split on every MoE layer at M > 1 (bit-exact, measured at TP=4); 0 turns it off, and
       # GR8_TR3_W2ALL in EXTRA_ENV keeps its own value
       case " ${EXTRA_ENV:-} " in *" GR8_TR3_W2ALL="*) ;; *) TR3_W2ALL=${TR3_W2ALL:-1} ;; esac
       # the decode options measured at TP=4 (#85 draft head, #86 shared-expert overlap, #87 attention set, the decode
       # buckets and indexer, S2 scheduling, prefill selx); 0 turns one off, and a GR8_ name in EXTRA_ENV keeps its own value
       tp4_default ROUTE_OVL 1; [ "$MTP" = 0 ] || tp4_default DRAFT_HEAD_I4 1
       tp4_default ATTN_SC2 4; tp4_default ATTN_SC2_TR 2; tp4_default ATTN_CTX2 4; tp4_default ATTN_CTX3 2
       tp4_default ATTN_SEL2 16; tp4_default IDX_GRP 16; tp4_default ATTN_QT 1; tp4_default ATTN_FRONT 1
       tp4_default IDX_VEC 1; tp4_default DEC_BK 512; [ "$MTP" = 0 ] || tp4_default S2_ADAPT 1
       tp4_default S2_DECODE_PF 4096; tp4_default S2_ADMIT_HOLD 1; tp4_default PF_SELX_CTX 53248 ;;
    6) SHARE_LIMIT=${SHARE_LIMIT:-131072}
       if [ "${KV_FORMAT:-fp8}" = int4 ]; then   # 614,400 is 3,307 fp8 pages a rank at 131072 and 2,026 at 32768
         if [ "$SHARE_LIMIT" = 131072 ]; then CAP=${CAP:-$(pool_cap 3307)} || return 2; else CAP=${CAP:-$(pool_cap 2026)} || return 2; fi
       else [[ "$SHARE_LIMIT" = 32768 || "$SHARE_LIMIT" = 131072 ]] && CAP=${CAP:-614400} || CAP=${CAP:-$(share_cap 2026)} || return 2; fi; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       # DFLASH=1: the DFlash2 drafter with its body replicated on every rank (dflash.cuh dfl::REP; the drafter
       # converted per rank in the replicated layout, --drafter). Off unless set
       DFLASH=${DFLASH:-0}; SPEC_ADAPT=${SPEC_ADAPT:-$DFLASH}; PF_OWN=${PF_OWN:-3}; REUSE=${REUSE:-1} ;;
    3) SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-$(pool_cap 1941)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       [ "${DFLASH:-0}" = 0 ] || { echo "engine_defaults: DFLASH=1 at TP=3: the drafter's 8 KV heads do not split three ways" >&2; return 2; }
       DFLASH=0; SPEC_ADAPT=${SPEC_ADAPT:-0}; PF_OWN=${PF_OWN:-1}; REUSE=${REUSE:-1} ;;
    5) [ "${KV_FORMAT:-fp8}" = fp8 ] || [ -n "${CAP:-}" ] || { echo "engine_defaults: KV_FORMAT=$KV_FORMAT at TP=5: no TP5 int4 page count; set CAP" >&2; return 2; }
       SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-$(pool_cap 1981)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       # 1,981 pages a rank = TP4's default (not boot-proven at TP5; a rank holds about 1/5 of the experts, so more fit)
       [ "${DFLASH:-0}" = 0 ] || { echo "engine_defaults: DFLASH=1 at TP=5: the drafter's 8 KV heads do not split five ways" >&2; return 2; }
       # PF_OWN=3: cold 8K prefill 1,554 tok/s against 935 with 1, at the same CAP; decode within 1 %
       DFLASH=0; SPEC_ADAPT=${SPEC_ADAPT:-0}; PF_OWN=${PF_OWN:-3}; REUSE=${REUSE:-1} ;;
    *) echo "engine_defaults: TP=$TP; want 4, 6, 3 or 5" >&2; return 2 ;;
  esac
}

# The engine environment for every rank (docs/SETTINGS.md has each variable). Prefill: 64-row attention tiles, dense
# attention up to 8192 tokens of context and the indexer's sparse attention past it, reduce-scatter in place of
# all-reduce with o_proj writing into the send slot, two lanes on two streams, attention and cuBLAS in 1024-row chunks,
# the residual hash on the GPU, and GR8_PF_OWN as the attention ownership. Decode: two step
# graphs split at layer 3, one scratch shared with prefill. KV: the first SHARE_LIMIT tokens replicated on every rank
# and later pages spread across ranks (CP), every staged page checked against a HASH2 hash.
# TP=4 defaults GR8_PF_OWN to 4. OWN=3 needs the 13 GB full-head QO pack on every rank, and that pack does not fit next to
# a 90 GB TP=4 rank pack. OWN=4 runs the same attention over two head pairs, with the pair weights rolled in per layer
# (about 1.3-1.4 GB of device memory plus 84 MB pinned per rank). About 1.4 KB per position of it scales with QP_CTX.
# GLM-5.3-Flash: the prefill speed stack at top-k 8 (docs/FLASH.md). GR8_PF_RS, GR8_PF_OWN and CP stay off.
flash_env() {
  ENGINE_BIN=g6m4_flash
  ENGINE_ENV="GR8_CP=0 GR8_PF_RS=0 GR8_PF_OWN=0 GR8_PF_POOL_OWNER=0 GR8_FPF_P1=0 GR8_FPF_DSA_P1=0 \
GR8_FPF_ATTN_P1=0 GR8_PF_SPARSE_MLA=1 GR8_PF_SPARSE_MLA_512=1 \
GR8_PF_SPARSE_MLA_V2=1 GR8_PF_SPARSE_MLA_H16C=1 GR8_PF_DTR8=1 GR8_PF_SHFP8=1 GR8_PF_FC2V2=1 GR8_PF_KDA_WS=1 \
GR8_PF_DSA_FULL_GV=1 GR8_PF_ROUTER_SPLIT=1 GR8_PF_DSA_F16GEMM=1 GR8_PF_PREWARM=1 GR8_PF_MHC_STAGE=1 GR8_PF_MHC_FUSE=2 \
GR8_PF_STAGE_ASYNC=1"
  # the peer-partial staging (GR8_PF_STAGE_ASYNC) runs at any rank count; the G9 KDA scan is a precision arm, not the
  # same bytes as the scan it replaces, so it stays where it was chosen: TP2 (it would change TP4's output)
  [ "$TP" = 4 ] || ENGINE_ENV="$ENGINE_ENV GR8_GLM53F_KDA_SCAN=1 GR8_GLM53F_KDA_VS=2"
  # FLASH_CAP_POLICY=borrow-idle: the KV pool in chunks, lent once idle (the supervisor adds GR8_GLM53F_KV_LEDGER, the box's)
  if [ "${FLASH_CAP_POLICY:-protected}" = borrow-idle ]; then
    ENGINE_ENV="$ENGINE_ENV GR8_GLM53F_KV_CHUNKS=1 GR8_GLM53F_KV_LEND_MS=$((${FLASH_KV_IDLE_S:-60} * 1000))"
  fi
  # FLASH_CACHE: session checkpoints at prefill quantum boundaries (docs/FLASH.md), off unless set
  if [ -n "${FLASH_CACHE:-}" ] && [ "$FLASH_CACHE" != off ]; then
    ENGINE_ENV="$ENGINE_ENV GR8_GLM53F_CACHE=$FLASH_CACHE"
    [ -n "${FLASH_STATE_BUDGET_MIB:-}" ] && ENGINE_ENV="$ENGINE_ENV GR8_GLM53F_STATE_BUDGET_MIB=$FLASH_STATE_BUDGET_MIB"
  fi
  # its KV_STORE on a group lease: when it gives segments to other groups (KV_SHARE_WHEN; docs/SETTINGS.md)
  ENGINE_ENV="$ENGINE_ENV${KV_SHARE_WHEN:+ GR8_KV_SHARE_WHEN=$KV_SHARE_WHEN}"
  # TP=3 ring: leave GR8_PF_STRIPE to the engine default (on for a three-rank ring: large prefill writes also use the
  # devices on the other PCIe root); other TPs keep it off as before
  [ "$TP" = 3 ] || ENGINE_ENV="$ENGINE_ENV GR8_PF_STRIPE=0"
  ENGINE_ENV="$ENGINE_ENV ${EXTRA_ENV:-}"
}

engine_env() {
  [ "${GR8_MODEL:-glm53}" = glm53f ] && { flash_env; return 0; }
  [ "${GR8_MODEL:-glm53}" = pseudo ] && { ENGINE_BIN=g6pseudo; ENGINE_ENV=""; return 0; }
  ENGINE_BIN=g6m4; [ "${KV_FORMAT:-fp8}" = int4 ] && ENGINE_BIN=g6m4-int4
  ENGINE_ENV="GR8_PF_TILE=64 GR8_PF_DENSE=8192 GR8_PF_RS=1 GR8_PF_STREAMS=2 GR8_PF_ATTN_ROWS=${PF_ATTN_ROWS:-1024} GR8_PF_OWN=$PF_OWN \
GR8_PF_BLAS_ROWS=1024 GR8_PF_HASH_GPU=1 GR8_PF_RSDIRECT=1 GR8_DEC_SPLIT=3 GR8_DEC_SHARED_SCRATCH=1 \
GR8_CP=1 GR8_CP_R=$SHARE_LIMIT GR8_CP_S3=1 GR8_CP_PF_VERIFY=1 GR8_CP_PF_HASH2=1 ${EXTRA_ENV:-}"
  if [ "$TP" = 4 ] && [ "$PF_OWN" = 4 ]; then
    ENGINE_ENV="$ENGINE_ENV GR8_PF_QP_ROLL=${QP_ROLL:-2} GR8_PF_QP_CTX=${QP_CTX:-$CAP} GR8_PF_QP_EARLY=${QP_EARLY:-1}"
  fi
  # DFlash2 after a prefix-reuse hit (DFL_REUSE, DFL_REUSE_MIN; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DFL_REUSE:+ GR8_DFL_REUSE=$DFL_REUSE}"
  # a KV_STORE group= lease: when it gives segments to other groups (KV_SHARE_WHEN; docs/SETTINGS.md)
  ENGINE_ENV="$ENGINE_ENV${KV_SHARE_WHEN:+ GR8_KV_SHARE_WHEN=$KV_SHARE_WHEN}"
  ENGINE_ENV="$ENGINE_ENV${DFL_REUSE_MIN:+ GR8_DFL_REUSE_MIN=$DFL_REUSE_MIN}"
  # the MTP confidence stop for solo MTP rounds (MTP_KSTOP, MTP_KSTOP_KMAX, MTP_KSTOP_TAU; docs/SETTINGS.md): off
  # unless set. GR8_MTP_KSTOP_SEED3 and the round log GR8_MTP_LPLOG stay engine environment (EXTRA_ENV)
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOP:+ GR8_MTP_KSTOP=$MTP_KSTOP}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOP_KMAX:+ GR8_MTP_KSTOP_KMAX=$MTP_KSTOP_KMAX}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOP_TAU:+ GR8_MTP_KSTOP_TAU=$MTP_KSTOP_TAU}"
  # the DFlash2 drafter step under MTP's first form on its own stream (DRAFT_PAR; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DRAFT_PAR:+ GR8_DRAFT_PAR=$DRAFT_PAR}"
  # the replicated decode GEMVs split across the ranks (QA_SPLIT, IDXQ_SPLIT, MTP_SPLIT; docs/SETTINGS.md): unset = the
  # engine's default, which at TP > 1 is on (QA_SPLIT 2, IDXQ_SPLIT 1, MTP_SPLIT 1 with --mtp; g6m4.cu); 0 turns one off
  ENGINE_ENV="$ENGINE_ENV${QA_SPLIT:+ GR8_QA_SPLIT=$QA_SPLIT}"
  ENGINE_ENV="$ENGINE_ENV${IDXQ_SPLIT:+ GR8_IDXQ_SPLIT=$IDXQ_SPLIT}"
  ENGINE_ENV="$ENGINE_ENV${MTP_SPLIT:+ GR8_MTP_SPLIT=$MTP_SPLIT}"
  # draft weights on INT4 g32 copies (DRAFT_HEAD_I4, DFL_HEAD_I4, DFL_BODY_I4; docs/SETTINGS.md): off unless set; DRAFT_HEAD_I4 on at TP=4
  ENGINE_ENV="$ENGINE_ENV${DRAFT_HEAD_I4:+ GR8_DRAFT_HEAD_I4=$DRAFT_HEAD_I4}"
  ENGINE_ENV="$ENGINE_ENV${DFL_HEAD_I4:+ GR8_DFL_HEAD_I4=$DFL_HEAD_I4}"
  ENGINE_ENV="$ENGINE_ENV${DFL_BODY_I4:+ GR8_DFL_BODY_I4=$DFL_BODY_I4}"
  # the trellis shared expert on a forked stream during the router (ROUTE_OVL; docs/SETTINGS.md): off unless set; on at TP=4
  ENGINE_ENV="$ENGINE_ENV${ROUTE_OVL:+ GR8_ROUTE_OVL=$ROUTE_OVL}"
  # exact decode attention kernel variants (ATTN_SC2 .. ATTN_FRONT; docs/SETTINGS.md): off unless set; the measured set at TP=4
  ENGINE_ENV="$ENGINE_ENV${ATTN_SC2:+ GR8_ATTN_SC2=$ATTN_SC2}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_SC2_TR:+ GR8_ATTN_SC2_TR=$ATTN_SC2_TR}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_CTX2:+ GR8_ATTN_CTX2=$ATTN_CTX2}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_CTX3:+ GR8_ATTN_CTX3=$ATTN_CTX3}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_SEL2:+ GR8_ATTN_SEL2=$ATTN_SEL2}"
  ENGINE_ENV="$ENGINE_ENV${IDX_GRP:+ GR8_IDX_GRP=$IDX_GRP}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_QT:+ GR8_ATTN_QT=$ATTN_QT}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_FRONT:+ GR8_ATTN_FRONT=$ATTN_FRONT}"
  # the decode indexer's vector loads, the decode context buckets, the S2 MTP draft count from accept rates, S2 prefill
  # quanta and decode rounds while sessions decode, burst admissions, and prefill selx past PF_SELX_CTX (docs/SETTINGS.md):
  # off unless set; on at TP=4. DEC_BK 0 and PF_SELX_CTX 0 leave them to the engine (DEC_BK 2048, selx from 53248)
  ENGINE_ENV="$ENGINE_ENV${IDX_VEC:+ GR8_IDX_VEC=$IDX_VEC}"
  [ "${DEC_BK:-0}" = 0 ] || ENGINE_ENV="$ENGINE_ENV GR8_DEC_BK=$DEC_BK"
  ENGINE_ENV="$ENGINE_ENV${S2_ADAPT:+ GR8_S2_ADAPT=$S2_ADAPT}"
  ENGINE_ENV="$ENGINE_ENV${S2_DECODE_PF:+ GR8_S2_DECODE_PF=$S2_DECODE_PF}"
  ENGINE_ENV="$ENGINE_ENV${S2_ADMIT_HOLD:+ GR8_S2_ADMIT_HOLD=$S2_ADMIT_HOLD}"
  [ "${PF_SELX_CTX:-0}" = 0 ] || ENGINE_ENV="$ENGINE_ENV GR8_PF_SELX_CTX=$PF_SELX_CTX"
  # KV_ROWS (remote attention rows): STORE,size=N(M|G|T)[,local=P][,pool=S][,place=P][,fmt=F] -> the engine's GR8_RT_*
  # (docs/SETTINGS.md). size is each rank's lease in bytes; options.py has checked the form
  if [ -n "${KV_ROWS:-}" ] && [ "$KV_ROWS" != off ]; then
    local f n u parts IFS=','
    read -ra parts <<< "$KV_ROWS"
    unset IFS
    ENGINE_ENV="$ENGINE_ENV GR8_RT_STORE=${parts[0]}"
    for f in "${parts[@]:1}"; do
      case $f in
        size=*) n=${f#size=}; u=${n: -1}; n=${n%?}
                ENGINE_ENV="$ENGINE_ENV GR8_RT_SIZE=$(awk -v n="$n" -v u="$u" 'BEGIN { e = u == "M" ? 20 : u == "G" ? 30 : 40; printf "%.0f", n * 2 ^ e }')";;
        local=*) ENGINE_ENV="$ENGINE_ENV GR8_RT_LOCAL=${f#local=}";;
        pool=*) ENGINE_ENV="$ENGINE_ENV GR8_RT_POOL=${f#pool=}";;
        place=*) ENGINE_ENV="$ENGINE_ENV GR8_RT_PLACE=${f#place=}";;
        fmt=*) ENGINE_ENV="$ENGINE_ENV GR8_RT_FMT=${f#fmt=}";;
      esac
    done
  fi
  # the drafter choice's frozen cost table (DSEL_MODEL, DSEL_TABLE; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DSEL_MODEL:+ GR8_S2_DSEL_MODEL=$DSEL_MODEL}"
  ENGINE_ENV="$ENGINE_ENV${DSEL_TABLE:+ GR8_S2_DSEL_TABLE=$DSEL_TABLE}"
  # the n-gram table as a draft source (NGRAM, NGRAM_KMAX, NGRAM_KMIN, NGRAM_MARGIN; docs/SETTINGS.md, #56): off unless
  # set. GR8_NGRAM_LOG stays engine environment (EXTRA_ENV)
  ENGINE_ENV="$ENGINE_ENV${NGRAM:+ GR8_NGRAM=$NGRAM}"
  ENGINE_ENV="$ENGINE_ENV${NGRAM_KMAX:+ GR8_NGRAM_KMAX=$NGRAM_KMAX}"
  ENGINE_ENV="$ENGINE_ENV${NGRAM_KMIN:+ GR8_NGRAM_KMIN=$NGRAM_KMIN}"
  ENGINE_ENV="$ENGINE_ENV${NGRAM_MARGIN:+ GR8_NGRAM_MARGIN=$NGRAM_MARGIN}"
  # the per-session MTP confidence stop in shared rounds (MTP_KSTOPB, MTP_KSTOPB_KMIN, MTP_KSTOPB_TAU, MTP_KSTOPB_TAU_S;
  # docs/SETTINGS.md): off unless set. MTP_KSTOPB_TAU_S="1:0.7,0.7/2:0.7,0.8" sets GR8_MTP_KSTOPB_TAU_S1 and _S2.
  # GR8_MTP_KSTOPB_EXPLORE stays engine environment (EXTRA_ENV)
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB:+ GR8_MTP_KSTOPB=$MTP_KSTOPB}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB_KMIN:+ GR8_MTP_KSTOPB_KMIN=$MTP_KSTOPB_KMIN}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB_TAU:+ GR8_MTP_KSTOPB_TAU=$MTP_KSTOPB_TAU}"
  if [ -n "${MTP_KSTOPB_TAU_S:-}" ]; then
    local e IFS=/
    for e in $MTP_KSTOPB_TAU_S; do
      case $e in
        [1-8]:?*) ENGINE_ENV="$ENGINE_ENV GR8_MTP_KSTOPB_TAU_S${e%%:*}=${e#*:}" ;;
        *) echo "engine_env: MTP_KSTOPB_TAU_S: want n:t1,t2,... entries (n 1..8) separated by /, not '$e'" >&2; return 2 ;;
      esac
    done
  fi
  # the paced L2 prefetch in the decode step's idle-DRAM windows (L2PF; docs/SETTINGS.md): off unless set. Its
  # tuning variables GR8_L2PF_T, _GBPS, _WAVE, _US, _MB and _OPMB stay engine environment (EXTRA_ENV)
  ENGINE_ENV="$ENGINE_ENV${L2PF:+ GR8_L2PF=$L2PF}"
  # the row cap for S2 steps past SHARE_LIMIT, which also sizes the CP fabric (CP_STEP_ROWS, CP_STEP_SEG;
  # docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${CP_STEP_ROWS:+ GR8_CP_STEP_ROWS=$CP_STEP_ROWS}"
  ENGINE_ENV="$ENGINE_ENV${CP_STEP_SEG:+ GR8_CP_STEP_SEG=$CP_STEP_SEG}"
  # CP decode's remote head groups as one attention call (CP_ONECHAIN; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${CP_ONECHAIN:+ GR8_CP_ONECHAIN=$CP_ONECHAIN}"
  # kv_b (UK / UV) as the release's block FP8 (KVB=fp8, #50; docs/SETTINGS.md): BF16 unless set. Needs V2_DENSE
  ENGINE_ENV="$ENGINE_ENV${KVB:+ GR8_KVB=$KVB}"
  # deterministic fp-order attention variants (ATTN_CTX3_SP, ATTN_POSIDX; E1, docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${ATTN_CTX3_SP:+ GR8_ATTN_CTX3_SP=$ATTN_CTX3_SP}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_POSIDX:+ GR8_ATTN_POSIDX=$ATTN_POSIDX}"
  # prefill o_proj FP32 epilogue, MTP draft-step speed-ups (PF_GEMMF32, MTP_LASTROW, DRAFT_HEAD2, MTP_L2PF,
  # MTP_KSTOPB_WIDTHS; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${PF_GEMMF32:+ GR8_PF_GEMMF32=$PF_GEMMF32}"
  ENGINE_ENV="$ENGINE_ENV${MTP_LASTROW:+ GR8_MTP_LASTROW=$MTP_LASTROW}"
  ENGINE_ENV="$ENGINE_ENV${DRAFT_HEAD2:+ GR8_DRAFT_HEAD2=$DRAFT_HEAD2}"
  ENGINE_ENV="$ENGINE_ENV${MTP_L2PF:+ GR8_MTP_L2PF=$MTP_L2PF}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB_WIDTHS:+ GR8_MTP_KSTOPB_WIDTHS=$MTP_KSTOPB_WIDTHS}"
  # the window-2 fc1 split on every MoE layer at M > 1 (TR3_W2ALL; docs/SETTINGS.md): off unless set; on at TP=4
  ENGINE_ENV="$ENGINE_ENV${TR3_W2ALL:+ GR8_TR3_W2ALL=$TR3_W2ALL}"
  # QO cache, prefill index-score rows, fc1 kernels, P11 attention, dense trellis (QO_CACHE, PF_ISGRT_RG, PF_F12,
  # ATTN_P11, DTR, PF_AOP; docs/SETTINGS.md): the engine default unless set
  ENGINE_ENV="$ENGINE_ENV${QO_CACHE:+ GR8_QO_CACHE=$QO_CACHE}"
  ENGINE_ENV="$ENGINE_ENV${PF_ISGRT_RG:+ GR8_PF_ISGRT_RG=$PF_ISGRT_RG}"
  ENGINE_ENV="$ENGINE_ENV${PF_F12:+ GR8_PF_F12=$PF_F12}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_P11:+ GR8_ATTN_P11=$ATTN_P11}"
  ENGINE_ENV="$ENGINE_ENV${DTR:+ GR8_DTR=$DTR}"
  ENGINE_ENV="$ENGINE_ENV${PF_AOP:+ GR8_PF_AOP=$PF_AOP}"
  return 0
}

engine_args() {
  local r=$1 rp=$2 ap=$3 home=$4 epoch=${5:-0} a src="" df="" qo=""
  [ -n "${V2:-}" ] && src="--v2 $V2 --v2-slices $V2_SLICES --v2-widths $V2_WIDTHS"
  # V2_DENSE: the image comes from the release alone. It holds PF_OWN=3's full-head weights too, so the QO pack is
  # left out.
  [ -n "${V2_DENSE:-}" ] && src="$src --v2-dense $V2_DENSE ${V2_DENSE_FORM:+--v2-dense-form $V2_DENSE_FORM}"
  [ -n "${LAYOUT:-}" ] && src="$src --layout $LAYOUT"   # the engine refuses it without --v2-dense
  [ -n "${GRAMMAR_DIR:-}" ] && src="$src --grammar-dir $GRAMMAR_DIR"
  local dd=${DFLASH_DIR:-$ap}   # the drafter packs: with the other packs, or where the supervisor converted them
  [ "${DFLASH:-0}" = 1 ] && df="--dflash-manifest $dd/dflash.r$r.manifest.json --dflash-pack $dd/dflash.r$r.pack ${DFLASH_SEL:+--dflash-sel $DFLASH_SEL} ${DFLASH_K:+--dflash-k $DFLASH_K}"
  [ "$PF_OWN" = 3 ] && [ -z "${V2_DENSE:-}" ] && qo="--qo-manifest $ap/qo.manifest.json --qo-pack $ap/qo.pack"
  a="--manifest $rp/rank$r.manifest.json --pack $rp/rank$r.pack --image --layers all"
  [ -n "${V2_DENSE:-}" ] && a="--image --layers all"
  if [ "${GR8_MODEL:-glm53}" = pseudo ]; then
    a="--model $V2_DENSE --cap $CAP --serve $home/s2 --s2-sessions $SESS --s2-epoch $epoch ${SET_ID:+--set-id $SET_ID}"
    (set -f; echo $a)
    return
  fi
  # a release only, SESS sessions (flash_defaults). --chunk 1: a prompt tail goes one row a step, and a session's rows
  # in a main step stay within its KDA state ring (MTP + 1)
  if [ "${GR8_MODEL:-glm53}" = glm53f ]; then
    a="--image --layers all $src --top-p-cands 256 --chunk 1 --temp 0.7 --cap $CAP --graph --prefill 4096 \
--pf-lanes 2 --mtp $MTP --serve $home/s2 --s2-port $S2PORT --s2-sessions $SESS --s2-epoch $epoch"
    [ "${NOHASH:-0}" = 1 ] && a="$a --no-hash"
    [ -n "${SET_ID:-}" ] && a="$a --set-id $SET_ID"
    # FLASH_CACHE keeps its checkpoints in the KV_STORE store (one at most), else in memory
    if [ -n "${FLASH_CACHE:-}" ] && [ "$FLASH_CACHE" != off ] && [ -n "${KV_STORE:-}" ] && [ "$KV_STORE" != off ]; then
      local t IFS=';'
      for t in $KV_STORE; do [ -n "$t" ] && a="$a --kv-store $t"; done
      unset IFS
    fi
    (set -f; echo $a)
    return
  fi
  a="$a $src --top-p-cands 256 --chunk 8 --temp 0.7 --cap $CAP --graph --wide --prefill 4096 --pf-lanes 2 --pfv2 \
--mtp $MTP --serve $home/s2 --s2-port $S2PORT --s2-sessions $SESS --s2-epoch $epoch"
  [ "${NOHASH:-0}" = 1 ] && a="$a --no-hash"
  [ "${REUSE:-0}" = 1 ] && a="$a --reuse-prefix"
  # KV_STORE: one --kv-store per store (stores separated by ;), and what they keep
  if [ -n "${KV_STORE:-}" ] && [ "$KV_STORE" != off ]; then
    local t IFS=';'
    for t in $KV_STORE; do [ -n "$t" ] && a="$a --kv-store $t"; done
    unset IFS
    [ -n "${KV_KEEP:-}" ] && a="$a --kv-keep $KV_KEEP"
    [ -n "${KV_SYSTEM_AFTER:-}" ] && a="$a --kv-system-after $KV_SYSTEM_AFTER"
    [ -n "${KV_WRITE_LIMIT:-}" ] && a="$a --kv-write-limit $KV_WRITE_LIMIT"
  fi
  [ "$SPEC_ADAPT" = 1 ] && a="$a --spec-adaptive ${SPEC_LEVELS:+--spec-levels $SPEC_LEVELS}"
  [ -n "${SPEC_ACCEPT:-}" ] && a="$a --spec-accept $SPEC_ACCEPT"
  a="$a $df $qo"
  [ -n "${STEERING_PROFILE:-}" ] && a="$a --steering-profile $STEERING_PROFILE \
--steering-model-revision $STEERING_MODEL_REVISION --steering-tokenizer-revision $STEERING_TOKENIZER_REVISION \
--steering-chat-template-sha256 $STEERING_CHAT_TEMPLATE_SHA256 --steering-chat-template $STEERING_CHAT_TEMPLATE \
--steering-weight-set ${STEERING_WEIGHT_SET:-$SET_ID} ${STEERING_FORCE:+--steering-force $STEERING_FORCE}"
  [ "$TP" = 6 ] && a="$a ${EXTRA_ARGS:-}"

  [ -n "${SET_ID:-}" ] && a="$a --set-id $SET_ID"
  [ "${KV_FORMAT:-fp8}" = int4 ] && a="$a --kv-format int4"   # bin/g6m4-int4 (the supervisor picks it), which needs it
  (set -f; echo $a)
}
