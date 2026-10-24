# The engine command line. The per-box supervisor (tools/launch) sources this file. KNOBS.md lists
# the knobs. docs/SETTINGS.md explains every engine variable and flag.
#
# GR8_MODEL picks the model: glm53 (unset) or glm53f, GLM-5.3-Flash (bin/g6m4_flash, TP 4 or 2).
#
#   engine_defaults          needs TP (4, 6 or 3); sets every knob that is unset to its default for this TP. An unset CAP
#                            follows SHARE_LIMIT, so the KV pool keeps the same physical pages per rank (share_cap below)
#   engine_env               sets ENGINE_ENV, the engine environment for every rank (K=V words), and ENGINE_BIN, the
#                            engine under bin/ for KV_FORMAT (g6m4, or g6m4-int4 for int4)
#   engine_args R RANK_PACKS AUX_PACKS HOME [EPOCH]
#                            prints the arguments that follow `--proxy-cpu N` for rank R. RANK_PACKS holds
#                            rank<R>.{manifest.json,pack} (unused with V2_DENSE: the release is the model), AUX_PACKS
#                            the drafter and QO packs, HOME is G6_HOME
#                            (the s2/ directory), EPOCH the S2 epoch. SET_ID, when set, adds --set-id.

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

# GLM-5.3-Flash under --serve takes one session and no CP, prefix reuse, DFlash or PF_OWN (engine/runtime/g6m4.cu
# refuses them). MTP=1: --serve needs a draft, and the KDA state does not roll back a rejected draft yet (docs/FLASH.md).
# CAP is a 262,144-token request plus a page for the guard rows (2 at MTP=1). Only the 11 DSA layers and the MTP layer
# keep KV, 1,664 B a token each, and without CP every rank holds the whole context: about 5.2 GB a rank at any TP
# (docs/FLASH.md, Context). With one session SHARE_LIMIT only sets the front end's default max_tokens.
flash_defaults() {
  case $TP in 4|2) ;; *) echo "engine_defaults: TP=$TP; GLM-5.3-Flash wants 4 or 2" >&2; return 2 ;; esac
  local k
  for k in DFLASH REUSE PF_OWN SPEC_ADAPT; do
    [ "${!k:-0}" = 0 ] || { echo "engine_defaults: $k=${!k}: GLM-5.3-Flash has no $k" >&2; return 2; }
  done
  [ "${SESS:-1}" = 1 ] || { echo "engine_defaults: SESS=$SESS: GLM-5.3-Flash serves one session" >&2; return 2; }
  SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-262208}; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-1}; SESS=1
  S2PORT=${S2PORT:-7341}; DFLASH=0; SPEC_ADAPT=0; PF_OWN=0; REUSE=0
}

engine_defaults() {
  [ "${GR8_MODEL:-glm53}" = glm53f ] && { flash_defaults; return; }
  case $TP in
    4) SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-$(pool_cap 2280)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       DFLASH=${DFLASH:-1}; SPEC_ADAPT=${SPEC_ADAPT:-$DFLASH}; PF_OWN=${PF_OWN:-4}; REUSE=${REUSE:-1} ;;
    6) SHARE_LIMIT=${SHARE_LIMIT:-131072}
       if [ "${KV_FORMAT:-fp8}" = int4 ]; then   # 614,400 is 3,307 fp8 pages a rank at 131072 and 2,026 at 32768
         if [ "$SHARE_LIMIT" = 131072 ]; then CAP=${CAP:-$(pool_cap 3307)} || return 2; else CAP=${CAP:-$(pool_cap 2026)} || return 2; fi
       else [[ "$SHARE_LIMIT" = 32768 || "$SHARE_LIMIT" = 131072 ]] && CAP=${CAP:-614400} || CAP=${CAP:-$(share_cap 2026)} || return 2; fi; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       [ "${DFLASH:-0}" = 0 ] || { echo "engine_defaults: DFLASH=1 at TP=6: the drafter's 8 KV heads do not split six ways" >&2; return 2; }
       DFLASH=0; SPEC_ADAPT=${SPEC_ADAPT:-0}; PF_OWN=${PF_OWN:-3}; REUSE=${REUSE:-1} ;;
    3) SHARE_LIMIT=${SHARE_LIMIT:-32768}; CAP=${CAP:-$(pool_cap 1941)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       [ "${DFLASH:-0}" = 0 ] || { echo "engine_defaults: DFLASH=1 at TP=3: the drafter's 8 KV heads do not split three ways" >&2; return 2; }
       DFLASH=0; SPEC_ADAPT=${SPEC_ADAPT:-0}; PF_OWN=${PF_OWN:-1}; REUSE=${REUSE:-1} ;;
    *) echo "engine_defaults: TP=$TP; want 4, 6 or 3" >&2; return 2 ;;
  esac
}

# The engine environment for every rank (docs/SETTINGS.md has each variable). Prefill: 64-row attention tiles, dense
# attention up to 8192 tokens of context and the indexer's sparse attention past it, reduce-scatter in place of
# all-reduce with o_proj writing into the send slot, two lanes on two streams, attention and cuBLAS in 1024-row chunks,
# timing lines on rank 0, the residual hash on the GPU, and G6_PF_OWN as the attention ownership. Decode: two step
# graphs split at layer 3, one scratch shared with prefill. KV: the first SHARE_LIMIT tokens replicated on every rank
# and later pages spread across ranks (CP), every staged page checked against a HASH2 hash.
# TP=4 defaults G6_PF_OWN to 4. OWN=3 needs the 13 GB full-head QO pack on every rank, and that pack does not fit next to
# a 90 GB TP=4 rank pack. OWN=4 runs the same attention over two head pairs, with the pair weights rolled in per layer
# (about 1.3-1.4 GB of device memory plus 84 MB pinned per rank). About 1.4 KB per position of it scales with QP_CTX.
# GLM-5.3-Flash: the prefill speed stack at top-k 8 (docs/FLASH.md). G6_PF_RS, G6_PF_OWN and CP stay off.
flash_env() {
  ENGINE_BIN=g6m4_flash
  ENGINE_ENV="G6_CP=0 G6_PF_RS=0 G6_PF_OWN=0 G6_PF_POOL_OWNER=0 G6_PF_STRIPE=0 G6_FPF_P1=0 G6_FPF_DSA_P1=0 \
G6_FPF_ATTN_P1=0 G6_PF_SPARSE_MLA_P1=0 G6_PF_SPARSE_MLA_HYBRID=0 G6_PF_SPARSE_MLA=1 G6_PF_SPARSE_MLA_512=1 \
G6_PF_SPARSE_MLA_V2=1 G6_PF_SPARSE_MLA_H16C=1 G6_PF_DTR8=1 G6_PF_SHFP8=1 G6_PF_FC2V2=1 G6_PF_KDA_WS=1 \
G6_PF_DSA_FULL_GV=1 G6_PF_ROUTER_SPLIT=1 G6_PF_DSA_F16GEMM=1 G6_PF_PREWARM=1 G6_PF_MHC_STAGE=1 G6_PF_MHC_FUSE=2 \
G6_PF_TIMES=1"
  if [ "$TP" = 4 ]; then ENGINE_ENV="$ENGINE_ENV G6_PF_STAGE_ASYNC=1"; else ENGINE_ENV="$ENGINE_ENV G9_KDA_SCAN=1 G9_KDA_VS=2"; fi
  ENGINE_ENV="$ENGINE_ENV ${G6ENV_X:-}"
}

engine_env() {
  [ "${GR8_MODEL:-glm53}" = glm53f ] && { flash_env; return 0; }
  ENGINE_BIN=g6m4; [ "${KV_FORMAT:-fp8}" = int4 ] && ENGINE_BIN=g6m4-int4
  ENGINE_ENV="G6_PF_TILE=64 G6_PF_DENSE=8192 G6_PF_RS=1 G6_PF_STREAMS=2 G6_PF_ATTN_ROWS=1024 G6_PF_OWN=$PF_OWN \
G6_PF_BLAS_ROWS=1024 G6_PF_TIMES=1 G6_PF_HASH_GPU=1 G6_PF_RSDIRECT=1 G6_DEC_SPLIT=3 G6_DEC_SHARED_SCRATCH=1 \
G6_CP=1 G6_CP_R=$SHARE_LIMIT G6_CP_S3=1 G6_CP_PF_VERIFY=1 G6_CP_PF_HASH2=1 ${G6ENV_X:-}"
  if [ "$TP" = 4 ] && [ "$PF_OWN" = 4 ]; then
    ENGINE_ENV="$ENGINE_ENV G6_PF_QP_ROLL=${QP_ROLL:-2} G6_PF_QP_CTX=${QP_CTX:-$CAP} G6_PF_QP_EARLY=${QP_EARLY:-1}"
  fi
  # DFlash2 after a prefix-reuse hit (DFL_REUSE, DFL_REUSE_MIN; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DFL_REUSE:+ G6_DFL_REUSE=$DFL_REUSE}"
  ENGINE_ENV="$ENGINE_ENV${DFL_REUSE_MIN:+ G6_DFL_REUSE_MIN=$DFL_REUSE_MIN}"
  # the MTP confidence stop for solo MTP rounds (MTP_KSTOP, MTP_KSTOP_KMAX, MTP_KSTOP_TAU; docs/SETTINGS.md): off
  # unless set. G6_MTP_KSTOP_SEED3 and the round log G6_MTP_LPLOG stay engine environment (G6ENV_X)
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOP:+ G6_MTP_KSTOP=$MTP_KSTOP}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOP_KMAX:+ G6_MTP_KSTOP_KMAX=$MTP_KSTOP_KMAX}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOP_TAU:+ G6_MTP_KSTOP_TAU=$MTP_KSTOP_TAU}"
  # the DFlash2 drafter step under MTP's first form on its own stream (DRAFT_PAR; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DRAFT_PAR:+ G6_DRAFT_PAR=$DRAFT_PAR}"
  # the replicated decode GEMVs split across the ranks (QA_SPLIT, IDXQ_SPLIT, MTP_SPLIT; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${QA_SPLIT:+ G6_QA_SPLIT=$QA_SPLIT}"
  ENGINE_ENV="$ENGINE_ENV${IDXQ_SPLIT:+ G6_IDXQ_SPLIT=$IDXQ_SPLIT}"
  ENGINE_ENV="$ENGINE_ENV${MTP_SPLIT:+ G6_MTP_SPLIT=$MTP_SPLIT}"
  # draft weights on INT4 g32 copies (DRAFT_HEAD_I4, DFL_HEAD_I4, DFL_BODY_I4; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DRAFT_HEAD_I4:+ G6_DRAFT_HEAD_I4=$DRAFT_HEAD_I4}"
  ENGINE_ENV="$ENGINE_ENV${DFL_HEAD_I4:+ G6_DFL_HEAD_I4=$DFL_HEAD_I4}"
  ENGINE_ENV="$ENGINE_ENV${DFL_BODY_I4:+ G6_DFL_BODY_I4=$DFL_BODY_I4}"
  # the trellis shared expert on a forked stream during the router (ROUTE_OVL; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${ROUTE_OVL:+ G6_ROUTE_OVL=$ROUTE_OVL}"
  # exact decode attention kernel variants (ATTN_SC2 .. ATTN_FRONT; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${ATTN_SC2:+ G6_ATTN_SC2=$ATTN_SC2}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_SC2_TR:+ G6_ATTN_SC2_TR=$ATTN_SC2_TR}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_CTX2:+ G6_ATTN_CTX2=$ATTN_CTX2}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_CTX3:+ G6_ATTN_CTX3=$ATTN_CTX3}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_SEL2:+ G6_ATTN_SEL2=$ATTN_SEL2}"
  ENGINE_ENV="$ENGINE_ENV${IDX_GRP:+ G6_IDX_GRP=$IDX_GRP}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_QT:+ G6_ATTN_QT=$ATTN_QT}"
  ENGINE_ENV="$ENGINE_ENV${ATTN_FRONT:+ G6_ATTN_FRONT=$ATTN_FRONT}"
  # KV_ROWS (remote attention rows): STORE,size=N(M|G|T)[,local=P][,pool=S][,place=P][,fmt=F] -> the engine's G6_RT_*
  # (docs/SETTINGS.md). size is each rank's lease in bytes; options.py has checked the form
  if [ -n "${KV_ROWS:-}" ] && [ "$KV_ROWS" != off ]; then
    local f n u parts IFS=','
    read -ra parts <<< "$KV_ROWS"
    unset IFS
    ENGINE_ENV="$ENGINE_ENV G6_RT_STORE=${parts[0]}"
    for f in "${parts[@]:1}"; do
      case $f in
        size=*) n=${f#size=}; u=${n: -1}; n=${n%?}
                ENGINE_ENV="$ENGINE_ENV G6_RT_SIZE=$(awk -v n="$n" -v u="$u" 'BEGIN { e = u == "M" ? 20 : u == "G" ? 30 : 40; printf "%.0f", n * 2 ^ e }')";;
        local=*) ENGINE_ENV="$ENGINE_ENV G6_RT_LOCAL=${f#local=}";;
        pool=*) ENGINE_ENV="$ENGINE_ENV G6_RT_POOL=${f#pool=}";;
        place=*) ENGINE_ENV="$ENGINE_ENV G6_RT_PLACE=${f#place=}";;
        fmt=*) ENGINE_ENV="$ENGINE_ENV G6_RT_FMT=${f#fmt=}";;
      esac
    done
  fi
  # the drafter choice's frozen cost table (DSEL_MODEL, DSEL_TABLE; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${DSEL_MODEL:+ G6_S2_DSEL_MODEL=$DSEL_MODEL}"
  ENGINE_ENV="$ENGINE_ENV${DSEL_TABLE:+ G6_S2_DSEL_TABLE=$DSEL_TABLE}"
  # the per-session MTP confidence stop in shared rounds (MTP_KSTOPB, MTP_KSTOPB_KMIN, MTP_KSTOPB_TAU, MTP_KSTOPB_TAU_S;
  # docs/SETTINGS.md): off unless set. MTP_KSTOPB_TAU_S="1:0.7,0.7/2:0.7,0.8" sets G6_MTP_KSTOPB_TAU_S1 and _S2.
  # G6_MTP_KSTOPB_WIDTHS and G6_MTP_KSTOPB_EXPLORE stay engine environment (G6ENV_X)
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB:+ G6_MTP_KSTOPB=$MTP_KSTOPB}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB_KMIN:+ G6_MTP_KSTOPB_KMIN=$MTP_KSTOPB_KMIN}"
  ENGINE_ENV="$ENGINE_ENV${MTP_KSTOPB_TAU:+ G6_MTP_KSTOPB_TAU=$MTP_KSTOPB_TAU}"
  if [ -n "${MTP_KSTOPB_TAU_S:-}" ]; then
    local e IFS=/
    for e in $MTP_KSTOPB_TAU_S; do
      case $e in
        [1-8]:?*) ENGINE_ENV="$ENGINE_ENV G6_MTP_KSTOPB_TAU_S${e%%:*}=${e#*:}" ;;
        *) echo "engine_env: MTP_KSTOPB_TAU_S: want n:t1,t2,... entries (n 1..8) separated by /, not '$e'" >&2; return 2 ;;
      esac
    done
  fi
  # the paced L2 prefetch in the decode step's idle-DRAM windows (L2PF; docs/SETTINGS.md): off unless set. Its
  # tuning variables G6_L2PF_T, _GBPS, _WAVE, _US, _MB and _OPMB stay engine environment (G6ENV_X)
  ENGINE_ENV="$ENGINE_ENV${L2PF:+ G6_L2PF=$L2PF}"
  # the row cap for S2 steps past SHARE_LIMIT, which also sizes the CP fabric (CP_STEP_ROWS, CP_STEP_SEG;
  # docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${CP_STEP_ROWS:+ G6_CP_STEP_ROWS=$CP_STEP_ROWS}"
  ENGINE_ENV="$ENGINE_ENV${CP_STEP_SEG:+ G6_CP_STEP_SEG=$CP_STEP_SEG}"
  # CP decode's remote head groups as one attention call (CP_ONECHAIN; docs/SETTINGS.md): off unless set
  ENGINE_ENV="$ENGINE_ENV${CP_ONECHAIN:+ G6_CP_ONECHAIN=$CP_ONECHAIN}"
  # kv_b (UK / UV) as the release's block FP8 (KVB=fp8, #50; docs/SETTINGS.md): BF16 unless set. Needs V2_DENSE
  ENGINE_ENV="$ENGINE_ENV${KVB:+ G6_KVB=$KVB}"
  return 0
}

engine_args() {
  local r=$1 rp=$2 ap=$3 home=$4 epoch=${5:-0} a src="" df="" qo=""
  [ -n "${V2:-}" ] && src="--v2 $V2 --v2-slices $V2_SLICES --v2-widths $V2_WIDTHS"
  # V2_DENSE: the image comes from the release alone. It holds PF_OWN=3's full-head weights too, so the QO pack is
  # left out.
  [ -n "${V2_DENSE:-}" ] && src="$src --v2-dense $V2_DENSE ${V2_DENSE_FORM:+--v2-dense-form $V2_DENSE_FORM}"
  local dd=${DFLASH_DIR:-$ap}   # the drafter packs: with the other packs, or where the supervisor converted them
  [ "${DFLASH:-0}" = 1 ] && df="--dflash-manifest $dd/dflash.r$r.manifest.json --dflash-pack $dd/dflash.r$r.pack ${DFLASH_SEL:+--dflash-sel $DFLASH_SEL} ${DFLASH_K:+--dflash-k $DFLASH_K}"
  [ "$PF_OWN" = 3 ] && [ -z "${V2_DENSE:-}" ] && qo="--qo-manifest $ap/qo.manifest.json --qo-pack $ap/qo.pack"
  a="--manifest $rp/rank$r.manifest.json --pack $rp/rank$r.pack --image --layers all"
  [ -n "${V2_DENSE:-}" ] && a="--image --layers all"
  # a release only, one session (flash_defaults). --chunk 1: a main step holds at most MTP + 1 rows, the KDA state ring
  if [ "${GR8_MODEL:-glm53}" = glm53f ]; then
    a="--image --layers all $src --top-p-cands 256 --chunk 1 --temp 0.7 --cap $CAP --graph --prefill 4096 \
--pf-lanes 2 --mtp $MTP --serve $home/s2 --s2-port $S2PORT --s2-sessions 1 --s2-epoch $epoch"
    [ "${NOHASH:-0}" = 1 ] && a="$a --no-hash"
    [ -n "${SET_ID:-}" ] && a="$a --set-id $SET_ID"
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
  [ "$TP" = 6 ] && a="$a ${G6ARGS_X:-}"

  [ -n "${SET_ID:-}" ] && a="$a --set-id $SET_ID"
  [ "${KV_FORMAT:-fp8}" = int4 ] && a="$a --kv-format int4"   # bin/g6m4-int4 (the supervisor picks it), which needs it
  (set -f; echo $a)
}
