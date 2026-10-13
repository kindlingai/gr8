# The engine command line. scripts/launch_s2.sh (every TP: launch_tp4.sh, launch_tp6.sh and launch_tp3.sh run it) and
# the per-box supervisor (tools/launch) source this file, so the ssh launchers and `./gr8 glm53 up` start the same engine. KNOBS.md lists
# the knobs. docs/SETTINGS.md explains every engine variable and flag.
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
    4:1920) echo 2390 ;;
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

engine_defaults() {
  case $TP in
    4) SHARE_LIMIT=${SHARE_LIMIT:-65536}; CAP=${CAP:-$(pool_cap 1920)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       DFLASH=${DFLASH:-1}; PACKS=${PACKS:-packs}; SPEC_ADAPT=${SPEC_ADAPT:-$DFLASH}; PF_OWN=${PF_OWN:-4}; REUSE=${REUSE:-1} ;;
    6) SHARE_LIMIT=${SHARE_LIMIT:-131072}
       if [ "${KV_FORMAT:-fp8}" = int4 ]; then   # 614,400 is 3,307 fp8 pages a rank at 131072 and 2,026 at 32768
         if [ "$SHARE_LIMIT" = 131072 ]; then CAP=${CAP:-$(pool_cap 3307)} || return 2; else CAP=${CAP:-$(pool_cap 2026)} || return 2; fi
       else [[ "$SHARE_LIMIT" = 32768 || "$SHARE_LIMIT" = 131072 ]] && CAP=${CAP:-614400} || CAP=${CAP:-$(share_cap 2026)} || return 2; fi; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       [ "${DFLASH:-0}" = 0 ] || { echo "engine_defaults: DFLASH=1 at TP=6: the drafter's 8 KV heads do not split six ways" >&2; return 2; }
       DFLASH=0; PACKS=${PACKS:-packs}; SPEC_ADAPT=${SPEC_ADAPT:-0}; PF_OWN=${PF_OWN:-3}; REUSE=${REUSE:-1} ;;
    3) SHARE_LIMIT=${SHARE_LIMIT:-65536}; CAP=${CAP:-$(pool_cap 1941)} || return 2; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       [ "${DFLASH:-0}" = 0 ] || { echo "engine_defaults: DFLASH=1 at TP=3: the drafter's 8 KV heads do not split three ways" >&2; return 2; }
       DFLASH=0; PACKS=${PACKS:-packs}; SPEC_ADAPT=${SPEC_ADAPT:-0}; PF_OWN=${PF_OWN:-1}; REUSE=${REUSE:-1} ;;
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
engine_env() {
  ENGINE_BIN=g6m4; [ "${KV_FORMAT:-fp8}" = int4 ] && ENGINE_BIN=g6m4-int4   # bin/ENGINE_BIN, for scripts/launch_s2.sh
  ENGINE_ENV="G6_PF_TILE=64 G6_PF_DENSE=8192 G6_PF_RS=1 G6_PF_STREAMS=2 G6_PF_ATTN_ROWS=1024 G6_PF_OWN=$PF_OWN \
G6_PF_BLAS_ROWS=1024 G6_PF_TIMES=1 G6_PF_HASH_GPU=1 G6_PF_RSDIRECT=1 G6_DEC_SPLIT=3 G6_DEC_SHARED_SCRATCH=1 \
G6_CP=1 G6_CP_R=$SHARE_LIMIT G6_CP_S3=1 G6_CP_PF_VERIFY=1 G6_CP_PF_HASH2=1 ${G6ENV_X:-}"
  if [ "$TP" = 4 ] && [ "$PF_OWN" = 4 ]; then
    ENGINE_ENV="$ENGINE_ENV G6_PF_QP_ROLL=${QP_ROLL:-2} G6_PF_QP_CTX=${QP_CTX:-$CAP} G6_PF_QP_EARLY=${QP_EARLY:-1}"
  fi
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
  [ "$TP" = 6 ] && a="$a ${G6ARGS_X:-}"
  [ -n "${SET_ID:-}" ] && a="$a --set-id $SET_ID"
  [ "${KV_FORMAT:-fp8}" = int4 ] && a="$a --kv-format int4"   # bin/g6m4-int4 (the supervisor picks it), which needs it
  (set -f; echo $a)
}
