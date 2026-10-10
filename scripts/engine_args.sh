# The engine command line. scripts/launch_s2.sh (TP=4), launch_tp6.sh, launch_tp3.sh and the per-box supervisor
# (tools/launch) all source this file, so the ssh launchers and `./gr8 glm53 up` start the same engine. KNOBS.md lists
# the knobs. docs/SETTINGS.md explains every engine variable and flag.
#
#   engine_defaults          needs TP (4, 6 or 3); sets every knob that is unset to its default for this TP, and S2=1
#                            for the multi-session engine (TP=4), else S2=0 (TP=6 and TP=3 serve one request at a time)
#   engine_env               sets ENGINE_ENV, the engine environment for every rank (K=V words)
#   engine_args R RANK_PACKS AUX_PACKS HOME [EPOCH]
#                            prints the arguments that follow `--proxy-cpu N` for rank R. RANK_PACKS holds
#                            rank<R>.{manifest.json,pack}, AUX_PACKS the drafter and QO packs, HOME is G6_HOME
#                            (the serve/ and s2/ directories), EPOCH the S2 epoch. SET_ID, when set, adds --set-id.

engine_defaults() {
  case $TP in
    4) S2=1; CAP=${CAP:-393216}; READY_TMO=${READY_TMO:-1800}; MTP=${MTP:-2}; SESS=${SESS:-8}; S2PORT=${S2PORT:-7341}
       DFLASH=${DFLASH:-1}; PACKS=${PACKS:-packs}; SPEC_ADAPT=${SPEC_ADAPT:-$DFLASH}; PF_OWN=${PF_OWN:-4}; REUSE=${REUSE:-1} ;;
    6) S2=0; CAP=${CAP:-614400}; MTP=${MTP:-2}; READY_TMO=${READY_TMO:-900}; PF_OWN=3 ;;
    3) S2=0; CAP=${CAP:-307200}; MTP=${MTP:-2}; READY_TMO=${READY_TMO:-900}; PF_OWN=${PF_OWN:-1} ;;
    *) echo "engine_defaults: TP=$TP; want 4, 6 or 3" >&2; return 2 ;;
  esac
}

# The engine environment for every rank (docs/SETTINGS.md has each variable). Prefill: 64-row attention tiles, dense
# attention up to 8192 tokens of context and the indexer's sparse attention past it, reduce-scatter in place of
# all-reduce with o_proj writing into the send slot, two lanes on two streams, attention and cuBLAS in 1024-row chunks,
# timing lines on rank 0, the residual hash on the GPU, and G6_PF_OWN as the attention ownership. Decode: two step
# graphs split at layer 3, one scratch shared with prefill. KV: the first 32,768 tokens replicated on every rank and
# later pages spread across ranks (CP), every staged page checked against a HASH2 hash.
# TP=4 defaults G6_PF_OWN to 4. OWN=3 needs the 13 GB full-head QO pack on every rank, and that pack does not fit next to
# a 90 GB TP=4 rank pack. OWN=4 runs the same attention over two head pairs, with the pair weights rolled in per layer
# (about 1.3-1.4 GB of device memory plus 84 MB pinned per rank). About 1.4 KB per position of it scales with QP_CTX.
engine_env() {
  ENGINE_ENV="G6_PF_TILE=64 G6_PF_DENSE=8192 G6_PF_RS=1 G6_PF_STREAMS=2 G6_PF_ATTN_ROWS=1024 G6_PF_OWN=$PF_OWN \
G6_PF_BLAS_ROWS=1024 G6_PF_TIMES=1 G6_PF_HASH_GPU=1 G6_PF_RSDIRECT=1 G6_DEC_SPLIT=3 G6_DEC_SHARED_SCRATCH=1 \
G6_CP=1 G6_CP_R=32768 G6_CP_S3=1 G6_CP_PF_VERIFY=1 G6_CP_PF_HASH2=1 ${G6ENV_X:-}"
  if [ "$TP" = 4 ] && [ "$PF_OWN" = 4 ]; then
    ENGINE_ENV="$ENGINE_ENV G6_PF_QP_ROLL=${QP_ROLL:-2} G6_PF_QP_CTX=${QP_CTX:-$CAP} G6_PF_QP_EARLY=${QP_EARLY:-1}"
  fi
  return 0
}

engine_args() {
  local r=$1 rp=$2 ap=$3 home=$4 epoch=${5:-0} a src="" df="" qo=""
  [ -n "${V2:-}" ] && src="--v2 $V2 --v2-slices $V2_SLICES --v2-widths $V2_WIDTHS"
  local dd=${DFLASH_DIR:-$ap}   # the drafter packs: with the other packs, or where the supervisor converted them
  [ "${DFLASH:-0}" = 1 ] && df="--dflash-manifest $dd/dflash.r$r.manifest.json --dflash-pack $dd/dflash.r$r.pack ${DFLASH_SEL:+--dflash-sel $DFLASH_SEL} ${DFLASH_K:+--dflash-k $DFLASH_K}"
  [ "$PF_OWN" = 3 ] && qo="--qo-manifest $ap/qo.manifest.json --qo-pack $ap/qo.pack"
  a="--manifest $rp/rank$r.manifest.json --pack $rp/rank$r.pack --image --layers all"
  case $TP in
    4)
      a="$a $src --top-p-cands 256 --chunk 8 --temp 0.7 --cap $CAP --graph --wide --prefill 4096 --pf-lanes 2 --pfv2 \
--mtp $MTP --s2 $home/s2 --s2-port $S2PORT --s2-sessions $SESS --s2-epoch $epoch"
      [ "${NOHASH:-0}" = 1 ] && a="$a --no-hash"
      [ "${REUSE:-0}" = 1 ] && a="$a --reuse-prefix"
      [ "$SPEC_ADAPT" = 1 ] && a="$a --spec-adaptive ${SPEC_LEVELS:+--spec-levels $SPEC_LEVELS}"
      [ -n "${SPEC_ACCEPT:-}" ] && a="$a --spec-accept $SPEC_ACCEPT"
      a="$a $df" ;;
    *)
      a="$a $src --serve $home/serve --top-p-cands 256 --chunk 8 --temp 0.7 --cap $CAP --graph --wide --mtp $MTP --prefill 4096 \
--pf-lanes 2 --pfv2 $qo"
      [ "${NOHASH:-0}" = 1 ] && a="$a --no-hash"
      a="$a ${G6ARGS_X:-}" ;;
  esac
  [ -n "${SET_ID:-}" ] && a="$a --set-id $SET_ID"
  (set -f; echo $a)
}
