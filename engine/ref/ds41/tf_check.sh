#!/usr/bin/env bash
# tf_check.sh ENGINE_DUMP_DIR PROMPT OUT WEIGHTS_DIR [EXPECT_LAYERS=$EXPECT_LAYERS]   (an empty PROMPT = prompts/p6.json, an empty OUT = /tmp/tf-oracle-$$)
# EXPECT_LAYERS (5th arg or env, e.g. 0,2,20,39) = the --layers the engine run was asked to dump.  When given, the dump must hold embed_T and both
# L<k>.resid_attn / L<k>.resid for exactly those layers (no more, no fewer), else exit 2 before any oracle work.  Without it the layer list is
# read from the files present (a dump that silently lost a layer then passes with fewer tensors): pass it.
# Always: every file must be non-empty and the same size as embed_T; and the final summary must read exactly '<1+2*nlayers> tensors compared, 0 failed',
# else exit 3 (the compare's own nonzero exit is passed through).
# Same-input check of a W4A16 engine run: the oracle (W4A16 MoE on the main chain, faithful attention) is run on the layers present in the
# engine dump (L<k>.resid.bin), each layer fed the ENGINE's own previous trunk, then every engine slot is gated with the "tf" tolerances.
# No GPU, ~2-5 min for 4 layers at M=6.  The engine dump needs embed_T.bin and L<k>.resid_attn.bin / L<k>.resid.bin (bf16 [1,S,4,5120]).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENG="$1"; PROMPT="${2:-$HERE/prompts/p6.json}"; OUT="${3:-/tmp/tf-oracle-$$}"
W="${4:?usage: tf_check.sh ENGINE_DUMP_DIR PROMPT OUT WEIGHTS_DIR [EXPECT_LAYERS]}"
EXPECT="${5:-${EXPECT_LAYERS:-}}"
die() { echo "tf_check FAIL: $*" >&2; exit 2; }
PRESENT=$(ls "$ENG" | sed -n 's/^L\([0-9]\+\)\.resid\(_attn\)\?\.bin$/\1/p' | sort -nu | paste -sd,)
if [ -n "$EXPECT" ]; then
  WANT=$(echo "$EXPECT" | tr ',' '\n' | sort -nu | paste -sd,)
  [ "$PRESENT" = "$WANT" ] || die "layers present in the dump ($PRESENT) differ from the expected layers ($WANT)"
  LAYERS="$WANT"
else
  LAYERS="$PRESENT"
fi
[ -n "$LAYERS" ] || die "no L<k>.resid / L<k>.resid_attn files in $ENG"
[ -s "$ENG/embed_T.bin" ] || die "$ENG/embed_T.bin missing or empty"
ESZ=$(stat -c %s "$ENG/embed_T.bin")
NAMES=embed_T
for k in ${LAYERS//,/ }; do
  for n in L$k.resid_attn L$k.resid; do
    [ -s "$ENG/$n.bin" ] || die "$ENG/$n.bin missing or empty"
    [ "$(stat -c %s "$ENG/$n.bin")" = "$ESZ" ] || die "$n.bin is $(stat -c %s "$ENG/$n.bin") bytes, embed_T.bin is $ESZ"
    NAMES="$NAMES,$n"
  done
done
NEXPECT=$(( 1 + 2 * $(echo "$LAYERS" | tr ',' '\n' | wc -l) ))
echo "layers=$LAYERS names=$NAMES"
cd /tmp
python3 "$HERE/ds41_oracle.py" --weights "$W" --prompt "$PROMPT" --layers "$LAYERS" --out "$OUT" --main-w4a16 --engine-trunk "$ENG" --threads "${THREADS:-8}" --no-final
CLOG="$(mktemp)"
set +e
python3 "$HERE/ds41_compare.py" "$ENG" "$OUT" --tol "$HERE/fixtures/manifest.json" --regime tf --names "$NAMES" 2>&1 | tee "$CLOG"
RC=${PIPESTATUS[0]}
set -e
LAST=$(grep -E '^[0-9]+ tensors compared, [0-9]+ failed$' "$CLOG" | tail -1)
rm -f "$CLOG"
[ "$RC" = 0 ] || exit "$RC"
[ "$LAST" = "$NEXPECT tensors compared, 0 failed" ] || { echo "tf_check FAIL: summary '${LAST:-<none>}' != '$NEXPECT tensors compared, 0 failed'" >&2; exit 3; }
echo "tf_check OK: $LAST"
