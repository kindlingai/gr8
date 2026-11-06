#!/usr/bin/env bash
# Redraw the README's recipe cards: docs/cards/<recipe>.png from each recipe's kindling.json, with bench/gif.py card.
#   docs/cards/make.sh                         fetch each kindling.json from the recipe's main branch on GitHub
#   KINDLING_DIR=DIR docs/cards/make.sh        read DIR/<recipe>/kindling.json instead (local clones)
#   FONT_DIR=DIR docs/cards/make.sh            draw with DIR/JetBrainsMono-Regular.ttf and DIR/SpaceGrotesk-Bold.ttf
# Without FONT_DIR, gif.py uses JetBrains Mono and Space Grotesk if they are installed, else its fallback fonts.
# The GLM-5.3-Flash 2x recipe has no kindling.json yet, so it has no card.
set -euo pipefail
cd "$(dirname "$0")/../.."
RECIPES="glm-5.3-flash-exlr8-1x-spark deepseek-v4.1-flash-exlr8-4x-spark"
fonts=()
[ -n "${FONT_DIR:-}" ] && fonts=(--font "$FONT_DIR/JetBrainsMono-Regular.ttf" --display-font "$FONT_DIR/SpaceGrotesk-Bold.ttf")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for r in $RECIPES; do
  if [ -n "${KINDLING_DIR:-}" ]; then
    src="$KINDLING_DIR/$r/kindling.json"
  else
    src="$tmp/$r.json"
    curl -fsSL "https://raw.githubusercontent.com/kindlingai/$r/main/kindling.json" -o "$src"
  fi
  python3 bench/gif.py card "$src" --out "docs/cards/$r.png" "${fonts[@]}"
done
