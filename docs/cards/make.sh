#!/usr/bin/env bash
# Redraw the README's recipe cards: docs/cards/<recipe>.png from the kindling.json of each recipe that recipes.json
# marks "card": true, with bench/gif.py card, and the switch card docs/cards/switching.png from docs/cards/switching.json with docs/cards/switch_card.py.
#   docs/cards/make.sh                         fetch each kindling.json from the recipe's main branch on GitHub
#   KINDLING_DIR=DIR docs/cards/make.sh        read DIR/<recipe>/kindling.json instead (local clones)
#   FONT_DIR=DIR docs/cards/make.sh            draw with DIR/JetBrainsMono-Regular.ttf and DIR/SpaceGrotesk-Bold.ttf
# Without FONT_DIR, gif.py uses JetBrains Mono and Space Grotesk if they are installed, else its fallback fonts.
set -euo pipefail
cd "$(dirname "$0")/../.."
RECIPES=$(python3 -c "import json; print(' '.join(r['repo'] for r in json.load(open('recipes.json'))['recipes'] if r['card']))")
fonts=()
[ -n "${FONT_DIR:-}" ] && fonts=(--font "$FONT_DIR/JetBrainsMono-Regular.ttf" --display-font "$FONT_DIR/SpaceGrotesk-Bold.ttf")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for repo in $RECIPES; do
  r=${repo#*/}
  if [ -n "${KINDLING_DIR:-}" ]; then
    src="$KINDLING_DIR/$r/kindling.json"
  else
    src="$tmp/$r.json"
    curl -fsSL "https://raw.githubusercontent.com/$repo/main/kindling.json" -o "$src"
  fi
  python3 bench/gif.py card "$src" --out "docs/cards/$r.png" "${fonts[@]}"
done
python3 docs/cards/switch_card.py docs/cards/switching.json --out docs/cards/switching.png "${fonts[@]}"
