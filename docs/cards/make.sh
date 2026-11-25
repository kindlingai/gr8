#!/usr/bin/env bash
# Redraw the README's recipe cards: docs/cards/<id>.png from the kindling.json of each recipe that recipes.json
# marks "card": true, with bench/gif.py card, and the switch card docs/cards/switching.png from docs/cards/switching.json with docs/cards/switch_card.py.
# The recipes live in kindlingai/recipes; its index.json lists each recipe id with its folder and its kindling.json.
#   docs/cards/make.sh                         fetch index.json once from kindlingai/recipes' main branch on GitHub
#   KINDLING_DIR=DIR docs/cards/make.sh        read DIR/<path>/kindling.json instead (DIR: a local clone of kindlingai/recipes;
#                                              DIR/index.json gives each id's path, else DIR/*/*/*/*/kindling.json are searched)
#   FONT_DIR=DIR docs/cards/make.sh            draw with DIR/JetBrainsMono-Regular.ttf and DIR/SpaceGrotesk-Bold.ttf
# Without FONT_DIR, gif.py uses JetBrains Mono and Space Grotesk if they are installed, else its fallback fonts.
set -euo pipefail
cd "$(dirname "$0")/../.."
fonts=()
[ -n "${FONT_DIR:-}" ] && fonts=(--font "$FONT_DIR/JetBrainsMono-Regular.ttf" --display-font "$FONT_DIR/SpaceGrotesk-Bold.ttf")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ -z "${KINDLING_DIR:-}" ]; then
  curl -fsSL https://raw.githubusercontent.com/kindlingai/recipes/main/index.json -o "$tmp/index.json"
fi
# Write $tmp/<id>.json for each card: true id in recipes.json and print the ids; stop if one has no kindling.json.
RECIPES=$(python3 - "$tmp" "${KINDLING_DIR:-}" <<'EOF'
import glob, json, os, sys
tmp, local = sys.argv[1], sys.argv[2]
ids = [r['id'] for r in json.load(open('recipes.json'))['recipes'] if r['card']]
cards = {}
if not local:
    cards = {e['id']: e['kindling'] for e in json.load(open(os.path.join(tmp, 'index.json')))['recipes']}
elif os.path.exists(os.path.join(local, 'index.json')):
    for e in json.load(open(os.path.join(local, 'index.json')))['recipes']:
        if e['id'] in ids:
            cards[e['id']] = json.load(open(os.path.join(local, e['path'], 'kindling.json')))
else:
    for f in glob.glob(os.path.join(local, '*', '*', '*', '*', 'kindling.json')):
        k = json.load(open(f))
        cards[k['id']] = k
missing = [i for i in ids if i not in cards]
if missing:
    sys.exit('no kindling.json for: ' + ' '.join(missing))
for i in ids:
    with open(os.path.join(tmp, i + '.json'), 'w') as f:
        json.dump(cards[i], f)
print(' '.join(ids))
EOF
)
for id in $RECIPES; do
  python3 bench/gif.py card "$tmp/$id.json" --out "docs/cards/$id.png" ${fonts[@]+"${fonts[@]}"}
done
python3 docs/cards/switch_card.py docs/cards/switching.json --out docs/cards/switching.png ${fonts[@]+"${fonts[@]}"}
