#!/bin/bash
# fetch-tokenizer.sh DIR: GLM-5.3's tokenizer files from the exlr8 release, at a fixed revision and checked, into DIR
# (skipped when DIR already holds them). The pseudo engine's tests need them: GR8_PSEUDO_TOKENIZER=DIR.
set -euo pipefail
DIR=${1:?usage: fetch-tokenizer.sh DIR}
URL=https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/resolve/e0075a6cb5f24fdf9b94c2c0d5268b4b925d0d9a
SUMS="3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c  chat_template.jinja
ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55  generation_config.json
19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d  tokenizer.json
98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc  tokenizer_config.json"
mkdir -p "$DIR" && cd "$DIR"
echo "$SUMS" | shasum -a 256 -c --status 2> /dev/null && exit 0
for f in $(echo "$SUMS" | awk '{print $2}'); do curl -fsSL --retry 3 -o "$f" "$URL/$f"; done
echo "$SUMS" | shasum -a 256 -c
