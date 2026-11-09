#!/usr/bin/env python3
"""Draw the README's switch card: the models as chips in a triangle, the seconds of each switch on its arrow.

    python3 docs/cards/switch_card.py docs/cards/switching.json --out docs/cards/switching.png [--font F] [--display-font F]

The JSON has title, subtitle, models (three: id, name, detail), switches (from, to, seconds: [runs]), command and
notes. Each arrow shows one number: the mean of its runs, in whole seconds. Needs Pillow. Fonts: JetBrains Mono and
Space Grotesk if installed, else Pillow's default font.
"""
import argparse
import json
import math
import sys
from pathlib import Path

W, H, S = 800, 600, 2                   # drawn at S times the size, then scaled down: smooth lines
BG_TOP, BG_BOTTOM, PANEL, LINE = '#2A160C', '#120A06', '#361E11', '#5A3218'   # ember: dark brown to near black
TEXT, MUTED, CHIP = '#FFF1E2', '#C49A7A', '#FFA23A'
ORANGE, AMBER = '#FF6B1A', '#FFC24A'    # Kindling orange and amber, as in bench/gif.py
BOW_OUT, BOW_IN = 0.16, 0.048           # curve depth as a share of the chip distance; inner arrows bow less
MONO = ['JetBrainsMono-Regular.ttf', 'JetBrainsMono[wght].ttf', 'DejaVuSansMono.ttf']
DISPLAY = ['SpaceGrotesk-Bold.ttf', 'SpaceGrotesk[wght].ttf', 'DejaVuSans-Bold.ttf']


def font(path, names, size):
    from PIL import ImageFont
    for p in ([path] if path else names):
        try:
            return ImageFont.truetype(p, size * S)
        except OSError:
            if path:
                sys.exit(f'{path}: cannot load this font')
    return ImageFont.load_default(size * S)


def seconds(runs):
    """The mean of the runs in whole seconds, halves up: [37.1, 39.2] -> '38 s'."""
    return f'{math.floor(sum(runs) / len(runs) + 0.5)} s'


ARROWS = '→↘↓↙←↖↑↗'                     # by angle, 0 = east, clockwise (screen y grows down)


def glyph(u):
    return ARROWS[round(math.degrees(math.atan2(u[1], u[0])) / 45) % 8]


def edge_point(c, half, d):
    """Where the ray from the chip centre c in direction d leaves the chip's box (half width, half height)."""
    n = math.hypot(*d)
    d = (d[0] / n, d[1] / n)
    tx = half[0] / abs(d[0]) if d[0] else math.inf
    ty = half[1] / abs(d[1]) if d[1] else math.inf
    t = min(tx, ty)
    return c[0] + d[0] * t, c[1] + d[1] * t


def bezier(p0, c, p1, t):
    return ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1])


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(rgb(a), rgb(b)))


def curved_arrow(dr, p0, ctrl, p1, width):
    """orange where the switch starts, amber at the head"""
    pts = [bezier(p0, ctrl, p1, i / 40) for i in range(41)]
    for i in range(40):
        c = mix(ORANGE, AMBER, i / 39)
        dr.line(pts[i:i + 2], fill=c, width=width)
        x, y = pts[i + 1]
        dr.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=c)
    color = AMBER
    b, a = pts[-1], pts[-3]
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    L, sp = 12 * S, 0.45
    dr.polygon([b, (b[0] - L * math.cos(ang - sp), b[1] - L * math.sin(ang - sp)),
                (b[0] - L * math.cos(ang + sp), b[1] - L * math.sin(ang + sp))], fill=color)


def text_box(dr, text, f):
    l, t, r, b = dr.textbbox((0, 0), text, font=f)
    return l, t, r - l, b - t


def label(dr, anchor, nrm, gap, text, f):
    """text on a small panel, its nearest edge `gap` from anchor along nrm, so it clears the line at any angle."""
    l, t, w, h = text_box(dr, text, f)
    pad = 6 * S
    reach = abs(nrm[0]) * (w / 2 + pad) + abs(nrm[1]) * (h / 2 + pad)
    x = anchor[0] + nrm[0] * (gap + reach) - w / 2
    y = anchor[1] + nrm[1] * (gap + reach) - h / 2
    dr.rounded_rectangle([x - pad, y - pad, x + w + pad, y + h + pad], radius=6 * S, fill=PANEL)
    dr.text((x - l, y - t), text, font=f, fill=AMBER)


def draw(doc, out, mono_path=None, display_path=None):
    from PIL import Image, ImageDraw   # here, not at the top: the tests of seconds() and glyph() run without Pillow
    models = {m['id']: m for m in doc['models']}
    if len(models) != 3:
        sys.exit('switch_card.py: the card draws exactly three models')
    title_f, chip_f = font(display_path, DISPLAY, 24), font(display_path, DISPLAY, 17)
    small_f, num_f, note_f = font(mono_path, MONO, 12), font(mono_path, MONO, 14), font(mono_path, MONO, 11)
    im = Image.new('RGB', (W * S, H * S))
    dr = ImageDraw.Draw(im)
    for y in range(H * S):
        dr.line([(0, y), (W * S, y)], fill=mix(BG_TOP, BG_BOTTOM, y / (H * S)))
    dr.rounded_rectangle([1 * S, 1 * S, (W - 2) * S, (H - 2) * S], radius=14 * S, outline=LINE, width=2 * S)
    dr.text((28 * S, 22 * S), doc['title'].upper(), font=title_f, fill=TEXT)
    dr.text((28 * S, 56 * S), doc['subtitle'], font=small_f, fill=MUTED)

    l, t, w, h = text_box(dr, doc['command'], num_f)    # the command, top right
    x, y = (W - 30) * S - w, 30 * S
    dr.rounded_rectangle([x - 14 * S, y - 8 * S, x + w + 14 * S, y + h + 8 * S], radius=8 * S,
                         fill=PANEL, outline=LINE, width=S)
    dr.text((x - l, y - t), doc['command'], font=num_f, fill=AMBER)

    ids = [m['id'] for m in doc['models']]
    centre = {ids[0]: (400 * S, 135 * S), ids[1]: (165 * S, 392 * S), ids[2]: (635 * S, 392 * S)}
    half = {}
    for i in ids:
        m = models[i]
        w = max(dr.textlength(m['name'], font=chip_f), dr.textlength(m['detail'], font=small_f))
        half[i] = (w / 2 + 18 * S, 30 * S)
    mid = (sum(c[0] for c in centre.values()) / 3, sum(c[1] for c in centre.values()) / 3)

    for sw in doc['switches']:
        a, b = centre[sw['from']], centre[sw['to']]
        d = (b[0] - a[0], b[1] - a[1])
        n = math.hypot(*d)
        u = (d[0] / n, d[1] / n)
        nrm = (-u[1], u[0])                               # left of the direction of travel
        m0 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        outward = (m0[0] - mid[0]) * nrm[0] + (m0[1] - mid[1]) * nrm[1] > 0
        k = (BOW_OUT if outward else BOW_IN) * n
        ctrl = (m0[0] + nrm[0] * k, m0[1] + nrm[1] * k)
        p0 = edge_point(a, (half[sw['from']][0] + 8 * S, half[sw['from']][1] + 8 * S), (ctrl[0] - a[0], ctrl[1] - a[1]))
        p1 = edge_point(b, (half[sw['to']][0] + 8 * S, half[sw['to']][1] + 8 * S), (ctrl[0] - b[0], ctrl[1] - b[1]))
        curved_arrow(dr, p0, ctrl, p1, 3 * S)
        label(dr, bezier(p0, ctrl, p1, 0.5), nrm, 10 * S, glyph(u) + ' ' + seconds(sw['seconds']), num_f)

    for i in ids:
        (cx, cy), (hw, hh) = centre[i], half[i]
        dr.rounded_rectangle([cx - hw, cy - hh, cx + hw, cy + hh], radius=6 * S, fill=PANEL, outline=CHIP,
                             width=2 * S)
        m = models[i]
        for text, f, color, dy in ((m['name'], chip_f, TEXT, -9 * S), (m['detail'], small_f, MUTED, 13 * S)):
            l, t, w, h = text_box(dr, text, f)
            dr.text((cx - w / 2 - l, cy + dy - h / 2 - t), text, font=f, fill=color)

    y = 504 * S
    for line in doc.get('notes') or []:
        dr.text((28 * S, y), line, font=note_f, fill=MUTED)
        y += 19 * S
    im.resize((W, H), Image.LANCZOS).save(out, optimize=True)
    print(f'{out}: {W}x{H}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('json')
    ap.add_argument('--out', default='switching.png')
    ap.add_argument('--font', help='a monospace TrueType file (default JetBrains Mono if installed)')
    ap.add_argument('--display-font', help='the title and chip TrueType file (default Space Grotesk if installed)')
    a = ap.parse_args(argv)
    draw(json.loads(Path(a.json).read_text()), a.out, a.font, a.display_font)


if __name__ == '__main__':
    main()
