#!/usr/bin/env python3
"""Draw a recording of streamed replies as an animated GIF, for a README.

  gif.py REC.json [--out OUT.gif] [--title CAPTION] [--speed X] [--fps N] [--grid CxR] [--cols C --rows R | --size WxH]
  gif.py card kindling.json [--config ID | --all] [--out card.png | card.gif] [--width 800] [--light]

REC.json is one of:
  a concurrency file from bench.py --record (DIR/concurrency-cN.json): one pane per stream, side by side, each
    started at its real offset; --batch picks the batch (default the first);
  an agentic file from bench.py --record (DIR/agentic-TASK.json): one pane, the task's requests one after another;
  a recording from tools/query/query.py --record: one pane.

Each pane has a title row (the stream number, its tok/s as it runs, then its final tok/s) above the tail of its
wrapped text: thinking dim, the answer bright, tool calls tagged. The header line holds the caption and the total
tok/s so far: every stream's tokens over the time since the first token of any stream, which at the end is the
bench's number for the batch. Token counts come from each stream's usage; between deltas a stream's count is
its share of characters so far times its final count. A recording without usage counts deltas, shown with "~".

gif.py card draws one config of a kindling.json (bench.py kindling writes one) as a model card in the same look: a
header, a bar for each decode workload, prefill size and stream count, and a footer with the limits and quality
lines. Each section scales its own bars, and a section the config lacks is left out.

Needs Pillow (pip install pillow). The bench itself does not."""
import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path

def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# kindlingai.dev's colors, from https://kindlingai.dev/assets/brand.css (ink, paper, ember, flame core, ai text) and
# /styles.css (surface, surface-2, line, muted, meter-track), dark and light
THEMES = {
    'dark': {k: _rgb(v) for k, v in {
        'bg': '#16130F', 'surface': '#1E1A15', 'surface2': '#29241D', 'line': '#342E26', 'text': '#F2EDE4',
        'muted': '#A39A8C', 'track': '#2E2820', 'accent': '#FF6B1A', 'accent2': '#FFC24A', 'value': '#FFC24A'}.items()},
    'light': {k: _rgb(v) for k, v in {
        'bg': '#F2EDE4', 'surface': '#FBF8F3', 'surface2': '#EAE3D7', 'line': '#D9D0C2', 'text': '#16130F',
        'muted': '#6B6358', 'track': '#E4DCCF', 'accent': '#E8590C', 'accent2': '#FFB020', 'value': '#C2410C'}.items()},
}
GROUNDS = ('bg', 'surface')   # the fills text sits on: the palette ramps from each to every other color


def text_styles(t):
    """The recording panes' styles in a theme's colors."""
    return {'think': t['muted'], 'answer': t['text'], 'tool': t['value']}


RAMP = 4          # palette steps from a fill to each color, for antialiased edges
PAD = 6
FONTS = ['JetBrainsMono-Regular.ttf', 'JetBrainsMono[wght].ttf',
         '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf', 'DejaVuSansMono.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
         '/usr/share/fonts/TTF/DejaVuSansMono.ttf', '/usr/share/fonts/dejavu/DejaVuSansMono.ttf',
         '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', '/System/Library/Fonts/Menlo.ttc',
         '/System/Library/Fonts/Monaco.ttf', '/Library/Fonts/Courier New.ttf', 'C:/Windows/Fonts/consola.ttf']
DISPLAY_FONTS = ['SpaceGrotesk-Bold.ttf', 'SpaceGrotesk[wght].ttf', 'SpaceGrotesk-Regular.ttf']   # card titles
DEFAULT_PANE = {1: (80, 24), 2: (60, 20), 4: (52, 14)}   # columns x rows by pane count, 40 x 12 for more panes


def pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        sys.exit('gif.py needs Pillow: pip install pillow')
    return Image, ImageDraw, ImageFont


# ---------------------------------------------------------------- recordings


class Stream:
    """One pane's content: text pieces (time from the start of the recording, style, text) and its token count."""

    def __init__(self, name):
        self.name, self.pieces, self.tokens, self.exact, self.end = name, [], 0, False, 0.0

    def finish(self, usage_tokens, deltas):
        self.exact = bool(usage_tokens)
        self.tokens = usage_tokens or deltas
        self.chars = sum(len(p[2]) for p in self.pieces if p[1] not in ('mark', 'sep'))
        self.first = next((p[0] for p in self.pieces if p[1] not in ('mark', 'sep')), None)
        return self


def events_to(st, events, t0):
    """Append one request's events to a stream; returns (completion tokens from usage, text deltas)."""
    usage, deltas, call = 0, 0, None
    for t, kind, v in events:
        t += t0
        st.end = max(st.end, t)
        if kind in ('think', 'answer') and v:
            st.pieces.append((t, kind, v))
            deltas += 1
        elif kind == 'tool':
            if v.get('name') or v.get('index') != call:
                call = v.get('index')
                st.pieces.append((t, 'mark', f"[tool] {v.get('name', '')} "))
            if v.get('arguments'):
                st.pieces.append((t, 'tool', v['arguments']))
            deltas += 1
        elif kind == 'usage':
            usage += v.get('completion_tokens') or 0
    return usage, deltas


def load(path, batch):
    """-> (streams, meta)"""
    d = json.loads(Path(path).read_text())
    meta = d.get('meta') or {}
    streams = []
    if 'batches' in d:
        b = d['batches'][batch]
        for s in b['streams']:
            st = Stream(f"#{s['i'] + 1}")
            u, n = events_to(st, s['events'], s['t0'])
            streams.append(st.finish(u or s.get('tokens'), n))
    elif 'requests' in d:
        st = Stream(d.get('task', 'agent'))
        u = n = 0
        for r in d['requests']:
            if r['i']:
                st.pieces.append((r['t0'], 'sep', f"-- request {r['i'] + 1} --"))
            du, dn = events_to(st, r['events'], r['t0'])
            u, n = u + du, n + dn
        streams.append(st.finish(u, n))
    elif 'events' in d:
        st = Stream('#1')
        u, n = events_to(st, d['events'], 0.0)
        streams.append(st.finish(u, n))
        meta = {k: d[k] for k in ('prompt', 'effort') if d.get(k)}
    else:
        sys.exit(f'{path}: not a recording (no batches, requests or events)')
    return streams, meta


# ---------------------------------------------------------------- text


def width(c):
    return 2 if unicodedata.east_asian_width(c) in 'WF' else 1


class Wrap:
    """The rows of a pane's text, wrapped as tools/query/query.py --window wraps them, built up a piece at a time.
    A change of style starts a new row, and a row breaks at its last space when a Latin word would cross the edge."""

    def __init__(self, cols):
        self.cols, self.rows, self.row, self.w, self.style = cols, [], [], 0, None

    def add(self, style, text):
        """style: think, answer or tool, or mark (a tool call's tag, followed by its arguments on the same row) or
        sep (a line between requests)"""
        if style in ('mark', 'sep') or (self.style is not None and style != self.style and self.style != 'mark'):
            if self.row or self.rows:
                self.newline()
        self.style = style
        for c in text.replace('**', ''):
            self.put({'mark': 'tool', 'sep': 'think'}.get(style, style), c)

    def newline(self):
        self.rows.append(self.row)
        self.row, self.w = [], 0

    def put(self, style, c):
        if c == '\n':
            return self.newline()
        if c == '\t':
            c = ' '
        elif unicodedata.category(c)[0] == 'C':
            return
        cw = width(c)
        if self.w + cw > self.cols:
            row, cut = self.row, len(self.row)
            if c != ' ' and c.isascii():
                k = next((k for k in range(len(row) - 1, -1, -1) if row[k][1] == ' '), -1)
                if k > 0 and all(ch.isascii() for _, ch in row[k + 1:]):
                    cut = k + 1
            self.rows.append(row[:cut])
            self.row = [x for x in row[cut:] if self.rows[-1] or x[1] != ' ']
            self.w = sum(width(ch) for _, ch in self.row)
            if c == ' ' and not self.row:
                return
        self.row.append((style, c))
        self.w += cw

    def tail(self, n):
        return (self.rows + [self.row])[-n:] if n > 0 else []


# ---------------------------------------------------------------- drawing


class Painter:
    def __init__(self, font, cw):
        self.font, self.cw = font, cw
        self.mono = abs(font.getlength('i') - font.getlength('M')) < 0.01

    def line(self, draw, x, y, cells, colors):
        """cells: [(style, char)] at x, y on the character grid. ASCII runs of one style go in one call when the font
        is monospace; anything else goes a character at a time, at its cell."""
        col, i = 0, 0
        while i < len(cells):
            style, c = cells[i]
            if self.mono and c.isascii():
                j = i
                while j < len(cells) and cells[j][0] == style and cells[j][1].isascii():
                    j += 1
                run = ''.join(ch for _, ch in cells[i:j])
                draw.text((x + col * self.cw, y), run, font=self.font, fill=colors[style])
                col += j - i
                i = j
            else:
                draw.text((x + col * self.cw, y), c, font=self.font, fill=colors[style])
                col += width(c)
                i += 1

    def text(self, draw, x, y, s, color):
        self.line(draw, x, y, [('c', ch) for ch in s], {'c': color})


def palette(Image, theme):
    """A fixed palette: each fill in GROUNDS, then RAMP steps from it to every other color of the theme. Every frame
    maps onto it."""
    cols = []
    for g in GROUNDS:
        ground = theme[g]
        cols.append(ground)
        for c in dict.fromkeys(theme.values()):
            if c != ground:
                cols += [tuple(round(b + (v - b) * k / RAMP) for b, v in zip(ground, c)) for k in range(1, RAMP + 1)]
    cols = list(dict.fromkeys(cols))
    pal = Image.new('P', (1, 1))
    flat = [v for c in cols for v in c]
    pal.putpalette(flat + flat[:3] * (256 - len(cols)))
    return pal, len(cols)


def rate_text(st, t):
    """-> (tokens so far, the title row's right side) at recording time t"""
    if st.first is None or t < st.first:
        return 0, 'waiting' if st.first is not None else 'no tokens'
    shown = sum(len(p[2]) for p in st.pieces if p[0] <= t and p[1] not in ('mark', 'sep'))
    toks = st.tokens * shown / st.chars if st.chars else 0
    approx = '' if st.exact else '~'
    if t >= st.end:
        tps = (st.tokens - 1) / (st.end - st.first) if st.end > st.first and st.tokens > 1 else 0
        return st.tokens, f'done  {approx}{tps:.1f} tok/s  {approx}{st.tokens} tok'
    if t - st.first < 0.5:
        return toks, f'{approx}{toks:.0f} tok'
    return toks, f'{approx}{toks / (t - st.first):.1f} tok/s  {approx}{toks:.0f} tok'


def grid_for(n, given):
    if given:
        c, _, r = given.partition('x')
        return int(c), int(r)
    for c, r in ((1, 1), (2, 1), (2, 2), (4, 2), (4, 4)):
        if n <= c * r:
            return c, r
    c = math.ceil(math.sqrt(n))
    return c, math.ceil(n / c)


def load_font(ImageFont, path, size, names=FONTS, fallback=True):
    """path, else the first of names that loads (Pillow also looks in the system font folders), else Pillow's own
    font, or None without fallback. -> (font, its name)"""
    for p in ([path] if path else names):
        try:
            return ImageFont.truetype(p, size), p
        except OSError:
            if path:
                sys.exit(f'{path}: cannot load this font')
    if not fallback:
        return None, None
    try:
        return ImageFont.load_default(size), "Pillow's default"
    except TypeError:   # Pillow before 10.1: a fixed bitmap font
        return ImageFont.load_default(), "Pillow's default"


def cell(ImageFont, a):
    """-> (font, its name, cell width, line height in pixels)"""
    font, name = load_font(ImageFont, a.font, a.font_size)
    asc, desc = font.getmetrics() if hasattr(font, 'getmetrics') else (a.font_size, 2)
    return font, name, font.getlength('M'), asc + desc + 2


def save(Image, out, frames, durations, **kw):
    """Write the frames (an iterator of palette images) as a GIF, or the first one alone as any other format.
    -> frames stored, file size in bytes"""
    first = next(frames)
    if not out.lower().endswith('.gif'):
        first.save(out)
        return 1, Path(out).stat().st_size
    first.save(out, save_all=True, append_images=frames, duration=durations, disposal=1, optimize=True, **kw)
    with Image.open(out) as im:
        return im.n_frames, Path(out).stat().st_size


def render(a):
    Image, ImageDraw, ImageFont = pillow()
    theme = THEMES['light' if a.light else 'dark']
    colors, bg = text_styles(theme), theme['bg']
    streams, meta = load(a.recording, a.batch)
    gc, gr = grid_for(len(streams), a.grid)
    if gc * gr < len(streams):
        sys.exit(f'--grid {a.grid} holds {gc * gr} panes, the recording has {len(streams)} streams')
    font, font_name, cw, lh = cell(ImageFont, a)
    if a.size:
        w, _, h = a.size.partition('x')
        cols, rows = max(8, int((int(w) - 2 * PAD) // cw)), max(2, int((int(h) - 2 * PAD) // lh))
    else:
        cols, rows = DEFAULT_PANE.get(gc * gr, (40, 12))
        cols, rows = a.cols or cols, a.rows or rows
    pw, ph = math.ceil(cols * cw) + 2 * PAD, rows * lh + 2 * PAD
    hh = lh + 2 * PAD
    W, H = gc * pw + (gc + 1), hh + gr * ph + gr + 1
    paint = Painter(font, cw)
    pal, ncolors = palette(Image, theme)
    caption = a.title or ' · '.join(x for x in (meta.get('model'), meta.get('label'),
                                                 f'{len(streams)} stream' + ('s' if len(streams) > 1 else '')) if x)

    t_end = max(s.end for s in streams)
    if a.max_seconds:
        t_end = min(t_end, a.max_seconds)
    first_any = min((s.first for s in streams if s.first is not None), default=None)
    step = a.speed / a.fps
    nframes = max(1, math.ceil(t_end / step) + 1)

    canvas = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(canvas)
    for i in range(gc + 1):
        d.line([(i * (pw + 1), hh), (i * (pw + 1), H - 1)], fill=theme['line'])
    for j in range(gr + 1):
        d.line([(0, hh + j * (ph + 1)), (W - 1, hh + j * (ph + 1))], fill=theme['line'])
    wraps = [Wrap(cols) for _ in streams]
    fed = [0] * len(streams)
    shown_title = [None] * len(streams)

    def frame(t):
        total = 0.0
        for k, st in enumerate(streams):
            x0, y0 = (k % gc) * (pw + 1) + 1 + PAD, hh + (k // gc) * (ph + 1) + 1 + PAD
            toks, right = rate_text(st, t)
            total += toks
            title = (st.name, right)
            if title != shown_title[k]:
                shown_title[k] = title
                d.rectangle([x0 - PAD, y0 - PAD, x0 + pw - PAD - 1, y0 + lh - 1], fill=bg)
                paint.text(d, x0, y0, st.name, theme['accent'])
                right = right[:max(0, cols - len(st.name) - 1)]
                paint.text(d, x0 + (cols - sum(width(c) for c in right)) * cw, y0, right,
                           theme['value'] if right.startswith('done') else theme['muted'])
            n = fed[k]
            while n < len(st.pieces) and st.pieces[n][0] <= t:
                wraps[k].add(st.pieces[n][1], st.pieces[n][2])
                n += 1
            if n != fed[k]:
                fed[k] = n
                d.rectangle([x0 - PAD, y0 + lh, x0 + pw - PAD - 1, y0 + ph - PAD - 1], fill=bg)
                for r, cells in enumerate(wraps[k].tail(rows - 1)):
                    paint.line(d, x0, y0 + (r + 1) * lh, cells, colors)
        # the header: the caption on the left, the total rate and the clock on the right
        d.rectangle([0, 0, W - 1, hh - 1], fill=bg)
        paint.text(d, PAD, PAD, caption, theme['text'])
        tps = total / (t - first_any) if first_any is not None and t - first_any >= 0.5 else None
        clock = f'{t:5.1f} s' + (f' (x{a.speed:g})' if a.speed != 1 else '')
        right = (f'total {tps:6.1f} tok/s   ' if tps else '') + clock
        paint.text(d, W - PAD - sum(width(c) for c in right) * cw, PAD, right, theme['accent'])
        return canvas.quantize(palette=pal, dither=Image.Dither.NONE)

    frames = (frame(min(k * step, t_end)) for k in range(nframes))
    durations = [round(1000 / a.fps)] * (nframes - 1) + [max(round(1000 / a.fps), round(a.hold * 1000))]
    stored, size = save(Image, a.out, frames, durations, loop=0)
    secs = (nframes - 1) / a.fps + max(1 / a.fps, a.hold)
    print(f'{a.out}: {W}x{H}, {len(streams)} pane{"s" if len(streams) > 1 else ""} in {gc}x{gr}, {nframes} frames '
          f'drawn, {stored} stored (unchanged frames merged), {secs:.1f} s, {size / 1024:.0f} KiB, '
          f'{ncolors} colors, font {font_name}')


# ---------------------------------------------------------------- model card


def num(v):
    return f'{v:,}' if isinstance(v, int) else f'{v:,.1f}'


def context_label(n):
    return f'{n // 1024}K' if n % 1024 == 0 else f'{n:,}'


def card_rows(doc, c):
    """A config of a kindling.json as rows to draw, in theme color names. A row is ('title', text, right text),
    ('text', color, text), ('section', heading) or
    ('bar', label, value, section maximum, value text, section). Only fields present show. Optional: a concurrency
    entry's ttft_p50_s (median first token, seconds) and the config's decode_prompt_tokens and
    concurrency_prompt_tokens ([min, max] prompt length) add to the bar text and the section headings.
    Optional max_tokens_per_chat ([{"chats": N, "tokens": T, "source": "measured"|"computed"}]) and
    max_tokens_per_chat_note appear under Limits."""
    nodes = c.get('nodes')
    right = ' · '.join(x for x in (c.get('label'), nodes and f"{nodes} node{'s' if nodes != 1 else ''}") if x)
    rows = [('title', ' · '.join(str(doc[k]) for k in ('title', 'variant') if doc.get(k)), right)]
    for k, color in (('model', 'text'), ('hardware', 'muted')):
        if doc.get(k):
            rows.append(('text', color, str(doc[k])))
    if doc.get('stack'):
        rows.append(('text', 'muted', ' · '.join(doc['stack'])))

    def bars(heading, items):
        if items:
            rows.append(('section', heading))
            top = max(v for _, v, _ in items) or 1
            rows.extend(('bar', label, v, top, text, heading) for label, v, text in items)

    def prompts(key):
        r = c.get(key)
        return f', prompts {r[0]}-{r[1]} tokens' if isinstance(r, list) and len(r) == 2 else ''

    dec = c.get('decode') or {}
    bars('Decode · tok/s, one stream' + prompts('decode_prompt_tokens'),
         [(w, v, num(v)) for w, v in dec.items() if isinstance(v, (int, float))])
    bars('Prefill · tok/s, cold prompt', [(context_label(x['context']), x['tps'], num(x['tps']))
                                          for x in c.get('prefill') or [] if 'tps' in x])

    def per_stream(x):
        v = next((x[k] for k in ('per_stream_tps', 'per_stream') if k in x), None)
        each = f'  {num(v)} each' if v is not None else ''
        t = x.get('ttft_p50_s')
        return each + (f'  first token {t:.2f} s' if isinstance(t, (int, float)) else '')
    bars('Concurrency · total tok/s' + prompts('concurrency_prompt_tokens'),
         [(f"{x['streams']} stream{'s' if x['streams'] != 1 else ''}", x['aggregate_tps'],
           num(x['aggregate_tps']) + per_stream(x)) for x in c.get('concurrency') or [] if 'aggregate_tps' in x])
    limits = [f'{name} {num(c[k])}' for k, name in (('max_context', 'max context'), ('kv_pool_tokens', 'KV pool'),
                                                       ('max_concurrent', 'max concurrent')) if k in c]
    per_chat = [f"{x['chats']} {'chat' if x['chats'] == 1 else 'chats'} {num(x['tokens'])}"
                + (f" ({x['source']})" if x.get('source') else '')
                for x in c.get('max_tokens_per_chat') or [] if 'chats' in x and 'tokens' in x]
    if c.get('max_tokens_per_chat_note'):
        per_chat.append(str(c['max_tokens_per_chat_note']))
    quality = [f"{q['name']}: {q['value']}" for q in c.get('quality') or [] if 'name' in q and 'value' in q]
    if limits or per_chat or quality:
        rows.append(('section', 'Limits' if (limits or per_chat) else 'Quality'))
        if limits:
            rows.append(('text', 'muted', ' · '.join(limits)))
        if per_chat:
            rows.append(('text', 'text', 'Max tokens per active chat: ' + ' · '.join(per_chat)))
        rows.extend(('text', 'text', q) for q in quality)
    return rows


def fit(rows, cols):
    """Text rows longer than cols split at their ' · ' separators into several rows."""
    out = []
    for row in rows:
        if row[0] != 'text' or len(row[2]) <= cols:
            out.append(row)
            continue
        line = ''
        for part in row[2].split(' · '):
            if line and len(line) + 3 + len(part) > cols:
                out.append(('text', row[1], line))
                line = part
            else:
                line = f'{line} · {part}' if line else part
        out.append(('text', row[1], line))
    return out


def spaced(d, x, y, text, font, fill, gap):
    """text a character at a time, gap pixels apart: the site's letter-spaced capitals. -> the width drawn"""
    x0 = x
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + gap
    return x - x0 - gap


def card(a):
    Image, ImageDraw, ImageFont = pillow()
    theme = THEMES['light' if a.light else 'dark']
    doc = json.loads(Path(a.kindling).read_text())
    configs = doc.get('configs') or []
    if a.all:
        chosen = configs
    else:
        want = a.config or doc.get('default_config') or (configs[0]['id'] if configs else None)
        chosen = [c for c in configs if c.get('id') == want]
        if not chosen:
            sys.exit(f"{a.kindling}: no config {want!r}; configs: {', '.join(c.get('id', '?') for c in configs)}")
    if not chosen:
        sys.exit(f'{a.kindling}: no configs')
    font, font_name, cw, lh = cell(ImageFont, a)
    head, head_name = load_font(ImageFont, a.display_font, a.font_size + 2, DISPLAY_FONTS, fallback=False)
    head = head or font
    paint = Painter(font, cw)
    pal, ncolors = palette(Image, theme)
    across = max(1, min(a.across, len(chosen)))
    gap = 2 * PAD
    cardw = (a.width - gap * (across - 1)) // across
    cols = int((cardw - 2 * PAD) // cw) - 4          # a border cell and a space on each side
    if cols < 30:
        sys.exit(f'--width {a.width} leaves {cols} characters a card: too narrow')
    layouts = [fit(card_rows(doc, c), cols) for c in chosen]
    # with several cards a section scales to its largest value on any of them, so the cards compare
    top = {}
    for row in (r for rows in layouts for r in rows if r[0] == 'bar'):
        top[row[5]] = max(top.get(row[5], 0), row[2])
    layouts = [[r[:3] + (top[r[5]] or 1,) + r[4:] if r[0] == 'bar' else r for r in rows] for rows in layouts]
    heights = [(len(rows) + 2) * lh + 2 * PAD for rows in layouts]
    lines = [max(heights[i:i + across]) for i in range(0, len(chosen), across)]
    W, H = a.width, sum(lines) + gap * (len(lines) - 1)
    bar_rows = [r for rows in layouts for r in rows if r[0] == 'bar']
    lw = max([len(r[1]) for r in bar_rows] + [0]) + 2      # the label column, the same on every card
    vw = max([len(r[4]) for r in bar_rows] + [0]) + 2      # the value column
    canvas = Image.new('RGB', (W, H), theme['bg'])
    d = ImageDraw.Draw(canvas)
    surface = theme['surface']

    def draw(p):
        """Every card with its bars p (0 to 1) of the way to full."""
        for k, rows in enumerate(layouts):
            x, y = (k % across) * (cardw + gap), sum(lines[:k // across]) + gap * (k // across)
            box = [x + PAD, y + PAD + lh // 2, x + cardw - PAD - 1, y + heights[k] - PAD - lh // 2]
            d.rectangle(box, fill=surface, outline=theme['line'])
            x0 = x + PAD + 2 * cw
            for r, row in enumerate(rows):
                y0 = y + PAD + (r + 1) * lh
                if row[0] == 'title':
                    spaced(d, x0, y0 - (head.size - font.size) // 2, row[1].upper(), head, theme['text'],
                           round(0.08 * head.size))
                    paint.text(d, x0 + (cols - len(row[2])) * cw, y0, row[2], theme['value'])
                elif row[0] == 'text':
                    paint.text(d, x0, y0, row[2][:cols], theme[row[1]])
                elif row[0] == 'section':
                    mid = y0 + lh // 2
                    d.line([(box[0], mid), (box[2], mid)], fill=theme['line'])
                    label = f' {row[1]} '
                    d.rectangle([x0, y0, x0 + len(label) * cw, y0 + lh - 1], fill=surface)
                    paint.text(d, x0, y0, label, theme['muted'])
                else:
                    _, label, v, peak, text, _ = row
                    paint.text(d, x0, y0, label, theme['muted'])
                    bx0, bx1 = x0 + lw * cw, x0 + (cols - vw) * cw
                    b0, b1 = y0 + lh // 3, y0 + lh - lh // 3
                    d.rectangle([bx0, b0, bx1, b1], fill=theme['track'])
                    end = bx0 + (bx1 - bx0) * v / peak * p
                    if end > bx0:
                        d.rectangle([bx0, b0, end, b1], fill=theme['accent'])
                    paint.text(d, x0 + (cols - len(text)) * cw, y0, text, theme['value'])
        return canvas.quantize(palette=pal, dither=Image.Dither.NONE)

    gif = a.out.lower().endswith('.gif')
    n = max(2, round(a.fill * a.fps)) if gif else 1
    # ease out: quick at first, settling at full
    frames = (draw(1 - (1 - k / (n - 1)) ** 3 if n > 1 else 1.0) for k in range(n))
    durations = [round(1000 / a.fps)] * (n - 1) + [max(round(1000 / a.fps), round(a.hold * 1000))]
    stored, size = save(Image, a.out, frames, durations)   # no loop count: the bars fill once
    print(f"{a.out}: {W}x{H}, {len(chosen)} config{'s' if len(chosen) > 1 else ''} "
          f"({', '.join(c.get('id', '?') for c in chosen)}), {stored} frame{'s' if stored > 1 else ''}, "
          f"{size / 1024:.0f} KiB, font {font_name}, title font {head_name or font_name}")


def card_main(argv):
    ap = argparse.ArgumentParser(prog='gif.py card', description='Draw a config of a kindling.json as a model card: '
                                 'a PNG, or with --out NAME.gif the same card with its bars filling once.')
    ap.add_argument('kindling')
    ap.add_argument('--config', help='a config id (default: default_config, else the first)')
    ap.add_argument('--all', action='store_true', help='every config, stacked (see --across)')
    ap.add_argument('--across', type=int, default=1, help='with --all: cards side by side in each row')
    ap.add_argument('--out', default='card.png')
    ap.add_argument('--width', type=int, default=800, help='image width in pixels')
    ap.add_argument('--light', action='store_true', help='the light theme (default dark)')
    ap.add_argument('--font', help='a monospace TrueType file (default JetBrains Mono if installed, else as gif.py)')
    ap.add_argument('--display-font',
                    help="the title's TrueType file (default Space Grotesk if installed, else the --font one)")
    ap.add_argument('--font-size', type=int, default=14)
    ap.add_argument('--fps', type=float, default=25.0, help='GIF only')
    ap.add_argument('--fill', type=float, default=1.2, help='GIF only: seconds the bars take to fill')
    ap.add_argument('--hold', type=float, default=4.0, help='GIF only: seconds to show the full card')
    a = ap.parse_args(argv)
    if a.fps <= 0:
        ap.error('--fps must be above 0')
    card(a)


def main():
    if sys.argv[1:2] == ['card']:
        return card_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog='\n'.join(__doc__.split('\n')[2:]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording')
    ap.add_argument('--out', help='default: the recording\'s name with .gif')
    ap.add_argument('--title', default='', help='the header caption (default: model, label and stream count)')
    ap.add_argument('--batch', type=int, default=0, help='concurrency files: which batch to draw')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed-up: 2 plays a 60 s run in 30 s')
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--grid', metavar='CxR', help='panes across x down (default from the stream count)')
    ap.add_argument('--cols', type=int, help='characters across a pane')
    ap.add_argument('--rows', type=int, help='rows in a pane, the title row included')
    ap.add_argument('--size', metavar='WxH', help='pane size in pixels, instead of --cols and --rows')
    ap.add_argument('--font', help='a TrueType font file (default: the first monospace font found, else Pillow\'s)')
    ap.add_argument('--font-size', type=int, default=14)
    ap.add_argument('--max-seconds', type=float, help='stop after this many seconds of the recording')
    ap.add_argument('--hold', type=float, default=3.0, help='seconds to show the last frame')
    ap.add_argument('--light', action='store_true', help='the light theme (default dark)')
    a = ap.parse_args()
    if a.speed <= 0 or a.fps <= 0:
        ap.error('--speed and --fps must be above 0')
    a.out = a.out or str(Path(a.recording).with_suffix('.gif'))
    render(a)


if __name__ == '__main__':
    main()
