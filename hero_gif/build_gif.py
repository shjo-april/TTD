# Copyright (C) 2026 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

"""
Hero GIF (CPU only): compose figures/hero.gif from panels of the paper's own figures.

Four scenes: the single tag bias (paper Fig. 1), tag selection by pixel-tag scoring (Fig. 4(a)), text-tag
self-distillation (Fig. 2) and the results (Fig. 6, Fig. 7, Tables 2 to 4). Every picture is a panel of a paper
figure stored in hero_gif/panels and every number is written in hero_gif/scenes.py, so no model is needed.

    python hero_gif/build_gif.py                  # writes figures/hero.gif
    python hero_gif/build_gif.py --preview DIR    # writes every frame to DIR as a PNG instead

The committed GIF was rendered on Windows with Segoe UI. On other systems the script falls back to the
DejaVu Sans bundled with matplotlib: same layout, slightly different glyphs.
"""
import argparse
import json
import math
import os
import re
import runpy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'hero.gif'))
ap.add_argument('--preview', metavar='DIR', help='write the frames as PNGs to DIR and skip the GIF')
args = ap.parse_args()

W, H = 1280, 960
TEAL, TEAL_BG, MINT_LINE = (0, 102, 102), (232, 245, 244), (150, 205, 200)
INK, MUTED, LINE, WHITE, GREY = (16, 24, 31), (91, 105, 117), (220, 229, 233), (255, 255, 255), (178, 188, 196)
TAB_DIM = (191, 217, 217)                                          # inactive scene tab
BASE = (0, 110, 175)                                               # without TTD: CLIP variants, MaskCLIP, TCL
RED, GREEN = (222, 33, 33), (0, 153, 0)                            # the paper's colors: biased or irrelevant · debiased
TEAL_SOFT = (215, 238, 236)                                        # small text on a teal card
M, TOP, MSG_Y, NOTE_Y = 24, 214, 872, 938                          # side margin · top of the panels · message · note
V = runpy.run_path(os.path.join(HERE, 'scenes.py'))['SCENES']

WIN_FONTS = 'C:/Windows/Fonts'
FALLBACK = {'segoeui.ttf': 'DejaVuSans.ttf', 'seguisb.ttf': 'DejaVuSans-Bold.ttf', 'segoeuib.ttf': 'DejaVuSans-Bold.ttf'}


def F(name, size):
    path = os.path.join(WIN_FONTS, name)
    if not os.path.exists(path):
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', FALLBACK[name])
    return ImageFont.truetype(path, size)


F_TITLE, F_BADGE, F_SUBT, F_TAB = F('segoeuib.ttf', 40), F('seguisb.ttf', 16), F('segoeui.ttf', 17), F('segoeuib.ttf', 16)
F_HEAD, F_H, F_HS, F_HSUB = F('segoeuib.ttf', 22), F('segoeuib.ttf', 20), F('segoeui.ttf', 16), F('segoeui.ttf', 13)
F_HB, F_HBSUB, F_SB = F('segoeuib.ttf', 17), F('segoeuib.ttf', 13), F('seguisb.ttf', 15)
F_MSG, F_NOTE, F_TICK = F('segoeuib.ttf', 20), F('segoeui.ttf', 15), F('segoeui.ttf', 14)
F_BIG, F_FORM = F('segoeuib.ttf', 34), F('segoeui.ttf', 22)


def logo(name, height):
    im = Image.open(os.path.join(HERE, 'logos', name)).convert('RGBA')
    return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)


LOGOS_L = [logo(n, h) for n, h in V['logos']['left']]              # the affiliations in the order of the paper
LOGOS_R = [logo(n, h) for n, h in V['logos']['right']]


def load(rel, size):
    return Image.open(os.path.join(HERE, rel)).convert('RGB').resize(size, Image.LANCZOS)


def rrect(dr, box, r, fill=None, outline=None, width=1):
    dr.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def rich(dr, x, y, s, font, sub, fill, anchor='l'):
    """Text with subscripts written as _{...}; y is the baseline. Returns the width."""
    parts = re.split(r'_\{([^}]*)\}', s)                           # even items: normal text, odd items: subscripts
    widths = [dr.textlength(p, font=sub if i % 2 else font) for i, p in enumerate(parts)]
    x -= {'l': 0, 'm': sum(widths) / 2, 'r': sum(widths)}[anchor]
    for i, (p, w) in enumerate(zip(parts, widths)):
        dr.text((x, y + (round(sub.size * 0.3) if i % 2 else 0)), p, font=sub if i % 2 else font, fill=fill, anchor='ls')
        x += w
    return sum(widths)


def wrap(dr, text, font, width):
    lines, cur = [], ''
    for word in text.split():
        t = (cur + ' ' + word).strip()
        if cur and dr.textlength(t, font=font) > width:
            lines.append(cur); cur = word
        else:
            cur = t
    return lines + [cur]


def line(dr, y, text, font, fill):
    """One centered line; the build stops if it does not fit between the side margins."""
    assert dr.textlength(text, font=font) <= W - 2 * M, text
    dr.text((W / 2, y), text, font=font, fill=fill, anchor='mm')


def fits(dr, text, font, width):
    assert dr.textlength(text, font=font) <= width, text
    return text


def canvas(scene, head, note):
    """White frame with the header: logos, title and venue, subtitle, scene tabs, the scene headline and its note."""
    im = Image.new('RGB', (W, H), WHITE); dr = ImageDraw.Draw(im)
    x = 34
    for lg in LOGOS_L:
        im.paste(lg, (x, 38 - lg.height // 2), lg); x += lg.width + 24
    x = W - 34
    for lg in reversed(LOGOS_R):
        x -= lg.width; im.paste(lg, (x, 38 - lg.height // 2), lg); x -= 20
    tw = dr.textlength(V['title'], font=F_TITLE); bw = dr.textlength(V['badge'], font=F_BADGE) + 28
    x0 = (W - (tw + 14 + bw)) / 2
    dr.text((x0, 6), V['title'], font=F_TITLE, fill=TEAL)
    rrect(dr, (x0 + tw + 14, 22, x0 + tw + 14 + bw, 50), 14, fill=TEAL)
    dr.text((x0 + tw + 14 + bw / 2, 36), V['badge'], font=F_BADGE, fill=WHITE, anchor='mm')
    line(dr, 74, V['subtitle'], F_SUBT, MUTED)
    tabs = V['tabs']                                               # fixed position and size, only the current one is solid
    tw_ = (W - 2 * M - 16 * (len(tabs) - 1)) / len(tabs); x = M
    for i, t in enumerate(tabs):
        rrect(dr, (x, 96, x + tw_, 122), 8, fill=TEAL if i == scene else TAB_DIM)
        dr.text((x + tw_ / 2, 109), t, font=F_TAB, fill=WHITE if i == scene else TEAL, anchor='mm')
        x += tw_ + 16
    line(dr, 154, head, F_HEAD, INK)
    line(dr, NOTE_Y, note, F_NOTE, MUTED)
    return im, dr


def framed(im, dr, pic, x, y, col, width=3):
    im.paste(pic, (x, y))
    dr.rectangle((x - 2, y - 2, x + pic.width + 1, y + pic.height + 1), outline=col, width=width)


def placeholder(dr, x, y, w, h, label=''):
    rrect(dr, (x, y, x + w - 1, y + h - 1), 10, fill=TEAL_BG, outline=MINT_LINE)
    dr.text((x + w / 2, y + h / 2), label, font=F_HEAD, fill=MINT_LINE, anchor='mm')


def chip(dr, x, y, text, font, fg, bg, pad=10, h=28, anchor='l', outline=None):
    """A one-line label on a rounded box; (x, y) is its left, middle or right top corner. Returns the box."""
    w = dr.textlength(text, font=font) + 2 * pad
    x0 = x - {'l': 0, 'm': w / 2, 'r': w}[anchor]
    rrect(dr, (x0, y, x0 + w, y + h), 6, fill=bg, outline=outline)
    dr.text((x0 + w / 2, y + h / 2 - 1), text, font=font, fill=fg, anchor='mm')
    return x0, y, x0 + w, y + h


def strokes(im, items, ss=3):
    """Arrows and brackets, drawn on a larger layer and scaled down to smooth them.
    items: (points, color, width, arrow head at the last point or not)."""
    layer = Image.new('RGBA', (W * ss, H * ss), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    for pts, col, width, head in items:
        P = [(x * ss, y * ss) for x, y in pts]
        ld.line(P, fill=col + (255,), width=width * ss, joint='curve')
        if head:
            (x0, y0), (x1, y1) = P[-2], P[-1]
            a, s = math.atan2(y1 - y0, x1 - x0), (5 + 2.2 * width) * ss
            tip = (x1 + 0.6 * s * math.cos(a), y1 + 0.6 * s * math.sin(a))
            ld.polygon([tip] + [(x1 + s * math.cos(a + d), y1 + s * math.sin(a + d)) for d in (2.5, -2.5)], fill=col + (255,))
    im.paste(Image.alpha_composite(im.convert('RGBA'), layer.resize((W, H), Image.LANCZOS)).convert('RGB'))
    return ImageDraw.Draw(im)


# ---------------------------------------------------------------- scene 1: the single tag bias (paper Fig. 1)
B = V['bias']
PW, PH, GAP = 394, 290, 25
BIAS = [(load(p, (PW, PH)), t) for p, t in B['panels']]
BIAS_X = [M + k * (PW + GAP) for k in range(3)]
BIAS_COL = [MUTED, BASE, TEAL]
CDW, CDH, CHW, CHH = 606, 233, 590, 217                            # chart card · score chart (rendered from the vector figure)
CHARTS = [load(p, (CHW, CHH)) for p in B['charts']]
CHART_X, CHART_Y = [M, W - M - CDW], TOP + PH + 92
BIAS_MSG = [('CLIP aligns an image with its text. Which tags of the text does it see?', INK),
            ('Without TTD, the similarity map and the tag scores highlight a single tag: chameleon', BASE),
            ('With TTD, the other relevant tags (green, branch) are activated as well', TEAL)]


def bias_frame(s):
    im, dr = canvas(0, 'Single tag bias: CLIP-based models focus on one tag of the text', B['note'])
    for k, (pic, title) in enumerate(BIAS):
        x, col, on = BIAS_X[k], BIAS_COL[k], k <= s
        dr.text((x, TOP - 10), title, font=F_H, fill=(INK if k == 0 else col) if on else GREY, anchor='ls')
        if not on:
            placeholder(dr, x, TOP, PW, PH); continue
        framed(im, dr, pic, x, TOP, LINE if k == 0 else col)
        if k == 1:                                                 # the red box of the paper: the region of the single tag
            x0, y0, x1, y1 = (round(v * (PW if i % 2 == 0 else PH)) for i, v in enumerate(B['box']))
            dr.rectangle((x + x0, TOP + y0, x + x1, TOP + y1), outline=RED, width=3)
            chip(dr, x + x1 + 1, TOP + y1 - 2, 'Biased Prediction', F_SB, WHITE, RED, anchor='r')
        if k == 2:
            chip(dr, x + PW, TOP + PH - 30, 'Debiased Prediction', F_SB, WHITE, GREEN, anchor='r')
    y = TOP + PH + 14                                              # the text of the image, under the image
    rrect(dr, (BIAS_X[0], y, BIAS_X[0] + PW, y + 34), 8, fill=TEAL_BG, outline=MINT_LINE)
    dr.text((BIAS_X[0] + 14, y + 16), 'Text', font=F_SB, fill=TEAL, anchor='lm')
    dr.text((BIAS_X[0] + 58, y + 16), fits(dr, '“%s”' % B['text'], F_HS, PW - 72), font=F_HS, fill=INK, anchor='lm')
    for k, pic in enumerate(CHARTS):
        x, col, on = CHART_X[k], BIAS_COL[k + 1], k + 1 <= s
        tw = dr.textlength(BIAS[k + 1][1], font=F_HB)
        dr.text((x, CHART_Y - 12), BIAS[k + 1][1], font=F_HB, fill=col if on else GREY, anchor='ls')
        dr.text((x + tw + 8, CHART_Y - 12), '· similarity between the image and each tag', font=F_HS, fill=MUTED if on else GREY, anchor='ls')
        if on:
            rrect(dr, (x, CHART_Y, x + CDW - 1, CHART_Y + CDH - 1), 10, fill=WHITE, outline=LINE)
            im.paste(pic, (x + (CDW - CHW) // 2, CHART_Y + (CDH - CHH) // 2))
        else:
            placeholder(dr, x, CHART_Y, CDW, CDH)
    line(dr, MSG_Y, BIAS_MSG[s][0], F_MSG, BIAS_MSG[s][1])
    return im


# ---------------------------------------------------------------- scene 2: tag selection by pixel-tag scoring (paper Fig. 4(a))
S = V['selection']
SORTED, GAPS, NSEL = S['sorted'], S['gaps'], S['selected']
assert all(abs(SORTED[k][1] - SORTED[k + 1][1] - g) < 0.006 for k, g in enumerate(GAPS)), 'the gaps must follow from the scores'
assert GAPS.index(max(GAPS)) == NSEL - 1, 'the largest gap must be the boundary of the selected tags'
SI, HWD, HY = 210, 198, 486
SEL_IMG = load(S['image'], (SI, SI))
HEAT = [load('panels/pillow_heatmap_%d.jpg' % (k + 1), (HWD, HWD)) for k in range(6)]
HX = [M + round(k * (HWD + (W - 2 * M - 6 * HWD) / 5)) for k in range(6)]
TX0, TX1, FX0 = 250, 760, 800                                      # text and tags · the score card
SEL_MSG = ['Split the text into tags, then compare every tag with every pixel of the image',
           'Each tag gets a heatmap. Its score is the maximum similarity in that heatmap',
           'Sorted by score: how large is the gap between consecutive tags?',
           'The largest gap is the boundary: %s and %s are selected as pseudo-tags' % (', '.join(t for t, _ in SORTED[:NSEL - 1]), SORTED[NSEL - 1][0])]


def selection_frame(shown, gaps=False, final=False):
    im, dr = canvas(1, 'Tag selection by pixel-tag scoring: which tags of the text are in the image?', S['note'])
    dr.text((M, TOP - 10), 'Image', font=F_H, fill=INK, anchor='ls')
    framed(im, dr, SEL_IMG, M, TOP, LINE)
    dr.text((TX0, TOP - 10), 'Text and its tags', font=F_H, fill=INK, anchor='ls')
    rrect(dr, (TX0, TOP, TX1, TOP + 80), 10, fill=TEAL_BG, outline=MINT_LINE)
    dr.text(((TX0 + TX1) / 2, TOP + 39), fits(dr, '“%s”' % S['text'], F_H, TX1 - TX0 - 24), font=F_H, fill=INK, anchor='mm')
    dr.text(((TX0 + TX1) / 2 + 14, TOP + 122), 'word split', font=F_TICK, fill=MUTED, anchor='lm')
    cw = (TX1 - TX0 - 8 * 5) / 6
    for k, word in enumerate(S['words']):
        x = TX0 + k * (cw + 8)
        rrect(dr, (x, TOP + SI - 40, x + cw, TOP + SI), 8, fill=WHITE, outline=MUTED)
        dr.text((x + cw / 2, TOP + SI - 21), fits(dr, word, F_HS, cw - 8), font=F_HS, fill=INK, anchor='mm')
    dr.text((FX0, TOP - 10), 'Pixel-tag score of a tag', font=F_H, fill=TEAL, anchor='ls')
    dr.text((W - M, TOP - 10), 'Eq. (2)', font=F_HS, fill=MUTED, anchor='rs')
    rrect(dr, (FX0, TOP, W - M, TOP + SI), 10, fill=WHITE, outline=MINT_LINE, width=2)
    fx = (FX0 + W - M) / 2
    assert rich(dr, fx, TOP + 68, 's_{pixel}(t_{i})  =  max_{h, w}  sim( E_{V}(I)_{hw} , E_{T}(t_{i}) )', F_FORM, F_HSUB, INK, 'm') <= W - M - FX0 - 24
    for n, t in enumerate(wrap(dr, 'the similarity between a tag and its most correlated pixel', F_HS, W - M - FX0 - 40)):
        dr.text((fx, TOP + 114 + n * 23), t, font=F_HS, fill=INK, anchor='mm')
    rich(dr, fx, TOP + 184, 'E_{V}: image encoder (pixel embeddings) · E_{T}: text encoder', F_TICK, F_HSUB, MUTED, 'm')
    dr.text((M, HY - 14), 'Sorting by the maximum similarity score per heatmap', font=F_HB, fill=INK, anchor='ls')
    for k, (tag, score) in enumerate(SORTED):
        x = HX[k]
        if k >= shown:
            placeholder(dr, x, HY, HWD, HWD, str(k + 1)); continue
        col = LINE if not final else TEAL if k < NSEL else RED
        framed(im, dr, HEAT[k], x, HY, col, 4 if final else 3)
        label = '%s: %.2f' % (tag, score)
        dr.rectangle((x, HY, x + dr.textlength(label, font=F_HB) + 16, HY + 30), fill=WHITE)
        dr.text((x + 8, HY + 14), label, font=F_HB, fill=INK, anchor='lm')
    gy, marks = HY + HWD + 16, [((TX0 + TX1) / 2, TOP + 88), ((TX0 + TX1) / 2, TOP + SI - 50)]
    items = [(marks, MUTED, 2, True)]
    if gaps:
        for k, g in enumerate(GAPS):
            cx, big = (HX[k] + HWD + HX[k + 1]) / 2, final and k == NSEL - 1
            items.append(([(cx, gy - 1), (cx, gy - 12)], TEAL if big else MUTED, 2, False))
            label = 'Gap %d: %.2f' % (k + 1, g) + (' · the largest gap' if big else '')
            chip(dr, cx, gy, label, F_SB if big else F_HS, WHITE if big else INK, TEAL if big else LINE, h=32, anchor='m')
    if final:
        by = gy + 54
        for a, b, col, label in ((0, NSEL - 1, TEAL, 'tags relevant to the image (selected)'), (NSEL, 5, RED, 'tags irrelevant to the image')):
            items.append(([(HX[a], by - 9), (HX[a], by), (HX[b] + HWD, by), (HX[b] + HWD, by - 9)], col, 3, False))
            dr.text(((HX[a] + HX[b] + HWD) / 2, by + 22), label, font=F_HB, fill=col, anchor='mm')
    dr = strokes(im, items)
    msg = SEL_MSG[3 if final else 2 if gaps else 1 if shown else 0]
    line(dr, MSG_Y, msg, F_MSG, TEAL if final else INK)
    return im


# ---------------------------------------------------------------- scene 3: text-tag self-distillation (paper Fig. 2)
D = V['distillation']
DW, DH = 360, 252
AX, BX, CX = M, M + DW + 76, 934                                   # image and tags · heatmaps · the four steps
R1, R2 = TOP, TOP + DH + 78
D_IMG, D_CLIP, D_UNION = (load(D[k], (DW, DH)) for k in ('image', 'clip', 'union'))
NSEL_D = D['selected']
DGAPS = [a[1] - b[1] for a, b in zip(D['tags'], D['tags'][1:])]
assert DGAPS.index(max(DGAPS)) == NSEL_D - 1, 'the largest gap must be the boundary of the pseudo-tags'
TW, TH = 174, 121                                                  # heatmap of one pseudo-tag (105 x 73 in the paper figure)
D_TAGS = [load('panels/jacuzzi_tag_%s.jpg' % t, (TW, TH)) for t, _ in D['tags'][:NSEL_D]]
SH = (R2 + DH - R1 - 3 * 12) // 4                                  # height of one step card


def distillation_frame(s):
    im, dr = canvas(2, 'Text-tag self-distillation: the pseudo-tags teach the image-text map', D['note'])
    items = []
    dr.text((AX, R1 - 10), 'Image and text', font=F_H, fill=INK, anchor='ls')
    framed(im, dr, D_IMG, AX, R1, LINE)
    rrect(dr, (AX, R1 + DH + 10, AX + DW, R1 + DH + 42), 8, fill=TEAL_BG, outline=MINT_LINE)
    dr.text((AX + DW / 2, R1 + DH + 25), fits(dr, '“%s”' % D['text'], F_HS, DW - 16), font=F_HS, fill=INK, anchor='mm')
    tw = dr.textlength('CLIP heatmap', font=F_H)
    dr.text((BX, R1 - 10), 'CLIP heatmap', font=F_H, fill=BASE, anchor='ls')
    dr.text((BX + tw + 8, R1 - 10), '· image-text map', font=F_HS, fill=MUTED, anchor='ls')
    framed(im, dr, D_CLIP, BX, R1, BASE)
    items.append(([(AX + DW + 12, R1 + DH / 2), (BX - 16, R1 + DH / 2)], MUTED, 3, True))
    dr.text(((AX + DW + BX) / 2, R1 + DH / 2 - 16), 'Cosine', font=F_TICK, fill=MUTED, anchor='mm')
    dr.text((AX, R2 - 10), 'Pseudo-tags by pixel-tag scoring', font=F_H, fill=TEAL if s >= 1 else GREY, anchor='ls')
    if s < 1:
        placeholder(dr, AX, R2, DW, DH)
    else:                                                          # the tags sorted by score, cut at the largest gap
        x = AX
        for tag, score in D['tags'][:NSEL_D]:
            x = chip(dr, x, R2, '%s  %.2f' % (tag, score), F_SB, WHITE, TEAL, h=30)[2] + 8
        dr.text((x + 4, R2 + 14), 'selected', font=F_SB, fill=TEAL, anchor='lm')
        gw = dr.textlength('the largest gap', font=F_TICK)
        dr.line((AX, R2 + 45, AX + (DW - gw) / 2 - 10, R2 + 45), fill=GREY, width=1)
        dr.line((AX + (DW + gw) / 2 + 10, R2 + 45, AX + DW, R2 + 45), fill=GREY, width=1)
        dr.text((AX + DW / 2, R2 + 44), 'the largest gap', font=F_TICK, fill=MUTED, anchor='mm')
        x = AX
        for tag, score in D['tags'][NSEL_D:]:
            x = chip(dr, x, R2 + 60, '%s  %.2f' % (tag, score), F_NOTE, MUTED, WHITE, pad=8, h=30, outline=GREY)[2] + 6
        assert x - 6 <= AX + DW
        for k, (tag, _) in enumerate(D['tags'][:NSEL_D]):          # the heatmaps of the two pseudo-tags
            x = AX + k * (TW + 12)
            framed(im, dr, D_TAGS[k], x, R2 + DH - TH, TEAL)
            dr.rectangle((x, R2 + DH - TH, x + dr.textlength(tag, font=F_SB) + 14, R2 + DH - TH + 24), fill=WHITE)
            dr.text((x + 7, R2 + DH - TH + 11), tag, font=F_SB, fill=INK, anchor='lm')
    tw = dr.textlength('Union of heatmaps', font=F_H)
    dr.text((BX, R2 - 10), 'Union of heatmaps', font=F_H, fill=TEAL if s >= 2 else GREY, anchor='ls')
    dr.text((BX + tw + 8, R2 - 10), '· ideal image-text map', font=F_HS, fill=MUTED if s >= 2 else GREY, anchor='ls')
    if s >= 2:
        framed(im, dr, D_UNION, BX, R2, TEAL)
        items.append(([(AX + DW + 12, R2 + DH / 2), (BX - 16, R2 + DH / 2)], TEAL, 3, True))
        dr.text(((AX + DW + BX) / 2, R2 + DH / 2 - 16), 'union', font=F_SB, fill=TEAL, anchor='mm')
    else:
        placeholder(dr, BX, R2, DW, DH)
    if s >= 3:
        x0, x1 = BX + DW + 8, BX + DW + 40
        items.append(([(x0, R2 + DH / 2), (x1, R2 + DH / 2), (x1, R1 + DH / 2), (x0 + 8, R1 + DH / 2)], TEAL, 4, True))
        dr.text((x1 + 10, (R1 + R2 + DH) / 2), 'distill', font=F_HB, fill=TEAL, anchor='lm')
    for k, (title, text) in enumerate(D['steps']):
        y, on, cur = R1 + k * (SH + 12), k <= s, k == s
        rrect(dr, (CX, y, W - M, y + SH), 12, fill=TEAL if cur else TEAL_BG if on else WHITE, outline=None if cur else MINT_LINE if on else LINE)
        dr.text((CX + 16, y + 26), title, font=F_HB, fill=WHITE if cur else INK if on else GREY, anchor='lm')
        if on:
            for n, t in enumerate(wrap(dr, text, F_NOTE, W - M - CX - 32)):
                assert n < 4, text
                dr.text((CX + 16, y + 56 + n * 21), t, font=F_NOTE, fill=TEAL_SOFT if cur else MUTED, anchor='lm')
    dr = strokes(im, items)
    if s >= 3:
        f = 'Final loss:  L = L_{distill} + L_{tag}   (Eq. 6)'
        rich(dr, W / 2, 888, f, F_MSG, F_HBSUB, TEAL, 'm')
    return im


# ---------------------------------------------------------------- scene 4: results (paper Fig. 6 and Fig. 7, Tables 2 to 4)
R = V['results']
RW = 198
RX = [M + round(k * (RW + (W - 2 * M - 6 * RW) / 5)) for k in range(6)]
RCOL = {'image': MUTED, 'gt': MUTED, 'maskclip': BASE, 'maskclip_ours': TEAL, 'tcl': BASE, 'tcl_ours': TEAL}


def panel_file(prefix, row, col):
    for ext in ('.jpg', '.png'):
        if os.path.exists(os.path.join(HERE, 'panels', '%s_%s_%s%s' % (prefix, row, col, ext))):
            return 'panels/%s_%s_%s%s' % (prefix, row, col, ext)


def results_frame(part):
    P = R[part]
    im, dr = canvas(3, P['head'], P['note'])
    for j, (key, title) in enumerate(R['columns']):
        dr.text((RX[j] + RW / 2, TOP - 10), title, font=F_H, fill=INK if RCOL[key] == MUTED else RCOL[key], anchor='ms')
    y = TOP
    for row, caption, h in P['rows']:
        if part == 'text':                                         # the input text of the row, above its pictures
            dr.text((M, y + 11), 'Text', font=F_SB, fill=TEAL, anchor='lm')
            dr.text((M + 42, y + 11), '“%s”' % caption, font=F_HS, fill=INK, anchor='lm')
            y += 26
        for j, (key, _) in enumerate(R['columns']):                # as in the paper figure, the pictures of a row share one box
            framed(im, dr, load(panel_file('text' if part == 'text' else 'tag', row, key), (RW, h)), RX[j], y, LINE if RCOL[key] == MUTED else RCOL[key])
        if part == 'tags':                                         # the benchmark of the row, on its image
            chip(dr, RX[0], y, caption, F_SB, WHITE, INK, pad=8, h=24)
        y += h + 12
    y += 8
    k = len(P['cards']); cw = (W - 2 * M - 16 * (k - 1)) / k
    for j, (value, metric, desc) in enumerate(P['cards']):
        x = M + j * (cw + 16)
        rrect(dr, (x, y, x + cw, y + 120), 12, fill=TEAL_BG, outline=MINT_LINE)
        dr.text((x + cw / 2, y + 36), value, font=F_BIG, fill=TEAL, anchor='mm')
        dr.text((x + cw / 2, y + 74), metric, font=F_HB, fill=INK, anchor='mm')
        dr.text((x + cw / 2, y + 99), fits(dr, desc, F_NOTE, cw - 16), font=F_NOTE, fill=MUTED, anchor='mm')
    if 'line' in P:
        line(dr, y + 120 + 34, P['line'], F_MSG, TEAL)
    return im


frames = []
add = lambda im, ms: frames.append((im, ms))
for s, ms in enumerate((2200, 3200, 4000)):
    add(bias_frame(s), ms)
add(selection_frame(0), 2400)
for k in range(1, 7):
    add(selection_frame(k), 700)
add(selection_frame(6, gaps=True), 2400)
add(selection_frame(6, gaps=True, final=True), 4200)
for s, ms in enumerate((2400, 2400, 2600, 4000)):
    add(distillation_frame(s), ms)
add(results_frame('text'), 4800)
add(results_frame('tags'), 5400)

# GIF palette: the flat UI colors and the main logo colors are reserved so text and logos stay clean; the rest of each frame is quantized.
def logo_colors(im, n=8):
    flat = Image.new('RGB', im.size, WHITE); flat.paste(im, (0, 0), im)
    p = flat.quantize(colors=n, method=Image.MEDIANCUT).getpalette()[:3 * n]
    return [tuple(p[i:i + 3]) for i in range(0, len(p), 3)]


RESERVED = list(dict.fromkeys([TEAL, TEAL_BG, MINT_LINE, INK, MUTED, LINE, WHITE, (0, 0, 0), GREY, TAB_DIM, BASE, RED, GREEN, TEAL_SOFT]
                              + [c for lg in LOGOS_L + LOGOS_R for c in logo_colors(lg)]))


def to_palette(f):
    k = len(RESERVED)
    q = f.quantize(colors=256 - k, method=Image.MEDIANCUT)
    p = (q.getpalette() or [])[:3 * (256 - k)]
    p = [v for c in RESERVED for v in c] + p + [0] * (3 * (256 - k) - len(p))
    pi = Image.new('P', (1, 1)); pi.putpalette(p)
    out = f.quantize(palette=pi, dither=Image.FLOYDSTEINBERG)
    idx, rgb = np.array(out), np.array(f.convert('RGB'))
    for j, col in enumerate(RESERVED):
        idx[(rgb == col).all(-1)] = j
    fixed = Image.frombytes('P', out.size, idx.astype(np.uint8).tobytes()); fixed.putpalette(p)
    return fixed


total_ms = sum(ms for _, ms in frames)
if args.preview:
    os.makedirs(args.preview, exist_ok=True)
    for i, (f, _) in enumerate(frames):
        f.save(os.path.join(args.preview, f'frame_{i:03d}.png'))
    json.dump([ms for _, ms in frames], open(os.path.join(args.preview, 'durations.json'), 'w'))
    print(f'{len(frames)} frames ({total_ms / 1000:.1f} s) written to {args.preview}')
else:
    before = open(args.out, 'rb').read() if os.path.exists(args.out) else None
    pal = [to_palette(f) for f, _ in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=1)
    after = open(args.out, 'rb').read()
    state = 'new file' if before is None else 'identical to the previous file' if before == after else 'differs from the previous file'
    print(f'{args.out}: {len(frames)} frames, {total_ms / 1000:.1f} s, {len(after) / 1e6:.1f} MB ({state})')
