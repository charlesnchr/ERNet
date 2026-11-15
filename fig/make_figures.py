"""Regenerate the README figures from the bundled test images.

    python segment.py testdata/input/*.png
    python segment.py --model sheets testdata/input_sim/*.png
    python fig/make_figures.py

Every segmentation and graph shown is the output of segment.py with the
released weights. The script reads the class maps from the output folder and
rebuilds the skeleton graph from them. Display contrast of the inputs is
stretched per crop. Needs ffmpeg for the GIF.

    python fig/make_figures.py --sweep

prints the fragmentation of the local threshold for every setting tried, the
numbers quoted next to the threshold figure in the README.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import filters

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import segment as S  # noqa: E402

BG = (13, 17, 23)
INK = (240, 246, 252)
MUTED = (139, 148, 158)
RULE = (48, 54, 61)

PREFIX = "190624_COS7_3_LAMP1-mCherry_ER-YFP_optokin_2s_40x_1pt5_01_2_MMStack_Pos0-1"
# key: (file, model, description)
SAMPLES = {
    "early": (os.path.join("testdata", "input", PREFIX + "0000.png"), "tubules", "Frame 0"),
    "late": (os.path.join("testdata", "input", PREFIX + "0125.png"), "tubules", "Frame 125, 250 s later"),
    "sim": (os.path.join("testdata", "input_sim", "TestImage1.png"), "sheets", "SIM"),
}

# crops as (x, y, width, height) in native pixels
CELL = (340, 150, 880, 660)        # the whole cell
LOBE = (800, 470, 360, 360)        # the lobe that fills in during the recording
NETWORK = (455, 200, 360, 360)     # the flat network above the nucleus
SIM_CROP = (470, 330, 360, 360)


def font(size, weight="Regular"):
    for path in ("/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight,
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight != "Regular" else "")):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_cache = {}


def sample(key, seg_dir):
    """Input, class map and skeleton graph of a test image."""
    if key not in _cache:
        path, model, _ = SAMPLES[key]
        arr = next(S.read_frames(os.path.join(ROOT, path)))
        img = S.normalise(arr, S.MODELS[model]["stretch"])
        name = os.path.splitext(os.path.basename(path))[0]
        classes = np.asarray(Image.open(os.path.join(seg_dir, name + "_classes.png")))
        skel, graph, positions = S.tubule_graph(classes == S.TUBULE)
        _cache[key] = dict(img=img, classes=classes, skel=skel, graph=graph, positions=positions)
    return _cache[key]


def stretch(crop):
    lo, hi = np.percentile(crop, (1, 99.8))
    return np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1)


def grey_tile(img, crop, size, dim=1.0):
    x, y, w, h = crop
    g = (stretch(img[y:y + h, x:x + w]) * 255 * dim).astype("uint8")
    return Image.fromarray(g).resize(size, Image.LANCZOS).convert("RGB")


def blend(tile, mask, crop, colour, strength=0.62):
    """Blend a colour over the tile where the native-resolution mask is set."""
    x, y, w, h = crop
    alpha = Image.fromarray((mask[y:y + h, x:x + w] * 255).astype("uint8")).resize(tile.size, Image.BILINEAR)
    alpha = alpha.point(lambda v: int(v * strength))
    tile.paste(Image.new("RGB", tile.size, colour), (0, 0), alpha)
    return tile


def render_input(s, crop, size):
    return grey_tile(s["img"], crop, size)


def render_overlay(s, crop, size):
    tile = grey_tile(s["img"], crop, size)
    for grey, colour in S.COLOURS.items():
        blend(tile, s["classes"] == grey, crop, colour)
    return tile


def render_mask(img, mask, crop, size):
    return blend(grey_tile(img, crop, size, dim=0.55), mask, crop, S.COLOURS[S.TUBULE], strength=0.85)


def render_graph(s, crop, size, line=None, radius=None):
    """Skeleton as anti-aliased lines over the dimmed input, with the nodes on top."""
    x, y, w, h = crop
    scale = size[0] / w
    line = line or max(2.0, 1.9 * scale)
    radius = radius or max(2.8, 2.8 * scale)
    aa = 3  # draw at three times the size, then shrink
    tile = grey_tile(s["img"], crop, size, dim=0.42).resize((size[0] * aa, size[1] * aa), Image.BILINEAR)
    draw = ImageDraw.Draw(tile)
    k = scale * aa

    def pt(yy, xx):
        return ((xx - x + 0.5) * k, (yy - y + 0.5) * k)

    skel = s["skel"]
    pad = 4
    sub = skel[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    oy, ox = max(y - pad, 0), max(x - pad, 0)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a, b = S.shifted(sub, dy, dx)
        yy, xx = np.nonzero(a & b)
        # a holds the pixel at +(dy, dx) of the pixel in b
        y0, x0 = yy + max(-dy, 0), xx + max(-dx, 0)
        for py, px in zip(y0.tolist(), x0.tolist()):
            draw.line([pt(py + oy, px + ox), pt(py + dy + oy, px + dx + ox)],
                      fill=S.SKELETON_COLOUR, width=int(round(line * aa)))
    r = radius * aa
    for node, degree in s["graph"].degree():
        if degree == 2:
            continue
        ny, nx_ = s["positions"][node]
        if not (y - 3 <= ny < y + h + 3 and x - 3 <= nx_ < x + w + 3):
            continue
        cx, cy = pt(ny, nx_)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=S.node_colour(degree))
    return tile.resize(size, Image.LANCZOS)


def pill(draw, xy, text, fnt, anchor="la", fill=(0, 0, 0), ink=INK):
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=fnt, anchor=anchor)
    draw.rounded_rectangle((x0 - 8, y0 - 5, x1 + 8, y1 + 5), radius=6, fill=fill)
    draw.text(xy, text, font=fnt, anchor=anchor, fill=ink)


def legend(draw, x, y, fnt, items):
    """A row of colour dots with labels. Returns the x position after the last item."""
    for colour, text in items:
        draw.ellipse((x, y - 6, x + 12, y + 6), fill=colour)
        draw.text((x + 18, y), text, font=fnt, fill=MUTED, anchor="lm")
        x += 18 + draw.textlength(text, font=fnt) + 22
    return x


NODE_LEGEND = ((S.TIP_COLOUR, "tip"), (S.JUNCTION_COLOUR, "three-way junction"), (S.MULTIWAY_COLOUR, "more than three"))

# ---------------------------------------------------------------- hero animation

HERO_TILES = (("late", CELL, (640, 480), "COS-7 cell, wide-field", "tubule model"),
              ("sim", SIM_CROP, (480, 480), "ER in a SIM image", "sheet and tubule model"))
STAGES = ("INPUT", "SEGMENTATION", "GRAPH")


def hero_frame(layers, top, under, p, margin=16, gap=12, caption_h=46):
    """One frame of the wipe. `top` is revealed from the left over `under`, p in [0, 1]."""
    width = 2 * margin + sum(t[2][0] for t in HERO_TILES) + gap * (len(HERO_TILES) - 1)
    height = HERO_TILES[0][2][1]
    canvas = Image.new("RGB", (width, margin + height + caption_h), BG)
    draw = ImageDraw.Draw(canvas)
    label, caption, small = font(15, "Bold"), font(17, "Bold"), font(14)
    x0 = margin
    for i, (_, _, (w, h), name, model) in enumerate(HERO_TILES):
        split = int(round(p * w))
        tile = layers[i][under].copy()
        tile.paste(layers[i][top].crop((0, 0, split, h)), (0, 0))
        td = ImageDraw.Draw(tile)
        if split > 150:
            pill(td, (12, 12), STAGES[top], label)
        if w - split > 150:
            pill(td, (w - 12, 12), STAGES[under], label, anchor="ra")
        if 0 < split < w:
            td.line((split, 0, split, h), fill=INK, width=2)
        canvas.paste(tile, (x0, margin))
        draw.text((x0, margin + h + 12), name, font=caption, fill=INK)
        draw.text((x0 + w, margin + h + 14), model, font=small, fill=MUTED, anchor="ra")
        x0 += w + gap
    return canvas


def make_hero(seg_dir, out, fps=20):
    layers = []
    for key, crop, size, _, _ in HERO_TILES:
        s = sample(key, seg_dir)
        layers.append((render_input(s, crop, size), render_overlay(s, crop, size), render_graph(s, crop, size)))

    def ease(n):
        return [(1 - np.cos(np.pi * t)) / 2 for t in np.linspace(0, 1, n)]

    sweep = ease(int(fps * 1.6))
    # (revealed layer, layer underneath, position of the wipe)
    frames = [(1, 0, 0.0)] * int(fps * 0.9)
    frames += [(1, 0, p) for p in sweep] + [(1, 0, 1.0)] * int(fps * 1.2)
    frames += [(2, 1, p) for p in sweep] + [(2, 1, 1.0)] * int(fps * 2.0)
    frames += [(2, 0, p) for p in sweep[::-1]]
    tmp = tempfile.mkdtemp()
    try:
        for i, (top, under, p) in enumerate(frames):
            hero_frame(layers, top, under, p).save(os.path.join(tmp, "f%04d.png" % i))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
             "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none",
             out], check=True)
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------- results grid

GRID = (("late", NETWORK, "Peripheral network", "wide-field, tubule model"),
        ("late", LOBE, "Polygonal network", "wide-field, tubule model"),
        ("sim", SIM_CROP, "Sheets and tubules", "SIM, sheet and tubule model"))


def make_grid(seg_dir, out, size=380):
    gap, margin, left, head, foot = 10, 20, 44, 78, 44
    width = left + margin + len(GRID) * size + (len(GRID) - 1) * gap + margin
    height = head + 3 * size + 2 * gap + foot
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    for i, (key, crop, name, sub) in enumerate(GRID):
        s = sample(key, seg_dir)
        x0 = left + margin + i * (size + gap)
        draw.text((x0, 12), name, font=font(26, "Bold"), fill=INK)
        draw.text((x0, 46), sub, font=font(18), fill=MUTED)
        for row, render in enumerate((render_input, render_overlay, render_graph)):
            canvas.paste(render(s, crop, (size, size)), (x0, head + row * (size + gap)))
    for row, text in enumerate(STAGES):
        strip = Image.new("RGB", (size, 36), BG)
        ImageDraw.Draw(strip).text((size // 2, 18), text, font=font(23, "Bold"), fill=INK if row else MUTED, anchor="mm")
        canvas.paste(strip.rotate(90, expand=True), (margin - 2, head + row * (size + gap)))
    y = height - foot // 2
    x = legend(draw, left + margin, y, font(17),
               ((S.COLOURS[S.TUBULE], "tubule"), (S.COLOURS[S.SHEET], "sheet")))
    legend(draw, x + 30, y, font(17), NODE_LEGEND)
    canvas.save(out, optimize=True)


# ---------------------------------------------------------------- remodelling over 250 s

def read_metrics(seg_dir):
    with open(os.path.join(seg_dir, "metrics.csv")) as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def make_remodelling(seg_dir, out, size=(560, 420)):
    """The same cell early and late in the recording, with the measurements side by side."""
    metrics = read_metrics(seg_dir)
    margin, gap, head, table_w = 20, 12, 60, 470
    width = margin + 2 * size[0] + gap + 28 + table_w + margin
    height = head + size[1] + margin + 30
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    rows = []
    for i, key in enumerate(("early", "late")):
        s = sample(key, seg_dir)
        x0 = margin + i * (size[0] + gap)
        draw.text((x0, 14), SAMPLES[key][2], font=font(26, "Bold"), fill=INK)
        canvas.paste(render_graph(s, CELL, size), (x0, head))
        rows.append(metrics[os.path.basename(SAMPLES[key][0])])
    legend(draw, margin, head + size[1] + 24, font(17), NODE_LEGEND)

    table = (("ER area", "er_fraction", lambda v: "%.1f%%" % (100 * float(v)), "of the frame"),
             ("Skeleton length", "skeleton_length_px", lambda v: "{:,} px".format(int(v)), ""),
             ("Three-way junctions", "three_way_junctions", str, ""),
             ("Tubules (edges)", "edges", str, ""),
             ("Free tips", "tips", str, ""),
             ("Fragments", "fragments", str, "disconnected pieces"),
             ("Largest fragment", "largest_fragment_share", lambda v: "%.0f%%" % (100 * float(v)), "of all nodes"))
    x0 = margin + 2 * size[0] + gap + 28
    cols = (x0 + 275, x0 + 380, x0 + table_w)
    y = head + 6
    for x, text in zip(cols, ("Frame 0", "Frame 125", "Change")):
        draw.text((x, y), text, font=font(17, "Bold"), fill=MUTED, anchor="ra")
    y += 34
    for name, column, fmt, note in table:
        a, b = float(rows[0][column]), float(rows[1][column])
        draw.line((x0, y - 12, x0 + table_w, y - 12), fill=RULE, width=1)
        draw.text((x0, y), name, font=font(19, "Bold"), fill=INK)
        if note:
            draw.text((x0, y + 24), note, font=font(15), fill=MUTED)
        draw.text((cols[0], y), fmt(rows[0][column]), font=font(19), fill=INK, anchor="ra")
        draw.text((cols[1], y), fmt(rows[1][column]), font=font(19), fill=INK, anchor="ra")
        draw.text((cols[2], y), "%+.0f%%" % (100 * (b - a) / a), font=font(19, "Bold"), fill=INK, anchor="ra")
        y += 52
    canvas.save(out, optimize=True)


# ---------------------------------------------------------------- intensity thresholds

# smoothing sigma, window size and offset of the local threshold. This is the
# setting with the best connected network out of the ones in SWEEP.
LOCAL = (2.0, 101, -0.01)
SWEEP = [(sigma, block, offset) for sigma in (1.0, 2.0) for block in (31, 51, 101, 151)
         for offset in (-0.01, -0.02, -0.03, -0.05)]


def otsu_mask(img):
    smooth = ndimage.gaussian_filter(img, 1.0)
    return smooth > filters.threshold_otsu(smooth)


def local_mask(img, sigma, block, offset):
    smooth = ndimage.gaussian_filter(img, sigma)
    return smooth > filters.threshold_local(smooth, block_size=block, offset=offset)


def network_stats(mask):
    """Run the graph extraction of segment.py on any binary mask."""
    skel, graph, _ = S.tubule_graph(mask)
    return S.graph_metrics(skel, graph)


def sweep(seg_dir, key="late"):
    s = sample(key, seg_dir)
    print("sigma block offset fragments largest_fragment_share three_way_junctions")
    for sigma, block, offset in SWEEP:
        m = network_stats(local_mask(s["img"], sigma, block, offset))
        print(sigma, block, offset, m["fragments"], m["largest_fragment_share"], m["three_way_junctions"])
    m = network_stats(s["classes"] == S.TUBULE)
    print("ERNet", m["fragments"], m["largest_fragment_share"], m["three_way_junctions"])


def make_threshold(seg_dir, out, size=(450, 338), key="late"):
    """Intensity thresholds against ERNet on the whole cell, with the fragmentation of each mask."""
    s = sample(key, seg_dir)
    masks = (("Otsu threshold", otsu_mask(s["img"])),
             ("Local threshold", local_mask(s["img"], *LOCAL)),
             ("ERNet", s["classes"] == S.TUBULE))
    tiles = [("Input", "wide-field image as recorded", render_input(s, CELL, size))]
    for name, mask in masks:
        m = network_stats(mask)
        note = "%d fragments, the largest holds %.0f%% of the nodes" % (
            m["fragments"], 100 * float(m["largest_fragment_share"]))
        tiles.append((name, note, render_mask(s["img"], mask, CELL, size)))
    gap, margin, head = 12, 20, 70
    canvas = Image.new("RGB", (2 * margin + 2 * size[0] + gap, 2 * (head + size[1]) + margin), BG)
    draw = ImageDraw.Draw(canvas)
    for i, (name, sub, tile) in enumerate(tiles):
        x0 = margin + (i % 2) * (size[0] + gap)
        y0 = (i // 2) * (head + size[1])
        draw.text((x0, y0 + 12), name, font=font(24, "Bold"), fill=INK)
        draw.text((x0, y0 + 43), sub, font=font(17), fill=MUTED)
        canvas.paste(tile, (x0, y0 + head))
    canvas.save(out, optimize=True)


# ---------------------------------------------------------------- social preview

def make_social(seg_dir, out):
    """1280x640 image for the repository's social preview."""
    w, h, size, gap = 1280, 640, 330, 22
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 48), "ERNet", font=font(84, "Black"), fill=INK)
    draw.text((66, 152), "Segment the endoplasmic reticulum and measure its network",
              font=font(31, "Medium"), fill=MUTED)
    draw.text((w - 64, 96), "Science Advances 2020", font=font(22), fill=MUTED, anchor="ra")
    s = sample("late", seg_dir)
    x = (w - (3 * size + 2 * gap)) // 2
    for text, render in zip(STAGES, (render_input, render_overlay, render_graph)):
        tile = render(s, LOBE, (size, size))
        pill(ImageDraw.Draw(tile), (12, 12), text, font(14, "Bold"))
        canvas.paste(tile, (x, 222))
        x += size + gap
    x = (w - (3 * size + 2 * gap)) // 2
    draw.text((x, 222 + size + 26), "Live COS-7 cell, wide-field microscope", font=font(22, "Bold"), fill=INK, anchor="lm")
    canvas.save(out, optimize=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg", default=os.path.join(ROOT, "segmented"), help="folder written by segment.py")
    p.add_argument("--out", default=HERE)
    p.add_argument("--only", nargs="*", help="figures to make, out of hero results remodelling threshold social")
    p.add_argument("--sweep", action="store_true", help="print the local threshold sweep and exit")
    args = p.parse_args()
    if args.sweep:
        sweep(args.seg)
        sys.exit()
    figures = {"hero": (make_hero, "hero.gif"), "results": (make_grid, "results.png"),
               "remodelling": (make_remodelling, "remodelling.png"),
               "threshold": (make_threshold, "threshold.png"), "social": (make_social, "social-preview.png")}
    for name in args.only or figures:
        make, filename = figures[name]
        make(args.seg, os.path.join(args.out, filename))
