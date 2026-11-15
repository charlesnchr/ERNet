"""Segment endoplasmic reticulum images with the pre-trained ERNet models and
measure the tubular network.

    python segment.py testdata/input/*.png

Inputs are greyscale images (PNG, JPG, TIFF) or multi-frame TIFF stacks. For
every frame the script writes the class map, a colour overlay and the skeleton
graph of the tubules drawn over the input. One row per frame goes to
metrics.csv in the output folder.

The model is the one in pretrained_model/, which separates tubules from
background in wide-field images.
"""

import argparse
import csv
import os

import networkx as nx
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageSequence
from scipy import ndimage
from skimage.morphology import skeletonize

from models import RCAN

HERE = os.path.dirname(os.path.abspath(__file__))

# grey values of the class maps, the same as evaluate.py writes
BACKGROUND, SHEET, TUBULE = 0, 127, 255

MODELS = {
    "tubules": dict(
        path=os.path.join(HERE, "pretrained_model", "final.pth"),
        n_resgroups=5, grey=(BACKGROUND, TUBULE),
        stretch=None,  # trained on 8-bit images as they are stored
    ),
}

COLOURS = {TUBULE: (0, 200, 255), SHEET: (255, 200, 0)}
TIP_COLOUR, JUNCTION_COLOUR, MULTIWAY_COLOUR = (70, 230, 100), (255, 80, 60), (200, 120, 255)
SKELETON_COLOUR = (0, 200, 255)

COLUMNS = ["file", "frame", "model", "er_fraction", "tubule_fraction", "sheet_fraction",
           "skeleton_length_px", "nodes", "edges", "tips", "three_way_junctions",
           "multiway_junctions", "fragments", "largest_fragment_share",
           "assortativity", "clustering"]


def load_model(name, device, weights=None):
    """Build the RCAN of the chosen model and load its weights."""
    spec = MODELS[name]
    path = weights or spec["path"]
    if not os.path.isfile(path):
        raise SystemExit("No such file: %s" % path)
    opt = argparse.Namespace(nch_in=1, nch_out=len(spec["grey"]), n_resgroups=spec["n_resgroups"],
                             n_resblocks=10, n_feats=64, reduction=16, narch=0)
    net = RCAN(opt)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()})
    return net.to(device).eval()


def read_frames(path):
    """Yield the frames of an image file as 2D arrays in their stored bit depth."""
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            arr = np.asarray(frame)
            if arr.ndim == 3:  # colour image, keep the brightest channel
                arr = arr[..., :3].max(axis=2)
            yield arr


def normalise(arr, stretch):
    """Return the frame as float32 in [0, 1], quantised to 8 bit like the training data."""
    if stretch is None and arr.dtype == np.uint8:
        return arr.astype("float32") / 255
    lo, hi = np.percentile(arr, stretch or (0, 99.9))
    img = np.clip((arr.astype("float32") - lo) / max(hi - lo, 1e-9), 0, 1)
    return (np.round(img * 255) / 255).astype("float32")


def tile_starts(length, tile, margin):
    """Start positions and size of equal tiles that cover a side with overlap."""
    if length <= tile:
        return [0], length
    n = int(np.ceil((length - 2 * margin) / (tile - 2 * margin)))
    size = int(np.ceil((length + (n - 1) * 2 * margin) / n))
    return [round(i * (length - size) / (n - 1)) for i in range(n)], size


def segment(net, img, device, tile=1536, margin=32):
    """Return the class index of every pixel.

    Frames that fit in one tile are processed in a single pass. Larger frames
    are split into overlapping tiles, and the overlap is discarded.
    """
    h, w = img.shape
    ys, th = tile_starts(h, tile, margin)
    xs, tw = tile_starts(w, tile, margin)
    out = np.zeros((h, w), dtype="uint8")
    x = torch.from_numpy(img)[None, None].to(device)
    for ty in ys:
        for tx in xs:
            with torch.no_grad():
                pred = net(x[:, :, ty:ty + th, tx:tx + tw])[0].argmax(0).cpu().numpy()
            y0 = 0 if ty == 0 else margin
            x0 = 0 if tx == 0 else margin
            y1 = th if ty + th == h else th - margin
            x1 = tw if tx + tw == w else tw - margin
            out[ty + y0:ty + y1, tx + x0:tx + x1] = pred[y0:y1, x0:x1]
    return out


EIGHT = np.ones((3, 3), dtype=bool)
SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def shifted(a, dy, dx):
    """Return a view of a and of a shifted by (dy, dx) that line up pixel by pixel."""
    h, w = a.shape
    return (a[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)],
            a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)])


def prune_spurs(skel, min_branch):
    """Remove branches that end in a tip and are shorter than min_branch pixels."""
    skel = skel.copy()
    for _ in range(3):
        neighbours = ndimage.convolve(skel.astype("uint8"), EIGHT.astype("uint8"), mode="constant") - 1
        chain = skel & (neighbours <= 2)  # tips and the pixels of unbranched lines
        labels, n = ndimage.label(chain, EIGHT)
        if n == 0:
            break
        size = ndimage.sum(chain, labels, index=np.arange(1, n + 1))
        has_tip = ndimage.maximum((neighbours == 1).astype("uint8"), labels, index=np.arange(1, n + 1)) > 0
        # only branches attached to a junction are spurs, free-standing lines are kept
        junction = ndimage.binary_dilation(skel & (neighbours > 2), EIGHT).astype("uint8")
        attached = ndimage.maximum(junction, labels, index=np.arange(1, n + 1)) > 0
        drop = np.flatnonzero(has_tip & attached & (size < min_branch)) + 1
        if drop.size == 0:
            break
        skel[np.isin(labels, drop)] = False
        skel = skeletonize(skel)
    return skel


def tubule_graph(tubules, min_area=50, min_branch=6):
    """Thin the tubule mask to a skeleton and convert it to a graph.

    Pixels of the skeleton with one neighbour (tips) or more than two
    (junctions) become nodes, and the lines between them become edges.
    Returns the skeleton, the graph and the node positions.
    """
    labels, n = ndimage.label(tubules, EIGHT)
    if n:
        area = ndimage.sum(tubules, labels, index=np.arange(1, n + 1))
        tubules = np.isin(labels, np.flatnonzero(area >= min_area) + 1)
    skel = prune_spurs(skeletonize(tubules), min_branch)

    neighbours = ndimage.convolve(skel.astype("uint8"), EIGHT.astype("uint8"), mode="constant") - 1
    node_labels, n_nodes = ndimage.label(skel & (neighbours != 2), EIGHT)
    line_labels, n_lines = ndimage.label(skel & (neighbours == 2), EIGHT)

    graph = nx.MultiGraph()
    graph.add_nodes_from(range(1, n_nodes + 1))
    ends = [set() for _ in range(n_lines + 1)]  # the nodes each line touches
    for dy, dx in SHIFTS:
        a, b = shifted(line_labels, dy, dx)[0], shifted(node_labels, dy, dx)[1]
        touching = (a > 0) & (b > 0)
        for line, node in set(zip(a[touching].tolist(), b[touching].tolist())):
            ends[line].add(node)
    length = ndimage.sum(line_labels > 0, line_labels, index=np.arange(1, n_lines + 1)) if n_lines else []
    for line in range(1, n_lines + 1):
        nodes = sorted(ends[line])
        if len(nodes) == 2:
            graph.add_edge(nodes[0], nodes[1], length=int(length[line - 1]))
        elif len(nodes) == 1:  # a loop that leaves a junction and returns to it
            graph.add_edge(nodes[0], nodes[0], length=int(length[line - 1]))
    # junction pixels that touch without a line in between belong to one node,
    # so two nodes are never directly adjacent
    graph.remove_nodes_from([n for n, d in dict(graph.degree()).items() if d == 0])
    centres = ndimage.center_of_mass(node_labels > 0, node_labels, index=list(graph.nodes)) if len(graph) else []
    positions = dict(zip(graph.nodes, centres))
    return skel, graph, positions


def graph_metrics(skel, graph):
    degrees = [d for _, d in graph.degree()]
    simple = nx.Graph(graph)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    fragments = list(nx.connected_components(simple))
    out = {
        "skeleton_length_px": int(skel.sum()),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "tips": sum(d == 1 for d in degrees),
        "three_way_junctions": sum(d == 3 for d in degrees),
        "multiway_junctions": sum(d > 3 for d in degrees),
        "fragments": len(fragments),
        "largest_fragment_share": "", "assortativity": "", "clustering": "",
    }
    if graph.number_of_edges() > 1:
        out["largest_fragment_share"] = round(max(len(f) for f in fragments) / len(simple), 4)
        with np.errstate(all="ignore"):
            assortativity = nx.degree_assortativity_coefficient(simple)
        out["assortativity"] = "" if np.isnan(assortativity) else round(float(assortativity), 4)
        out["clustering"] = round(nx.average_clustering(simple), 4)
    return out


def to_rgb(img):
    return np.repeat((np.clip(img, 0, 1) * 255).astype("uint8")[..., None], 3, axis=2)


def render_overlay(img, classes, strength=0.6):
    """Blend the class colours over the greyscale input."""
    rgb = to_rgb(img).astype("float32")
    for grey, colour in COLOURS.items():
        sel = classes == grey
        rgb[sel] = (1 - strength) * rgb[sel] + strength * np.asarray(colour, dtype="float32")
    return Image.fromarray(rgb.astype("uint8"))


def node_colour(degree):
    return TIP_COLOUR if degree == 1 else JUNCTION_COLOUR if degree == 3 else MULTIWAY_COLOUR


def render_graph(img, skel, graph, positions, dim=0.45, radius=3):
    """Draw the skeleton and its nodes over the dimmed input.

    Tips are green, three-way junctions red and junctions of more tubules violet.
    """
    rgb = (to_rgb(img).astype("float32") * dim).astype("uint8")
    rgb[ndimage.binary_dilation(skel, EIGHT)] = SKELETON_COLOUR
    out = Image.fromarray(rgb)
    draw = ImageDraw.Draw(out)
    for node, degree in graph.degree():
        if degree == 2:
            continue
        y, x = positions[node]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=node_colour(degree))
    return out


def analyse(net, model, arr, device, tile=1536, stretch=None):
    """Segment one frame. Returns the normalised input, the class map, the graph parts and the metrics."""
    spec = MODELS[model]
    img = normalise(arr, stretch or spec["stretch"])
    classes = np.asarray(spec["grey"], dtype="uint8")[segment(net, img, device, tile)]
    tubules, sheets = classes == TUBULE, classes == SHEET
    skel, graph, positions = tubule_graph(tubules)
    er = tubules.sum() + sheets.sum()
    metrics = {
        "model": model,
        "er_fraction": round(float(er) / classes.size, 4),
        "tubule_fraction": round(float(tubules.sum()) / er, 4) if er else "",
        "sheet_fraction": round(float(sheets.sum()) / er, 4) if er else "",
    }
    metrics.update(graph_metrics(skel, graph))
    return img, classes, (skel, graph, positions), metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="greyscale images or TIFF stacks")
    p.add_argument("--model", choices=sorted(MODELS), default="tubules", help="which pre-trained model to use")
    p.add_argument("--weights", help="use this checkpoint instead of the released one")
    p.add_argument("--out", default="segmented", help="output folder")
    p.add_argument("--stretch", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   help="stretch each frame between these percentiles before segmenting, e.g. 1 99")
    p.add_argument("--tile", type=int, default=1536, help="largest side processed in one pass, in pixels")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = load_model(args.model, device, args.weights)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for path in args.images:
        name = os.path.splitext(os.path.basename(path))[0]
        frames = list(read_frames(path))
        for i, arr in enumerate(frames):
            img, classes, (skel, graph, positions), metrics = analyse(
                net, args.model, arr, device, args.tile, args.stretch)
            base = os.path.join(args.out, name if len(frames) == 1 else "%s_f%04d" % (name, i))
            Image.fromarray(classes).save(base + "_classes.png")
            render_overlay(img, classes).save(base + "_overlay.png")
            render_graph(img, skel, graph, positions).save(base + "_graph.png")
            rows.append(dict(file=os.path.basename(path), frame=i, **metrics))
            print("%s frame %d: %d nodes, %d edges, %d tips, %d three-way junctions" % (
                os.path.basename(path), i, metrics["nodes"], metrics["edges"],
                metrics["tips"], metrics["three_way_junctions"]))

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("Results are in", args.out)


if __name__ == "__main__":
    main()
