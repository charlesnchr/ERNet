"""Segment endoplasmic reticulum images with the pre-trained ERNet model.

    python segment.py testdata/input/*.png

Inputs are greyscale images (PNG, JPG, TIFF) or multi-frame TIFF stacks. For
every frame the script writes the class map and a colour overlay.

The model is the one in pretrained_model/, which separates tubules from
background in wide-field images.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageSequence

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


def to_rgb(img):
    return np.repeat((np.clip(img, 0, 1) * 255).astype("uint8")[..., None], 3, axis=2)


def render_overlay(img, classes, strength=0.6):
    """Blend the class colours over the greyscale input."""
    rgb = to_rgb(img).astype("float32")
    for grey, colour in COLOURS.items():
        sel = classes == grey
        rgb[sel] = (1 - strength) * rgb[sel] + strength * np.asarray(colour, dtype="float32")
    return Image.fromarray(rgb.astype("uint8"))


def analyse(net, model, arr, device, tile=1536, stretch=None):
    """Segment one frame. Returns the normalised input and the class map."""
    spec = MODELS[model]
    img = normalise(arr, stretch or spec["stretch"])
    classes = np.asarray(spec["grey"], dtype="uint8")[segment(net, img, device, tile)]
    return img, classes



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

    for path in args.images:
        name = os.path.splitext(os.path.basename(path))[0]
        frames = list(read_frames(path))
        for i, arr in enumerate(frames):
            img, classes = analyse(net, args.model, arr, device, args.tile, args.stretch)
            base = os.path.join(args.out, name if len(frames) == 1 else "%s_f%04d" % (name, i))
            Image.fromarray(classes).save(base + "_classes.png")
            render_overlay(img, classes).save(base + "_overlay.png")
            print("Segmented %s frame %d" % (os.path.basename(path), i))
    print("Results are in", args.out)


if __name__ == "__main__":
    main()
