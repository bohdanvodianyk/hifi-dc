"""Depth visualisation: colour maps and side-by-side comparison figures (numpy + PIL only).

Depth is shown as inverse depth on a turbo-like map (near = red, far = blue), with one display range
per case taken from the valid GT so every method in a figure shares the same colours. Errors are
absolute relative error |pred - gt| / gt on an inferno-like map, clipped at ``max_rel``.
Invalid pixels are black.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .metrics import depth_metrics

_MAPS = {
    'turbo': [(48, 18, 59), (67, 94, 213), (41, 170, 247), (26, 228, 182), (113, 254, 95),
              (199, 239, 52), (251, 185, 56), (234, 80, 13), (122, 4, 3)],
    'inferno': [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
}


def _apply_map(t, name):
    anchors = np.asarray(_MAPS[name], dtype=np.float32)
    x = np.linspace(0.0, 1.0, len(anchors))
    t = np.clip(t, 0.0, 1.0)
    return np.stack([np.interp(t, x, anchors[:, c]) for c in range(3)], axis=-1).astype(np.uint8)


def colorize(values, valid, lo, hi, cmap='turbo'):
    """Map values in [lo, hi] linearly onto ``cmap``; invalid pixels black. Returns (H, W, 3) uint8."""
    t = (np.asarray(values, np.float32) - lo) / max(hi - lo, 1e-12)
    image = _apply_map(np.nan_to_num(t), cmap)
    image[~valid] = 0
    return image


def depth_range(gt, valid, low=2.0, high=98.0):
    """Display range (near, far) in native units from percentiles of the valid GT."""
    lo, hi = np.percentile(gt[valid], [low, high])
    return float(lo), float(max(hi, lo * 1.0001))


def colorize_depth(depth, valid, near, far):
    """Inverse-depth colouring: near = red end, far = blue end."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(valid, 1.0 / np.maximum(depth, 1e-12), 0.0)
    return colorize(inv, valid & np.isfinite(inv), 1.0 / far, 1.0 / near, 'turbo')


def error_map(pred, gt, valid, max_rel=0.2):
    """Absolute relative error, 0 .. max_rel on the inferno map; invalid GT black."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(valid, np.abs(pred - gt) / np.maximum(gt, 1e-12), 0.0)
    return colorize(rel, valid, 0.0, max_rel, 'inferno')


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _panel(image, width, label, font, label_h, resample=Image.NEAREST):
    pil = Image.fromarray(image)
    height = max(1, round(pil.height * width / pil.width))
    pil = pil.resize((width, height), resample)
    canvas = Image.new('RGB', (width, height + label_h), (255, 255, 255))
    canvas.paste(pil, (0, label_h))
    ImageDraw.Draw(canvas).text((4, 2), label, fill=(0, 0, 0), font=font)
    return canvas


def _sparse_panel(rgb, sparse, near, far, width, font, label_h):
    h, w = sparse.shape
    height = max(1, round(h * width / w))
    base = (np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR), np.float32) * 0.35).astype(np.uint8)
    ys, xs = np.nonzero(sparse > 0)
    colours = colorize_depth(sparse[ys, xs][None], np.ones((1, ys.size), bool), near, far)[0]
    sy = np.clip(np.round(ys * height / h).astype(int), 0, height - 1)
    sx = np.clip(np.round(xs * width / w).astype(int), 0, width - 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            base[np.clip(sy + dy, 0, height - 1), np.clip(sx + dx, 0, width - 1)] = colours
    return _panel(base, width, f'sparse input ({ys.size} pts)', font, label_h)


def comparison_figure(rgb, sparse, gt, valid, preds, title='', panel_width=400, max_rel=0.2):
    """Rows: [RGB | sparse input | GT], [prediction per method], [relative error per method].

    ``preds`` maps a label to an (H, W) depth in the same native units as ``gt``. Returns (H, W, 3) uint8.
    """
    font, label_h = _font(14), 20
    near, far = depth_range(gt, valid)
    rows = [[_panel(rgb, panel_width, 'RGB', font, label_h, Image.BILINEAR),
             _sparse_panel(rgb, sparse, near, far, panel_width, font, label_h),
             _panel(colorize_depth(gt, valid, near, far), panel_width, 'ground truth', font, label_h)]]
    pred_row, err_row = [], []
    for name, pred in preds.items():
        pred = np.asarray(pred, np.float32)
        abs_rel = depth_metrics(pred, gt, valid)['abs_rel']
        finite = np.isfinite(pred) & (pred > 0)
        pred_row.append(_panel(colorize_depth(pred, finite, near, far), panel_width,
                               f'{name}   AbsRel {abs_rel:.4f}', font, label_h))
        err_row.append(_panel(error_map(pred, gt, valid, max_rel), panel_width,
                              f'{name}   error 0-{int(max_rel * 100)}%', font, label_h))
    rows += [pred_row, err_row]
    columns = max(len(r) for r in rows)
    cell_w, cell_h = panel_width, max(p.height for r in rows for p in r)
    gap, title_h = 6, 26
    figure = Image.new('RGB', (columns * cell_w + (columns - 1) * gap,
                               title_h + len(rows) * cell_h + (len(rows) - 1) * gap), (255, 255, 255))
    ImageDraw.Draw(figure).text((4, 4), title, fill=(0, 0, 0), font=_font(16))
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            figure.paste(panel, (c * (cell_w + gap), title_h + r * (cell_h + gap)))
    return np.asarray(figure)
