"""Rung L6: RGB-guided edge-aware filtering of the final depth (He et al. guided filter on log-depth).

The solver leaves pixel-level speckle and small dents at the sparse points. A guided filter with the
grey image as guide smooths log-depth where the image is smooth and keeps steps where the image
has an edge. Working in log-depth makes it unit-free (filter(c * d) = c * filter(d)). The radius
is given in pixels of a 384-px grid and scaled to the output size.
"""
import numpy as np
import torch
import torch.nn.functional as F

GRID = 384


def _box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='replicate'), 2 * r + 1, stride=1)


def guided_log_depth(depth, rgb, radius=2, eps=1e-3):
    """depth (H, W) > 0 (non-positive values are floored), rgb (H, W, 3) uint8 -> filtered depth (H, W) float32."""
    depth = np.asarray(depth, np.float64)
    positive = depth[np.isfinite(depth) & (depth > 0)]
    floor = np.percentile(positive, 1) * 0.5 if positive.size else 1e-6
    log_depth = np.log(np.maximum(np.nan_to_num(depth, nan=floor), floor))
    p = torch.from_numpy(log_depth)[None, None]
    g = torch.from_numpy(np.asarray(rgb, np.float64).mean(-1) / 255)[None, None]
    r = max(1, round(radius * max(depth.shape) / GRID))
    mean_g, mean_p = _box(g, r), _box(p, r)
    a = (_box(g * p, r) - mean_g * mean_p) / (_box(g * g, r) - mean_g * mean_g + eps)
    b = mean_p - a * mean_g
    return np.exp((_box(a, r) * g + _box(b, r))[0, 0].numpy()).astype(np.float32)
