"""Closed-form metric fit of a relative (affine-invariant) depth prior to sparse points.

No latent optimisation: the prior keeps its own structure (edges, flat surfaces), and only a
global least-squares affine map, plus optionally the L4 low-frequency residual field, is fitted to
the sparse points. This is what the inert HiFi-DC v1 actually produced (prior + global affine); the
field variant adds smooth regional correction that a single affine map cannot express.
"""
import numpy as np
import torch

from .field import evaluate_field, fit_residual_field


def fit_prior_to_sparse(rel, sparse, field=False, grid=(6, 8), lam=1.0):
    """rel (H, W) relative depth, sparse (H, W) metric depth with 0 = no point -> metric (H, W) float32."""
    rel = np.asarray(rel, np.float64)
    sparse = np.asarray(sparse, np.float64)
    mask = (sparse > 0) & np.isfinite(sparse) & np.isfinite(rel)
    if mask.sum() < 3:
        raise ValueError(f'need at least 3 sparse points, got {int(mask.sum())}')
    a = np.stack([rel[mask], np.ones(int(mask.sum()))], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(a, sparse[mask], rcond=None)
    out = scale * rel + shift
    if field:
        rows, cols = np.nonzero(mask)
        h, w = rel.shape
        residual = torch.from_numpy(sparse[mask] - out[mask])
        uv = torch.from_numpy(np.stack([cols / max(w - 1, 1), rows / max(h - 1, 1)], axis=1))
        q = fit_residual_field(residual, uv, torch.ones_like(residual), grid, lam)
        out = out + evaluate_field(q, (h, w))[0, 0].numpy()
    return out.astype(np.float32)
