"""Robust (Huber IRLS) weighting of the sparse anchor loss."""
import torch


def robust_weights(residuals, k=1.345):
    """Detached Huber weights min(1, k*sigma / |r - median r|), sigma = 1.4826 * MAD; sigma = 0 -> all ones."""
    with torch.no_grad():
        centred = residuals.detach() - residuals.detach().median()
        sigma = 1.4826 * centred.abs().median()
        if not torch.isfinite(sigma) or sigma <= 0:
            return torch.ones_like(centred)
        return torch.clamp(k * sigma / centred.abs().clamp_min(torch.finfo(centred.dtype).tiny), max=1.0)


def robust_l1l2(predicted, target, k=1.345):
    """Weighted L1 + L2 at the sparse points, normalised by the weight sum (unit weights = L0 loss)."""
    residual = predicted - target
    weights = robust_weights(residual, k)
    return (weights * (residual.abs() + residual.square())).sum() / weights.sum()
