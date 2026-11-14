"""Phase 2 warm start helpers: initial scale/shift from a least-squares fit of the prior to the sparse points."""
import torch


def affine_from_prior(prior, target, sparse_range, sparse_lower):
    """LS fit target ~ a * prior + b at the sparse points, in the solver's parametrisation.

    The solver uses depth = scale^2 * sparse_range * prior + shift^2 * sparse_lower, so b must be >= 0:
    a negative b is refitted through zero. A degenerate fit (no spread or a <= 0) returns (1, 1),
    the cold-start initialisation. Returns 0-d float tensors (scale, shift).
    """
    p, t = prior.double().flatten(), target.double().flatten()
    one = torch.tensor(1.0)
    if p.numel() < 2 or float(p.var()) < 1e-12:
        return one, one.clone()
    a = float(((p - p.mean()) * (t - t.mean())).sum() / ((p - p.mean()) ** 2).sum())
    b = float(t.mean() - a * p.mean())
    if b < 0:
        a, b = float((p * t).sum() / (p * p).sum()), 0.0
    if a <= 0 or sparse_range <= 0 or sparse_lower <= 0:
        return one, one.clone()
    return torch.tensor(a / sparse_range).sqrt().float(), torch.tensor(b / sparse_lower).sqrt().float()
