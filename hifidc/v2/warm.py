"""Phase 2 warm start helpers: initial scale/shift from a least-squares fit of the prior to the sparse points."""
import torch


def affine_from_prior(prior, target, sparse_range, sparse_lower, method='ls'):
    """Fit target ~ a * prior + b at the sparse points, in the solver's parametrisation.

    Noise in the prior shrinks the LS slope towards 0 and raises the offset (regression dilution),
    and the one-step prior is noisy while the sparse depth is near exact. ``method``: 'ls' least
    squares, a = cov / var(p); 'rma' reduced major axis, a = sign(cov) std(t) / std(p) (half the
    dilution, in log terms); 'inverse' regresses the prior on the target and inverts,
    a = var(t) / cov (unbiased when the noise is in the prior only). The solver uses
    depth = scale^2 * sparse_range * prior + shift^2 * sparse_lower, so b must be >= 0: a negative b is
    refitted through zero. A degenerate fit (no spread or a <= 0) returns (1, 1), the cold-start
    initialisation. Returns 0-d float tensors (scale, shift).
    """
    if method not in ('ls', 'rma', 'inverse'):
        raise ValueError('method must be ls, rma or inverse')
    p, t = prior.double().flatten(), target.double().flatten()
    one = torch.tensor(1.0)
    if p.numel() < 2 or float(p.var()) < 1e-12:
        return one, one.clone()
    pc, tc = p - p.mean(), t - t.mean()
    a = slope(float((pc * tc).sum()), float((pc * pc).sum()), float((tc * tc).sum()), method)
    b = float(t.mean() - a * p.mean())
    if b < 0:  # the same estimator through the origin (uncentred moments)
        a, b = slope(float((p * t).sum()), float((p * p).sum()), float((t * t).sum()), method), 0.0
    if a <= 0 or sparse_range <= 0 or sparse_lower <= 0:
        return one, one.clone()
    return torch.tensor(a / sparse_range).sqrt().float(), torch.tensor(b / sparse_lower).sqrt().float()


def slope(spt, spp, stt, method):
    """Slope of t on p from second moments (cross, prior, target); <= 0 when they are anti-correlated."""
    if spt <= 0:
        return 0.0
    if method == 'ls':
        return spt / spp
    if method == 'rma':
        return (stt / spp) ** 0.5
    return stt / spt
