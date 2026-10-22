"""Low-frequency metric residual field on a coarse bilinear grid (rung L4).

The field q (gh x gw control values, corners at the image corners) is a weighted ridge least-squares
fit to residuals r_i = y_i - D_global(x_i) at the sparse points:

    min_q  sum_i w_i (B(x_i) q - r_i)^2 + lam * (q^T L q + RIDGE * |q|^2)

with B the bilinear basis and L the grid-graph Laplacian (first-order smoothness). The ridge makes
the field decay to 0 (the global fit) where there are no anchors.
"""
import torch
import torch.nn.functional as F

RIDGE = 0.3


def _basis(uv, grid):
    gh, gw = grid
    n = uv.shape[0]
    x = uv[:, 0].clamp(0, 1) * (gw - 1)
    y = uv[:, 1].clamp(0, 1) * (gh - 1)
    j0 = x.floor().clamp(max=gw - 2).long()
    i0 = y.floor().clamp(max=gh - 2).long()
    tx, ty = x - j0, y - i0
    basis = uv.new_zeros(n, gh * gw)
    rows = torch.arange(n)
    for di, dj, weight in ((0, 0, (1 - ty) * (1 - tx)), (0, 1, (1 - ty) * tx), (1, 0, ty * (1 - tx)), (1, 1, ty * tx)):
        basis[rows, (i0 + di) * gw + (j0 + dj)] += weight
    return basis


def _grid_laplacian(grid, dtype, device):
    gh, gw = grid
    lap = torch.zeros(gh * gw, gh * gw, dtype=dtype, device=device)
    for i in range(gh):
        for j in range(gw):
            for ni, nj in ((i + 1, j), (i, j + 1)):
                if ni < gh and nj < gw:
                    a, b = i * gw + j, ni * gw + nj
                    lap[a, a] += 1
                    lap[b, b] += 1
                    lap[a, b] -= 1
                    lap[b, a] -= 1
    return lap


def fit_residual_field(residual, uv, weights, grid, lam):
    """residual [N], uv [N,2] in [0,1] (x=column, y=row), weights [N] -> control values [gh, gw]."""
    dtype = torch.float64
    r, w, coords = residual.to(dtype), weights.to(dtype), uv.to(dtype)
    basis = _basis(coords, grid)
    gram = basis.T @ (w[:, None] * basis)
    reg = _grid_laplacian(grid, dtype, basis.device) + RIDGE * torch.eye(grid[0] * grid[1], dtype=dtype, device=basis.device)
    q = torch.linalg.solve(gram + lam * reg, basis.T @ (w * r))
    return q.reshape(grid).to(residual.dtype)


def evaluate_field(q, hw):
    """Control values [gh, gw] -> dense field [1,1,H,W] (bilinear, corners aligned)."""
    return F.interpolate(q[None, None], size=tuple(hw), mode='bilinear', align_corners=True)


def log_residual_field(depth, sparse, grid=(6, 8), lam=0.3, robust=True, tukey_c=4.685, iters=5, min_points=8):
    """Rung L7: post-hoc smooth multiplicative correction of the final depth toward the sparse points.

    Fits the field to log(sparse) - log(depth) (unit-free), with Tukey IRLS weights (scale = 1.4826 MAD)
    when robust, so gross outliers in the input do not bend it. Returns depth * exp(field), float32.
    With fewer than min_points valid points the depth is returned unchanged.
    """
    import numpy as np  # local: keeps the torch-only helpers above free of numpy

    depth = np.asarray(depth, np.float64)
    sparse = np.asarray(sparse, np.float64)
    mask = (sparse > 0) & np.isfinite(sparse) & np.isfinite(depth) & (depth > 0)
    if mask.sum() < min_points:
        return depth.astype(np.float32)
    rows, cols = np.nonzero(mask)
    h, w = depth.shape
    r = torch.from_numpy(np.log(sparse[mask]) - np.log(depth[mask]))
    uv = torch.from_numpy(np.stack([cols / max(w - 1, 1), rows / max(h - 1, 1)], axis=1).astype(np.float64))
    weights = torch.ones_like(r)
    q = fit_residual_field(r, uv, weights, grid, lam)
    if robust:
        basis = _basis(uv, grid)
        for _ in range(iters):
            e = r - basis @ q.reshape(-1)
            scale = (1.4826 * (e - e.median()).abs().median()).clamp(min=1e-4)
            u = (e / (tukey_c * scale)).clamp(-1, 1)
            weights = (1 - u ** 2) ** 2
            q = fit_residual_field(r, uv, weights, grid, lam)
    return (depth * np.exp(evaluate_field(q, (h, w))[0, 0].numpy())).astype(np.float32)
