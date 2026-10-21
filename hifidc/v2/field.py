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
