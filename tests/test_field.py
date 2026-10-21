"""Low-frequency metric residual field (rung L4)."""
import torch

from hifidc.v2.field import evaluate_field, fit_residual_field


def scattered(n=500, seed=0):
    return torch.rand(n, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_zero_residual_zero_field():
    uv = scattered()
    q = fit_residual_field(torch.zeros(len(uv), dtype=torch.float64), uv, torch.ones(len(uv), dtype=torch.float64),
                           (6, 8), 1.0)
    assert torch.allclose(q, torch.zeros(6, 8, dtype=torch.float64), atol=1e-12)


def test_two_regime_offsets_recovered():
    uv = scattered()
    residual = torch.where(uv[:, 0] < 0.5, 0.5, -0.5).to(torch.float64)
    q = fit_residual_field(residual, uv, torch.ones(len(uv), dtype=torch.float64), (6, 8), 0.01)
    field = evaluate_field(q, (60, 80))[0, 0]
    assert float((field[:, :25] - 0.5).abs().max()) < 0.1   # left, away from the seam
    assert float((field[:, -25:] + 0.5).abs().max()) < 0.1  # right
    global_only = (residual - residual.mean()).abs().min()
    assert float(global_only) >= 0.45  # one global offset cannot fit both halves


def test_empty_cells_shrink_to_zero():
    uv = scattered()
    uv[:, 0] *= 0.4  # points only in the left 40 %
    q = fit_residual_field(torch.full((len(uv),), 0.3, dtype=torch.float64), uv,
                           torch.ones(len(uv), dtype=torch.float64), (6, 8), 1.0)
    assert torch.isfinite(q).all()
    assert float(q[:, -1].abs().max()) < 0.05


def test_lambda_monotone():
    uv = scattered()
    residual = torch.sin(6 * uv[:, 0]) * 0.2
    weights = torch.ones(len(uv), dtype=torch.float64)
    norms = [float(fit_residual_field(residual, uv, weights, (6, 8), lam).norm()) for lam in (0.1, 1.0, 10.0)]
    assert norms[0] > norms[1] > norms[2]


def test_weights_respected():
    uv = scattered()
    residual = torch.full((len(uv),), 0.4, dtype=torch.float64)
    weights = torch.ones(len(uv), dtype=torch.float64)
    residual[:50], weights[:50] = -5.0, 0.0  # zero-weight outliers must not matter
    q = fit_residual_field(residual, uv, weights, (6, 8), 0.01)
    assert float((evaluate_field(q, (30, 40)) - 0.4).abs().max()) < 0.05
