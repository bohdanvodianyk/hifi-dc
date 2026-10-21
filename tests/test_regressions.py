"""Regression tests that fail on the two historical v1 defects (inert guidance, w^2 WLS)."""
import pytest
import torch

from conftest import fixture, mock_pipe
from hifidc.config import Config
from hifidc.core import affine_fit, complete
from hifidc.pipeline import DiffusionBackend


class InertBackend(DiffusionBackend):
    """The v1 pattern: clean-sample prediction computed under no_grad."""

    def predict_clean(self, latent, timestep, encoded):
        with torch.no_grad():
            return super().predict_clean(latent, timestep, encoded)


def test_inert_guidance_is_rejected():
    _, _, rgb, sparse, normal = fixture()
    backend = InertBackend(mock_pipe(), torch.Generator().manual_seed(2024))
    with pytest.raises(RuntimeError, match='latent gradient'):
        complete(backend, rgb, sparse, normal, Config(steps=6, bilateral=False))


def test_live_guidance_moves_latent():
    _, backend, rgb, sparse, normal = fixture()
    live = complete(backend, rgb, sparse, normal, Config(steps=6, bilateral=False))
    assert all(r['latent_gradient_norm'] > 0 for r in live['trace'])
    assert all(r['adam_max_update'] > 0 for r in live['trace'])
    _, backend, *_ = fixture()
    frozen = complete(backend, rgb, sparse, normal, Config(steps=6, bilateral=False, learning_rate=1e-12))
    assert float((live['depth'] - frozen['depth']).abs().max()) > 1e-6


def _analytic_wls(z, y, w):
    X = torch.stack((z, torch.ones_like(z)), 1).double()
    W = torch.diag(w.double())
    return torch.linalg.solve(X.T @ W @ X, X.T @ W @ y.double())


def test_squared_weight_wls_is_detected():
    g = torch.Generator().manual_seed(3)
    z = torch.rand(40, generator=g)
    y = 1.7 * z + 0.4 + 0.3 * torch.randn(40, generator=g)
    w = torch.tensor([1., .5, .25, .1] * 10)
    reference = _analytic_wls(z, y, w)
    shape = (1, 1, 5, 8)
    args = (z.reshape(shape), y.reshape(shape), torch.ones(shape, dtype=torch.bool))
    raw = Config(ema_beta=1.0, ridge=0.0)
    start = (torch.tensor(1.), torch.tensor(0.))
    correct = torch.stack(affine_fit(*args, w.reshape(shape), start, raw)).double()
    squared = torch.stack(affine_fit(*args, w.reshape(shape) ** 2, start, raw)).double()  # v1 bug: weights applied twice
    torch.testing.assert_close(correct, reference, rtol=0, atol=1e-6)
    assert float((squared - reference).abs().max()) > 1e-3
