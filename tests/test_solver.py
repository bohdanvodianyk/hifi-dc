"""research-plan.md test-first solver set: affine recovery, outlier weighting, local scale, guidance gradient."""
import pytest
import torch

from conftest import fixture
from hifidc.config import Config
from hifidc.core import affine_fit, confidence_weights, data_terms

RAW_FIT = Config(ema_beta=1.0, ridge=0.0)  # EMA off: the fit itself is returned
START = (torch.tensor(1.), torch.tensor(0.))


def synthetic_affine(a=2.3, b=-0.7, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(1, 1, size, size, generator=g)
    y = a * z + b
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask.view(-1)[::3] = True
    return z, y, mask


def test_global_affine_recovery():
    z, y, mask = synthetic_affine()
    a, b = affine_fit(z, y, mask, mask.float(), START, RAW_FIT)
    assert abs(float(a) - 2.3) < 1e-6
    assert abs(float(b) + 0.7) < 1e-6


def test_weighted_outlier_rejection():
    z, y, mask = synthetic_affine()
    index = int(mask.view(-1).nonzero()[5])
    y_bad = y.clone()
    y_bad.view(-1)[index] += 50.
    weights = mask.float()
    weights.view(-1)[index] = 1e-4
    a, b = affine_fit(z, y_bad, mask, weights, START, RAW_FIT)
    assert abs(float(a) - 2.3) < 0.01 and abs(float(b) + 0.7) < 0.01
    # Control: at full weight the same outlier visibly corrupts the fit, so the weight is what protects it.
    a_full, b_full = affine_fit(z, y_bad, mask, mask.float(), START, RAW_FIT)
    assert abs(float(a_full) - 2.3) + abs(float(b_full) + 0.7) > 0.1


@pytest.mark.xfail(strict=True, reason='global affine cannot represent two regimes; target of P1 local metric graph')
def test_local_scale_recovery():
    z, _, mask = synthetic_affine()
    y = torch.empty_like(z)
    half = z.shape[-1] // 2
    y[..., :half] = 2.0 * z[..., :half] + 0.1
    y[..., half:] = 0.5 * z[..., half:] + 1.0
    a, b = affine_fit(z, y, mask, mask.float(), START, RAW_FIT)
    error = (a * z + b - y).abs()
    assert float(error[..., :half].max()) < 1e-3
    assert float(error[..., half:].max()) < 1e-3


def _guidance_setup():
    _, backend, rgb, sparse, _ = fixture()
    backend.set_timesteps(6)
    encoded = backend.encode_rgb(rgb)
    latent = backend.initial_latent(encoded)
    return backend, encoded, latent, sparse


def _sparse_loss(backend, latent, timestep, encoded, sparse):
    mask = sparse > 0
    weights = confidence_weights(mask, Config())
    decoded = backend.decode(backend.predict_clean(latent, timestep, encoded))
    quadratic, l1 = data_terms(decoded, sparse, mask, weights, Config())
    return quadratic + l1


def test_guidance_gradient():
    backend, encoded, latent, sparse = _guidance_setup()
    t0 = backend.timesteps[0]
    perturbed = sparse.clone()
    first = perturbed.view(-1).nonzero()[0]
    perturbed.view(-1)[first] *= 1.5  # perturb one sparse observation

    x = latent.clone().requires_grad_(True)
    loss = _sparse_loss(backend, x, t0, encoded, perturbed)
    grad = torch.autograd.grad(loss, x)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    with torch.no_grad():
        stepped = _sparse_loss(backend, latent - 1e-3 * grad / grad.norm(), t0, encoded, perturbed)
    assert float(stepped) < float(loss)

    x_clean = latent.clone().requires_grad_(True)
    grad_clean = torch.autograd.grad(_sparse_loss(backend, x_clean, t0, encoded, sparse), x_clean)[0]
    assert not torch.allclose(grad, grad_clean)
