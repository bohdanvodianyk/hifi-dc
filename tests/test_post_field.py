"""Post-hoc robust log residual field (rung L7): smooth multiplicative correction fitted to the sparse points."""
import numpy as np
import pytest

from hifidc.v2.config import V2Config
from hifidc.v2.field import log_residual_field


def _scene(h=60, w=80, n=600, seed=0):
    rng = np.random.default_rng(seed)
    gt = 2.0 + rng.uniform(0, 0.5) + np.zeros((h, w))
    gt[:, w // 2:] += 1.5                                      # a depth step
    ramp = np.linspace(-1, 1, w)[None, :] * np.ones((h, 1))
    pred = gt * np.exp(0.15 * ramp)                            # smooth multiplicative bias
    sparse = np.zeros((h, w))
    idx = rng.choice(h * w, n, replace=False)
    sparse.flat[idx] = gt.flat[idx]
    return gt, pred, sparse, idx


def _abs_rel(x, gt):
    return float(np.mean(np.abs(x - gt) / gt))


def test_smooth_bias_is_removed():
    gt, pred, sparse, _ = _scene()
    out = log_residual_field(pred, sparse, lam=0.3)
    assert _abs_rel(pred, gt) > 0.05
    assert _abs_rel(out, gt) < 0.25 * _abs_rel(pred, gt)


def test_robust_fit_ignores_gross_outliers():
    gt, pred, sparse, idx = _scene()
    bad = idx[:30]                                             # 5 % of points 3x too far
    sparse.flat[bad] *= 3.0
    plain = log_residual_field(pred, sparse, lam=0.3, robust=False)
    robust = log_residual_field(pred, sparse, lam=0.3, robust=True)
    assert _abs_rel(robust, gt) < 0.5 * _abs_rel(plain, gt)
    assert _abs_rel(robust, gt) < 0.25 * _abs_rel(pred, gt)


def test_field_is_scale_equivariant():
    _, pred, sparse, _ = _scene()
    a = log_residual_field(pred, sparse)
    b = log_residual_field(1000 * pred, 1000 * sparse)
    np.testing.assert_allclose(b, 1000 * a, rtol=1e-5)


def test_too_few_points_returns_input_unchanged():
    _, pred, sparse, idx = _scene()
    few = np.zeros_like(sparse)
    few.flat[idx[:2]] = sparse.flat[idx[:2]]
    np.testing.assert_array_equal(log_residual_field(pred, few), pred.astype(np.float32))


def test_config_post_field_validated():
    V2Config(post_field='log_robust').validate()
    with pytest.raises(ValueError):
        V2Config(post_field='bogus').validate()
    with pytest.raises(ValueError):
        V2Config(post_field_lambda=0.0).validate()
