import numpy as np

from hifidc.v2.prior_fit import fit_prior_to_sparse


def _grid(h=60, w=80):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return yy / (h - 1), xx / (w - 1)


def _points(shape, n=400, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, bool)
    mask.flat[rng.choice(mask.size, n, replace=False)] = True
    return mask


def test_global_affine_is_recovered_exactly():
    y, x = _grid()
    rel = 0.2 + 0.6 * x + 0.1 * y
    truth = 3.0 * rel + 2.0
    mask = _points(rel.shape)
    sparse = np.where(mask, truth, 0).astype(np.float32)
    out = fit_prior_to_sparse(rel, sparse, field=False)
    np.testing.assert_allclose(out, truth, rtol=1e-4)


def test_field_corrects_a_smooth_regional_error_that_affine_cannot():
    y, x = _grid()
    rel = (0.2 + 0.6 * x + 0.1 * y).astype(np.float32)
    truth = 3.0 * rel + 2.0 + 0.8 * np.sin(np.pi * x) * y      # smooth, non-affine deviation
    mask = _points(rel.shape)
    sparse = np.where(mask, truth, 0).astype(np.float32)
    err_affine = np.abs(fit_prior_to_sparse(rel, sparse, field=False) - truth).mean()
    err_field = np.abs(fit_prior_to_sparse(rel, sparse, field=True) - truth).mean()
    assert err_field < 0.5 * err_affine


def test_edges_of_the_prior_are_kept():
    y, x = _grid()
    rel = np.where(x > 0.5, 0.8, 0.3).astype(np.float32)          # a sharp step edge
    truth = 2.0 * rel + 1.0
    mask = _points(rel.shape)
    sparse = np.where(mask, truth, 0).astype(np.float32)
    out = fit_prior_to_sparse(rel, sparse, field=True)
    jump = out[:, 41].mean() - out[:, 38].mean()
    assert abs(jump - 1.0) < 0.05                                  # step height 2.0 * 0.5 preserved


def test_too_few_points_raises():
    rel = np.ones((10, 10), np.float32)
    sparse = np.zeros_like(rel)
    sparse[0, 0] = 1.0
    try:
        fit_prior_to_sparse(rel, sparse)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
