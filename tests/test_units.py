"""Median normalisation of sparse depth (unit-free inputs)."""
import numpy as np
import pytest

from hifidc.v2.units import normalize_sparse


def sparse_with(values, size=64):
    sparse = np.zeros((size, size), np.float32)
    sparse.flat[:len(values)] = values
    return sparse


def test_median_becomes_one():
    normalized, scale = normalize_sparse(sparse_with(np.linspace(2., 10., 101)))
    assert scale == pytest.approx(6.0)
    assert np.median(normalized[normalized > 0]) == pytest.approx(1.0)


def test_scale_roundtrip():
    sparse = sparse_with(np.random.default_rng(0).uniform(500, 5000, 300))
    normalized, scale = normalize_sparse(sparse)
    mask = sparse > 0
    np.testing.assert_allclose(normalized[mask] * scale, sparse[mask], rtol=1e-6)
    assert np.all(normalized[~mask] == 0)


def test_degenerate_inputs():
    normalized, scale = normalize_sparse(sparse_with([3.0] * 10))
    assert scale == 3.0 and np.all(normalized[normalized > 0] == 1.0)
    values = np.ones(1000)
    values[0] = 1e6
    assert normalize_sparse(sparse_with(values))[1] == 1.0
    noisy = sparse_with([np.nan, -1., 0., 2., 4., 6., 8.])
    normalized, scale = normalize_sparse(noisy)
    assert scale == 5.0 and np.isfinite(normalized).all() and int((normalized > 0).sum()) == 4
    with pytest.raises(ValueError):
        normalize_sparse(sparse_with([1., 2., 3.]))
