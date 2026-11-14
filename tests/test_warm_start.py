"""Phase 2 warm start: start the solver from a one-step Marigold prediction instead of pure noise.

The prior latent is re-noised to an intermediate timestep (SDEdit) and the solver runs only the
remaining steps; scale and shift start from a least-squares fit of the prior to the sparse points.
"""
import numpy as np
import pytest
import torch

from test_v2_pipeline import run
from hifidc.v2.config import V2Config
from hifidc.v2.warm import affine_from_prior


def test_affine_from_prior_recovers_an_exact_affine_map():
    prior = torch.linspace(0.1, 0.9, 50)
    target = 2.0 * prior + 0.5
    scale, shift = affine_from_prior(prior, target, sparse_range=1.6, sparse_lower=0.7)
    # the solver's parametrisation: depth = scale^2 * range * prior + shift^2 * lower
    assert float(scale ** 2 * 1.6) == pytest.approx(2.0, rel=1e-5)
    assert float(shift ** 2 * 0.7) == pytest.approx(0.5, rel=1e-5)


def test_affine_from_prior_negative_shift_refits_through_zero():
    prior = torch.linspace(0.1, 0.9, 50)
    target = 2.0 * prior - 0.1                               # shift < 0 cannot be represented (shift^2 >= 0)
    scale, shift = affine_from_prior(prior, target, sparse_range=1.0, sparse_lower=0.1)
    assert float(shift) == 0.0
    expected = float((prior * target).sum() / (prior * prior).sum())
    assert float(scale ** 2) == pytest.approx(expected, rel=1e-5)


def test_affine_from_prior_degenerate_falls_back_to_one():
    prior = torch.full((10,), 0.5)                           # no spread: slope undefined
    scale, shift = affine_from_prior(prior, torch.linspace(1, 2, 10), sparse_range=1.0, sparse_lower=1.0)
    assert (float(scale), float(shift)) == (1.0, 1.0)


def test_warm_start_runs_only_the_remaining_steps(tiny_v2_pipeline):
    calls = []
    tiny_v2_pipeline.unet.register_forward_hook(lambda *a: calls.append(1))
    result = run(tiny_v2_pipeline, config=V2Config(steps=6, warm_start=0.5))
    assert len(result['trace']) == 3                         # steps 3..5 of a 6-step schedule
    assert len(calls) == 1 + 3                               # one prior call + the solver steps
    assert np.isfinite(result['depth']).all()


def test_warm_start_is_deterministic_and_changes_the_output(tiny_v2_pipeline):
    first = run(tiny_v2_pipeline, config=V2Config(steps=6, warm_start=0.5))['depth']
    second = run(tiny_v2_pipeline, config=V2Config(steps=6, warm_start=0.5))['depth']
    cold = run(tiny_v2_pipeline, config=V2Config(steps=6))['depth']
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, cold)


@pytest.mark.parametrize('value', [-0.1, 1.0, float('nan')])
def test_warm_start_is_validated(value):
    with pytest.raises(ValueError, match='warm_start'):
        V2Config(warm_start=value).validate()
