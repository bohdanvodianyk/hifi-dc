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


def test_warm_start_can_keep_the_cold_affine_init(tiny_v2_pipeline, monkeypatch):
    import hifidc.v2.pipeline as pipeline_module

    def fail(*args, **kwargs):
        raise AssertionError("affine_from_prior must not run when warm_affine is none")

    monkeypatch.setattr(pipeline_module, "affine_from_prior", fail)
    result = run(tiny_v2_pipeline, config=V2Config(steps=6, warm_start=0.5, warm_affine="none"))
    assert len(result["trace"]) == 3
    assert np.isfinite(result["depth"]).all()


def test_noise_in_the_prior_dilutes_ls_partly_rma_and_not_the_inverse_fit():
    gen = torch.Generator().manual_seed(0)
    clean = torch.rand(20000, generator=gen)
    target = 3.0 * clean + 1.0                                # exact sparse depth
    prior = clean + 0.2 * torch.randn(20000, generator=gen)  # noisy prior; attenuation var(c)/var(p) ~ 0.68
    slope = {m: float(affine_from_prior(prior, target, 1.0, 1.0, method=m)[0] ** 2) for m in ("ls", "rma", "inverse")}
    assert slope["ls"] == pytest.approx(3.0 * 0.676, rel=0.03)
    assert slope["rma"] == pytest.approx(3.0 * 0.676 ** 0.5, rel=0.03)
    assert slope["inverse"] == pytest.approx(3.0, rel=0.03)
    rma_scale, rma_shift = affine_from_prior(prior, target, 1.0, 1.0, method="rma")
    expected = float(target.std() / prior.std())
    assert float(rma_scale ** 2) == pytest.approx(expected, rel=1e-4)
    assert float(rma_shift ** 2) == pytest.approx(float(target.mean() - expected * prior.mean()), rel=1e-4)


@pytest.mark.parametrize("method", ["rma", "inverse"])
def test_every_fit_recovers_an_exact_affine_map(method):
    prior = torch.linspace(0.1, 0.9, 50)
    scale, shift = affine_from_prior(prior, 2.0 * prior + 0.5, sparse_range=1.6, sparse_lower=0.7, method=method)
    assert float(scale ** 2 * 1.6) == pytest.approx(2.0, rel=1e-5)
    assert float(shift ** 2 * 0.7) == pytest.approx(0.5, rel=1e-5)


@pytest.mark.parametrize("method", ["ls", "rma", "inverse"])
def test_negatively_correlated_prior_falls_back_to_cold_init(method):
    prior = torch.linspace(0.1, 0.9, 50)
    scale, shift = affine_from_prior(prior, 3.0 - prior, 1.0, 1.0, method=method)
    assert (float(scale), float(shift)) == (1.0, 1.0)


def test_warm_affine_is_validated():
    with pytest.raises(ValueError, match="warm_affine"):
        V2Config(warm_affine="bogus").validate()
