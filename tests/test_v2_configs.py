"""The shipped v2 configs load and validate, and the two frozen settings carry the measured values."""
from pathlib import Path

import pytest

from hifidc.config import load_dataclass
from hifidc.v2.config import V2Config

CONFIGS = Path(__file__).resolve().parent.parent / 'configs' / 'v2'


@pytest.mark.parametrize('path', sorted(CONFIGS.glob('*.yaml')), ids=lambda p: p.name)
def test_every_v2_config_loads(path):
    assert isinstance(load_dataclass(V2Config, path), V2Config)


def test_candidate_is_l1_l3_l7_l6():
    c = load_dataclass(V2Config, CONFIGS / 'candidate.yaml')
    assert (c.steps, c.latent_lr, c.affine_lr, c.normalize_input) == (50, 0.05, 0.005, True)
    assert (c.normal_weight, c.kappa_gate) == (0.025, False)
    assert (c.post_field, c.post_field_lambda, c.post_filter, c.filter_radius) == ('log_robust', 0.3, 'guided', 2)


def test_fast_is_25_steps_2x_lr_without_normals():
    c = load_dataclass(V2Config, CONFIGS / 'fast.yaml')
    assert (c.steps, c.latent_lr, c.affine_lr, c.normalize_input) == (25, 0.1, 0.01, True)
    assert c.normal_weight == 0.0  # measured without DSINE guidance; needs no normals
    assert (c.post_field, c.post_field_lambda, c.post_filter, c.filter_radius) == ('log_robust', 0.3, 'guided', 2)
