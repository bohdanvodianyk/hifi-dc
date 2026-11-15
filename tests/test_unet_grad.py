"""Phase 3 speed: treat the UNet output as constant within a step after a fraction of the schedule.

Marigold is v-prediction, x0 = sqrt(abar) x_t - sqrt(1 - abar) v: early (abar ~ 0) the latent reaches
x0 almost only through the UNet, late the direct path dominates. ``unet_grad_steps`` is the fraction of
steps, from the start, that backpropagate through the UNet; 1.0 (default) is Marigold-DC.
"""
import numpy as np
import pytest

from test_v2_pipeline import run
from hifidc.v2.config import V2Config


@pytest.mark.parametrize("fraction, expected", [(1.0, [True] * 4), (0.5, [True, True, False, False]),
                                                (0.0, [False] * 4)])
def test_unet_backward_only_in_the_first_fraction_of_steps(tiny_v2_pipeline, fraction, expected):
    seen = []
    handle = tiny_v2_pipeline.unet.register_forward_hook(lambda m, i, o: seen.append(o[0].requires_grad))
    try:
        result = run(tiny_v2_pipeline, config=V2Config(steps=4, unet_grad_steps=fraction))
    finally:
        handle.remove()
    assert seen == expected
    assert all(step["latent_grad_norm"] > 0 for step in result["trace"])   # the decoder path still guides
    assert np.isfinite(result["depth"]).all()


def test_unet_grad_off_changes_the_result(tiny_v2_pipeline):
    on = run(tiny_v2_pipeline, config=V2Config(steps=3))["depth"]
    off = run(tiny_v2_pipeline, config=V2Config(steps=3, unet_grad_steps=0.0))["depth"]
    assert not np.array_equal(on, off)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_unet_grad_steps_is_validated(value):
    with pytest.raises(ValueError, match="unet_grad_steps"):
        V2Config(unet_grad_steps=value).validate()
