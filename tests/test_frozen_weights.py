"""Speed: the guidance needs the gradient w.r.t. the latent only, never w.r.t. the UNet / VAE weights.

Freezing the weights skips the weight-gradient half of every backward (16.8 -> 11.3 s per case at
384 px, peak VRAM 5.9 -> 4.0 GB). Same maths; input-only backward kernels round differently (~1 ulp),
so it is off by default (bit-exact Marigold-DC) and switched on by run_benchmark --fast.
"""
import numpy as np

from test_loop_decoder import tiny_taesd
from test_v2_pipeline import run
from hifidc.v2.config import V2Config


def weights(pipe):
    modules = [pipe.unet, pipe.vae] + ([pipe.tiny_vae] if pipe.tiny_vae is not None else [])
    return [p for m in modules for p in m.parameters()]


def test_no_weight_gradients_are_computed(tiny_v2_pipeline):
    tiny_v2_pipeline.freeze_weights = True
    tiny_v2_pipeline.tiny_vae = tiny_taesd()
    result = run(tiny_v2_pipeline, config=V2Config(steps=3, loop_decoder='taesd', loop_decoder_steps=0.5))
    assert all(p.grad is None for p in weights(tiny_v2_pipeline))
    assert all(step['latent_grad_norm'] > 0 for step in result['trace'])


def test_default_keeps_weight_gradients_like_marigold_dc(tiny_v2_pipeline):
    run(tiny_v2_pipeline, config=V2Config(steps=3))
    assert any(p.grad is not None for p in weights(tiny_v2_pipeline))


def test_freezing_changes_the_result_only_by_rounding(tiny_v2_pipeline):
    unfrozen = run(tiny_v2_pipeline, config=V2Config(steps=4))
    tiny_v2_pipeline.freeze_weights = True
    frozen = run(tiny_v2_pipeline, config=V2Config(steps=4))
    np.testing.assert_allclose(frozen['depth'], unfrozen['depth'], rtol=1e-5, atol=0)
    np.testing.assert_allclose([s['latent_grad_norm'] for s in frozen['trace']],
                               [s['latent_grad_norm'] for s in unfrozen['trace']], rtol=1e-5)
