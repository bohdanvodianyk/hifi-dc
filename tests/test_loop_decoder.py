"""Phase 3 speed: decode the in-loop guidance prediction with a tiny VAE decoder (TAESD).

The final depth is always decoded with the full VAE. Default loop_decoder=vae is Marigold-DC.
"""
import numpy as np
import pytest
from diffusers import AutoencoderTiny

from test_v2_pipeline import run
from hifidc.v2.config import V2Config


def tiny_taesd():
    # two blocks = one 2x upsample, matching the 2x tiny test VAE
    return AutoencoderTiny(encoder_block_out_channels=(8, 8), decoder_block_out_channels=(8, 8),
                           num_encoder_blocks=(1, 1), num_decoder_blocks=(1, 1), latent_channels=4).eval()


def count(module):
    calls = []
    module.register_forward_hook(lambda *a: calls.append(1))
    return calls


@pytest.mark.parametrize("loop_decoder, tiny_calls, full_calls", [("vae", 0, 4), ("taesd", 3, 1)])
def test_loop_decoder_routes_the_in_loop_decodes(tiny_v2_pipeline, loop_decoder, tiny_calls, full_calls):
    tiny_v2_pipeline.tiny_vae = tiny_taesd()
    tiny = count(tiny_v2_pipeline.tiny_vae.decoder)
    full = count(tiny_v2_pipeline.vae.decoder)
    result = run(tiny_v2_pipeline, config=V2Config(steps=3, loop_decoder=loop_decoder))
    assert (len(tiny), len(full)) == (tiny_calls, full_calls)    # 3 steps + 1 final decode
    assert all(step["latent_grad_norm"] > 0 for step in result["trace"])
    assert np.isfinite(result["depth"]).all()


def test_full_vae_takes_over_after_loop_decoder_steps(tiny_v2_pipeline):
    tiny_v2_pipeline.tiny_vae = tiny_taesd()
    order = []
    tiny_v2_pipeline.tiny_vae.decoder.register_forward_hook(lambda *a: order.append("tiny"))
    tiny_v2_pipeline.vae.decoder.register_forward_hook(lambda *a: order.append("full"))
    run(tiny_v2_pipeline, config=V2Config(steps=4, loop_decoder="taesd", loop_decoder_steps=0.5))
    assert order == ["tiny", "tiny", "full", "full", "full"]     # 2 TAESD steps, 2 full steps, final decode


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_loop_decoder_steps_is_validated(value):
    with pytest.raises(ValueError, match="loop_decoder_steps"):
        V2Config(loop_decoder_steps=value).validate()


def test_taesd_loop_decoder_needs_a_tiny_vae(tiny_v2_pipeline):
    tiny_v2_pipeline.tiny_vae = None
    with pytest.raises(ValueError, match="tiny_vae"):
        run(tiny_v2_pipeline, config=V2Config(steps=3, loop_decoder="taesd"))


def test_loop_decoder_is_validated():
    with pytest.raises(ValueError, match="loop_decoder"):
        V2Config(loop_decoder="other").validate()
