import numpy as np
import pytest

from hifidc.v2.config import V2Config
from hifidc.v2.postfilter import guided_log_depth


def _flat_rgb(h=96, w=128, value=128):
    return np.full((h, w, 3), value, np.uint8)


def test_filter_is_scale_equivariant():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1, 5, (96, 128)).astype(np.float32)
    rgb = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    a = guided_log_depth(depth, rgb)
    b = guided_log_depth(depth * 1000, rgb)
    np.testing.assert_allclose(b, a * 1000, rtol=1e-5)


def test_filter_removes_pixel_noise_on_a_flat_surface():
    rng = np.random.default_rng(1)
    noisy = (3.0 * np.exp(rng.normal(0, 0.05, (96, 128)))).astype(np.float32)
    out = guided_log_depth(noisy, _flat_rgb())
    assert np.std(np.log(out)) < 0.3 * np.std(np.log(noisy))
    assert abs(np.median(out) - 3.0) < 0.05


def test_filter_keeps_a_depth_edge_that_the_image_shows():
    depth = np.full((96, 128), 2.0, np.float32)
    depth[:, 64:] = 6.0
    rgb = _flat_rgb()
    rgb[:, 64:] = 230
    rgb[:, :64] = 20
    out = guided_log_depth(depth, rgb)
    assert abs(out[:, 60].mean() - 2.0) < 0.1 and abs(out[:, 68].mean() - 6.0) < 0.3


def test_nonpositive_depth_does_not_produce_nan():
    depth = np.full((32, 48), 2.0, np.float32)
    depth[0, 0] = 0.0
    depth[1, 1] = -1.0
    assert np.isfinite(guided_log_depth(depth, _flat_rgb(32, 48))).all()


def test_config_post_filter_validated():
    assert V2Config().post_filter == 'none'
    V2Config(post_filter='guided').validate()
    with pytest.raises(ValueError):
        V2Config(post_filter='median').validate()
    with pytest.raises(ValueError):
        V2Config(post_filter='guided', filter_radius=0).validate()
