"""HiFi-DC v2 pipeline on tiny random modules (CPU)."""
import numpy as np
from PIL import Image
import pytest

from conftest import sparse_grid
from hifidc.config import load_dataclass
from hifidc.v2.config import V2Config


def run(pipe, config=None, sparse=None, size=(32, 48), **kwargs):
    h, w = size
    image = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8))
    sparse = sparse_grid(h, w) if sparse is None else sparse
    return pipe(image=image, sparse_depth=sparse, config=config or V2Config(steps=3), seed=2024,
                processing_resolution=48, return_details=True, **kwargs)


def test_v2_returns_original_grid_and_counts_steps(tiny_v2_pipeline):
    result = run(tiny_v2_pipeline)
    assert result['depth'].shape == (32, 48)
    assert np.isfinite(result['depth']).all()
    assert len(result['trace']) == 3


def test_v2_latent_gradient_live(tiny_v2_pipeline):
    result = run(tiny_v2_pipeline)
    assert all(t['latent_grad_norm'] > 0 for t in result['trace'])


def test_v2_scale_shift_positive(tiny_v2_pipeline):
    result = run(tiny_v2_pipeline, config=V2Config(steps=4))
    assert all(t['scale'] >= 0 and t['shift'] >= 0 for t in result['trace'])


def test_v2_is_deterministic(tiny_v2_pipeline):
    first = run(tiny_v2_pipeline)['depth']
    second = run(tiny_v2_pipeline)['depth']
    np.testing.assert_array_equal(first, second)


def test_v2_rejects_mismatched_sparse(tiny_v2_pipeline):
    with pytest.raises(ValueError):
        run(tiny_v2_pipeline, sparse=np.zeros((10, 10), np.float32))


def test_load_dataclass_typed(tmp_path):
    path = tmp_path / 'v2.yaml'
    path.write_text('steps: 20\nlatent_lr: 5e-2\n')
    config = load_dataclass(V2Config, path, {'affine_lr': '1e-3'})
    assert config.steps == 20 and config.latent_lr == 0.05 and config.affine_lr == 1e-3
    with pytest.raises(ValueError, match='bogus'):
        load_dataclass(V2Config, path, {'bogus': 1})


def test_l0_default_unchanged(tiny_v2_pipeline):
    """Default V2Config must reproduce the recorded L0 output (every later rung defaults off)."""
    import json
    from pathlib import Path
    from test_golden import golden_tolerance, platform_fingerprint
    data = Path(__file__).parent / 'data'
    rtol, atol = golden_tolerance(json.loads((data / 'golden_platform.json').read_text()), platform_fingerprint())
    depth = run(tiny_v2_pipeline, config=V2Config(steps=4))['depth']
    np.testing.assert_allclose(depth, np.load(data / 'v2_l0_tiny.npy'), rtol=rtol, atol=atol)


def test_unit_invariance_with_normalisation(tiny_v2_pipeline):
    sparse = sparse_grid(32, 48)
    config = V2Config(steps=3, normalize_input=True)
    metres = run(tiny_v2_pipeline, config=config, sparse=sparse)['depth']
    millimetres = run(tiny_v2_pipeline, config=config, sparse=sparse * 1000)['depth']
    np.testing.assert_allclose(millimetres, metres * 1000, rtol=1e-4)


def test_without_normalisation_units_matter(tiny_v2_pipeline):
    sparse = sparse_grid(32, 48)
    metres = run(tiny_v2_pipeline, config=V2Config(steps=3), sparse=sparse)['depth']
    millimetres = run(tiny_v2_pipeline, config=V2Config(steps=3), sparse=sparse * 1000)['depth']
    assert not np.allclose(millimetres, metres * 1000, rtol=1e-4)


def test_robust_anchor_resists_an_outlier(tiny_v2_pipeline):
    sparse = sparse_grid(32, 48)
    corrupted = sparse.copy()
    corrupted[0, 0] = 50.0  # one gross outlier among 96 points in [0.2, 2]
    clean = run(tiny_v2_pipeline, config=V2Config(steps=4, robust='huber'), sparse=sparse)['depth']
    robust = run(tiny_v2_pipeline, config=V2Config(steps=4, robust='huber'), sparse=corrupted)['depth']
    plain = run(tiny_v2_pipeline, config=V2Config(steps=4), sparse=corrupted)['depth']
    assert np.isfinite(robust).all()
    assert np.abs(robust - clean).mean() < np.abs(plain - clean).mean()


def tilted_normals(h=32, w=48):
    n = np.zeros((h, w, 3), np.float32)
    n[..., 0], n[..., 2] = 0.4, 1.0
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def test_normal_guidance_changes_output(tiny_v2_pipeline):
    base = V2Config(steps=3, normal_start=0.0)
    without = run(tiny_v2_pipeline, config=base)['depth']
    guided = run(tiny_v2_pipeline, config=V2Config(steps=3, normal_start=0.0, normal_weight=1.0),
                 normals=tilted_normals(), kappa=np.ones((32, 48), np.float32))
    assert np.isfinite(guided['depth']).all()
    assert np.abs(guided['depth'] - without).max() > 1e-6
    assert all('normal_loss' in t for t in guided['trace'])


def test_normal_ramp_is_late():
    from hifidc.v2.pipeline import normal_ramp
    config = V2Config(steps=50, normal_weight=0.1, normal_start=0.66)
    assert normal_ramp(0, config) == 0.0 and normal_ramp(32, config) == 0.0
    assert abs(normal_ramp(49, config) - 0.1) < 1e-12
    assert 0 < normal_ramp(40, config) < 0.1


def test_normal_weight_requires_normals(tiny_v2_pipeline):
    with pytest.raises(ValueError, match='normals'):
        run(tiny_v2_pipeline, config=V2Config(steps=3, normal_weight=0.1))


def test_field_lowers_final_anchor_error(tiny_v2_pipeline):
    sparse = sparse_grid(32, 48)
    sparse[:, 24:] *= 1.5  # right half on a different scale
    result = run(tiny_v2_pipeline, config=V2Config(steps=4, field=True, field_lambda=0.1, field_start=0.0,
                                                   field_every=1), sparse=sparse)
    mask = sparse > 0
    with_field = ((result['depth'] - sparse)[mask] ** 2).mean()
    global_only = ((result['depth_global'] - sparse)[mask] ** 2).mean()
    assert np.isfinite(result['depth']).all()
    assert with_field < global_only
    assert any('field_rms' in t for t in result['trace'])


def test_field_off_has_no_field_outputs(tiny_v2_pipeline):
    result = run(tiny_v2_pipeline, config=V2Config(steps=3))
    assert 'depth_global' not in result and all('field_rms' not in t for t in result['trace'])


def test_post_filter_is_applied_to_the_final_depth(tiny_v2_pipeline):
    from hifidc.v2.postfilter import guided_log_depth
    h, w = 32, 48
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, w // 2:] = 200
    image = Image.fromarray(rgb)
    plain = tiny_v2_pipeline(image=image, sparse_depth=sparse_grid(h, w), config=V2Config(steps=3), seed=2024,
                             processing_resolution=48)
    filtered = tiny_v2_pipeline(image=image, sparse_depth=sparse_grid(h, w),
                                config=V2Config(steps=3, post_filter="guided"), seed=2024, processing_resolution=48)
    np.testing.assert_allclose(filtered, guided_log_depth(plain, rgb), rtol=1e-5)
