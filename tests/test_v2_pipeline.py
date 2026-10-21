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
