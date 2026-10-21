"""Progressive (half -> full) resolution path at paper scale and through real Diffusers modules."""
import numpy as np
from PIL import Image
import torch

from conftest import mock_pipe, sparse_grid
from hifidc.config import Config
from hifidc.core import complete
from hifidc.pipeline import DiffusionBackend


def test_progressive_384_to_768():
    backend = DiffusionBackend(mock_pipe(), torch.Generator().manual_seed(2024))
    rgb = torch.linspace(.05, .95, 3 * 768 * 768).reshape(1, 3, 768, 768)
    sparse = torch.zeros(1, 1, 768, 768)
    sparse[:, :, ::10, ::10] = torch.linspace(.5, 5., 77 * 77).reshape(77, 77)  # ~1 % of pixels
    result = complete(backend, rgb, sparse, None, Config(steps=3, bilateral=False, use_normals=False))
    assert [r['height'] for r in result['trace']] == [384, 768, 768]
    assert [r['width'] for r in result['trace']] == [384, 768, 768]
    assert result['depth'].shape == (1, 1, 768, 768)
    assert torch.isfinite(result['depth']).all()


def test_progressive_real_tiny_diffusers(tiny_real_pipeline):
    image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    normals = np.zeros((64, 64, 3), dtype=np.float32)
    normals[:, :, 2] = 1
    result = tiny_real_pipeline(image=image, sparse_depth=sparse_grid(64, 64), normal_prior=normals,
                                num_inference_steps=3, processing_resolution=64,
                                config=Config(bilateral=False), return_details=True)
    assert [t['height'] for t in result['trace']] == [32, 64, 64]
    assert all(t['adam_max_update'] > 0 for t in result['trace'])
    assert result['depth'].shape == (64, 64)
