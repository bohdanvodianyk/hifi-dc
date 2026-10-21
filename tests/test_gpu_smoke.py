"""Real-weight smoke tests (CUDA + Marigold weights + local data + cached normals). Run: pytest -m gpu"""
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA, NORMALS = ROOT / 'data' / 'test', ROOT / 'cache' / 'normals'
STEM = 'office_4_1240'

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA'),
    pytest.mark.skipif(not (NORMALS / f'{STEM}_normals.npy').is_file(), reason='needs data + cached normals'),
]


def load():
    from hifidc.data import load_manifest, load_scene
    spec = next(s for s in load_manifest(ROOT / 'configs' / 'scenes.yaml') if s.stem == STEM)
    return load_scene(DATA, spec), np.load(NORMALS / f'{STEM}_normals.npy')


def test_cached_normals_are_unit_and_aligned():
    scene, normals = load()
    assert normals.shape == (*scene.gt.shape, 3)
    assert np.abs(np.linalg.norm(normals, axis=-1) - 1).max() < 1e-3


def test_hifidc_smoke_128px():
    from hifidc.bench import make_hifidc_method
    from hifidc.config import load_config
    scene, normals = load()
    method = make_hifidc_method(load_config(ROOT / 'configs' / 'paper_v1.yaml', {'steps': 6}), torch.device('cuda'))
    pred, info = method(scene, normals, 2024, 128)
    assert pred.shape == scene.gt.shape and np.isfinite(pred).all()
    assert info['unet_calls'] == 12
    assert all(t['latent_gradient_norm'] > 0 for t in info['trace'])
    del method
    torch.cuda.empty_cache()


def test_marigold_dc_smoke_128px():
    from hifidc.bench import make_marigold_dc_method
    scene, _ = load()
    method = make_marigold_dc_method(torch.device('cuda'), steps=6)
    pred, info = method(scene, None, 2024, 128)
    assert pred.shape == scene.gt.shape and np.isfinite(pred).all()
    assert info['unet_calls'] == 6
    del method
    torch.cuda.empty_cache()
