"""v2 L0 must reproduce official Marigold-DC (CUDA, real weights, local data). Run: pytest -m gpu"""
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'test'
STEMS = ('office_4_1240', '3FO4MMLDUKDJ_Living_room_8')

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA'),
    pytest.mark.skipif(not DATA.is_dir(), reason='needs local data'),
]


def test_l0_matches_official_mdc():
    from hifidc.bench import make_hifidc_v2_method, make_marigold_dc_method
    from hifidc.config import load_dataclass
    from hifidc.data import load_manifest, load_scene
    from hifidc.v2.config import V2Config
    specs = {s.stem: s for s in load_manifest(ROOT / 'configs' / 'scenes.yaml')}
    scenes = [load_scene(DATA, specs[s]) for s in STEMS]
    device = torch.device('cuda')
    mdc = make_marigold_dc_method(device, steps=6)
    reference = [mdc(scene, None, 2024, 128)[0] for scene in scenes]
    del mdc
    torch.cuda.empty_cache()
    v2 = make_hifidc_v2_method(load_dataclass(V2Config, ROOT / 'configs' / 'v2' / 'L0.yaml', {'steps': 6}), device)
    ours = [v2(scene, None, 2024, 128) for scene in scenes]
    for (pred, info), ref in zip(ours, reference):
        assert info['unet_calls'] == 6
        assert np.array_equal(pred, ref), f'max |diff| {np.abs(pred - ref).max()}'
