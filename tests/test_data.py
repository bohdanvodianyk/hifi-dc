"""Per-scene manifest and gated loading: sentinels are removed from inputs and GT, per scene."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from hifidc.data import SceneSpec, load_manifest, load_scene, valid_mask

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / 'configs' / 'scenes.yaml'
DATA = ROOT / 'data' / 'test'


def spec(cap=None, stem='toy'):
    return SceneSpec(stem=stem, source='toy', to_metres=1.0, invalid_at_or_above=cap,
                     unit_uncertain=False, outdoor=False, note='')


def write_scene(directory, stem, sparse, gt, rgb_shape=None):
    h, w = rgb_shape or sparse.shape
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(directory / f'{stem}_rgb.png')
    np.save(directory / f'{stem}_sparse.npy', sparse)
    np.save(directory / f'{stem}_gt.npy', gt)


def test_sentinel_removed_from_sparse_and_gt(tmp_path):
    sparse = np.zeros((4, 4), np.float32)
    sparse[0, 0], sparse[1, 1] = 65535, 7.
    gt = np.full((4, 4), 5., np.float32)
    gt[2, 2] = 65535
    write_scene(tmp_path, 'toy', sparse, gt)
    scene = load_scene(tmp_path, spec(cap=65535))
    assert scene.sparse[0, 0] == 0 and scene.sparse[1, 1] == 7.
    assert scene.gt[2, 2] == 0 and not scene.valid_gt[2, 2]
    assert int(scene.valid_gt.sum()) == 15


def test_value_100_kept_without_cap_and_removed_with_cap(tmp_path):
    sparse = np.zeros((4, 4), np.float32)
    sparse[0, 0], sparse[3, 3] = 100., 50.
    gt = np.full((4, 4), 100., np.float32)
    write_scene(tmp_path, 'toy', sparse, gt)
    kept = load_scene(tmp_path, spec(cap=None))
    assert kept.sparse[0, 0] == 100. and kept.valid_gt.all()
    capped = load_scene(tmp_path, spec(cap=100))
    assert capped.sparse[0, 0] == 0 and capped.sparse[3, 3] == 50. and not capped.valid_gt.any()


def test_nan_and_negative_zeroed_and_float64_cast(tmp_path):
    sparse = np.zeros((4, 4), np.float64)
    sparse[0, 0], sparse[0, 1], sparse[0, 2] = np.nan, -3., 2.
    gt = np.full((4, 4), 3., np.float64)
    gt[1, 1] = np.inf
    write_scene(tmp_path, 'toy', sparse, gt)
    scene = load_scene(tmp_path, spec())
    assert scene.sparse.dtype == np.float32 and scene.gt.dtype == np.float32
    assert scene.sparse[0, 0] == 0 and scene.sparse[0, 1] == 0 and scene.sparse[0, 2] == 2.
    assert not scene.valid_gt[1, 1] and scene.gt[1, 1] == 0


def test_shape_mismatch_raises_with_stem(tmp_path):
    write_scene(tmp_path, 'odd', np.zeros((4, 4), np.float32), np.ones((4, 5), np.float32))
    with pytest.raises(ValueError, match='odd'):
        load_scene(tmp_path, spec(stem='odd'))


def test_valid_mask():
    depth = np.array([np.nan, -1., 0., 1., 99., 100.], np.float32)
    assert valid_mask(depth, spec()).tolist() == [False, False, False, True, True, True]
    assert valid_mask(depth, spec(cap=100)).tolist() == [False, False, False, True, True, False]


def test_manifest_contents():
    specs = {s.stem: s for s in load_manifest(SCENES)}
    assert len(specs) == 10
    assert specs['Scene20_00836'].invalid_at_or_above == 65535
    assert specs['market_2_frame_0017'].invalid_at_or_above == 100
    assert sum(s.invalid_at_or_above is not None for s in specs.values()) == 2
    assert {k for k, s in specs.items() if s.unit_uncertain} == {
        'ai_055_004_90', 'office_4_1240', 'student_lounge_0001_out_379', 'study_room_0004_out_305'}
    assert {k for k, s in specs.items() if s.outdoor} == {'2011_10_03_drive_0047_sync_0000000791_03', 'Scene20_00836'}
    assert all(s.to_metres > 0 for s in specs.values())


@pytest.mark.skipif(not DATA.is_dir(), reason='local 10-scene data not present (CI)')
def test_manifest_matches_local_data():
    stems = {p.name[:-len('_rgb.png')] for p in DATA.glob('*_rgb.png')}
    specs = load_manifest(SCENES)
    assert {s.stem for s in specs} == stems
    for s in specs:
        scene = load_scene(DATA, s)
        assert int((scene.sparse > 0).sum()) >= 4
        assert scene.rgb.shape[:2] == scene.gt.shape
    scene20 = load_scene(DATA, next(s for s in specs if s.stem == 'Scene20_00836'))
    assert int((scene20.sparse > 0).sum()) == 1000 - 247


def test_invalid_below_gates_sparse_and_gt(tmp_path):
    sparse = np.zeros((4, 4), np.float32)
    sparse[0, 0], sparse[1, 1] = 45., 600.
    gt = np.full((4, 4), 700., np.float32)
    gt[2, 2] = 95.
    write_scene(tmp_path, "toy", sparse, gt)
    s = SceneSpec(stem="toy", source="toy", to_metres=1., invalid_at_or_above=None, unit_uncertain=False,
                  outdoor=False, note="", invalid_below=500.)
    scene = load_scene(tmp_path, s)
    assert scene.sparse[0, 0] == 0 and scene.sparse[1, 1] == 600.
    assert not scene.valid_gt[2, 2] and int(scene.valid_gt.sum()) == 15


def test_office4_lower_limit_in_manifest():
    specs = {s.stem: s for s in load_manifest(SCENES)}
    assert specs["office_4_1240"].invalid_below == 500
    assert sum(s.invalid_below is not None for s in specs.values()) == 1


@pytest.mark.skipif(not DATA.is_dir(), reason="local 10-scene data not present (CI)")
def test_office4_low_anchors_removed():
    spec_ = next(s for s in load_manifest(SCENES) if s.stem == "office_4_1240")
    scene = load_scene(DATA, spec_)
    assert int((scene.sparse > 0).sum()) == 997
    assert float(scene.gt[scene.valid_gt].min()) >= 500
