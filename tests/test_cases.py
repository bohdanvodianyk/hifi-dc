"""Reproducible evaluation cases: resampled sparse patterns, outliers and held-out points."""
import csv
from pathlib import Path

import numpy as np

from hifidc.bench import run_benchmark
from hifidc.cases import DENSE_GT_STEMS, dev50, make_case, stress48
from hifidc.data import Scene, load_manifest

SCENES = Path(__file__).resolve().parent.parent / 'configs' / 'scenes.yaml'


def toy_scene(stem='toy', n_provided=100, size=64, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1., 5., (size, size)).astype(np.float32)
    sparse = np.zeros_like(gt)
    idx = rng.choice(gt.size, n_provided, replace=False)
    sparse.flat[idx] = gt.flat[idx] * 1.01
    spec = next((s for s in load_manifest(SCENES) if s.stem == stem), None)
    if spec is None:
        from hifidc.data import SceneSpec
        spec = SceneSpec(stem=stem, source='toy', to_metres=1., invalid_at_or_above=None,
                         unit_uncertain=False, outdoor=False, note='')
    return Scene(spec=spec, rgb=np.zeros((size, size, 3), np.uint8), sparse=sparse, gt=gt, valid_gt=gt > 0)


def points(a):
    return set(zip(*np.nonzero(a)))


def test_pattern_zero_keeps_provided_points_minus_holdout():
    scene = toy_scene()
    case = make_case(scene, pattern=0)
    assert len(points(case.sparse)) == 80 and len(points(case.holdout)) == 20
    assert points(case.sparse) | points(case.holdout) == points(scene.sparse)
    assert not points(case.sparse) & points(case.holdout)
    assert np.array_equal(case.holdout[case.holdout > 0], scene.sparse[case.holdout > 0])
    assert case.case_id == 'toy/p0'


def test_resampled_pattern_is_deterministic_and_on_valid_gt():
    scene = toy_scene()
    a, b = make_case(scene, pattern=1), make_case(scene, pattern=1)
    assert np.array_equal(a.sparse, b.sparse) and np.array_equal(a.holdout, b.holdout)
    both = (a.sparse > 0) | (a.holdout > 0)
    assert int(both.sum()) == 1000 and scene.valid_gt[both].all()
    assert np.array_equal(a.sparse[a.sparse > 0], scene.gt[a.sparse > 0])
    assert a.case_id == 'toy/p1_n1000_o0'


def test_patterns_differ():
    scene = toy_scene()
    assert not np.array_equal(make_case(scene, pattern=1).sparse, make_case(scene, pattern=2).sparse)


def test_outliers_are_scaled_and_holdout_clean():
    scene = toy_scene()
    case = make_case(scene, pattern=1, outlier_frac=0.05)
    mask = case.sparse > 0
    ratio = case.sparse[mask] / scene.gt[mask]
    moved = ~np.isclose(ratio, 1.0)
    assert int(mask.sum()) == 800 and int(moved.sum()) == 40
    assert np.all((ratio[moved] <= 0.8) | (ratio[moved] >= 1.25))
    assert np.array_equal(case.holdout[case.holdout > 0], scene.gt[case.holdout > 0])
    assert case.case_id == 'toy/p1_n1000_o5'


def test_case_set_sizes():
    ten = [toy_scene(s.stem, seed=i) for i, s in enumerate(load_manifest(SCENES))]
    dev, stress = dev50(ten), stress48(ten)
    assert len(dev) == 50 and len(stress) == 48
    ids = [c.case_id for c in dev + stress]
    assert len(set(ids)) == len(ids)
    assert len(DENSE_GT_STEMS) == 8
    assert {c.spec.stem for c in stress} == set(DENSE_GT_STEMS)


def test_runner_writes_case_and_holdout_columns(tmp_path):
    case = make_case(toy_scene(), pattern=1)

    def fake(scene, normals, seed, resolution):
        return scene.gt * 1.1, {'unet_calls': 1}

    run_benchmark(fake, [case], None, [7], 64, tmp_path / 'run', {})
    with open(tmp_path / 'run' / 'metrics.csv', newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['case'] == 'toy/p1_n1000_o0' and row['stem'] == 'toy'
    expected = 0.1 * float(case.holdout[case.holdout > 0].astype(np.float64).mean())
    assert abs(float(row['holdout_residual']) - expected) < 1e-5
    assert (tmp_path / 'run' / 'pred' / 'toy__p1_n1000_o0_s7.npy').is_file()
