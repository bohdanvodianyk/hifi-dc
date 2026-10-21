"""Reproducible evaluation cases built from the scenes.

A case is a copy of a Scene whose sparse input is either the provided one (pattern 0) or
``n_points`` drawn uniformly from valid GT (pattern >= 1). A fraction of the points is withheld
from the input and kept clean in ``holdout`` (held-out anchor error); optional outliers scale a
fraction of the remaining input points by a factor in [0.5, 0.79] or [1.26, 2.0].
"""
from dataclasses import replace
import zlib

import numpy as np

# Scenes with (near-)dense GT: all except semi-dense KITTI and 79 %-filled VOID workbench.
DENSE_GT_STEMS = frozenset({
    'Scene20_00836', '3FO4MMLDUKDJ_Living_room_8', 'ai_055_004_90', 'market_2_frame_0017',
    'office_4_1240', 'student_lounge_0001_out_379', 'study_room_0004_out_305', 'visionlab1_1552089403.9210',
})
STRESS_SETTINGS = ((100, 0.0), (300, 0.0), (1000, 0.05))


def make_case(scene, pattern, n_points=1000, outlier_frac=0.0, holdout_frac=0.2, seed=0):
    stem = scene.spec.stem
    rng = np.random.default_rng([seed, pattern, zlib.crc32(stem.encode())])
    if pattern == 0:
        index = np.flatnonzero(scene.sparse > 0)
        values = scene.sparse.ravel()[index]
        case_id = f'{stem}/p0'
    else:
        valid = np.flatnonzero(scene.valid_gt)
        if valid.size < n_points:
            raise ValueError(f'{stem}: only {valid.size} valid GT pixels for {n_points} points')
        index = np.sort(rng.choice(valid, n_points, replace=False))
        values = scene.gt.ravel()[index]
        case_id = f'{stem}/p{pattern}_n{n_points}_o{int(round(outlier_frac * 100))}'
    order = rng.permutation(index.size)
    n_hold = int(round(holdout_frac * index.size))
    hold, keep = order[:n_hold], order[n_hold:]
    sparse = np.zeros_like(scene.gt, dtype=np.float32)
    holdout = np.zeros_like(scene.gt, dtype=np.float32)
    sparse.flat[index[keep]] = values[keep]
    holdout.flat[index[hold]] = values[hold]
    if outlier_frac > 0:
        n_out = int(round(outlier_frac * keep.size))
        chosen = index[keep[rng.choice(keep.size, n_out, replace=False)]]
        factor = np.where(rng.random(n_out) < 0.5, rng.uniform(0.5, 0.79, n_out), rng.uniform(1.26, 2.0, n_out))
        sparse.flat[chosen] = (sparse.flat[chosen] * factor).astype(np.float32)
    return replace(scene, sparse=sparse, holdout=holdout, case_id=case_id)


def dev50(scenes):
    """Patterns 0-4 (1000 points, 20 % held out) for every scene: 50 cases for 10 scenes."""
    return [make_case(scene, pattern) for scene in scenes for pattern in range(5)]


def stress48(scenes):
    """Dense-GT scenes x {100, 300, 1000+5 % outliers} x patterns 1-2: 48 cases."""
    return [make_case(scene, pattern, n, outliers)
            for scene in scenes if scene.spec.stem in DENSE_GT_STEMS
            for n, outliers in STRESS_SETTINGS for pattern in (1, 2)]
