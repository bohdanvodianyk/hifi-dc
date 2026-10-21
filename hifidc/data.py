"""Evaluation scenes: per-scene manifest (units, sentinels) and gated loading.

Every scene keeps its native depth unit. ``to_metres`` converts it for the method (whose
hyper-parameters assume metres); metrics are computed in native units, per scene only.
Invalid values (non-finite, <= 0, or >= the scene's sentinel/cap) are zeroed in BOTH the sparse
input and the ground truth, so a sentinel can never act as an anchor.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


@dataclass(frozen=True)
class SceneSpec:
    stem: str
    source: str
    to_metres: float
    invalid_at_or_above: float | None
    unit_uncertain: bool
    outdoor: bool
    note: str
    invalid_below: float | None = None  # depths below this are invalid (sparse and GT)


@dataclass
class Scene:
    spec: SceneSpec
    rgb: np.ndarray        # (H, W, 3) uint8
    sparse: np.ndarray     # (H, W) float32, native units, 0 = no measurement
    gt: np.ndarray         # (H, W) float32, native units, 0 = invalid
    valid_gt: np.ndarray   # (H, W) bool
    case_id: str | None = None           # set for resampled evaluation cases (hifidc.cases)
    holdout: np.ndarray | None = None    # (H, W) withheld clean points, 0 elsewhere


def load_manifest(path):
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    specs = []
    for entry in document['scenes']:
        cap = entry.get('invalid_at_or_above')
        specs.append(SceneSpec(stem=str(entry['stem']), source=entry['source'], to_metres=float(entry['to_metres']),
                               invalid_at_or_above=None if cap is None else float(cap),
                               unit_uncertain=bool(entry['unit_uncertain']), outdoor=bool(entry['outdoor']),
                               note=entry.get('note', ''),
                               invalid_below=None if entry.get('invalid_below') is None else float(entry['invalid_below'])))
    return specs


def valid_mask(depth, spec):
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(depth) & (depth > 0)
        if spec.invalid_at_or_above is not None:
            valid &= depth < spec.invalid_at_or_above
        if spec.invalid_below is not None:
            valid &= depth >= spec.invalid_below
    return valid


def load_scene(data_dir, spec):
    data_dir = Path(data_dir)
    rgb = np.asarray(Image.open(data_dir / f'{spec.stem}_rgb.png').convert('RGB'))
    sparse = np.load(data_dir / f'{spec.stem}_sparse.npy', allow_pickle=False)
    gt = np.load(data_dir / f'{spec.stem}_gt.npy', allow_pickle=False)
    if not (rgb.shape[:2] == sparse.shape == gt.shape):
        raise ValueError(f'{spec.stem}: shape mismatch rgb {rgb.shape[:2]}, sparse {sparse.shape}, gt {gt.shape}')
    sparse_ok, gt_ok = valid_mask(sparse, spec), valid_mask(gt, spec)
    return Scene(spec=spec, rgb=rgb,
                 sparse=np.where(sparse_ok, sparse, 0).astype(np.float32),
                 gt=np.where(gt_ok, gt, 0).astype(np.float32),
                 valid_gt=gt_ok)
