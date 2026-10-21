"""Select and import new evaluation scenes from the Google Drive dataset folders.

Expected layout (one folder per dataset):

    <root>/output_<name>/rgb/<stem>.(png|jpg)
    <root>/output_<name>/npy_sparse_depth/<stem>.npy
    <root>/output_<name>/npy_ground_truth/<stem>.npy
    <root>/output_<name>/validity_map/<stem>.(npy|png)      ignored: VOID-style mask of the sparse points

Selection draws whole *sequences* first (one frame per sequence while there are enough), so the
held-out test set does not contain near-duplicate frames, keeps test and dev sequences disjoint,
and skips every sequence already used by the 10-scene development set. Imported scenes are written
in the ``data/test`` layout (``<stem>_rgb.png``, ``_sparse.npy``, ``_gt.npy``) plus a scenes YAML
per split, so ``scripts/run_benchmark.py --data ... --scenes ...`` runs on them unchanged.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import zlib

import numpy as np
from PIL import Image
import yaml

# Per-dataset defaults, copied from the matching scene in configs/scenes.yaml (same export).
# Units marked unit_uncertain are guesses; headline metrics are scale-free.
DATASETS = {
    'kitti': dict(source='kitti', to_metres=1 / 256, invalid_at_or_above=None, unit_uncertain=False, outdoor=True,
                  intrinsics=[718.856, 718.856, 594.69, 161.22],
                  note='KITTI uint16/256; intrinsics approximate (2011_10_03 calibration, depth-completion crop).'),
    'vkitti': dict(source='vkitti2', to_metres=0.01, invalid_at_or_above=65535, unit_uncertain=False, outdoor=True,
                   intrinsics=[725.0087, 725.0087, 620.5, 187.0], note='Virtual KITTI 2 cm; 65535 = sky.'),
    'nyu': dict(source='nyuv2', to_metres=10 / 255, invalid_at_or_above=None, unit_uncertain=True, outdoor=False,
                intrinsics=[518.8579, 519.4696, 325.5824, 253.7362], note='Looks 8-bit; guessed 10 m / 255.',
                overrides=[(r'^nyu2_test_', dict(to_metres=0.001, unit_uncertain=False,
                                                    note='NYUv2 labelled test frame, uint16 mm.'))]),
    'void': dict(source='void', to_metres=1 / 256, invalid_at_or_above=None, unit_uncertain=False, outdoor=False,
                 intrinsics=None, note='VOID uint16/256.'),
    'hypersim': dict(source='hypersim', to_metres=10 / 255, invalid_at_or_above=None, unit_uncertain=True,
                     outdoor=False, intrinsics=None,
                     note='Per-image normalised 8-bit encoding (every frame spans 30-215): not metric, affine-distorted; '
                          'report separately.'),
    'interiornet': dict(source='interiornet', to_metres=0.001, invalid_at_or_above=None, unit_uncertain=False,
                        outdoor=False, intrinsics=[600.0, 600.0, 320.0, 240.0], note='InteriorNet mm.'),
    'icl': dict(source='icl_nuim', to_metres=0.0002, invalid_at_or_above=None, invalid_below=500,
                unit_uncertain=True, outdoor=False, intrinsics=[481.2, 480.0, 319.5, 239.5],
                note='Guessed ICL-NUIM 1/5000; depths < 500 native gated (as office_4).'),
    'sintel': dict(source='sintel', to_metres=1.0, invalid_at_or_above=100, unit_uncertain=False, outdoor=False,
                   intrinsics=None, note='MPI-Sintel float depth; 100 = far cap.'),
}

_SEQUENCE = {
    'kitti': r'^(\d{4}_\d{2}_\d{2}_drive_\d{4})',
    'vkitti2': r'^(Scene\d+)',
    'nyuv2': r'^(.+)_out_\d+$',
    'void': r'^(.+?)_\d+\.\d+$',
    'sintel': r'^(.+)_frame_\d+$',
    'hypersim': r'^(ai_\d{3}_\d{3})',
    'icl_nuim': r'^(.+)_\d+$',
    'interiornet': r'^([A-Za-z0-9]+)_',
}
_SUFFIX = re.compile(r'(\s*\(\d+\))?(_(rgb|sparse|gt|depth|validity|valid|mask))?$')
_KINDS = {'rgb': 'rgb', 'npy_sparse_depth': 'sparse', 'npy_ground_truth': 'gt', 'validity_map': 'validity'}
_EXTS = ('.png', '.jpg', '.jpeg', '.npy')


def normalize_stem(filename):
    """File name -> scene stem: drop a known extension, a ' (1)' copy mark and a _rgb/_gt/... suffix."""
    name = Path(filename).name
    for ext in _EXTS:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    return _SUFFIX.sub('', name)


def pair_files(listing):
    """{folder: [file names]} -> {stem: {'rgb', 'sparse', 'gt', 'validity'}} for stems with rgb+sparse+gt."""
    found = defaultdict(dict)
    for folder, names in listing.items():
        kind = _KINDS.get(folder)
        if kind is None:
            continue
        for name in sorted(names):
            if name.lower().endswith(_EXTS):
                found[normalize_stem(name)].setdefault(kind, name)
    return {stem: {'rgb': f['rgb'], 'sparse': f['sparse'], 'gt': f['gt'], 'validity': f.get('validity')}
            for stem, f in sorted(found.items()) if {'rgb', 'sparse', 'gt'} <= set(f)}


def sequence_key(source, stem):
    match = re.match(_SEQUENCE.get(source, r'^(.+)$'), stem)
    return match.group(1) if match else stem


def dev_sequences(specs):
    """(source, sequence) pairs used by an existing scene list (e.g. the 10 development scenes)."""
    return {(s.source, sequence_key(s.source, s.stem)) for s in specs}


def _round_robin(groups, n, rng):
    """Up to n frames, one per sequence per pass, sequences in the given (shuffled) order."""
    queues = [list(rng.permutation(frames)) for frames in groups]
    picks = []
    while len(picks) < n and any(queues):
        for queue in queues:
            if queue and len(picks) < n:
                picks.append(str(queue.pop(0)))
    return picks


def select_cases(stems_by_dataset, n_test, n_dev, seed, exclude):
    """Reproducible split: {'test': [(dataset, stem)], 'dev': [...], 'short': {dataset: msg}}."""
    selection = {'test': [], 'dev': [], 'short': {}}
    for name in sorted(stems_by_dataset):
        source = DATASETS[name]['source'] if name in DATASETS else name
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        groups = defaultdict(list)
        for stem in sorted(stems_by_dataset[name]):
            key = sequence_key(source, stem)
            if (source, key) not in exclude:
                groups[key].append(stem)
        keys = [str(k) for k in rng.permutation(sorted(groups))] if groups else []
        if len(keys) >= n_test + n_dev or n_dev == 0:
            k_test = min(n_test, len(keys)) if len(keys) >= n_test + n_dev else len(keys)
        else:
            k_test = max(1, round(len(keys) * n_test / (n_test + n_dev))) if keys else 0
        test_keys, dev_keys = keys[:k_test], keys[k_test:]
        if n_dev and len(keys) >= n_test + n_dev:
            dev_keys = dev_keys[:n_dev]
        test = _round_robin([groups[k] for k in test_keys], n_test, rng)
        dev = _round_robin([groups[k] for k in dev_keys], n_dev, rng)
        selection['test'] += [(name, s) for s in test]
        selection['dev'] += [(name, s) for s in dev]
        if len(test) < n_test or len(dev) < n_dev or len(keys) < n_test + n_dev:
            selection['short'][name] = (f'{len(keys)} usable sequences for {n_test}+{n_dev}: '
                                        f'took {len(test)} test / {len(dev)} dev')
    return selection


def scene_meta(name, stem):
    """Dataset defaults with any stem-pattern override applied (e.g. NYU mm test frames vs 8-bit frames)."""
    meta = {k: v for k, v in DATASETS[name].items() if k != 'overrides'}
    for pattern, override in DATASETS[name].get('overrides', []):
        if re.search(pattern, stem):
            meta.update(override)
    return meta


def _load_array(path):
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    else:
        array = np.asarray(Image.open(path))
    array = np.squeeze(array)
    if array.ndim == 3:  # e.g. an RGB-encoded validity map
        array = array[..., 0]
    return array


def _import_one(base, stem, files, data_dir):
    rgb = np.asarray(Image.open(base / 'rgb' / files['rgb']).convert('RGB'))
    sparse = _load_array(base / 'npy_sparse_depth' / files['sparse']).astype(np.float32)
    gt = _load_array(base / 'npy_ground_truth' / files['gt']).astype(np.float32)
    if not (rgb.shape[:2] == sparse.shape == gt.shape):
        raise ValueError(f'shape mismatch rgb {rgb.shape[:2]}, sparse {sparse.shape}, gt {gt.shape}')
    sparse = np.where(np.isfinite(sparse), sparse, 0).astype(np.float32)
    gt = np.where(np.isfinite(gt), gt, 0).astype(np.float32)
    Image.fromarray(rgb).save(data_dir / f'{stem}_rgb.png')
    np.save(data_dir / f'{stem}_sparse.npy', sparse)
    np.save(data_dir / f'{stem}_gt.npy', gt)
    valid_gt = gt[gt > 0]
    return {'shape': list(gt.shape), 'gt_fill': float((gt > 0).mean()), 'sparse_points': int((sparse > 0).sum()),
            'gt_min': float(valid_gt.min()) if valid_gt.size else 0.0,
            'gt_median': float(np.median(valid_gt)) if valid_gt.size else 0.0,
            'gt_max': float(valid_gt.max()) if valid_gt.size else 0.0}


def list_dataset(root, name):
    base = Path(root) / f'output_{name}'
    listing = {folder: [p.name for p in (base / folder).iterdir() if p.is_file()]
               for folder in _KINDS if (base / folder).is_dir()}
    return pair_files(listing)


def import_selection(root, selection, out_dir):
    """Write the selected scenes to <out>/data, scenes_<split>.yaml and selection.json; return yaml paths."""
    root, out_dir = Path(root), Path(out_dir)
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs = {}
    paths, record = {}, {'test': [list(x) for x in selection['test']], 'dev': [list(x) for x in selection['dev']],
                         'short': selection.get('short', {}), 'skipped': [], 'stats': {}}
    seen = set()
    for split in ('test', 'dev'):
        entries = []
        for name, stem in selection[split]:
            if name not in pairs:
                pairs[name] = list_dataset(root, name)
            if stem in seen:
                raise ValueError(f'stem {stem} selected twice')
            seen.add(stem)
            try:
                stats = _import_one(root / f'output_{name}', stem, pairs[name][stem], data_dir)
            except (ValueError, KeyError, OSError) as error:
                record['skipped'].append([name, stem, str(error)])
                continue
            record['stats'][stem] = dict(stats, dataset=name, split=split)
            meta = scene_meta(name, stem)
            entry = {'stem': stem, 'source': meta['source'], 'to_metres': float(meta['to_metres']),
                     'invalid_at_or_above': meta['invalid_at_or_above'], 'unit_uncertain': meta['unit_uncertain'],
                     'outdoor': meta['outdoor'], 'note': meta['note']}
            if meta.get('invalid_below') is not None:
                entry['invalid_below'] = meta['invalid_below']
            if meta.get('intrinsics') is not None:
                entry['intrinsics'] = list(meta['intrinsics'])
            entries.append(entry)
        paths[split] = out_dir / f'scenes_{split}.yaml'
        paths[split].write_text(yaml.safe_dump({'scenes': entries}, sort_keys=False))
    (out_dir / 'selection.json').write_text(json.dumps(record, indent=1))
    return paths
