import json

import numpy as np
from PIL import Image
import yaml

from hifidc.data import load_manifest, load_scene
from hifidc.drive_import import (DATASETS, import_selection, normalize_stem, pair_files, select_cases,
                                 sequence_key)


def test_normalize_stem_strips_extensions_suffixes_and_copy_marks():
    assert normalize_stem('office_4_1240.png') == 'office_4_1240'
    assert normalize_stem('office_4_1240 (1).npy') == 'office_4_1240'
    assert normalize_stem('Scene20_00836_gt.npy') == 'Scene20_00836'
    assert normalize_stem('market_2_frame_0017_sparse.npy') == 'market_2_frame_0017'
    assert normalize_stem('visionlab1_1552089403.9210.png') == 'visionlab1_1552089403.9210'


def test_pair_files_keeps_only_complete_triplets():
    listing = {'rgb': ['a.png', 'b.jpg', 'c.png'], 'npy_sparse_depth': ['a.npy', 'b.npy'],
               'npy_ground_truth': ['a.npy', 'b.npy', 'c.npy'], 'validity_map': ['a.npy']}
    pairs = pair_files(listing)
    assert sorted(pairs) == ['a', 'b']
    assert pairs['a']['validity'] == 'a.npy' and pairs['b']['validity'] is None


def test_sequence_keys_group_frames_of_one_sequence():
    assert sequence_key('kitti', '2011_10_03_drive_0047_sync_0000000791_03') == '2011_10_03_drive_0047'
    assert sequence_key('vkitti2', 'Scene20_00836') == 'Scene20'
    assert sequence_key('nyuv2', 'study_room_0004_out_305') == 'study_room_0004'
    assert sequence_key('void', 'workbench0_1551915136.1459') == 'workbench0'
    assert sequence_key('sintel', 'market_2_frame_0017') == 'market_2'
    assert sequence_key('hypersim', 'ai_055_004_90') == 'ai_055_004'
    assert sequence_key('icl_nuim', 'office_4_1240') == 'office_4'
    assert sequence_key('interiornet', '3FO4MMLDUKDJ_Living_room_8') == '3FO4MMLDUKDJ'


def test_select_is_reproducible_disjoint_and_excludes_dev_sequences():
    stems = {'nyu': [f'room_{s:04d}_out_{f}' for s in range(12) for f in range(3)]}
    excluded = {('nyuv2', 'room_0000')}
    a = select_cases(stems, n_test=5, n_dev=3, seed=7, exclude=excluded)
    b = select_cases(stems, n_test=5, n_dev=3, seed=7, exclude=excluded)
    assert a == b
    test_seq = {sequence_key('nyuv2', s) for _, s in a['test']}
    dev_seq = {sequence_key('nyuv2', s) for _, s in a['dev']}
    assert len(a['test']) == 5 and len(a['dev']) == 3
    assert len(test_seq) == 5 and not test_seq & dev_seq           # one frame per sequence, disjoint
    assert 'room_0000' not in test_seq | dev_seq


def test_select_reuses_sequences_only_when_there_are_too_few():
    stems = {'icl': [f'office_{s}_{f}' for s in range(2) for f in range(10)]}
    sel = select_cases(stems, n_test=4, n_dev=0, seed=0, exclude=set())
    assert len(sel['test']) == 4 and len(set(sel['test'])) == 4


def _write_dataset(root, name, stems, gt_value=1000.0, shape=(24, 32)):
    base = root / f'output_{name}'
    for folder in ('rgb', 'npy_sparse_depth', 'npy_ground_truth', 'validity_map'):
        (base / folder).mkdir(parents=True)
    for stem in stems:
        Image.fromarray(np.full(shape + (3,), 100, np.uint8)).save(base / 'rgb' / f'{stem}.png')
        sparse = np.zeros(shape, np.float64)
        sparse[::4, ::4] = gt_value
        np.save(base / 'npy_sparse_depth' / f'{stem}.npy', sparse)
        np.save(base / 'npy_ground_truth' / f'{stem}.npy', np.full(shape + (1,), gt_value))
        validity = np.ones(shape, np.uint8)
        validity[0, :] = 0
        np.save(base / 'validity_map' / f'{stem}.npy', validity)


def test_import_selection_writes_loadable_scenes_and_manifest(tmp_path):
    root = tmp_path / 'main_datasets_new'
    _write_dataset(root, 'void', [f'lab{s}_{1000 + s}.5' for s in range(4)])
    _write_dataset(root, 'kitti', [f'2011_09_26_drive_{s:04d}_sync_0000000005_02' for s in range(4)])
    selection = select_cases({'void': [f'lab{s}_{1000 + s}.5' for s in range(4)],
                              'kitti': [f'2011_09_26_drive_{s:04d}_sync_0000000005_02' for s in range(4)]},
                             n_test=2, n_dev=1, seed=0, exclude=set())
    out = tmp_path / 'newdata'
    paths = import_selection(root, selection, out)
    test_specs = load_manifest(paths['test'])
    assert len(test_specs) == 4 and {s.source for s in test_specs} == {'void', 'kitti'}
    kitti = next(s for s in test_specs if s.source == 'kitti')
    assert kitti.to_metres == DATASETS['kitti']['to_metres'] and kitti.intrinsics is not None
    scene = load_scene(out / 'data', kitti)
    assert scene.gt.dtype == np.float32 and scene.gt.shape == (24, 32)
    assert not scene.valid_gt[0].any() and scene.valid_gt[1:].all()     # validity map applied to GT
    assert json.loads((out / 'selection.json').read_text())['test'] == [list(x) for x in selection['test']]
    assert len(yaml.safe_load(paths['dev'].read_text())['scenes']) == 2
