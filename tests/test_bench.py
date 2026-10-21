"""Benchmark runner: output layout, failure isolation (error / OOM), overwrite protection."""
import csv
import json

import numpy as np
import pytest
import torch

from hifidc.bench import CSV_COLUMNS, run_benchmark
from hifidc.data import Scene, SceneSpec


def make_scene(stem, n_sparse=10, outdoor=False):
    rng = np.random.default_rng(len(stem))
    gt = rng.uniform(1., 5., (8, 8)).astype(np.float32)
    sparse = np.zeros_like(gt)
    sparse.flat[:n_sparse] = gt.flat[:n_sparse]
    spec = SceneSpec(stem=stem, source='toy', to_metres=1., invalid_at_or_above=None,
                     unit_uncertain=False, outdoor=outdoor, note='')
    return Scene(spec=spec, rgb=np.zeros((8, 8, 3), np.uint8), sparse=sparse, gt=gt, valid_gt=gt > 0)


def fake_method(scene, normals, seed, resolution):
    return scene.gt * 1.1, {'unet_calls': 7, 'trace': [{'step': 1}]}


def read_rows(out):
    with open(out / 'metrics.csv', newline='') as handle:
        return list(csv.DictReader(handle))


def test_layout_and_rows(tmp_path):
    out = tmp_path / 'run'
    scenes = [make_scene('a'), make_scene('b', outdoor=True)]
    rows = run_benchmark(fake_method, scenes, None, [1, 2], 64, out, {'method': 'fake'})
    assert len(rows) == 4 and all(r['status'] == 'ok' for r in rows)
    table = read_rows(out)
    assert list(table[0]) == CSV_COLUMNS
    assert [(r['stem'], r['seed']) for r in table] == [('a', '1'), ('a', '2'), ('b', '1'), ('b', '2')]
    assert abs(float(table[0]['abs_rel']) - 0.1) < 1e-6 and table[0]['unet_calls'] == '7'
    assert (out / 'pred' / 'a_s1.npy').is_file() and (out / 'trace' / 'b_s2.json').is_file()
    summary = json.loads((out / 'summary.json').read_text())
    assert abs(summary['all']['abs_rel'] - 0.1) < 1e-6
    assert summary['indoor']['n_scenes'] == 2 and summary['outdoor']['n_scenes'] == 2
    assert 'rmse' not in summary['all']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert len(manifest['git_sha']) == 40 and manifest['run']['method'] == 'fake'
    assert 'torch' in manifest['versions']


def test_too_few_sparse_points_is_recorded_and_run_continues(tmp_path):
    rows = run_benchmark(fake_method, [make_scene('few', n_sparse=3), make_scene('ok')], None, [1], 64,
                         tmp_path / 'run', {})
    assert rows[0]['status'] == 'error' and 'sparse' in rows[0]['message']
    assert rows[1]['status'] == 'ok'


def test_oom_is_recorded_and_run_continues(tmp_path):
    def flaky(scene, normals, seed, resolution):
        if scene.spec.stem == 'big':
            raise torch.OutOfMemoryError('CUDA out of memory (simulated)')
        return fake_method(scene, normals, seed, resolution)

    rows = run_benchmark(flaky, [make_scene('big'), make_scene('small')], None, [1], 64, tmp_path / 'run', {})
    assert [r['status'] for r in rows] == ['oom', 'ok']
    assert not (tmp_path / 'run' / 'pred' / 'big_s1.npy').exists()


def test_existing_output_refused_without_overwrite(tmp_path):
    out = tmp_path / 'run'
    run_benchmark(fake_method, [make_scene('a')], None, [1], 64, out, {})
    with pytest.raises(FileExistsError):
        run_benchmark(fake_method, [make_scene('a')], None, [1], 64, out, {})
    rows = run_benchmark(fake_method, [make_scene('a')], None, [1], 64, out, {}, overwrite=True)
    assert rows[0]['status'] == 'ok'


def test_normals_loaded_per_scene(tmp_path):
    normals_dir = tmp_path / 'normals'
    normals_dir.mkdir()
    n = np.zeros((8, 8, 3), np.float32)
    n[..., 2] = 1
    np.save(normals_dir / 'a_normals.npy', n)
    seen = {}

    def spy(scene, normals, seed, resolution):
        seen[scene.spec.stem] = normals
        return fake_method(scene, normals, seed, resolution)

    run_benchmark(spy, [make_scene('a')], normals_dir, [1], 64, tmp_path / 'run', {})
    assert seen['a'].shape == (8, 8, 3)


def test_configure_determinism_sets_flags_and_manifest_records_them(tmp_path):
    import os
    from hifidc.bench import configure_determinism
    try:
        state = configure_determinism()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert state["deterministic_algorithms"] is True
        out = tmp_path / "run"
        run_benchmark(fake_method, [make_scene("a")], None, [1], 64, out, {})
        recorded = json.loads((out / "manifest.json").read_text())["determinism"]
        assert recorded["deterministic_algorithms"] is True and recorded["cublas_workspace_config"] == ":4096:8"
        assert "cudnn_allow_tf32" in recorded and "matmul_allow_tf32" in recorded
    finally:
        torch.use_deterministic_algorithms(False)


def test_determinism_forces_math_attention_unless_disabled():
    from hifidc.bench import configure_determinism
    cuda = torch.backends.cuda
    try:
        state = configure_determinism()
        assert not cuda.flash_sdp_enabled() and not cuda.mem_efficient_sdp_enabled() and not cuda.cudnn_sdp_enabled()
        assert cuda.math_sdp_enabled() and state["sdpa_math_only"] is True
        state = configure_determinism(deterministic_attention=False)
        assert cuda.flash_sdp_enabled() and cuda.mem_efficient_sdp_enabled()
        assert state["sdpa_math_only"] is False
    finally:
        torch.use_deterministic_algorithms(False)
        cuda.enable_flash_sdp(True)
        cuda.enable_mem_efficient_sdp(True)
        cuda.enable_cudnn_sdp(True)
