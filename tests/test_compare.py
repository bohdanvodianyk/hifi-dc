"""Paired, scene-clustered comparison of two benchmark runs."""
import math

import pytest

from hifidc.compare import GROUPS, clustered_bootstrap, compare_rows, paired


def row(case, stem, source, abs_rel, seed=1, status='ok', **extra):
    base = {'case': case, 'stem': stem, 'source': source, 'seed': seed, 'status': status,
            'abs_rel': abs_rel, 'si_log_rmse': abs_rel, 'delta1': 0.9, 'holdout_residual': abs_rel}
    base.update(extra)
    return base


def grid(scale=1.0, scenes=(('s1', 'nyuv2'), ('s2', 'hypersim'), ('s3', 'kitti')), per_scene=5):
    return [row(f'{stem}/p{i}', stem, source, scale * (0.1 + 0.01 * i + 0.05 * k))
            for k, (stem, source) in enumerate(scenes) for i in range(per_scene)]


def test_constant_ratio_has_zero_width_ci():
    records, dropped = paired(grid(0.9), grid(1.0), 'abs_rel')
    result = clustered_bootstrap(records, n_boot=500)
    assert dropped == []
    assert abs(result['mean'] - math.log(0.9)) < 1e-12
    assert abs(result['ci_low'] - result['mean']) < 1e-12 and abs(result['ci_high'] - result['mean']) < 1e-12


def test_clusters_counted_by_scene():
    records, _ = paired(grid(0.9), grid(1.0), 'abs_rel')
    result = clustered_bootstrap(records, n_boot=200)
    assert result['n_clusters'] == 3 and result['n_cases'] == 15


def test_failed_and_missing_cases_reported():
    a, b = grid(0.9), grid(1.0)
    a[0] = dict(a[0], status='oom')
    b.append(row('s9/p0', 's9', 'void', 0.2))
    records, dropped = paired(a, b, 'abs_rel')
    assert len(records) == 14
    assert any('s1/p0' in d for d in dropped) and any('s9/p0' in d for d in dropped)


def test_seeds_averaged_within_run():
    a = [row('s1/p0', 's1', 'nyuv2', 0.1, seed=1), row('s1/p0', 's1', 'nyuv2', 0.4, seed=2)]
    b = [row('s1/p0', 's1', 'nyuv2', 0.2, seed=7)]
    records, _ = paired(a, b, 'abs_rel')
    assert len(records) == 1 and abs(records[0]['log_ratio'] - math.log(0.2 / 0.2)) < 1e-12  # geo-mean(0.1,0.4)=0.2


def test_group_ratio_and_delta1_error_rate():
    a = [r for r in grid(1.0)]
    for r in a:
        if r['source'] == 'hypersim':
            r['abs_rel'] *= 0.8
        r['delta1'] = 0.95
    b = grid(1.0)
    result = compare_rows(a, b, metrics=('abs_rel', 'delta1'), n_boot=200)
    groups = result['abs_rel']['groups']
    assert abs(groups['synthetic'] - 0.8) < 1e-9 and abs(groups['driving'] - 1.0) < 1e-9
    expected = math.log((1 - 0.95 + 1e-3) / (1 - 0.9 + 1e-3))
    assert abs(result['delta1']['mean'] - expected) < 1e-12
    assert set(GROUPS) == {'real_indoor', 'synthetic', 'driving'}


def test_ci_contains_mean_and_is_reproducible():
    a = grid(1.0)
    for i, r in enumerate(a):
        r['abs_rel'] *= 0.8 + 0.05 * (i % 4)
    records, _ = paired(a, grid(1.0), 'abs_rel')
    first, second = clustered_bootstrap(records, n_boot=1000, seed=3), clustered_bootstrap(records, n_boot=1000, seed=3)
    assert first == second
    assert first['ci_low'] <= first['mean'] <= first['ci_high']


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match='mystery'):
        compare_rows([row('x/p0', 'x', 'mystery', 0.1)], [row('x/p0', 'x', 'mystery', 0.2)], metrics=('abs_rel',))


def test_legacy_runs_without_case_column(tmp_path):
    import csv as _csv
    from hifidc.compare import load_rows
    path = tmp_path / "run"
    path.mkdir()
    with open(path / "metrics.csv", "w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=["stem", "source", "seed", "status", "abs_rel"])
        writer.writeheader()
        writer.writerow({"stem": "s1", "source": "nyuv2", "seed": 1, "status": "ok", "abs_rel": 0.1})
    rows = load_rows(path)
    assert rows[0]["case"] == "s1"
