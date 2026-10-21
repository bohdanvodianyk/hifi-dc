"""Paired comparison of two benchmark runs, with a scene-clustered bootstrap.

Cases are paired by case id (seeds within a run are averaged geometrically). The effect is the
log ratio A/B of a metric per case (for delta1: of the error rate 1 - delta1 + 1e-3); lower is
better for A. Scenes are the clusters: the estimate is the mean over scenes of the per-scene mean,
and the bootstrap resamples scenes, then cases within each drawn scene.
"""
import csv
import math
from pathlib import Path

import numpy as np

GROUPS = {'real_indoor': {'nyuv2', 'void'},
          'synthetic': {'hypersim', 'interiornet', 'icl_nuim', 'sintel'},
          'driving': {'kitti', 'vkitti2'}}
DEFAULT_METRICS = ('abs_rel', 'si_log_rmse', 'delta1', 'holdout_residual')
EPS = 1e-3


def group_of(source):
    for name, sources in GROUPS.items():
        if source in sources:
            return name
    raise ValueError(f'source {source!r} is not in any comparison group')


def load_rows(run_dir):
    with open(Path(run_dir) / 'metrics.csv', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:  # runs written before the `case` column: one case per scene
        if not r.get('case'):
            r['case'] = r['stem']
    return rows


def _value(row, metric):
    raw = row.get(metric, '')
    if raw in ('', None):
        return None
    value = float(raw)
    if metric == 'delta1':
        value = 1.0 - value + EPS
    return value if value > 0 and math.isfinite(value) else None


def _per_case(rows, metric, label, dropped):
    values, info = {}, {}
    for r in rows:
        case = r['case']
        if r['status'] != 'ok':
            dropped.append(f"{case}: status {r['status']} in {label}")
            continue
        value = _value(r, metric)
        if value is None:
            dropped.append(f'{case}: no usable {metric} in {label}')
            continue
        values.setdefault(case, []).append(math.log(value))
        info[case] = (r['stem'], r['source'])
    return {c: float(np.mean(v)) for c, v in values.items()}, info


def paired(rows_a, rows_b, metric):
    """-> (records [{case, stem, source, log_ratio}], dropped [str])."""
    dropped = []
    a, info_a = _per_case(rows_a, metric, 'A', dropped)
    b, info_b = _per_case(rows_b, metric, 'B', dropped)
    for case in sorted(set(a) ^ set(b)):
        dropped.append(f"{case}: missing in {'B' if case in a else 'A'}")
    records = [{'case': c, 'stem': info_a[c][0], 'source': info_a[c][1], 'log_ratio': a[c] - b[c]}
               for c in sorted(set(a) & set(b))]
    return records, sorted(set(dropped))


def _scene_means(by_scene):
    return float(np.mean([np.mean(v) for v in by_scene.values()]))


def clustered_bootstrap(records, n_boot=10000, seed=0):
    by_scene = {}
    for r in records:
        by_scene.setdefault(r['stem'], []).append(r['log_ratio'])
    scenes = [np.asarray(v) for v in by_scene.values()]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        drawn = rng.integers(len(scenes), size=len(scenes))
        boots[i] = np.mean([scenes[j][rng.integers(len(scenes[j]), size=len(scenes[j]))].mean() for j in drawn])
    return {'mean': _scene_means(by_scene), 'ci_low': float(np.percentile(boots, 2.5)),
            'ci_high': float(np.percentile(boots, 97.5)), 'n_cases': len(records), 'n_clusters': len(scenes)}


def compare_rows(rows_a, rows_b, metrics=DEFAULT_METRICS, n_boot=10000, seed=0):
    result, dropped = {}, set()
    for metric in metrics:
        records, lost = paired(rows_a, rows_b, metric)
        dropped.update(lost)
        if not records:
            continue
        stats = clustered_bootstrap(records, n_boot=n_boot, seed=seed)
        groups = {}
        for name in GROUPS:
            by_scene = {}
            for r in records:
                if group_of(r['source']) == name:
                    by_scene.setdefault(r['stem'], []).append(r['log_ratio'])
            if by_scene:
                groups[name] = math.exp(_scene_means(by_scene))
        for r in records:
            group_of(r['source'])
        stats.update(ratio=math.exp(stats['mean']), ratio_ci=[math.exp(stats['ci_low']), math.exp(stats['ci_high'])],
                     groups=groups)
        result[metric] = stats
    result['dropped'] = sorted(dropped)
    return result


def compare(run_a, run_b, metrics=DEFAULT_METRICS, n_boot=10000, seed=0):
    return compare_rows(load_rows(run_a), load_rows(run_b), metrics, n_boot, seed)
