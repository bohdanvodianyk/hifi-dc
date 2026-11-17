"""Edge quality of finished runs, from their saved predictions (no GPU): sharpness and edge AbsRel.

  python scripts/edge_runs.py RUN_MDC RUN_A RUN_B --labels mdc a b --scenes S.yaml --data DATA [--span-grids 384 768]

Per run, writes <run>/edge_metrics.csv (reused when it has the columns) with, per case, edge_abs_rel
and edge_sharpness_<grid> for each span grid (hifidc.metrics.edge_metrics; dense-GT scenes only,
others are NaN). Prints the scene-mean sharpness per run (1 = as sharp as GT) and the paired
comparison of every run with the first one (scene-clustered bootstrap, hifidc.compare): for
edge_abs_rel < 1 is better, for edge_sharpness > 1 is sharper.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hifidc.compare import compare_rows, load_rows  # noqa: E402
from hifidc.data import load_manifest, load_scene  # noqa: E402
from hifidc.metrics import edge_metrics  # noqa: E402

KEEP = ('stem', 'source', 'seed', 'case', 'status')


def columns(grids):
    return ['edge_abs_rel'] + [f'edge_sharpness_{g}' for g in grids]


def edge_rows(run, specs, data, grids):
    """Per-case edge metrics of one run (cached in <run>/edge_metrics.csv)."""
    cache = Path(run) / 'edge_metrics.csv'
    if cache.is_file():
        with open(cache, newline='') as handle:
            rows = list(csv.DictReader(handle))
        if rows and all(c in rows[0] for c in columns(grids)):
            return rows
    out, scene, stem = [], None, None
    for r in sorted(load_rows(run), key=lambda r: r['stem']):
        row = {k: r[k] for k in KEEP}
        if r['status'] == 'ok':
            if r['stem'] != stem:
                stem, scene = r['stem'], load_scene(data, specs[r['stem']])
            dense = scene.valid_gt.mean() >= 0.9
            pred = np.load(Path(run) / 'pred' / f"{r['case'].replace('/', '__')}_s{r['seed']}.npy") if dense else None
            for g in grids:
                m = (edge_metrics(pred, scene.gt, scene.valid_gt, span_grid=g) if dense
                     else {'edge_abs_rel': math.nan, 'edge_sharpness': math.nan})
                row['edge_abs_rel'] = m['edge_abs_rel']       # does not depend on the span grid
                row[f'edge_sharpness_{g}'] = m['edge_sharpness']
        out.append(row)
    with open(cache, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(KEEP) + columns(grids))
        writer.writeheader()
        writer.writerows(out)
    return [{k: str(v) for k, v in row.items()} for row in out]


def scene_mean(rows, key):
    by_scene = {}
    for r in rows:
        v = float(r[key]) if r.get(key) not in ('', None) else math.nan
        if r['status'] == 'ok' and math.isfinite(v):
            by_scene.setdefault(r['stem'], []).append(v)
    return (float(np.mean([np.mean(v) for v in by_scene.values()])) if by_scene else math.nan), len(by_scene)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run dirs; the first is the reference')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--scenes', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--span-grids', type=int, nargs='+', default=[384])
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit('--labels must match the runs')
    specs = {s.stem: s for s in load_manifest(args.scenes)}
    rows = {label: edge_rows(run, specs, args.data, args.span_grids) for label, run in zip(labels, args.runs)}
    keys = columns(args.span_grids)
    result = {'span_grids': args.span_grids, 'means': {}, 'vs_ref': {}}
    print('scene-mean edge metrics (dense-GT scenes; sharpness 1 = as sharp as GT)')
    for label in labels:
        means = {k: scene_mean(rows[label], k) for k in keys}
        result['means'][label] = {k: v[0] for k, v in means.items()}
        print(f'  {label:14s} ' + '  '.join(f'{k}={v[0]:.3f}' for k, v in means.items()) +
              f'  scenes={means[keys[0]][1]}')
    ref = labels[0]
    for label in labels[1:]:
        r = compare_rows(rows[label], rows[ref], keys, args.n_boot)
        result['vs_ref'][label] = r
        print(f'### {label} vs {ref}  (edge_abs_rel < 1 better; edge_sharpness > 1 sharper)')
        for k in keys:
            if k in r:
                groups = '  '.join(f'{g}={v:.3f}' for g, v in r[k]['groups'].items())
                print(f"  {k:20s} ratio={r[k]['ratio']:.3f}  95% CI [{r[k]['ratio_ci'][0]:.3f}, "
                      f"{r[k]['ratio_ci'][1]:.3f}]  cases={r[k]['n_cases']} scenes={r[k]['n_clusters']}  | {groups}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
