"""Apply the L6 guided filter to a finished run's predictions and write a new, comparable run.

  python scripts/postfilter_run.py results/ref_mdcnorm_384_s2024 results/v2_L1F_dev50_384 [--radius 2 --eps 1e-3]

Rebuilds the run's cases (same --cases as the source run), filters every saved prediction, and writes
pred/, metrics.csv (metrics recomputed; seconds include the filter time), summary.json and a manifest
that records the source run and the filter settings. No GPU needed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hifidc.bench import _summary, _write_csv  # noqa: E402
from hifidc.cases import dev50, stress48, stress_dense  # noqa: E402
from hifidc.data import load_manifest, load_scene  # noqa: E402
from hifidc.metrics import anchor_residual, depth_metrics  # noqa: E402
from hifidc.v2.postfilter import guided_log_depth  # noqa: E402

CASE_SETS = {'dev50': dev50, 'stress48': stress48, 'stress': stress_dense}
FLOATS = ('seconds', 'peak_vram_gb')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source')
    parser.add_argument('out')
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--scenes', default=str(ROOT / 'configs' / 'scenes.yaml'))
    parser.add_argument('--data', default=str(ROOT / 'data' / 'test'))
    args = parser.parse_args(argv)

    source, out = Path(args.source), Path(args.out)
    manifest = json.loads((source / 'manifest.json').read_text())
    run_argv = manifest.get('argv', [])
    kind = run_argv[run_argv.index('--cases') + 1] if '--cases' in run_argv else 'provided'
    scenes = [load_scene(args.data, s) for s in load_manifest(args.scenes)]
    cases = {c.case_id or c.spec.stem: c for c in CASE_SETS.get(kind, lambda s: s)(scenes)}

    (out / 'pred').mkdir(parents=True, exist_ok=True)
    rows = []
    with open(source / 'metrics.csv', newline='') as handle:
        for old in csv.DictReader(handle):
            row = {k: old[k] for k in ('stem', 'source', 'seed', 'case', 'status', 'message', 'unet_calls')}
            row.update(unit_uncertain=old['unit_uncertain'] == 'True', outdoor=old['outdoor'] == 'True')
            for key in FLOATS:
                row[key] = float(old[key]) if old.get(key) else ''
            if old['status'] == 'ok':
                case = cases[old['case']]
                name = f"{old['case'].replace('/', '__')}_s{old['seed']}.npy"
                start = time.perf_counter()
                pred = guided_log_depth(np.load(source / 'pred' / name), case.rgb, args.radius, args.eps)
                row['seconds'] = (row['seconds'] or 0.0) + time.perf_counter() - start
                np.save(out / 'pred' / name, pred)
                row.update(depth_metrics(pred, case.gt, case.valid_gt))
                row['anchor_residual'] = anchor_residual(pred, case.sparse)
                if case.holdout is not None and (case.holdout > 0).any():
                    row['holdout_residual'] = anchor_residual(pred, case.holdout)
            rows.append(row)
    _write_csv(out / 'metrics.csv', rows)
    (out / 'summary.json').write_text(json.dumps(_summary(rows), indent=2))
    manifest.update(derived_from=str(source), post_filter={'kind': 'guided', 'radius': args.radius, 'eps': args.eps})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(f'filtered {sum(r["status"] == "ok" for r in rows)}/{len(rows)} predictions -> {out}')


if __name__ == '__main__':
    main()
