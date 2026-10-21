"""Write one side-by-side PNG per case comparing benchmark runs.

Example:
  python scripts/visualize_runs.py results/ref_mdc_384_s2024 results/v2_L0_dev50_384 \
      --labels Marigold-DC "HiFi-DC v2 L1" --out results/viz/mdc_vs_L1

Each figure shows RGB, the sparse input and GT on the top row, each run's depth below (same
colour range per case, near = red), and each run's relative-error map (0-20 %) at the bottom.
The runs must share a case set (same --cases); predictions are read from <run>/pred/.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hifidc.cases import dev50, stress48, stress_dense  # noqa: E402
from hifidc.data import load_manifest, load_scene  # noqa: E402
from hifidc.viz import comparison_figure  # noqa: E402


def _cases_kind(run_dir):
    argv = json.loads((Path(run_dir) / 'manifest.json').read_text()).get('argv', [])
    return argv[argv.index('--cases') + 1] if '--cases' in argv else 'provided'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories (results/<run>)')
    parser.add_argument('--labels', nargs='+', help='one label per run (default: run directory names)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--match', default=None, help='regex on the case id, e.g. "p0$" or "office_4"')
    parser.add_argument('--panel-width', type=int, default=400)
    parser.add_argument('--max-rel', type=float, default=0.2, help='error map saturates at this relative error')
    parser.add_argument('--scenes', default=str(ROOT / 'configs' / 'scenes.yaml'))
    parser.add_argument('--data', default=str(ROOT / 'data' / 'test'))
    args = parser.parse_args(argv)

    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit('need one --labels entry per run')
    kinds = {_cases_kind(r) for r in args.runs}
    if len(kinds) != 1:
        raise SystemExit(f'runs use different case sets: {sorted(kinds)}')
    kind = kinds.pop()
    scenes = [load_scene(args.data, s) for s in load_manifest(args.scenes)]
    cases = {'dev50': dev50, 'stress48': stress48, 'stress': stress_dense}.get(kind, lambda s: s)(scenes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for case in cases:
        case_id = case.case_id or case.spec.stem
        if args.match and not re.search(args.match, case_id):
            continue
        name = case_id.replace('/', '__')
        files = [Path(r) / 'pred' / f'{name}_s{args.seed}.npy' for r in args.runs]
        if not all(f.exists() for f in files):
            print(f'skip {case_id}: missing prediction in {[str(f.parent.parent) for f in files if not f.exists()]}')
            continue
        preds = {label: np.load(f) for label, f in zip(labels, files)}
        figure = comparison_figure(case.rgb, case.sparse, case.gt, case.valid_gt, preds,
                                   title=f'{case_id}  ({case.spec.source})', panel_width=args.panel_width,
                                   max_rel=args.max_rel)
        Image.fromarray(figure).save(out / f'{name}.png', optimize=True)
        written += 1
    print(f'wrote {written} figures to {out}')


if __name__ == '__main__':
    main()
