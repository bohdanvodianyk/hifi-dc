"""Compare two benchmark runs case by case (A relative to B; ratio < 1 means A is better).

  python scripts/compare_runs.py results/v2_L1_dev50_384 results/ref_mdc_384_s2024 [--out compare.json]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hifidc.compare import DEFAULT_METRICS, compare  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_a')
    parser.add_argument('run_b')
    parser.add_argument('--metrics', nargs='+', default=list(DEFAULT_METRICS))
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    result = compare(args.run_a, args.run_b, args.metrics, args.n_boot)
    print(f'A = {args.run_a}\nB = {args.run_b}   (ratio A/B; < 1 = A better; delta1 compares error rates)')
    for metric in args.metrics:
        if metric not in result:
            print(f'{metric:17s} no paired cases')
            continue
        r = result[metric]
        groups = '  '.join(f'{g}={v:.3f}' for g, v in r['groups'].items())
        print(f"{metric:17s} ratio={r['ratio']:.3f}  95% CI [{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}]  "
              f"cases={r['n_cases']} scenes={r['n_clusters']}  | {groups}")
    if result['dropped']:
        print(f"dropped ({len(result['dropped'])}): " + '; '.join(result['dropped']))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
