"""Pick random test / dev scenes from the Drive dataset folders and import them for run_benchmark.

  python scripts/import_drive_datasets.py --root /content/drive/MyDrive/.../main_datasets_new \
      --out /content/newdata --n-test 10 --n-dev 3 --seed 2026 [--selection saved/selection.json]

With --selection, the saved split is re-imported exactly (the test set stays frozen across sessions).
Writes <out>/data/, <out>/scenes_test.yaml, <out>/scenes_dev.yaml and <out>/selection.json.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hifidc.data import load_manifest  # noqa: E402
from hifidc.drive_import import (DATASETS, dev_sequences, import_selection, list_dataset,  # noqa: E402
                                 select_cases, sequence_key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='folder holding output_<dataset>/ folders')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-test', type=int, default=10, help='test scenes per dataset')
    parser.add_argument('--n-dev', type=int, default=3, help='extra development scenes per dataset')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--datasets', nargs='*', default=None, help='default: every output_<name> with known units')
    parser.add_argument('--selection', default=None, help='re-import this saved selection.json instead of drawing')
    parser.add_argument('--exclude-scenes', default=str(ROOT / 'configs' / 'scenes.yaml'),
                        help='scenes whose sequences must not be selected (the development set)')
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.selection:
        saved = json.loads(Path(args.selection).read_text())
        selection = {'test': [tuple(x) for x in saved['test']], 'dev': [tuple(x) for x in saved['dev']],
                     'short': saved.get('short', {})}
        print(f'reusing saved selection {args.selection}')
    else:
        names = args.datasets or sorted(p.name[len('output_'):] for p in root.glob('output_*') if p.is_dir())
        unknown = [n for n in names if n not in DATASETS]
        if unknown:
            print(f'skipping datasets without unit defaults: {unknown}')
        names = [n for n in names if n in DATASETS]
        exclude = dev_sequences(load_manifest(args.exclude_scenes))
        stems = {}
        for name in names:
            pairs = list_dataset(root, name)
            stems[name] = sorted(pairs)
            source = DATASETS[name]['source']
            sequences = Counter(sequence_key(source, s) for s in pairs)
            print(f'{name:12s} {len(pairs):6d} complete scenes in {len(sequences):5d} sequences')
        selection = select_cases(stems, args.n_test, args.n_dev, args.seed, exclude)
    for name, message in selection.get('short', {}).items():
        print(f'NOTE {name}: {message}')
    paths = import_selection(root, selection, args.out)
    record = json.loads((Path(args.out) / 'selection.json').read_text())
    record.update(seed=args.seed, n_test=args.n_test, n_dev=args.n_dev, root=str(root))
    (Path(args.out) / 'selection.json').write_text(json.dumps(record, indent=1))
    for stem, error in ((s, e) for _, s, e in record['skipped']):
        print(f'SKIPPED {stem}: {error}')
    print(f"\n{'dataset':12s} {'split':5s} {'stem':45s} {'HxW':>10s} {'GT fill':>7s} {'GT min/median/max (native)':>30s}")
    for stem, s in record['stats'].items():
        shape = 'x'.join(map(str, s['shape']))
        print(f"{s['dataset']:12s} {s['split']:5s} {stem[:45]:45s} {shape:>10s} {s['gt_fill']:7.1%} "
              f"{s['gt_min']:>9.4g} / {s['gt_median']:<8.4g} / {s['gt_max']:<9.4g}")
    counts = Counter((s['split'], s['dataset']) for s in record['stats'].values())
    print('\nimported:', {split: sum(v for (sp, _), v in counts.items() if sp == split) for split in ('test', 'dev')},
          '->', {k: str(v) for k, v in paths.items()})


if __name__ == '__main__':
    main()
