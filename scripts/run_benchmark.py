"""Run a method over the evaluation scenes and write results/<run_id>/.

  python scripts/run_benchmark.py --method hifidc --resolution 384 --seeds 2024 --out results/local_hifidc_384
  python scripts/run_benchmark.py --method hifidc --override learning_rate=1e-12 --out results/noguide_384
  python scripts/run_benchmark.py --method marigold_dc --resolution 768 --out results/mdc_768
"""
import argparse
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hifidc.bench import (configure_determinism, make_hifidc_method, make_hifidc_v2_method,  # noqa: E402
                          make_marigold_dc_method,
                          run_benchmark, sha256_json)
from hifidc.config import load_config, load_dataclass, parse_overrides  # noqa: E402
from hifidc.v2.config import V2Config  # noqa: E402
from hifidc.cases import dev50, stress48  # noqa: E402
from hifidc.data import load_manifest, load_scene  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=['hifidc', 'hifidc_v2', 'marigold_dc', 'marigold_dc_norm'], required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'paper_v1.yaml'))
    parser.add_argument('--override', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--v2-config', default=str(ROOT / 'configs' / 'v2' / 'L0.yaml'),
                        help='V2Config YAML for --method hifidc_v2 (overrides apply to it)')
    parser.add_argument('--mdc-steps', type=int, default=50, help='Marigold-DC denoising steps')
    parser.add_argument('--scenes', default=str(ROOT / 'configs' / 'scenes.yaml'))
    parser.add_argument('--data', default=str(ROOT / 'data' / 'test'))
    parser.add_argument('--normals', default=str(ROOT / 'cache' / 'normals'))
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024])
    parser.add_argument('--cases', choices=['provided', 'dev50', 'stress48'], default='provided',
                        help='provided sparse inputs, or reproducible resampled cases (hifidc.cases)')
    parser.add_argument('--only', nargs='*', default=None, help='Restrict to these scene stems')
    parser.add_argument('--out', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--fast-attention', action='store_true',
                        help='Keep fused SDPA kernels (less memory, but runs are not bit-repeatable)')
    args = parser.parse_args(argv)
    if args.fast_attention:
        os.environ['HIFIDC_DETERMINISTIC_ATTENTION'] = '0'
    configure_determinism()

    specs = load_manifest(args.scenes)
    if args.only:
        unknown = set(args.only) - {s.stem for s in specs}
        if unknown:
            raise SystemExit(f'unknown scenes: {sorted(unknown)}')
        specs = [s for s in specs if s.stem in args.only]
    scenes = [load_scene(args.data, s) for s in specs]
    if args.cases == 'dev50':
        scenes = dev50(scenes)
    elif args.cases == 'stress48':
        scenes = stress48(scenes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.method == 'hifidc':
        config = load_config(args.config, parse_overrides(args.override))
        method = make_hifidc_method(config, device)
        normals_dir = Path(args.normals) if config.use_normals else None
        if normals_dir is not None:
            missing = [s.stem for s in specs if not (normals_dir / f'{s.stem}_normals.npy').is_file()]
            if missing:
                raise SystemExit(f'missing cached normals for {missing}; run scripts/cache_normals.py first')
    elif args.method == 'hifidc_v2':
        v2_config = load_dataclass(V2Config, args.v2_config, parse_overrides(args.override))
        normals_dir = Path(args.normals) if Path(args.normals).is_dir() else None
        method = make_hifidc_v2_method(v2_config, device, normals_dir=normals_dir)
    else:
        method = make_marigold_dc_method(device, steps=args.mdc_steps, normalize=args.method == 'marigold_dc_norm')
        normals_dir = None
    meta = dict(method.meta, overrides=args.override,
                config_file=args.v2_config if args.method == 'hifidc_v2' else args.config,
                scenes_file_sha256=sha256_json(Path(args.scenes).read_text()))
    rows = run_benchmark(method, scenes, normals_dir, args.seeds, args.resolution, Path(args.out), meta,
                         overwrite=args.overwrite)
    failed = [r for r in rows if r['status'] != 'ok']
    print(f'done: {len(rows) - len(failed)}/{len(rows)} ok -> {args.out}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
