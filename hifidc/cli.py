"""Command-line inference for one image (legacy `python hifi_dc.py ...` interface)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .config import Config
from .dsine import default_dsine_paths, load_local_dsine_model
from .pipeline import MarigoldDepthCompletionPipelineWithNormals

MARIGOLD_CHECKPOINT = 'prs-eth/marigold-depth-v1-0'
MARIGOLD_REVISION = 'f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c'
PACKAGE_DIR = Path(__file__).resolve().parent


def main(argv=None):
    dsine_ckpt, dsine_cfg = default_dsine_paths()
    parser = argparse.ArgumentParser(description='HiFi-DC depth completion (corrected v1 reconstruction)')
    parser.add_argument('--in-image', default='data/image.png')
    parser.add_argument('--in-depth', default='data/sparse_1000.npy')
    parser.add_argument('--out-depth', default='data/dense_depth_new_1000.npy')
    parser.add_argument('--checkpoint', default=MARIGOLD_CHECKPOINT)
    parser.add_argument('--checkpoint-revision', default=MARIGOLD_REVISION, help='Immutable Hugging Face model revision')
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument('--processing_resolution', type=int, default=768)
    parser.add_argument('--normal_weight', type=float, default=None)
    parser.add_argument('--lambda_l1', type=float, default=None)
    parser.add_argument('--lsq-interval', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dsine-ckpt', default=dsine_ckpt, help='Default: $HIFIDC_DSINE_ROOT/' + 'projects/dsine/checkpoints/...')
    parser.add_argument('--dsine-config', default=dsine_cfg)
    parser.add_argument('--normal-prior', help='Optional aligned DSINE normal .npy, HWC or CHW')
    parser.add_argument('--config', help='JSON containing additional Config fields')
    parser.add_argument('--depth-unit', choices=['m', 'mm'], default=None, help='Input/output depth unit; optimisation uses metres')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing existing prediction files')
    args = parser.parse_args(argv)
    path = Path(args.out_depth)
    outputs = [path if path.suffix == '.npy' else Path(str(path) + '.npy'),
               path.with_name(path.stem + '_uncertainty.npy'), path.with_name(path.stem + '_run.json')]
    if not args.overwrite and any(p.exists() for p in outputs):
        raise FileExistsError('Output already exists; choose a new directory or pass --overwrite explicitly')
    config = Config(**json.loads(Path(args.config).read_text())) if args.config else Config()
    if args.depth_unit:
        config.depth_unit = args.depth_unit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal = np.load(args.normal_prior, allow_pickle=False) if args.normal_prior else None
    dsine, dsine_args = (None, None)
    if normal is None and config.use_normals:
        dsine, dsine_args = load_local_dsine_model(args.dsine_ckpt, args.dsine_config, device)
    pipe = MarigoldDepthCompletionPipelineWithNormals.from_pretrained(
        args.checkpoint, revision=args.checkpoint_revision, dsine_model=dsine, dsine_args=dsine_args, torch_dtype=torch.float32).to(device)
    result = pipe(image=Image.open(args.in_image).convert('RGB'), sparse_depth=np.load(args.in_depth, allow_pickle=False),
                  num_inference_steps=args.num_inference_steps, processing_resolution=args.processing_resolution,
                  normal_weight=args.normal_weight, lambda_l1=args.lambda_l1, lsq_interval=args.lsq_interval,
                  seed=args.seed, config=config, normal_prior=normal, return_details=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result['depth'])
    np.save(path.with_name(path.stem + '_uncertainty.npy'), result['uncertainty'])
    import diffusers
    manifest = dict(pipe.last_run, checkpoint=args.checkpoint, checkpoint_revision=args.checkpoint_revision,
                    torch_version=torch.__version__,
                    diffusers_version=diffusers.__version__, device=str(device),
                    gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                    input_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in [args.in_image, args.in_depth] + ([args.normal_prior] if args.normal_prior else [])},
                    dsine_files={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in [args.dsine_ckpt, args.dsine_config]
                                 if normal is None and config.use_normals and Path(p).is_file()},
                    code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(PACKAGE_DIR.glob('*.py'))})
    path.with_name(path.stem + '_run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved depth, uncertainty and run manifest beside {path}')


if __name__ == '__main__':
    main()
