"""Run DSINE once per scene on the original-grid RGB and cache unit normals (HxWx3 float32).

Keeps DSINE out of GPU memory during diffusion runs and makes local and Colab runs share the
exact same normal prior. Intrinsics: DSINE's nominal 60-degree FOV (paper parity).
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hifidc.data import load_manifest  # noqa: E402
from hifidc.dsine import default_dsine_paths, dsine_infer, dsine_root, load_local_dsine_model  # noqa: E402


def main(argv=None):
    ckpt, cfg = default_dsine_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data' / 'test'))
    parser.add_argument('--scenes', default=str(ROOT / 'configs' / 'scenes.yaml'))
    parser.add_argument('--out', default=str(ROOT / 'cache' / 'normals'))
    parser.add_argument('--dsine-ckpt', default=ckpt)
    parser.add_argument('--dsine-config', default=cfg)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_args = load_local_dsine_model(args.dsine_ckpt, args.dsine_config, device)
    written = {}
    for spec in load_manifest(args.scenes):
        image = Image.open(Path(args.data) / f'{spec.stem}_rgb.png').convert('RGB')
        output = dsine_infer(model, model_args, image, device)
        normals = F.normalize(output[:, :3], p=2, dim=1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        np.save(out / f'{spec.stem}_normals.npy', normals)
        written[spec.stem] = list(normals.shape)
        if output.shape[1] > 3:
            np.save(out / f'{spec.stem}_kappa.npy', output[0, 3].cpu().numpy().astype(np.float32))
        print(f'{spec.stem}: normals {normals.shape}, extra channels {output.shape[1] - 3}', flush=True)
    del model
    torch.cuda.empty_cache()
    commit = subprocess.run(['git', '-C', str(dsine_root()), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    manifest = {'dsine_commit': commit, 'checkpoint': args.dsine_ckpt,
                'checkpoint_sha256': hashlib.sha256(Path(args.dsine_ckpt).read_bytes()).hexdigest(),
                'config': args.dsine_config, 'fov_degrees': 60.0, 'grid': 'original RGB', 'scenes': written}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'wrote {len(written)} normal maps to {out}')


if __name__ == '__main__':
    main()
