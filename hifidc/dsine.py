"""DSINE surface-normal prior (external project, pinned by scripts/setup_external.sh).

The DSINE source tree is located through ``$HIFIDC_DSINE_ROOT`` (default: ``external/DSINE``
next to this package) and added to ``sys.path`` only when a DSINE function is called.
"""
import importlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DSINE_ROOT = Path(__file__).resolve().parent.parent / 'external' / 'DSINE'
DSINE_CKPT = 'projects/dsine/checkpoints/exp002_kappa/dsine_plus_scannet.pt'
DSINE_CONFIG = 'projects/dsine/experiments/exp002_kappa/dsine_plus_scannet.txt'


def dsine_root():
    return Path(os.environ.get('HIFIDC_DSINE_ROOT', DEFAULT_DSINE_ROOT))


def default_dsine_paths():
    root = dsine_root()
    return str(root / DSINE_CKPT), str(root / DSINE_CONFIG)


def _ensure_import_path():
    root = str(dsine_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_local_dsine_model(ckpt, cfg, device):
    _ensure_import_path()
    from projects.dsine import config as dsine_config
    from utils import utils as dsine_utils
    args = dsine_config.get_args(test=True, config_path=cfg)
    arch = args.NNET_architecture
    classes = {'v00': 'DSINE_v00', 'v01': 'DSINE_v01', 'v02': 'DSINE_v02', 'v02_kappa': 'DSINE_v02_kappa'}
    if arch not in classes:
        raise ValueError(f'Unsupported DSINE architecture: {arch}')
    model = getattr(importlib.import_module(f'models.dsine.{arch}'), classes[arch])(args).to(device)
    model = dsine_utils.load_checkpoint(ckpt, model)
    model.eval().requires_grad_(False)
    return model, args


def dsine_infer_normals(dsine_model, dsine_args, pil_img, device):
    """Normals [1,3,H,W] on the input grid (60-degree nominal FOV)."""
    return dsine_infer(dsine_model, dsine_args, pil_img, device)[:, :3]


def dsine_infer(dsine_model, dsine_args, pil_img, device):
    """Full final DSINE output [1,C,H,W]: channels 0-2 normals, channel 3 kappa for v02_kappa."""
    _ensure_import_path()
    from utils import utils as dsine_utils
    from utils.projection import intrins_from_fov
    image = torch.from_numpy(np.asarray(pil_img.convert('RGB')).copy()).permute(2, 0, 1)[None].to(device).float() / 255
    h, w = image.shape[-2:]
    pad = dsine_utils.get_padding(h, w)
    left, right, top, bottom = pad
    image = F.pad(image, pad, value=0)
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    intrinsics = intrins_from_fov(60., h, w, device=device).unsqueeze(0)
    intrinsics[:, 0, 2] += left
    intrinsics[:, 1, 2] += top
    with torch.no_grad():
        normal = dsine_model((image - mean) / std, intrins=intrinsics)[-1]
    return normal[:, :, top:top + h, left:left + w]
