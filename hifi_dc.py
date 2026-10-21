"""Clean publication-guided HiFi-DC entry point.

Preserves the original class name, keyword call arguments and NumPy return.
Requires hifi_core.py beside this file. Original experimental files are untouched.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, MarigoldDepthPipeline

# Supports old scripts loading this file by its filesystem path.
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
from hifi_core import Config, complete, resize_sparse


def load_local_dsine_model(ckpt, cfg, device):
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
    return normal[:, :3, top:top + h, left:left + w]


class DiffusionBackend:
    def __init__(self, pipeline, generator):
        self.pipe = pipeline
        self.generator = generator
        self.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
        for module in (pipeline.unet, pipeline.vae, pipeline.text_encoder):
            module.eval().requires_grad_(False)
        with torch.no_grad():
            ids = pipeline.tokenizer('', return_tensors='pt', padding='do_not_pad', truncation=True).input_ids.to(pipeline._execution_device)
            self.embedding = pipeline.text_encoder(ids)[0]

    def set_timesteps(self, steps):
        self.scheduler.set_timesteps(steps, device=self.pipe._execution_device)
        self.timesteps = self.scheduler.timesteps

    def encode_rgb(self, rgb):
        with torch.no_grad():
            distribution = self.pipe.vae.encode(rgb * 2 - 1).latent_dist
            return distribution.mode() * self.pipe.vae.config.scaling_factor

    def initial_latent(self, encoded):
        return torch.randn(encoded.shape, generator=self.generator, device=encoded.device, dtype=encoded.dtype)

    def noise(self, latent, timestep, encoded):
        return self.pipe.unet(torch.cat((encoded, latent), 1), timestep,
                              encoder_hidden_states=self.embedding, return_dict=False)[0]

    def predict_clean(self, latent, timestep, encoded):
        # Intentionally differentiable through frozen UNet and VAE to the latent.
        return self.scheduler.step(self.noise(latent, timestep, encoded), timestep, latent,
                                   eta=0., generator=self.generator).pred_original_sample

    def decode(self, latent):
        return self.pipe.decode_prediction(latent)

    def advance(self, latent, timestep, encoded):
        return self.scheduler.step(self.noise(latent, timestep, encoded), timestep, latent,
                                   eta=0., generator=self.generator).prev_sample


class MarigoldDepthCompletionPipelineWithNormals(MarigoldDepthPipeline):
    def __init__(self, unet, vae, scheduler, text_encoder, tokenizer,
                 dsine_model=None, dsine_args=None, scale_invariant=True,
                 shift_invariant=True, prediction_type='depth', default_denoising_steps=None,
                 default_processing_resolution=None):
        super().__init__(unet=unet, vae=vae, scheduler=scheduler, text_encoder=text_encoder,
                         tokenizer=tokenizer, prediction_type=prediction_type, scale_invariant=scale_invariant, shift_invariant=shift_invariant,
                         default_denoising_steps=default_denoising_steps,
                         default_processing_resolution=default_processing_resolution)
        self.dsine_model, self.dsine_args = dsine_model, dsine_args
        self.last_run = None

    @torch.enable_grad()
    def __call__(self, *, image, sparse_depth, num_inference_steps=None,
                 processing_resolution=768, seed=2024, normal_weight=None,
                 lambda_l1=None, lsq_interval=None, config=None, normal_prior=None,
                 return_details=False):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('Call HiFi-DC outside torch.inference_mode(); latent optimisation needs autograd')
        if not isinstance(image, Image.Image):
            raise TypeError('image must be a PIL Image')
        sparse_depth = np.asarray(sparse_depth)
        if sparse_depth.ndim != 2 or sparse_depth.shape != (image.height, image.width):
            raise ValueError('Sparse depth must match the input image grid')
        if processing_resolution < 16:
            raise ValueError('processing_resolution must be at least 16')
        c = replace(config) if config is not None else Config()
        # Existing call arguments keep their meaning; config supplies additional settings.
        for key, value in [('steps', num_inference_steps), ('normal_weight', normal_weight),
                           ('lambda_l1', lambda_l1), ('lsq_interval', lsq_interval)]:
            if value is not None:
                setattr(c, key, value)
        c.validate()
        output_unit = c.depth_unit
        unit_scale = 1000. if output_unit == 'mm' else 1.
        c.depth_unit = 'm'
        warnings.warn('Publication-guided reconstruction: see RECONSTRUCTION_NOTES.md for assumed settings; historical metrics not yet reproduced.', stacklevel=2)
        device = self._execution_device
        if self.unet.dtype != torch.float32 or self.vae.dtype != torch.float32:
            raise ValueError('This reconstruction is validated in float32; load the pipeline with torch_dtype=torch.float32')
        original = (image.height, image.width)
        ratio = processing_resolution / max(original)
        resized = tuple(max(1, round(n * ratio)) for n in original)
        padded = tuple(math_ceil16(n) for n in resized)
        rgb_image = image.convert('RGB').resize((resized[1], resized[0]), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(rgb_image).copy()).permute(2, 0, 1)[None].to(device).float() / 255
        padding = (0, padded[1] - resized[1], 0, padded[0] - resized[0])
        rgb = F.pad(rgb, padding, mode='replicate')
        sparse = torch.from_numpy(sparse_depth.astype(np.float32) / unit_scale).to(device)[None, None]
        sparse = F.pad(resize_sparse(sparse, resized), padding)
        normal = None
        if c.use_normals:
            if normal_prior is None:
                if self.dsine_model is None:
                    raise ValueError('Provide a DSINE model or normal_prior array')
                self.dsine_model.eval().requires_grad_(False)
                # DSINE sees RGB in its expected range, never the VAE [-1,1] tensor.
                normal = dsine_infer_normals(self.dsine_model, self.dsine_args, rgb_image, device)
            else:
                normal = torch.as_tensor(normal_prior, dtype=torch.float32, device=device)
                if normal.ndim == 3 and normal.shape[-1] == 3:
                    normal = normal.permute(2, 0, 1)
                if normal.ndim == 3:
                    normal = normal[None]
                if normal.shape != (1, 3, *original):
                    raise ValueError('normal_prior must match the original RGB grid, CHW or HWC')
                normal = F.interpolate(normal, size=resized, mode='bilinear', align_corners=False)
            normal = F.normalize(F.pad(normal, padding, mode='replicate'), p=2, dim=1)
        backend = DiffusionBackend(self, torch.Generator(device=device).manual_seed(seed))
        result = complete(backend, rgb, sparse, normal, c)
        for name in ('depth', 'uncertainty'):
            value = result[name][..., :resized[0], :resized[1]]
            result[name] = F.interpolate(value, size=original, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        result['depth'] *= unit_scale
        result.update({'seed': seed, 'processing_resolution': processing_resolution,
                       'original_shape': original, 'prepared_shape': padded,
                       'depth_unit': output_unit, 'internal_depth_unit': 'm'})
        self.last_run = {k: v for k, v in result.items() if k not in ('depth', 'uncertainty')}
        return result if return_details else result['depth']


def math_ceil16(value):
    return (value + 15) // 16 * 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image', default='data/image.png')
    parser.add_argument('--in-depth', default='data/sparse_1000.npy')
    parser.add_argument('--out-depth', default='data/dense_depth_new_1000.npy')
    parser.add_argument('--checkpoint', default='prs-eth/marigold-depth-v1-0')
    parser.add_argument('--checkpoint-revision', default=None, help='Optional immutable Hugging Face model revision')
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument('--processing_resolution', type=int, default=768)
    parser.add_argument('--normal_weight', type=float, default=None)
    parser.add_argument('--lambda_l1', type=float, default=None)
    parser.add_argument('--lsq-interval', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2024)
    base = '/content/drive/MyDrive/ColabNotebooks/marigold_dsine/projects/dsine/'
    parser.add_argument('--dsine-ckpt', default=base + 'checkpoints/exp002_kappa/dsine_plus_scannet.pt')
    parser.add_argument('--dsine-config', default=base + 'experiments/exp002_kappa/dsine_plus_scannet.txt')
    parser.add_argument('--normal-prior', help='Optional aligned DSINE normal .npy, HWC or CHW')
    parser.add_argument('--config', help='JSON containing additional Config fields')
    parser.add_argument('--depth-unit', choices=['m','mm'], default=None, help='Input/output depth unit; optimisation uses metres')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing existing prediction files')
    args = parser.parse_args()
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
                                 for p in (Path(__file__), Path(__file__).with_name('hifi_core.py'))})
    path.with_name(path.stem + '_run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved depth, uncertainty and run manifest beside {path}')


if __name__ == '__main__':
    main()
