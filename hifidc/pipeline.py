"""Diffusers integration: frozen Marigold backend and the legacy-compatible pipeline class."""
from dataclasses import replace
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, MarigoldDepthPipeline

from .config import Config
from .core import complete, resize_sparse
from .dsine import dsine_infer_normals


class DiffusionBackend:
    def __init__(self, pipeline, generator, embedding=None):
        self.pipe = pipeline
        self.generator = generator
        self.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
        for module in (pipeline.unet, pipeline.vae, pipeline.text_encoder):
            if module is not None:
                module.eval().requires_grad_(False)
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = empty_text_embedding(pipeline)

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


def empty_text_embedding(pipeline):
    """Empty-prompt CLIP embedding used as fixed UNet conditioning."""
    with torch.no_grad():
        ids = pipeline.tokenizer('', return_tensors='pt', padding='do_not_pad', truncation=True).input_ids.to(pipeline._execution_device)
        return pipeline.text_encoder(ids)[0]


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
        self.cached_embedding = None

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
        backend = DiffusionBackend(self, torch.Generator(device=device).manual_seed(seed),
                                   embedding=getattr(self, 'cached_embedding', None))
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

    def free_text_encoder(self):
        """Cache the empty-prompt embedding, then drop the CLIP text encoder (~1.4 GB fp32)."""
        if self.cached_embedding is None:
            self.cached_embedding = empty_text_embedding(self)
        self.text_encoder = None
        return self.cached_embedding


def math_ceil16(value):
    return (value + 15) // 16 * 16
