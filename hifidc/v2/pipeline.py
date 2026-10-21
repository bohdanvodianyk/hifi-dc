"""HiFi-DC v2 solver.

Rung L0 reproduces the official Marigold-DC test-time guidance operation for operation
(prs-eth/Marigold-DC @ c6d756b, ``marigold_dc.py``, Apache-2.0, Viola, Qu, Metzger, Obukhov):
scale/shift parametrised as squares times the sparse range/minimum and optimised jointly with the
depth latent by Adam; the latent gradient is rescaled to the norm of the predicted noise; L1 + L2
loss at the sparse points on the original grid; the DDIM step reuses the pre-update noise
prediction (one UNet call per step). Later rungs add terms through the small methods below; with
default ``V2Config`` values the computation is identical to L0.
"""
import numpy as np
from PIL import Image
import torch
from diffusers import MarigoldDepthPipeline

from .config import V2Config
from .geometry import intrinsics_matrix, normal_guidance_loss
from .field import evaluate_field, fit_residual_field
from .robust import robust_l1l2, robust_weights
from .units import normalize_sparse


class HiFiDCv2Pipeline(MarigoldDepthPipeline):

    def __call__(self, *, image, sparse_depth, config=None, seed=2024, processing_resolution=768,
                 normals=None, kappa=None, intrinsics=None, return_details=False):
        config = (config or V2Config()).validate()
        if not isinstance(image, Image.Image):
            raise TypeError('image must be a PIL Image')
        if not isinstance(sparse_depth, np.ndarray) or sparse_depth.ndim != 2:
            raise ValueError('Sparse depth should be a 2D numpy ndarray with zeros at missing positions')
        input_scale = 1.0
        if config.normalize_input:
            sparse_depth, input_scale = normalize_sparse(sparse_depth)
        device = self._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)
        state = self._prepare(image, sparse_depth, processing_resolution, generator)
        self._prepare_normals(state, config, normals, kappa, intrinsics)
        trace = []
        self.scheduler.set_timesteps(config.steps, device=device)
        scale = torch.nn.Parameter(torch.ones(1, device=device))
        shift = torch.nn.Parameter(torch.ones(1, device=device))
        latent = torch.nn.Parameter(state['pred_latent'])
        optimizer = torch.optim.Adam([{'params': [scale, shift], 'lr': config.affine_lr},
                                      {'params': [latent], 'lr': config.latent_lr}])
        for index, t in enumerate(self.scheduler.timesteps):
            optimizer.zero_grad()
            noise = self.unet(torch.cat([state['image_latent'], latent], dim=1), t,
                              encoder_hidden_states=self.empty_text_embedding, return_dict=False)[0]
            with torch.no_grad():
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                pred_epsilon = (alpha_prod_t ** 0.5) * noise + ((1 - alpha_prod_t) ** 0.5) * latent
            step_output = self.scheduler.step(noise, t, latent, generator=generator)
            prediction = self._metric(step_output.pred_original_sample, scale, shift, state)
            field_parts = {}
            if config.field:
                start = int(config.field_start * config.steps)
                if index >= start and (index - start) % config.field_every == 0:
                    state['field'] = self._fit_field(prediction.detach(), state, config)
                if 'field' in state:
                    prediction = prediction + state['field']
                    field_parts = {'field_rms': float(state['field'].square().mean().sqrt())}
            loss, parts = self._guidance_loss(prediction, state, index, config)
            parts.update(field_parts)
            loss.backward()
            with torch.no_grad():
                epsilon_norm = torch.linalg.norm(pred_epsilon).item()
                grad_norm = torch.linalg.norm(latent.grad).item()
                latent.grad *= epsilon_norm / max(grad_norm, 1e-8)
            optimizer.step()
            with torch.no_grad():
                latent.data = self.scheduler.step(noise, t, latent, generator=generator).prev_sample
            trace.append({'step': index + 1, 'loss': float(loss.detach()), 'latent_grad_norm': grad_norm,
                          'scale': float(scale.detach() ** 2) * state['sparse_range'],
                          'shift': float(shift.detach() ** 2) * state['sparse_lower'], **parts})
            del step_output, prediction, pred_epsilon, noise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        extra = {}
        with torch.no_grad():
            prediction = self._metric(latent.detach(), scale, shift, state)
            if config.field:
                global_depth = self.image_processor.pt_to_numpy(prediction).squeeze() * input_scale
                extra['depth_global'] = global_depth
                prediction = prediction + self._fit_field(prediction, state, config)
        depth = self.image_processor.pt_to_numpy(prediction).squeeze()
        if input_scale != 1.0:
            depth = depth * input_scale
        self.maybe_free_model_hooks()
        if not return_details:
            return depth
        return {'depth': depth, 'trace': trace, 'input_scale': input_scale, **extra}

    def _prepare(self, image, sparse_depth, processing_resolution, generator):
        device = self._execution_device
        with torch.no_grad():
            if self.empty_text_embedding is None:
                text_inputs = self.tokenizer('', padding='do_not_pad', max_length=self.tokenizer.model_max_length,
                                             truncation=True, return_tensors='pt')
                self.empty_text_embedding = self.text_encoder(text_inputs.input_ids.to(device))[0]
        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution=processing_resolution, device=device, dtype=self.dtype)
        if sparse_depth.shape != original_resolution:
            raise ValueError(f'Sparse depth dimensions ({sparse_depth.shape}) must match the image ({original_resolution})')
        with torch.no_grad():
            image_latent, pred_latent = self.prepare_latents(image, None, generator, 1, 1)
        sparse = torch.from_numpy(sparse_depth)[None, None].float().to(device)
        mask = sparse > 0
        if int(mask.sum()) < 2:
            raise ValueError('Need at least two sparse points')
        return {'image_latent': image_latent, 'pred_latent': pred_latent, 'padding': padding,
                'original_resolution': original_resolution, 'sparse': sparse, 'mask': mask,
                'sparse_range': (sparse[mask].max() - sparse[mask].min()).item(),
                'sparse_lower': sparse[mask].min().item()}

    def _prepare_normals(self, state, config, normals, kappa, intrinsics):
        """L3 inputs on the original grid: prior normals [1,3,H,W], optional kappa [1,1,H,W], K."""
        if config.normal_weight <= 0:
            return
        if normals is None:
            raise ValueError('normal_weight > 0 needs prior normals (HxWx3) on the input grid')
        device, size = self._execution_device, state['original_resolution']
        prior = torch.as_tensor(normals, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        if tuple(prior.shape[-2:]) != tuple(size):
            raise ValueError(f'normals {tuple(prior.shape[-2:])} must match the input grid {tuple(size)}')
        state['normals'] = torch.nn.functional.normalize(prior, dim=1)
        state['kappa'] = (torch.as_tensor(kappa, dtype=torch.float32, device=device)[None, None]
                          if kappa is not None and config.kappa_gate else None)
        state['K'] = intrinsics_matrix(intrinsics, size).to(device)

    def _fit_field(self, prediction, state, config):
        """L4: closed-form residual field from the sparse residuals of a (global) prediction, [1,1,H,W]."""
        mask = state['mask'][0, 0]
        rows, cols = torch.nonzero(mask, as_tuple=True)
        h, w = mask.shape
        residual = (state['sparse'] - prediction.detach().float())[0, 0][mask]
        uv = torch.stack((cols.float() / max(w - 1, 1), rows.float() / max(h - 1, 1)), dim=1)
        weights = robust_weights(residual, config.huber_k) if config.robust == 'huber' else torch.ones_like(residual)
        q = fit_residual_field(residual.cpu(), uv.cpu(), weights.cpu(), (config.field_grid_h, config.field_grid_w),
                               config.field_lambda)
        return evaluate_field(q.to(prediction.device), (h, w)).to(prediction.dtype)

    def _metric(self, latent, scale, shift, state):
        """Decode a latent to metric depth on the original grid (Marigold-DC Eq. 2)."""
        affine_invariant = self.decode_prediction(latent)
        prediction = (scale ** 2) * state['sparse_range'] * affine_invariant + (shift ** 2) * state['sparse_lower']
        prediction = self.image_processor.unpad_image(prediction, state['padding'])
        return self.image_processor.resize_antialias(prediction, state['original_resolution'], 'bilinear', is_aa=False)

    def _guidance_loss(self, prediction, state, index, config):
        """L0: L1 + L2 at the sparse points; L2: Huber-weighted; L3: + late gated normal term."""
        predicted, target = prediction[state['mask']], state['sparse'][state['mask']]
        if config.robust == 'huber':
            loss = robust_l1l2(predicted, target, config.huber_k)
        else:
            loss = torch.nn.functional.l1_loss(predicted, target) + torch.nn.functional.mse_loss(predicted, target)
        parts = {}
        if config.normal_weight > 0:
            weight = normal_ramp(index, config)
            normal_term = prediction.new_zeros(())
            if weight > 0:
                normal_term = normal_guidance_loss(prediction, state['normals'], state['kappa'], state['K'],
                                                   config.edge_tau)
                loss = loss + weight * normal_term
            parts = {'normal_weight': weight, 'normal_loss': float(normal_term.detach())}
        return loss, parts


def normal_ramp(index, config):
    """Normal-term weight at 0-based step ``index``: 0 before normal_start*steps, linear to normal_weight at the end."""
    start = config.normal_start * config.steps
    span = max(config.steps - start, 1e-9)
    return config.normal_weight * min(1.0, max(0.0, (index + 1 - start) / span))
