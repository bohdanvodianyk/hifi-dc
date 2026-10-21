"""Publication-guided reconstruction; see RECONSTRUCTION_NOTES.md for choices.

No historical-result reproduction is implied. This module needs only torch.
"""
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn.functional as F


@dataclass
class Config:
    # Values stated in the paper.
    steps: int = 50
    normal_weight: float = 0.1
    lambda_l1: float = 0.5
    lsq_interval: int = 10
    ema_beta: float = 0.5
    edge_tau: float = 0.20
    stability_threshold: float = 0.10
    bilateral_sigma_space: float = 75.0
    bilateral_sigma_range: float = 75.0
    # Reconstruction choices where the paper is not a complete specification.
    learning_rate: float = 0.02  # surviving code; not a recovered run config
    gradient_clip: float = 1.0  # surviving code
    ridge: float = 1e-6
    variance_guard: float = 1e-8
    confidence_window: int = 5
    confidence_gain: float = 10.0
    confidence_midpoint: float = 0.5
    bilateral_diameter: int = 5  # paper gives {5, 7}
    depth_unit: str = 'm'
    # Controls are prospective interventions, not recovered historical rows.
    use_normals: bool = True
    use_l1: bool = True
    use_quadratic_correction: bool = True
    periodic_fit: bool = True
    warm_start: bool = True
    final_fit: bool = True
    progressive: bool = True
    bilateral: bool = True
    confidence: bool = True
    normal_schedule: str = 'paper'  # paper, constant, ramp_only

    def validate(self):
        for key in ('steps', 'lsq_interval', 'confidence_window', 'bilateral_diameter'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.steps < 3 and self.progressive:
            raise ValueError('Progressive inference needs at least three steps')
        if self.confidence_window % 2 != 1 or self.bilateral_diameter % 2 != 1:
            raise ValueError('Window diameters must be odd')
        for key in ('learning_rate', 'gradient_clip', 'edge_tau',
                    'bilateral_sigma_space', 'bilateral_sigma_range', 'confidence_gain'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('ridge', 'variance_guard', 'normal_weight', 'lambda_l1', 'stability_threshold'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if not 0 <= self.ema_beta <= 1 or not 0 <= self.confidence_midpoint <= 1:
            raise ValueError('EMA beta and sigmoid midpoint must be in [0, 1]')
        if self.normal_schedule not in ('paper', 'constant', 'ramp_only'):
            raise ValueError('Unknown normal schedule')
        if self.depth_unit not in ('m', 'mm'):
            raise ValueError('depth_unit must be m or mm')
        return self


def confidence_weights(mask, config):
    if not config.confidence:
        return mask.to(torch.float32)
    k = config.confidence_window
    density = F.avg_pool2d(mask.float(), k, 1, k // 2, count_include_pad=False)
    maximum = density.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    normalized = density / maximum
    return torch.sigmoid(config.confidence_gain * (normalized - config.confidence_midpoint)) * mask


def affine_fit(decoded, sparse, mask, weights, previous, config):
    """Paper centred ridge fit, EMA, and fallback; fitted coefficients detached.

    The ridge penalises slope only. Slopes are unconstrained as in the equations.
    """
    with torch.no_grad():
        valid = mask & torch.isfinite(decoded) & torch.isfinite(sparse) & (weights > 0)
        if valid.sum().item() < 4:
            return previous
        z, y, w = (v[valid].double() for v in (decoded, sparse, weights))
        sw = w.sum()
        if not torch.isfinite(sw) or sw <= 0:
            return previous
        zm, ym = (w * z).sum() / sw, (w * y).sum() / sw
        variance = (w * (z - zm).square()).sum() + config.ridge
        if not torch.isfinite(variance) or variance <= config.variance_guard:
            return previous
        a = (w * (z - zm) * (y - ym)).sum() / variance
        b = ym - a * zm
        if not torch.isfinite(a) or not torch.isfinite(b):
            return previous
        beta = config.ema_beta
        return tuple(((1 - beta) * p + beta * q).to(decoded.dtype).detach()
                     for p, q in zip(previous, (a, b)))


def data_terms(prediction, sparse, mask, weights, config):
    r, w = (prediction - sparse)[mask], weights[mask]
    n = r.numel()
    if n == 0:
        raise ValueError('No valid sparse observations')
    quadratic = (w * r.square()).sum() / n
    if config.use_quadratic_correction:
        quadratic = quadratic - 0.5 * (w * r).sum().square() / (n * n)
    l1 = config.lambda_l1 * (w * r.abs()).sum() / n if config.use_l1 else quadratic.new_zeros(())
    return quadratic, l1


def depth_normals(depth, tau):
    # Sobel / 8, replicate border: explicit reconstruction convention.
    kx = depth.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    padded = F.pad(depth, (1, 1, 1, 1), mode='replicate')
    dx = F.conv2d(padded, kx[None, None])
    dy = F.conv2d(padded, kx.T[None, None])
    # vector_norm has a defined zero-gradient at the origin.
    magnitude = torch.linalg.vector_norm(torch.cat((dx, dy), dim=1), dim=1, keepdim=True)
    attenuation = torch.exp(-magnitude / tau)
    return F.normalize(torch.cat((-attenuation * dx, -attenuation * dy, torch.ones_like(depth)), 1), p=2, dim=1)


def normal_loss(depth, prior, config):
    predicted = depth_normals(depth, config.edge_tau)
    result = depth.new_zeros(())
    for factor, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
        size = tuple(max(1, n // factor) for n in depth.shape[-2:])
        a = F.normalize(F.interpolate(predicted, size=size, mode='area'), p=2, dim=1)
        b = F.normalize(F.interpolate(prior, size=size, mode='area'), p=2, dim=1)
        result = result + weight * (1 - (a * b).sum(1)).mean()
    return result


def normal_weight(step, steps, data_change, config):
    if config.normal_schedule == 'constant':
        return config.normal_weight
    ramp = max(0., min(1., (step - 0.66 * steps) / (0.34 * steps)))
    gamma = 1.0 if config.normal_schedule == 'ramp_only' else 1 + 0.5 * math.tanh(5 * (config.stability_threshold - data_change))
    return config.normal_weight * ramp * gamma


def resize_sparse(depth, size):
    """Map valid sample centres; average collisions. Never interpolate missing zeros."""
    if tuple(depth.shape[-2:]) == tuple(size):
        return depth.clone()
    valid = (depth[0, 0] > 0) & torch.isfinite(depth[0, 0])
    y, x = valid.nonzero(as_tuple=True)
    h, w = depth.shape[-2:]
    nh, nw = size
    yy = ((y.float() + .5) * nh / h).long().clamp(0, nh - 1)
    xx = ((x.float() + .5) * nw / w).long().clamp(0, nw - 1)
    index = yy * nw + xx
    sums = depth.new_zeros(nh * nw).scatter_add_(0, index, depth[0, 0][valid])
    counts = depth.new_zeros(nh * nw).scatter_add_(0, index, torch.ones_like(index, dtype=depth.dtype))
    return (sums / counts.clamp_min(1)).reshape(1, 1, nh, nw)


def bilateral_filter(depth, config):
    radius = config.bilateral_diameter // 2
    padded = F.pad(depth, (radius,) * 4, mode='replicate')
    total, denom = torch.zeros_like(depth), torch.zeros_like(depth)
    h, w = depth.shape[-2:]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbour = padded[..., radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            weight = torch.exp(-((neighbour - depth) / config.bilateral_sigma_range).square() / 2
                               - (dx * dx + dy * dy) / (2 * config.bilateral_sigma_space ** 2))
            total += weight * neighbour
            denom += weight
    return total / denom.clamp_min(1e-12)


def uncertainty_map(mask, changes):
    # scipy is imported only for the exact Euclidean distance transform.
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    distance = distance_transform_edt(~mask[0, 0].detach().cpu().numpy())
    distance /= max(float(distance.max()), 1e-8)
    stability = float(np.mean(changes[-10:])) if changes else 0.
    return torch.as_tensor(distance, device=mask.device, dtype=torch.float32)[None, None] * (1 + stability)


def complete(backend, rgb, sparse, normals, config):
    """Backend encodes RGB, differentiably decodes depth and runs DDIM.

    Arrays use one image, NCHW. RGB is [0,1]; sparse depths use the declared
    metric unit; normals share the image grid. No ground truth is accepted.
    """
    config.validate()
    if config.depth_unit != 'm':
        raise ValueError('Core tensors must use metres; the compatibility wrapper converts mm inputs')
    if rgb.ndim != 4 or rgb.shape[:2] != (1, 3) or sparse.shape != (1, 1, *rgb.shape[-2:]):
        raise ValueError('Expected RGB [1,3,H,W] and sparse depth [1,1,H,W]')
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('RGB must be finite and in [0,1]')
    if config.use_normals and (normals is None or normals.shape != rgb.shape or not torch.isfinite(normals).all()
                               or (torch.linalg.vector_norm(normals, dim=1) < 1e-8).any()):
        raise ValueError('Supply finite, nonzero aligned normal vectors [1,3,H,W]')
    full_mask = torch.isfinite(sparse) & (sparse > 0)
    if full_mask.sum() < 4:
        raise ValueError('At least four valid positive sparse observations are required')
    sparse = torch.where(full_mask, sparse, torch.zeros_like(sparse))
    h, w = sparse.shape[-2:]
    if h % 16 or w % 16:
        raise ValueError('Prepared grid dimensions must be divisible by 16')
    backend.set_timesteps(config.steps)
    params = (sparse.new_tensor(1.), sparse.new_tensor(0.))
    changes, trace = [], []
    previous_data = None
    latent = optimizer = None
    current_size = None
    low_steps = config.steps // 3 if config.progressive else 0
    for index, timestep in enumerate(backend.timesteps):
        size = (h // 2, w // 2) if index < low_steps else (h, w)
        if size != current_size:
            stage_rgb = F.interpolate(rgb, size=size, mode='bilinear', align_corners=False)
            stage_sparse = resize_sparse(sparse, size)
            mask = stage_sparse > 0
            weights = confidence_weights(mask, config)
            prior = F.normalize(F.interpolate(normals, size=size, mode='bilinear', align_corners=False), dim=1) if config.use_normals else None
            encoded = backend.encode_rgb(stage_rgb)
            latent = torch.nn.Parameter(backend.initial_latent(encoded) if latent is None else
                                       F.interpolate(latent.detach(), size=encoded.shape[-2:], mode='bilinear', align_corners=False))
            optimizer = torch.optim.Adam([latent], lr=config.learning_rate)
            # Explicit choice: reset Adam and stability history across resolutions.
            previous_data = None
            current_size = size
        optimizer.zero_grad(set_to_none=True)
        prediction = backend.predict_clean(latent, timestep, encoded)
        decoded = backend.decode(prediction)
        if (index == 0 and config.warm_start) or (index > 0 and config.periodic_fit and index % config.lsq_interval == 0):
            params = affine_fit(decoded, stage_sparse, mask, weights, params, config)
        aligned = params[0] * decoded + params[1]
        quadratic, l1 = data_terms(aligned, stage_sparse, mask, weights, config)
        data = quadratic + l1
        dv = float(data.detach())
        change = 0. if previous_data is None else abs(dv - previous_data) / (previous_data + 1e-8)
        previous_data = dv
        changes.append(change)
        wn = normal_weight(index + 1, config.steps, change, config)
        geometric = normal_loss(aligned, prior, config) if config.use_normals else data.new_zeros(())
        loss = data + wn * geometric
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss; run stopped rather than repairing outputs')
        loss.backward()
        if latent.grad is None or not torch.isfinite(latent.grad).all():
            raise RuntimeError('Missing or nonfinite latent gradient')
        norm = float(torch.nn.utils.clip_grad_norm_([latent], config.gradient_clip))
        before = latent.detach().clone()
        optimizer.step()
        update = float((latent.detach() - before).abs().max())
        trace.append({'step': index + 1, 'height': size[0], 'width': size[1], 'data_loss': dv,
                      'normal_loss': float(geometric.detach()), 'normal_weight': wn, 'loss': float(loss.detach()),
                      'data_change': change, 'latent_gradient_norm': norm, 'adam_max_update': update,
                      'affine_scale': float(params[0]), 'affine_shift': float(params[1])})
        with torch.no_grad():
            # Recompute the denoiser after the Adam update; do not use stale noise.
            latent.copy_(backend.advance(latent, timestep, encoded))
    with torch.no_grad():
        depth = backend.decode(latent)
        if config.final_fit:
            params = affine_fit(depth, sparse, full_mask, confidence_weights(full_mask, config), params, config)
        depth = params[0] * depth + params[1]
        if config.bilateral:
            depth = bilateral_filter(depth, config)
        if not torch.isfinite(depth).all():
            raise FloatingPointError('Nonfinite final depth')
        uncertainty = uncertainty_map(full_mask, changes)
    return {'depth': depth, 'uncertainty': uncertainty, 'trace': trace, 'config': asdict(config),
            'status': 'publication-guided reconstruction; benchmark reproduction unverified'}
