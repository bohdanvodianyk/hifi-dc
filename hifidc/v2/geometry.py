"""Camera-aware surface normals from depth and the gated normal-guidance loss (rung L3).

Axes are (x right, y down, z forward) and normals point away from the camera, i.e.
cross(dP/du, dP/dv) of the back-projected points P = D * K^-1 [u, v, 1]. The cached DSINE
normals follow the same convention on our data (tests/test_geometry.py::test_dsine_convention_on_gt),
even though DSINE's README describes them as outward.
"""
import math

import torch
import torch.nn.functional as F

FALLBACK_FOV_DEG = 60.0


def intrinsics_matrix(intrinsics, shape_hw):
    """3x3 K from (fx, fy, cx, cy) in image pixels, or a 60-degree horizontal FOV when unknown."""
    h, w = (int(n) for n in shape_hw)
    if intrinsics is None:
        f = (w / 2) / math.tan(math.radians(FALLBACK_FOV_DEG / 2))
        fx, fy, cx, cy = f, f, w / 2, h / 2
    else:
        fx, fy, cx, cy = intrinsics
    return torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=torch.float32)


def _valid_stencil(depth):
    """Pixels whose whole 3x3 neighbourhood has positive depth."""
    valid = (depth > 0).to(depth.dtype)
    return -F.max_pool2d(-F.pad(valid, (1, 1, 1, 1), value=0.), 3, stride=1) > 0.5


def normals_from_depth(depth, K):
    """[1,1,H,W] depth -> [1,3,H,W] unit normals (central differences of back-projected points)."""
    h, w = depth.shape[-2:]
    K = K.to(device=depth.device, dtype=depth.dtype)
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                          torch.arange(w, device=depth.device, dtype=depth.dtype), indexing='ij')
    d = depth[:, 0]
    points = torch.stack(((u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d), dim=1)
    padded = F.pad(points, (1, 1, 1, 1), mode='replicate')
    du = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]) / 2
    dv = (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]) / 2
    normals = torch.cross(du, dv, dim=1)
    forward = torch.zeros_like(normals)
    forward[:, 2] = 1.0
    good = _valid_stencil(depth) & (torch.linalg.vector_norm(normals, dim=1, keepdim=True) > 0)
    return F.normalize(torch.where(good, normals, forward), dim=1)


def normal_guidance_loss(depth, prior, kappa, K, edge_tau):
    """Weighted mean of 1 - <n_D, n_prior>.

    Weight (detached) = kappa scaled to [0, 1] by its 99th percentile (1 if None)
    x exp(-|grad log D| / edge_tau) x [whole 3x3 stencil has positive depth].
    """
    normals = normals_from_depth(depth, K)
    with torch.no_grad():
        weight = _valid_stencil(depth).to(depth.dtype)
        log_d = torch.log(depth.clamp_min(1e-6))
        padded = F.pad(log_d, (1, 1, 1, 1), mode='replicate')
        gx = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]) / 2
        gy = (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]) / 2
        weight = weight * torch.exp(-torch.sqrt(gx ** 2 + gy ** 2) / edge_tau)
        if kappa is not None:
            k = kappa.to(depth.dtype)
            scale = torch.quantile(k.flatten().float()[::max(1, k.numel() // 1_000_000)], 0.99).to(depth.dtype)
            weight = weight * (k / scale.clamp_min(1e-12)).clamp(0, 1)
    cosine = (normals * prior.to(normals.dtype)).sum(1, keepdim=True)
    return (weight * (1 - cosine)).sum() / (weight.sum() + 1e-8)
