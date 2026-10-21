"""Camera-aware normals from depth, and the kappa/edge-gated normal guidance loss."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hifidc.data import load_manifest, load_scene
from hifidc.v2.geometry import intrinsics_matrix, normal_guidance_loss, normals_from_depth

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'test'


def plane_depth(K, h, w, normal, offset):
    """Depth of the plane n . P = offset seen through K (P = D * K^-1 [u, v, 1])."""
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing='ij')
    rays = torch.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)))
    n = torch.tensor(normal, dtype=torch.float64)
    return (offset / (n[:, None, None] * rays).sum(0))[None, None]


def angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(1).clamp(-1, 1)))


def test_plane_normal_analytic():
    K = intrinsics_matrix((300., 300., 32., 24.), (48, 64)).double()
    normal = torch.nn.functional.normalize(torch.tensor([0.3, -0.2, 1.0]), dim=0).tolist()
    depth = plane_depth(K, 48, 64, normal, 2.0)
    n = normals_from_depth(depth, K)
    expected = torch.tensor(normal, dtype=n.dtype)[None, :, None, None].expand_as(n)
    assert float(angle(n, expected)[..., 2:-2, 2:-2].max()) < math.degrees(1e-3)


def test_fronto_parallel_points_away_from_camera():
    K = intrinsics_matrix(None, (32, 32))
    n = normals_from_depth(torch.full((1, 1, 32, 32), 3.0), K)
    assert torch.allclose(n[0, :, 4:-4, 4:-4].mean((1, 2)), torch.tensor([0., 0., 1.]), atol=1e-6)


def test_invalid_depth_finite_and_gated():
    depth = torch.full((1, 1, 32, 32), 2.0)
    depth[..., :8] = 0.0
    depth[..., 8:12, :] = -1.0
    depth[..., 20:, 16:] = 5.0  # step edge
    K = intrinsics_matrix(None, (32, 32))
    n = normals_from_depth(depth, K)
    assert torch.isfinite(n).all() and torch.allclose(n.norm(dim=1), torch.ones(1, 32, 32), atol=1e-5)
    depth.requires_grad_(True)
    prior = torch.nn.functional.normalize(torch.randn(1, 3, 32, 32), dim=1)
    loss = normal_guidance_loss(depth, prior, None, K, 0.2)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(depth.grad).all()
    assert float(depth.grad[..., :8].abs().sum()) == 0.0  # invalid pixels carry no weight


def test_matching_prior_gives_zero_loss():
    K = intrinsics_matrix(None, (32, 32))
    depth = plane_depth(K.double(), 32, 32, [0.0, 0.0, 1.0], 2.0).float()
    loss = normal_guidance_loss(depth, normals_from_depth(depth, K), torch.ones(1, 1, 32, 32), K, 0.2)
    assert float(loss) < 1e-6


def test_fov_fallback_matches_60deg():
    K = intrinsics_matrix(None, (480, 640))
    assert abs(float(K[0, 0]) - 320 / math.tan(math.radians(30))) < 1e-3
    assert float(K[0, 2]) == 320 and float(K[1, 2]) == 240 and float(K[0, 0]) == float(K[1, 1])


def test_manifest_intrinsics_loaded():
    specs = {s.stem: s for s in load_manifest(ROOT / 'configs' / 'scenes.yaml')}
    assert specs['student_lounge_0001_out_379'].intrinsics == (518.8579, 519.4696, 325.5824, 253.7362)
    assert specs['market_2_frame_0017'].intrinsics is None


@pytest.mark.skipif(not (ROOT / 'cache' / 'normals').is_dir() or not DATA.is_dir(), reason='needs data + cached normals')
def test_dsine_convention_on_gt():
    spec = next(s for s in load_manifest(ROOT / 'configs' / 'scenes.yaml') if s.stem == '3FO4MMLDUKDJ_Living_room_8')
    scene = load_scene(DATA, spec)
    depth = torch.from_numpy(scene.gt * spec.to_metres)[None, None]
    n = normals_from_depth(depth, intrinsics_matrix(spec.intrinsics, depth.shape[-2:]))
    prior = torch.from_numpy(np.load(ROOT / 'cache' / 'normals' / f'{spec.stem}_normals.npy')).permute(2, 0, 1)[None]
    errors = angle(n, prior)[torch.from_numpy(scene.valid_gt)[None]]
    assert float(errors.median()) < 30.0
