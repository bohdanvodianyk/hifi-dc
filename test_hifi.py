"""CPU component/integration tests, no model downloads or benchmark data."""
import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, MarigoldDepthPipeline
from hifi_core import Config, affine_fit, data_terms, depth_normals, normal_loss, normal_weight, resize_sparse, bilateral_filter, complete
from hifi_dc import DiffusionBackend, MarigoldDepthCompletionPipelineWithNormals

torch.set_num_threads(1)


class TinyVAE(torch.nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1., latent_channels=4)
    def __init__(self):
        super().__init__()
        self.fixed = torch.nn.Parameter(torch.tensor(.5))
    def encode(self, image):
        encoded = F.avg_pool2d(image, 8)
        encoded = torch.cat((encoded, encoded[:, :1]), 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: encoded))
    def decode(self, latent, return_dict=False):
        return (torch.tanh(F.interpolate(latent[:, :3], scale_factor=8, mode='bilinear', align_corners=False) * self.fixed),)


class TinyUNet(torch.nn.Module):
    dtype = torch.float32
    def __init__(self):
        super().__init__()
        self.fixed = torch.nn.Parameter(torch.tensor(.12))
    def forward(self, mix, timestep, **kwargs):
        return (self.fixed * torch.tanh(mix[:, -4:] + .2 * mix[:, :4]),)


class TinyText(torch.nn.Module):
    def forward(self, ids):
        return (torch.zeros(1, 1, 1),)


def fixture(seed=2024):
    vae, unet, text = TinyVAE(), TinyUNet(), TinyText()
    pipe = SimpleNamespace(vae=vae, unet=unet, text_encoder=text, _execution_device=torch.device('cpu'),
                           scheduler=DDIMScheduler(num_train_timesteps=100, clip_sample=False),
                           tokenizer=lambda *a, **k: SimpleNamespace(input_ids=torch.zeros(1, 1, dtype=torch.long)),
                           dsine_model=None, dsine_args=None)
    pipe.decode_prediction = lambda latent: MarigoldDepthPipeline.decode_prediction(pipe, latent)
    backend = DiffusionBackend(pipe, torch.Generator().manual_seed(seed))
    rgb = torch.linspace(.05, .95, 3 * 32 * 32).reshape(1, 3, 32, 32)
    sparse = torch.zeros(1, 1, 32, 32)
    sparse[:, :, ::4, ::4] = torch.linspace(.2, 2., 64).reshape(8, 8)
    normal = F.normalize(torch.ones_like(rgb), dim=1)
    return pipe, backend, rgb, sparse, normal


class CoreTests(unittest.TestCase):
    def test_weighted_ridge_matches_normal_equations_and_ema(self):
        z = torch.tensor([0., 1., 2., 3.]).reshape(1, 1, 2, 2)
        y = torch.tensor([1., 2., 3., 20.]).reshape_as(z)
        w = torch.tensor([1., 1., 1., .01]).reshape_as(z)
        c = Config(ridge=.1, ema_beta=.5)
        previous = (torch.tensor(1.), torch.tensor(0.))
        a, b = affine_fit(z, y, torch.ones_like(z, dtype=torch.bool), w, previous, c)
        X = torch.stack((z.flatten(), torch.ones(4)), 1).double()
        W = torch.diag(w.flatten().double())
        expected = torch.linalg.solve(X.T @ W @ X + torch.diag(torch.tensor([.1, 0.], dtype=torch.float64)), X.T @ W @ y.flatten().double())
        actual = torch.stack((a, b)).double()
        torch.testing.assert_close(actual, .5 * torch.tensor([1., 0.]).double() + .5 * expected, rtol=1e-6, atol=1e-6)

    def test_fit_fallback(self):
        d = torch.ones(1, 1, 2, 2)
        previous = (d.new_tensor(2.), d.new_tensor(3.))
        self.assertIs(affine_fit(d, d, d.bool(), d * 0, previous, Config()), previous)

    def test_published_data_formula(self):
        p = torch.tensor([1., 3., 5., 7.]).reshape(1, 1, 2, 2)
        t, w = torch.ones_like(p), torch.tensor([1., .5, .25, .1]).reshape_as(p)
        q, l1 = data_terms(p, t, t.bool(), w, Config())
        r = p - t
        self.assertAlmostEqual(float(q), float((w * r*r).sum()/4 - (w*r).sum()**2/32), places=6)
        self.assertAlmostEqual(float(l1), float(.5*(w*r.abs()).sum()/4), places=6)

    def test_matching_normals_have_zero_loss_and_finite_gradient(self):
        depth = torch.ones(1, 1, 16, 16, requires_grad=True)
        prior = depth_normals(depth.detach(), .2)
        loss = normal_loss(depth, prior, Config())
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 0., places=6)
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_schedule(self):
        c = Config()
        self.assertEqual(normal_weight(33, 50, .1, c), 0.)
        self.assertAlmostEqual(normal_weight(50, 50, .1, c), .1)
        self.assertGreater(normal_weight(50, 50, 0., c), normal_weight(50, 50, 1., c))
        c.normal_schedule = 'constant'
        self.assertEqual(normal_weight(1, 50, 100., c), .1)

    def test_sparse_resize_preserves_single_sample_value(self):
        depth = torch.zeros(1, 1, 16, 16)
        depth[0, 0, 5, 7] = 8.
        resized = resize_sparse(depth, (8, 8))
        self.assertEqual(int((resized > 0).sum()), 1)
        self.assertEqual(float(resized.max()), 8.)

    def test_bilateral_constant(self):
        depth = torch.full((1, 1, 8, 8), 3.)
        torch.testing.assert_close(bilateral_filter(depth, Config()), depth)

    def test_progressive_gradient_frozen_models_and_repeatability(self):
        pipe, backend, rgb, sparse, normal = fixture()
        before = pipe.unet.fixed.detach().clone()
        result = complete(backend, rgb, sparse, normal, Config(steps=6, bilateral=False))
        self.assertEqual([r['height'] for r in result['trace']], [16,16,32,32,32,32])
        self.assertTrue(all(r['latent_gradient_norm'] > 0 for r in result['trace']))
        self.assertTrue(all(r['adam_max_update'] > 0 for r in result['trace']))
        self.assertIsNone(pipe.unet.fixed.grad)
        self.assertIsNone(pipe.vae.fixed.grad)
        torch.testing.assert_close(pipe.unet.fixed, before)
        _, backend2, *unused = fixture()
        again = complete(backend2, rgb, sparse, normal, Config(steps=6, bilateral=False))
        torch.testing.assert_close(result['depth'], again['depth'], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result['uncertainty']).all())

    def test_normal_guidance_changes_output(self):
        _, backend, rgb, sparse, normal = fixture()
        c = Config(steps=6, bilateral=False, normal_weight=1., normal_schedule='constant')
        with_normals = complete(backend, rgb, sparse, normal, c)
        _, backend, *unused = fixture()
        without = complete(backend, rgb, sparse, normal, replace(c, use_normals=False))
        self.assertGreater(float((with_normals['depth'] - without['depth']).abs().max()), 1e-6)

    def test_original_gradient_break_and_positive_control(self):
        # Reproduce the exact data-flow operation, using real DDIM and frozen mock networks.
        for detached in (True, False):
            _, backend, rgb, sparse, _ = fixture()
            backend.set_timesteps(6)
            encoded = backend.encode_rgb(rgb)
            latent = torch.nn.Parameter(backend.initial_latent(encoded))
            optimizer = torch.optim.Adam([latent], lr=.02)
            with torch.set_grad_enabled(not detached):
                prediction = backend.predict_clean(latent, backend.timesteps[0], encoded)
            # Like the source, affine parameters require grad but are not in this optimiser.
            scale = torch.nn.Parameter(torch.ones(()))
            loss = ((backend.decode(prediction) * scale - sparse)**2).mean()
            loss.backward()
            old = latent.detach().clone()
            optimizer.step()
            self.assertEqual(latent.grad is None, detached)
            self.assertEqual(torch.equal(old, latent), detached)

    def test_original_normal_and_log_helpers(self):
        original = Path(__file__).resolve().parent.parent/'HIFI-DC/new_exp_pipeline_2_ablation.py'
        if not original.exists():
            self.skipTest('Original snapshot is outside this distributed clean package')
        tree = ast.parse(original.read_text(encoding='utf-8-sig'))
        metric = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_depth_metrics')
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EnhancedAblationPipeline')
        cosine = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'cosine_loss')
        cosine.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[metric, cosine], type_ignores=[]))
        ns = {'torch':torch, 'F':F}
        exec(compile(module, str(original), 'exec'), ns)
        v = F.normalize(torch.ones(1,3,1,1), dim=1)
        self.assertAlmostEqual(float(ns['cosine_loss'](v,v)), 2/3, places=6)
        metrics = ns['_compute_depth_metrics'](torch.tensor([1., np.exp(2)]), torch.ones(2), torch.ones(2,dtype=torch.bool))
        self.assertAlmostEqual(metrics['rmse_log'], 1., places=6)
        self.assertNotAlmostEqual(metrics['rmse_log'], np.sqrt(2), places=3)

    def test_legacy_call_returns_original_grid(self):
        pipe, _, _, sparse, normal = fixture()
        image = Image.fromarray(np.full((32,32,3), 128, dtype=np.uint8))
        result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
            sparse_depth=sparse[0,0].numpy(), normal_prior=normal[0].numpy(),
            num_inference_steps=6, processing_resolution=32, config=Config(bilateral=False))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (32,32))
        self.assertTrue(np.isfinite(result).all())

    def test_millimetre_roundtrip(self):
        image = Image.fromarray(np.full((32,32,3), 128, dtype=np.uint8))
        results = []
        for unit, factor in [('m', 1.), ('mm', 1000.)]:
            pipe, _, _, sparse, normal = fixture()
            result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
                sparse_depth=sparse[0,0].numpy() * factor, normal_prior=normal[0].numpy(),
                num_inference_steps=6, processing_resolution=32,
                config=Config(bilateral=False, depth_unit=unit))
            results.append(result/factor)
        np.testing.assert_allclose(*results, rtol=2e-5, atol=2e-5)

    def test_config_is_used_without_silent_default_override(self):
        pipe, _, _, sparse, normal = fixture()
        image = Image.fromarray(np.full((32,32,3), 128, dtype=np.uint8))
        result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
            sparse_depth=sparse[0,0].numpy(), normal_prior=normal[0].numpy(),
            processing_resolution=32, config=Config(steps=3, normal_weight=.25, bilateral=False), return_details=True)
        self.assertEqual(len(result['trace']), 3)
        self.assertEqual(result['config']['normal_weight'], .25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
