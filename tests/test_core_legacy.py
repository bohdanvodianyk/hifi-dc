"""Component/integration tests carried over from the clean reconstruction (CPU, no downloads)."""
from dataclasses import replace
import unittest

import numpy as np
from PIL import Image
import pytest
import torch

from conftest import TinyText, fixture, mock_pipe, sparse_grid
from hifidc.config import Config
from hifidc.core import (affine_fit, bilateral_filter, complete, data_terms, depth_normals, normal_loss,
                         normal_weight, resize_sparse)
from hifidc.pipeline import DiffusionBackend, MarigoldDepthCompletionPipelineWithNormals


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
        self.assertAlmostEqual(float(q), float((w * r * r).sum() / 4 - (w * r).sum() ** 2 / 32), places=6)
        self.assertAlmostEqual(float(l1), float(.5 * (w * r.abs()).sum() / 4), places=6)

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
        self.assertEqual([r['height'] for r in result['trace']], [16, 16, 32, 32, 32, 32])
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
            loss = ((backend.decode(prediction) * scale - sparse) ** 2).mean()
            loss.backward()
            old = latent.detach().clone()
            optimizer.step()
            self.assertEqual(latent.grad is None, detached)
            self.assertEqual(torch.equal(old, latent), detached)

    def test_legacy_call_returns_original_grid(self):
        pipe, _, _, sparse, normal = fixture()
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
            sparse_depth=sparse[0, 0].numpy(), normal_prior=normal[0].numpy(),
            num_inference_steps=6, processing_resolution=32, config=Config(bilateral=False))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (32, 32))
        self.assertTrue(np.isfinite(result).all())

    def test_millimetre_roundtrip(self):
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        results = []
        for unit, factor in [('m', 1.), ('mm', 1000.)]:
            pipe, _, _, sparse, normal = fixture()
            result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
                sparse_depth=sparse[0, 0].numpy() * factor, normal_prior=normal[0].numpy(),
                num_inference_steps=6, processing_resolution=32,
                config=Config(bilateral=False, depth_unit=unit))
            results.append(result / factor)
        np.testing.assert_allclose(*results, rtol=2e-5, atol=2e-5)

    def test_config_is_used_without_silent_default_override(self):
        pipe, _, _, sparse, normal = fixture()
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
            sparse_depth=sparse[0, 0].numpy(), normal_prior=normal[0].numpy(),
            processing_resolution=32, config=Config(steps=3, normal_weight=.25, bilateral=False), return_details=True)
        self.assertEqual(len(result['trace']), 3)
        self.assertEqual(result['config']['normal_weight'], .25)


def test_local_save_load_and_gradient(tiny_real_pipeline):
    image = Image.fromarray(np.full((32, 48, 3), 128, dtype=np.uint8))
    sparse = sparse_grid(32, 48)
    normals = np.zeros((32, 48, 3), dtype=np.float32)
    normals[:, :, 2] = 1
    result = tiny_real_pipeline(image=image, sparse_depth=sparse, normal_prior=normals,
                                num_inference_steps=3, processing_resolution=48,
                                config=Config(bilateral=False), return_details=True)
    assert result['depth'].shape == (32, 48)
    assert np.isfinite(result['depth']).all()
    assert all(t['latent_gradient_norm'] > 0 for t in result['trace'])
    assert all(t['adam_max_update'] > 0 for t in result['trace'])
    assert all(not p.requires_grad and p.grad is None for p in tiny_real_pipeline.unet.parameters())


def test_legacy_call_odd_grid():
    pipe = mock_pipe()
    image = Image.fromarray(np.full((37, 53, 3), 128, dtype=np.uint8))
    normals = np.zeros((37, 53, 3), dtype=np.float32)
    normals[:, :, 2] = 1
    result = MarigoldDepthCompletionPipelineWithNormals.__call__(pipe, image=image,
        sparse_depth=sparse_grid(37, 53), normal_prior=normals,
        num_inference_steps=3, processing_resolution=48, config=Config(bilateral=False))
    assert result.shape == (37, 53)
    assert np.isfinite(result).all()


class RaisingText(TinyText):
    def forward(self, ids):
        raise AssertionError('text encoder must not run when an embedding is supplied')


def test_backend_uses_supplied_embedding():
    pipe = mock_pipe()
    pipe.text_encoder = RaisingText()
    backend = DiffusionBackend(pipe, torch.Generator().manual_seed(0), embedding=torch.zeros(1, 1, 1))
    assert torch.equal(backend.embedding, torch.zeros(1, 1, 1))
    with pytest.raises(AssertionError):
        DiffusionBackend(pipe, torch.Generator().manual_seed(0))


def test_free_text_encoder_keeps_results_identical(tiny_real_pipeline):
    image = Image.fromarray(np.full((32, 48, 3), 128, dtype=np.uint8))
    normals = np.zeros((32, 48, 3), dtype=np.float32)
    normals[:, :, 2] = 1
    kwargs = dict(image=image, sparse_depth=sparse_grid(32, 48), normal_prior=normals, num_inference_steps=3,
                  processing_resolution=48, config=Config(bilateral=False))
    before = tiny_real_pipeline(**kwargs)
    tiny_real_pipeline.free_text_encoder()
    assert tiny_real_pipeline.text_encoder is None
    after = tiny_real_pipeline(**kwargs)
    np.testing.assert_array_equal(before, after)
