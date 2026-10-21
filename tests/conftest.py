"""Shared CPU fixtures: tiny mock frozen networks and a tiny random real Diffusers pipeline."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, MarigoldDepthPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from hifidc.pipeline import DiffusionBackend, MarigoldDepthCompletionPipelineWithNormals

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


def mock_pipe():
    vae, unet, text = TinyVAE(), TinyUNet(), TinyText()
    pipe = SimpleNamespace(vae=vae, unet=unet, text_encoder=text, _execution_device=torch.device('cpu'),
                           scheduler=DDIMScheduler(num_train_timesteps=100, clip_sample=False),
                           tokenizer=lambda *a, **k: SimpleNamespace(input_ids=torch.zeros(1, 1, dtype=torch.long)),
                           dsine_model=None, dsine_args=None)
    pipe.decode_prediction = lambda latent: MarigoldDepthPipeline.decode_prediction(pipe, latent)
    return pipe


def fixture(seed=2024):
    """Mock pipeline, backend, RGB [1,3,32,32], sparse 8x8 grid, constant normals."""
    pipe = mock_pipe()
    backend = DiffusionBackend(pipe, torch.Generator().manual_seed(seed))
    rgb = torch.linspace(.05, .95, 3 * 32 * 32).reshape(1, 3, 32, 32)
    sparse = torch.zeros(1, 1, 32, 32)
    sparse[:, :, ::4, ::4] = torch.linspace(.2, 2., 64).reshape(8, 8)
    normal = F.normalize(torch.ones_like(rgb), dim=1)
    return pipe, backend, rgb, sparse, normal


def tiny_modules(directory):
    """Randomly initialised tiny UNet, VAE, CLIP text encoder and tokenizer (seeded)."""
    torch.manual_seed(13)
    (directory / 'vocab.json').write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1}))
    (directory / 'merges.txt').write_text('#version: 0.2\n')
    tokenizer = CLIPTokenizer(str(directory / 'vocab.json'), str(directory / 'merges.txt'), model_max_length=8)
    text = CLIPTextModel(CLIPTextConfig(vocab_size=2, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                                        bos_token_id=0, eos_token_id=1, pad_token_id=1))
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
                        up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'),
                        block_out_channels=(8, 8), layers_per_block=1, norm_num_groups=4)
    unet = UNet2DConditionModel(sample_size=16, in_channels=8, out_channels=4,
                                down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
                                block_out_channels=(8, 8), layers_per_block=1, cross_attention_dim=8,
                                attention_head_dim=2, norm_num_groups=4)
    return unet, vae, text, tokenizer


def build_tiny_real_pipeline(directory):
    """Randomly initialised tiny UNet/VAE/CLIP saved and reloaded through from_pretrained."""
    unet, vae, text, tokenizer = tiny_modules(directory)
    pipe = MarigoldDepthCompletionPipelineWithNormals(unet, vae,
                                                      DDIMScheduler(num_train_timesteps=100, clip_sample=False), text, tokenizer)
    model = directory / 'model'
    pipe.save_pretrained(model)
    return MarigoldDepthCompletionPipelineWithNormals.from_pretrained(
        model, local_files_only=True, dsine_model=None, dsine_args=None, torch_dtype=torch.float32).to('cpu')


@pytest.fixture
def tiny_real_pipeline(tmp_path):
    return build_tiny_real_pipeline(tmp_path)


def sparse_grid(h, w, step=4, low=.2, high=2.):
    sparse = np.zeros((h, w), dtype=np.float32)
    rows, cols = len(range(0, h, step)), len(range(0, w, step))
    sparse[::step, ::step] = np.linspace(low, high, rows * cols).reshape(rows, cols)
    return sparse


def build_tiny_v2_pipeline(directory):
    """Tiny random modules in a HiFiDCv2Pipeline with Marigold-DC's trailing DDIM schedule."""
    from hifidc.v2.pipeline import HiFiDCv2Pipeline
    unet, vae, text, tokenizer = tiny_modules(directory)
    scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False, timestep_spacing='trailing')
    return HiFiDCv2Pipeline(unet=unet, vae=vae, scheduler=scheduler, text_encoder=text, tokenizer=tokenizer,
                            prediction_type='depth').to('cpu')


@pytest.fixture
def tiny_v2_pipeline(tmp_path):
    return build_tiny_v2_pipeline(tmp_path)
