"""Offline smoke test with randomly initialised, tiny real Diffusers modules.

Validates local save/load, legacy call, VAE/UNet gradient integration and shapes.
This is not a trained-model quality test or a benchmark reproduction.
"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from hifi_dc import MarigoldDepthCompletionPipelineWithNormals
from hifi_core import Config


class RealComponents(unittest.TestCase):
    def test_local_save_load_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'vocab.json').write_text(json.dumps({'<|startoftext|>':0, '<|endoftext|>':1}))
            (path/'merges.txt').write_text('#version: 0.2\n')
            tokenizer = CLIPTokenizer(str(path/'vocab.json'), str(path/'merges.txt'), model_max_length=8)
            text = CLIPTextModel(CLIPTextConfig(vocab_size=2, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                bos_token_id=0, eos_token_id=1, pad_token_id=1))
            vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
                up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'),
                block_out_channels=(8,8), layers_per_block=1, norm_num_groups=4)
            unet = UNet2DConditionModel(sample_size=16, in_channels=8, out_channels=4,
                down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),
                up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
                block_out_channels=(8,8), layers_per_block=1, cross_attention_dim=8,
                attention_head_dim=2, norm_num_groups=4)
            pipe = MarigoldDepthCompletionPipelineWithNormals(unet, vae,
                DDIMScheduler(num_train_timesteps=100, clip_sample=False), text, tokenizer)
            model = path/'model'
            pipe.save_pretrained(model)
            loaded = MarigoldDepthCompletionPipelineWithNormals.from_pretrained(model, local_files_only=True,
                dsine_model=None, dsine_args=None, torch_dtype=torch.float32).to('cpu')
            image = Image.fromarray(np.full((32,48,3), 128, dtype=np.uint8))
            sparse = np.zeros((32,48), dtype=np.float32)
            sparse[::4, ::4] = np.linspace(.2, 2., 96).reshape(8,12)
            normals = np.zeros((32,48,3), dtype=np.float32)
            normals[:,:,2] = 1
            result = loaded(image=image, sparse_depth=sparse, normal_prior=normals,
                            num_inference_steps=3, processing_resolution=48,
                            config=Config(bilateral=False), return_details=True)
            self.assertEqual(result['depth'].shape, (32,48))
            self.assertTrue(np.isfinite(result['depth']).all())
            self.assertTrue(all(t['latent_gradient_norm'] > 0 for t in result['trace']))
            self.assertTrue(all(t['adam_max_update'] > 0 for t in result['trace']))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in loaded.unet.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
