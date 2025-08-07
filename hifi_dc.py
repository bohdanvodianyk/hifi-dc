# Copyright 2025 Bohdan Vodianyk, University of Málaga.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper:
# "HIFI-DC: High-Fidelity Detail-Preserving Depth Completion"
# BibTeX and project page available at:
# https://github.com/bohdanvdnyk/hifi-dc

import os, argparse, logging, warnings
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers import DDIMScheduler, MarigoldDepthPipeline
from diffusers.configuration_utils import FrozenDict
from typing import Optional, Tuple
import cv2

import projects.dsine.config as dsine_config
import utils.utils as dsine_utils

warnings.simplefilter("ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()


def load_local_dsine_model(ckpt: str, cfg: str, device: torch.device):
    args = dsine_config.get_args(test=True, config_path=cfg)
    if args.NNET_architecture == "v00":
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == "v01":
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == "v02":
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == "v02_kappa":
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise ValueError(f"Unsupported DSINE architecture: {args.NNET_architecture}")
    model = DSINE(args).to(device)
    model = dsine_utils.load_checkpoint(ckpt, model)
    model.eval()
    return model, args


def dsine_infer_normals(dsine_model, dsine_args, pil_img: Image.Image, device):
    from torchvision import transforms
    arr = np.array(pil_img).astype(np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, H, W = img.shape
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(
              F.pad(img, dsine_utils.get_padding(H, W), value=0))
    l, r, t, b = dsine_utils.get_padding(H, W)
    from utils.projection import intrins_from_fov
    intr = intrins_from_fov(60.0, H, W, device=device).unsqueeze(0)
    intr[:, 0, 2] += l;  intr[:, 1, 2] += t
    with torch.no_grad():
        n = dsine_model(img, intrins=intr)[-1][:, :, t:t+H, l:l+W]
    return n


class MarigoldDepthCompletionPipelineEnhanced(MarigoldDepthPipeline):

    # --- remove scalar keys coming from model_index.json ---------------------
    @classmethod
    def extract_init_dict(cls, cfg, **kw):
        init_d, unused, remain = super().extract_init_dict(cfg, **kw)
        for k in ("scale_invariant", "shift_invariant", "default_denoising_steps",
                  "default_processing_resolution", "prediction_type"):
            init_d.pop(k, None)
        return init_d, unused, remain

    # ------------------------------------------------------------------
    def __init__(self, *, unet, vae, scheduler, text_encoder, tokenizer,
                 dsine_model=None, dsine_args=None):
        super().__init__(unet=unet, vae=vae, scheduler=scheduler,
                         text_encoder=text_encoder, tokenizer=tokenizer)

        # replace internal FrozenDict with modules‑only version
        object.__setattr__(self, "_internal_dict", FrozenDict({
            "unet": unet, "vae": vae, "scheduler": scheduler,
            "text_encoder": text_encoder, "tokenizer": tokenizer
        }))

        self.dsine_model = dsine_model
        self.dsine_args  = dsine_args

    # ---------- Enhanced helpers ------------------------------------------------
    
    @staticmethod
    def edge_aware_depth_to_normals(d: torch.Tensor, edge_threshold: float = 0.1):
        # Sobel operators for more robust gradient estimation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=d.dtype, device=d.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=d.dtype, device=d.device).view(1, 1, 3, 3)
        
        # Pad depth for convolution
        d_pad = F.pad(d, (1, 1, 1, 1), mode='replicate')
        
        # Compute gradients using Sobel
        dx = F.conv2d(d_pad, sobel_x, padding=0) / 8.0
        dy = F.conv2d(d_pad, sobel_y, padding=0) / 8.0
        
        # Edge detection for adaptive weighting
        edges = torch.sqrt(dx**2 + dy**2)
        edge_weight = torch.exp(-edges / edge_threshold)
        
        # Weighted normal computation
        Nx, Ny, Nz = -dx * edge_weight, -dy * edge_weight, torch.ones_like(dx)
        norm = torch.sqrt(Nx**2 + Ny**2 + Nz**2 + 1e-8)
        
        return torch.cat([Nx/norm, Ny/norm, Nz/norm], 1)
    
    @staticmethod
    def multi_scale_cosine_loss(pred_normals, target_normals, scales=[1, 2, 4]):
        total_loss = 0.0
        weights = [1.0, 0.5, 0.25]  # Higher weight for finer scales
        
        for scale, weight in zip(scales, weights):
            if scale > 1:
                pred_s = F.avg_pool2d(pred_normals, scale, scale)
                target_s = F.avg_pool2d(target_normals, scale, scale)
            else:
                pred_s = pred_normals
                target_s = target_normals
            
            cosine_sim = (F.normalize(pred_s, dim=1) * F.normalize(target_s, dim=1)).sum(1)
            loss = (1 - cosine_sim).mean()
            total_loss += weight * loss
        
        return total_loss / sum(weights)
    
    @staticmethod
    def compute_confidence_weights(sparse_depth: torch.Tensor, kernel_size: int = 7):
        mask = (sparse_depth > 0).float()
        
        # Count nearby valid points
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=sparse_depth.device)
        local_density = F.conv2d(mask, kernel, padding=kernel_size//2)
        
        # Normalize to [0, 1]
        max_density = kernel_size * kernel_size
        confidence = local_density / max_density
        
        # Apply sigmoid for smooth transition
        confidence = torch.sigmoid(5 * (confidence - 0.3))
        
        return confidence * mask

    def predict_normals_from_image(self, pil, device):
        if self.dsine_model is None:
            raise RuntimeError("DSINE model missing.")
        return dsine_infer_normals(self.dsine_model, self.dsine_args, pil, device)[:, :3]

    # Enhanced loss functions
    @staticmethod
    def weighted_scale_inv_l2(p, t, m, w):
        d = (p - t)[m]
        weights = w[m]
        weighted_d = d * weights
        n_eff = weights.sum()
        
        if n_eff < 1e-6:
            return torch.tensor(0.0, device=p.device)
        
        return (weighted_d.pow(2).sum()/n_eff - 
                0.5*(weighted_d.sum()**2)/(n_eff**2))
    
    @staticmethod
    def adaptive_weight_schedule(step: int, total_steps: int, 
                               convergence_metric: Optional[float] = None) -> float:
        # Base schedule: ramp in last third
        base_schedule = max(0.0, min(1.0, (step - 0.66*total_steps) / (0.34*total_steps)))
        
        # Adaptive component based on convergence
        if convergence_metric is not None:
            # Higher weight if convergence is good
            adaptive_factor = 1.0 + 0.5 * np.tanh(5 * (0.1 - convergence_metric))
            return base_schedule * adaptive_factor
        
        return base_schedule

    # ---------- Enhanced forward method ------------------------------------------------
    def __call__(self, *, image: Image.Image, sparse_depth: np.ndarray,
                 num_inference_steps: int = 50, processing_resolution: int = 768,
                 seed: int = 2024, normal_weight: float = 0.1, lambda_l1: float = 0.5,
                 lsq_interval: int = 10, use_confidence_weighting: bool = True,
                 use_edge_aware_normals: bool = True, use_multi_scale_loss: bool = True,
                 bilateral_filter_d: int = 0, bilateral_sigma_color: float = 75.0,
                 bilateral_sigma_space: float = 75.0, 
                 estimate_uncertainty: bool = False,
                 progressive_resolution: bool = False):

        device = self._execution_device
        g = torch.Generator(device=device).manual_seed(seed)
        if sparse_depth.ndim != 2:
            raise ValueError("sparse_depth must be H×W")

        # empty prompt
        if self.empty_text_embedding is None:
            ids = self.tokenizer("", return_tensors="pt", padding="do_not_pad", truncation=True
            ).input_ids.to(device)
            self.empty_text_embedding = self.text_encoder(ids)[0]

        # Progressive resolution handling
        if progressive_resolution and processing_resolution > 384:
            # Start with lower resolution for initial steps
            initial_resolution = processing_resolution // 2
            switch_step = num_inference_steps // 3
        else:
            initial_resolution = processing_resolution
            switch_step = num_inference_steps

        img_t, pad, orig = self.image_processor.preprocess(
            image, processing_resolution=initial_resolution,
            device=device, dtype=self.dtype
        )
        if sparse_depth.shape != orig:
            raise ValueError("Sparse depth size mismatch.")

        with torch.no_grad():
            img_lat, init_lat = self.prepare_latents(img_t, None, g, 1, 1)
        depth_lat = torch.nn.Parameter(init_lat.detach())

        sparse_t = torch.from_numpy(sparse_depth).float().to(device)[None, None]
        mask = (sparse_t > 0) & torch.isfinite(sparse_t)
        v = sparse_t[mask]
        sparse_range = (v.max() - v.min()).item() if v.numel() > 1 else 1.0
        if sparse_range < 1e-6:
            sparse_range = 1.0

        # Compute confidence weights if enabled
        if use_confidence_weighting:
            confidence = self.compute_confidence_weights(sparse_t)
        else:
            confidence = torch.ones_like(sparse_t)

        # ---- affine parameters (exp‑scale) ----------------------------
        scale_raw = torch.nn.Parameter(torch.zeros(1, device=device))
        shift_raw = torch.nn.Parameter(torch.zeros(1, device=device))
        affine = lambda d: (torch.exp(scale_raw) + 1e-4) * sparse_range * d + shift_raw

        # LSQ warm‑start with confidence weighting
        with torch.no_grad():
            d0 = self.decode_prediction(init_lat)
            d0 = self.image_processor.resize_antialias(
                     self.image_processor.unpad_image(d0, pad),
                     orig, "bilinear", is_aa=False)
            ok = mask & torch.isfinite(d0)
            if ok.sum() > 3:
                A = d0[ok].view(-1, 1)
                B = sparse_t[ok].view(-1, 1)
                W = confidence[ok].view(-1, 1) if use_confidence_weighting else None
                
                if W is not None:
                    # Weighted least squares
                    AW = A * W
                    s, b = torch.linalg.lstsq(
                        torch.cat([AW, W], 1), B * W
                    ).solution.view(-1)
                else:
                    s, b = torch.linalg.lstsq(
                        torch.cat([A, torch.ones_like(A)], 1), B
                    ).solution.view(-1)
                    
                if torch.isfinite(s) and torch.isfinite(b):
                    scale_raw.data.copy_(torch.log(torch.clamp(s / sparse_range, min=1e-4)))
                    shift_raw.data.copy_(b)

        # DSINE normals
        dsine_n = self.predict_normals_from_image(
            TF.to_pil_image(img_t[0].cpu()), device
        )
        if dsine_n.shape[-2:] != img_t.shape[-2:]:
            dsine_n = F.interpolate(dsine_n, size=img_t.shape[-2:], mode="bilinear",
                                    align_corners=False)

        # optimisers ----------------------------------------------------------------
        opt_lat = torch.optim.Adam([depth_lat], lr=2e-2)
        
        # Tracking for adaptive scheduling
        prev_loss = float('inf')
        convergence_metrics = []

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        T = len(self.scheduler.timesteps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps, desc="Marigold+DSINE")):
            # Handle progressive resolution
            if progressive_resolution and i == switch_step:
                # Switch to full resolution
                img_t, pad, _ = self.image_processor.preprocess(
                    image, processing_resolution=processing_resolution,
                    device=device, dtype=self.dtype
                )
                with torch.no_grad():
                    # Resize latents
                    depth_lat_resized = F.interpolate(
                        depth_lat.data, size=img_t.shape[-2:]//8, mode='bilinear'
                    )
                    depth_lat = torch.nn.Parameter(depth_lat_resized)
                    img_lat = self.vae.encode(img_t).latent_dist.mode()
                    
                # Update optimizer
                opt_lat = torch.optim.Adam([depth_lat], lr=2e-2)
                
                # Update DSINE normals
                dsine_n = self.predict_normals_from_image(
                    TF.to_pil_image(img_t[0].cpu()), device
                )

            opt_lat.zero_grad()

            mix = torch.cat([img_lat, depth_lat], 1)
            eps = self.unet(mix, t, encoder_hidden_states=self.empty_text_embedding,
                            return_dict=False)[0]

            with torch.no_grad():
                pred_orig = self.scheduler.step(
                    eps, t, depth_lat, generator=g
                ).pred_original_sample

            # analytic LSQ update with confidence weighting
            if (i % lsq_interval) == 0:
                with torch.no_grad():
                    d_aff_tmp = self.decode_prediction(pred_orig)
                    d_aff_tmp = self.image_processor.resize_antialias(
                        self.image_processor.unpad_image(d_aff_tmp, pad),
                        orig, "bilinear", is_aa=False)
                    ok = mask & torch.isfinite(d_aff_tmp)
                    if ok.sum() > 3:
                        A = d_aff_tmp[ok].view(-1, 1)
                        B = sparse_t[ok].view(-1, 1)
                        W = confidence[ok].view(-1, 1) if use_confidence_weighting else None
                        
                        if W is not None:
                            AW = A * W
                            s, b = torch.linalg.lstsq(
                                torch.cat([AW, W], 1), B * W
                            ).solution.view(-1)
                        else:
                            s, b = torch.linalg.lstsq(
                                torch.cat([A, torch.ones_like(A)], 1), B
                            ).solution.view(-1)
                            
                        scale_raw.data.copy_(torch.log(torch.clamp(s / sparse_range, min=1e-4)))
                        shift_raw.data.copy_(b)

            # build current affine‑aligned depth for loss terms
            d_aff = self.decode_prediction(pred_orig)
            d_aff = self.image_processor.resize_antialias(
                        self.image_processor.unpad_image(d_aff, pad),
                        orig, "bilinear", is_aa=False)
            d_m = torch.nan_to_num(affine(d_aff), nan=0.0, posinf=1e4, neginf=0.0)

            # Compute convergence metric
            current_loss = F.l1_loss(d_m[mask], sparse_t[mask]).item()
            convergence_metric = abs(prev_loss - current_loss) / (prev_loss + 1e-8)
            convergence_metrics.append(convergence_metric)
            prev_loss = current_loss

            # Adaptive weight scheduling
            w_n = normal_weight * self.adaptive_weight_schedule(
                i, T, convergence_metric if len(convergence_metrics) > 5 else None
            )

            # Compute losses
            if use_confidence_weighting:
                data_loss = self.weighted_scale_inv_l2(d_m, sparse_t, mask, confidence)
                l1_loss = (F.l1_loss(d_m[mask], sparse_t[mask], reduction='none') * 
                          confidence[mask]).mean()
            else:
                data_loss = self.scale_inv_l2(d_m, sparse_t, mask)
                l1_loss = F.l1_loss(d_m[mask], sparse_t[mask])
            
            # Normal consistency loss
            if use_edge_aware_normals:
                pred_normals = self.edge_aware_depth_to_normals(d_m)
            else:
                pred_normals = self.depth_to_normals(d_m)
            
            target_normals = F.interpolate(dsine_n, size=d_m.shape[-2:], mode='bilinear',
                                         align_corners=False)
            
            if use_multi_scale_loss:
                normal_loss = self.multi_scale_cosine_loss(pred_normals, target_normals)
            else:
                normal_loss = self.cosine_loss(pred_normals, target_normals)
            
            loss = data_loss + lambda_l1 * l1_loss + w_n * normal_loss
            loss.backward()

            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([depth_lat], 1.0)
            opt_lat.step()

            with torch.no_grad():
                depth_lat.set_(self.scheduler.step(
                    eps, t, depth_lat, generator=g).prev_sample)

        # final decode + LSQ re‑fit ------------------------------------------------
        with torch.no_grad():
            d = self.image_processor.resize_antialias(
                    self.image_processor.unpad_image(
                        self.decode_prediction(depth_lat), pad),
                    orig, "bilinear", is_aa=False)
            d = torch.nan_to_num(affine(d), nan=0.0, posinf=1e4, neginf=0.0)
            ok = mask & torch.isfinite(d)
            if ok.sum() > 3:
                A = d[ok].view(-1, 1)
                B = sparse_t[ok].view(-1, 1)
                W = confidence[ok].view(-1, 1) if use_confidence_weighting else None
                
                if W is not None:
                    AW = A * W
                    s, b = torch.linalg.lstsq(
                        torch.cat([AW, W], 1), B * W
                    ).solution.view(-1)
                else:
                    s, b = torch.linalg.lstsq(
                        torch.cat([A, torch.ones_like(A)], 1), B
                    ).solution.view(-1)
                d = s * d + b

        out = self.image_processor.pt_to_numpy(d).squeeze()
        
        # Optional bilateral filtering
        if bilateral_filter_d > 0:
            out_filtered = cv2.bilateralFilter(
                out.astype(np.float32), 
                bilateral_filter_d, 
                bilateral_sigma_color, 
                bilateral_sigma_space
            )
            out = out_filtered
        
        if not np.isfinite(out).any():
            logging.warning("All-NaN output – returning zeros.")
            out = np.zeros_like(out)
        
        # Uncertainty estimation if requested
        if estimate_uncertainty:
            # Simple uncertainty based on distance to nearest sparse point
            from scipy.ndimage import distance_transform_edt
            sparse_mask = sparse_depth > 0
            distance_to_sparse = distance_transform_edt(~sparse_mask)
            
            # Normalize and invert (closer = more certain)
            max_dist = distance_to_sparse.max()
            uncertainty = distance_to_sparse / (max_dist + 1e-6)
            
            # Also consider convergence metrics
            avg_convergence = np.mean(convergence_metrics[-10:]) if len(convergence_metrics) > 10 else 0.1
            uncertainty = uncertainty * (1 + avg_convergence)
            
            return out, uncertainty
        
        return out

# ═════════════════════════════════════ MAIN ═════════════════════════════════

def main():
    parser = argparse.ArgumentParser("Enhanced Marigold + DSINE Normal Consistency (v4.1)")
    parser.add_argument("--in-image",  default="data/image.png")
    parser.add_argument("--in-depth",  default="data/sparse_1000.npy")
    parser.add_argument("--out-depth", default="data/dense_depth_new_1000.npy")
    parser.add_argument("--checkpoint", default="prs-eth/marigold-depth-v1-0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--processing_resolution", type=int, default=768)
    parser.add_argument("--normal_weight", type=float, default=0.1)
    parser.add_argument("--lambda_l1", type=float, default=0.5)
    
    # New v4.1 arguments
    parser.add_argument("--use_confidence_weighting", action="store_true", 
                       help="Use confidence-based weighting for sparse points")
    parser.add_argument("--use_edge_aware_normals", action="store_true",
                       help="Use edge-aware normal computation")
    parser.add_argument("--use_multi_scale_loss", action="store_true",
                       help="Use multi-scale normal consistency loss")
    parser.add_argument("--bilateral_filter_d", type=int, default=0,
                       help="Bilateral filter diameter (0 to disable)")
    parser.add_argument("--estimate_uncertainty", action="store_true",
                       help="Estimate uncertainty map for completed depth")
    parser.add_argument("--progressive_resolution", action="store_true",
                       help="Use progressive resolution for faster inference")
    
    parser.add_argument("--dsine-ckpt",
        default="/content/drive/MyDrive/ColabNotebooks/marigold_dsine/projects/dsine/checkpoints/exp002_kappa/dsine_plus_scannet.pt")
    parser.add_argument("--dsine-config",
        default="/content/drive/MyDrive/ColabNotebooks/marigold_dsine/projects/dsine/experiments/exp002_kappa/dsine_plus_scannet.txt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("Running on CPU – slow.")

    dsine_model, dsine_args = load_local_dsine_model(
        args.dsine_ckpt, args.dsine_config, device
    )
    logging.info(f"Loading Marigold checkpoint: {args.checkpoint}")
    pipe = MarigoldDepthCompletionPipelineEnhanced.from_pretrained(
        args.checkpoint, dsine_model=dsine_model, dsine_args=dsine_args
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    if device.type == "cpu":
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device)

    img    = Image.open(args.in_image).convert("RGB")
    sparse = np.load(args.in_depth)
    logging.info(f"Input  : {args.in_image}")
    logging.info(f"Sparse : {args.in_depth}  {sparse.shape}")

    result = pipe(
        image=img, sparse_depth=sparse,
        num_inference_steps=args.num_inference_steps,
        processing_resolution=args.processing_resolution,
        normal_weight=args.normal_weight,
        lambda_l1=args.lambda_l1,
        use_confidence_weighting=args.use_confidence_weighting,
        use_edge_aware_normals=args.use_edge_aware_normals,
        use_multi_scale_loss=args.use_multi_scale_loss,
        bilateral_filter_d=args.bilateral_filter_d,
        estimate_uncertainty=args.estimate_uncertainty,
        progressive_resolution=args.progressive_resolution
    )
    
    if args.estimate_uncertainty:
        dense, uncertainty = result
        np.save(args.out_depth.replace('.npy', '_uncertainty.npy'), uncertainty)
        logging.info(f"Saved uncertainty: {args.out_depth.replace('.npy', '_uncertainty.npy')}")
    else:
        dense = result
    
    np.save(args.out_depth, dense)
    logging.info(f"Saved  : {args.out_depth}")

    fin = np.isfinite(dense)
    if not fin.any():
        logging.warning("Dense depth has no finite values – skipping preview.")
    else:
        vmin, vmax = float(dense[fin].min()), float(dense[fin].max())
        if vmin == vmax: vmax = vmin + 1.0
        vis = pipe.image_processor.visualize_depth(
                np.nan_to_num(dense, nan=vmin, posinf=vmax, neginf=vmin),
                val_min=vmin, val_max=vmax)[0]
        out_vis = os.path.splitext(args.out_depth)[0] + "_vis.jpg"
        vis.save(out_vis)
        logging.info(f"Preview: {out_vis}")
        
        # Also save uncertainty visualization if available
        if args.estimate_uncertainty:
            uncertainty_vis = (uncertainty * 255).astype(np.uint8)
            uncertainty_img = Image.fromarray(uncertainty_vis, mode='L')
            uncertainty_vis_path = os.path.splitext(args.out_depth)[0] + "_uncertainty_vis.jpg"
            uncertainty_img.save(uncertainty_vis_path)
            logging.info(f"Uncertainty preview: {uncertainty_vis_path}")

if __name__ == "__main__":
    main()