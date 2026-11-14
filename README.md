# HiFi-DC: high-fidelity detail-preserving depth completion

HiFi-DC densifies sparse metric depth (LiDAR, ToF, SfM points) into a full depth map guided by an
RGB image. It is zero-shot: a frozen image-conditioned latent diffusion depth prior (Marigold) is
steered at test time by weighted least-squares metric anchoring, confidence-weighted sparse
losses and convergence-aware multi-scale surface-normal guidance (DSINE), with a coarse-to-fine
resolution schedule.

## HiFi-DC v2 (release `v2.0`)

v2 builds on the Marigold-DC test-time solver (`hifidc/v2/`) and adds, in this order: a unit-free
median-normalised input, light DSINE surface-normal guidance, a robust multiplicative low-frequency
correction fitted to the sparse points (`log_residual_field`), and an RGB-guided filter on the final
log-depth. Config: [`configs/v2/candidate.yaml`](configs/v2/candidate.yaml).

On 80 held-out scenes from 8 datasets (NYUv2, VOID, Hypersim, InteriorNet, ICL-NUIM, Sintel, KITTI,
vKITTI2; 384 px, never used for tuning, evaluated once), AbsRel relative to Marigold-DC
(< 1 = better, 95 % scene-clustered bootstrap CI):

| cases | v2 / Marigold-DC | real indoor / synthetic / driving |
|---|---|---|
| 5 sparse patterns, 1000 points (400) | **0.803 [0.775, 0.830]** | 0.871 / 0.846 / 0.666 |
| 100 / 300 points, 5 % outliers (336) | **0.848 [0.828, 0.868]** | 0.863 / 0.847 / 0.650 |

Every group and metric (AbsRel, scale-invariant log RMSE, δ1, held-out point error) improves, at
Marigold-DC's runtime plus about 1 s for DSINE. Details, development results and the one post-test fix:
[`docs/V2_RESULTS.md`](docs/V2_RESULTS.md); every keep/drop decision: [`docs/V2_LADDER.md`](docs/V2_LADDER.md).

```bash
python scripts/cache_normals.py --data data/test --out cache/normals
python scripts/run_benchmark.py --method hifidc_v2 --v2-config configs/v2/candidate.yaml \
    --cases dev50 --resolution 384 --out results/v2_candidate
```

From Python, `hifidc.bench.make_hifidc_v2_method(config, device)` returns
`method(scene, normals, seed, resolution) -> (depth, info)`; the pipeline itself is
`hifidc.v2.pipeline.HiFiDCv2Pipeline`.

## v1.1-corrected

> **Status — v1.1-corrected (tag `v1.1-corrected`).** In the originally released `hifi_dc.py`, the
> clean-sample prediction used for the guidance losses was computed under `torch.no_grad()`, so
> the losses never reached the optimised latent: the output was plain Marigold depth plus a global
> affine fit. This branch contains a corrected, paper-guided implementation in which the
> guidance is live (and a test fails if it ever goes inert again), the weighted fits use the
> confidence weights exactly once, every publication setting lives in `configs/paper_v1.yaml`, and
> runs are reproducible from a pinned environment. Published numbers were produced by the original
> code and have not yet been re-established with this implementation; see `docs/BASELINE.md`.

## Install

```bash
conda env create -f environment.yml        # Python 3.12, torch 2.8.0 (CUDA 12.8), diffusers 0.35.1
conda activate hifidc-v2
pip install -e ".[dev,dsine]"
bash scripts/setup_external.sh             # pinned DSINE (+ import shims) and Marigold-DC into external/
```

DSINE weights are not downloaded automatically. Put the `exp002_kappa` checkpoint
(`dsine_plus_scannet.pt`) under `external/DSINE/projects/dsine/checkpoints/exp002_kappa/`, or run
`HIFIDC_DSINE_CKPT_SRC=<your DSINE projects/dsine dir> bash scripts/setup_external.sh dsine` to link it.
Marigold weights (`prs-eth/marigold-depth-v1-0`, pinned revision) download on first use.

## Run on one image

```bash
python hifi_dc.py --in-image image.png --in-depth sparse.npy --out-depth out/dense.npy \
    --depth-unit m --processing_resolution 768
```

`sparse.npy` is an `H×W` float array with `0` where there is no measurement. The output folder gets
the dense depth, a heuristic uncertainty map and a JSON manifest (inputs, code hashes, per-step
diagnostics). Settings come from `configs/paper_v1.yaml`; pass `--config other.yaml` to change them.
From Python:

```python
from hifi_dc import MarigoldDepthCompletionPipelineWithNormals
pipe = MarigoldDepthCompletionPipelineWithNormals.from_pretrained(
    "prs-eth/marigold-depth-v1-0", dsine_model=dsine, dsine_args=dsine_args, torch_dtype=torch.float32).to("cuda")
dense = pipe(image=rgb_pil, sparse_depth=sparse)   # or normal_prior=<HxWx3 normals> instead of DSINE
```

The method keeps the full UNet/VAE backward graph in float32; on an 8 GB GPU use
`--processing_resolution 384` or lower.

## Benchmark

```bash
python scripts/cache_normals.py                                   # DSINE normals once per scene
python scripts/run_benchmark.py --method hifidc --resolution 384 --seeds 2024 --out results/my_run
python scripts/run_benchmark.py --method marigold_dc --resolution 768 --out results/mdc_768
```

Scenes, depth units and per-scene invalid values are declared in `configs/scenes.yaml`; invalid
values are removed from both the sparse input and the ground truth. Each run writes
`results/<run_id>/{manifest.json, metrics.csv, summary.json, pred/, trace/}`. Native-unit errors
(RMSE, MAE) are reported per scene only; cross-scene summaries use scale-free metrics
(AbsRel, δ1, scale-invariant log RMSE). Large runs: see [`docs/COLAB.md`](docs/COLAB.md).

## Tests

```bash
python -m pytest -m "not gpu"      # CPU: solver, regression, config, data, metrics, runner tests
python -m pytest -m gpu            # needs CUDA, weights, local data and cached normals
```

## Citation

```bibtex
@article{vodianyk2026hifidc,
  title   = {{HIFI-DC}: High-fidelity detail-preserving depth completion},
  author  = {Vodianyk, Bohdan and Nava-Baro, Enrique and Popov, Anton},
  journal = {Image and Vision Computing},
  volume  = {173},
  pages   = {106024},
  year    = {2026},
  doi     = {10.1016/j.imavis.2026.106024}
}
```

## License

Apache-2.0 (see `LICENSE`). DSINE and Marigold-DC are fetched separately under their own licences.
