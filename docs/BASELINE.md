# v1.1-corrected baseline

Reference numbers for the corrected HiFi-DC v1 (`configs/paper_v1.yaml`) on the 10-scene
diagnostic set, next to a guidance-off control and the official Marigold-DC. Every later v2
change is compared against these runs. This is a diagnostic A/B set, not a leaderboard: 10 scenes,
one seed, native units per scene.

## Local runs (8 GB laptop GPU, 384 px)

| run | command | commit |
|---|---|---|
| A, B | `python scripts/run_benchmark.py --method hifidc --resolution 384 --seeds 2024 --out results/local_hifidc_paper_v1_384_{a,b}` | `152a45a` |
| C (guidance off) | same with `--override learning_rate=1e-12` | `152a45a` |
| D (Marigold-DC) | `python scripts/run_benchmark.py --method marigold_dc --resolution 384 --seeds 2024 --out results/local_marigold_dc_384` | `152a45a` |

Environment: Python 3.12.15, torch 2.8.0+cu128, diffusers 0.35.1, NVIDIA RTX 2000 Ada Laptop (8 GB).
HiFi-DC runs in float32; Marigold-DC uses its own default (bfloat16, 50 steps, ensemble 1).
Deterministic mode (`configure_determinism`: deterministic algorithms, cuBLAS workspace, math
attention) is on; run B reproduces run A **bit-for-bit** across separate processes.

### Scale-free metrics per scene (AbsRel / δ1 / si-log RMSE)

"unit?" marks scenes whose depth unit is a guess (`configs/scenes.yaml`); their scale-free metrics
are unaffected by the guess, but the method's unit-sensitive hyper-parameters are.

| scene | unit? | HiFi-DC v1.1 (A) | guidance off (C) | Marigold-DC (D) |
|---|---|---|---|---|
| 2011_10_03_drive_0047_sync_0000000791_03 (KITTI) |  | 0.387 / 0.516 / 0.526 | 0.648 / 0.231 / 0.567 | 0.078 / 0.946 / 0.128 |
| Scene20_00836 (vKITTI2) |  | 0.324 / 0.715 / 0.304 | 1.828 / 0.103 / 1.187 | 0.288 / 0.495 / 0.389 |
| 3FO4MMLDUKDJ_Living_room_8 (InteriorNet) |  | 0.042 / 0.976 / 0.077 | 0.118 / 0.945 / 0.168 | 0.011 / 0.997 / 0.030 |
| ai_055_004_90 (Hypersim) | yes | 0.229 / 0.582 / 0.316 | 0.290 / 0.548 / 0.371 | 0.096 / 0.881 / 0.182 |
| market_2_frame_0017 (Sintel) |  | 0.443 / 0.418 / 0.467 | 1.285 / 0.235 / 0.806 | 0.130 / 0.868 / 0.228 |
| office_4_1240 (ICL-NUIM?) | yes | 0.317 / 0.933 / 0.297 | 0.433 / 0.814 / 0.323 | 0.325 / 0.941 / 0.780 |
| student_lounge_0001_out_379 (NYU) | yes | 0.077 / 0.950 / 0.183 | 0.074 / 0.958 / 0.102 | 0.013 / 0.999 / 0.026 |
| study_room_0004_out_305 (NYU) | yes | 0.112 / 0.864 / 0.201 | 0.416 / 0.371 / 0.429 | 0.028 / 0.989 / 0.058 |
| visionlab1_1552089403.9210 (VOID) |  | 0.148 / 0.761 / 0.240 | 0.322 / 0.411 / 0.336 | 0.036 / 0.983 / 0.078 |
| workbench0_1551915136.1459 (VOID) |  | 0.384 / 0.692 / 0.397 | 2.859 / 0.116 / 1.177 | 0.059 / 0.982 / 0.165 |
| **indoor mean (8)** |  | **0.219 / 0.772 / 0.272** | 0.725 / 0.550 / 0.464 | **0.088 / 0.955 / 0.193** |
| **outdoor mean (2)** |  | **0.355 / 0.615 / 0.415** | 1.238 / 0.167 / 0.877 | **0.183 / 0.720 / 0.259** |

### Native-unit errors per scene (RMSE / MAE / mean |pred − sparse| at the anchors)

Never averaged across scenes (units differ).

| scene | HiFi-DC v1.1 (A) | guidance off (C) | Marigold-DC (D) |
|---|---|---|---|
| KITTI (÷256 m) | 2204 / 1264 / 1247 | 2637 / 1953 / 1973 | 709.8 / 295.9 / 212 |
| vKITTI2 (cm) | 1132 / 644.1 / 547.7 | 3858 / 3179 / 3111 | 1056 / 567 / 381.2 |
| InteriorNet (mm) | 202.3 / 133.4 / 129.4 | 423.3 / 327.5 / 334 | 76.37 / 33.42 / 25.4 |
| Hypersim | 22.75 / 16.58 / 16.27 | 28.18 / 18.98 / 19.76 | 16.33 / 7.305 / 2.605 |
| Sintel | 13.31 / 9.28 / 8.684 | 22.27 / 18.02 / 17.83 | 6.453 / 3.286 / 1.789 |
| office_4 | 1523 / 668.5 / 589.8 | 2393 / 1757 / 1816 | 1633 / 556.7 / 457.6 |
| student_lounge | 11.58 / 6.604 / 6.566 | 9.654 / 6.854 / 7.043 | 2.546 / 1.208 / 0.8686 |
| study_room | 14.08 / 9.636 / 8.816 | 47.43 / 34.2 / 34.1 | 5.792 / 2.907 / 1.869 |
| visionlab1 (÷256 m) | 166.8 / 97.13 / 93.12 | 273.8 / 206.7 / 221.3 | 71.36 / 30.19 / 19.73 |
| workbench0 (÷256 m) | 195.5 / 113.8 / 114.9 | 624.8 / 545.2 / 547.1 | 94.15 / 30.51 / 18.35 |

### Runtime (384 px, 10 scenes)

| run | s / image | peak VRAM | UNet calls / image |
|---|---|---|---|
| HiFi-DC v1.1 (A) | 19.2 | 6.91 GB | 100 (2 per step) |
| guidance off (C) | 18.7 | 6.91 GB | 100 |
| Marigold-DC (D) | 18.7 | 6.23 GB | 50 (1 per step) |

## What the baseline shows

1. **The corrected test-time optimisation is live and matters.** Turning guidance off (C) is far
   worse than the corrected method (indoor AbsRel 0.725 vs 0.219; anchor residual 2–5× larger).
2. **Corrected v1 is well behind official Marigold-DC** on 9 of 10 scenes (indoor AbsRel 0.219 vs
   0.088), and it fits the sparse anchors much less tightly (e.g. KITTI 1247 vs 212 native).
3. **Likely cause — the global affine anchor.** Per-step traces (`results/.../trace/`) show:
   - the warm-start fit at step 1 is made from an almost pure-noise prediction; its slope is
     unconstrained and comes out **negative on KITTI (−24) and Sintel (−29)** and near zero on
     Hypersim (0.03), so those depth maps are fitted upside-down or degenerate;
   - the scale is refitted only every 10 steps with EMA 0.5 and keeps moving to the end
     (workbench 2.9 → 6.9), so the latent chases a moving target and the data loss is still
     falling at step 50.
   Marigold-DC instead optimises a positive (squared) scale and shift jointly with the latent at
   every step. This makes a robust, positivity-constrained anchor the first v2 change to test.
4. **office_4 is dominated by 629 near-zero GT pixels** (native 45–450 on the scene's regular
   50-unit ladder, i.e. < 0.1 m at the guessed unit). Excluding GT < 500: AbsRel A 0.317 → 0.097,
   C 0.433 → 0.193, D 0.325 → 0.103 (δ1 nearly unchanged). They are kept in the official numbers
   because the scoring rule is finite & > 0 & not-sentinel; whether to gate them is an open
   decision.
5. **Caveats.** One seed; 384 px instead of the paper's 768 px (the A100 runs below close this);
   six GTs are coarsely quantised (96–420 distinct values), which puts a floor under their errors;
   four scenes have guessed units.

## Colab A100 runs (768 px, 5 seeds; 384 px cross-check; Marigold-DC 768 px)

Pending — run `notebooks/colab_a100.ipynb` (see `docs/COLAB.md`) at the commit above and add the
results here.
