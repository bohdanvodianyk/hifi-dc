# HiFi-DC v2 — Phase 1 results

Goal: beat Marigold-DC (MDC) on accuracy, and on speed where possible, at low resolution (384 px).
All numbers are ratios of the scene-clustered geometric mean AbsRel, ours / MDC (< 1 = ours better),
with a 95 % scene-clustered bootstrap CI. Scoring: finite, depth > 0, not-sentinel valid GT only, per
scene in its native units (never pooled across scenes). Decisions per rung are in `V2_LADDER.md`.

## The two frozen settings

| name | config | what it is | runtime / case (384 px, local 8 GB laptop GPU) |
|---|---|---|---|
| `cand` | `configs/v2/candidate.yaml` | MDC solver + median-normalised input (L1) + DSINE normal guidance w 0.025 (L3) + robust log residual field (L7) + RGB-guided log-depth filter r 2 (L6) | 21.5 s (MDC 21.2 s) + ~1 s DSINE |
| `fast` | `configs/v2/fast.yaml` | L1 with 25 steps and 2× learning rates + L7 + L6, no normals | ~11 s |

L7 and L6 are post-hoc (< 0.1 s): they can be applied to any depth-completion output.

## Development data (10 local scenes; used for tuning)

| set | `cand` vs MDC | real_indoor / synthetic / driving | `fast` vs MDC |
|---|---|---|---|
| dev50 (5 patterns × 10 scenes), seed 2024 | **0.829 [0.746, 0.901]** | 0.902 / 0.873 / 0.631 | 1.007 [0.970, 1.048] |
| dev50, seed 2025 (offline) | **0.828 [0.757, 0.886]** | 0.874 / 0.863 / 0.683 | — |
| stress48 (100 / 300 points, 5 % outliers) | **0.843 [0.783, 0.896]** | 0.882 / 0.848 / 0.715 | — |

Where the gain comes from (dev50, vs the previous kept rung): L1 0.980, L3 0.983–0.986,
L6 ≈ 0.91, L7 0.921. The filter and field fix the two visible MDC failure modes: blocky texture
copied from the 384 latent grid (L6) and smooth low-frequency scale drift between the sparse points
(L7, robust to outliers in the sparse input).

## New scenes (Colab, `main_datasets_new`, never used for tuning)

Dev split (3 scenes per dataset): L1 0.988 [0.971, 1.004]; L1 + L6 0.894 [0.863, 0.922] (patterns),
0.906 (stress). End-to-end `cand` and `fast` on this split: **pending** (notebook cell 8).

Test split (10 scenes per dataset, run once with the frozen settings): **pending**.

| set | `cand` vs MDC | groups | `fast` vs MDC | groups |
|---|---|---|---|---|
| test patterns | | | | |
| test stress | | | | |

## Figures

Side by side (RGB, sparse input, GT; MDC vs `cand` depth and relative error):
`results/viz/dev50_cand/` (one pattern per scene) and `results/viz/stress48_cand/` (100 points and
outlier cases). New-scene figures are written by the notebook to `MyDrive/hifidc-v2/newdata_v1/viz/`.

## What did not help

Huber anchoring (L2), fewer steps at the same learning rate (L5), the inert v1 "prior fit" output and
blends of it (copies image texture into depth), joint bilateral upsampling, edge-protected and
prior-guided filters. Details and numbers: `V2_LADDER.md`, "Tried and not kept".
