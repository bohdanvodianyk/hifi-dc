# HiFi-DC v2 — Phase 1 results

Goal: beat Marigold-DC (MDC) on accuracy, and on speed where possible, at low resolution (384 px).
All numbers are ratios of the scene-clustered geometric mean AbsRel, ours / MDC (< 1 = ours better),
with a 95 % scene-clustered bootstrap CI. Scoring: finite, depth > 0, not-sentinel valid GT only, per
scene in its native units (never pooled across scenes). Decisions per rung are in `V2_LADDER.md`.

## The two frozen settings

| name | config | what it is | runtime / case (384 px, local 8 GB laptop GPU) |
|---|---|---|---|
| `cand` | `configs/v2/candidate.yaml` | MDC solver + median-normalised input (L1) + DSINE normal guidance w 0.025 (L3) + robust log residual field (L7) + RGB-guided log-depth filter r 2 (L6) | 21.5 s (MDC 21.2 s) + ~1 s DSINE |
| `fast` (dropped) | `configs/v2/fast.yaml` | L1 with 25 steps and 2× learning rates + L7 + L6, no normals | ~11 s; 1.06 vs MDC on new scenes |

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

Dev split (3 scenes per dataset, 24 scenes; Colab L4, end to end):

| set | `cand` vs MDC | real_indoor / synthetic / driving | si-log / δ1 / held-out | `fast` vs MDC | L1 vs MDC |
|---|---|---|---|---|---|
| dev patterns (120 cases) | **0.821 [0.781, 0.863]** | 0.881 / 0.856 / 0.704 | 0.845 / 0.825 / 0.834 | 1.060 [1.013, 1.113] | 0.983 |
| dev stress (114 cases) | **0.847 [0.813, 0.881]** | 0.885 / 0.861 / 0.743 | 0.866 / 0.851 / 0.849 | 1.057 [1.013, 1.108] | 0.979 |

`cand` reproduces the local gain on unseen scenes (local 0.829 / 0.843). `fast` does not: it tied MDC
on the local scenes (1.007) but is 6 % worse here (real_indoor 1.16), so it is dropped from the test.

Test split (10 scenes per dataset, 80 scenes, never seen before; run once with the frozen `cand`,
Colab L4, after the dev gate passed). `fast` was already dropped but ran too (the notebook in that
session predated the change); it is reported for completeness.

| set | `cand` vs MDC | real_indoor / synthetic / driving | si-log / δ1 / held-out | `fast` vs MDC |
|---|---|---|---|---|
| test patterns (400 cases) | **0.803 [0.775, 0.830]** | 0.871 / 0.846 / 0.666 | 0.836 / 0.765 / 0.846 | 1.041 [1.010, 1.073] |
| test stress (336 cases) | **0.848 [0.828, 0.868]** | 0.863 / 0.847 / 0.650 | 0.869 / 0.837 / 0.880 | 1.055 [1.023, 1.089] |

**Result:** on held-out scenes the frozen candidate has about 20 % lower AbsRel than Marigold-DC with
5 sparse patterns and about 15 % lower under stress (100 / 300 points, 5 % outliers). Every group and
every metric improves, and the CIs exclude 1. The local, new-dev and test estimates agree
(0.829 → 0.821 → 0.803; stress 0.843 → 0.847 → 0.848), so the gain is not a tuning artefact.
Largest gain on driving (KITTI, vKITTI2: ~0.66), smallest on real indoor (~0.87).
`fast` confirms its dev result: halving the steps costs more than L7 + L6 give back.

**Post-test fix to L7 (found in the final code review).** The test ran code in which the Tukey
weights used the raw residual (not centred at its median) with a scale floor of 1e-4. On near-noise-free
synthetic inputs that rejected clean points (10 % scale error, 50 clean points: 0.022–0.030 AbsRel vs
0.011 for the plain fit). Fixed: centred residual, floor 0.01 in log units
(`tests/test_post_field.py::test_robust_fit_matches_plain_fit_on_clean_points`). On real data the
change is within 0.002: dev50 0.829 → 0.827 (s2024), 0.828 → 0.827 (s2025); stress48 (L1 base)
0.854 → 0.853; the 5 %-outlier cases are unchanged (0.768). The test numbers above are for the code
as run (commit `84f8aa1`); the fixed method was not re-run on the test split, which stays used once.

**Known notebook limitation (not fixed; the test is already run):** the dev gate checks the comparison
JSON on Drive but not the git SHA or config of the `cand` run that produced it, so a stale dev run could
authorise a later test run. In this session the dev `cand` run was produced fresh, by the same code as
the test run.

## Figures

Side by side (RGB, sparse input, GT; MDC vs `cand` depth and relative error):
`results/viz/dev50_cand/` (one pattern per scene) and `results/viz/stress48_cand/` (100 points and
outlier cases). New-scene figures (dev 24 + 57, test 80 + 168 PNGs, MDC / `cand` / `fast`) are in
`MyDrive/hifidc-v2/newdata_v1/viz/`; the comparison JSONs are in `MyDrive/hifidc-v2/newdata_v1/`.

## What did not help

Huber anchoring (L2), fewer steps at the same learning rate (L5), the inert v1 "prior fit" output and
blends of it (copies image texture into depth), joint bilateral upsampling, edge-protected and
prior-guided filters. Details and numbers: `V2_LADDER.md`, "Tried and not kept".
