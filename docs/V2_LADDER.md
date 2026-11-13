# HiFi-DC v2 ladder — decisions and evidence

Development set: `dev50` (10 scenes × 5 sparse patterns, 20 % of points held out), 384 px, seed 2024
unless noted. Ratios are A/B of the scene-clustered geometric mean (< 1 = A better), 95 % CI from a
scene-clustered bootstrap (`scripts/compare_runs.py`). Groups: real_indoor (NYUv2, VOID),
synthetic (Hypersim, InteriorNet, ICL-NUIM, Sintel), driving (KITTI, vKITTI2).

Rung rule: keep a rung if its AbsRel ratio vs the last kept rung is < 1 and no group is worse by
> 5 %. Because the seed noise floor is about ±4 % (below), a close call (|ratio − 1| < 0.04) is
re-run with a second seed before it is kept.

## Reference and noise floor

| comparison | AbsRel ratio [95 % CI] | si-log | held-out residual | notes |
|---|---|---|---|---|
| L0 (ours) vs official Marigold-DC | **identical** | — | — | 50/50 predictions bit-equal (`tests/test_gpu_v2.py` passes) |
| MDC seed 2025 vs 2024 | 0.974 [0.931, 1.014] | 0.960 | 0.995 | noise floor ≈ ±4 % |
| MDC-norm vs MDC | 0.980 [0.943, 1.012] | 1.001 | 0.993 | driving 0.894, real_indoor 1.016 |
| corrected v1 vs MDC | 2.839 [2.010, 3.914] | 2.172 | 2.674 | `paper_v1`, for reference |

## Rungs

| rung | change | vs | AbsRel ratio [95 % CI] | groups (ri / syn / drv) | decision |
|---|---|---|---|---|---|
| L0 | Marigold-DC solver in `hifidc.v2` | MDC | 1.000 (bit-equal) | — | base |
| L1 | median-normalised input | L0 (= MDC) | 0.980 [0.943, 1.012] | 1.016 / 0.989 / 0.894 | **keep** — no group worse > 5 %, removes dependence on guessed units |
| L1 (stress48) | median-normalised input | MDC | 0.969 [0.933, 1.001] | 0.992 / 0.974 / 0.884 | kept; largest per-case gains are the 5 %-outlier cases (up to 0.76) |
| L2 | Huber IRLS anchor, k = 1.345 | L1 | 1.090 [1.029, 1.157] | 1.114 / 1.120 / 0.988 | **drop** — also worse on stress48 (1.095 [1.033, 1.160]), even with 5 % outliers; downweighting large-but-legitimate residuals slows the metric fit |
| L2b | Huber, k = 3.0 | L1 | 1.012 [0.970, 1.055] | 1.039 / 1.011 / 0.962 | **drop** — stress48 1.030 [0.999, 1.066] |
| L3 | normal guidance w 0.025, κ gate off (best of w ∈ {0.025, 0.05, 0.1} × κ on/off) | L1 | 0.986 [0.973, 0.996] (s2024), 0.983 [0.970, 0.994] (s2025) | 0.981 / 0.983 / 0.988 (s2025) | **keep** — small but the same on two seeds, all groups better; with L6 on top 0.987 [0.973, 1.000]. w 0.1 hurts real_indoor. Needs DSINE normals (~1 s/image) |
| L4 | in-loop metric residual field | — | superseded by L7 | | the post-hoc field is simpler and robust |
| L5 | fewer steps (same lr), with L6 | L1+L6 | 25 steps 1.651 [1.371, 2.041], 10 steps 3.327 [2.255, 5.074] | driving worst (2.8 / 8.7) | **drop** — 12.4 / 4.9 s vs 21 s per case, but the latent does not converge; an affine refit does not rescue it. 25 steps with 2× lr (+ L7 + L6): 1.208 [1.123, 1.331] vs the 50-step version, 1.007 [0.970, 1.048] vs MDC at ~12 s/case — a possible "fast" setting (MDC accuracy, half the time) |
| L6 | RGB-guided filter on log-depth (grey guide, r 2 on the 384 grid, eps 1e-3) | L1 | 0.896 vs MDC (s2024); stress48 0.881 | 0.942 / 0.909 / 0.787 vs MDC | **keep** — r 4 gives the same AbsRel but softer edges (edge sharpness 0.28 vs 0.34 on new scenes); r 2 is also sharper than MDC on dev50 (0.393 vs 0.366) |
| L7 | post-hoc robust log residual field (6×8, Tukey IRLS, λ 0.3), before L6 | L1+L6 | 0.921 [0.861, 0.964] (s2025) | 0.925 / 0.969 / 0.827 | **keep** — vs MDC: 0.834 (s2024), 0.834 (s2025), stress48 0.854. A plain metric field helped dev50 (0.841) but hurt stress48 (0.891 vs 0.881, real_indoor 0.958); the robust log fit fixes that |

## Current best (candidate) — `configs/v2/candidate.yaml`

L1 + L3 + L7 + L6, vs Marigold-DC on dev50 at 384 px. End-to-end run `cand_dev50_384` reproduces the offline result exactly (ratio 1.000); stress48 `cand_stress_384` 0.843 [0.783, 0.896] (ri 0.882 / syn 0.848 / drv 0.715), 21.5 s/case vs MDC 21.2 s. Offline (from saved L3 predictions;
the end-to-end candidate run is the check): **0.829 [0.746, 0.901]** (s2024), **0.828 [0.757, 0.886]**
(s2025); groups real_indoor 0.874–0.902, synthetic 0.863–0.873, driving 0.631–0.683.

New scenes (Colab, 24 dev scenes from `main_datasets_new`, never used for tuning): L1 0.988
[0.971, 1.004]; L1 + L6 (r 2) 0.894 [0.863, 0.922] (patterns) and 0.906 (stress). End-to-end on these scenes: `cand` **0.821 [0.781, 0.863]** (patterns), **0.847 [0.813, 0.881]** (stress), every group and metric better; `fast` 1.060 / 1.057 (dropped). Summary: `V2_RESULTS.md`.

## Tried and not kept (offline probes on saved predictions)

| idea | result vs the kept version | why not |
|---|---|---|
| prior fit (plain Marigold + global LS affine, = the inert v1 "bug" output) | 3.73× MDC; + metric field 1.97×; 4 steps + field 1.83× at 0.42 s/case | edges well placed but jump sizes wrong; copies image texture as depth relief (wall pictures in Living_room_8) |
| detail blend (high-pass of prior added to ours, or prior-guided filter) | 0.913–1.073 vs MDC, sharpness 0.19–0.34 | worse than ours (0.887, 0.389) on both |
| edge-protected filter (blend by own / prior / filtered edge strength) | loses most of the L6 gain | no sharpness regained |
| trilateral / prior-guided filter | not better than L6 | — |
| joint bilateral upsampling from the 384 grid instead of bilinear | +0.5–1 % with L6, sharpness unchanged | softness comes from the 384 solve, not the upsampling |
| median + L6 | +0.5–1 % | within noise |
