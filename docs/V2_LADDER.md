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
| L2b | Huber, k = 3.0 | L1 | pending | | |
| L3 | normal guidance sweep (weight × κ gate), on L1 | L1 | pending | | |
