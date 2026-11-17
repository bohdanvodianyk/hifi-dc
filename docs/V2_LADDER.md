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
| L6 | RGB-guided filter on log-depth (grey guide, r 2 on the 384 grid, eps 1e-3) | L1 | 0.896 vs MDC (s2024); stress48 0.881 | 0.942 / 0.909 / 0.787 vs MDC | **keep** — r 4 gives the same AbsRel but softer edges (edge sharpness 0.28 vs 0.34 on new scenes); r 2 is also sharper than MDC on dev50 (0.393 vs 0.366; **correction:** that is the median over cases — scene-clustered, the full candidate is softer than MDC, 0.80, see "Edge quality" below) |
| L7 | post-hoc robust log residual field (6×8, Tukey IRLS, λ 0.3), before L6 | L1+L6 | 0.921 [0.861, 0.964] (s2025) | 0.925 / 0.969 / 0.827 | **keep** — vs MDC: 0.834 (s2024), 0.834 (s2025), stress48 0.854. A plain metric field helped dev50 (0.841) but hurt stress48 (0.891 vs 0.881, real_indoor 0.958); the robust log fit fixes that |

## Current best (candidate) — `configs/v2/candidate.yaml`

L1 + L3 + L7 + L6, vs Marigold-DC on dev50 at 384 px. End-to-end run `cand_dev50_384` reproduces the offline result exactly (ratio 1.000); stress48 `cand_stress_384` 0.843 [0.783, 0.896] (ri 0.882 / syn 0.848 / drv 0.715), 21.5 s/case vs MDC 21.2 s. Offline (from saved L3 predictions;
the end-to-end candidate run is the check): **0.829 [0.746, 0.901]** (s2024), **0.828 [0.757, 0.886]**
(s2025); groups real_indoor 0.874–0.902, synthetic 0.863–0.873, driving 0.631–0.683.

New scenes (Colab, 24 dev scenes from `main_datasets_new`, never used for tuning): L1 0.988
[0.971, 1.004]; L1 + L6 (r 2) 0.894 [0.863, 0.922] (patterns) and 0.906 (stress). End-to-end on these scenes: `cand` **0.821 [0.781, 0.863]** (patterns), **0.847 [0.813, 0.881]** (stress), every group and metric better; `fast` 1.060 / 1.057 (dropped). Summary: `V2_RESULTS.md`.

**Final held-out test (80 scenes, run once):** `cand` vs MDC **0.803 [0.775, 0.830]** (patterns),
**0.848 [0.828, 0.868]** (stress); every group and metric better.

## Edge quality, scene-clustered (`scripts/edge_runs.py`, dev50 384 px, 7 dense-GT scenes)

edge_sharpness = median over GT edge pairs of |jump in pred| / |jump in GT| (1 = as sharp as GT); ratios vs MDC
from the paired scene-clustered bootstrap (edge AbsRel < 1 better, sharpness > 1 sharper).

| run | edge AbsRel | sharpness (scene mean) | edge AbsRel vs MDC | sharpness vs MDC |
|---|---|---|---|---|
| MDC | 0.218 | 0.467 | 1 | 1 |
| L3 (solver only) | 0.218 | 0.452 | 0.997 | 0.958 [0.902, 1.025] |
| cand (L3 + L7 + L6) | 0.207 | 0.380 | **0.942** [0.890, 1.000] | **0.802 [0.655, 0.963]** (real_indoor 0.978, synthetic 0.692) |
| prior fit (`ls`, the v1 "bug" look) | 0.341 | 0.140 | 1.620 | 0.220 |

The post-processing (L7 + L6) makes the edge region more accurate but shrinks depth jumps, mostly on the
synthetic scenes; the "wrong" prior-fit look is the least sharp by this measure (its jumps are far too small).

## Tried and not kept (offline probes on saved predictions)

| idea | result vs the kept version | why not |
|---|---|---|
| prior fit (plain Marigold + global LS affine, = the inert v1 "bug" output) | 3.73× MDC; + metric field 1.97×; 4 steps + field 1.83× at 0.42 s/case | edges well placed but jump sizes wrong; copies image texture as depth relief (wall pictures in Living_room_8) |
| detail blend (high-pass of prior added to ours, or prior-guided filter) | 0.913–1.073 vs MDC, sharpness 0.19–0.34 | worse than ours (0.887, 0.389) on both |
| edge-protected filter (blend by own / prior / filtered edge strength) | loses most of the L6 gain | no sharpness regained |
| trilateral / prior-guided filter | not better than L6 | — |
| joint bilateral upsampling from the 384 grid instead of bilinear | +0.5–1 % with L6, sharpness unchanged | softness comes from the 384 solve, not the upsampling |
| median + L6 | +0.5–1 % | within noise |

## Phase 2 — speed (dev50, 384 px, local RTX 2000 Ada 8 GB; candidate = 22.7 s/case)

| change | s/case | vs MDC | vs cand | decision |
|---|---|---|---|---|
| fused SDPA alone / no per-step cache flush alone | 22.5 / 21.1 | — | — | small each |
| `--fast` = fused SDPA + no per-step cache flush | 16.9 (−26 %) | 0.826 [0.746, 0.896] | 0.997 [0.983, 1.010] | **keep** as the runtime default for speed runs; method unchanged, not bit-repeatable |
| torch.compile (UNet or VAE decoder) | — | — | — | fails in dynamo with this torch/diffusers pair |
| warm start 0.5 (one-step prior re-noised to mid-schedule, 25 solver steps; prior LS affine init) | 11.4 | 0.916 [0.756, 1.179] | 1.105 | **not kept**: office_4 0.70×, Living_room_8 2.75× |
| warm start 0.7 (15 steps; prior affine init) | 6.8 | 1.017 | 1.227 | not kept |
| warm start 0.3 (35 steps; prior affine init) | 15.2 | 0.872 [0.710, 1.135] | 1.052 | not kept |
| prior affine init only, 50 steps, latent from noise (`warm_start=0.005`) | 21.9 | 0.875 [0.710, 1.131] | 1.056 | **not kept** — and the cause of the warm-start losses: per scene it matches warm 0.3 (Living_room_8 2.64×, office_4 0.70×), so the LS affine init from the one-step prior, not the warm latent, moves the result. warm 0.3 vs this: same accuracy at 70 % of the time |
| 25 steps, 2× lr, prior affine init (latent from noise) | 11.2 | 1.024 | 1.236 | not kept (≈ the old fast setting, 1.017) |
| warm start 0.3 / 0.5 / 0.7, cold affine init (`warm_affine: none`, `--fast`) | 12.0 / 8.5 / 5.3 | 0.904 / 1.060 / 1.147 | 1.094 / 1.283 / 1.389 (vs `--fast`) | **not kept**: Living_room_8 is fixed at 0.3 (1.06) but driving 1.45 and Scene20 1.68 / office_4 1.21 lose what the prior init gave; at 0.5+ Living_room_8 fails again (2.3, texture). The warm latent needs a matching affine, and the LS one is wrong for some scenes |
| why the LS affine init fails (hypothesis) | — | — | — | the LS fit of the one-step prior gives a low scale and a high shift (Living_room_8: 0.74 / 0.58 vs 1.00 / 0.31 cold; office_4 shift 0.56 vs 0.05), and the solver then barely moves it (0.738 → 0.738). Guess: regression dilution by noise in the prior — **rejected** by the next row |
| warm start 0.3, inverse-regression affine init (undiluted when the noise is in the prior; `--fast`) | 11.9 | 0.894 [0.738, 1.134] | 1.082 [0.916, 1.352] | **not kept**: Living_room_8 still 2.60× (ls 2.81, cold 1.06), market_2 1.26; driving and Scene20 kept. Whatever the prior's affine gets wrong on Living_room_8 is not slope dilution |
| warm start 0.3 `rma`; 0.5 `inverse` (`--fast`) | 11.9 / 8.7 | 0.876 / 0.926 | 1.060 / 1.121 | not kept: between `ls` and `inverse`, as expected; synthetic 1.19 / 1.29 |
| **Phase 2 conclusion** | | | | `--fast` is the speed result (16.9 s vs MDC 21.2 s, 0.826 vs MDC). Every warm start costs 5–12 % at 0.3 and more beyond, whatever the affine init; the latent needs the full noise-to-data schedule |
| two starts per case, keep the one with the lower anchor residual (input points only; offline from saved runs) | ~24 (2 runs) | 0.791 [0.705, 0.864] (warm-ls 0.3 + warm-cold 0.3) | 0.958 [0.881, 1.024] | **not kept**: control — the same selection between two seeds of cand gives 0.976, so half the gain is best-of-two noise; the warm start adds ~2 % (n.s.) at twice the cost. Deciding early does not work: picking by the solver loss at any step up to the last gives 1.01–1.06 (agreement with the final pick 22–36/50); only the residual after L7 + L6 selects well |

## Phase 3 — cheaper steps (dev50, 384 px, all with `--fast`; reference = cand + `--fast`, 16.9 s/case)

| change | s/case | vs MDC | vs cand `--fast` | decision |
|---|---|---|---|---|
| no UNet backward (`unet_grad_steps: 0`) | 11.2 | 1.730 [1.279, 2.603] | 2.094 | **drop**: Marigold is v-prediction, x0 = sqrt(abar) x_t - sqrt(1-abar) v, so early on the latent reaches x0 almost only through the UNet (sqrt(abar) = 0.07 at step 1) |
| UNet backward for the first 50 % / 70 % of steps | 13.9 / 15.0 | 1.169 / 1.026 | 1.415 / 1.242 | **drop**: the UNet Jacobian matters late as well |
| TAESD for the in-loop decode, full VAE for the final depth (`loop_decoder: taesd`) | **9.7** | **0.863 [0.789, 0.926]** | 1.044 [1.002, 1.081] | promising: 2.2x faster than MDC, still clearly better. TAESD vs full VAE on saved predictions: corr 0.998, mean abs diff 0.02-0.03 on [-1, 1], edge strength 0.94-1.01 |
| TAESD for the first 80 % / 70 % of steps, full VAE after (`loop_decoder_steps`) | 10.8 / 12.0 | 0.869 / 0.851 | 1.051 / 1.030 | not kept: the loss builds up while TAESD guides the late steps, not from a final affine mismatch |
| **TAESD for the first 50 % of steps** (`configs/v2/candidate_taesd.yaml`) | **13.4** | **0.825 [0.754, 0.886]** | **0.999 [0.969, 1.028]** | **keep** as the speed setting (pending the Colab new-scene check): groups 0.999 / 0.970 / 1.060 (driving just over 5 %, one KITTI scene); stress48 **0.990 [0.947, 1.031]** vs cand, 0.834 vs MDC, groups 1.005 / 0.969 / 1.031, 13.8 s. 1.6x faster than MDC (21.2 s) at the candidate's accuracy. **New scenes (Colab, 24 dev scenes): vs MDC 0.841 [0.788, 0.895] (patterns), 0.870 [0.823, 0.918] (stress), every group and metric better; vs cand (ratio of the two MDC ratios, same cases) 1.024 / 1.027, real_indoor 1.082 / 1.063, synthetic 1.013 / 1.034, driving 0.990 / 0.947.** So not equal accuracy: real_indoor loses 6-8 % (the local dev50 has too few real-indoor scenes to show it). Kept as a speed/accuracy trade-off option, not as the default. Colab L4 median s/case: MDC 17.7, cand (no `--fast`) 17.9, cand_taesd 11.3 (patterns) / 11.4 (stress) = 1.57x faster than MDC |

Local runs from here on use `.superpowers/half.sh` (two foreground 5-scene halves, ~5 min and 4.4 GB host RSS each): background shells were killed four times by the host-memory reaper.

### Frozen weights (free speed-up, now part of `--fast`)

A per-part profile of one case (`.superpowers/profile_case.py`, 384 px, latent 40x48) put 67 % of the time in
`loss.backward()` and showed every UNet and VAE-decoder weight receiving a gradient each step (never used: the
optimiser holds only the latent, scale and shift; Marigold-DC does the same). Per step: UNet fwd+bwd 159 -> 95 ms,
VAE decode fwd+bwd 186 -> 136 ms once the weights are frozen. Same maths; input-only conv backward kernels round
differently (~1 ulp on the CPU golden test), so `HiFiDCv2Pipeline.freeze_weights` stays off by default (bit-exact
Marigold-DC) and `run_benchmark --fast` switches it on (with fused SDPA and no per-step cache flush).

| setting (dev50, 384 px, local RTX 2000 Ada) | s/case | peak VRAM | vs MDC | vs cand + old `--fast` | decision |
|---|---|---|---|---|---|
| MDC (official settings) | 21.2 | 6.23 GB | 1 | | reference |
| cand + old `--fast` (fused SDPA, no cache flush) | 16.9 | 5.88 GB | 0.826 | 1 | |
| **cand + `--fast` incl. frozen weights** | **11.4** | **4.07 GB** | **0.824 [0.744, 0.894]** | **0.997 [0.989, 1.006]**, groups 0.998 / 0.998 / 0.992 | **keep: the v2 default way to run** — 1.86x faster than MDC at unchanged accuracy |
| new Colab scenes, cand + old `--fast` (two repeat runs, both at e46d16b; "cand_fast" / "cand_frozen" labels) | | | patterns 0.823 / 0.826, stress 0.855 / 0.850 (cand 0.821 / 0.847) | ratio of MDC ratios 1.002 / 1.006 (patterns), 1.009 / 1.004 (stress) | old `--fast` is accuracy-neutral on new scenes too; repeat noise ~0.5 % |
| cand_taesd + `--fast` incl. frozen weights | **8.7** | | 0.827 [0.753, 0.890] | 1.001 [0.968, 1.031], groups 1.004 / 0.973 / 1.050 | trade-off option as before (new scenes: ~2.5 % worse than cand, real_indoor 6-8 %): 2.4x faster than MDC |
