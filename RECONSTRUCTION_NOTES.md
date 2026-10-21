# Method correspondence and explicit reconstruction decisions

Reference: the provided `cas-dc-template.tex`. User instruction establishes it as the primary reference for the intended method. The current experimental files are comparison material, not the target specification. Historical code identity is not inferred from the new implementation.

## Implemented mechanisms

| Paper location | Experimental snapshot | Clean implementation |
| --- | --- | --- |
| 229, 423: frozen models; latent Adam | Prediction for the loss formed under no-grad | Differentiable UNet/DDIM-predicted-clean-sample/VAE path; frozen model parameters; assertion that latent gradients exist and are finite |
| 237–309: WLS, ridge, EMA | Unweighted `lstsq`; no ridge/EMA | Centred weighted moments, ridge on slope, EMA on slope/intercept, sample/weight/variance guards |
| 318–339: confidence-weighted losses | Uniform sparse penalties | Density-derived confidence used in both fitting and the exact published quadratic and L1 expressions |
| 349–375: normals | Single-scale forward differences; L1-normalized dot product | Sobel, exponential attenuation, Euclidean unit normals, scales 1/2/4 weighted 1/.5/.25 |
| 386–423: normal schedule | Late linear ramp only | Late linear ramp multiplied by the stated stability factor; explicit constant and ramp-only alternatives |
| 426–438: progressive resolution | Fixed grid | First floor(T/3) steps at half resolution, remaining steps at full resolution |
| 440–463: final processing | No bilateral/uncertainty stages | Final WLS/EMA alignment, optional bilateral filter, distance/stability uncertainty |
| Component studies | Later distortions and hyperparameter interventions | No such additions; individual Config switches are prospective controls, not recovered historic table definitions |

The new code recomputes denoiser predictions after Adam before advancing DDIM. It does not apply a denoising update using stale pre-optimisation noise. This costs an additional UNet pass; publication timings must not be attributed to this reconstruction without measurement.

## Paper ambiguities and how this implementation resolves them

These are **declared choices**, not recovered experiment facts. All numeric choices are exposed in Config or documented here. Do not tune them on the benchmark test set just to match a published number.

- Confidence: paper gives density, normalisation and sigmoid but no window, padding or sigmoid parameters. Use a 5-pixel window, border-aware averaging, maximum-density normalisation, sigmoid gain 10 and midpoint .5. Recompute at each resolution.
- Ridge and variance guard: paper omits magnitudes. Use `1e-6` and `1e-8`; decoded prior values follow Marigold's [0,1] output convention. Guard the ridge-augmented variance as written in the algorithm. Slope is unconstrained as in the paper, rather than importing the experimental exp-scale parameterisation. Negative slopes/depths are not silently clipped; inspect outputs before downstream fusion.
- WLS objective: the displayed unregularised argmin and ridge formula are not literally the same objective. The implementation follows the stated ridge formula, equivalent to adding a slope penalty with an unpenalised intercept. The thesis already records this mathematical qualification.
- Initialise affine parameters to (1,0), use EMA beta .5 from the paper algorithm, and apply EMA even on warm/final refits. The paper does not fully distinguish these update cases.
- Learning rate .02 and clipping 1 come from the surviving base code. They are reconstruction settings, not independently recovered published run settings. No ablation changes them automatically.
- Sobel uses division by eight and replicated image boundaries. The paper does not specify derivative scaling/padding. Normals are area-downsampled then L2-renormalised at each scale.
- The paper's prose says smaller tau under-attenuates and larger tau over-attenuates. Its equation implies the reverse: for positive gradient magnitude, smaller tau gives stronger attenuation. Implement the equation with tau .20. Do not reproduce that prose reversal.
- Step indexing is one-based, so the ramp reaches its nominal endpoint on step T. Stability weights are computed from detached scalar loss history; the paper does not specify whether to differentiate through the weight itself. First-step relative change is zero. The stability history is reset at the resolution change to avoid comparing objectives on different sample grids.
- Sparse resizing maps valid sample centres and averages collisions, without interpolating missing zeros. RGB and normal maps use bilinear resizing with preserved aspect ratio, then padding to multiples of 16. The paper says 768x768 but does not fully specify aspect-ratio/padding behavior; this implementation uses the old interface's longest-side resolution convention. Output is cropped/resized back to the original image grid.
- Half-resolution step count uses floor(T/3). At the resolution transition, RGB is re-encoded, the latent is bilinearly resized and Adam is reset. These transition details are absent from the paper.
- Input/output depth units must be declared (m or mm). Internal optimisation uses metres. The paper does not unambiguously define the units/pre-normalisation attached to all hyperparameters. Especially check tau and bilateral range sigma when comparing with original runs.
- Bilateral uses diameter 5 (paper permits 5 or 7), both sigmas 75, replicated border and raw metric depth, with no hidden intensity conversion. This literal range-sigma choice may produce weak range discrimination on metre-scale scenes; it is flagged rather than silently replaced.
- Uncertainty uses Euclidean distance to observed pixels divided by the image maximum and the mean final-ten-step relative loss change. It is a heuristic, not calibrated probability; it can exceed one. It uses the prepared grid, then is resized to the output grid.
- DSINE normal inference retains the original external project API, ImageNet normalisation, padding and 60-degree nominal field of view. These preprocessing details come from the surviving helper rather than a complete paper specification. Actual camera-intrinsics handling remains a possible reconstruction difference. The VAE-normalised [-1,1] RGB tensor is never sent to the PIL conversion for DSINE.

## Verification limits and integration

No pretrained model, benchmark input/output pair or original evaluator is bundled. The author chose to run the clean code through GeoFuse3D and the old script. Local tests establish component arithmetic, latent updates, frozen weights, schedule/shape behavior, normal influence, reproducibility for the CPU fixture and local real-Diffusers save/load. They do not establish published accuracy or runtime.

The old ablation evaluator's log-error expression is a mean absolute log error despite its `rmse_log` label. The clean inference code contains no replacement evaluator; verify the evaluator used by the author before interpreting that column. Extended metrics, SSIM and DBE were not invented or substituted.

The author's existing correction says historical DBE was unclipped and bidirectional; the paper describes clipped/directional DBE. This conflict is flagged. Making the paper primary for method reconstruction does not justify relabelling unchanged historical table values as having a different evaluation protocol.

The historical experimental files and tables are preserved. A new run manifest describes the new code. It does not claim this is the historical implementation or that its results reproduce the publication until measured.
