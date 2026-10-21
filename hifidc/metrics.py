"""Per-scene depth metrics (native units) and a cross-scene aggregate of scale-free metrics only."""
import numpy as np

SCALE_FREE = ('abs_rel', 'delta1', 'si_log_rmse')


def depth_metrics(pred, gt, valid):
    """Metrics over valid GT pixels with a finite prediction.

    Log-based metrics use pixels with pred > 0; non-positive predictions count as delta1 failures
    and are reported as ``frac_pred_nonpositive``.
    """
    mask = valid & np.isfinite(pred)
    if not mask.any():
        raise ValueError('No valid pixels to score')
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    err = p - g
    positive = p > 0
    ratio = np.full_like(p, np.inf)
    ratio[positive] = np.maximum(p[positive] / g[positive], g[positive] / p[positive])
    if positive.any():
        d = np.log(p[positive]) - np.log(g[positive])
        si_log = float(np.sqrt(max(np.mean(d ** 2) - np.mean(d) ** 2, 0.)))
    else:
        si_log = float('nan')
    return {
        'abs_rel': float(np.mean(np.abs(err) / g)),
        'sq_rel': float(np.mean(err ** 2 / g)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'delta1': float(np.mean(ratio < 1.25)),
        'si_log_rmse': si_log,
        'frac_pred_nonpositive': float(1. - positive.mean()),
        'n_valid': int(mask.sum()),
    }


def anchor_residual(pred, sparse):
    """Mean |pred - sparse| at the sparse input points (native units)."""
    mask = (sparse > 0) & np.isfinite(pred)
    return float(np.mean(np.abs(pred[mask].astype(np.float64) - sparse[mask])))


def aggregate(rows):
    """Mean of scale-free metrics across scenes; native-unit metrics are deliberately excluded."""
    summary = {key: float(np.nanmean([row[key] for row in rows])) for key in SCALE_FREE}
    summary['n_scenes'] = len(rows)
    return summary


def edge_metrics(pred, gt, valid, jump=0.1, band=2, min_fill=0.9):
    """Depth-edge quality on dense GT (NaN when less than ``min_fill`` of the GT is valid).

    Edges are neighbouring valid GT pixels whose log-depth differs by more than ``jump`` (~10 %).
    edge_abs_rel: AbsRel inside a band of ``band`` px (on a 384-px grid, scaled) around the edges.
    edge_sharpness: median over edge pairs of |d log pred| / |d log gt| across the same pair, clipped
    to [0, 2]; 1 = as sharp as GT, a blurred edge is < 1. Both are unit-free.
    """
    nan = {'edge_abs_rel': float('nan'), 'edge_sharpness': float('nan'), 'n_edge': 0}
    if valid.mean() < min_fill:
        return nan
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = np.where(valid, np.log(np.where(valid, gt, 1.0)), np.nan).astype(np.float64)
        ok_pred = np.isfinite(pred) & (pred > 0)
        lp = np.where(ok_pred, np.log(np.where(ok_pred, pred, 1.0)), np.nan).astype(np.float64)
    edges = np.zeros(gt.shape, bool)
    ratios = []
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                 ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        dg = lg[a] - lg[b]
        dp = lp[a] - lp[b]
        pair = np.abs(dg) > jump
        pair &= np.isfinite(dg)
        edges[a] |= pair
        edges[b] |= pair
        use = pair & np.isfinite(dp)
        ratios.append(np.clip(np.abs(dp[use]) / np.abs(dg[use]), 0, 2))
    ratios = np.concatenate(ratios)
    if not edges.any() or ratios.size == 0:
        return nan
    r = max(1, round(band * max(gt.shape) / 384))
    grown = edges.copy()
    for _ in range(r):
        step = grown.copy()
        step[1:] |= grown[:-1]
        step[:-1] |= grown[1:]
        step[:, 1:] |= grown[:, :-1]
        step[:, :-1] |= grown[:, 1:]
        grown = step
    mask = grown & valid & ok_pred
    err = np.abs(pred[mask].astype(np.float64) - gt[mask]) / gt[mask]
    return {'edge_abs_rel': float(err.mean()) if err.size else float('nan'),
            'edge_sharpness': float(np.median(ratios)), 'n_edge': int(edges.sum())}
