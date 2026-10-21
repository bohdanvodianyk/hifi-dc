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
