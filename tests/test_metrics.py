"""Scale-aware depth metrics; native-unit metrics are never pooled across scenes."""
import numpy as np
import pytest

from hifidc.metrics import aggregate, anchor_residual, depth_metrics


def grid(seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1., 10., (16, 16)).astype(np.float32)
    return gt, np.ones_like(gt, dtype=bool)


def test_perfect_prediction():
    gt, valid = grid()
    m = depth_metrics(gt.copy(), gt, valid)
    assert m['abs_rel'] == 0 and m['rmse'] == 0 and m['mae'] == 0 and m['sq_rel'] == 0
    assert m['delta1'] == 1 and abs(m['si_log_rmse']) < 1e-6
    assert m['frac_pred_nonpositive'] == 0 and m['n_valid'] == 256


def test_double_prediction_is_scale_invariant_in_si_log():
    gt, valid = grid()
    m = depth_metrics(2 * gt, gt, valid)
    assert abs(m['abs_rel'] - 1) < 1e-6
    assert abs(m['si_log_rmse']) < 1e-6
    assert m['delta1'] == 0
    assert abs(m['mae'] - float(gt.astype(np.float64).mean())) < 1e-4


def test_invalid_pixels_excluded():
    gt, valid = grid()
    pred = gt.copy()
    pred[0, 0] = 1e6
    valid[0, 0] = False
    assert depth_metrics(pred, gt, valid)['abs_rel'] == 0


def test_nonpositive_predictions_stay_finite():
    gt, valid = grid()
    pred = gt.copy()
    pred.flat[::10] = -1.
    m = depth_metrics(pred, gt, valid)
    assert all(np.isfinite(v) for v in m.values())
    assert abs(m['frac_pred_nonpositive'] - len(pred.flat[::10]) / pred.size) < 1e-9
    assert m['delta1'] < 1


def test_no_valid_pixels_raises():
    gt, valid = grid()
    with pytest.raises(ValueError):
        depth_metrics(gt, gt, np.zeros_like(valid))


def test_anchor_residual():
    pred = np.full((4, 4), 2., np.float32)
    sparse = np.zeros((4, 4), np.float32)
    sparse[0, 0], sparse[1, 1] = 3., 1.5
    assert abs(anchor_residual(pred, sparse) - 0.75) < 1e-9


def test_aggregate_only_scale_free_metrics():
    rows = [dict(abs_rel=.1, delta1=.9, si_log_rmse=.2, rmse=5., mae=3., sq_rel=1.),
            dict(abs_rel=.3, delta1=.7, si_log_rmse=.4, rmse=500., mae=300., sq_rel=9.)]
    summary = aggregate(rows)
    assert abs(summary['abs_rel'] - .2) < 1e-12 and abs(summary['delta1'] - .8) < 1e-12
    assert abs(summary['si_log_rmse'] - .3) < 1e-12 and summary['n_scenes'] == 2
    assert not {'rmse', 'mae', 'sq_rel'} & set(summary)
