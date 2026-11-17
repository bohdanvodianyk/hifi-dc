import numpy as np
import torch
import torch.nn.functional as F

from hifidc.metrics import edge_metrics


def _step_scene(h=64, w=96):
    gt = np.full((h, w), 2.0, np.float32)
    gt[:, w // 2:] = 5.0
    return gt, np.ones_like(gt, bool)


def _blur(depth, k=7):
    t = torch.from_numpy(depth)[None, None]
    t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode='replicate'), k, stride=1)
    return t[0, 0].numpy()


def test_perfect_prediction_is_sharp_with_zero_edge_error():
    gt, valid = _step_scene()
    m = edge_metrics(gt.copy(), gt, valid)
    assert m['edge_abs_rel'] == 0.0 and abs(m['edge_sharpness'] - 1.0) < 1e-6 and m['n_edge'] > 0


def test_blurred_edge_loses_sharpness_and_gains_edge_error():
    gt, valid = _step_scene()
    m = edge_metrics(_blur(gt), gt, valid)
    assert m['edge_sharpness'] < 0.5 and m['edge_abs_rel'] > 0.05


def test_flat_noise_away_from_edges_does_not_count():
    gt, valid = _step_scene()
    pred = gt.copy()
    pred[:, :20] *= 1.2                 # error far from the edge
    assert edge_metrics(pred, gt, valid)['edge_abs_rel'] == 0.0


def test_semi_dense_gt_gives_nan():
    gt, valid = _step_scene()
    valid[::2] = False
    valid[:, ::2] = False
    m = edge_metrics(gt, gt, valid)
    assert np.isnan(m['edge_abs_rel']) and np.isnan(m['edge_sharpness'])


def test_metric_is_unit_free():
    gt, valid = _step_scene()
    pred = _blur(gt, 3)
    a, b = edge_metrics(pred, gt, valid), edge_metrics(pred * 1000, gt * 1000, valid)
    assert abs(a['edge_abs_rel'] - b['edge_abs_rel']) < 1e-6 and abs(a['edge_sharpness'] - b['edge_sharpness']) < 1e-6


def test_upsampled_but_correct_edge_still_counts_as_sharp():
    h, w = 120, 640
    gt = np.full((h, w), 2.0, np.float32)
    gt[:, 333:] = 5.0                                   # edge not on the coarse grid
    coarse = torch.from_numpy(gt)[None, None]
    coarse = F.interpolate(coarse, size=(72, 384), mode="area")
    pred = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    assert edge_metrics(pred, gt, np.ones_like(gt, bool))["edge_sharpness"] > 0.8
    assert edge_metrics(_blur(gt, 15), gt, np.ones_like(gt, bool))["edge_sharpness"] < 0.6


def _upsampled_from(gt, width):
    h, w = gt.shape
    coarse = F.interpolate(torch.from_numpy(gt)[None, None], size=(round(h * width / w), width), mode="area")
    return F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


def test_finer_span_grid_tells_a_384_solve_from_a_768_solve():
    h, w = 240, 1280
    gt = np.full((h, w), 2.0, np.float32)
    gt[:, 667:] = 5.0
    valid = np.ones_like(gt, bool)
    from_384, from_768 = _upsampled_from(gt, 384), _upsampled_from(gt, 768)
    # the default tolerance (384 grid) counts both as sharp; the 768 tolerance only the 768 solve
    assert edge_metrics(from_384, gt, valid)["edge_sharpness"] > 0.8
    assert edge_metrics(from_768, gt, valid, span_grid=768)["edge_sharpness"] > 0.8
    # an ideal (area-down, bilinear-up) 384 edge: 0.94 vs 1.0; real 384 solves are blurrier than that
    assert (edge_metrics(from_384, gt, valid, span_grid=768)["edge_sharpness"]
            < edge_metrics(from_768, gt, valid, span_grid=768)["edge_sharpness"] - 0.03)
    assert edge_metrics(gt, gt, valid, span_grid=768) == edge_metrics(gt, gt, valid)  # exact edge: same either way
