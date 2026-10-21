import numpy as np

from hifidc.viz import colorize, comparison_figure, depth_range, error_map


def test_colorize_marks_invalid_black_and_spans_the_map():
    values = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    valid = values > 0
    image = colorize(values, valid, 1.0, 3.0, 'turbo')
    assert image.shape == (2, 2, 3) and image.dtype == np.uint8
    assert (image[1, 1] == 0).all()                      # invalid pixel is black
    assert not (image[0, 0] == image[1, 0]).all()        # both ends of the range differ


def test_depth_range_uses_valid_gt_percentiles():
    gt = np.linspace(1, 100, 10000, dtype=np.float32).reshape(100, 100)
    gt[0, :10] = 0
    lo, hi = depth_range(gt, gt > 0)
    assert 1.0 < lo < 5.0 and 95.0 < hi < 100.0


def test_error_map_is_relative_error_clipped():
    gt = np.full((4, 4), 10.0, dtype=np.float32)
    pred = gt * 1.1
    err = error_map(pred, gt, gt > 0, max_rel=0.2)
    assert err.shape == (4, 4, 3)
    gt[0, 0] = 0
    err2 = error_map(pred, gt, gt > 0, max_rel=0.2)
    assert (err2[0, 0] == 0).all()


def test_comparison_figure_has_one_column_per_run_and_three_rows():
    h, w = 40, 60
    rgb = np.zeros((h, w, 3), np.uint8)
    gt = np.full((h, w), 5.0, np.float32)
    sparse = np.zeros((h, w), np.float32)
    sparse[::7, ::7] = 5.0
    preds = {'Marigold-DC': gt * 1.05, 'HiFi-DC v2': gt * 0.98, 'third': gt}
    fig = comparison_figure(rgb, sparse, gt, gt > 0, preds, title='case', panel_width=60)
    assert fig.ndim == 3 and fig.shape[2] == 3
    assert fig.shape[1] >= 3 * 60          # at least three columns
    assert fig.shape[0] >= 3 * 40          # three rows: inputs, predictions, errors
