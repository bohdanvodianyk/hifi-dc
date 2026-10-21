"""Huber IRLS weights with a MAD scale for the sparse anchor loss."""
import torch

from hifidc.v2.robust import robust_l1l2, robust_weights


def test_inliers_full_weight():
    residuals = torch.randn(1000, generator=torch.Generator().manual_seed(0))
    weights = robust_weights(residuals, 1.345)
    assert float((weights == 1).float().mean()) >= 0.80
    assert torch.all((weights > 0) & (weights <= 1))


def test_outlier_downweighted():
    residuals = torch.randn(500, generator=torch.Generator().manual_seed(1)) * 0.01
    residuals[0] = 0.5  # 50 sigma
    weights = robust_weights(residuals, 1.345)
    assert float(weights[0]) < 0.05 and float(weights[1:].mean()) > 0.9


def test_mad_zero_all_ones():
    for residuals in (torch.full((20,), 0.3), torch.zeros(20)):
        weights = robust_weights(residuals, 1.345)
        assert torch.equal(weights, torch.ones(20)) and torch.isfinite(weights).all()


def test_weights_are_detached():
    residuals = torch.randn(50, requires_grad=True)
    assert not robust_weights(residuals, 1.345).requires_grad


def test_weighted_loss_gradient_bounded():
    target = torch.zeros(500)
    predicted = (torch.randn(500, generator=torch.Generator().manual_seed(2)) * 0.01).requires_grad_(True)
    with torch.no_grad():
        predicted[0] = 0.5
    robust_l1l2(predicted, target, 1.345).backward()
    inlier = predicted.grad[1:].abs().max()
    assert float(predicted.grad[0].abs()) <= 3 * float(inlier)


def test_unit_weights_match_l1_plus_l2():
    predicted, target = torch.tensor([1., 2., 3., 4.]), torch.tensor([1.1, 1.9, 3.2, 4.0])
    expected = torch.nn.functional.l1_loss(predicted, target) + torch.nn.functional.mse_loss(predicted, target)
    assert torch.allclose(robust_l1l2(predicted, target, k=1e9), expected)
