from __future__ import annotations

import torch

from octdenoiser.engine.losses import charbonnier_loss, compute_total_loss, gradient_l1


def test_losses_are_finite_and_differentiable() -> None:
    prediction = torch.randn(2, 1, 8, 9, requires_grad=True)
    target = torch.randn(2, 1, 8, 9)
    loss = compute_total_loss(prediction, target, w_charb=0.8, w_grad=0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_matching_images_have_zero_gradient_loss() -> None:
    image = torch.randn(1, 1, 5, 6)
    assert gradient_l1(image, image).item() == 0.0
    assert charbonnier_loss(image, image).item() > 0.0
