from __future__ import annotations

import torch


def compute_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    w_charb: float,
    w_grad: float,
) -> torch.Tensor:
    """Combine pixel-domain Charbonnier loss with first-gradient L1 loss."""

    return w_charb * charbonnier_loss(pred, target) + w_grad * gradient_l1(pred, target)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    if pred.shape[-2] > 1:
        pred_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + (pred_y - target_y).abs().mean()
    if pred.shape[-1] > 1:
        pred_x = pred[..., :, 1:] - pred[..., :, :-1]
        target_x = target[..., :, 1:] - target[..., :, :-1]
        loss = loss + (pred_x - target_x).abs().mean()
    return loss
