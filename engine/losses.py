from __future__ import annotations

import torch


def unpack_batch(batch, device: str):
    """Unpack (x, y, meta) or (x, y) batch and move tensors to device."""
    if len(batch) == 2:
        x, y = batch
        meta = None
    else:
        x, y, meta = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta


def compute_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    w_charb: float,
    w_grad: float,
) -> torch.Tensor:
    """Compute the combined Charbonnier + gradient L1 loss."""
    return w_charb * charbonnier_loss(pred, target) + w_grad * gradient_l1(pred, target)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean()
