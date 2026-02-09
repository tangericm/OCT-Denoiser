from __future__ import annotations

import torch
from engine.metrics import roi_bounds, bg_bounds

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))

def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean()


def roi_snr_loss(
    pred: torch.Tensor,
    snr_sig_y0: int,
    snr_sig_y1: int,
    weight_bg: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_img = pred[:, 0] if pred.dim() == 4 else pred
    height, width = pred_img.shape[-2], pred_img.shape[-1]
    sig_roi = roi_bounds(height, width, snr_sig_y0, snr_sig_y1)
    sy0, sy1, sx0, sx1 = sig_roi
    by0, by1, bx0, bx1 = bg_bounds(height, width, x0=sx0, x1=sx1)

    sig = pred_img[..., sy0:sy1, sx0:sx1]
    bg = pred_img[..., by0:by1, bx0:bx1]

    sig_mean = sig.mean(dim=(-2, -1))
    bg_std = bg.std(dim=(-2, -1), unbiased=False)
    snr_loss = -20.0 * torch.log10((sig_mean + eps) / (bg_std + eps))
    bg_penalty = weight_bg * bg_std
    return (snr_loss + bg_penalty).mean()
