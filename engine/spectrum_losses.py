"""Image-domain loss for spectrum training."""
from __future__ import annotations

import torch


def charbonnier_1d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def spectrum_image_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_depth: tuple[int, int],
    log_eps: float,
    apply_fftshift: bool,
) -> torch.Tensor:
    """Image-domain loss via differentiable IFFT.

    pred, target: [B, 2, L] or [B, 2, L, W] (real, imag channels)
    Returns scalar loss comparing log-magnitude depth profiles after IFFT.
    """
    if pred.dim() == 4:
        B, C, L, W = pred.shape
        pred = pred.permute(0, 3, 1, 2).reshape(B * W, C, L).contiguous()
        target = target.permute(0, 3, 1, 2).reshape(B * W, C, L).contiguous()

    pred_c = torch.complex(pred[:, 0].float(), pred[:, 1].float())
    tgt_c = torch.complex(target[:, 0].float(), target[:, 1].float())

    pred_mag = torch.fft.ifft(pred_c, dim=-1).abs()
    tgt_mag = torch.fft.ifft(tgt_c, dim=-1).abs()

    if apply_fftshift:
        pred_mag = torch.fft.fftshift(pred_mag, dim=-1)
        tgt_mag = torch.fft.fftshift(tgt_mag, dim=-1)

    z0, z1 = crop_depth
    pred_crop = torch.log10(pred_mag[:, z0:z1] + log_eps)
    tgt_crop = torch.log10(tgt_mag[:, z0:z1] + log_eps)
    return charbonnier_1d(pred_crop, tgt_crop)


def compute_spectrum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    crop_depth: tuple[int, int],
    log_eps: float,
    apply_fftshift: bool,
) -> torch.Tensor:
    """Pure image-domain spectrum training loss."""
    return spectrum_image_loss(pred, target, crop_depth, log_eps, apply_fftshift)
