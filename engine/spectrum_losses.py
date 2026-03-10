"""Image and spectrum-domain losses for spectrum training (real spectra)."""
from __future__ import annotations

import torch

from engine.losses import charbonnier_loss as charbonnier_1d


def gradient_l1_1d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-1] <= 1:
        return pred.new_tensor(0.0)
    dp = pred[..., 1:] - pred[..., :-1]
    dt = target[..., 1:] - target[..., :-1]
    return (dp - dt).abs().mean()


def _flatten_optional_width(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() == 4:
        b, c, l, w = pred.shape
        pred = pred.permute(0, 3, 1, 2).reshape(b * w, c, l).contiguous()
        target = target.permute(0, 3, 1, 2).reshape(b * w, c, l).contiguous()
    return pred, target


def spectrum_image_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_depth: tuple[int, int],
    log_eps: float,
    apply_fftshift: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-domain loss for real spectra [B, 1, L] or [B, 1, L, W]."""
    pred, target = _flatten_optional_width(pred, target)

    # pred, target: [B, 1, L] — real spectra; cast to complex for ifft
    pred_mag = torch.fft.ifft(pred[:, 0].to(torch.complex64), dim=-1).abs()
    tgt_mag  = torch.fft.ifft(target[:, 0].to(torch.complex64), dim=-1).abs()

    if apply_fftshift:
        pred_mag = torch.fft.fftshift(pred_mag, dim=-1)
        tgt_mag  = torch.fft.fftshift(tgt_mag,  dim=-1)

    z0, z1 = crop_depth
    pred_crop = torch.log10(pred_mag[:, z0:z1] + log_eps)
    tgt_crop  = torch.log10(tgt_mag[:, z0:z1]  + log_eps)

    charb = charbonnier_1d(pred_crop, tgt_crop)
    grad  = gradient_l1_1d(pred_crop, tgt_crop)
    return charb, grad


def spectral_mag_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on spectral magnitude for real spectra [B, 1, L]."""
    pred, target = _flatten_optional_width(pred, target)
    return (pred[:, 0].abs() - target[:, 0].abs()).abs().mean()


def compute_spectrum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    crop_depth: tuple[int, int],
    log_eps: float,
    apply_fftshift: bool,
    w_charb: float,
    w_grad: float,
    w_spec_mag: float,
) -> torch.Tensor:
    """Combined image-domain and spectral-magnitude loss for real spectra."""
    charb, grad = spectrum_image_terms(pred, target, crop_depth, log_eps, apply_fftshift)
    spec_mag = spectral_mag_term(pred, target)
    return w_charb * charb + w_grad * grad + w_spec_mag * spec_mag
