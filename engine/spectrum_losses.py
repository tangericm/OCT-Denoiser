"""Image and spectrum-domain losses for spectrum training."""
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
    pred, target = _flatten_optional_width(pred, target)

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

    charb = charbonnier_1d(pred_crop, tgt_crop)
    grad = gradient_l1_1d(pred_crop, tgt_crop)
    return charb, grad


def spectral_complex_terms(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred, target = _flatten_optional_width(pred, target)

    pred_c = torch.complex(pred[:, 0].float(), pred[:, 1].float())
    tgt_c = torch.complex(target[:, 0].float(), target[:, 1].float())

    mag_loss = (pred_c.abs() - tgt_c.abs()).abs().mean()
    phase_delta = torch.angle(pred_c) - torch.angle(tgt_c)
    phase_loss = (1.0 - torch.cos(phase_delta)).mean()
    return mag_loss, phase_loss


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
    w_spec_phase: float,
) -> torch.Tensor:
    """Combined image-domain and spectral-domain complex loss."""
    charb, grad = spectrum_image_terms(pred, target, crop_depth, log_eps, apply_fftshift)
    spec_mag, spec_phase = spectral_complex_terms(pred, target)
    return (
        w_charb * charb
        + w_grad * grad
        + w_spec_mag * spec_mag
        + w_spec_phase * spec_phase
    )
