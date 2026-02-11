from __future__ import annotations

import torch
import torch.nn.functional as F


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
    w_snr_loss: float = 0.0,
    snr_loss_t_peak: float = 0.1,
    snr_loss_t_bg: float = 0.1,
) -> torch.Tensor:
    """Compute the combined Charbonnier + gradient L1 + optional SNR loss."""
    loss = w_charb * charbonnier_loss(pred, target) + w_grad * gradient_l1(pred, target)
    if w_snr_loss > 0:
        snr_l, _ = smooth_snr_loss(pred, t_peak=snr_loss_t_peak, t_bg=snr_loss_t_bg)
        loss = loss + w_snr_loss * snr_l
    return loss


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean()


# ---------------------------------------------------------------------------
# Differentiable smooth SNR-based loss  (ROI-independent)
# ---------------------------------------------------------------------------
#
# Mathematical definition:
#   For image x [B, 1, H, W], flatten spatial dims to [B, N]:
#
#   1) Soft peak (signal strength):
#      w_peak  = softmax(x_flat / T_peak)          [B, N]
#      soft_peak = sum(w_peak * x_flat, dim=-1)     [B]
#
#   2) Soft background noise std:
#      w_bg    = softmax(-x_flat / T_bg)            [B, N]
#      mu_bg   = sum(w_bg * x_flat, dim=-1)         [B]
#      var_bg  = sum(w_bg * (x_flat - mu_bg)^2)     [B]
#      std_bg  = sqrt(var_bg + eps)                  [B]
#
#   3) SNR surrogate (higher = better):
#      snr     = soft_peak / std_bg                  [B]
#      loss    = -log(clamp(snr, min=eps)).mean()    scalar
#
# Properties:
#   - Fully differentiable (softmax + sqrt + log, no argmax/masks).
#   - No hand-crafted ROI: softmax weights discover bright / dark regions.
#   - Temperature params (T_peak, T_bg) control selection sharpness.
#   - Numerically stable via eps, clamp, and log-sum-exp in softmax.
#   - Operates on linear or log-scale images (set linear_mode accordingly).
# ---------------------------------------------------------------------------

def smooth_snr_loss(
    pred: torch.Tensor,
    *,
    t_peak: float = 0.1,
    t_bg: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Differentiable, ROI-free SNR surrogate loss.

    Parameters
    ----------
    pred : [B, 1, H, W]  predicted image (log-compressed float, any scale)
    t_peak : temperature for soft-peak weighting (lower = sharper)
    t_bg   : temperature for soft-background weighting (lower = sharper)
    eps    : numerical stability constant

    Returns
    -------
    loss : scalar tensor (lower = higher SNR)
    info : dict with 'soft_peak', 'soft_std_bg', 'snr' for logging
    """
    B = pred.shape[0]
    x = pred.reshape(B, -1)  # [B, N]

    # --- soft peak signal strength ---
    w_peak = F.softmax(x / t_peak, dim=-1)          # [B, N]
    soft_peak = (w_peak * x).sum(dim=-1)             # [B]

    # --- soft background noise std ---
    w_bg = F.softmax(-x / t_bg, dim=-1)              # [B, N]
    mu_bg = (w_bg * x).sum(dim=-1, keepdim=True)     # [B, 1]
    var_bg = (w_bg * (x - mu_bg) ** 2).sum(dim=-1)   # [B]
    std_bg = torch.sqrt(var_bg + eps)                 # [B]

    # --- SNR surrogate ---
    snr = soft_peak / std_bg                          # [B]
    loss = -torch.log(snr.clamp(min=eps)).mean()      # scalar

    info = {
        "soft_peak": soft_peak.detach().mean(),
        "soft_std_bg": std_bg.detach().mean(),
        "snr": snr.detach().mean(),
    }
    return loss, info
