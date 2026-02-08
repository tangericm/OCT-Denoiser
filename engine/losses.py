from __future__ import annotations

import torch
import torch.nn.functional as F

def _select_hw(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:
        return t[:, 0]
    if t.dim() == 3:
        return t
    raise ValueError(f"Expected tensor with shape [B,1,H,W] or [B,H,W], got {t.shape}")


def _roi_bounds(height: int, width: int, y0: int, y1: int, x_pad: int = 0) -> tuple[int, int, int, int]:
    """Clamp ROI with fixed x-range [x_pad, width - x_pad]."""
    x0 = max(0, x_pad)
    x1 = max(x0 + 1, width - x_pad)
    y0c = max(0, min(height - 1, int(y0)))
    y1c = max(y0c + 1, min(height, int(y1)))
    return y0c, y1c, x0, x1


def _bg_bounds(
    height: int,
    width: int,
    *,
    x0: int,
    x1: int,
    rows: int = 20,
    x_pad: int = 0,
) -> tuple[int, int, int, int]:
    y1 = height
    y0 = max(0, height - rows)
    x_min = max(0, x_pad)
    x_max = max(x_min + 1, width - x_pad)
    x0c = max(x_min, min(x_max - 1, int(x0)))
    x1c = max(x0c + 1, min(x_max, int(x1)))
    return y0, y1, x0c, x1c


def _roi_snr_cnr(
    img: torch.Tensor,
    *,
    sig_y0: int,
    sig_y1: int,
    bg_y0: int,
    bg_y1: int,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    img_hw = _select_hw(img)
    b, h, w = img_hw.shape
    sy0, sy1, sx0, sx1 = _roi_bounds(h, w, sig_y0, sig_y1)
    x0 = sx0
    x1 = sx1
    by0, by1, bx0, bx1 = _bg_bounds(h, w, x0=x0, x1=x1)
    x0 = max(x0, bx0)
    x1 = min(x1, bx1)
    if x1 <= x0:
        nan = img_hw.new_full((b,), float("nan"))
        return nan, nan

    sig = img_hw[:, sy0:sy1, x0:x1]
    bg = img_hw[:, by0:by1, x0:x1]

    # sig_lin = (10 ** sig) - 1e-6
    # bg_lin = (10 ** bg) - 1e-6
    sig_lin = sig
    bg_lin = bg

    max_sig_per_x = torch.logsumexp(sig_lin * 10.0, dim=1) / 10.0
    mean_peak = max_sig_per_x.mean(dim=1)
    std_bg = bg_lin.flatten(1).var(dim=1, unbiased=False).sqrt()
    snr_db = mean_peak / (std_bg + eps) 
    mean_sig = sig_lin.flatten(1).mean(dim=1)
    cnr_db = mean_sig / (std_bg + eps) 
    return snr_db, cnr_db

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    pred_linear = (10 ** pred) - 1e-6
    target_linear = (10 ** target) - 1e-6
    linear_diff = pred_linear - target_linear
    loss_linear = torch.mean(torch.sqrt(linear_diff ** 2 + eps**2))
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2)) #+ 1e-4*loss_linear


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_linear = (10 ** pred) - 1e-6
    target_linear = (10 ** target) - 1e-6
    dy_p_linear = pred_linear[..., 1:, :] - pred_linear[..., :-1, :]
    dx_p_linear = pred_linear[..., :, 1:] - pred_linear[..., :, :-1]
    dy_t_linear = target_linear[..., 1:, :] - target_linear[..., :-1, :]
    dx_t_linear = target_linear[..., :, 1:] - target_linear[..., :, :-1]
    linear_grad_loss = (dy_p_linear - dy_t_linear).abs().mean() + (dx_p_linear - dx_t_linear).abs().mean()

    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean() #+ 1e-4*linear_grad_loss


def snr_cnr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sig_y0: int,
    sig_y1: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_snr, pred_cnr = _roi_snr_cnr(pred, sig_y0=sig_y0, sig_y1=sig_y1, bg_y0=0, bg_y1=0)
    target_snr, target_cnr = _roi_snr_cnr(target, sig_y0=sig_y0, sig_y1=sig_y1, bg_y0=0, bg_y1=0)
    mask = torch.isfinite(pred_snr) & torch.isfinite(pred_cnr) & torch.isfinite(target_snr) & torch.isfinite(target_cnr)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    delta_snr = pred_snr[mask] - target_snr[mask]
    delta_cnr = pred_cnr[mask] - target_cnr[mask]  
    snr_cnr_loss = (F.softplus(-delta_snr) + F.softplus(-delta_cnr)).mean()
    return snr_cnr_loss
