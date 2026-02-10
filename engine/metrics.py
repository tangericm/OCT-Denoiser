from __future__ import annotations

import numpy as np


def align_prediction_to_gt(
    pred2d: np.ndarray,
    gt2d: np.ndarray,
    bg_roi,
) -> tuple[np.ndarray, float]:
    """
    Align prediction to GT by matching the background/noise-floor mean (dB offset).

    Returns:
      aligned_pred: pred2d shifted by a constant offset
      shift_db: additive offset applied to prediction
    """
    y0b, y1b, x0b, x1b = bg_roi
    pred_bg = pred2d[y0b:y1b, x0b:x1b]
    gt_bg = gt2d[y0b:y1b, x0b:x1b]

    pred_bg = np.where(np.isfinite(pred_bg), pred_bg, np.nan)
    gt_bg = np.where(np.isfinite(gt_bg), gt_bg, np.nan)

    pred_bg_mean = float(np.nanmean(pred_bg))
    gt_bg_mean = float(np.nanmean(gt_bg))
    if not np.isfinite(pred_bg_mean) or not np.isfinite(gt_bg_mean):
        return pred2d, 0.0

    shift_db = gt_bg_mean - pred_bg_mean
    return pred2d + shift_db, float(shift_db)


def roi_snr_cnr(img2d: np.ndarray, sig_roi, bg_roi, eps: float = 1e-8) -> tuple[float, float]:
    """
    Compute SNR and CNR in dB using shared ROIs.

    img2d: [H,W] float32 (linear or log-compressed is fine as long as you're consistent)
    ROI format: (y0, y1, x0, x1), y1/x1 exclusive
    """
    y0s, y1s, x0s, x1s = sig_roi
    y0b, y1b, x0b, x1b = bg_roi

    x0 = max(x0s, x0b)
    x1 = min(x1s, x1b)
    if x1 <= x0:
        return float("nan"), float("nan")

    sig = img2d[y0s:y1s, x0:x1]
    bg = img2d[y0b:y1b, x0:x1]

    sig = (10 ** sig) - 1e-6
    bg = (10 ** bg) - 1e-6

    sig = np.where(np.isfinite(sig), sig, np.nan)
    bg = np.where(np.isfinite(bg), bg, np.nan)

    mean_max_sig = float(np.nanmean(np.nanmax(sig, axis=0)))
    mean_sig = float(np.nanmean(sig))
    std_bg = float(np.nanstd(bg))

    snr = 20.0 * np.log10((mean_max_sig + eps) / (std_bg + eps))
    cnr = 20.0 * np.log10((mean_sig + eps) / (std_bg + eps))
    if not np.isfinite(snr):
        snr = float("nan")
    if not np.isfinite(cnr):
        cnr = float("nan")
    return float(snr), float(cnr)


def roi_bounds(height: int, width: int, y0: int, y1: int, x_pad: int = 10) -> tuple[int, int, int, int]:
    """Clamp ROI with fixed x-range [x_pad, width - x_pad]."""
    x0 = max(0, x_pad)
    x1 = max(x0 + 1, width - x_pad)
    y0c = max(0, min(height - 1, int(y0)))
    y1c = max(y0c + 1, min(height, int(y1)))
    return y0c, y1c, x0, x1


def bg_bounds(
    height: int,
    width: int,
    *,
    x0: int,
    x1: int,
    rows: int = 20,
    x_pad: int = 10,
) -> tuple[int, int, int, int]:
    y1 = height
    y0 = max(0, height - rows)
    x_min = max(0, x_pad)
    x_max = max(x_min + 1, width - x_pad)
    x0c = max(x_min, min(x_max - 1, int(x0)))
    x1c = max(x0c + 1, min(x_max, int(x1)))
    return y0, y1, x0c, x1c
