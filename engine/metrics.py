from __future__ import annotations

import numpy as np


def roi_snr_cnr(
    img2d: np.ndarray,
    sig_roi,
    bg_roi,
    eps: float = 1e-8,
    sig_stat: str = "max",
) -> tuple[float, float]:
    """
    Compute SNR and CNR in dB using shared ROIs.

    img2d: [H,W] float32 in the domain you want to evaluate (e.g., linear intensity)
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

    sig = np.where(np.isfinite(sig), sig, np.nan)
    bg = np.where(np.isfinite(bg), bg, np.nan)

    sig_stat_key = sig_stat.lower().strip()
    if sig_stat_key == "max":
        signal_level = float(np.nanmean(np.nanmax(sig, axis=0)))
    elif sig_stat_key.startswith("p"):
        try:
            percentile_q = float(sig_stat_key[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid sig_stat '{sig_stat}'. Expected 'max' or 'p<percentile>' (e.g. 'p95').") from exc
        if not (0.0 <= percentile_q <= 100.0):
            raise ValueError(f"Invalid percentile in sig_stat '{sig_stat}'. Percentile must be in [0, 100].")
        signal_level = float(np.nanpercentile(sig, q=percentile_q))
    else:
        raise ValueError(f"Invalid sig_stat '{sig_stat}'. Expected 'max' or 'p<percentile>' (e.g. 'p95').")

    mean_sig = float(np.nanmean(sig))
    std_bg = float(np.nanstd(bg))

    snr = 20.0 * np.log10((signal_level + eps) / (std_bg + eps))
    cnr = 20.0 * np.log10((mean_sig + eps) / (std_bg + eps))
    if not np.isfinite(snr):
        snr = float("nan")
    if not np.isfinite(cnr):
        cnr = float("nan")
    return float(snr), float(cnr)


def to_physical_intensity(img: np.ndarray, meta: dict | None) -> np.ndarray:
    """Convert normalized log-domain image back to linear physical intensity."""
    if not meta:
        return img
    img_log = img * float(meta["target_sd"]) + float(meta["target_mu"])
    return np.maximum(10.0 ** img_log - float(meta["log_eps"]), 0.0)


def roi_bounds(height: int, width: int, y0: int, y1: int, x_pad: int = 10) -> tuple[int, int, int, int]:
    """Compute signal ROI clamped to image bounds with x-padding."""
    x0 = max(0, x_pad)
    x1 = max(x0 + 1, width - x_pad)
    y0c = max(0, min(height - 1, int(y0)))
    y1c = max(y0c + 1, min(height, int(y1)))
    return y0c, y1c, x0, x1


def bg_bounds(height: int, width: int, *, x0: int, x1: int, rows: int = 20, x_pad: int = 10) -> tuple[int, int, int, int]:
    """Compute background ROI (bottom rows of image) clamped to image bounds."""
    y1 = height
    y0 = max(0, height - rows)
    x_min = max(0, x_pad)
    x_max = max(x_min + 1, width - x_pad)
    x0c = max(x_min, min(x_max - 1, int(x0)))
    x1c = max(x0c + 1, min(x_max, int(x1)))
    return y0, y1, x0c, x1c
