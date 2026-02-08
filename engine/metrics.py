from __future__ import annotations

import numpy as np


def roi_snr_cnr(img2d: np.ndarray, sig_roi, bg_roi, eps: float = 1e-8) -> tuple[float, float]:
    """
    Compute SNR and CNR in dB using shared ROIs.

    For each A-line (x column) in the ROI:
      SNR_x(dB) = 20*log10( (max(signal_y_range, x) + eps) / (std(background_y_range, x) + eps) )

    Then return mean(SNR_x) over all x in the ROI where bg std is finite.

    img2d: [H,W] float32 (linear or log-compressed is fine as long as you're consistent)
    ROI format: (y0, y1, x0, x1), y1/x1 exclusive
    """
    y0s, y1s, x0s, x1s = sig_roi
    y0b, y1b, x0b, x1b = bg_roi

    # Ensure the A-line (x) ranges match; otherwise you aren't averaging comparable columns.
    x0 = max(x0s, x0b)
    x1 = min(x1s, x1b)
    if x1 <= x0:
        return float("nan"), float("nan")

    # Extract ROIs, aligned in x
    sig = img2d[y0s:y1s, x0:x1]  # shape [Hs, Wx]
    bg = img2d[y0b:y1b, x0:x1]  # shape [Hb, Wx]

    sig = (10 ** sig) - 1e-6
    bg = (10 ** bg) - 1e-6

    # Per-column metrics
    max_sig_per_x = np.max(sig, axis=0)  # [Wx]
    std_bg_per_x = np.std(bg, axis=0)  # [Wx]

    # Compute SNR/CNR in dB (match training loss)
    snr_per_x = 20.0 * np.log10((np.mean(max_sig_per_x) + eps) / (np.std(bg) + eps))
    mean_sig = float(np.mean(sig))
    std_bg = float(np.std(bg))
    cnr = 20.0 * np.log10((mean_sig + eps) / (std_bg + eps))

    # # Robustness: ignore any non-finite values (can happen if img2d contains NaNs/Infs)
    # snr_per_x = snr_per_x[np.isfinite(snr_per_x)]
    # if snr_per_x.size == 0:
    #     return float("nan")

    # return float(np.mean(snr_per_x))
    return float(snr_per_x), float(cnr)


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
