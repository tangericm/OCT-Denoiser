"""Preprocessing helpers for registration: normalization, CLAHE, etc."""
from __future__ import annotations

import numpy as np
import cv2


def normalize_percentile(
    img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0
) -> np.ndarray:
    """Percentile-clip and rescale *img* to [0, 1] float32."""
    lo, hi = np.percentile(img, [p_lo, p_hi])
    out = (img.astype(np.float32) - lo) / (hi - lo + 1e-7)
    return np.clip(out, 0.0, 1.0)


def apply_clahe(
    img: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 4
) -> np.ndarray:
    """Apply CLAHE to a [0,1] float32 image; returns [0,1] float32."""
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)
    )
    enhanced = clahe.apply(u8)
    return enhanced.astype(np.float32) / 255.0


def prepare_for_registration(
    stack: np.ndarray,
    use_clahe: bool = True,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    clahe_clip: float = 3.0,
) -> np.ndarray:
    """Create a preprocessed copy of *stack* optimized for registration.

    Parameters
    ----------
    stack : [N, H, W] float32 raw images.
    use_clahe : apply CLAHE after percentile normalization.
    p_lo, p_hi : percentile bounds for normalization.
    clahe_clip : CLAHE clip limit.

    Returns
    -------
    [N, H, W] float32 array in [0, 1].
    """
    N = stack.shape[0]
    out = np.empty_like(stack)
    for i in range(N):
        frame = normalize_percentile(stack[i], p_lo, p_hi)
        if use_clahe:
            frame = apply_clahe(frame, clip_limit=clahe_clip)
        out[i] = frame
    return out
