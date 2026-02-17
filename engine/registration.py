"""Robust image registration for OCT frame alignment.

Aligns predicted frames to reference frames under translation and orientation
changes.  Designed for OCT B-scans where a thin tissue band occupies a
fraction of the image and the rest is dark/noisy background.

Strategy (coarse-to-fine, tissue-aware):

  0. **Tissue-ROI detection** — automatically find the row band that contains
     retinal tissue in both images; crop to that band so the background cannot
     dominate the correlation.
  1. **Coarse orientation search** over the dihedral group D4
     (4 rotations x 2 optional flips = up to 8 hypotheses).
  2. **Translation estimation** via Gaussian-smoothed FFT cross-correlation
     for a robust coarse integer-pixel shift, followed by subpixel NCC
     optimisation with Powell's method — similar to MATLAB's ``imregister``.
  3. **Optional fine angle refinement** around the best coarse orientation.
  4. **NCC scoring on the original tissue crop** for a meaningful quality
     metric.

All operations are deterministic and reproducible.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    rotate as ndi_rotate,
    shift as ndi_shift,
    sobel as ndi_sobel,
)
from scipy.optimize import minimize as sp_minimize
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameRegistrationResult:
    """Per-frame registration metadata."""
    frame_idx: int
    orientation_deg: int        # coarse orientation: 0, 90, 180, 270
    flip_lr: bool               # whether horizontal flip was applied
    refined_angle_deg: float    # final angle after optional refinement
    dy: float                   # translation in y (rows)
    dx: float                   # translation in x (cols)
    score: float                # NCC on tissue ROI of registered result
    success: bool               # whether registration met quality threshold
    tissue_y0: int = 0          # detected tissue band start row
    tissue_y1: int = 0          # detected tissue band end row
    note: str = ""              # e.g. "low_texture", "low_score"


# ---------------------------------------------------------------------------
# Tissue detection
# ---------------------------------------------------------------------------

def _detect_tissue_rows(
    img: np.ndarray,
    smooth_sigma: float = 5.0,
    threshold_frac: float = 0.35,
    min_band_height: int = 10,
) -> tuple[int, int]:
    """Find the row range containing retinal tissue.

    Computes a smoothed row-wise mean intensity profile, thresholds it to
    separate the bright tissue band from the dark background, and returns
    the first/last rows above threshold.

    Parameters
    ----------
    img : 2-D array [H, W]
    smooth_sigma : Gaussian smoothing sigma for the row profile
    threshold_frac : fraction between profile min and max to use as threshold
    min_band_height : minimum number of tissue rows to be considered valid

    Returns
    -------
    (y0, y1) : row range (y1 exclusive).  Falls back to full height if
                detection fails.
    """
    H = img.shape[0]
    row_means = img.astype(np.float64).mean(axis=1)
    smoothed = gaussian_filter1d(row_means, sigma=smooth_sigma)

    lo, hi = float(smoothed.min()), float(smoothed.max())
    if hi - lo < 1e-10:
        return 0, H

    threshold = lo + threshold_frac * (hi - lo)
    above = np.where(smoothed > threshold)[0]
    if len(above) < min_band_height:
        return 0, H

    return int(above[0]), int(above[-1] + 1)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _apply_orientation(img: np.ndarray, angle_deg: int, flip_lr: bool) -> np.ndarray:
    """Apply a coarse orientation transform (flip then rotate) to a 2-D image."""
    out = img
    if flip_lr:
        out = np.ascontiguousarray(out[:, ::-1])
    if angle_deg != 0:
        k = angle_deg // 90
        out = np.rot90(out, k=k)
    return out


def _apply_transform(
    img: np.ndarray,
    angle_deg: float,
    flip_lr: bool,
    dy: float,
    dx: float,
) -> np.ndarray:
    """Apply the full transform sequence: flip -> rotate -> translate.

    For exact 90-degree multiples, ``np.rot90`` is used for speed and
    precision.  Arbitrary angles fall back to ``scipy.ndimage.rotate``.
    """
    out = img.copy()
    if flip_lr:
        out = np.ascontiguousarray(out[:, ::-1])
    if angle_deg != 0.0:
        if angle_deg % 90.0 == 0.0:
            k = int(round(angle_deg)) // 90
            out = np.rot90(out, k=k)
        else:
            out = ndi_rotate(out, angle_deg, reshape=False, order=3,
                             mode="constant", cval=0.0)
    if dy != 0.0 or dx != 0.0:
        out = ndi_shift(out, (dy, dx), order=3, mode="constant", cval=0.0)
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation (Pearson) between two images.

    Returns 0.0 when either image has near-zero variance.
    """
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    a_std = a64.std()
    b_std = b64.std()
    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0
    return float(np.mean((a64 - a64.mean()) * (b64 - b64.mean())) / (a_std * b_std))


def _find_translation(
    ref_crop: np.ndarray,
    mov_crop: np.ndarray,
    max_shift: int = 30,
    smooth_sigma: float = 1.5,
) -> tuple[float, float]:
    """Estimate (dy, dx) to align *mov_crop* onto *ref_crop*.

    1. Compute Sobel gradient magnitude of both crops — this captures
       structural edges in both axes and is invariant to global
       intensity / noise-level differences (denoised-vs-noisy).
    2. Lightly Gaussian-smooth the edge maps to reduce noise.
    3. FFT cross-correlation for fast integer-pixel coarse alignment.
    4. Powell optimisation of NCC on the *original* crops for subpixel
       refinement.

    The key difference from scikit-image's ``phase_cross_correlation``
    is that plain FFT cross-correlation preserves the natural amplitude
    weighting of structural features, whereas phase normalisation can
    amplify noise and produce unreliable peaks in OCT data.

    Parameters
    ----------
    ref_crop, mov_crop : 2-D float arrays (same shape).
    max_shift : maximum pixel shift to consider (constrains search).
    smooth_sigma : Gaussian sigma applied to the Sobel edge maps before
        FFT cross-correlation.

    Returns
    -------
    (dy, dx) : shift for ``ndi_shift(mov_crop, (dy, dx))`` to align it
        to *ref_crop*.
    """
    ref64 = ref_crop.astype(np.float64)
    mov64 = mov_crop.astype(np.float64)

    # --- Edge enhancement: Sobel gradient magnitude ---
    # Captures structural boundaries in both y and x while suppressing
    # DC offset and noise-level differences between pred and ref.
    def _edge_mag(img: np.ndarray) -> np.ndarray:
        gx = ndi_sobel(img, axis=1)
        gy = ndi_sobel(img, axis=0)
        return np.sqrt(gx * gx + gy * gy)

    ref_e = gaussian_filter(_edge_mag(ref64), sigma=smooth_sigma)
    mov_e = gaussian_filter(_edge_mag(mov64), sigma=smooth_sigma)

    ref_zm = ref_e - ref_e.mean()
    mov_zm = mov_e - mov_e.mean()

    if ref_zm.std() < 1e-10 or mov_zm.std() < 1e-10:
        return 0.0, 0.0

    # --- Coarse: FFT cross-correlation on smoothed edge maps ---
    xcorr = fftconvolve(ref_zm, mov_zm[::-1, ::-1], mode="full")

    H, W = ref64.shape
    cy, cx = H - 1, W - 1  # zero-shift location in the xcorr array

    # Restrict peak search to ±max_shift
    y_lo = max(0, cy - max_shift)
    y_hi = min(xcorr.shape[0], cy + max_shift + 1)
    x_lo = max(0, cx - max_shift)
    x_hi = min(xcorr.shape[1], cx + max_shift + 1)

    roi = xcorr[y_lo:y_hi, x_lo:x_hi]
    peak = np.unravel_index(roi.argmax(), roi.shape)
    coarse_dy = float(peak[0] + y_lo - cy)
    coarse_dx = float(peak[1] + x_lo - cx)

    # --- Fine: optimise NCC on original (unsmoothed) crops ---
    def _neg_ncc(params):
        shifted = ndi_shift(mov64, (params[0], params[1]),
                            order=3, mode="constant", cval=0.0)
        return -_ncc(shifted, ref64)

    res = sp_minimize(
        _neg_ncc,
        [coarse_dy, coarse_dx],
        method="Powell",
        options={"xtol": 0.05, "ftol": 1e-8, "maxiter": 200},
    )

    return float(res.x[0]), float(res.x[1])


# ---------------------------------------------------------------------------
# Per-frame registration
# ---------------------------------------------------------------------------

def register_frame(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    frame_idx: int = 0,
    include_flips: bool = True,
    refine_angles: bool = True,
    refine_range_deg: float = 10.0,
    refine_step_deg: float = 1.0,
    max_shift: int = 30,
    min_texture_std: float = 1e-6,
    success_threshold: float = 0.3,
    # Tissue ROI options
    tissue_roi: tuple[int, int] | None = None,
    roi_pad: int = 30,
) -> tuple[np.ndarray, FrameRegistrationResult]:
    """Register a single predicted frame to a reference frame.

    Parameters
    ----------
    pred : 2-D array
        Predicted image.
    ref : 2-D array
        Reference image (same shape as *pred*).
    frame_idx : int
        Frame index stored in the returned metadata.
    include_flips : bool
        If True, also test left-right flipped orientations.
    refine_angles : bool
        Whether to refine the angle around the best coarse orientation.
    refine_range_deg / refine_step_deg : float
        Range and step of the fine angle search.
    max_shift : int
        Maximum translation in pixels to consider during alignment.
    min_texture_std : float
        Minimum standard deviation to consider a frame registrable.
    success_threshold : float
        Minimum NCC (on tissue ROI) for the result to be flagged as
        successful.
    tissue_roi : (y0, y1) or None
        Explicit row range of the tissue band.  If *None*, the tissue
        band is auto-detected from the reference image.
    roi_pad : int
        Padding (in rows) added above/below the tissue band to
        accommodate vertical shifts during registration.

    Returns
    -------
    registered : 2-D array
        Prediction aligned to the reference (full-frame).
    result : FrameRegistrationResult
        Per-frame metadata (transform parameters, score, status).
    """
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")

    H, W = ref.shape

    # --- Guard: low-texture frames ---
    if pred.std() < min_texture_std or ref.std() < min_texture_std:
        return pred.copy(), FrameRegistrationResult(
            frame_idx=frame_idx,
            orientation_deg=0,
            flip_lr=False,
            refined_angle_deg=0.0,
            dy=0.0, dx=0.0,
            score=0.0,
            success=False,
            note="low_texture",
        )

    # ------------------------------------------------------------------
    # Stage 0: Tissue ROI detection
    # ------------------------------------------------------------------
    if tissue_roi is not None:
        t_y0, t_y1 = tissue_roi
    else:
        # Detect tissue band from both images and take the union
        ref_y0, ref_y1 = _detect_tissue_rows(ref)
        pred_y0, pred_y1 = _detect_tissue_rows(pred)
        t_y0 = min(ref_y0, pred_y0)
        t_y1 = max(ref_y1, pred_y1)

    # Add padding so vertical shifts are visible inside the crop
    t_y0 = max(0, t_y0 - roi_pad)
    t_y1 = min(H, t_y1 + roi_pad)

    # Crop reference for ROI-based operations
    ref_crop = ref[t_y0:t_y1, :]

    # ------------------------------------------------------------------
    # Stage 1: Coarse orientation search (D4 subgroup)
    # ------------------------------------------------------------------
    best_score = -np.inf
    best_orientation = 0
    best_flip = False
    best_shift = (0.0, 0.0)

    flip_options = [False, True] if include_flips else [False]

    for angle_deg in (0, 90, 180, 270):
        for flip_lr in flip_options:
            candidate_full = _apply_orientation(pred, angle_deg, flip_lr)

            # 90/270 rotations change (H,W) -> (W,H) for non-square images.
            if candidate_full.shape != ref.shape:
                continue

            # Crop the oriented candidate to the same tissue ROI
            cand_crop = candidate_full[t_y0:t_y1, :]

            try:
                dy, dx = _find_translation(ref_crop, cand_crop,
                                           max_shift=max_shift)
            except Exception:
                continue

            # Score using NCC on original tissue crop
            aligned_crop = ndi_shift(cand_crop, (dy, dx), order=3,
                                     mode="constant", cval=0.0)
            score = _ncc(aligned_crop, ref_crop)

            if score > best_score:
                best_score = score
                best_orientation = angle_deg
                best_flip = flip_lr
                best_shift = (dy, dx)

    # ------------------------------------------------------------------
    # Stage 2: Fine angle refinement around best coarse orientation
    # ------------------------------------------------------------------
    best_angle = float(best_orientation)

    if refine_angles and refine_range_deg > 0:
        fine_angles = np.arange(
            best_orientation - refine_range_deg,
            best_orientation + refine_range_deg + refine_step_deg * 0.5,
            refine_step_deg,
        )
        for angle in fine_angles:
            # Skip the coarse angle itself (already evaluated)
            if abs(angle - best_orientation) < 0.01:
                continue

            candidate_full = pred.copy()
            if best_flip:
                candidate_full = np.ascontiguousarray(candidate_full[:, ::-1])
            candidate_full = ndi_rotate(candidate_full, angle, reshape=False,
                                        order=3, mode="constant", cval=0.0)

            if candidate_full.shape != ref.shape:
                continue

            cand_crop = candidate_full[t_y0:t_y1, :]

            try:
                dy, dx = _find_translation(ref_crop, cand_crop,
                                           max_shift=max_shift)
            except Exception:
                continue

            aligned_crop = ndi_shift(cand_crop, (dy, dx), order=3,
                                     mode="constant", cval=0.0)
            score = _ncc(aligned_crop, ref_crop)

            if score > best_score:
                best_score = score
                best_angle = float(angle)
                best_shift = (dy, dx)

    # ------------------------------------------------------------------
    # Stage 3: Apply final transform to *full frame* and report
    # ------------------------------------------------------------------
    dy_final, dx_final = best_shift
    registered = _apply_transform(pred, best_angle, best_flip, dy_final, dx_final)

    # Final score: NCC on tissue ROI of the registered full frame
    final_score = _ncc(registered[t_y0:t_y1, :], ref_crop)
    success = final_score >= success_threshold

    return registered, FrameRegistrationResult(
        frame_idx=frame_idx,
        orientation_deg=best_orientation,
        flip_lr=best_flip,
        refined_angle_deg=best_angle,
        dy=dy_final,
        dx=dx_final,
        score=final_score,
        success=success,
        tissue_y0=t_y0,
        tissue_y1=t_y1,
        note="" if success else "low_score",
    )


# ---------------------------------------------------------------------------
# Stack-level operations
# ---------------------------------------------------------------------------

def register_stack(
    preds: np.ndarray,
    refs: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, list[FrameRegistrationResult]]:
    """Register every frame in *preds* to the corresponding frame in *refs*.

    Parameters
    ----------
    preds, refs : [F, H, W] arrays
    **kwargs : forwarded to :func:`register_frame`

    Returns
    -------
    registered : [F, H, W] registered prediction stack
    results : list of :class:`FrameRegistrationResult`
    """
    F = preds.shape[0]
    if refs.shape[0] != F:
        raise ValueError(f"Stack length mismatch: preds {preds.shape[0]} vs refs {refs.shape[0]}")

    registered = np.zeros_like(preds)
    results: list[FrameRegistrationResult] = []

    for i in range(F):
        reg_frame, result = register_frame(
            preds[i], refs[i], frame_idx=i, **kwargs
        )
        registered[i] = reg_frame
        results.append(result)

        status = "OK" if result.success else "FALLBACK"
        print(
            f"[REG] Frame {i + 1}/{F}: orient={result.orientation_deg}\u00b0 "
            f"flip_lr={result.flip_lr} angle={result.refined_angle_deg:.1f}\u00b0 "
            f"shift=({result.dy:.2f}, {result.dx:.2f}) "
            f"score={result.score:.4f} "
            f"tissue=[{result.tissue_y0}:{result.tissue_y1}] [{status}]"
        )

    return registered, results


def apply_registration_to_stack(
    stack: np.ndarray,
    results: list[FrameRegistrationResult],
) -> np.ndarray:
    """Apply previously computed registration transforms to another stack.

    Useful for transforming the ground-truth stack with the same parameters
    derived from prediction-to-reference alignment.

    Parameters
    ----------
    stack : [F, H, W]
    results : per-frame :class:`FrameRegistrationResult` list

    Returns
    -------
    transformed : [F, H, W]
    """
    F = stack.shape[0]
    if len(results) != F:
        raise ValueError(
            f"Stack has {F} frames but {len(results)} registration results"
        )

    transformed = np.zeros_like(stack)
    for i, res in enumerate(results):
        if not res.success:
            transformed[i] = stack[i]
            continue
        transformed[i] = _apply_transform(
            stack[i], res.refined_angle_deg, res.flip_lr, res.dy, res.dx,
        )

    return transformed


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_registration_csv(
    path: str,
    results: list[FrameRegistrationResult],
) -> None:
    """Write per-frame registration metadata to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "frame_idx", "orientation_deg", "flip_lr", "refined_angle_deg",
        "dy", "dx", "score", "success", "tissue_y0", "tissue_y1", "note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def save_registration_json(
    path: str,
    results: list[FrameRegistrationResult],
) -> None:
    """Write per-frame registration metadata to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
