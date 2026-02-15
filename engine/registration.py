"""Robust image registration for OCT frame alignment.

Aligns predicted frames to reference frames under translation and orientation
changes using a coarse-to-fine approach:

  1. Coarse orientation search over the dihedral group D4
     (4 rotations x 2 optional flips = up to 8 hypotheses)
  2. Subpixel translation estimation via phase cross-correlation
  3. Optional fine angle refinement around best coarse orientation

All operations are deterministic and reproducible.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np
from scipy.ndimage import rotate as ndi_rotate, shift as ndi_shift
from skimage.registration import phase_cross_correlation


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
    score: float                # normalized cross-correlation of registered result
    success: bool               # whether registration met quality threshold
    note: str = ""              # e.g. "low_texture", "low_score"


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
    upsample_factor: int = 10,
    min_texture_std: float = 1e-6,
    success_threshold: float = 0.1,
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
        If True, also test left-right flipped orientations (doubles the
        number of coarse hypotheses).
    refine_angles : bool
        Whether to refine the angle around the best coarse orientation.
    refine_range_deg / refine_step_deg : float
        Range and step of the fine angle search.
    upsample_factor : int
        Sub-pixel precision factor for ``phase_cross_correlation``.
    min_texture_std : float
        Minimum standard deviation to consider a frame registrable.
    success_threshold : float
        Minimum NCC for the result to be flagged as successful.

    Returns
    -------
    registered : 2-D array
        Prediction aligned to the reference.
    result : FrameRegistrationResult
        Per-frame metadata (transform parameters, score, status).
    """
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")

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
    # Stage 1: Coarse orientation search (D4 subgroup)
    # ------------------------------------------------------------------
    best_score = -np.inf
    best_orientation = 0
    best_flip = False
    best_shift = (0.0, 0.0)

    flip_options = [False, True] if include_flips else [False]

    for angle_deg in (0, 90, 180, 270):
        for flip_lr in flip_options:
            candidate = _apply_orientation(pred, angle_deg, flip_lr)

            # 90/270 rotations change (H,W) -> (W,H) for non-square images.
            if candidate.shape != ref.shape:
                continue

            try:
                result_tuple = phase_cross_correlation(
                    ref.astype(np.float64),
                    candidate.astype(np.float64),
                    upsample_factor=upsample_factor,
                )
                shifts = result_tuple[0]
                dy, dx = float(shifts[0]), float(shifts[1])
            except Exception:
                continue

            aligned = ndi_shift(candidate, (dy, dx), order=3,
                                mode="constant", cval=0.0)
            score = _ncc(aligned, ref)

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

            candidate = pred.copy()
            if best_flip:
                candidate = np.ascontiguousarray(candidate[:, ::-1])
            candidate = ndi_rotate(candidate, angle, reshape=False, order=3,
                                   mode="constant", cval=0.0)

            if candidate.shape != ref.shape:
                continue

            try:
                result_tuple = phase_cross_correlation(
                    ref.astype(np.float64),
                    candidate.astype(np.float64),
                    upsample_factor=upsample_factor,
                )
                shifts = result_tuple[0]
                dy, dx = float(shifts[0]), float(shifts[1])
            except Exception:
                continue

            aligned = ndi_shift(candidate, (dy, dx), order=3,
                                mode="constant", cval=0.0)
            score = _ncc(aligned, ref)

            if score > best_score:
                best_score = score
                best_angle = float(angle)
                best_shift = (dy, dx)

    # ------------------------------------------------------------------
    # Stage 3: Apply final transform and report
    # ------------------------------------------------------------------
    dy_final, dx_final = best_shift
    registered = _apply_transform(pred, best_angle, best_flip, dy_final, dx_final)

    final_score = _ncc(registered, ref)
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
            f"score={result.score:.4f} [{status}]"
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
        "dy", "dx", "score", "success", "note",
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
