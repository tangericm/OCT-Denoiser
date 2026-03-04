"""
registration.py — Sub-pixel rigid + strip-wise registration & averaging for OCT B-scans.

Loads a multi-page TIFF stack, aligns all frames to an iteratively refined
reference using phase-correlation (sub-pixel translation) and optional rotation,
then exports a sharp averaged image.

Usage
-----
    Edit the parameters in the ``if __name__`` block at the bottom, then
    run the file directly (F5 in VS Code).

Dependencies
------------
    numpy, scipy, tifffile, matplotlib

Author: Eric Tang (tangericm)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tifffile
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import fourier_shift, shift as ndi_shift, rotate as ndi_rotate


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """All tunables for the registration pipeline."""

    # --- sub-pixel translation ---
    upsample_factor: int = 100
    """Phase-correlation up-sampling factor (100 → 0.01 px accuracy)."""

    # --- rotation ---
    estimate_rotation: bool = True
    """If True, estimate small rotation via brute-force angular search."""
    max_rotation_deg: float = 2.0
    """Maximum rotation angle to search (degrees, symmetric ±)."""
    rotation_step_deg: float = 0.05
    """Angular resolution for the search grid."""

    # --- strip-wise (non-rigid) registration ---
    strip_registration: bool = True
    """Split each frame into horizontal strips and register each independently
    to correct for non-rigid axial motion (common in OCT)."""
    n_strips: int = 8
    """Number of horizontal strips to divide each frame into."""
    strip_overlap: int = 16
    """Overlap in pixels between adjacent strips (for blending)."""

    # --- iterative reference refinement ---
    n_refine_iters: int = 3
    """Number of register→average→re-register iterations."""

    # --- quality gating ---
    correlation_threshold: float = 0.3
    """Frames with peak cross-correlation below this are discarded."""

    # --- output ---
    output_dtype: str = "uint16"
    """TIFF output dtype: 'uint8', 'uint16', or 'float32'."""
    p_lo: float = 0.5
    """Low percentile for robust intensity scaling."""
    p_hi: float = 99.5
    """High percentile for robust intensity scaling."""


# ---------------------------------------------------------------------------
#  Core: up-sampled phase cross-correlation  (translation)
# ---------------------------------------------------------------------------

def _upsampled_cross_correlation(
    ref: np.ndarray,
    mov: np.ndarray,
    upsample_factor: int = 100,
) -> Tuple[float, float, float]:
    """Return (shift_row, shift_col, peak_correlation) using the method of
    Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 33 (2008).

    Two stages:
      1. Pixel-level peak from the full cross-power spectrum.
      2. Sub-pixel refinement via DFT up-sampling around the peak.
    """
    src_freq = fft2(ref)
    tgt_freq = fft2(mov)
    shape = ref.shape

    # Cross-power spectrum
    cross_power = src_freq * np.conj(tgt_freq)
    eps = np.finfo(cross_power.dtype).eps
    cross_power /= np.abs(cross_power) + eps

    # --- Stage 1: pixel-level peak ---
    cc_image = np.real(ifft2(cross_power))
    maxima = np.unravel_index(np.argmax(cc_image), shape)
    midpoints = np.array([dim // 2 for dim in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        peak = cc_image[maxima]
        return float(shifts[0]), float(shifts[1]), float(peak)

    # --- Stage 2: sub-pixel refinement via up-sampled DFT ---
    shifts = np.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = int(np.ceil(upsample_factor * 1.5))
    dft_shift = np.fix(upsampled_region_size / 2.0)

    sample_region_offset = dft_shift - shifts * upsample_factor

    cc_up = _upsampled_dft(
        cross_power,
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    )
    cc_up = np.conj(cc_up)

    maxima_up = np.unravel_index(np.argmax(np.abs(cc_up)), cc_up.shape)
    peak = np.abs(cc_up[maxima_up])

    shifts_up = np.array(maxima_up, dtype=np.float64) - dft_shift
    shifts = shifts + shifts_up / upsample_factor

    # Normalise peak to [0, 1]
    ref_norm = np.sqrt(np.sum(ref * ref))
    mov_norm = np.sqrt(np.sum(mov * mov))
    peak /= (ref_norm * mov_norm + eps)

    return float(shifts[0]), float(shifts[1]), float(peak)


def _upsampled_dft(
    data: np.ndarray,
    upsampled_region_size: int,
    upsample_factor: int,
    axis_offsets: np.ndarray,
) -> np.ndarray:
    """Compute small regions of the DFT of *data* on a finer grid, using the
    matrix-multiply DFT algorithm of Guizar-Sicairos et al."""
    nr, nc = data.shape
    row_kern = np.exp(
        (-1j * 2 * np.pi / (nr * upsample_factor))
        * (np.fft.ifftshift(np.arange(nr))[:, None] - np.floor(nr / 2))
        @ (np.arange(upsampled_region_size)[None, :] - axis_offsets[0])
    )
    col_kern = np.exp(
        (-1j * 2 * np.pi / (nc * upsample_factor))
        * (np.arange(upsampled_region_size)[:, None] - axis_offsets[1])
        @ (np.fft.ifftshift(np.arange(nc))[None, :] - np.floor(nc / 2))
    )
    return row_kern.T @ data @ col_kern


# ---------------------------------------------------------------------------
#  Core: rotation estimation
# ---------------------------------------------------------------------------

def _estimate_rotation(
    ref: np.ndarray,
    mov: np.ndarray,
    max_deg: float = 2.0,
    step_deg: float = 0.05,
) -> float:
    """Brute-force small-angle rotation search maximising cross-correlation.

    Returns the rotation angle (degrees, counter-clockwise) that best aligns
    *mov* to *ref*.  Only searches ±max_deg in increments of step_deg.
    """
    best_corr = -np.inf
    best_angle = 0.0
    angles = np.arange(-max_deg, max_deg + step_deg / 2, step_deg)

    # Use a centre crop to speed up correlation
    rh, rw = ref.shape
    margin_r, margin_c = rh // 6, rw // 6
    ref_crop = ref[margin_r:rh - margin_r, margin_c:rw - margin_c]
    ref_crop = ref_crop - ref_crop.mean()
    ref_norm = np.linalg.norm(ref_crop) + 1e-12

    for angle in angles:
        rotated = ndi_rotate(mov, angle, reshape=False, order=3, mode="reflect")
        rot_crop = rotated[margin_r:rh - margin_r, margin_c:rw - margin_c]
        rot_crop = rot_crop - rot_crop.mean()
        corr = np.sum(ref_crop * rot_crop) / (ref_norm * (np.linalg.norm(rot_crop) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_angle = angle

    return best_angle


# ---------------------------------------------------------------------------
#  Core: strip-wise non-rigid registration
# ---------------------------------------------------------------------------

def _blend_weights(strip_h: int, overlap: int) -> np.ndarray:
    """Create a 1-D vertical blending ramp for strip overlap regions."""
    w = np.ones(strip_h, dtype=np.float64)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


def _register_stripwise(
    ref: np.ndarray,
    mov: np.ndarray,
    n_strips: int = 8,
    overlap: int = 16,
    upsample_factor: int = 100,
) -> np.ndarray:
    """Register *mov* to *ref* using independent per-strip translations.

    Each horizontal strip is shifted independently, then strips are blended
    with cosine ramps in the overlap regions.  This corrects the non-rigid
    axial bulk motion that is the main source of blurriness in OCT averaging.
    """
    h, w = ref.shape
    strip_h = h // n_strips
    out = np.zeros_like(mov, dtype=np.float64)
    weight_map = np.zeros(h, dtype=np.float64)

    for i in range(n_strips):
        y0 = max(i * strip_h - overlap, 0)
        y1 = min((i + 1) * strip_h + overlap, h)

        ref_strip = ref[y0:y1, :]
        mov_strip = mov[y0:y1, :]

        dy, dx, _ = _upsampled_cross_correlation(ref_strip, mov_strip, upsample_factor)

        shifted_strip = ndi_shift(mov_strip, (dy, dx), order=3, mode="reflect")

        bw = _blend_weights(y1 - y0, overlap)
        for row_idx in range(y1 - y0):
            out[y0 + row_idx, :] += shifted_strip[row_idx, :] * bw[row_idx]
            weight_map[y0 + row_idx] += bw[row_idx]

    # Normalise by accumulated weight
    weight_map = np.maximum(weight_map, 1e-12)
    for row_idx in range(h):
        out[row_idx, :] /= weight_map[row_idx]

    return out


# ---------------------------------------------------------------------------
#  High-level pipeline
# ---------------------------------------------------------------------------

def register_and_average(
    stack: np.ndarray,
    cfg: Optional[RegistrationConfig] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register a [N, H, W] float32 stack and return (averaged_image, weights).

    Parameters
    ----------
    stack : ndarray, shape (N, H, W)
        Input B-scan stack (float, any range).
    cfg : RegistrationConfig, optional
        Pipeline parameters.  Uses defaults if None.
    verbose : bool
        Print progress info.

    Returns
    -------
    avg : ndarray, shape (H, W)
        Registered average.
    weights : ndarray, shape (N,)
        Per-frame quality weight (0 for rejected frames).
    """
    if cfg is None:
        cfg = RegistrationConfig()

    n_frames, h, w = stack.shape
    if verbose:
        print(f"[reg] {n_frames} frames  {h}×{w} px  |  "
              f"strips={cfg.n_strips if cfg.strip_registration else 'off'}  "
              f"rotation={'on' if cfg.estimate_rotation else 'off'}  "
              f"upsample={cfg.upsample_factor}  "
              f"refine_iters={cfg.n_refine_iters}")

    stack = stack.astype(np.float64, copy=True)

    # ---- Initial reference: median of the stack (robust to outliers) ----
    reference = np.median(stack, axis=0)

    weights = np.ones(n_frames, dtype=np.float64)
    aligned = np.empty_like(stack)

    for iteration in range(cfg.n_refine_iters):
        if verbose:
            print(f"[reg] === iteration {iteration + 1}/{cfg.n_refine_iters} ===")

        frame_shifts = []

        for fi in range(n_frames):
            frame = stack[fi]

            # --- Rotation ---
            angle = 0.0
            if cfg.estimate_rotation:
                angle = _estimate_rotation(
                    reference, frame,
                    max_deg=cfg.max_rotation_deg,
                    step_deg=cfg.rotation_step_deg,
                )
                if abs(angle) > 1e-4:
                    frame = ndi_rotate(frame, angle, reshape=False, order=3, mode="reflect")

            # --- Global rigid translation ---
            dy, dx, peak = _upsampled_cross_correlation(
                reference, frame, cfg.upsample_factor,
            )

            if peak < cfg.correlation_threshold:
                weights[fi] = 0.0
                aligned[fi] = 0.0
                if verbose:
                    print(f"  frame {fi:3d}:  REJECTED  (corr={peak:.3f})")
                continue

            weights[fi] = peak  # quality weighting

            # Apply global shift
            frame = ndi_shift(frame, (dy, dx), order=3, mode="reflect")

            # --- Strip-wise non-rigid correction ---
            if cfg.strip_registration:
                frame = _register_stripwise(
                    reference, frame,
                    n_strips=cfg.n_strips,
                    overlap=cfg.strip_overlap,
                    upsample_factor=cfg.upsample_factor,
                )

            aligned[fi] = frame
            frame_shifts.append((fi, dy, dx, angle, peak))

        # --- Weighted average for new reference ---
        w = weights.copy()
        w_sum = w.sum()
        if w_sum < 1e-12:
            print("[reg] WARNING: all frames rejected, returning simple mean")
            return np.mean(stack, axis=0).astype(np.float32), weights.astype(np.float32)

        reference = np.einsum("fhw,f->hw", aligned, w) / w_sum

        if verbose:
            n_kept = int(np.sum(w > 0))
            print(f"[reg]   kept {n_kept}/{n_frames} frames  "
                  f"(mean corr = {np.mean(w[w > 0]):.4f})")

    return reference.astype(np.float32), weights.astype(np.float32)


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def load_tiff_stack(path: str) -> np.ndarray:
    """Load a (possibly multi-page) TIFF and return [N, H, W] float32."""
    with tifffile.TiffFile(path) as tif:
        stack = tif.asarray()
    stack = np.squeeze(stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    stack = stack.astype(np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D TIFF, got shape {stack.shape}")
    return stack


def save_tiff(
    path: str,
    image: np.ndarray,
    dtype: str = "uint16",
    p_lo: float = 0.5,
    p_hi: float = 99.5,
) -> None:
    """Save a single 2-D image to TIFF with robust percentile scaling."""
    img = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(img, [p_lo, p_hi])
    img = np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    if dtype == "float32":
        out = img.astype(np.float32)
    elif dtype == "uint8":
        out = (img * 255.0).astype(np.uint8)
    elif dtype == "uint16":
        out = (img * 65535.0).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(path, out, photometric="minisblack")
    print(f"[reg] saved → {path}  ({dtype}, {out.shape})")


# ---------------------------------------------------------------------------
#  Run directly in VS Code — edit parameters below
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---- Edit these ----
    direc = r"runs\A-Line\1D_npatch=256\predictions_tiff\6mm_1024Aline"
    file = "gt_6mm_1024Aline_s005_g060.tiff"
    output = None  # defaults to <direc>/registered_<file>

    cfg = RegistrationConfig(
        upsample_factor=100,        # 100 → 0.01 px accuracy
        estimate_rotation=True,
        max_rotation_deg=2.0,
        rotation_step_deg=0.05,
        strip_registration=True,
        n_strips=8,
        strip_overlap=16,
        n_refine_iters=3,
        correlation_threshold=0.3,
        output_dtype="uint16",
    )
    # --------------------

    tiff_path = os.path.join(direc, file)
    output_path = output or os.path.join(direc, f"registered_{file}")

    print(f"[reg] loading {tiff_path}")
    stack = load_tiff_stack(tiff_path)
    print(f"[reg] stack shape: {stack.shape}  dtype: {stack.dtype}")

    avg, weights = register_and_average(stack, cfg, verbose=True)

    n_kept = int(np.sum(weights > 0))
    print(f"[reg] done — {n_kept}/{stack.shape[0]} frames used")

    save_tiff(output_path, avg, dtype=cfg.output_dtype)

    # ---- Display comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    naive_avg = np.mean(stack, axis=0)

    axes[0].imshow(naive_avg, cmap="gray", aspect="auto")
    axes[0].set_title("Naive mean (no registration)")
    axes[1].imshow(avg, cmap="gray", aspect="auto")
    axes[1].set_title(f"Registered mean ({n_kept} frames)")
    axes[2].imshow(np.abs(avg - naive_avg), cmap="hot", aspect="auto")
    axes[2].set_title("| registered − naive |")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path.replace(".tif", "_comparison.png"), dpi=150)
    plt.show()
