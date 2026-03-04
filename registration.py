"""
registration.py — Sub-pixel rigid + strip-wise registration & averaging for OCT B-scans.

Loads a multi-page TIFF stack, aligns all frames to an iteratively refined
reference using phase-correlation (sub-pixel translation) and optional rotation,
then exports a sharp averaged image.

Usage
-----
    Edit the parameters in the ``if __name__`` block at the bottom, then
    run the file directly (F5 in VS Code).

Pipeline stages
---------------
    1. Global translation  — gradient-image phase correlation, optionally
                             with a 2-level coarse-to-fine pyramid.
    2. Optional rotation   — brute-force NCC search (disabled by default).
    3. Non-rigid strip-wise — per-strip residual shifts on gradient images,
                              signal-gated so dark regions are never warped.
    4. Iterative refinement — repeat from a quality-weighted average reference.
    5. Robust averaging    — mean / trimmed-mean / quality-weighted.
    6. Optional sharpening — mild unsharp mask on the final average.

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
from numpy.fft import fft2, ifft2
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    rotate as ndi_rotate,
    shift as ndi_shift,
)


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """All tunables for the registration pipeline."""

    # --- sub-pixel translation ---
    upsample_factor: int = 100
    """Phase-correlation up-sampling factor (100 → 0.01 px accuracy)."""

    # --- multi-resolution pyramid ---
    use_pyramid: bool = True
    """Estimate global shift with a 2-level coarse-to-fine pyramid.
    Level 1: 4× subsampled for fast coarse estimate.
    Level 2: full-resolution sub-pixel refinement seeded by level 1.
    More robust than single-scale for larger displacements."""

    # --- rotation ---
    estimate_rotation: bool = False
    """If True, estimate small rotation via brute-force NCC search.
    Disabled by default: rotation between repeated OCT B-scans is negligible
    and the brute-force search is expensive (O(n_angles) per frame)."""
    max_rotation_deg: float = 2.0
    """Maximum rotation angle to search (degrees, symmetric ±)."""
    rotation_step_deg: float = 0.05
    """Angular resolution for the search grid."""

    # --- strip-wise (non-rigid) registration ---
    strip_registration: bool = True
    """Split each frame into horizontal strips and register each independently
    to correct for non-rigid axial motion (common in OCT averaging)."""
    n_strips: int = 16
    """Number of horizontal strips.  More strips → finer non-rigid correction."""
    strip_overlap: int = 16
    """Pixels of overlap added to each strip before correlation (wider context
    gives a more stable shift estimate for narrow strips)."""
    mask_signal_frac: float = 0.25
    """Strips whose mean gradient energy falls below this fraction of the
    brightest strip's energy are treated as featureless (vitreous, deep noise)
    and assigned zero residual shift.  Prevents dark strips from producing
    spurious large displacements that corrupt the warp field."""

    # --- iterative reference refinement ---
    n_refine_iters: int = 3
    """Number of register → average → re-register iterations."""

    # --- quality gating ---
    correlation_threshold: float = 0.1
    """Frames with raw phase-correlation peak below this are discarded.
    With z-score normalised images, good OCT frames land around 0.15–0.30;
    artefacts (blinks, gross motion) land near 0.  0.1 is a safe default."""

    # --- averaging ---
    avg_mode: str = "quality_weighted"
    """How to combine the aligned frames.
    'mean'            — simple arithmetic mean of accepted frames.
    'trimmed'         — drop the top/bottom trim_frac fraction per pixel,
                        then mean the remainder.
    'quality_weighted'— weight each frame by its peak correlation score."""
    trim_frac: float = 0.1
    """Fraction of frames to drop from each tail for trimmed mean.
    Ignored for other avg_mode values."""

    # --- post-processing ---
    sharpen: bool = False
    """Apply a mild unsharp mask to the final average.  Useful when the
    registered average still looks slightly soft."""
    sharpen_amount: float = 0.3
    """Unsharp mask strength.  0 = no effect, 1 = strong."""
    sharpen_sigma: float = 1.5
    """Gaussian blur sigma for the unsharp mask kernel (pixels)."""

    # --- output ---
    output_dtype: str = "uint16"
    """TIFF output dtype: 'uint8', 'uint16', or 'float32'."""
    p_lo: float = 0.5
    """Low percentile for robust intensity scaling."""
    p_hi: float = 99.5
    """High percentile for robust intensity scaling."""


# ---------------------------------------------------------------------------
#  Core: upsampled-DFT phase cross-correlation  (translation)
# ---------------------------------------------------------------------------

def _upsampled_cross_correlation(
    ref: np.ndarray,
    mov: np.ndarray,
    upsample_factor: int = 100,
) -> Tuple[float, float, float]:
    """Return (shift_row, shift_col, quality) via Guizar-Sicairos et al. (2008).

    quality is the normalised upsampled-DFT peak for upsample_factor > 1, or
    the pixel-level phase-correlation peak (≈ [0, 1]) for upsample_factor == 1.

    Images are z-score normalised internally so quality values are meaningful
    regardless of input intensity range.
    """
    ref = (ref - ref.mean()) / (ref.std() + 1e-12)
    mov = (mov - mov.mean()) / (mov.std() + 1e-12)

    src_freq = fft2(ref)
    tgt_freq = fft2(mov)
    shape = ref.shape

    cross_power = src_freq * np.conj(tgt_freq)
    eps = np.finfo(cross_power.dtype).eps
    cross_power /= np.abs(cross_power) + eps

    cc_image = np.real(ifft2(cross_power))
    maxima = np.unravel_index(np.argmax(cc_image), shape)
    midpoints = np.array([dim // 2 for dim in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        return float(shifts[0]), float(shifts[1]), float(cc_image[maxima])

    # Sub-pixel refinement via upsampled DFT
    shifts = np.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = int(np.ceil(upsample_factor * 1.5))
    dft_shift = np.fix(upsampled_region_size / 2.0)
    sample_region_offset = dft_shift - shifts * upsample_factor

    cc_up = _upsampled_dft(
        cross_power, upsampled_region_size, upsample_factor, sample_region_offset
    )
    cc_up = np.conj(cc_up)

    maxima_up = np.unravel_index(np.argmax(np.abs(cc_up)), cc_up.shape)
    peak = np.abs(cc_up[maxima_up])
    shifts_up = np.array(maxima_up, dtype=np.float64) - dft_shift
    shifts = shifts + shifts_up / upsample_factor

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
    """Matrix-multiply DFT on a fine grid (Guizar-Sicairos et al.)."""
    nr, nc = data.shape
    row_kern = np.exp(
        (-1j * 2 * np.pi / (nr * upsample_factor))
        * (np.fft.ifftshift(np.arange(nr))[:, None] - np.floor(nr / 2))
        @ (np.arange(upsampled_region_size)[None, :] - axis_offsets[0])
    )
    col_kern = np.exp(
        (-1j * 2 * np.pi / (nc * upsample_factor))
        * (np.fft.ifftshift(np.arange(nc))[:, None] - np.floor(nc / 2))
        @ (np.arange(upsampled_region_size)[None, :] - axis_offsets[1])
    )
    return row_kern.T @ data @ col_kern


# ---------------------------------------------------------------------------
#  Core: gradient representation
# ---------------------------------------------------------------------------

def _gradient_image(img: np.ndarray) -> np.ndarray:
    """Axial-gradient magnitude — emphasises retinal layer edges.

    Retinal layer boundaries produce strong, consistent axial edges that are
    far more reliable registration features than raw speckle intensity, which
    is a random multiplicative process that decorrelates between frames.
    """
    return np.abs(np.gradient(img, axis=0))


# ---------------------------------------------------------------------------
#  Core: coarse-to-fine global shift (image pyramid)
# ---------------------------------------------------------------------------

def _pyramid_shift(
    ref_g: np.ndarray,
    mov_g: np.ndarray,
    upsample_factor: int,
) -> Tuple[float, float]:
    """Two-level coarse-to-fine global translation estimation.

    Level 1 (coarse): 4× subsampled via slicing — fast, no interpolation.
                      Gives ±4 px accuracy at low cost.
    Level 2 (fine):   full-resolution, seeded by the integer coarse shift,
                      then sub-pixel refined with *upsample_factor*.

    The pyramid guards against the full-resolution correlator latching onto a
    sidelobe when large bulk shifts push the true peak off-centre.
    """
    # Coarse: 4× subsample via slicing (no interpolation needed)
    ref_c = ref_g[::4, ::4]
    mov_c = mov_g[::4, ::4]
    dy4, dx4, _ = _upsampled_cross_correlation(ref_c, mov_c, upsample_factor=10)
    dy_int = int(round(dy4 * 4))
    dx_int = int(round(dx4 * 4))

    # Fine: apply integer coarse shift, then find sub-pixel residual
    mov_seeded = ndi_shift(mov_g, (dy_int, dx_int), order=1, mode="reflect")
    ddy, ddx, _ = _upsampled_cross_correlation(
        ref_g, mov_seeded, upsample_factor=upsample_factor
    )
    return float(dy_int + ddy), float(dx_int + ddx)


# ---------------------------------------------------------------------------
#  Core: rotation estimation
# ---------------------------------------------------------------------------

def _estimate_rotation(
    ref: np.ndarray,
    mov: np.ndarray,
    max_deg: float = 2.0,
    step_deg: float = 0.05,
) -> float:
    """Brute-force small-angle rotation search maximising NCC.

    Returns the rotation angle (degrees, counter-clockwise) that best aligns
    *mov* to *ref*.  Only searches ±max_deg in steps of step_deg.
    """
    best_corr = -np.inf
    best_angle = 0.0
    angles = np.arange(-max_deg, max_deg + step_deg / 2, step_deg)

    rh, rw = ref.shape
    mr, mc = rh // 6, rw // 6
    ref_c = ref[mr:rh - mr, mc:rw - mc]
    ref_c = ref_c - ref_c.mean()
    ref_norm = np.linalg.norm(ref_c) + 1e-12

    for angle in angles:
        rot = ndi_rotate(mov, angle, reshape=False, order=3, mode="reflect")
        rot_c = rot[mr:rh - mr, mc:rw - mc]
        rot_c = rot_c - rot_c.mean()
        corr = np.sum(ref_c * rot_c) / (ref_norm * (np.linalg.norm(rot_c) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_angle = angle

    return best_angle


# ---------------------------------------------------------------------------
#  Core: strip-wise non-rigid registration
# ---------------------------------------------------------------------------

def _register_stripwise(
    ref: np.ndarray,
    mov: np.ndarray,
    n_strips: int = 16,
    overlap: int = 16,
    upsample_factor: int = 100,
    mask_signal_frac: float = 0.25,
) -> np.ndarray:
    """Smooth per-row non-rigid registration via gradient-image phase correlation.

    Algorithm
    ---------
    1. Compute per-strip gradient energy from the *reference* to identify which
       strips contain retinal structure.  Strips with energy < mask_signal_frac
       × max_strip_energy receive dy = dx = 0 (no warp) — this prevents
       featureless vitreous / deep-noise strips from polluting the warp field.
    2. For signal-bearing strips, estimate the residual shift from gradient
       images (insensitive to frame-to-frame speckle change).
    3. Outlier-clamp any remaining bad shifts with a 3×MAD filter.
    4. Interpolate strip-centre shifts into a smooth per-row displacement field
       and apply it as a single warp via map_coordinates — no blending seams.
    """
    h, w = ref.shape
    strip_h = max(h // n_strips, 16)
    actual_n = h // strip_h

    ref_g = _gradient_image(ref)
    mov_g = _gradient_image(mov)

    # Per-strip signal energy measured on the core (non-overlapping) zone
    strip_energies = np.array([
        np.mean(ref_g[i * strip_h : min((i + 1) * strip_h, h), :])
        for i in range(actual_n)
    ])
    energy_threshold = mask_signal_frac * (strip_energies.max() + 1e-12)

    centers: list[float] = []
    dy_list: list[float] = []
    dx_list: list[float] = []

    for i in range(actual_n):
        y0 = max(i * strip_h - overlap, 0)
        y1 = min((i + 1) * strip_h + overlap, h)
        centers.append(0.5 * (y0 + y1))

        if strip_energies[i] < energy_threshold:
            # Featureless strip: assign zero residual shift
            dy_list.append(0.0)
            dx_list.append(0.0)
            continue

        dy, dx, _ = _upsampled_cross_correlation(
            ref_g[y0:y1, :], mov_g[y0:y1, :], upsample_factor
        )
        dy_list.append(dy)
        dx_list.append(dx)

    # Secondary outlier clamp: any remaining wild shifts → median
    dy_arr = np.array(dy_list)
    dx_arr = np.array(dx_list)
    for arr in (dy_arr, dx_arr):
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 0.5   # +0.5 avoids zero-MAD collapse
        arr[np.abs(arr - med) > 3.0 * mad] = med

    # Smooth per-row displacement field via linear interpolation
    row_coords = np.arange(h, dtype=np.float64)
    dy_field = np.interp(row_coords, centers, dy_arr)
    dx_field = np.interp(row_coords, centers, dx_arr)

    # Single smooth warp — no blending seams
    row_grid = (row_coords + dy_field)[:, None] * np.ones((1, w))
    col_grid = np.arange(w, dtype=np.float64)[None, :] + dx_field[:, None]

    return map_coordinates(mov, [row_grid, col_grid], order=3, mode="reflect")


# ---------------------------------------------------------------------------
#  Core: robust frame averaging
# ---------------------------------------------------------------------------

def _robust_average(
    aligned: np.ndarray,
    weights: np.ndarray,
    mode: str,
    trim_frac: float = 0.1,
) -> np.ndarray:
    """Combine aligned frames using the chosen averaging strategy.

    Parameters
    ----------
    aligned : (N, H, W) float64 — aligned frame stack.
    weights : (N,) float64     — per-frame quality weights (0 = rejected).
    mode    : str              — 'mean', 'trimmed', or 'quality_weighted'.
    trim_frac : float          — tail fraction for trimmed mean.
    """
    valid = weights > 0
    if valid.sum() == 0:
        return np.mean(aligned, axis=0)

    if mode == "mean":
        return np.mean(aligned[valid], axis=0)

    elif mode == "trimmed":
        frames = aligned[valid]
        n = frames.shape[0]
        n_trim = max(0, int(n * trim_frac))
        if n_trim == 0 or 2 * n_trim >= n:
            return np.mean(frames, axis=0)
        # Per-pixel trimmed mean: sort along frame axis, discard tails
        sorted_f = np.sort(frames, axis=0)
        return np.mean(sorted_f[n_trim : n - n_trim], axis=0)

    else:  # quality_weighted
        w = weights.copy()
        w[~valid] = 0.0
        return np.einsum("fhw,f->hw", aligned, w) / (w.sum() + 1e-12)


# ---------------------------------------------------------------------------
#  High-level pipeline
# ---------------------------------------------------------------------------

def register_and_average(
    stack: np.ndarray,
    cfg: Optional[RegistrationConfig] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register a [N, H, W] float stack and return (averaged_image, weights).

    Parameters
    ----------
    stack   : (N, H, W) float — input B-scan stack (any intensity range).
    cfg     : RegistrationConfig, optional.
    verbose : bool — print per-iteration progress.

    Returns
    -------
    avg     : (H, W) float32 — registered average.
    weights : (N,) float32  — per-frame quality weight (0 = rejected).
    """
    if cfg is None:
        cfg = RegistrationConfig()

    n_frames, h, w = stack.shape
    if verbose:
        print(
            f"[reg] {n_frames} frames  {h}×{w} px  |  "
            f"pyramid={'on' if cfg.use_pyramid else 'off'}  "
            f"strips={cfg.n_strips if cfg.strip_registration else 'off'}  "
            f"rotation={'on' if cfg.estimate_rotation else 'off'}  "
            f"avg={cfg.avg_mode}  "
            f"iters={cfg.n_refine_iters}"
        )

    stack = stack.astype(np.float64, copy=True)

    # Initial reference: pixel-wise median (robust to single-frame outliers)
    reference = np.median(stack, axis=0)

    weights = np.ones(n_frames, dtype=np.float64)
    aligned = np.empty_like(stack)

    for iteration in range(cfg.n_refine_iters):
        if verbose:
            print(f"[reg] === iteration {iteration + 1}/{cfg.n_refine_iters} ===")

        # Pre-compute reference gradient once per iteration (reused for every frame)
        ref_g = _gradient_image(reference)

        for fi in range(n_frames):
            frame = stack[fi]

            # --- Optional rotation correction ---
            angle = 0.0
            if cfg.estimate_rotation:
                angle = _estimate_rotation(
                    reference, frame,
                    max_deg=cfg.max_rotation_deg,
                    step_deg=cfg.rotation_step_deg,
                )
                if abs(angle) > 1e-4:
                    frame = ndi_rotate(
                        frame, angle, reshape=False, order=3, mode="reflect"
                    )

            # --- Global translation ---
            # Use gradient images for shift estimation (speckle-robust).
            # Use raw images at pixel resolution for the quality gate (reliable [0,1]).
            frame_g = _gradient_image(frame)
            if cfg.use_pyramid:
                dy, dx = _pyramid_shift(ref_g, frame_g, cfg.upsample_factor)
            else:
                dy, dx, _ = _upsampled_cross_correlation(
                    ref_g, frame_g, cfg.upsample_factor
                )

            _, _, peak = _upsampled_cross_correlation(reference, frame, upsample_factor=1)

            if peak < cfg.correlation_threshold:
                weights[fi] = 0.0
                aligned[fi] = 0.0
                if verbose:
                    print(f"  frame {fi:3d}:  REJECTED  (corr={peak:.3f})")
                continue

            weights[fi] = peak

            # Apply global shift to the raw (non-gradient) frame
            frame = ndi_shift(frame, (dy, dx), order=3, mode="reflect")

            # --- Strip-wise non-rigid correction ---
            if cfg.strip_registration:
                frame = _register_stripwise(
                    reference, frame,
                    n_strips=cfg.n_strips,
                    overlap=cfg.strip_overlap,
                    upsample_factor=cfg.upsample_factor,
                    mask_signal_frac=cfg.mask_signal_frac,
                )

            aligned[fi] = frame

        # --- Update reference via robust average ---
        w_sum = weights.sum()
        if w_sum < 1e-12:
            print("[reg] WARNING: all frames rejected, returning simple mean")
            return np.mean(stack, axis=0).astype(np.float32), weights.astype(np.float32)

        reference = _robust_average(aligned, weights, cfg.avg_mode, cfg.trim_frac)

        if verbose:
            n_kept = int(np.sum(weights > 0))
            mean_q = float(np.mean(weights[weights > 0])) if n_kept > 0 else 0.0
            print(f"[reg]   kept {n_kept}/{n_frames}  (mean corr = {mean_q:.4f})")

    # --- Optional unsharp-mask sharpening ---
    if cfg.sharpen:
        blurred = gaussian_filter(reference, sigma=cfg.sharpen_sigma)
        reference = reference + cfg.sharpen_amount * (reference - blurred)

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
    import re
    import matplotlib.pyplot as plt

    # ---- Edit these ----
    direc = r"runs\A-Line\1D_npatch=256\predictions_tiff\6mm_1024Aline"
    file = "gt_6mm_1024Aline_s005_g060.tiff"
    output = None       # defaults to <direc>/registered_<file>
    n_frames = None     # number of frames to use; None = all

    cfg = RegistrationConfig(
        upsample_factor=100,        # 100 → 0.01 px accuracy
        use_pyramid=True,           # coarse-to-fine for robust large-shift handling
        estimate_rotation=False,    # rarely needed for same-position B-scans
        strip_registration=True,
        n_strips=16,                # finer non-rigid correction
        strip_overlap=16,
        mask_signal_frac=0.25,      # skip vitreous/noise strips
        n_refine_iters=3,
        correlation_threshold=0.1,
        avg_mode="quality_weighted",
        trim_frac=0.1,
        sharpen=False,              # enable if output still looks soft
        sharpen_amount=0.3,
        sharpen_sigma=1.5,
        output_dtype="uint16",
    )
    # --------------------

    tiff_path = os.path.join(direc, file)
    output_path = output or os.path.join(direc, f"registered_{file}")

    print(f"[reg] loading {tiff_path}")
    stack = load_tiff_stack(tiff_path)
    if n_frames is not None:
        stack = stack[:n_frames]
    print(f"[reg] stack shape: {stack.shape}  dtype: {stack.dtype}")

    avg, weights = register_and_average(stack, cfg, verbose=True)

    n_kept = int(np.sum(weights > 0))
    print(f"[reg] done — {n_kept}/{stack.shape[0]} frames used")

    save_tiff(output_path, avg, dtype=cfg.output_dtype, p_lo=cfg.p_lo, p_hi=cfg.p_hi)

    # ---- Display comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    naive_avg = np.mean(stack, axis=0)

    axes[0].imshow(naive_avg, cmap="gray", aspect="auto")
    axes[0].set_title("Naive mean (no registration)")
    axes[1].imshow(avg, cmap="gray", aspect="auto")
    axes[1].set_title(f"Registered ({cfg.avg_mode}, {n_kept}/{stack.shape[0]} frames)")
    axes[2].imshow(np.abs(avg.astype(np.float64) - naive_avg), cmap="hot", aspect="auto")
    axes[2].set_title("| registered − naive |")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(re.sub(r"\.tiff?$", "_comparison.png", output_path), dpi=150)
    plt.show()
