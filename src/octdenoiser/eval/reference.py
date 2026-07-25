"""Registration and averaging of repeated B-scans into a near-clean reference.

Purpose
-------
The four Maestro2 line-scan acquisitions each hold 50 repeats at a single
position. Registered and averaged they give the only near-clean reference on
real retina that this dataset admits. That reference is NOT a training target --
four line positions is far too narrow for that -- it exists to demonstrate that
the held-out-mask self-validation metric ranks models the same way a clean
reference does. See eval/selfval.py.

Why registration is mandatory
-----------------------------
Motion is real and it is not monotonic in time. Measured linear-magnitude
correlation against frame 0: 0.975 at frame 1, 0.972 at frame 2, then 0.700 at
frame 4, recovering to 0.863 at frame 8. Frame 4 is a motion event, not drift.
Averaging without registration would blur the reference by exactly the
structure the reference is supposed to resolve.

That 0.975 baseline also carries the finding that motivates the whole method:
repeats of the same tissue barely decorrelate speckle. Averaging suppresses
detector noise well, and speckle only to the extent motion decorrelates it.

Motion model
------------
Repeated B-scans at one position are dominated by axial (depth) bulk motion,
with smaller lateral shift. Rigid sub-pixel translation captures most of it; an
optional per-A-line axial refinement handles the residual tilt and shear left by
eye motion during the scan. Rotation is omitted deliberately -- at a fixed line
position it is not a physically meaningful degree of freedom, and fitting it
invites the optimiser to absorb speckle.

Domain conventions
------------------
Registration runs on LOG-domain images, where layer structure dominates the
correlation. Averaging runs on LINEAR intensity, which is the correct estimator
of mean reflectivity and matches the convention already used by
data/avg_targets.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import shift as ndshift


@dataclass
class RegistrationResult:
    """Per-frame diagnostics from registering a stack."""

    shifts: np.ndarray                       # [N, 2] (dz, dx), sub-pixel
    correlations: np.ndarray                 # [N] post-registration vs reference
    correlations_before: np.ndarray          # [N] pre-registration vs reference
    kept: np.ndarray                         # [N] bool, survived outlier rejection
    reference_index: int = 0
    per_aline: np.ndarray | None = None      # [N, alines] axial shift, if enabled
    notes: list[str] = field(default_factory=list)

    @property
    def n_kept(self) -> int:
        return int(self.kept.sum())

    def improvement(self) -> float:
        """Mean correlation gain over the frames that were kept."""
        k = self.kept
        if not k.any():
            return float("nan")
        return float(self.correlations[k].mean() - self.correlations_before[k].mean())

    def summary(self) -> str:
        k = self.kept
        return (
            f"registered {self.n_kept}/{len(self.kept)} frames "
            f"(reference index {self.reference_index})\n"
            f"  correlation before : {self.correlations_before[k].mean():.4f} "
            f"[{self.correlations_before[k].min():.4f}, {self.correlations_before[k].max():.4f}]\n"
            f"  correlation after  : {self.correlations[k].mean():.4f} "
            f"[{self.correlations[k].min():.4f}, {self.correlations[k].max():.4f}]\n"
            f"  mean |shift|       : dz={np.abs(self.shifts[k, 0]).mean():.3f} px, "
            f"dx={np.abs(self.shifts[k, 1]).mean():.3f} px\n"
            f"  rejected           : {(~k).sum()} frame(s)"
        )


def _normalise(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    a = a - a.mean()
    s = a.std()
    return a / s if s > 0 else a


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two images."""
    x, y = _normalise(a).ravel(), _normalise(b).ravel()
    n = x.size
    return float(x @ y / n) if n else float("nan")


def _parabolic_offset(y_minus: float, y_zero: float, y_plus: float) -> float:
    """Sub-pixel peak offset from three samples around a correlation maximum.

    Standard three-point parabolic fit. Accurate to well under 0.1 px for a
    peak that is not pathologically flat, which is ample here: bulk axial motion
    between adjacent OCT frames is a few pixels at most.
    """
    denom = y_minus - 2.0 * y_zero + y_plus
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_minus - y_plus) / denom, -0.5, 0.5))


def phase_correlation_shift(
    ref: np.ndarray, mov: np.ndarray, *, max_shift: int | None = None
) -> tuple[float, float]:
    """Sub-pixel (dz, dx) translation carrying `mov` onto `ref`.

    Phase correlation rather than plain cross-correlation: normalising by the
    magnitude makes the peak sharp and insensitive to the strong low-frequency
    brightness gradient an OCT B-scan carries down its depth axis.
    """
    if ref.shape != mov.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {mov.shape}")

    a = _normalise(ref)
    b = _normalise(mov)

    # Hann window suppresses wrap-around edge artefacts from the circular FFT.
    wz = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    win = wz * wx

    fa = sfft.fft2(a * win)
    fb = sfft.fft2(b * win)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    corr = np.real(sfft.ifft2(cross / np.where(mag > 1e-12, mag, 1e-12)))

    if max_shift is not None:
        # Restrict the peak search to the plausible motion range so a spurious
        # far peak cannot win. Shifts wrap, so mask both ends of each axis.
        #
        # The mask is applied to a SEPARATE array used only for argmax. Masking
        # `corr` itself with -inf would poison the parabolic fit whenever the
        # peak lands on the mask boundary: -inf - 2*y + -inf is -inf, and the
        # resulting (-inf - -inf)/-inf is NaN.
        m = int(max_shift)
        keep = np.zeros_like(corr, dtype=bool)
        keep[: m + 1, : m + 1] = True
        keep[: m + 1, -m:] = True
        keep[-m:, : m + 1] = True
        keep[-m:, -m:] = True
        search = np.where(keep, corr, -np.inf)
    else:
        search = corr

    pz, px = np.unravel_index(int(np.argmax(search)), search.shape)
    nz, nx = corr.shape

    sub_z = _parabolic_offset(
        corr[(pz - 1) % nz, px], corr[pz, px], corr[(pz + 1) % nz, px]
    )
    sub_x = _parabolic_offset(
        corr[pz, (px - 1) % nx], corr[pz, px], corr[pz, (px + 1) % nx]
    )

    dz = pz + sub_z
    dx = px + sub_x
    # Unwrap to signed shifts.
    if dz > nz // 2:
        dz -= nz
    if dx > nx // 2:
        dx -= nx
    return float(dz), float(dx)


def per_aline_axial_shift(
    ref: np.ndarray, mov: np.ndarray, *, group: int = 32, max_shift: int = 8
) -> np.ndarray:
    """Residual axial shift per group of A-lines, linearly interpolated.

    Eye motion during a scan leaves tilt and shear that a single global
    translation cannot remove. Grouping A-lines keeps the estimate stable —
    a single A-line is far too noisy to correlate reliably.
    """
    if ref.shape != mov.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {mov.shape}")
    h, w = ref.shape
    centres, shifts = [], []

    for x0 in range(0, w, group):
        x1 = min(x0 + group, w)
        a = _normalise(ref[:, x0:x1].mean(axis=1))
        b = _normalise(mov[:, x0:x1].mean(axis=1))
        c = np.correlate(a, b, mode="full")
        lags = np.arange(-h + 1, h)
        keep = np.abs(lags) <= max_shift
        c, lags = c[keep], lags[keep]
        k = int(np.argmax(c))
        sub = (
            _parabolic_offset(c[k - 1], c[k], c[k + 1])
            if 0 < k < len(c) - 1
            else 0.0
        )
        centres.append(0.5 * (x0 + x1 - 1))
        shifts.append(float(lags[k] + sub))

    return np.interp(np.arange(w), np.asarray(centres), np.asarray(shifts))


def _apply_shift(img: np.ndarray, dz: float, dx: float, order: int = 3) -> np.ndarray:
    return ndshift(img, (dz, dx), order=order, mode="nearest")


def _apply_per_aline(img: np.ndarray, shifts: np.ndarray, order: int = 3) -> np.ndarray:
    out = np.empty_like(img)
    for x in range(img.shape[1]):
        out[:, x] = ndshift(img[:, x], shifts[x], order=order, mode="nearest")
    return out


def register_stack(
    frames: np.ndarray,
    *,
    reference_index: int = 0,
    roi: tuple[int, int] | None = None,
    max_shift: int = 64,
    min_correlation: float | None = None,
    refine_per_aline: bool = False,
    aline_group: int = 32,
) -> tuple[np.ndarray, RegistrationResult]:
    """Register a stack of log-domain B-scans onto one reference frame.

    `roi` restricts the depth rows used to ESTIMATE the shift while the shift is
    applied to the full frame. The prior MATLAB work cropped rows 130:600 for
    exactly this reason: correlating over the vitreous and the DC roll-off
    dilutes the tissue signal that should be driving alignment.

    `min_correlation` drops frames that remain poorly aligned. Frame 4 of the
    Maestro2 stack sits at 0.700 against 0.975 for its neighbours, and a frame
    that registration cannot rescue must not enter the average.
    """
    if frames.ndim != 3:
        raise ValueError(f"expected [N, H, W], got {frames.shape}")
    n = frames.shape[0]
    if not 0 <= reference_index < n:
        raise ValueError(f"reference_index {reference_index} out of range for {n} frames")

    def crop(a: np.ndarray) -> np.ndarray:
        return a if roi is None else a[roi[0]:roi[1], :]

    ref_full = frames[reference_index].astype(np.float64)
    ref = crop(ref_full)

    registered = np.empty_like(frames, dtype=np.float32)
    shifts = np.zeros((n, 2))
    corr_before = np.zeros(n)
    corr_after = np.zeros(n)
    per_aline = np.zeros((n, frames.shape[2])) if refine_per_aline else None

    for i in range(n):
        mov = frames[i].astype(np.float64)
        corr_before[i] = correlation(ref, crop(mov))

        dz, dx = phase_correlation_shift(ref, crop(mov), max_shift=max_shift)
        shifts[i] = (dz, dx)
        out = _apply_shift(mov, dz, dx)

        if refine_per_aline:
            resid = per_aline_axial_shift(ref, crop(out), group=aline_group)
            per_aline[i] = resid  # type: ignore[index]
            out = _apply_per_aline(out, resid)

        registered[i] = out.astype(np.float32)
        corr_after[i] = correlation(ref, crop(out))

    kept = np.ones(n, dtype=bool)
    notes: list[str] = []
    if min_correlation is not None:
        kept = corr_after >= min_correlation
        dropped = np.flatnonzero(~kept)
        if dropped.size:
            notes.append(
                f"dropped frames {dropped.tolist()} below correlation {min_correlation}"
            )
        # The reference correlates 1.0 with itself and so always survives, which
        # means "nothing survived" is unreachable. The condition that actually
        # matters is having too few frames left to average.
        if kept.sum() < 2:
            others = np.delete(corr_after, reference_index)
            raise RuntimeError(
                f"only the reference frame cleared min_correlation={min_correlation}; "
                f"averaging needs at least two. Best non-reference correlation was "
                f"{others.max():.4f}. Either the stack is dominated by motion or the "
                f"threshold is too strict."
            )

    return registered, RegistrationResult(
        shifts=shifts,
        correlations=corr_after,
        correlations_before=corr_before,
        kept=kept,
        reference_index=reference_index,
        per_aline=per_aline,
        notes=notes,
    )


def average_linear(
    linear_frames: np.ndarray, kept: np.ndarray | None = None
) -> np.ndarray:
    """Average registered frames in LINEAR intensity.

    Linear rather than log because the mean of linear intensity is the unbiased
    estimator of mean reflectivity; averaging log values computes a geometric
    mean and biases dark regions. Matches data/avg_targets.py.
    """
    if linear_frames.ndim != 3:
        raise ValueError(f"expected [N, H, W], got {linear_frames.shape}")
    sel = linear_frames if kept is None else linear_frames[kept]
    if sel.shape[0] == 0:
        raise ValueError("no frames selected for averaging")
    return sel.mean(axis=0)


def speckle_contrast(img: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> float:
    """std/mean over a region — the standard speckle metric.

    Fully developed speckle sits near 1.0; averaging N decorrelated realisations
    drives it toward 1/sqrt(N). Comparing this before and after averaging shows
    how much speckle the motion in a stack actually decorrelated, which on this
    data is the quantity in question.
    """
    a = img if roi is None else img[roi[0]:roi[1], roi[2]:roi[3]]
    m = float(np.mean(a))
    return float(np.std(a) / m) if m > 0 else float("nan")
