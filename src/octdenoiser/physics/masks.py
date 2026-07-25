"""Complementary spectral masks for self-supervised OCT denoising.

STATUS: measured and NOT recommended as the supervision scheme. See below.

The idea
--------
The existing bandgap method splits the spectrum into two Gaussian sub-bands and
regresses them onto the FULL-band reconstruction. Measured defects:

1. The target contains the input (full band contains both sub-bands), so target
   noise is correlated with input noise. Confirmed: speckle correlation between
   a sub-band input and its full-band target is +0.138, against +0.003 for a
   complementary view -- roughly 40x worse. This is bias, not variance, and it
   is the dominant defect.
2. A sub-band has narrower k-support and therefore a broader axial PSF.
   Measured at 2.43x the full-bandwidth width.
3. Gaussian windows overlap in the tails -- VOID at the tuned settings.
   Measured support overlap at sigma=0.05, gap=0.60 is exactly 0.0000, so the
   two windows are already disjoint and this was never a real defect.

A complementary mask pair over the k-linearised spectrum:

    m1 : random ~50% duty spread across the FULL k-range
    m2 = 1 - m1

MEASURED OUTCOME — the masks lose
---------------------------------
Measured on real Maestro2 data with the laterally-smooth component removed, so
speckle is isolated rather than swamped by shared anatomy:

    construction                 speckle corr   axial PSF   signal mismatch
    repeat frames                     +0.008        1.00x       ~0
    contiguous Gaussian sub-bands     +0.003        2.43x       0.269
    complementary random masks        +0.257        1.01x       0.058

The masks were proposed expecting disjoint k-support to decorrelate speckle
better than Gaussian windows. It does the opposite, by 75x. Speckle decorrelates
because two views sample DIFFERENT spectral regions; contiguous sub-bands do
exactly that, whereas interleaved masks spread across the SAME k-range. That is
precisely the property which preserves the PSF, so decorrelation and resolution
are in direct tension and cannot both be had from one frame.

What the masks do win is signal consistency: 0.058 versus 0.269 relative
mismatch between the two views' mean depth profiles, with a systematic
depth-dependent slope ~100x smaller. Contiguous sub-bands sit at different k and
scattering is wavelength-dependent, so their expected signals genuinely differ —
a Noise2Noise bias term, not noise.

Retained because `make_mask_partition` and the PSF utilities serve the
held-out-mask validation metric in eval/selfval.py, and because the comparison
above belongs in the ablation table.

Why random rather than a periodic comb
--------------------------------------
Periodic sampling with spacing dk replicates the image at depth spacing 2*pi/dk,
producing discrete ghost artifacts. Randomising scatters that energy into a
noise-like pedestal instead.

`smooth` and `power` variants were included as lower-sidelobe fallbacks but
retain +0.63 and +0.87 reconstruction correlation on synthetic spectra, so they
are diagnostics only.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

MaskKind = Literal["binary", "smooth", "power"]


def _smooth_random_field(pixels: int, rng: np.random.Generator, correlation_px: float) -> np.ndarray:
    """Random field in [0,1], smooth on a scale of `correlation_px` samples.

    Built by low-pass filtering white noise in the Fourier domain, which keeps
    the field strictly band-limited so the resulting mask has less high-frequency
    content and therefore lower PSF sidelobes than a binary mask.
    """
    white = rng.standard_normal(pixels)
    freqs = np.fft.rfftfreq(pixels, d=1.0)
    cutoff = 1.0 / max(correlation_px, 1.0)
    spectrum = np.fft.rfft(white) * np.exp(-0.5 * (freqs / cutoff) ** 2)
    field = np.fft.irfft(spectrum, n=pixels)

    lo, hi = field.min(), field.max()
    if hi - lo < 1e-12:
        return np.full(pixels, 0.5)
    return (field - lo) / (hi - lo)


def make_complementary_masks(
    pixels: int,
    *,
    kind: MaskKind = "binary",
    duty: float = 0.5,
    seed: int | None = None,
    correlation_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a complementary mask pair (m1, m2) over `pixels` spectral samples.

    kind:
      "binary" — disjoint 0/1 masks, m1 + m2 == 1. Strongest decorrelation,
                 highest sidelobe pedestal.
      "smooth" — band-limited random field f and 1-f. Still sums to 1, lower
                 sidelobes, weaker decorrelation because supports now overlap.
      "power"  — sqrt(f) and sqrt(1-f), so m1^2 + m2^2 == 1. Preserves total
                 POWER rather than amplitude, which keeps each view's intensity
                 statistics closer to the full-band reconstruction.

    duty: fraction of spectral samples assigned to m1 (binary only). 0.5 splits
          photons evenly; deviating trades SNR between the two views.
    """
    if pixels <= 0:
        raise ValueError(f"pixels must be positive, got {pixels}")
    if not 0.0 < duty < 1.0:
        raise ValueError(f"duty must be in (0, 1), got {duty}")

    rng = np.random.default_rng(seed)

    if kind == "binary":
        # Exact count rather than per-sample Bernoulli: fixes the photon split
        # so it does not fluctuate between training samples.
        n_on = int(round(duty * pixels))
        n_on = min(max(n_on, 1), pixels - 1)
        m1 = np.zeros(pixels, dtype=np.float32)
        m1[rng.permutation(pixels)[:n_on]] = 1.0
        return m1, (1.0 - m1).astype(np.float32)

    field = _smooth_random_field(pixels, rng, correlation_px)

    if kind == "smooth":
        m1 = field.astype(np.float32)
        return m1, (1.0 - field).astype(np.float32)

    if kind == "power":
        m1 = np.sqrt(field).astype(np.float32)
        m2 = np.sqrt(1.0 - field).astype(np.float32)
        return m1, m2

    raise ValueError(f"unknown mask kind {kind!r}; expected 'binary', 'smooth' or 'power'")


def make_mask_partition(
    pixels: int, n_parts: int, *, seed: int | None = None
) -> list[np.ndarray]:
    """Partition the k-axis into `n_parts` mutually disjoint binary masks.

    Needed for held-out-mask validation, which requires a THIRD mask m3 whose
    noise is independent of both the training input and target, and for the
    multi-view input ablation where several disjoint views are stacked as
    channels while a further disjoint view serves as the target.
    """
    if n_parts < 2:
        raise ValueError(f"n_parts must be >= 2, got {n_parts}")
    if pixels < n_parts:
        raise ValueError(f"pixels ({pixels}) must be >= n_parts ({n_parts})")

    rng = np.random.default_rng(seed)
    assignment = rng.permutation(pixels) % n_parts
    return [(assignment == i).astype(np.float32) for i in range(n_parts)]


def masks_are_disjoint(masks: list[np.ndarray], tol: float = 1e-6) -> bool:
    """True if no spectral sample carries weight in more than one mask."""
    stack = np.stack(masks)
    return bool(np.all((stack > tol).sum(axis=0) <= 1))


def axial_psf(mask: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """Axial PSF magnitude implied by a spectral mask.

    The reconstruction is an IFFT along k, so the mask's own IFFT magnitude is
    the impulse response. Used to verify the central claim that a full-range
    random mask preserves the full-bandwidth main-lobe width.
    """
    n = n_fft or mask.size
    return np.abs(np.fft.ifft(mask.astype(np.float64), n=n))


def psf_main_lobe_width(mask: np.ndarray, n_fft: int | None = None) -> float:
    """Full width at half maximum of the PSF main lobe, in depth samples.

    Measured by walking outward from the peak to the first half-maximum
    crossing on each side with sub-pixel interpolation, so a sidelobe pedestal
    cannot inflate the result.
    """
    psf = np.fft.fftshift(axial_psf(mask, n_fft))
    pk_i = int(np.argmax(psf))
    half = psf[pk_i] / 2.0

    def cross(indices) -> float | None:
        prev = pk_i
        for i in indices:
            if psf[i] < half:
                y0, y1 = psf[prev], psf[i]
                t = (y0 - half) / (y0 - y1) if y0 != y1 else 0.0
                return prev + t * (i - prev)
            prev = i
        return None

    left = cross(range(pk_i - 1, -1, -1))
    right = cross(range(pk_i + 1, psf.size))
    if left is None or right is None:
        return float("nan")
    return float(right - left)


def sidelobe_ratio(mask: np.ndarray, main_lobe_halfwidth: int = 8, n_fft: int | None = None) -> float:
    """RMS sidelobe energy relative to the peak, excluding the main lobe.

    This is the cost of randomisation: it quantifies the noise-like pedestal a
    random mask introduces, and is the number to watch if the method
    underperforms.
    """
    psf = np.fft.fftshift(axial_psf(mask, n_fft))
    pk_i = int(np.argmax(psf))
    peak = psf[pk_i]
    keep = np.ones(psf.size, dtype=bool)
    lo = max(0, pk_i - main_lobe_halfwidth)
    hi = min(psf.size, pk_i + main_lobe_halfwidth + 1)
    keep[lo:hi] = False
    if not keep.any() or peak <= 0:
        return float("nan")
    return float(np.sqrt(np.mean(psf[keep] ** 2)) / peak)
