"""Complementary spectral masks for self-supervised OCT denoising.

The idea
--------
The existing bandgap method splits the spectrum into two Gaussian sub-bands and
regresses them onto the FULL-band reconstruction. That breaks the Noise2Noise
conditions three ways:

1. The target contains the input (full band contains both sub-bands), so target
   noise is correlated with input noise. This is bias, not variance, and it is
   the dominant defect.
2. A sub-band has narrower k-support and therefore a broader axial PSF, so the
   network is asked to denoise AND extrapolate bandwidth at once.
3. Gaussian windows overlap in the tails, so the two views are never disjoint.

A complementary mask pair over the k-linearised spectrum fixes all three:

    m1 : random ~50% duty spread across the FULL k-range
    m2 = 1 - m1

  * detector noise is independent across k, and disjoint masks therefore give
    independent noise in the two reconstructions;
  * speckle correlation scales with k-support overlap, so disjoint supports
    decorrelate speckle by construction, without relying on motion (measured
    repeat-frame correlation on this data is 0.975, i.e. repeats do NOT
    decorrelate speckle);
  * total support still spans the full k-range, so the PSF main-lobe width is
    the full-bandwidth width — no resolution mismatch;
  * the target cannot contain the input.

Why random rather than a periodic comb
--------------------------------------
Periodic sampling with spacing dk replicates the image at depth spacing 2*pi/dk,
producing discrete ghost artifacts. Randomising the mask scatters that energy
into a noise-like pedestal instead. Crucially the pedestal differs between m1
and m2, so under Noise2Noise it is treated as noise to suppress rather than
signal to reproduce.

The cost is that pedestal, which raises the effective noise floor. `smooth` and
`power` variants trade decorrelation strength for lower sidelobe energy and
exist as fallbacks if the binary pedestal proves too expensive.
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
