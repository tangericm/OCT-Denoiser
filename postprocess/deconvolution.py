"""Axial (column-wise) 1-D deconvolution for OCT images.

Supports Wiener and Richardson-Lucy methods with safeguards against
ringing and noise amplification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────
@dataclass
class DeconvConfig:
    """Parameters for axial deconvolution."""

    method: str = "wiener"        # "wiener" or "richardson_lucy"
    psf_sigma: float = 1.5        # Gaussian PSF sigma in axial pixels
    psf_kernel: Optional[np.ndarray] = None  # explicit 1-D kernel overrides sigma

    # Wiener-specific
    wiener_nsr: float = 0.01      # noise-to-signal ratio regularization

    # Richardson-Lucy-specific
    rl_iterations: int = 15
    rl_tv_lambda: float = 0.002   # TV regularization weight (0 = off)

    # Pre-smoothing (reduces ringing on noisy input)
    pre_smooth_sigma: float = 0.0  # 0 = disabled

    # Post-processing
    clip_range: tuple = (0.0, 1.0)
    non_negative: bool = True
    post_smooth_sigma: float = 0.0  # 0 = disabled


# ── PSF construction ──────────────────────────────────────────────────────
def make_gaussian_psf(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """1-D Gaussian PSF, normalized to unit sum.

    Parameters
    ----------
    sigma : standard deviation in pixels.
    truncate : kernel half-length in multiples of sigma.
    """
    radius = int(np.ceil(sigma * truncate))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


# ── Wiener deconvolution (frequency domain, column-wise) ─────────────────
def _wiener_deconv_columns(
    img: np.ndarray, psf: np.ndarray, nsr: float
) -> np.ndarray:
    """Wiener deconvolution applied independently to each column (A-line).

    Parameters
    ----------
    img : [H, W] float64 image.
    psf : 1-D kernel (length <= H).
    nsr : noise-to-signal power ratio for regularization.
    """
    H, W = img.shape
    # Zero-pad PSF to image height
    h = np.zeros(H, dtype=np.float64)
    offset = len(psf) // 2
    h[:len(psf)] = psf
    # Shift so PSF center is at index 0 (for correct frequency alignment)
    h = np.roll(h, -offset)
    H_freq = np.fft.rfft(h)

    # Wiener filter: H* / (|H|^2 + NSR)
    H_conj = np.conj(H_freq)
    H_power = np.abs(H_freq) ** 2
    W_filter = H_conj / (H_power + nsr)

    # Apply column-wise via 2-D rfft along axis 0
    IMG = np.fft.rfft(img, axis=0)
    # Broadcast: W_filter is [H//2+1], IMG is [H//2+1, W]
    result = np.fft.irfft(IMG * W_filter[:, np.newaxis], n=H, axis=0)
    return result


# ── Richardson-Lucy with optional TV regularization ───────────────────────
def _rl_deconv_columns(
    img: np.ndarray,
    psf: np.ndarray,
    iterations: int,
    tv_lambda: float,
) -> np.ndarray:
    """Richardson-Lucy deconvolution applied column-wise.

    Optionally includes a total-variation-like gradient penalty to
    suppress ringing artifacts.

    Parameters
    ----------
    img : [H, W] float64, non-negative.
    psf : 1-D kernel.
    iterations : max iteration count.
    tv_lambda : TV regularization strength (0 = standard RL).
    """
    H, W = img.shape
    eps = 1e-12
    psf_flip = psf[::-1].copy()

    # Initialize with the observed image
    estimate = img.copy()
    estimate = np.clip(estimate, eps, None)

    for _ in range(iterations):
        # Forward model: convolve estimate with PSF along axis 0
        blurred = _convolve_columns(estimate, psf)
        blurred = np.clip(blurred, eps, None)

        ratio = img / blurred
        correction = _convolve_columns(ratio, psf_flip)

        # TV regularization: penalize axial gradients
        if tv_lambda > 0:
            grad = np.zeros_like(estimate)
            grad[1:-1, :] = (
                2 * estimate[1:-1, :] - estimate[:-2, :] - estimate[2:, :]
            )
            denom = 1.0 + tv_lambda * grad / (estimate + eps)
            denom = np.clip(denom, 0.5, 2.0)  # stability clamp
            correction = correction / denom

        estimate = estimate * correction
        estimate = np.clip(estimate, eps, None)

    return estimate


def _convolve_columns(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve each column of *img* with 1-D *kernel* along axis 0.

    Uses FFT-based convolution for efficiency.
    """
    H, W = img.shape
    k_len = len(kernel)
    # Pad for linear (non-circular) convolution
    n_fft = H + k_len - 1
    K = np.fft.rfft(kernel, n=n_fft)
    IMG = np.fft.rfft(img, n=n_fft, axis=0)
    conv_full = np.fft.irfft(IMG * K[:, np.newaxis], n=n_fft, axis=0)
    # Trim to original size, centered
    start = k_len // 2
    return conv_full[start:start + H, :]


# ── Public API ────────────────────────────────────────────────────────────
def deconvolve_image(
    img: np.ndarray, cfg: Optional[DeconvConfig] = None
) -> np.ndarray:
    """Axial deconvolution of a single [H, W] image.

    Parameters
    ----------
    img : 2-D float32 image, expected in [0, 1].
    cfg : deconvolution settings.

    Returns
    -------
    Deconvolved image, same shape, clipped to cfg.clip_range.
    """
    if cfg is None:
        cfg = DeconvConfig()

    work = img.astype(np.float64)

    # Pre-smoothing
    if cfg.pre_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        work = gaussian_filter1d(work, cfg.pre_smooth_sigma, axis=0)

    # Build PSF
    if cfg.psf_kernel is not None:
        psf = np.asarray(cfg.psf_kernel, dtype=np.float64).ravel()
        psf /= psf.sum()
    else:
        psf = make_gaussian_psf(cfg.psf_sigma)

    # Deconvolve
    if cfg.method == "wiener":
        result = _wiener_deconv_columns(work, psf, cfg.wiener_nsr)
    elif cfg.method == "richardson_lucy":
        if cfg.non_negative:
            work = np.clip(work, 0, None)
        result = _rl_deconv_columns(
            work, psf, cfg.rl_iterations, cfg.rl_tv_lambda
        )
    else:
        raise ValueError(f"Unknown deconvolution method: {cfg.method}")

    # Non-negativity
    if cfg.non_negative:
        result = np.clip(result, 0, None)

    # Clip to valid range
    lo, hi = cfg.clip_range
    result = np.clip(result, lo, hi)

    # Post-smoothing
    if cfg.post_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        result = gaussian_filter1d(result, cfg.post_smooth_sigma, axis=0)
        result = np.clip(result, lo, hi)

    return result.astype(np.float32)


def deconvolve_stack(
    stack: np.ndarray, cfg: Optional[DeconvConfig] = None
) -> np.ndarray:
    """Axial deconvolution on every frame of an [N, H, W] stack.

    Parameters
    ----------
    stack : float32 image stack.
    cfg : deconvolution settings.

    Returns
    -------
    Deconvolved stack, same shape and dtype.
    """
    if cfg is None:
        cfg = DeconvConfig()

    N = stack.shape[0]
    out = np.empty_like(stack)
    for i in range(N):
        out[i] = deconvolve_image(stack[i], cfg)
    logger.info(
        "Deconvolved %d frames  method=%s  psf_sigma=%.2f",
        N, cfg.method, cfg.psf_sigma,
    )
    return out
