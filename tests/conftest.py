"""Shared fixtures.

The important one is `synthetic_dataset`: it writes a physically plausible
miniature OCT acquisition (raw uint16 interferograms plus a .CLB calibration
file) to a temp directory, so the whole preprocessing and training path can be
exercised on CI with no instrument data present.

Synthetic frames are modelled as a Gaussian source envelope modulating the
interference of a few discrete scattering layers, with per-A-line random phase
to produce speckle and additive detector noise. That matters: tests run against
white noise would not catch a broken FFT, crop or window.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from octdenoiser.configs.default import FolderSpec

PIXELS = 256
ALINES = 64
N_FRAMES = 6
CROP = (0, PIXELS // 2)

# Layer depths as a fraction of the FFT length, and reflectivity. Kept clear of
# both the DC roll-off and the Nyquist edge so every layer is resolvable.
LAYERS = [(0.10, 1.00), (0.16, 0.60), (0.24, 0.40), (0.34, 0.28)]


def _resampling_lut(pixels: int) -> np.ndarray:
    """Monotonic k-linearisation LUT in [0, 1].

    Shaped to match the real calibration: identity plus a smooth warp with a
    peak deviation from linear of ~0.07, which is what was measured on the
    Maestro3 .CLB and what makes the cubic-spline resample non-trivial.
    """
    x = np.linspace(0.0, 1.0, pixels, dtype=np.float64)
    lut = x + 0.07 * np.sin(np.pi * x)
    lut -= lut[0]
    lut /= lut[-1]
    assert np.all(np.diff(lut) > 0), "synthetic LUT must be strictly monotonic"
    return lut


def _clb_bytes(pixels: int) -> bytes:
    """Mimic the real .CLB layout: 64-byte header, then float32 payload.

    The genuine file carries 2*pixels floats; only the first `pixels` are read
    as the resampling LUT.
    """
    header = bytearray(64)
    header[0:4] = b"2.00"
    header[16:22] = b"999999"
    payload = np.concatenate([_resampling_lut(pixels), np.zeros(pixels)])
    return bytes(header) + payload.astype(np.float32).tobytes()


def _frame(pixels: int, alines: int, rng: np.random.Generator) -> np.ndarray:
    """One synthetic interferogram, uint16, shape (pixels, alines).

    Real spectrometers sample linearly in wavelength, so the fringes are NOT
    periodic in the raw sample index — the .CLB resample is what linearises
    them in k. The fixture reproduces that: fringe phase is generated against
    the inverse LUT so that after `resample_klinear_cubic_operator` the layers
    land at exactly `depth_frac * pixels`. Generating fringes on the uniform
    grid instead would shift every reconstructed peak, increasingly with depth.
    """
    lut = _resampling_lut(pixels)
    x = np.linspace(0.0, 1.0, pixels, dtype=np.float64)
    # Inverse warp: pipeline evaluates raw at lut[i], so pre-compose with lut^-1.
    k = np.interp(x, lut, x)[:, None]

    envelope = np.exp(-0.5 * ((x[:, None] - 0.5) / 0.22) ** 2)

    fringes = np.zeros((pixels, alines))
    for depth_frac, refl in LAYERS:
        # Random phase per A-line decorrelates speckle laterally.
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, alines))
        # Per-A-line reflectivity jitter gives the layer a textured appearance.
        amp = refl * (1.0 + 0.35 * rng.standard_normal((1, alines)))
        fringes += amp * np.cos(2.0 * np.pi * depth_frac * pixels * k + phase)

    signal = envelope * (1.0 + 0.45 * fringes)
    signal += 0.02 * rng.standard_normal((pixels, alines))  # detector noise

    lo, hi = signal.min(), signal.max()
    scaled = (signal - lo) / max(hi - lo, 1e-9)
    return (scaled * 60000.0).astype(np.uint16)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> str:
    """Write a miniature acquisition; return the root folder path.

    Layout mirrors the real one:  <root>/<data_folder>/bscan*.raw  +  <root>/*.CLB
    """
    root = tmp_path_factory.mktemp("oct_synth")
    data_dir = root / "synth_folder"
    data_dir.mkdir()

    (root / "SYNTH_000_FUNDUS.CLB").write_bytes(_clb_bytes(PIXELS))

    rng = np.random.default_rng(1234)
    for i in range(N_FRAMES):
        frame = _frame(PIXELS, ALINES, rng)
        # Real files are Fortran-ordered on disk.
        frame.ravel(order="F").tofile(data_dir / f"bscan{i:06d}.raw")

    return str(root)


@pytest.fixture(scope="session")
def synthetic_spec(synthetic_dataset) -> FolderSpec:
    return FolderSpec(
        root_folder=synthetic_dataset,
        data_folder="synth_folder",
        pixels=PIXELS,
        alines=ALINES,
        crop_depth=CROP,
        window_sigma=0.08,
        gap=0.30,
    )


@pytest.fixture(scope="session")
def synthetic_spec_multilevel(synthetic_dataset) -> FolderSpec:
    return FolderSpec(
        root_folder=synthetic_dataset,
        data_folder="synth_folder",
        pixels=PIXELS,
        alines=ALINES,
        crop_depth=CROP,
        window_sigma=0.05,
        gap=0.60,
        gap_offset=0.015,
        n_sub_windows=2,
        sub_window_spread=0.5,
    )


@pytest.fixture(scope="session")
def n_synthetic_frames() -> int:
    return N_FRAMES


def pytest_configure(config):
    """Keep numeric libraries single-threaded so CI timings are stable."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")


# Silence an unused-import warning while keeping struct available for anyone
# extending _clb_bytes with real header fields.
_ = struct
