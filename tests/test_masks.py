"""Complementary spectral mask tests.

The load-bearing test here is `test_random_mask_preserves_full_band_resolution`:
the whole justification for random full-range masks over contiguous sub-bands is
that they keep the full-bandwidth PSF. That claim gets measured, not assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from octdenoiser.physics.masks import (
    axial_psf,
    make_complementary_masks,
    make_mask_partition,
    masks_are_disjoint,
    psf_main_lobe_width,
    sidelobe_ratio,
)

PIXELS = 2048


# --------------------------------------------------------------------------
# Complementarity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["binary", "smooth"])
def test_amplitude_complementary_sums_to_one(kind):
    m1, m2 = make_complementary_masks(PIXELS, kind=kind, seed=0)
    assert np.allclose(m1 + m2, 1.0, atol=1e-6)


def test_power_complementary_squares_sum_to_one():
    m1, m2 = make_complementary_masks(PIXELS, kind="power", seed=0)
    assert np.allclose(m1**2 + m2**2, 1.0, atol=1e-5)


def test_binary_masks_are_disjoint_and_binary():
    m1, m2 = make_complementary_masks(PIXELS, kind="binary", seed=0)
    assert set(np.unique(m1)).issubset({0.0, 1.0})
    assert masks_are_disjoint([m1, m2])
    assert not np.any((m1 > 0) & (m2 > 0)), "no sample may appear in both views"


@pytest.mark.parametrize("duty", [0.25, 0.5, 0.75])
def test_binary_duty_is_respected(duty):
    m1, _ = make_complementary_masks(PIXELS, kind="binary", duty=duty, seed=1)
    assert m1.mean() == pytest.approx(duty, abs=1e-3)


def test_smooth_masks_overlap_by_design():
    """smooth trades disjointness for lower sidelobes — document that."""
    m1, m2 = make_complementary_masks(PIXELS, kind="smooth", seed=0)
    assert not masks_are_disjoint([m1, m2])


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_seed_is_deterministic():
    a1, a2 = make_complementary_masks(PIXELS, seed=42)
    b1, b2 = make_complementary_masks(PIXELS, seed=42)
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2)


def test_different_seeds_give_different_masks():
    """Masks are redrawn per training sample; they must actually differ."""
    a1, _ = make_complementary_masks(PIXELS, seed=1)
    b1, _ = make_complementary_masks(PIXELS, seed=2)
    assert not np.array_equal(a1, b1)
    # Two independent 50% masks agree on ~50% of samples by chance.
    assert 0.4 < float((a1 == b1).mean()) < 0.6


# --------------------------------------------------------------------------
# Resolution — the central claim
# --------------------------------------------------------------------------
def test_random_mask_preserves_full_band_resolution():
    """A random full-range mask must keep the full-bandwidth main-lobe width.

    This is what separates the method from the existing bandgap approach: no
    resolution loss, so no entangled deconvolution task.
    """
    full = np.ones(PIXELS, dtype=np.float32)
    m1, m2 = make_complementary_masks(PIXELS, kind="binary", seed=3)

    w_full = psf_main_lobe_width(full)
    w_m1 = psf_main_lobe_width(m1)
    w_m2 = psf_main_lobe_width(m2)

    assert np.isfinite(w_full) and np.isfinite(w_m1)
    assert w_m1 == pytest.approx(w_full, rel=0.25), (
        f"masked PSF main lobe {w_m1:.2f} vs full-band {w_full:.2f}"
    )
    assert w_m2 == pytest.approx(w_full, rel=0.25)


def test_contiguous_subband_loses_resolution():
    """The contrast case: a contiguous half-band roughly doubles the PSF width.

    This is the defect the existing Gaussian bandgap method carries and the
    reason it needed a PSF metric defensively.
    """
    full = np.ones(PIXELS, dtype=np.float32)
    half = np.zeros(PIXELS, dtype=np.float32)
    half[: PIXELS // 2] = 1.0  # contiguous half of the k-range

    w_full = psf_main_lobe_width(full)
    w_half = psf_main_lobe_width(half)
    w_random = psf_main_lobe_width(make_complementary_masks(PIXELS, seed=5)[0])

    assert w_half > 1.6 * w_full, f"expected a broader lobe, got {w_half:.2f} vs {w_full:.2f}"
    assert w_random < w_half, (
        f"random full-range mask ({w_random:.2f}) must beat a contiguous "
        f"sub-band ({w_half:.2f}) on resolution"
    )


def test_random_mask_pays_for_resolution_with_sidelobes():
    """The known cost: randomisation scatters energy into a pedestal.

    Recorded so the tradeoff is visible if the method underperforms.
    """
    full = np.ones(PIXELS, dtype=np.float32)
    binary, _ = make_complementary_masks(PIXELS, kind="binary", seed=7)
    smooth, _ = make_complementary_masks(PIXELS, kind="smooth", seed=7)

    assert sidelobe_ratio(binary) > sidelobe_ratio(full)
    assert sidelobe_ratio(smooth) < sidelobe_ratio(binary), (
        "smooth masks exist precisely to lower the pedestal"
    )


def test_periodic_comb_produces_ghosts_that_random_avoids():
    """Justifies randomising: a periodic comb replicates the image in depth."""
    comb = np.zeros(PIXELS, dtype=np.float32)
    comb[::2] = 1.0  # period-2 sampling -> a replica at half the depth range
    psf_comb = np.fft.fftshift(axial_psf(comb))
    psf_rand = np.fft.fftshift(axial_psf(make_complementary_masks(PIXELS, seed=9)[0]))

    def peak_excluding_centre(p):
        c = int(np.argmax(p))
        q = p.copy()
        q[max(0, c - 16): c + 17] = 0.0
        return float(q.max() / p[c])

    assert peak_excluding_centre(psf_comb) > 0.5, "comb should show a strong ghost"
    assert peak_excluding_centre(psf_rand) < 0.1, "random mask should have no discrete ghost"


# --------------------------------------------------------------------------
# Partitions — needed for held-out-mask validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_parts", [2, 3, 4, 8])
def test_partition_is_disjoint_and_complete(n_parts):
    masks = make_mask_partition(PIXELS, n_parts, seed=0)
    assert len(masks) == n_parts
    assert masks_are_disjoint(masks)
    assert np.allclose(np.sum(masks, axis=0), 1.0), "partition must cover every sample"


def test_partition_shares_photons_evenly():
    masks = make_mask_partition(PIXELS, 4, seed=0)
    for m in masks:
        assert m.mean() == pytest.approx(0.25, abs=0.01)


def test_three_way_partition_supports_held_out_validation():
    """m1 trains, m2 is the target, m3 is the untouched validation view."""
    m1, m2, m3 = make_mask_partition(PIXELS, 3, seed=11)
    assert masks_are_disjoint([m1, m2, m3])
    for a, b in ((m1, m2), (m1, m3), (m2, m3)):
        assert not np.any((a > 0) & (b > 0))


def test_partition_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_parts must be >= 2"):
        make_mask_partition(PIXELS, 1)
    with pytest.raises(ValueError, match="must be >= n_parts"):
        make_mask_partition(4, 8)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_duty_rejected(bad):
    with pytest.raises(ValueError, match="duty"):
        make_complementary_masks(PIXELS, duty=bad)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown mask kind"):
        make_complementary_masks(PIXELS, kind="gaussian")  # type: ignore[arg-type]
