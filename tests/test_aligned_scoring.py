"""Predictions must be scored against a reference in THEIR OWN coordinates.

`build_reference` used to compute registration shifts, average with them, then
throw them away -- so every frame but frame 0 was scored against a displaced
average. The bug was invisible in a ranking-shaped way: it is not a constant
offset that cancels, because misalignment costs a SHARP output more than a
blurred one, which is the direction PSNR/SSIM are already biased in.

These tests pin both halves: that `aligned_to` undoes the shift, and that
skipping it produces exactly the sharp-output penalty that motivated the fix.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift

from octdenoiser.experiments.run_fair_eval import Reference


def _structured_image(seed: int = 0, h: int = 96, w: int = 96) -> np.ndarray:
    """Smooth anatomy plus fine detail -- both are needed to show the bias."""
    rng = np.random.RandomState(seed)
    zz, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    anatomy = np.sin(zz / 11.0) * np.cos(xx / 13.0)
    detail = gaussian_filter(rng.randn(h, w), 0.8)
    return anatomy + 0.5 * detail


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# --------------------------------------------------------------------------
# aligned_to
# --------------------------------------------------------------------------
def test_zero_shift_returns_the_reference_untouched():
    """The identity fast path must not resample -- interpolation is not free."""
    img = _structured_image()
    ref = Reference(image=img, shifts=np.zeros((4, 2)))
    out = ref.aligned_to(2)
    assert out is img, "zero shift must short-circuit, not round-trip through ndshift"


def test_frame_index_past_the_shift_table_falls_back_to_the_reference():
    ref = Reference(image=_structured_image(), shifts=np.zeros((3, 2)))
    assert ref.aligned_to(99) is ref.image


def _frame_at(truth: np.ndarray, dz: float, dx: float) -> np.ndarray:
    """The frame whose registration result is `shifts[i] = (dz, dx)`.

    The convention is that applying `shifts[i]` CARRIES frame i onto the
    reference, i.e. ndshift(frame_i, shifts[i]) == reference. So frame i is the
    reference displaced by the negated shift.
    """
    return ndshift(truth, (-dz, -dx), order=3, mode="nearest")


@pytest.mark.parametrize("dz,dx", [(3.0, 0.0), (0.0, -4.0), (2.0, -3.0), (5.5, 2.5)])
def test_aligned_to_undoes_the_registration_shift(dz, dx):
    """`shifts[i]` carries frame i onto frame 0, so the reference must move by -shift."""
    truth = _structured_image()
    frame_i = _frame_at(truth, dz, dx)
    ref = Reference(image=truth, shifts=np.array([[0.0, 0.0], [dz, dx]]))

    aligned = ref.aligned_to(1)

    inner = (slice(12, -12), slice(12, -12))          # ignore edge extrapolation
    assert _mse(aligned[inner], frame_i[inner]) < _mse(ref.image[inner], frame_i[inner]), (
        "aligned reference must sit closer to the frame than the unaligned one"
    )


def test_alignment_recovers_almost_all_of_the_error():
    truth = _structured_image()
    dz, dx = 4.0, -3.0
    frame_i = _frame_at(truth, dz, dx)
    ref = Reference(image=truth, shifts=np.array([[0.0, 0.0], [dz, dx]]))

    inner = (slice(12, -12), slice(12, -12))
    before = _mse(ref.image[inner], frame_i[inner])
    after = _mse(ref.aligned_to(1)[inner], frame_i[inner])
    assert after < 0.05 * before, f"expected >20x error reduction, got {before / after:.1f}x"


def test_shift_sign_is_not_symmetric():
    """Guards the convention itself: negating the shift must NOT be equivalent."""
    truth = _structured_image()
    dz, dx = 4.0, -3.0
    frame_i = _frame_at(truth, dz, dx)
    inner = (slice(12, -12), slice(12, -12))

    right = Reference(truth, np.array([[0.0, 0.0], [dz, dx]])).aligned_to(1)
    wrong = Reference(truth, np.array([[0.0, 0.0], [-dz, -dx]])).aligned_to(1)
    assert _mse(right[inner], frame_i[inner]) < _mse(wrong[inner], frame_i[inner]), (
        "a sign flip in the shift convention must be detectable"
    )


# --------------------------------------------------------------------------
# The reason it matters
# --------------------------------------------------------------------------
def test_misalignment_penalises_a_sharp_output_more_than_a_blurred_one():
    """The bias that made the bug dangerous rather than merely noisy.

    A blurred prediction is already smooth, so displacing the reference costs it
    little. A sharp prediction is punished for detail that is present and simply
    offset. Scoring unaligned therefore pushes the ranking toward blur -- the
    same failure mode docs/FINDINGS.md section 9 documents for PSNR/SSIM.
    """
    truth = _structured_image()
    dz, dx = 4.0, 3.0
    ref = Reference(image=truth, shifts=np.array([[0.0, 0.0], [dz, dx]]))

    # Two predictions living in frame 1's coordinates: one faithful, one blurred.
    frame1_truth = _frame_at(truth, dz, dx)
    sharp = frame1_truth
    blurred = gaussian_filter(frame1_truth, 2.0)

    inner = (slice(12, -12), slice(12, -12))
    sharp_penalty = (_mse(ref.image[inner], sharp[inner])
                     - _mse(ref.aligned_to(1)[inner], sharp[inner]))
    blurred_penalty = (_mse(ref.image[inner], blurred[inner])
                       - _mse(ref.aligned_to(1)[inner], blurred[inner]))

    assert sharp_penalty > blurred_penalty, (
        f"misalignment must cost the sharp output more; "
        f"sharp={sharp_penalty:.4f} blurred={blurred_penalty:.4f}"
    )


def test_aligned_scoring_ranks_sharp_above_blurred_where_unaligned_does_not():
    """End-to-end consequence: the fix changes which model wins."""
    truth = _structured_image()
    dz, dx = 5.0, 4.0
    ref = Reference(image=truth, shifts=np.array([[0.0, 0.0], [dz, dx]]))

    frame1_truth = _frame_at(truth, dz, dx)
    sharp = frame1_truth
    blurred = gaussian_filter(frame1_truth, 2.5)
    inner = (slice(14, -14), slice(14, -14))

    aligned_ref = ref.aligned_to(1)
    assert _mse(aligned_ref[inner], sharp[inner]) < _mse(aligned_ref[inner], blurred[inner]), (
        "against an aligned reference the faithful prediction must win"
    )

    # And the defect: unaligned, the blurred output scores BETTER than the
    # faithful one. That inversion is what the fix removes.
    assert _mse(ref.image[inner], blurred[inner]) < _mse(ref.image[inner], sharp[inner]), (
        "precondition: unaligned scoring must prefer blur, else the test proves nothing"
    )
