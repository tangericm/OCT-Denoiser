"""Registration and reference-averaging tests.

Shifts are applied to synthetic frames and then recovered, so accuracy is
measured against known truth rather than eyeballed.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as ndshift

from octdenoiser.eval.reference import (
    average_linear,
    correlation,
    per_aline_axial_shift,
    phase_correlation_shift,
    register_stack,
    speckle_contrast,
)

H, W = 96, 64


def layered(seed: int = 0) -> np.ndarray:
    """Log-domain B-scan stand-in: bright layers over a dark background."""
    rng = np.random.default_rng(seed)
    z = np.arange(H)[:, None]
    img = np.zeros((H, W))
    for row, amp in ((20, 1.0), (30, 0.65), (48, 0.4), (66, 0.25)):
        img = img + amp * np.exp(-0.5 * ((z - row) / 2.2) ** 2)
    img = img * (1.0 + 0.10 * rng.standard_normal((1, W)))
    return img + 0.02 * rng.standard_normal((H, W))


# --------------------------------------------------------------------------
# Shift estimation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("truth", [(3.0, 0.0), (0.0, 2.0), (-4.0, 3.0), (5.0, -2.0)])
def test_recovers_integer_shifts(truth):
    ref = layered()
    mov = ndshift(ref, (-truth[0], -truth[1]), order=3, mode="nearest")
    dz, dx = phase_correlation_shift(ref, mov)
    assert dz == pytest.approx(truth[0], abs=0.25), f"dz {dz} vs {truth[0]}"
    assert dx == pytest.approx(truth[1], abs=0.25), f"dx {dx} vs {truth[1]}"


@pytest.mark.parametrize("truth", [(1.5, 0.0), (0.0, -0.5), (2.25, 1.75)])
def test_recovers_subpixel_shifts(truth):
    """Bulk axial motion between OCT frames is rarely a whole pixel."""
    ref = layered()
    mov = ndshift(ref, (-truth[0], -truth[1]), order=3, mode="nearest")
    dz, dx = phase_correlation_shift(ref, mov)
    assert dz == pytest.approx(truth[0], abs=0.35)
    assert dx == pytest.approx(truth[1], abs=0.35)


def test_zero_shift_recovered_as_zero():
    ref = layered()
    dz, dx = phase_correlation_shift(ref, ref.copy())
    assert abs(dz) < 0.1 and abs(dx) < 0.1


def test_shift_estimate_survives_added_noise():
    ref = layered()
    rng = np.random.default_rng(3)
    mov = ndshift(ref, (-2.0, -1.0), order=3, mode="nearest")
    mov = mov + 0.15 * rng.standard_normal(mov.shape)
    dz, dx = phase_correlation_shift(ref, mov)
    assert dz == pytest.approx(2.0, abs=0.4)
    assert dx == pytest.approx(1.0, abs=0.4)


def test_max_shift_rejects_implausible_peaks():
    ref = layered()
    mov = ndshift(ref, (-2.0, 0.0), order=3, mode="nearest")
    dz, _ = phase_correlation_shift(ref, mov, max_shift=6)
    assert abs(dz) <= 6.5


@pytest.mark.parametrize("max_shift", [1, 2, 4, 8, 16])
def test_max_shift_never_yields_nan(max_shift):
    """Regression: masking the correlation with -inf poisoned the parabolic fit.

    When the peak landed on the mask boundary, -inf - 2*y + -inf evaluated to
    -inf and the sub-pixel offset came out NaN, which then propagated silently
    into every shift. The mask now applies only to the argmax search.
    """
    ref = layered()
    for truth in (0.0, 1.0, 3.0, -3.0):
        mov = ndshift(ref, (-truth, 0.0), order=3, mode="nearest")
        dz, dx = phase_correlation_shift(ref, mov, max_shift=max_shift)
        assert np.isfinite(dz) and np.isfinite(dx), f"NaN at max_shift={max_shift}, truth={truth}"


def test_tight_max_shift_still_recovers_small_shift():
    ref = layered()
    mov = ndshift(ref, (-1.5, -1.0), order=3, mode="nearest")
    dz, dx = phase_correlation_shift(ref, mov, max_shift=4)
    assert dz == pytest.approx(1.5, abs=0.35)
    assert dx == pytest.approx(1.0, abs=0.35)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        phase_correlation_shift(np.zeros((8, 8)), np.zeros((8, 9)))


# --------------------------------------------------------------------------
# Stack registration
# --------------------------------------------------------------------------
def make_stack(n: int, seed: int = 0, motion_at: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frames with drifting shifts, optionally one large motion event."""
    base = layered(seed)
    rng = np.random.default_rng(seed + 100)
    frames, truth = [], []
    for i in range(n):
        dz = 0.35 * i + rng.normal(0, 0.2)
        dx = 0.12 * i + rng.normal(0, 0.15)
        if motion_at is not None and i == motion_at:
            dz += 9.0          # a bulk motion event, like Maestro2 frame 4
        f = ndshift(base, (dz, dx), order=3, mode="nearest")
        frames.append(f + 0.05 * rng.standard_normal(base.shape))
        truth.append((dz, dx))
    return np.stack(frames), np.asarray(truth)


def test_registration_raises_correlation():
    frames, _ = make_stack(12, seed=1)
    _, res = register_stack(frames, max_shift=24)
    assert res.improvement() > 0.0, res.summary()
    assert res.correlations.mean() > res.correlations_before.mean()
    assert res.correlations.min() > 0.9, res.summary()


def test_registration_recovers_applied_shifts():
    frames, truth = make_stack(10, seed=2)
    _, res = register_stack(frames, max_shift=24)
    # Shifts are measured relative to the reference frame.
    est = res.shifts[:, 0] - res.shifts[0, 0]
    ref = -(truth[:, 0] - truth[0, 0])
    assert np.allclose(est, ref, atol=0.5), f"\nest={np.round(est,2)}\nref={np.round(ref,2)}"


def test_motion_event_is_registered_not_silently_averaged():
    """Frame 4 of the Maestro2 stack drops to 0.700 correlation."""
    frames, _ = make_stack(10, seed=3, motion_at=4)
    _, res = register_stack(frames, max_shift=32)
    assert res.correlations_before[4] < res.correlations_before[3]
    assert res.correlations[4] > 0.9, (
        f"a recoverable motion event must register: {res.correlations[4]:.4f}"
    )


def test_unrecoverable_frame_is_rejected():
    frames, _ = make_stack(8, seed=4)
    rng = np.random.default_rng(0)
    frames[5] = rng.standard_normal(frames[5].shape)  # pure noise, unregisterable
    _, res = register_stack(frames, max_shift=24, min_correlation=0.8)
    assert not res.kept[5], res.summary()
    assert res.n_kept == 7
    assert res.notes and "dropped frames" in res.notes[0]


def test_raises_when_only_the_reference_survives():
    """The reference correlates 1.0 with itself, so it always survives.

    The failure that matters is having too few frames left to average, not
    having none.
    """
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((4, H, W))  # mutually unregisterable noise
    with pytest.raises(RuntimeError, match="only the reference frame cleared"):
        register_stack(frames, min_correlation=0.99)


def test_reference_always_clears_the_threshold():
    frames, _ = make_stack(6, seed=7)
    _, res = register_stack(frames, reference_index=3, min_correlation=0.5, max_shift=24)
    assert res.kept[3], "the reference cannot be rejected against itself"


def test_roi_restricts_estimation_but_not_application():
    """Prior MATLAB work cropped rows 130:600 to key on tissue, not vitreous."""
    frames, _ = make_stack(6, seed=5)
    reg, res = register_stack(frames, roi=(10, 80), max_shift=24)
    assert reg.shape == frames.shape, "the shift applies to the full frame"
    assert res.improvement() > 0.0


def test_worsening_shifts_are_reverted():
    """A spurious peak must never be accepted.

    On the Maestro2 "YM" stack a wide search produced mean shifts of dz=22,
    dx=20 px and drove one frame's correlation from 0.716 to 0.177. Zero shift
    is always available and always at least as good.
    """
    frames, _ = make_stack(6, seed=11)
    rng = np.random.default_rng(1)
    # A frame that cannot be aligned: any shift makes it worse or no better.
    frames[3] = frames[0] + 3.0 * rng.standard_normal(frames[0].shape)

    _, guarded = register_stack(frames, max_shift=24, reject_worsening=True)
    _, unguarded = register_stack(frames, max_shift=24, reject_worsening=False)

    assert (guarded.correlations >= unguarded.correlations - 1e-9).all(), (
        "the guard must never do worse than the unguarded fit"
    )
    assert (guarded.correlations >= guarded.correlations_before - 1e-9).all(), (
        "no frame may end up worse aligned than it started"
    )


def test_guard_leaves_good_registrations_untouched():
    frames, _ = make_stack(8, seed=12)
    _, guarded = register_stack(frames, max_shift=24, reject_worsening=True)
    _, plain = register_stack(frames, max_shift=24, reject_worsening=False)
    assert np.allclose(guarded.shifts, plain.shifts), "clean stacks should be unaffected"


def test_reference_frame_registers_to_itself():
    frames, _ = make_stack(6, seed=6)
    _, res = register_stack(frames, reference_index=2, max_shift=24)
    assert res.correlations[2] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(res.shifts[2], 0.0, atol=0.1)


def test_bad_reference_index_raises():
    frames, _ = make_stack(4)
    with pytest.raises(ValueError, match="out of range"):
        register_stack(frames, reference_index=9)


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match=r"expected \[N, H, W\]"):
        register_stack(np.zeros((8, 8)))


# --------------------------------------------------------------------------
# Per-A-line refinement
# --------------------------------------------------------------------------
def test_per_aline_recovers_a_tilt():
    """Eye motion during a scan leaves tilt a global shift cannot remove."""
    ref = layered()
    tilt = np.linspace(-3.0, 3.0, W)
    mov = np.stack(
        [ndshift(ref[:, x], tilt[x], order=3, mode="nearest") for x in range(W)], axis=1
    )
    est = per_aline_axial_shift(ref, mov, group=8)
    assert np.corrcoef(est, -tilt)[0, 1] > 0.95, "tilt should be recovered"


def test_per_aline_refinement_improves_a_tilted_stack():
    ref = layered()
    tilt = np.linspace(-2.5, 2.5, W)
    frames = np.stack([
        ref,
        np.stack([ndshift(ref[:, x], tilt[x], order=3, mode="nearest") for x in range(W)], axis=1),
    ])
    _, plain = register_stack(frames, max_shift=16)
    _, refined = register_stack(frames, max_shift=16, refine_per_aline=True, aline_group=8)
    assert refined.correlations[1] > plain.correlations[1], (
        f"per-A-line: {refined.correlations[1]:.4f} vs global-only {plain.correlations[1]:.4f}"
    )
    assert refined.per_aline is not None


# --------------------------------------------------------------------------
# Averaging
# --------------------------------------------------------------------------
def test_average_suppresses_speckle_toward_one_over_sqrt_n():
    """Averaging N decorrelated realisations cuts speckle contrast by sqrt(N)."""
    rng = np.random.default_rng(0)
    n = 25
    mean_level = 4.0
    # Rayleigh amplitude -> exponential intensity, the fully-developed case.
    stack = rng.exponential(mean_level, size=(n, H, W))

    before = speckle_contrast(stack[0])
    after = speckle_contrast(average_linear(stack))
    assert before == pytest.approx(1.0, abs=0.1), f"exponential intensity -> contrast 1, got {before}"
    assert after == pytest.approx(1.0 / np.sqrt(n), rel=0.25), f"got {after}"


def test_average_respects_kept_mask():
    stack = np.ones((4, 8, 8))
    stack[2] = 100.0
    kept = np.array([True, True, False, True])
    assert average_linear(stack, kept).mean() == pytest.approx(1.0)


def test_average_rejects_empty_selection():
    with pytest.raises(ValueError, match="no frames selected"):
        average_linear(np.ones((3, 4, 4)), np.zeros(3, dtype=bool))


def test_average_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expected \[N, H, W\]"):
        average_linear(np.ones((4, 4)))


def test_correlation_is_one_for_identical_images():
    a = layered()
    assert correlation(a, a) == pytest.approx(1.0, abs=1e-9)


def test_speckle_contrast_honours_roi():
    img = np.ones((20, 20))
    img[:10] = 5.0
    assert speckle_contrast(img, roi=(0, 10, 0, 20)) == pytest.approx(0.0, abs=1e-9)
