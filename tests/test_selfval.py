"""Held-out-mask self-validation tests.

The load-bearing test is `test_ranking_matches_true_mse_ranking`. The entire
evaluation plan rests on the claim that this metric ranks models the same way a
clean reference would, so that claim is demonstrated against known ground truth
rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from octdenoiser.eval.selfval import (
    SelfValidator,
    ValidationViews,
    held_out_mse,
    held_out_psnr_proxy,
    make_validation_views,
)

H, W = 64, 48


def clean_image(seed: int = 0) -> np.ndarray:
    """Layered structure, loosely OCT-like: bright bands over a dark background."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W))
    for row, amp in ((10, 1.0), (18, 0.7), (31, 0.45), (44, 0.3)):
        z = np.arange(H)[:, None]
        img += amp * np.exp(-0.5 * ((z - row) / 1.8) ** 2)
    img = img * (1.0 + 0.15 * rng.standard_normal((1, W)))  # lateral variation
    return img


def noisy_views(clean: np.ndarray, n_views: int, sigma: float, seed: int) -> list[np.ndarray]:
    """Independent noisy realisations — what disjoint masks produce."""
    rng = np.random.default_rng(seed)
    return [clean + sigma * rng.standard_normal(clean.shape) for _ in range(n_views)]


# --------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------
WIDTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def _mse_curves(n_frames: int, sigma: float = 0.25, seed0: int = 1):
    """Average true and held-out MSE over `n_frames` independent frames."""
    clean = clean_image()
    true = np.zeros(len(WIDTHS))
    self_ = np.zeros(len(WIDTHS))
    for f in range(n_frames):
        x1, x3 = noisy_views(clean, 2, sigma=sigma, seed=seed0 + f)
        for i, w in enumerate(WIDTHS):
            pred = x1 if w == 0.0 else gaussian_filter(x1, w)
            true[i] += float(np.mean((pred - clean) ** 2))
            self_[i] += held_out_mse(pred, x3)
    return true / n_frames, self_ / n_frames


def test_ranking_matches_true_mse_ranking():
    """Held-out-mask ranking must equal clean-reference ranking.

    Denoisers are Gaussian filters of varying width, giving a genuine ranking
    problem with an interior optimum: too little smoothing leaves noise, too
    much destroys structure.

    Aggregated over 32 frames, which is how the metric is actually used. A
    single frame is NOT enough — see the companion test below.
    """
    true_mse, self_mse = _mse_curves(n_frames=32)

    rho = spearmanr(true_mse, self_mse).statistic
    assert rho == pytest.approx(1.0, abs=1e-9), (
        f"rank correlation {rho:.4f}\n  true={np.round(true_mse, 5).tolist()}"
        f"\n  self={np.round(self_mse, 5).tolist()}"
    )
    assert int(np.argmin(true_mse)) == int(np.argmin(self_mse)), (
        "the metric must select the same model a clean reference would"
    )


def test_single_frame_ranking_is_noisier_than_aggregated():
    """Documents the sample-size requirement rather than hiding it.

    On one frame the metric can swap models that are within its own noise of
    each other: measured true MSE 0.00708 vs 0.00745 (5% apart) came back as
    0.07120 vs 0.07102 (0.25% apart), inverting that pair. Aggregation fixes it,
    so validation must run over enough frames to resolve close candidates.
    """
    rho_one = spearmanr(*_mse_curves(n_frames=1)).statistic
    rho_many = spearmanr(*_mse_curves(n_frames=32)).statistic
    assert rho_many >= rho_one
    assert rho_many == pytest.approx(1.0, abs=1e-9)


def test_score_differs_from_true_mse_by_a_constant():
    """The offset is E||eta_3||^2 -- constant in the model.

    This is why the metric ranks correctly but must never be quoted as an
    absolute quality number.
    """
    clean = clean_image()
    sigma = 0.25
    x1, x3 = noisy_views(clean, 2, sigma=sigma, seed=2)

    offsets, spread = [], []
    for w in (0.25, 0.5, 1.0, 1.5, 2.0):
        pred = gaussian_filter(x1, w)
        t = float(np.mean((pred - clean) ** 2))
        s = held_out_mse(pred, x3)
        offsets.append(s - t)
        spread.append(t)

    offsets = np.asarray(offsets)
    # The offset should sit near the noise variance of the held-out view...
    assert offsets.mean() == pytest.approx(sigma**2, rel=0.20)
    # ...and vary far less than the quantity being ranked.
    assert offsets.std() < 0.15 * (max(spread) - min(spread))


def test_more_independent_noise_raises_the_floor_not_the_ranking():
    """Doubling held-out noise shifts every score but preserves the order."""
    clean = clean_image()
    rng = np.random.default_rng(5)
    x1 = clean + 0.25 * rng.standard_normal(clean.shape)

    widths = [0.25, 0.5, 1.0, 2.0, 4.0]
    orders = []
    for s3 in (0.25, 0.5):
        x3 = clean + s3 * np.random.default_rng(9).standard_normal(clean.shape)
        scores = [held_out_mse(gaussian_filter(x1, w), x3) for w in widths]
        orders.append(np.argsort(scores).tolist())
    assert orders[0] == orders[1]


def test_correlated_held_out_view_breaks_the_guarantee():
    """Independence is doing real work: reusing the INPUT as the held-out view
    makes the metric prefer doing nothing, which is exactly the failure the
    disjoint-mask construction avoids."""
    clean = clean_image()
    x1, _ = noisy_views(clean, 2, sigma=0.25, seed=3)

    widths = [0.0, 0.5, 1.0, 2.0]
    bad = [held_out_mse(x1 if w == 0 else gaussian_filter(x1, w), x1) for w in widths]
    assert int(np.argmin(bad)) == 0, "a correlated target rewards the identity"


# --------------------------------------------------------------------------
# Aggregation and diagnostics
# --------------------------------------------------------------------------
def _views(clean, sigma, seed) -> ValidationViews:
    a, b, c = noisy_views(clean, 3, sigma=sigma, seed=seed)
    return ValidationViews(input_view=a, target_view=b, held_out_view=c, seed=seed)


def test_vs_identity_is_a_weak_floor():
    """`vs_identity` is a sanity check, NOT a usefulness test.

    The identity carries 2*sigma^2 — its own noise plus the held-out view's —
    so almost any smoothing beats it. Measured, ruinous sigma=12 blur still
    scored 0.888 against the identity. Only a degenerate constant output
    actually exceeds it. Selection must therefore use the held-out MSE ranking,
    not this ratio.
    """
    clean = clean_image()
    vs = [_views(clean, 0.25, s) for s in (1, 2, 3)]
    sv = SelfValidator(n_frames=3)

    good = [gaussian_filter(v.input_view, 1.0) for v in vs]
    ruinous = [gaussian_filter(v.input_view, 12.0) for v in vs]
    constant = [np.full_like(v.input_view, float(v.input_view.mean())) for v in vs]

    assert sv.score(good, vs)["vs_identity"] < 1.0
    assert sv.score(ruinous, vs)["vs_identity"] < 1.0, (
        "documents the weakness: heavy blur still beats the identity"
    )
    assert sv.score(constant, vs)["vs_identity"] > 1.0, (
        "a degenerate constant output must exceed the identity floor"
    )


def test_ranking_still_penalises_over_smoothing():
    """What `vs_identity` misses, the ranking catches."""
    clean = clean_image()
    vs = [_views(clean, 0.25, s) for s in (1, 2, 3)]
    sv = SelfValidator(n_frames=3)

    optimal = [gaussian_filter(v.input_view, 1.0) for v in vs]
    over = [gaussian_filter(v.input_view, 12.0) for v in vs]
    assert sv.score(over, vs)["held_out_mse"] > sv.score(optimal, vs)["held_out_mse"]


def test_score_reports_expected_keys():
    clean = clean_image()
    vs = [_views(clean, 0.25, 7)]
    out = SelfValidator(n_frames=1).score([gaussian_filter(vs[0].input_view, 1.0)], vs)
    for k in ("held_out_mse", "held_out_psnr_proxy", "identity_mse", "vs_identity", "n_frames", "std"):
        assert k in out
    assert out["n_frames"] == 1.0


def test_psnr_proxy_is_monotone_in_mse():
    clean = clean_image()
    x1, x3 = noisy_views(clean, 2, sigma=0.25, seed=4)
    widths = [0.25, 0.5, 1.0, 2.0, 4.0]
    mses = [held_out_mse(gaussian_filter(x1, w), x3) for w in widths]
    psnrs = [held_out_psnr_proxy(gaussian_filter(x1, w), x3) for w in widths]
    assert spearmanr(mses, psnrs).statistic == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_seeds_are_deterministic_across_instances():
    """Two runs must validate on identical masks, or scores are incomparable."""
    assert SelfValidator(n_frames=16).seeds == SelfValidator(n_frames=16).seeds


def test_different_base_seed_changes_masks():
    a = SelfValidator(n_frames=8, base_seed=1).seeds
    b = SelfValidator(n_frames=8, base_seed=2).seeds
    assert a != b


def test_seed_lookup_wraps():
    sv = SelfValidator(n_frames=4)
    assert sv.seed_for(0) == sv.seed_for(4)


def test_rejects_bad_frame_count():
    with pytest.raises(ValueError, match="n_frames must be positive"):
        SelfValidator(n_frames=0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        held_out_mse(np.zeros((4, 4)), np.zeros((4, 5)))


def test_score_rejects_length_mismatch():
    clean = clean_image()
    vs = [_views(clean, 0.25, 1)]
    with pytest.raises(ValueError, match="predictions vs"):
        SelfValidator(n_frames=1).score([], vs)


# --------------------------------------------------------------------------
# Integration through the real reconstruction path
# --------------------------------------------------------------------------
def test_views_from_real_spectrum_are_disjoint_and_distinct():
    rng = np.random.default_rng(0)
    pixels, alines = 256, 32
    spec = (rng.standard_normal((pixels, alines))
            + 1j * rng.standard_normal((pixels, alines))).astype(np.complex64)

    v = make_validation_views(spec, seed=17, crop=(0, 128))
    assert v.shape == (128, alines)
    for a in (v.input_view, v.target_view, v.held_out_view):
        assert np.isfinite(a).all()
    assert not np.allclose(v.input_view, v.target_view)
    assert not np.allclose(v.input_view, v.held_out_view)


def test_views_are_reproducible_for_a_given_seed():
    rng = np.random.default_rng(1)
    spec = (rng.standard_normal((256, 16)) + 1j * rng.standard_normal((256, 16))).astype(np.complex64)
    a = make_validation_views(spec, seed=5, crop=(0, 128))
    b = make_validation_views(spec, seed=5, crop=(0, 128))
    assert np.array_equal(a.input_view, b.input_view)
    assert np.array_equal(a.held_out_view, b.held_out_view)


def test_rejects_non_2d_spectrum():
    with pytest.raises(ValueError, match=r"expected \[pixels, alines\]"):
        make_validation_views(np.zeros((3, 8, 8), dtype=np.complex64), seed=0, crop=(0, 4))
