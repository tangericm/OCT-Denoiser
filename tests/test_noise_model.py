"""Noise-model calibration tests.

The R2R component's unbiasedness depends on this fit being right, so these use
synthetic frames with KNOWN (read_var, gain, rin) and check recovery. No
instrument data required.
"""
from __future__ import annotations

import numpy as np
import pytest

from octdenoiser.physics.noise_model import (
    ADU_FULL_SCALE,
    NoiseModel,
    fit_noise_model,
    load_background_frames,
    photon_transfer_curve,
)


def synth_frames(
    n_frames: int = 40,
    pixels: int = 256,
    alines: int = 64,
    read_var: float = 36.0,
    gain: float = 5.0,
    rin: float = 1.2e-3,
    seed: int = 7,
) -> np.ndarray:
    """Frames whose temporal variance follows read_var + gain*mu + rin*mu^2.

    The mean level sweeps across the spectral axis the way a source envelope
    does, giving the fit a wide lever arm — mirroring real background frames.

    The envelope is deliberately NARROW (sigma 0.15, not 0.25). read_var is only
    observable where the shot and RIN terms are small, so the sweep has to reach
    genuinely low mean levels. At sigma 0.25 the edges bottom out near mu=359,
    where read_var is under 2% of the total variance, and the intercept would be
    extrapolated from a range that never approaches zero. At sigma 0.15 the
    edges reach mu~18, where read_var is a third of the variance and the fit has
    real leverage on it. Real background frames have this property because the
    source envelope falls to near-dark at the spectrometer edges.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, pixels)[:, None]
    envelope = 2600.0 * np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2) + 8.0
    mu = np.repeat(envelope, alines, axis=1)

    sigma = np.sqrt(read_var + gain * mu + rin * mu * mu)
    return mu[None, ...] + sigma[None, ...] * rng.standard_normal((n_frames, pixels, alines))


# --------------------------------------------------------------------------
# Parameter recovery
# --------------------------------------------------------------------------
def test_recovers_known_parameters():
    truth = dict(read_var=36.0, gain=5.0, rin=1.2e-3)
    model, _ = fit_noise_model(synth_frames(n_frames=60, **truth), source="synthetic")

    assert model.is_physical(), model.summary()
    assert model.gain == pytest.approx(truth["gain"], rel=0.10), model.summary()
    assert model.rin == pytest.approx(truth["rin"], rel=0.25), model.summary()
    # Read noise is the hardest term: it only shows where mu is small, so it
    # gets the loosest tolerance.
    assert model.read_var == pytest.approx(truth["read_var"], rel=0.60), model.summary()
    assert model.r_squared > 0.99


@pytest.mark.parametrize("gain", [1.5, 4.46, 6.25])
def test_recovers_gain_across_instruments(gain):
    """4.46 and 6.25 are the values measured on two Maestro3 acquisitions."""
    model, _ = fit_noise_model(synth_frames(n_frames=60, gain=gain))
    assert model.gain == pytest.approx(gain, rel=0.12), model.summary()


def test_pure_shot_noise_gives_near_zero_rin():
    model, _ = fit_noise_model(synth_frames(n_frames=80, read_var=25.0, gain=3.0, rin=0.0))
    assert model.is_physical()
    assert abs(model.rin) < 1e-4, model.summary()
    assert model.gain == pytest.approx(3.0, rel=0.10)


# --------------------------------------------------------------------------
# The failure the real data exhibited
# --------------------------------------------------------------------------
def test_linear_model_would_fail_where_quadratic_succeeds():
    """A straight-line fit to RIN-dominated data returns a negative intercept.

    This is precisely what the real background frames produced (-1518 ADU^2),
    and it is why the quadratic term is not optional.
    """
    frames = synth_frames(n_frames=60, read_var=36.0, gain=5.0, rin=3e-3)
    curve = photon_transfer_curve(frames)

    A = np.vstack([np.ones_like(curve.mean), curve.mean]).T
    linear_intercept = np.linalg.lstsq(A, curve.variance, rcond=None)[0][0]
    assert linear_intercept < 0.0, "expected the linear model to misfit RIN data"

    model, _ = fit_noise_model(frames)
    assert model.read_var > 0.0, model.summary()
    assert model.is_physical()


def test_fit_never_returns_negative_parameters():
    """NNLS keeps every term in the physically valid region.

    On real data, M3_Macula_3x3mm produced gain = -0.0120 from an unconstrained
    fit -- numerically good (R^2 0.992) but impossible. Constrained, it returns
    exactly 0.0 with R^2 unchanged at 0.99217, i.e. the negative term explained
    nothing. RIN-dominated data with no separable shot component is the case
    that provokes this.
    """
    frames = synth_frames(n_frames=12, read_var=40.0, gain=0.0, rin=4e-3, seed=3)
    model, _ = fit_noise_model(frames)

    assert model.read_var >= 0.0, model.summary()
    assert model.gain >= 0.0, model.summary()
    assert model.rin >= 0.0, model.summary()
    assert model.is_physical(), model.summary()
    assert model.r_squared > 0.95, model.summary()


def test_pooled_fit_reports_poor_r2_when_gain_is_not_shared():
    """Pooling is a hypothesis test, not a production calibration path.

    Detector gain is an adjustable acquisition setting on this instrument (the
    Maestro2 folder names carry "gain165"/"gain167"), so a shared-gain fit
    should fit badly when the acquisitions differ. On real Maestro3 data pooling
    dropped mean R^2 from 0.995 to 0.323.
    """
    from octdenoiser.physics.noise_model import fit_pooled_noise_model, photon_transfer_curve

    curves = {
        "low_gain": photon_transfer_curve(synth_frames(n_frames=40, gain=1.0, seed=1)),
        "high_gain": photon_transfer_curve(synth_frames(n_frames=40, gain=9.0, seed=2)),
    }
    pooled = fit_pooled_noise_model(curves)
    assert len(pooled) == 2
    gains = {round(m.gain, 9) for m in pooled.values()}
    assert len(gains) == 1, "pooled fit must share one gain by construction"
    assert min(m.r_squared for m in pooled.values()) < 0.9, (
        "a shared-gain fit over genuinely different gains should fit poorly"
    )


def test_pooled_fit_recovers_a_genuinely_shared_gain():
    from octdenoiser.physics.noise_model import fit_pooled_noise_model, photon_transfer_curve

    curves = {
        f"acq{i}": photon_transfer_curve(synth_frames(n_frames=40, gain=4.0, seed=i))
        for i in range(3)
    }
    pooled = fit_pooled_noise_model(curves)
    shared = next(iter(pooled.values())).gain
    assert shared == pytest.approx(4.0, rel=0.15)
    assert all(m.r_squared > 0.95 for m in pooled.values())


def test_is_physical_rejects_negative_intercept():
    assert not NoiseModel(read_var=-1518.0, gain=6.25, rin=0.0).is_physical()
    assert not NoiseModel(read_var=10.0, gain=-1.0, rin=0.0).is_physical()
    assert NoiseModel(read_var=10.0, gain=5.0, rin=1e-3).is_physical()


def test_read_noise_std_is_nan_when_unphysical():
    assert np.isnan(NoiseModel(read_var=-5.0, gain=1.0, rin=0.0).read_noise_std)


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def test_saturated_pixels_are_excluded():
    """Clipped pixels have collapsed variance and would flatten the curve."""
    frames = synth_frames(n_frames=60)
    frames[:, :8, :] = ADU_FULL_SCALE  # a saturated band, zero variance
    model, curve = fit_noise_model(frames)
    assert model.is_physical(), model.summary()
    assert curve.mean.max() < ADU_FULL_SCALE


def test_hot_pixels_do_not_wreck_the_fit():
    frames = synth_frames(n_frames=60)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, frames.shape[1], size=40)
    frames[:, idx, :] += rng.standard_normal((60, 40, frames.shape[2])) * 800.0
    model, _ = fit_noise_model(frames)
    assert model.is_physical(), model.summary()
    assert model.gain == pytest.approx(5.0, rel=0.25), model.summary()


def test_variance_and_std_are_consistent():
    m = NoiseModel(read_var=36.0, gain=5.0, rin=1e-3)
    mu = np.array([0.0, 100.0, 2000.0])
    assert np.allclose(m.std(mu) ** 2, m.variance(mu))
    assert m.variance(0.0) == pytest.approx(36.0)


def test_too_few_frames_rejected():
    with pytest.raises(ValueError, match="at least 3 frames"):
        fit_noise_model(synth_frames(n_frames=2))


def test_missing_background_frames_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No background frames"):
        load_background_frames(str(tmp_path), 256, 64)


def test_wrong_dimensions_rejected(tmp_path):
    (tmp_path / "back000.raw").write_bytes(np.zeros(100, dtype=np.uint16).tobytes())
    with pytest.raises(ValueError, match="expected"):
        load_background_frames(str(tmp_path), 256, 64)
