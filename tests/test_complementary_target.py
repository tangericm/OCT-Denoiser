"""Complementary sub-band target and auto-ROI detection.

The complementary target fixes the defect that the full band CONTAINS both
sub-bands, so targeting it leaks input noise into the target. Measured on real
data: sub-band input against full-band target leaves speckle correlated at
+0.138, against +0.003 for the complementary view.
"""
from __future__ import annotations

import numpy as np
import pytest

from octdenoiser.configs.default import ConfigError, FolderSpec, TrainConfig
from octdenoiser.data.dataset import RawBscanDataset
from octdenoiser.eval.reference import auto_roi, register_stack


def make_spec(**kw) -> FolderSpec:
    base = dict(root_folder="r", data_folder="d", pixels=2048, alines=1024, crop_depth=(0, 1024))
    base.update(kw)
    return FolderSpec(**base)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------
def test_complementary_config_is_accepted():
    cfg = TrainConfig(
        model_name="resunet_pseudo3d", input_mode="bandgap", target_mode="complementary",
        folder_specs=[make_spec(n_sub_windows=0)],
    )
    assert cfg.target_mode == "complementary"


def test_complementary_rejects_multilevel_model():
    """One sub-band in is 1 channel; the multi-level stem cannot consume it."""
    with pytest.raises(ConfigError, match="multi-level spectral stem"):
        TrainConfig(
            model_name="resunet_pseudo3d_multilevel", target_mode="complementary",
            folder_specs=[make_spec(n_sub_windows=2)],
        )


def test_complementary_rejects_sub_windows():
    with pytest.raises(ConfigError, match="n_sub_windows must be 0"):
        TrainConfig(
            model_name="resunet_pseudo3d", target_mode="complementary",
            folder_specs=[make_spec(n_sub_windows=2)],
        )


def test_complementary_requires_bandgap_input():
    with pytest.raises(ConfigError, match='needs input_mode="bandgap"'):
        TrainConfig(
            model_name="resunet_pseudo3d", input_mode="fullband",
            target_mode="complementary", folder_specs=[make_spec()],
        )


# --------------------------------------------------------------------------
# Dataset behaviour
# --------------------------------------------------------------------------
def _dataset(spec, **kw):
    base = dict(folder_specs=[spec], split="train", train_frac=0.7, patch_h=32, patch_w=16,
                patches_per_frame=4, patch_mode="patch", augment=False, seed=0,
                cache_frames_per_worker=4, input_mode="bandgap", target_mode="complementary")
    base.update(kw)
    return RawBscanDataset(**base)


def test_complementary_yields_single_channel_input(synthetic_spec):
    ds = _dataset(synthetic_spec)
    x, y, _ = ds[0]
    assert x.shape == (1, 32, 16), f"one sub-band in, got {tuple(x.shape)}"
    assert y.shape == (1, 32, 16)


def test_input_and_target_are_different_sub_bands(synthetic_spec):
    """The whole point: the target must not contain the input."""
    ds = _dataset(synthetic_spec, patch_mode="patch", patch_h=64, patch_w=32)
    x, y, _ = ds[0]
    assert not np.allclose(x.numpy()[0], y.numpy()[0]), "input and target must differ"


def test_view_direction_is_randomised(synthetic_spec):
    """w1->w2 and w2->w1 must both occur, or a directional bias is learned.

    The bands sit at different k and scattering is wavelength-dependent, so
    their expected signals genuinely differ (0.269 measured mismatch);
    alternating keeps that bias symmetric.
    """
    ds = _dataset(synthetic_spec, patches_per_frame=16)
    seen = set()
    for i in range(min(len(ds), 40)):
        ds[i]
        seen.add(ds._swap_views)
    assert seen == {True, False}, f"both directions must appear, saw {seen}"


def test_full_frame_direction_is_deterministic(synthetic_spec):
    """Validation must not vary run to run."""
    ds = _dataset(synthetic_spec, split="val", full_frame=True)
    ds[0]
    assert ds._swap_views is False
    ds[0]
    assert ds._swap_views is False


def test_fullband_target_still_works(synthetic_spec):
    """The existing mode is untouched."""
    ds = _dataset(synthetic_spec, target_mode="fullband")
    x, y, _ = ds[0]
    assert x.shape == (2, 32, 16), "bandgap input stays 2-channel"
    assert y.shape == (1, 32, 16)


# --------------------------------------------------------------------------
# Auto ROI
# --------------------------------------------------------------------------
def _frame_with_band(h, w, lo, hi, amp=1.0, floor=0.05, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), floor)
    img[lo:hi] += amp
    img[:12] += 3.0  # DC roll-off near the surface
    return img + 0.01 * rng.standard_normal((h, w))


def test_auto_roi_finds_a_narrow_retina_band():
    roi = auto_roi(_frame_with_band(1024, 128, 150, 360), pad=20)
    assert roi[0] <= 150 and roi[1] >= 360
    assert roi[1] - roi[0] < 400, f"band should stay tight, got {roi}"


def test_auto_roi_finds_a_wide_anterior_band():
    """The anterior stack spans rows 79-973; a retina ROI would miss it."""
    roi = auto_roi(_frame_with_band(1024, 128, 80, 970), pad=20)
    assert roi[0] <= 80 and roi[1] >= 970


def test_auto_roi_excludes_the_dc_rolloff():
    """Without skip_top the surface spike anchors the band at row 0."""
    roi = auto_roi(_frame_with_band(1024, 128, 400, 600), skip_top=30, pad=10)
    assert roi[0] > 30, f"DC roll-off should be excluded, got {roi}"


def test_auto_roi_survives_a_flat_frame():
    lo, hi = auto_roi(np.zeros((256, 32)))
    assert (lo, hi) == (0, 256)


def test_register_stack_accepts_auto_roi():
    frames = np.stack([_frame_with_band(256, 64, 90, 160, seed=i) for i in range(5)])
    _, res = register_stack(frames, roi="auto", max_shift=16)
    assert res.roi is not None
    assert res.roi[0] < 90 and res.roi[1] > 160
    assert any("registration ROI" in n for n in res.notes)


def test_register_stack_rejects_bad_roi_string():
    frames = np.stack([_frame_with_band(128, 32, 40, 80, seed=i) for i in range(3)])
    with pytest.raises(ValueError, match="roi must be"):
        register_stack(frames, roi="tissue")
