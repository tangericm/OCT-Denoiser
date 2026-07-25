"""Configuration validation.

These guard the class of defect that shipped in tune.py: a config that is
internally inconsistent but fails only at the first forward pass, or worse,
silently trains a different architecture than intended.
"""
from __future__ import annotations

import pytest

from octdenoiser.configs.default import ConfigError, FolderSpec, TrainConfig


def make_spec(**kw) -> FolderSpec:
    base = dict(root_folder="r", data_folder="d", pixels=2048, alines=1024, crop_depth=(0, 1024))
    base.update(kw)
    return FolderSpec(**base)


# --------------------------------------------------------------------------
# FolderSpec
# --------------------------------------------------------------------------
def test_folderspec_accepts_shipped_values():
    fs = make_spec(window_sigma=0.05, gap=0.60, gap_offset=0.015,
                   n_sub_windows=2, sub_window_spread=0.5)
    assert fs.in_channels_bandgap == 6


@pytest.mark.parametrize("bad", [(1024, 1024), (600, 100), (-1, 512), (0, 4096)])
def test_folderspec_rejects_bad_crop_depth(bad):
    with pytest.raises(ConfigError, match="crop_depth"):
        make_spec(crop_depth=bad)


def test_folderspec_rejects_nonpositive_dims():
    with pytest.raises(ConfigError, match="positive"):
        make_spec(pixels=0)


def test_folderspec_rejects_negative_sub_windows():
    with pytest.raises(ConfigError, match="n_sub_windows"):
        make_spec(n_sub_windows=-1)


# --------------------------------------------------------------------------
# TrainConfig enum-likes
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value,match",
    [
        ("patch_mode", "strips", "patch_mode"),
        ("input_mode", "bandpass", "input_mode"),
        ("target_mode", "avg", "target_mode"),
        ("tiff_dtype", "int16", "tiff_dtype"),
        ("snr_sig_stat", "p", "snr_sig_stat"),
        ("snr_sig_stat", "percentile99", "snr_sig_stat"),
        ("train_frac", 1.0, "train_frac"),
        ("train_frac", 0.0, "train_frac"),
        ("batch_size", 0, "batch_size"),
        ("num_workers", -1, "num_workers"),
    ],
)
def test_trainconfig_rejects_bad_scalars(field, value, match):
    with pytest.raises(ConfigError, match=match):
        TrainConfig(**{field: value})


@pytest.mark.parametrize("stat", ["max", "p99", "p99.99", "p50.5"])
def test_trainconfig_accepts_valid_snr_stats(stat):
    assert TrainConfig(snr_sig_stat=stat).snr_sig_stat == stat


def test_trainconfig_rejects_inverted_roi():
    with pytest.raises(ConfigError, match="snr_sig_y0"):
        TrainConfig(snr_sig_y0=600, snr_sig_y1=111)


# --------------------------------------------------------------------------
# Cross-field consistency — the tune.py bug class
# --------------------------------------------------------------------------
def test_multilevel_model_requires_sub_windows():
    """The exact defect that shipped: multilevel model with n_sub_windows unset."""
    with pytest.raises(ConfigError, match="n_sub_windows > 0"):
        TrainConfig(
            model_name="resunet_pseudo3d_multilevel",
            folder_specs=[make_spec(n_sub_windows=0)],
        )


def test_multilevel_model_requires_bandgap_input():
    with pytest.raises(ConfigError, match="bandgap"):
        TrainConfig(
            model_name="resunet_pseudo3d_multilevel",
            input_mode="fullband",
            folder_specs=[make_spec(n_sub_windows=2)],
        )


def test_heterogeneous_sub_windows_rejected():
    """Only folder_specs[0] sizes the model stem, so a mismatch is silent."""
    with pytest.raises(ConfigError, match="share n_sub_windows"):
        TrainConfig(
            model_name="resunet_pseudo3d_multilevel",
            folder_specs=[make_spec(n_sub_windows=2), make_spec(n_sub_windows=4)],
        )


def test_fullband_input_rejects_sub_windows():
    with pytest.raises(ConfigError, match="n_sub_windows must be 0"):
        TrainConfig(
            model_name="resunet_pseudo3d",
            input_mode="fullband",
            folder_specs=[make_spec(n_sub_windows=2)],
        )


def test_shipped_training_config_is_valid():
    """The config in cli/train.py must survive validation."""
    cfg = TrainConfig(
        model_name="resunet_pseudo3d_multilevel",
        base=32,
        batch_size=12,
        patch_h=288,
        patch_w=32,
        patches_per_frame=32,
        patch_mode="strip",
        w_charb=0.0103,
        w_grad=0.0102,
        snr_sig_stat="p99.99",
        early_stop_patience=20,
        folder_specs=[make_spec(window_sigma=0.05, gap=0.60, gap_offset=0.015,
                                n_sub_windows=2, sub_window_spread=0.5)],
    )
    assert cfg.folder_specs[0].in_channels_bandgap == 6


def test_mirror_study_baseline_configs_are_valid():
    """The 1-channel fullband baselines (dncnn / unet2d / resunet-1ch)."""
    for name in ("dncnn", "unet2d", "resunet_pseudo3d"):
        cfg = TrainConfig(
            model_name=name,
            input_mode="fullband",
            target_mode="average",
            folder_specs=[make_spec(n_sub_windows=0)],
        )
        assert cfg.model_name == name


def test_folder_specs_none_is_permitted():
    """Defaults must remain constructible; the dataloader raises later."""
    assert TrainConfig().folder_specs is None
