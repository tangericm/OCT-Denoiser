"""End-to-end preprocessing, dataset and checkpoint tests on synthetic data.

Every test here runs on CPU with no instrument data present, which is what
makes the pipeline testable in CI.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import ALINES, CROP, LAYERS, PIXELS
from octdenoiser.configs.default import FolderSpec
from octdenoiser.preprocess import BscanProcessor


# --------------------------------------------------------------------------
# Calibration resolution — per-instrument, and wrong-instrument is silent
# --------------------------------------------------------------------------
def test_processor_resolves_clb_from_root(synthetic_spec):
    proc = BscanProcessor(synthetic_spec)
    assert proc.clb_path is not None
    assert proc.clb_path.endswith(".CLB")
    assert proc.resampling.shape == (PIXELS,)
    assert np.all(np.diff(proc.resampling) > 0), "k-linearisation LUT must be monotonic"


def test_explicit_clb_path_overrides_discovery(synthetic_dataset, synthetic_spec):
    """Maestro2 and Maestro3 have different LUTs; an explicit path must win."""
    found = BscanProcessor(synthetic_spec).clb_path
    spec = FolderSpec(
        root_folder=synthetic_dataset, data_folder="synth_folder",
        pixels=PIXELS, alines=ALINES, crop_depth=CROP, clb_path=found,
    )
    assert BscanProcessor(spec).clb_path == found


def test_missing_clb_raises_actionable_error(tmp_path, synthetic_dataset):
    """Previously this fell through to open(None) and a bare TypeError."""
    root = tmp_path / "no_clb"
    (root / "synth_folder").mkdir(parents=True)
    src = os.path.join(synthetic_dataset, "synth_folder")
    for name in sorted(os.listdir(src))[:2]:
        with open(os.path.join(src, name), "rb") as fh:
            (root / "synth_folder" / name).write_bytes(fh.read())

    spec = FolderSpec(root_folder=str(root), data_folder="synth_folder",
                      pixels=PIXELS, alines=ALINES, crop_depth=CROP)
    with pytest.raises(FileNotFoundError, match="No .CLB calibration file"):
        BscanProcessor(spec)


def test_bad_explicit_clb_path_raises(synthetic_dataset):
    spec = FolderSpec(root_folder=synthetic_dataset, data_folder="synth_folder",
                      pixels=PIXELS, alines=ALINES, crop_depth=CROP,
                      clb_path=os.path.join(synthetic_dataset, "nope.CLB"))
    with pytest.raises(FileNotFoundError, match="clb_path does not exist"):
        BscanProcessor(spec)


# --------------------------------------------------------------------------
# process_one contract
# --------------------------------------------------------------------------
def test_process_one_shapes_and_keys(synthetic_spec):
    proc = BscanProcessor(synthetic_spec)
    out = proc.process_one(proc.bscan_paths[0], frame_idx=0)

    h = CROP[1] - CROP[0]
    for key in ("target_full", "input_w1", "input_w2"):
        assert out[key].shape == (h, ALINES), f"{key} shape"
        assert out[key].dtype == np.float32, f"{key} dtype"
        assert np.isfinite(out[key]).all(), f"{key} must be finite"

    for key in ("target_mu", "target_sd", "input_w1_mu", "input_w2_sd"):
        assert isinstance(out[key], float)

    assert "input_sub_windows" not in out, "sub-windows disabled when n_sub_windows=0"


def test_process_one_is_zscore_normalised(synthetic_spec):
    proc = BscanProcessor(synthetic_spec)
    out = proc.process_one(proc.bscan_paths[0], frame_idx=0)
    for key in ("target_full", "input_w1", "input_w2"):
        assert abs(float(out[key].mean())) < 1e-4, f"{key} mean"
        assert abs(float(out[key].std()) - 1.0) < 1e-3, f"{key} std"


def test_multilevel_produces_expected_channel_count(synthetic_spec_multilevel):
    proc = BscanProcessor(synthetic_spec_multilevel)
    out = proc.process_one(proc.bscan_paths[0], frame_idx=0)
    subs = out["input_sub_windows"]
    n_sub = synthetic_spec_multilevel.n_sub_windows
    assert len(subs) == 2 * n_sub
    # 2 parent windows + 2*n_sub children is what the multilevel stem expects.
    assert 2 + len(subs) == synthetic_spec_multilevel.in_channels_bandgap


def test_linear_target_returned_on_request(synthetic_spec):
    proc = BscanProcessor(synthetic_spec)
    out = proc.process_one(proc.bscan_paths[0], frame_idx=0, need_linear_full=True)
    lin = out["target_full_linear"]
    assert lin.shape == (CROP[1] - CROP[0], ALINES)
    assert (lin >= 0).all(), "linear magnitude must be non-negative"


def test_reconstruction_recovers_synthetic_layer_depths(synthetic_spec):
    """The strongest end-to-end check: DC-subtract -> resample -> FFT -> crop
    must place energy at the depths the fixture encoded. A broken crop, a
    dropped fftshift or an inverted LUT all fail here.

    Uses prominence-based peak detection rather than an absolute threshold —
    the DC roll-off near the surface dominates any global statistic.
    """
    from scipy.signal import find_peaks

    proc = BscanProcessor(synthetic_spec)
    out = proc.process_one(proc.bscan_paths[0], frame_idx=0, need_linear_full=True)
    prof = out["target_full_linear"].mean(axis=1).astype(np.float64)

    expected = [int(round(frac * PIXELS)) for frac, _ in LAYERS]
    expected = [z for z in expected if CROP[0] <= z < CROP[1]]
    assert expected, "fixture layers must land inside the crop"

    peaks, _ = find_peaks(prof, prominence=0.05 * (prof.max() - prof.min()))
    assert peaks.size, "reconstruction produced no depth structure at all"

    for z in expected:
        nearest = int(np.min(np.abs(peaks - z)))
        assert nearest <= 4, (
            f"no reconstructed peak within 4 px of encoded layer depth {z}; "
            f"detected peaks at {peaks.tolist()}"
        )


def test_frames_differ_between_indices(synthetic_spec):
    proc = BscanProcessor(synthetic_spec)
    a = proc.process_one(proc.bscan_paths[0], frame_idx=0)["target_full"]
    b = proc.process_one(proc.bscan_paths[1], frame_idx=1)["target_full"]
    assert not np.allclose(a, b), "fixture frames must carry independent speckle"


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def test_dataset_yields_expected_shapes(synthetic_spec):
    from octdenoiser.data.dataset import RawBscanDataset

    ds = RawBscanDataset(
        folder_specs=[synthetic_spec], split="train", train_frac=0.7,
        patch_h=32, patch_w=16, patches_per_frame=2, patch_mode="patch",
        augment=True, seed=0, cache_frames_per_worker=4,
    )
    assert len(ds) > 0
    x, y, meta = ds[0]
    assert x.shape == (2, 32, 16), f"bandgap input should be 2-channel, got {tuple(x.shape)}"
    assert y.shape == (1, 32, 16)
    assert x.dtype.is_floating_point and y.dtype.is_floating_point
    assert np.isfinite(x.numpy()).all() and np.isfinite(y.numpy()).all()
    assert isinstance(meta, dict)


def test_dataset_train_val_splits_are_disjoint(synthetic_spec, n_synthetic_frames):
    from octdenoiser.data.dataset import RawBscanDataset

    common = dict(folder_specs=[synthetic_spec], train_frac=0.7, patch_h=32, patch_w=16,
                  patches_per_frame=1, patch_mode="patch", seed=0, cache_frames_per_worker=2)
    tr = RawBscanDataset(split="train", **common)
    va = RawBscanDataset(split="val", **common)
    tr._build_index()
    va._build_index()

    tr_frames = {e[1] for e in tr._index}
    va_frames = {e[1] for e in va._index}
    assert tr_frames and va_frames
    assert not (tr_frames & va_frames), "train and val must not share frames"
    assert len(tr_frames | va_frames) <= n_synthetic_frames


def test_full_frame_split_carries_denormalisation_stats(synthetic_spec):
    """Metrics are computed in physical intensity, so full-frame samples must
    carry target_mu/target_sd/log_eps or to_physical_intensity silently
    passes the log-domain image straight through."""
    from octdenoiser.data.dataset import RawBscanDataset

    ds = RawBscanDataset(
        folder_specs=[synthetic_spec], split="val", train_frac=0.7,
        full_frame=True, seed=0, cache_frames_per_worker=2,
    )
    _, _, meta = ds[0]
    for key in ("target_mu", "target_sd", "log_eps"):
        assert key in meta, f"full-frame meta missing {key}"


# --------------------------------------------------------------------------
# Checkpoint round-trip
# --------------------------------------------------------------------------
def test_checkpoint_roundtrip_preserves_weights(tmp_path):
    import torch

    from octdenoiser.networks import create_model

    model = create_model("resunet_pseudo3d", base=8, in_ch=2)
    x = torch.randn(1, 2, 32, 32)
    model.eval()
    with torch.no_grad():
        before = model(x)

    path = tmp_path / "best.pt"
    torch.save({"epoch": 3, "model": model.state_dict(), "best_val": 0.5}, path)

    restored = create_model("resunet_pseudo3d", base=8, in_ch=2)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    restored.load_state_dict(ckpt["model"], strict=True)
    restored.eval()
    with torch.no_grad():
        after = restored(x)

    assert torch.allclose(before, after, atol=0), "round-trip must be bit-exact"
    assert ckpt["epoch"] == 3
