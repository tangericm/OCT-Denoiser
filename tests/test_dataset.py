from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from octdenoiser.data.dataset import (
    BscanPairDataset,
    discover_volumes,
    normalize_frame,
    read_frame,
    restore_frame,
    split_pairs,
)


def test_discovers_tiff_stacks(processed_volumes: tuple[Path, Path]) -> None:
    volumes = discover_volumes([str(path) for path in processed_volumes])
    assert [len(volume.frames) for volume in volumes] == [8, 8]
    assert all(volume.shape == (32, 40) for volume in volumes)
    assert read_frame(volumes[0].frames[3]).dtype == np.uint16


def test_discovers_naturally_sorted_tiff_folder(tmp_path: Path) -> None:
    for index in (10, 2, 1):
        tifffile.imwrite(tmp_path / f"bscan{index}.tif", np.full((16, 18), index, np.uint16))
    volume = discover_volumes([str(tmp_path)])[0]
    values = [int(read_frame(ref)[0, 0]) for ref in volume.frames]
    assert values == [1, 2, 10]


def test_normalization_round_trip() -> None:
    frame = np.arange(30, dtype=np.uint16).reshape(5, 6)
    normalized, mean, std = normalize_frame(frame)
    restored = restore_frame(normalized, mean, std, frame.dtype)
    assert restored.dtype == frame.dtype
    assert np.array_equal(restored, frame)


def test_block_split_has_no_frame_overlap(processed_volumes: tuple[Path, Path]) -> None:
    volumes = discover_volumes([str(path) for path in processed_volumes])
    train_pairs, val_pairs = split_pairs(
        volumes,
        pair_offset=1,
        group_size=4,
        train_fraction=0.5,
        seed=7,
    )
    train_frames = {ref for pair in train_pairs for ref in pair}
    val_frames = {ref for pair in val_pairs for ref in pair}
    assert train_frames.isdisjoint(val_frames)


def test_pair_dataset_returns_matching_patches(processed_volumes: tuple[Path, Path]) -> None:
    volumes = discover_volumes([str(path) for path in processed_volumes])
    train_pairs, _ = split_pairs(
        volumes,
        pair_offset=1,
        group_size=4,
        train_fraction=0.5,
        seed=0,
    )
    dataset = BscanPairDataset(train_pairs, patch_size=(16, 20), augment=False)
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (1, 16, 20)
    assert inputs.dtype == targets.dtype
    assert inputs.isfinite().all()
