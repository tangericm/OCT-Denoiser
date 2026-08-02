from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".npy"}


@dataclass(frozen=True)
class FrameRef:
    path: Path
    index: int | None


@dataclass(frozen=True)
class Volume:
    name: str
    frames: tuple[FrameRef, ...]
    shape: tuple[int, int]


Pair: TypeAlias = tuple[FrameRef, FrameRef]


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def _inspect_file(path: Path) -> tuple[tuple[int, ...], np.dtype[np.generic]]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return tuple(array.shape), array.dtype
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        return tuple(series.shape), np.dtype(series.dtype)


def _file_refs(path: Path) -> tuple[list[FrameRef], tuple[int, int]]:
    shape, _ = _inspect_file(path)
    if len(shape) == 2:
        return [FrameRef(path, None)], (shape[0], shape[1])
    if len(shape) == 3:
        if shape[0] < 1:
            raise ValueError(f"B-scan stack is empty: {path}")
        return [FrameRef(path, index) for index in range(shape[0])], (shape[1], shape[2])
    raise ValueError(f"Expected a 2D B-scan or 3D stack, got shape {shape} in {path}")


def discover_volumes(inputs: tuple[str, ...] | list[str]) -> list[Volume]:
    """Discover ordered grayscale B-scans in TIFF stacks, TIFF folders, or NPY stacks."""

    volumes: list[Volume] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"B-scan input does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported B-scan file type: {path.suffix}")
            refs, shape = _file_refs(path)
            volumes.append(Volume(path.stem, tuple(refs), shape))
            continue

        files = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES),
            key=_natural_key,
        )
        if not files:
            raise ValueError(f"No TIFF or NPY B-scans found directly inside {path}")

        loose_frames: list[FrameRef] = []
        loose_shape: tuple[int, int] | None = None
        for file_path in files:
            refs, shape = _file_refs(file_path)
            if len(refs) == 1:
                if loose_shape is not None and shape != loose_shape:
                    raise ValueError(f"Inconsistent B-scan shapes in {path}: {loose_shape} and {shape}")
                loose_shape = shape
                loose_frames.extend(refs)
            else:
                volumes.append(Volume(file_path.stem, tuple(refs), shape))
        if loose_frames and loose_shape is not None:
            volumes.append(Volume(path.name, tuple(loose_frames), loose_shape))

    if not volumes:
        raise ValueError("No processed B-scans were discovered.")
    return volumes


def read_frame(ref: FrameRef) -> np.ndarray:
    """Read one referenced B-scan without loading unrelated NPY frames."""

    if ref.path.suffix.lower() == ".npy":
        array = np.load(ref.path, mmap_mode="r", allow_pickle=False)
        frame = array if ref.index is None else array[ref.index]
    else:
        frame = tifffile.imread(ref.path) if ref.index is None else tifffile.imread(ref.path, key=ref.index)
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D B-scan, got shape {frame.shape} from {ref.path}")
    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError(f"B-scan data must be numeric, got {frame.dtype} in {ref.path}")
    return frame


def normalize_frame(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(frame, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("B-scan contains NaN or infinite values.")
    mean = float(array.mean())
    std = max(float(array.std()), 1e-6)
    return ((array - mean) / std).astype(np.float32, copy=False), mean, std


def restore_frame(frame: np.ndarray, mean: float, std: float, dtype: np.dtype[np.generic]) -> np.ndarray:
    restored = np.asarray(frame, dtype=np.float32) * std + mean
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.rint(np.clip(restored, limits.min, limits.max)).astype(dtype)
    return restored.astype(np.float32)


def split_pairs(
    volumes: list[Volume],
    *,
    pair_offset: int,
    group_size: int,
    train_fraction: float,
    seed: int,
) -> tuple[list[Pair], list[Pair]]:
    """Split contiguous frame groups before constructing adjacent-frame pairs."""

    blocks: list[tuple[FrameRef, ...]] = []
    for volume in volumes:
        for start in range(0, len(volume.frames), group_size):
            block = volume.frames[start : start + group_size]
            if len(block) > pair_offset:
                blocks.append(block)
    if len(blocks) < 2:
        raise ValueError(
            "Training needs at least two contiguous frame groups. Add another volume "
            "or reduce --group-size so train and validation remain separate."
        )

    order = np.random.default_rng(seed).permutation(len(blocks))
    n_train = int(round(len(blocks) * train_fraction))
    n_train = min(max(n_train, 1), len(blocks) - 1)
    train_ids = {int(index) for index in order[:n_train]}

    train_pairs: list[Pair] = []
    val_pairs: list[Pair] = []
    for block_index, block in enumerate(blocks):
        destination = train_pairs if block_index in train_ids else val_pairs
        destination.extend((block[index], block[index + pair_offset]) for index in range(len(block) - pair_offset))
    return train_pairs, val_pairs


class BscanPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Self-supervised pairs made from ordered, already-processed B-scans."""

    def __init__(
        self,
        pairs: list[Pair],
        *,
        patch_size: tuple[int, int] | None,
        patches_per_pair: int = 1,
        augment: bool = False,
    ):
        if not pairs:
            raise ValueError("The B-scan pair dataset is empty.")
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_pair = patches_per_pair
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ref, target_ref = self.pairs[item % len(self.pairs)]
        input_frame, _, _ = normalize_frame(read_frame(input_ref))
        target_frame, _, _ = normalize_frame(read_frame(target_ref))
        if input_frame.shape != target_frame.shape:
            raise ValueError(
                f"Paired B-scans must have the same shape, got {input_frame.shape} and {target_frame.shape}"
            )

        if self.patch_size is not None:
            patch_h, patch_w = self.patch_size
            height, width = input_frame.shape
            if patch_h > height or patch_w > width:
                raise ValueError(
                    f"Patch {self.patch_size} is larger than B-scan {input_frame.shape}. "
                    "Choose smaller --patch-height/--patch-width values."
                )
            if self.augment:
                top = int(np.random.randint(0, height - patch_h + 1))
                left = int(np.random.randint(0, width - patch_w + 1))
            else:
                top = (height - patch_h) // 2
                left = (width - patch_w) // 2
            input_frame = input_frame[top : top + patch_h, left : left + patch_w]
            target_frame = target_frame[top : top + patch_h, left : left + patch_w]

        if self.augment and bool(np.random.randint(0, 2)):
            input_frame = np.flip(input_frame, axis=1).copy()
            target_frame = np.flip(target_frame, axis=1).copy()

        x = torch.from_numpy(np.ascontiguousarray(input_frame[None]))
        y = torch.from_numpy(np.ascontiguousarray(target_frame[None]))
        return x, y
