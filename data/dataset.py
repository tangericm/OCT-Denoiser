from __future__ import annotations

from collections import OrderedDict
from typing import List, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from preprocess import BscanProcessor


class _LRU:
    """Simple LRU cache for processed B-scan frames."""

    def __init__(self, max_items: int = 4):
        self.max_items = max_items
        self.d: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, val):
        self.d[key] = val
        self.d.move_to_end(key)
        if len(self.d) > self.max_items:
            self.d.popitem(last=False)


class RawBscanDataset(Dataset):
    """
    Unified dataset for both patch-based training and full-frame validation.

    When full_frame=False (default):
      Returns (x, y, meta) with x: [2, patch_h, patch_w], y: [1, patch_h, patch_w]
    When full_frame=True:
      Returns (x, y, meta) with x: [2, H, W], y: [1, H, W]
    """

    def __init__(
        self,
        folder_specs: List[Any],
        split: str,
        train_frac: float,
        patch_h: int = 128,
        patch_w: int = 128,
        patches_per_frame: int = 16,
        patch_mode: str = "strip",
        seed: int = 42,
        augment: bool = False,
        cache_frames_per_worker: int = 200,
        full_frame: bool = False,
        max_frames: int | None = None,
    ):
        self.folder_specs = folder_specs
        self.split = split
        self.train_frac = train_frac
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patches_per_frame = patches_per_frame
        self.patch_mode = patch_mode
        self.seed = seed
        self.augment = augment and not full_frame
        self.cache_frames_per_worker = cache_frames_per_worker
        self.full_frame = full_frame
        self.max_frames = max_frames

        self._procs = None
        self._paths = None
        self._index = None
        self._cache = None
        self._rng = None
        self._estimated_len = 1

    def _init_worker_state(self):
        if self._procs is not None:
            return

        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        base_seed = self.seed if self.full_frame else self.seed * wid

        self._procs = []
        self._paths = []
        for fs in self.folder_specs:
            proc = BscanProcessor(fs.root_folder, fs.to_preprocess_config())
            self._procs.append(proc)
            self._paths.append(proc.bscan_paths)

            if not self.full_frame:
                if fs.alines < self.patch_w:
                    raise ValueError(f"patch_w={self.patch_w} > alines={fs.alines} for folder={fs.data_folder}")
                z0, z1 = fs.crop_depth
                if (z1 - z0) < self.patch_h:
                    raise ValueError(f"patch_h={self.patch_h} > cropped_depth={z1-z0} for folder={fs.data_folder}")

        self._rng = np.random.RandomState(base_seed)
        index = []

        for fidx, paths in enumerate(self._paths):
            n = len(paths)
            order = np.arange(n)
            self._rng.shuffle(order)
            n_train = int(round(self.train_frac * n))
            chosen = order[:n_train] if self.split == "train" else order[n_train:]

            if self.full_frame:
                for frame_idx in chosen:
                    index.append((fidx, int(frame_idx)))
            else:
                for frame_idx in chosen:
                    for pr in range(self.patches_per_frame):
                        index.append((fidx, int(frame_idx), pr))

        if self.full_frame and self.max_frames is not None and len(index) > self.max_frames:
            subset_rng = np.random.RandomState(base_seed)
            pick = subset_rng.permutation(len(index))[: self.max_frames]
            index = [index[i] for i in pick]

        self._index = index
        self._estimated_len = len(index)
        self._cache = _LRU(max_items=self.cache_frames_per_worker)

    def _fetch_frame(self, fidx: int, frame_idx: int):
        cache_key = (fidx, frame_idx)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        out = self._procs[fidx].process_one(self._paths[fidx][frame_idx], frame_idx=frame_idx)
        self._cache.put(cache_key, out)
        return out

    def _random_crop(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = tgt.shape
        if self.patch_mode == "strip":
            x0 = self._rng.randint(0, W - self.patch_w + 1)
            x = np.stack([img1[:, x0:x0 + self.patch_w],
                          img2[:, x0:x0 + self.patch_w]], axis=0).astype(np.float32)
            y = tgt[:, x0:x0 + self.patch_w][None, ...].astype(np.float32)
            return x, y

        y0 = self._rng.randint(0, H - self.patch_h + 1)
        x0 = self._rng.randint(0, W - self.patch_w + 1)
        x = np.stack([img1[y0:y0 + self.patch_h, x0:x0 + self.patch_w],
                      img2[y0:y0 + self.patch_h, x0:x0 + self.patch_w]], axis=0).astype(np.float32)
        y = tgt[y0:y0 + self.patch_h, x0:x0 + self.patch_w][None, ...].astype(np.float32)
        return x, y

    def _random_flips(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._rng.rand() < 0.5:
            x = x[:, :, ::-1]
            y = y[:, :, ::-1]
        if self._rng.rand() < 0.5:
            x = x[:, ::-1, :]
            y = y[:, ::-1, :]
        return x, y

    def _build_meta(self, fidx: int, frame_idx: int) -> dict:
        fs = self.folder_specs[fidx]
        return {
            "folder_idx": fidx,
            "frame_idx": frame_idx,
            "data_folder": fs.data_folder,
            "window_sigma": fs.window_sigma,
            "gap": fs.gap,
            "pixels": fs.pixels,
            "alines": fs.alines,
        }

    def __len__(self):
        self._init_worker_state()
        return self._estimated_len

    def __getitem__(self, idx: int):
        self._init_worker_state()
        entry = self._index[idx]

        if self.full_frame:
            fidx, frame_idx = entry
            out = self._fetch_frame(fidx, frame_idx)
            x = np.stack([out["input_w1"], out["input_w2"]], axis=0).astype(np.float32)
            y = out["target_full"][None, ...].astype(np.float32)
        else:
            fidx, frame_idx, _pr = entry
            out = self._fetch_frame(fidx, frame_idx)
            x, y = self._random_crop(out["input_w1"], out["input_w2"], out["target_full"])
            if self.augment:
                x, y = self._random_flips(x, y)
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

        return torch.from_numpy(x), torch.from_numpy(y), self._build_meta(fidx, frame_idx)


# Backward-compatible aliases
RawBscanPatchDataset = RawBscanDataset
