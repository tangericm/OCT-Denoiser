from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import Config as PreprocessConfig, BscanProcessor


class _LRU:
    def __init__(self, max_items: int = 4):
        self.max_items = max_items
        self.d = OrderedDict()

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


class RawBscanFullFrameDataset(Dataset):
    """
    Returns (x, y, meta) where:
      x: [2, H, W]
      y: [1, H, W]
    """

    def __init__(
        self,
        folder_specs: List[Any],  # FolderSpec
        split: str,  # "train" or "val"
        train_frac: float,
        max_frames: int | None,
        seed: int = 0,
        cache_frames_per_worker: int = 2,
    ):
        self.folder_specs = folder_specs
        self.split = split
        self.train_frac = train_frac
        self.max_frames = max_frames
        self.seed = seed
        self.cache_frames_per_worker = cache_frames_per_worker

        self._procs = None
        self._paths = None
        self._index = None
        self._cache = None
        self._estimated_len = 1

    def _init_worker_state(self):
        if self._procs is not None:
            return

        base_seed = self.seed

        self._procs = []
        self._paths = []
        for fs in self.folder_specs:
            cfg = PreprocessConfig(
                pixels=fs.pixels,
                alines=fs.alines,
                data_folder=fs.data_folder,
                do_dc_subtract=fs.do_dc_subtract,
                window_type=fs.window_type,
                use_log=fs.use_log,
                log_eps=fs.log_eps,
                crop_depth=fs.crop_depth,
                apply_fftshift_depth=fs.apply_fftshift_depth,
                window_sigma=fs.window_sigma,
                gap=fs.gap,
                dispersion=fs.dispersion,
            )
            proc = BscanProcessor(fs.root_folder, cfg)
            self._procs.append(proc)
            self._paths.append(proc.bscan_paths)

        rng = np.random.RandomState(base_seed)
        index = []
        for fidx, paths in enumerate(self._paths):
            n = len(paths)
            order = np.arange(n)
            rng.shuffle(order)
            n_train = int(round(self.train_frac * n))
            if self.split == "train":
                chosen = order[:n_train]
            else:
                chosen = order[n_train:]
            for frame_idx in chosen:
                index.append((fidx, int(frame_idx)))

        if self.max_frames is not None and len(index) > self.max_frames:
            subset_rng = np.random.RandomState(base_seed + 12345)
            pick = subset_rng.permutation(len(index))[: self.max_frames]
            index = [index[i] for i in pick]

        self._index = index
        self._estimated_len = len(self._index)
        self._cache = _LRU(max_items=self.cache_frames_per_worker)

    def __len__(self):
        self._init_worker_state()
        return self._estimated_len

    def __getitem__(self, idx: int):
        self._init_worker_state()
        fidx, frame_idx = self._index[idx]

        cache_key = (fidx, frame_idx)
        cached = self._cache.get(cache_key)
        if cached is None:
            out = self._procs[fidx].process_one(self._paths[fidx][frame_idx], frame_idx=frame_idx)
            self._cache.put(cache_key, out)
        else:
            out = cached

        x = np.stack([out["input_w1"], out["input_w2"]], axis=0).astype(np.float32)
        y = out["target_full"][None, ...].astype(np.float32)

        meta = {
            "folder_idx": fidx,
            "frame_idx": frame_idx,
            "data_folder": self.folder_specs[fidx].data_folder,
            "window_sigma": self.folder_specs[fidx].window_sigma,
            "gap": self.folder_specs[fidx].gap,
            "pixels": self.folder_specs[fidx].pixels,
            "alines": self.folder_specs[fidx].alines,
        }

        return torch.from_numpy(x), torch.from_numpy(y), meta
