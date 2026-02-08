import os
import random
from collections import OrderedDict
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

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

class RawBscanPatchDataset(Dataset):
    """
    Returns (x, y, meta) where:
      x: [2, patch_h, patch_w]
      y: [1, patch_h, patch_w]
    """
    def __init__(
        self,
        folder_specs: List[Any],   # FolderSpec
        split: str,                # "train" or "val"
        train_frac: float,
        patch_h: int,
        patch_w: int,
        patches_per_frame: int,
        patch_mode: str = "strip",
        seed: int = 0,
        cache_frames_per_worker: int = 2,
        preprocess_debug: bool = False,
        preprocess_debug_dir: str | None = None,
    ):
        self.folder_specs = folder_specs
        self.split = split
        self.train_frac = train_frac
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patches_per_frame = patches_per_frame
        self.patch_mode = patch_mode
        self.seed = seed
        self.cache_frames_per_worker = cache_frames_per_worker
        self.preprocess_debug = preprocess_debug
        self.preprocess_debug_dir = preprocess_debug_dir

        # Will be created lazily per worker:
        self._procs = None
        self._paths = None
        self._index = None
        self._cache = None

        self._build_global_index()

    def _build_global_index(self):
        # We only build a lightweight index here; processors are built per-worker later.
        # Index entries: (folder_idx, frame_idx_within_folder, patch_rep_idx)
        # Total length = sum_frames * patches_per_frame (per split).
        rng = random.Random(self.seed)
        self._folder_frame_counts = []

        # We cannot enumerate bscan paths without constructing BscanProcessor, so do it lazily.
        # We'll just store a placeholder here and compute true lengths in _init_worker_state().
        self._estimated_len = 1  # will be fixed at first __len__ call after init

    def _init_worker_state(self):
        if self._procs is not None:
            return

        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        base_seed = self.seed + 1337 * wid

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
                debug_mode=self.preprocess_debug,
            )
            proc = BscanProcessor(fs.root_folder, cfg)
            if self.preprocess_debug and self.preprocess_debug_dir:
                os.makedirs(self.preprocess_debug_dir, exist_ok=True)
                proc._debug_out_dir = self.preprocess_debug_dir
                proc._dataset_name = f"{fs.data_folder}_{self.split}"
            self._procs.append(proc)
            self._paths.append(proc.bscan_paths)

            # Sanity: ensure patch_w fits
            if fs.alines < self.patch_w:
                raise ValueError(f"patch_w={self.patch_w} > alines={fs.alines} for folder={fs.data_folder}")

            # Sanity: ensure patch_h fits cropped depth
            z0, z1 = fs.crop_depth
            if (z1 - z0) < self.patch_h:
                raise ValueError(f"patch_h={self.patch_h} > cropped_depth={z1-z0} for folder={fs.data_folder}")

        # Split per folder deterministically
        rng = np.random.RandomState(base_seed)
        self._index = []
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
                for pr in range(self.patches_per_frame):
                    self._index.append((fidx, int(frame_idx), pr))

        self._estimated_len = len(self._index)
        self._cache = _LRU(max_items=self.cache_frames_per_worker)

    def __len__(self):
        self._init_worker_state()
        return self._estimated_len

    def __getitem__(self, idx: int):
        self._init_worker_state()
        fidx, frame_idx, pr = self._index[idx]

        cache_key = (fidx, frame_idx)
        cached = self._cache.get(cache_key)
        if cached is None:
            out = self._procs[fidx].process_one(self._paths[fidx][frame_idx], frame_idx=frame_idx)
            # out contains float32 arrays [H,W]
            self._cache.put(cache_key, out)
        else:
            out = cached

        img1 = out["input_w1"]
        img2 = out["input_w2"]
        tgt  = out["target_full"]

        H, W = tgt.shape
        if self.patch_mode == "strip":
            # full axial, random lateral strip
            y0 = 0
            patch_h = H
            x0 = np.random.randint(0, W - self.patch_w + 1)

            x = np.stack([
                img1[:, x0:x0+self.patch_w],
                img2[:, x0:x0+self.patch_w],
            ], axis=0).astype(np.float32)

            y = tgt[:, x0:x0+self.patch_w][None, ...].astype(np.float32)

        else:
            # standard 2D patch
            y0 = np.random.randint(0, H - self.patch_h + 1)
            x0 = np.random.randint(0, W - self.patch_w + 1)

            x = np.stack([
                img1[y0:y0+self.patch_h, x0:x0+self.patch_w],
                img2[y0:y0+self.patch_h, x0:x0+self.patch_w],
            ], axis=0).astype(np.float32)

            y = tgt[y0:y0+self.patch_h, x0:x0+self.patch_w][None, ...].astype(np.float32)

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
