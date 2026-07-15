from __future__ import annotations

from collections import OrderedDict
import glob
import os
from typing import List, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from preprocess import BscanProcessor




def _to_torch_float32(arr: np.ndarray) -> torch.Tensor:
    """Convert any numpy-like view to a safe float32 torch tensor.

    Forces a C-order copy so negative/non-contiguous strides from slicing/flips
    cannot propagate into torch.from_numpy.
    """
    return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True, order="C"))

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
        input_mode: str = "bandgap",
        target_mode: str = "fullband",
        avg_leave_one_out: bool = True,
        avg_cache_dir: str | None = None,
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
        self.input_mode = input_mode
        self.target_mode = target_mode
        self.avg_leave_one_out = avg_leave_one_out
        self.avg_cache_dir = avg_cache_dir

        self._procs = None
        self._paths = None
        self._index = None
        self._cache = None
        self._rng = None
        self._estimated_len = 1
        self._avg_sum = None
        self._avg_N = None

    def _build_index(self):
        if self._index is not None:
            return

        index_rng = np.random.RandomState(self.seed)
        index = []

        for fidx, fs in enumerate(self.folder_specs):
            data_dir = os.path.join(fs.root_folder, fs.data_folder)
            paths = sorted(glob.glob(os.path.join(data_dir, "bscan*.raw")))
            n = len(paths)
            if n == 0:
                raise FileNotFoundError(f"No bscan*.raw found in {data_dir}")

            order = np.arange(n)
            index_rng.shuffle(order)
            n_train = int(round(self.train_frac * n))
            chosen = order[:n_train] if self.split == "train" else order[n_train:]

            if self.full_frame:
                for frame_idx in chosen:
                    index.append((fidx, int(frame_idx)))
            else:
                if fs.alines < self.patch_w:
                    raise ValueError(f"patch_w={self.patch_w} > alines={fs.alines} for folder={fs.data_folder}")
                z0, z1 = fs.crop_depth
                if (z1 - z0) < self.patch_h:
                    raise ValueError(f"patch_h={self.patch_h} > cropped_depth={z1-z0} for folder={fs.data_folder}")

                for frame_idx in chosen:
                    for pr in range(self.patches_per_frame):
                        index.append((fidx, int(frame_idx), pr))

        if self.full_frame and self.max_frames is not None and len(index) > self.max_frames:
            subset_rng = np.random.RandomState(self.seed)
            pick = subset_rng.permutation(len(index))[: self.max_frames]
            index = [index[i] for i in pick]

        self._index = index
        self._estimated_len = len(index)

    def _init_worker_state(self):
        if self._procs is not None:
            return

        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        aug_seed = self.seed + wid

        self._procs = []
        self._paths = []
        for fs in self.folder_specs:
            proc = BscanProcessor(fs)
            self._procs.append(proc)
            self._paths.append(proc.bscan_paths)

        self._build_index()
        self._rng = np.random.RandomState(aug_seed)
        self._cache = _LRU(max_items=self.cache_frames_per_worker)

        # Load per-folder linear-magnitude sums for temporal-average targets.
        if self.target_mode == "average":
            from data.avg_targets import load_folder_sum

            if not self.avg_cache_dir:
                raise ValueError("target_mode='average' requires avg_cache_dir to be set.")
            self._avg_sum = []
            self._avg_N = []
            for fs in self.folder_specs:
                s, n = load_folder_sum(self.avg_cache_dir, fs)
                self._avg_sum.append(s)
                self._avg_N.append(n)

    def _fetch_frame(self, fidx: int, frame_idx: int):
        cache_key = (fidx, frame_idx)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        need_linear = (self.target_mode == "average")
        # Cap FFT threads to 1 inside a DataLoader worker so N workers don't each
        # spawn an all-core FFT (oversubscription). In-process (no worker) uses all cores.
        fft_workers = 1 if get_worker_info() is not None else -1
        out = self._procs[fidx].process_one(
            self._paths[fidx][frame_idx], frame_idx=frame_idx,
            need_linear_full=need_linear, fft_workers=fft_workers,
        )
        self._cache.put(cache_key, out)
        return out

    def _make_inputs(self, out: dict) -> list:
        """Input channels per input_mode: bandgap [w1,w2(+subs)] or single full-band image."""
        if self.input_mode == "fullband":
            return [out["target_full"]]
        return self._gather_inputs(out)

    def _make_target(self, out: dict, fidx: int) -> tuple:
        """Return (target [H,W] float32, target_mu, target_sd) per target_mode."""
        if self.target_mode != "average":
            return out["target_full"], float(out["target_mu"]), float(out["target_sd"])

        cfg = self._procs[fidx].cfg
        mag_i = out["target_full_linear"].astype(np.float64, copy=False)
        sum_mag = self._avg_sum[fidx]
        n = self._avg_N[fidx]
        if self.avg_leave_one_out and n > 1:
            avg = (sum_mag - mag_i) / (n - 1)
        else:
            avg = sum_mag / max(n, 1)
        t = np.log10(avg + cfg.log_eps) if cfg.use_log else avg
        tmu = float(t.mean())
        tsd = float(t.std()) + 1e-6
        tgt = ((t - tmu) / tsd).astype(np.float32)
        return tgt, tmu, tsd

    def _random_crop(
        self,
        inputs: list,
        tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = tgt.shape
        if self.patch_mode == "strip":
            x0 = self._rng.randint(0, W - self.patch_w + 1)
            x = np.stack([img[:, x0:x0 + self.patch_w] for img in inputs],
                         axis=0).astype(np.float32)
            y = tgt[:, x0:x0 + self.patch_w][None, ...].astype(np.float32)
            return x, y

        y0 = self._rng.randint(0, H - self.patch_h + 1)
        x0 = self._rng.randint(0, W - self.patch_w + 1)
        x = np.stack([img[y0:y0 + self.patch_h, x0:x0 + self.patch_w] for img in inputs],
                     axis=0).astype(np.float32)
        y = tgt[y0:y0 + self.patch_h, x0:x0 + self.patch_w][None, ...].astype(np.float32)
        return x, y

    def _random_flips(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._rng.rand() < 0.5:
            x = x[:, :, ::-1]
            y = y[:, :, ::-1]
        if self._rng.rand() < 0.5:
            x = x[:, ::-1, :]
            y = y[:, ::-1, :]
        return x.copy(order="C"), y.copy(order="C")

    def _build_meta(
        self,
        fidx: int,
        frame_idx: int,
        out: dict | None = None,
        target_mu: float | None = None,
        target_sd: float | None = None,
    ) -> dict:
        fs = self.folder_specs[fidx]
        meta = {
            "folder_idx": fidx,
            "frame_idx": frame_idx,
            "data_folder": fs.data_folder,
            "window_sigma": fs.window_sigma,
            "gap": fs.gap,
            "pixels": fs.pixels,
            "alines": fs.alines,
        }
        if out is not None:
            # target_mu/sd reflect the actual target used (averaged target overrides).
            mu = target_mu if target_mu is not None else out.get("target_mu")
            sd = target_sd if target_sd is not None else out.get("target_sd")
            if mu is not None:
                meta["target_mu"] = float(mu)
            if sd is not None:
                meta["target_sd"] = float(sd)
            if hasattr(self._procs[fidx], "cfg") and hasattr(self._procs[fidx].cfg, "log_eps"):
                meta["log_eps"] = float(self._procs[fidx].cfg.log_eps)
        return meta

    def __len__(self):
        self._build_index()
        return self._estimated_len

    @staticmethod
    def _gather_inputs(out: dict) -> list:
        """Collect all input channels: Level 1 (w1, w2) + optional Level 2 sub-windows."""
        inputs = [out["input_w1"], out["input_w2"]]
        if "input_sub_windows" in out:
            inputs.extend(out["input_sub_windows"])
        return inputs

    def __getitem__(self, idx: int):
        self._init_worker_state()
        entry = self._index[idx]

        if self.full_frame:
            fidx, frame_idx = entry
            out = self._fetch_frame(fidx, frame_idx)
            inputs = self._make_inputs(out)
            tgt, tmu, tsd = self._make_target(out, fidx)
            x = np.stack(inputs, axis=0).astype(np.float32)
            y = tgt[None, ...].astype(np.float32)
        else:
            fidx, frame_idx, _pr = entry
            out = self._fetch_frame(fidx, frame_idx)
            inputs = self._make_inputs(out)
            tgt, tmu, tsd = self._make_target(out, fidx)
            x, y = self._random_crop(inputs, tgt)
            if self.augment:
                x, y = self._random_flips(x, y)
            else:
                x = np.ascontiguousarray(x)
                y = np.ascontiguousarray(y)

        meta_out = out if self.full_frame else None
        return (
            _to_torch_float32(x),
            _to_torch_float32(y),
            self._build_meta(fidx, frame_idx, out=meta_out, target_mu=tmu, target_sd=tsd),
        )
