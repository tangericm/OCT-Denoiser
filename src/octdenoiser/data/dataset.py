from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from octdenoiser.preprocess import BscanProcessor


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

    Returns (x, y, meta). The target y is always 1-channel; the input channel
    count C depends on the mode, so it is NOT always 2:

      input_mode="fullband"                       -> C = 1
      input_mode="bandgap", n_sub_windows=0       -> C = 2          (w1, w2)
      input_mode="bandgap", n_sub_windows=k > 0   -> C = 2 + 2*k    (+ sub-windows)

    When full_frame=False (default):
      x: [C, patch_h, patch_w], y: [1, patch_h, patch_w]
    When full_frame=True:
      x: [C, H, W], y: [1, H, W], and meta carries target_mu / target_sd /
      log_eps so metrics can be computed in physical intensity.
    """

    def __init__(
        self,
        folder_specs: list[Any],
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

        self._procs: list | None = None
        self._paths: list | None = None
        self._index: list[tuple] | None = None
        self._cache: Any = None
        self._rng: Any = None
        self._swap_views = False
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
            from octdenoiser.data.avg_targets import load_folder_sum

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
        """Input channels per mode.

        fullband      -> [full-band image]                       (1 channel)
        complementary -> [one sub-band]; its complement is the target (1 channel)
        bandgap       -> [w1, w2, *sub-windows]                   (2 + 2*n_sub)
        """
        if self.input_mode == "fullband":
            return [out["target_full"]]
        if self.target_mode == "complementary":
            return self._gather_complementary_input(out)
        return self._gather_inputs(out)

    def _make_target(self, out: dict, fidx: int) -> tuple:
        """Return (target [H,W] float32, target_mu, target_sd) per target_mode."""
        if self.target_mode == "complementary":
            # The complementary sub-band, not the full band.
            #
            # Targeting the full band leaks: it CONTAINS both sub-bands, so the
            # target's noise is correlated with the input's and the network is
            # rewarded for passing noise through. Measured on real data, a
            # sub-band input against its full-band target leaves speckle
            # correlated at +0.138, against +0.003 for the complementary view --
            # roughly 40x worse. Noise2Noise needs that near zero.
            key = "input_w1" if self._swap_views else "input_w2"
            return out[key], float(out[f"{key}_mu"]), float(out[f"{key}_sd"])

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
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def _random_flips(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def _gather_complementary_input(self, out: dict) -> list:
        """Single sub-band input; its complement becomes the target.

        Which of the two is the input is randomised per sample so the network
        cannot learn a directional w1->w2 bias. The two bands sit at different k
        and scattering is wavelength-dependent, so their expected signals do
        differ (measured 0.269 relative mismatch in mean depth profile);
        alternating the direction keeps that bias symmetric rather than letting
        it accumulate in one direction.
        """
        return [out["input_w2" if self._swap_views else "input_w1"]]

    def __getitem__(self, idx: int):
        self._init_worker_state()
        entry = self._index[idx]

        # Which sub-band is the input this sample (complementary mode only).
        # Held fixed for full-frame validation so the metric is deterministic.
        self._swap_views = (
            False if self.full_frame else bool(self._rng.randint(0, 2))
        )

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
