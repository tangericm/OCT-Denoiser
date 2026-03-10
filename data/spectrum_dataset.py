"""Dataset for spectrum-domain training.

Returns complex spectra (as real/imag channels) for individual A-lines or
full frames, suitable for training 1D spectrum denoising networks.
"""
from __future__ import annotations

import glob
import os
from typing import List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

from data.dataset import _LRU, _to_torch_float32
from preprocess import BscanProcessor


class SpectrumDataset(Dataset):
    """
    Dataset returning complex spectra as real/imag channels.

    When full_frame=False:
      Returns (x, y, meta) with x: [6, pixels], y: [2, pixels]
      Each sample is a single A-line spectrum.
    When full_frame=True:
      Returns (x, y, meta) with x: [6, pixels, alines], y: [2, pixels, alines]
    """

    def __init__(
        self,
        folder_specs: List[Any],
        split: str,
        train_frac: float,
        patches_per_frame: int = 256,
        seed: int = 42,
        cache_frames_per_worker: int = 100,
        full_frame: bool = False,
        max_frames: int | None = None,
        patch_w: int = 1,
    ):
        self.folder_specs = folder_specs
        self.split = split
        self.train_frac = train_frac
        self.patches_per_frame = patches_per_frame
        self.seed = seed
        self.cache_frames_per_worker = cache_frames_per_worker
        self.full_frame = full_frame
        self.max_frames = max_frames
        self.patch_w = patch_w

        self._procs = None
        self._paths = None
        self._index = None
        self._cache = None
        self._rng = None
        self._estimated_len = 1

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

        self._procs = []
        self._paths = []
        for fs in self.folder_specs:
            proc = BscanProcessor(fs)
            self._procs.append(proc)
            self._paths.append(proc.bscan_paths)

        self._build_index()
        self._rng = np.random.RandomState(self.seed + wid)
        self._cache = _LRU(max_items=self.cache_frames_per_worker)

    def _fetch_frame(self, fidx: int, frame_idx: int):
        cache_key = (fidx, frame_idx)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        out = self._procs[fidx].process_one_spectrum(
            self._paths[fidx][frame_idx], frame_idx=frame_idx
        )
        self._cache.put(cache_key, out)
        return out

    def _build_meta(self, fidx: int, frame_idx: int, out: dict) -> dict:
        fs = self.folder_specs[fidx]
        return {
            "folder_idx": fidx,
            "frame_idx": frame_idx,
            "data_folder": fs.data_folder,
            "norm_factor": float(out["norm_factor"]),
            "pixels": fs.pixels,
            "alines": fs.alines,
            "crop_depth": fs.crop_depth,
            "use_log": fs.use_log,
            "log_eps": float(fs.log_eps),
            "apply_fftshift_depth": fs.apply_fftshift_depth,
        }

    def __len__(self):
        self._build_index()
        return self._estimated_len

    @staticmethod
    def _spec_to_channels(spec_w1, spec_w2, w1_mask, w2_mask):
        """Convert spectra + masks to [6, ...] channels."""
        mask_shape = (spec_w1.shape[0],) + (1,) * (spec_w1.ndim - 1)
        w1 = np.broadcast_to(np.asarray(w1_mask, dtype=np.float32).reshape(mask_shape), spec_w1.shape)
        w2 = np.broadcast_to(np.asarray(w2_mask, dtype=np.float32).reshape(mask_shape), spec_w2.shape)
        return np.stack([
            spec_w1.real, spec_w1.imag,
            spec_w2.real, spec_w2.imag,
            w1, w2,
        ], axis=0).astype(np.float32)

    @staticmethod
    def _spec_to_target(spec_full):
        """Convert complex spectrum to [2, ...] real/imag channels."""
        return np.stack([
            spec_full.real, spec_full.imag,
        ], axis=0).astype(np.float32)

    def __getitem__(self, idx: int):
        self._init_worker_state()
        entry = self._index[idx]

        if self.full_frame:
            fidx, frame_idx = entry
            out = self._fetch_frame(fidx, frame_idx)
            x = self._spec_to_channels(out["spec_w1"], out["spec_w2"], out["w1_mask"], out["w2_mask"])  # [6, pixels, alines]
            y = self._spec_to_target(out["spec_full"])  # [2, pixels, alines]
        else:
            fidx, frame_idx, _pr = entry
            out = self._fetch_frame(fidx, frame_idx)
            alines = out["spec_full"].shape[1]

            if self.patch_w > 1:
                # Sample a random contiguous block of patch_w A-lines
                max_start = max(0, alines - self.patch_w)
                j = self._rng.randint(0, max_start + 1)
                sl = slice(j, j + self.patch_w)
                x = self._spec_to_channels(out["spec_w1"][:, sl], out["spec_w2"][:, sl], out["w1_mask"], out["w2_mask"])  # [6, pixels, patch_w]
                y = self._spec_to_target(out["spec_full"][:, sl])  # [2, pixels, patch_w]
            else:
                # Single random A-line
                j = self._rng.randint(0, alines)
                x = self._spec_to_channels(out["spec_w1"][:, j:j+1], out["spec_w2"][:, j:j+1], out["w1_mask"], out["w2_mask"])  # [6, pixels, 1]
                y = self._spec_to_target(out["spec_full"][:, j:j+1])  # [2, pixels, 1]
                x = x[:, :, 0]  # [6, pixels]
                y = y[:, :, 0]  # [2, pixels]

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        meta = self._build_meta(fidx, frame_idx, out)
        return _to_torch_float32(x), _to_torch_float32(y), meta


class SpectrumDataModule:
    """DataLoader factory for spectrum training."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._train = None
        self._val = None
        self._val_full = None

    def setup(self):
        c = self.cfg
        self._train = SpectrumDataset(
            folder_specs=c.folder_specs,
            split="train",
            train_frac=c.train_frac,
            patches_per_frame=c.patches_per_frame,
            seed=c.seed,
            cache_frames_per_worker=c.cache_frames_per_worker,
            patch_w=c.patch_w,
        )
        self._val = SpectrumDataset(
            folder_specs=c.folder_specs,
            split="val",
            train_frac=c.train_frac,
            patches_per_frame=max(1, c.patches_per_frame // 2),
            seed=c.seed,
            cache_frames_per_worker=c.cache_frames_per_worker,
            patch_w=c.patch_w,
        )
        self._val_full = SpectrumDataset(
            folder_specs=c.folder_specs,
            split="val",
            train_frac=c.train_frac,
            seed=c.seed,
            cache_frames_per_worker=c.cache_frames_per_worker,
            full_frame=True,
        )

    def train_loader(self):
        nw = self.cfg.num_workers
        return DataLoader(
            self._train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
            collate_fn=_collate,
        )

    def val_loader(self):
        nw = max(0, self.cfg.num_workers // 2)
        return DataLoader(
            self._val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
            collate_fn=_collate,
        )

    def val_full_loader(self):
        return DataLoader(
            self._val_full,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=_collate,
        )


def _collate(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), metas
