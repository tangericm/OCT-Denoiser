from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.dataset import RawBscanDataset
from data.avg_targets import resolve_avg_cache_dir


class RawBscanDataModule:
    """Wraps train/val/val-full data loaders. Accepts a TrainConfig directly."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._train = None
        self._val = None
        self._val_full = None

    def setup(self):
        c = self.cfg
        mode_kwargs = dict(
            input_mode=getattr(c, "input_mode", "bandgap"),
            target_mode=getattr(c, "target_mode", "fullband"),
            avg_leave_one_out=getattr(c, "avg_leave_one_out", True),
            avg_cache_dir=resolve_avg_cache_dir(c),
        )
        self._train = RawBscanDataset(
            folder_specs=c.folder_specs,
            split="train",
            train_frac=c.train_frac,
            patch_h=c.patch_h,
            patch_w=c.patch_w,
            patches_per_frame=c.patches_per_frame,
            patch_mode=c.patch_mode,
            seed=c.seed,
            augment=c.augment,
            cache_frames_per_worker=c.cache_frames_per_worker,
            **mode_kwargs,
        )
        self._val = RawBscanDataset(
            folder_specs=c.folder_specs,
            split="val",
            train_frac=c.train_frac,
            patch_h=c.patch_h,
            patch_w=c.patch_w,
            patches_per_frame=max(1, c.patches_per_frame // 2),
            patch_mode=c.patch_mode,
            seed=c.seed,
            cache_frames_per_worker=c.cache_frames_per_worker,
            **mode_kwargs,
        )
        self._val_full = RawBscanDataset(
            folder_specs=c.folder_specs,
            split="val",
            train_frac=c.train_frac,
            seed=c.seed,
            cache_frames_per_worker=c.cache_frames_per_worker,
            full_frame=True,
            **mode_kwargs,
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
