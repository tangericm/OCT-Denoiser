from dataclasses import dataclass
from typing import List, Any
from torch.utils.data import DataLoader

from data.dataset import RawBscanPatchDataset
from data.full_frame_dataset import RawBscanFullFrameDataset

@dataclass
class RawDataConfig:
    folder_specs: List[Any]   # List[FolderSpec]
    train_frac: float
    patch_h: int
    patch_w: int
    patches_per_frame: int
    batch_size: int
    num_workers: int
    seed: int
    augment: bool = False
    patch_mode: str = "strip"
    cache_frames_per_worker: int = 2

class RawBscanDataModule:
    def __init__(self, cfg: RawDataConfig):
        self.cfg = cfg
        self._train = None
        self._val = None
        self._val_full = None

    def setup(self):
        self._train = RawBscanPatchDataset(
            folder_specs=self.cfg.folder_specs,
            split="train",
            train_frac=self.cfg.train_frac,
            patch_h=self.cfg.patch_h,
            patch_w=self.cfg.patch_w,
            patches_per_frame=self.cfg.patches_per_frame,
            patch_mode=self.cfg.patch_mode,
            seed=self.cfg.seed,
            augment=self.cfg.augment,
            cache_frames_per_worker=self.cfg.cache_frames_per_worker,
        )
        self._val = RawBscanPatchDataset(
            folder_specs=self.cfg.folder_specs,
            split="val",
            train_frac=self.cfg.train_frac,
            patch_h=self.cfg.patch_h,
            patch_w=self.cfg.patch_w,
            patches_per_frame=max(1, self.cfg.patches_per_frame // 2),
            patch_mode=self.cfg.patch_mode,
            seed=self.cfg.seed + 999,
            cache_frames_per_worker=1,
        )
        self._val_full = RawBscanFullFrameDataset(
            folder_specs=self.cfg.folder_specs,
            split="val",
            train_frac=self.cfg.train_frac,
            max_frames=1,
            seed=self.cfg.seed + 2024,
            cache_frames_per_worker=1,
        )

    def train_loader(self):
        return DataLoader(
            self._train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
            collate_fn=self._collate,
        )

    def val_loader(self):
        return DataLoader(
            self._val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=max(0, self.cfg.num_workers // 2),
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
            collate_fn=self._collate,
        )

    def val_full_loader(self):
        return DataLoader(
            self._val_full,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch):
        # batch entries: (x, y, meta)
        xs, ys, metas = zip(*batch)
        import torch
        return torch.stack(xs, 0), torch.stack(ys, 0), metas
