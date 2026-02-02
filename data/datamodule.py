from __future__ import annotations

from dataclasses import dataclass
from torch.utils.data import DataLoader
from .dataset_npz import NPZDataset

@dataclass
class DataConfig:
    npz_path: str
    train_frac: float
    patch_h: int
    patch_w: int
    patches_per_frame: int
    augment: bool
    batch_size: int
    num_workers: int
    seed: int

class NPZDataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None

    def setup(self):
        self.train_ds = NPZDataset(
            self.cfg.npz_path,
            split="train",
            train_frac=self.cfg.train_frac,
            patch_size=(self.cfg.patch_h, self.cfg.patch_w),
            patches_per_frame=self.cfg.patches_per_frame,
            augment=self.cfg.augment,
            seed=self.cfg.seed,
        )
        self.val_ds = NPZDataset(
            self.cfg.npz_path,
            split="val",
            train_frac=self.cfg.train_frac,
            patch_size=(self.cfg.patch_h, self.cfg.patch_w),
            patches_per_frame=max(4, self.cfg.patches_per_frame // 4),
            augment=False,
            seed=self.cfg.seed,
        )

    def train_loader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.cfg.num_workers > 0),
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
        )

    def val_loader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.cfg.num_workers > 0),
            prefetch_factor=4 if self.cfg.num_workers > 0 else None,
        )
