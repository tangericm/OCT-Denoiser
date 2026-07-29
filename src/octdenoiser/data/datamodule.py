from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from octdenoiser.configs.default import TrainConfig

from .dataset import BscanPairDataset, discover_volumes, split_pairs


class BscanDataModule:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.train_dataset: BscanPairDataset | None = None
        self.val_dataset: BscanPairDataset | None = None

    def setup(self) -> None:
        cfg = self.config
        volumes = discover_volumes(cfg.inputs)
        train_pairs, val_pairs = split_pairs(
            volumes,
            pair_offset=cfg.pair_offset,
            group_size=cfg.group_size,
            train_fraction=cfg.train_fraction,
            seed=cfg.seed,
        )
        self.train_dataset = BscanPairDataset(
            train_pairs,
            patch_size=(cfg.patch_height, cfg.patch_width),
            patches_per_pair=cfg.patches_per_pair,
            augment=cfg.augment,
        )
        self.val_dataset = BscanPairDataset(val_pairs, patch_size=None)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting a data loader.")
        cfg = self.config
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=cfg.device != "cpu",
            persistent_workers=cfg.workers > 0,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting a data loader.")
        cfg = self.config
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=cfg.device != "cpu",
            persistent_workers=cfg.workers > 0,
        )
