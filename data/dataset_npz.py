from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple

class NPZDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        *,
        split: str,
        train_frac: float = 0.85,
        patch_size: Optional[Tuple[int, int]] = (128, 128),
        patches_per_frame: int = 16,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "all")

        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"].astype(np.float32)   # [F,2,H,W]
        self.Y = data["Y"].astype(np.float32)   # [F,1,H,W]

        self.F, self.C, self.H, self.W = self.X.shape
        assert self.C == 2, f"Expected X channels=2, got {self.C}"
        assert self.Y.shape[:2] == (self.F, 1), f"Expected Y:[F,1,H,W], got {self.Y.shape}"

        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.augment = augment

        rng = np.random.RandomState(seed)
        all_idx = np.arange(self.F)
        rng.shuffle(all_idx)
        cut = int(np.floor(train_frac * self.F))

        if split == "train":
            self.frame_idx = all_idx[:cut]
        elif split == "val":
            self.frame_idx = all_idx[cut:]
        else:
            self.frame_idx = all_idx

        if self.patch_size is not None and split != "all":
            self.length = len(self.frame_idx) * patches_per_frame
        else:
            self.length = len(self.frame_idx)

    def __len__(self) -> int:
        return self.length

    def _random_crop(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ph, pw = self.patch_size  # type: ignore[misc]
        if self.H <= ph or self.W <= pw:
            return x, y
        top = np.random.randint(0, self.H - ph + 1)
        left = np.random.randint(0, self.W - pw + 1)
        x = x[:, top:top + ph, left:left + pw]
        y = y[:, top:top + ph, left:left + pw]
        return x, y

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        return x, y

    def __getitem__(self, idx: int):
        if self.patch_size is not None and self.length != len(self.frame_idx):
            frame_i = self.frame_idx[idx // self.patches_per_frame]
        else:
            frame_i = self.frame_idx[idx]

        x = self.X[frame_i]  # [2,H,W]
        y = self.Y[frame_i]  # [1,H,W]

        if self.patch_size is not None:
            x, y = self._random_crop(x, y)
        if self.augment:
            x, y = self._augment(x, y)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        return x_t, y_t
