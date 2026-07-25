from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopping:
    """
    Early stop on a monitored scalar (typically val loss).
    Counts *validation checks* (not epochs), which is what you want when val_every > 1.

    mode:
      - "min": lower is better (val loss)
      - "max": higher is better (accuracy/SSIM)
    """
    patience: int = 5            # number of val checks without improvement
    min_delta: float = 0.0        # required improvement
    mode: str = "min"             # "min" or "max"
    warmup: int = 0               # ignore stopping for first N val checks

    best: Optional[float] = None
    num_bad: int = 0
    num_checks: int = 0
    best_epoch: Optional[int] = None
    stop_epoch: int = -1           # epoch at which stopping was triggered

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        elif self.mode == "max":
            return value > (self.best + self.min_delta)
        else:
            raise ValueError(f"Unknown mode={self.mode}")

    def update(self, value: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        self.num_checks += 1

        if self._is_improvement(value):
            self.best = value
            self.num_bad = 0
            self.best_epoch = epoch
        else:
            self.num_bad += 1

        if self.num_checks <= self.warmup:
            return False
        
        if self.num_bad >= self.patience:
            self.stop_epoch = epoch
            return True

        return False
