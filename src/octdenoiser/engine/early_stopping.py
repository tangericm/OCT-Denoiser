from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    best: float | None = None
    bad_checks: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` for a validation loss."""

        improved = self.best is None or value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        should_stop = self.patience > 0 and self.bad_checks >= self.patience
        return improved, should_stop
