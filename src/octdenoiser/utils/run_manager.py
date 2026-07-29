from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    return re.sub(r"-+", "-", value).strip("-") or "oct-denoiser"


def make_run_dir(runs_root: str, experiment: str) -> Path:
    parent = Path(runs_root) / _slugify(experiment)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / timestamp
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{timestamp}_{suffix:02d}"
        suffix += 1
    (candidate / "checkpoints").mkdir(parents=True)
    return candidate
