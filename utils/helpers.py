from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    if is_dataclass(obj):
        payload = asdict(obj)
    else:
        payload = obj
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def nanmean(lst) -> float:
    """Return nanmean of a list of floats; returns nan if empty."""
    if not lst:
        return float("nan")
    a = np.asarray(lst, dtype=np.float64)
    return float(np.nanmean(np.where(np.isfinite(a), a, np.nan)))


def seed_all(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
