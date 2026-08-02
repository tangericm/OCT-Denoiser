from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def seed_all(seed: int, deterministic: bool = False) -> None:
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
