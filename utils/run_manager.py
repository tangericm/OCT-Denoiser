import os
import re
from datetime import datetime
from typing import Dict

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def make_run_dir(runs_root: str, experiment: str, suffix: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = slugify(experiment) if experiment else "experiment"
    suf = slugify(suffix) if suffix else ""
    name = f"{ts}_{suf}" if suf else ts

    base = os.path.join(runs_root, exp, name)
    run_dir = base
    k = 2
    while os.path.exists(run_dir):
        run_dir = f"{base}_{k:02d}"
        k += 1

    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def setup_run_dirs(run_dir: str) -> Dict[str, str]:
    paths = {
        "run": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "pred_tiff": os.path.join(run_dir, "predictions_tiff"),
    }
    for p in paths.values():
        ensure_dir(p)
    return paths
