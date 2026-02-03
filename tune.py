# tune_optuna.py
from __future__ import annotations

import os
import csv
import time
from dataclasses import dataclass, asdict
from typing import Dict

from engine.train import run_training  # uses run_training(cfg, paths)

# -----------------------------
# Config (mirror the fields run_training uses)
# -----------------------------
@dataclass
class TuneConfig:
    # data
    npz_path: str
    train_frac: float = 0.85
    patch_h: int = 192
    patch_w: int = 320
    patches_per_frame: int = 24
    augment: bool = True
    batch_size: int = 12
    num_workers: int = 4
    seed: int = 42

    # model
    model_name: str = "resunet_pseudo3d"
    base: int = 32

    # optimization
    epochs: int = 30            # short runs for tuning
    lr: float = 3e-04
    weight_decay: float = 8e-05
    amp: bool = True
    grad_clip: float = 1.0

    # loss weights 
    w_charb: float = 0.5
    w_grad: float = 0.1

    # logging/checkpoint cadence
    val_every: int = 5
    save_every: int = 999999    # disable periodic saves during tuning

    # Early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    early_stop_warmup_checks: int = 1

    # misc
    device: str = "cuda"
    experiment_name: str = "optuna_tune"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def make_trial_paths(root: str, trial_number: int) -> Dict[str, str]:
    run_dir = os.path.join(root, f"trial_{trial_number:04d}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    return {"run": run_dir, "checkpoints": ckpt_dir}


def write_results_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    # -----------------------------
    # User edits
    # -----------------------------
    NPZ_PATH = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Denoiser\images\processed\6mm_1024Aline_gapped_dataset.npz"
    TUNE_ROOT = os.path.join(r"runs\optuna", "optuna_tune_" + time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(TUNE_ROOT)

    # Lazy import so script still runs without optuna installed
    try:
        import optuna
    except ImportError as e:
        raise SystemExit("pip install optuna") from e

    # Store per-trial summary
    results_rows: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        # Sample hyperparameters (conservative ranges to start)
        cfg = TuneConfig(npz_path=NPZ_PATH)

        # # Core optimizer params
        # cfg.lr = trial.suggest_float("lr", 4e-5, 4e-4, log=True)
        # cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 9e-5, log=True)

        # Patch params
        # Keep multiples of 32 for tensor cores / conv efficiency
        ph = trial.suggest_categorical("patch_h", [192, 224, 256, 288, 320])
        pw = trial.suggest_categorical("patch_w", [192, 224, 256, 288, 320])
        cfg.patch_h, cfg.patch_w = int(ph), int(pw)
        cfg.patches_per_frame = trial.suggest_categorical("patches_per_frame", [16, 24, 32, 48])

        # Batch size (optional: comment out if you often OOM)
        cfg.batch_size = trial.suggest_categorical("batch_size", [4, 8, 12, 16])

        # Loss balance
        cfg.w_charb = trial.suggest_float("w_charb", 0.01, 0.8)
        cfg.w_grad = trial.suggest_float("w_grad", 0.01, 0.8)

        # Model capacity (optional knob)
        cfg.base = trial.suggest_categorical("base", [24, 32, 40])

        # Reduce variance for fair comparisons
        cfg.seed = 42
        cfg.experiment_name = f"trial_{trial.number:04d}"

        paths = make_trial_paths(TUNE_ROOT, trial.number)

        out = run_training(cfg, paths)
        history = out["history"]

        # Grab the last reported val loss (evaluate runs every val_every)
        val_entries = history.get("val_loss", [])
        if not val_entries:
            # if val never ran for some reason, penalize
            return 1e9
        
        # Since run_training() already early-stops, just score by the best val seen.
        best_entry = min(val_entries, key=lambda d: float(d["loss"]))
        best_val = float(best_entry["loss"])
        best_epoch = int(best_entry.get("epoch", -1))

        # Pull early-stop metadata saved by train.py (if present)
        es_meta = history.get("early_stop", {})
        stop_epoch = int(es_meta.get("stop_epoch", -1)) if isinstance(es_meta, dict) else -1

        trial.set_user_attr("best_val_loss", best_val)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("stop_epoch", stop_epoch)

        final_val = best_val

        # Save a small trial summary row
        row = {
            "trial": trial.number,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            **{k: v for k, v in asdict(cfg).items() if k != "npz_path"},
            "run_dir": paths["run"],
        }
        results_rows.append(row)
        write_results_csv(os.path.join(TUNE_ROOT, "study_results.csv"), results_rows)

        return final_val

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("\nBest trial:")
    print(study.best_trial.number)
    print(study.best_trial.value)
    print(study.best_trial.params)
    print(f"Results saved in: {TUNE_ROOT}")


if __name__ == "__main__":
    main()
