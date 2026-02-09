# tune.py
from __future__ import annotations

import os
import csv
import time
import copy
import optuna
from dataclasses import asdict
from typing import Dict, List

from configs.default import TrainConfig, FolderSpec
from engine.train import run_training  # uses run_training(cfg, paths)
from utils.seed import seed_all


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def make_trial_paths(root: str, trial_number: int) -> Dict[str, str]:
    run_dir = os.path.join(root, f"trial_{trial_number:04d}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    val_dir = os.path.join(run_dir, "val_outputs")
    ensure_dir(ckpt_dir)
    ensure_dir(val_dir)
    return {"run": run_dir, "checkpoints": ckpt_dir, "val_outputs": val_dir}   


def write_results_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _apply_folder_knobs(folder_specs: List[FolderSpec], *, window_sigma: float, gap: float) -> None:
    # Apply the same spectral-window knobs to all folders (common case).
    # If you want per-folder tuning later, sample separate values per folder.
    for fs in folder_specs:
        fs.window_sigma = float(window_sigma)
        fs.gap = float(gap)
        

def main():
    # -----------------------------
    # User edits
    # -----------------------------
    TUNE_ROOT = os.path.join(r"runs\optuna", "optuna_tune_" + time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(TUNE_ROOT)

    # Store per-trial summary
    results_rows: list[dict] = []

    # -----------------------------
    # Base config
    # -----------------------------
    base_cfg = TrainConfig(
        npz_path=None,                 # not used in raw-folder pipeline
        runs_root=TUNE_ROOT,            # not strictly used by run_training, but kept consistent
        experiment_name="optuna",   # overwritten per trial

        folder_specs=[
            FolderSpec(
                root_folder=r"images\Maestro3",
                data_folder="6mm_1024Aline",
                pixels=2048,
                alines=1024,
                crop_depth=(0, 1024),
                dispersion=[1.315892282e-06, 5.459678905e-10],
                window_sigma=0.08,
                gap=0.25,
            ),
        ],

        cache_frames_per_worker=1000,

        device="cuda",
        amp=True,
        deterministic=True,

        # keep tuning runs short-ish
        epochs=30,
        val_every=5,
        save_every=999999,  # effectively disable periodic checkpoint spam during tuning

        # patching
        patch_mode="strip",
        patch_h=288,        # unused when patch_mode="strip" (kept for completeness)
        patch_w=16,
        patches_per_frame=16,

        # model
        model_name="resunet_pseudo3d",
        base=32,

        # optim
        lr=3e-4,
        weight_decay=8e-5,
        grad_clip=1.0,
        augment=True,

        # loss
        w_charb=0.010307111599432855,
        w_grad=0.010163544565911599,

        # loader
        batch_size=12,
        num_workers=4,

        # early stopping
        early_stop_patience=5,
        early_stop_min_delta=1e-4,
        early_stop_warmup_checks=1,
    )

    def objective(trial: "optuna.Trial") -> float:
        # Sample hyperparameters (conservative ranges to start)
        cfg = copy.deepcopy(base_cfg)

        # Spectral-window knobs (the physics knobs you care about)
        window_sigma = trial.suggest_float("window_sigma", 0.01, 0.16)
        gap = trial.suggest_float("gap", -0.10, 0.50)
        _apply_folder_knobs(cfg.folder_specs, window_sigma=window_sigma, gap=gap)

        # pw = trial.suggest_categorical("patch_w", [8, 16, 32, 48, 64, 80, 96, 128, 160])
        # cfg.patch_w = int(pw)
        # cfg.patches_per_frame = trial.suggest_categorical("patches_per_frame", [16, 24, 32, 48, 64, 80])

        # # Batch size (optional: comment out if you often OOM)
        # cfg.batch_size = trial.suggest_categorical("batch_size", [4, 8, 12, 16])

        # # Loss balance
        # cfg.w_charb = trial.suggest_float("w_charb", 0.01, 0.8)
        # cfg.w_grad = trial.suggest_float("w_grad", 0.01, 0.8)

        # # Optimizer
        # cfg.lr = float(trial.suggest_float("lr", 4e-5, 6e-4, log=True))
        # cfg.weight_decay = float(trial.suggest_float("weight_decay", 1e-6, 2e-4, log=True))

        # Reduce variance for fair comparisons
        cfg.seed = 42
        cfg.experiment_name = f"trial_{trial.number:04d}"

        # Important: seed before loaders/workers get created
        seed_all(cfg.seed, deterministic=cfg.deterministic)

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
        last_entry = val_entries[-1]

        best_snr_pred = best_entry.get("snr_pred")
        best_snr_gt = best_entry.get("snr_gt")
        best_cnr_pred = best_entry.get("cnr_pred")
        best_cnr_gt = best_entry.get("cnr_gt")

        last_snr_pred = last_entry.get("snr_pred")
        last_snr_gt = last_entry.get("snr_gt")
        last_cnr_pred = last_entry.get("cnr_pred")
        last_cnr_gt = last_entry.get("cnr_gt")

        # Pull early-stop metadata saved by train.py (if present)
        es_meta = history.get("early_stop", {})
        stop_epoch = int(es_meta.get("stop_epoch", -1)) if isinstance(es_meta, dict) else -1

        trial.set_user_attr("best_val_loss", best_val)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("stop_epoch", stop_epoch)
        trial.set_user_attr("best_snr_pred", best_snr_pred)
        trial.set_user_attr("best_snr_gt", best_snr_gt)
        trial.set_user_attr("best_cnr_pred", best_cnr_pred)
        trial.set_user_attr("best_cnr_gt", best_cnr_gt)
        trial.set_user_attr("last_snr_pred", last_snr_pred)
        trial.set_user_attr("last_snr_gt", last_snr_gt)
        trial.set_user_attr("last_cnr_pred", last_cnr_pred)
        trial.set_user_attr("last_cnr_gt", last_cnr_gt)

        final_val = best_val

        # Write a row summary
        row = {
            "trial": trial.number,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "best_snr_pred": best_snr_pred,
            "best_snr_gt": best_snr_gt,
            "best_cnr_pred": best_cnr_pred,
            "best_cnr_gt": best_cnr_gt,
            "last_snr_pred": last_snr_pred,
            "last_snr_gt": last_snr_gt,
            "last_cnr_pred": last_cnr_pred,
            "last_cnr_gt": last_cnr_gt,
            "run_dir": paths["run"],
            # record the key trial params explicitly
            "window_sigma": window_sigma,
            "gap": gap,
            "patch_w": cfg.patch_w,
            "patches_per_frame": cfg.patches_per_frame,
            "base": cfg.base,
            "w_charb": cfg.w_charb,
            "w_grad": cfg.w_grad,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
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
