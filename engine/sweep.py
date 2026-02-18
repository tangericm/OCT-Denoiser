"""Grid sweep over spectral window parameters (window_sigma x gap)."""
from __future__ import annotations

import os
import csv
import copy
from dataclasses import asdict
from typing import Any, Dict, List

from engine.train import run_training
from engine.infer import predict_from_config
from utils.run_manager import make_run_dir, setup_run_dirs
from utils.helpers import seed_all, save_json


def _write_sweep_csv(path: str, rows: List[dict]) -> None:
    """Write incremental sweep results CSV."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_sweep(cfg) -> Dict[str, Any]:
    """
    Run a grid sweep over window_sigma x gap combinations.

    Creates a parent directory and one sub-run per (sigma, gap) pair:
        runs/<experiment>/<timestamp>_sweep/s008_g025/
        runs/<experiment>/<timestamp>_sweep/s008_g050/
        ...

    Each sub-run uses the standard training pipeline and produces
    checkpoints, validation outputs, and predictions.

    Returns dict with sweep_root path and per-run results.
    """
    sweep_root = make_run_dir(cfg.runs_root, cfg.experiment_name, suffix="sweep")
    save_json(os.path.join(sweep_root, "sweep_config.json"), asdict(cfg))

    sigmas = cfg.sweep_sigmas
    gaps = cfg.sweep_gaps
    total = len(sigmas) * len(gaps)

    results: List[dict] = []
    run_idx = 0

    for sigma in sigmas:
        for gap in gaps:
            run_idx += 1
            suffix = f"s{int(round(sigma * 100)):03d}_g{int(round(gap * 100)):03d}"
            run_dir = os.path.join(sweep_root, suffix)
            paths = setup_run_dirs(run_dir)

            # Deep copy config and set spectral parameters for this run
            run_cfg = copy.deepcopy(cfg)
            run_cfg.sweep_sigmas = None
            run_cfg.sweep_gaps = None
            for fs in run_cfg.folder_specs:
                fs.window_sigma = float(sigma)
                fs.gap = float(gap)

            print(f"\n{'=' * 60}")
            print(f"[SWEEP {run_idx}/{total}] sigma={sigma:.4f}  gap={gap:.4f}")
            print(f"{'=' * 60}")

            seed_all(run_cfg.seed, deterministic=run_cfg.deterministic)
            result = run_training(run_cfg, paths)

            # Run inference with best checkpoint
            for fs in run_cfg.folder_specs:
                predict_from_config(
                    run_cfg, fs, result["best_ckpt_path"], os.path.join(paths["pred_tiff"], fs.data_folder)
                )

            # Extract best validation entry
            history = result["history"]
            val_entries = history.get("val_loss", [])
            best_entry = (
                min(val_entries, key=lambda d: float(d["loss"]))
                if val_entries
                else {}
            )
            es_meta = history.get("early_stop", {})

            row = {
                "run": run_idx,
                "sigma": sigma,
                "gap": gap,
                "best_val_loss": float(best_entry.get("loss", float("inf"))),
                "best_epoch": int(best_entry.get("epoch", -1)),
                "stop_epoch": (
                    int(es_meta.get("stop_epoch", -1))
                    if isinstance(es_meta, dict)
                    else -1
                ),
                "snr_pred": best_entry.get("snr_pred"),
                "snr_gt": best_entry.get("snr_gt"),
                "cnr_pred": best_entry.get("cnr_pred"),
                "cnr_gt": best_entry.get("cnr_gt"),
                "run_dir": run_dir,
            }
            results.append(row)
            _write_sweep_csv(
                os.path.join(sweep_root, "sweep_results.csv"), results
            )

    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE: {total} runs")
    print(f"Results: {os.path.join(sweep_root, 'sweep_results.csv')}")
    print(f"{'=' * 60}")

    if results:
        best = min(results, key=lambda r: r["best_val_loss"])
        print(
            f"Best: sigma={best['sigma']:.4f} gap={best['gap']:.4f} "
            f"val_loss={best['best_val_loss']:.6f}"
        )

    return {"sweep_root": sweep_root, "results": results}
