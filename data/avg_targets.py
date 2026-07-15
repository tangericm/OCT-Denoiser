"""Temporal-average target support for the mirror baseline study.

For static data (e.g. a mirror), the per-pixel temporal mean of the linear
full-band magnitude is a clean pseudo-reference. We precompute the per-folder
SUM of linear magnitude across all frames once (cached to disk); the training
dataset then forms a leave-one-out average per frame as:

    loo_i = (sum_mag - mag_i) / (N - 1)

Averaging is done in LINEAR magnitude (never in log/z-score domain), then the
result is log-compressed + z-normalized to match the target domain.
"""
from __future__ import annotations

import hashlib
import os
from typing import Tuple

import numpy as np


def _folder_key(fs) -> str:
    data_dir = os.path.abspath(os.path.join(fs.root_folder, fs.data_folder))
    sig = f"{data_dir}|crop={tuple(fs.crop_depth)}|dc={fs.do_dc_subtract}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
    base = os.path.basename(fs.data_folder.rstrip("/\\")) or "folder"
    return f"{base}_{h}"


def resolve_avg_cache_dir(cfg) -> str:
    """Resolve cfg.avg_cache_dir; relative paths are placed under cfg.runs_root."""
    d = getattr(cfg, "avg_cache_dir", "avg_cache")
    if os.path.isabs(d):
        return d
    return os.path.join(cfg.runs_root, d)


def folder_cache_path(cache_dir: str, fs) -> str:
    return os.path.join(cache_dir, f"summag_{_folder_key(fs)}.npz")


def build_folder_sum(fs) -> Tuple[np.ndarray, int]:
    """Sum linear full-band magnitude over all frames in the folder.

    Returns (sum_mag [H,W] float64, N frames).
    """
    from preprocess import BscanProcessor

    proc = BscanProcessor(fs)
    paths = proc.bscan_paths
    sum_mag = None
    for i, p in enumerate(paths):
        out = proc.process_one(p, frame_idx=i, need_linear_full=True)
        mag = out["target_full_linear"].astype(np.float64, copy=False)
        if sum_mag is None:
            sum_mag = np.zeros_like(mag, dtype=np.float64)
        sum_mag += mag
    if sum_mag is None:
        raise RuntimeError(f"No frames found for {fs.data_folder}")
    return sum_mag, len(paths)


def ensure_folder_averages(folder_specs, cache_dir: str, verbose: bool = True) -> None:
    """Build and cache the linear-magnitude sum for each folder if missing.

    Must be called ONCE from the main process (not per DataLoader worker).
    """
    os.makedirs(cache_dir, exist_ok=True)
    for fs in folder_specs:
        path = folder_cache_path(cache_dir, fs)
        if os.path.exists(path):
            if verbose:
                print(f"[avg] cache hit: {path}")
            continue
        if verbose:
            print(f"[avg] building linear-magnitude sum for {fs.data_folder} ...")
        sum_mag, n = build_folder_sum(fs)
        np.savez(path, sum_mag=sum_mag, n=np.int64(n))
        if verbose:
            print(f"[avg] cached {path}  N={n}  shape={sum_mag.shape}")


def load_folder_sum(cache_dir: str, fs) -> Tuple[np.ndarray, int]:
    path = folder_cache_path(cache_dir, fs)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Average-target cache missing for {fs.data_folder}: {path}. "
            f"Call ensure_folder_averages() in the main process first."
        )
    d = np.load(path)
    return d["sum_mag"].astype(np.float64, copy=False), int(d["n"])
