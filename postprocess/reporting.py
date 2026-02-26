"""QC reporting: CSV/JSON export, summary statistics, and self-test."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .registration import FrameResult, TransformType

logger = logging.getLogger(__name__)


# ── CSV / JSON reports ────────────────────────────────────────────────────
def save_csv_report(
    results: List[FrameResult], path: str | Path
) -> Path:
    """Write per-frame registration metrics to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame", "method", "dx", "dy", "rotation_deg",
        "ncc_before", "ncc_after", "ncc_gain",
        "inlier_count", "inlier_ratio", "fallback_reason",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "frame": r.index,
                "method": r.method.value,
                "dx": f"{r.dx:.4f}",
                "dy": f"{r.dy:.4f}",
                "rotation_deg": f"{r.rotation_deg:.4f}",
                "ncc_before": f"{r.ncc_before:.6f}",
                "ncc_after": f"{r.ncc_after:.6f}",
                "ncc_gain": f"{r.ncc_after - r.ncc_before:.6f}",
                "inlier_count": r.inlier_count,
                "inlier_ratio": f"{r.inlier_ratio:.4f}",
                "fallback_reason": r.fallback_reason,
            })
    logger.info("Saved CSV report: %s  (%d frames)", path, len(results))
    return path


def save_json_report(
    results: List[FrameResult], path: str | Path
) -> Path:
    """Write per-frame registration metrics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for r in results:
        records.append({
            "frame": r.index,
            "method": r.method.value,
            "dx": round(r.dx, 4),
            "dy": round(r.dy, 4),
            "rotation_deg": round(r.rotation_deg, 4),
            "ncc_before": round(r.ncc_before, 6),
            "ncc_after": round(r.ncc_after, 6),
            "ncc_gain": round(r.ncc_after - r.ncc_before, 6),
            "inlier_count": r.inlier_count,
            "inlier_ratio": round(r.inlier_ratio, 4),
            "fallback_reason": r.fallback_reason,
            "matrix": r.matrix.tolist(),
        })

    # Summary stats
    ncc_gains = [r.ncc_after - r.ncc_before for r in results]
    methods = [r.method.value for r in results]
    summary = {
        "total_frames": len(results),
        "mean_ncc_gain": round(float(np.mean(ncc_gains)), 6),
        "median_ncc_gain": round(float(np.median(ncc_gains)), 6),
        "method_counts": {
            m: methods.count(m)
            for m in sorted(set(methods))
        },
        "fallback_count": sum(1 for r in results if r.fallback_reason),
    }

    payload = {"summary": summary, "frames": records}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved JSON report: %s", path)
    return path


def print_summary(results: List[FrameResult]) -> None:
    """Print a human-readable summary of registration results."""
    ncc_gains = [r.ncc_after - r.ncc_before for r in results]
    methods = [r.method.value for r in results]
    fallbacks = sum(1 for r in results if r.fallback_reason)

    print("\n===== Registration Summary =====")
    print(f"  Frames:          {len(results)}")
    print(f"  Mean NCC gain:   {np.mean(ncc_gains):+.6f}")
    print(f"  Median NCC gain: {np.median(ncc_gains):+.6f}")
    print(f"  Min NCC gain:    {np.min(ncc_gains):+.6f}")
    print(f"  Max NCC gain:    {np.max(ncc_gains):+.6f}")
    print(f"  Methods used:    ", end="")
    for m in sorted(set(methods)):
        print(f"{m}={methods.count(m)} ", end="")
    print(f"\n  Fallbacks:       {fallbacks}")
    print("================================\n")


# ── Synthetic self-test ───────────────────────────────────────────────────
def run_self_test(verbose: bool = True) -> bool:
    """Create synthetic transformed copies, verify recovery.

    Returns True if all checks pass.
    """
    from .preproc import prepare_for_registration
    from .registration import (
        RegistrationConfig,
        register_stack,
    )

    rng = np.random.default_rng(42)
    H, W = 256, 256
    ok = True

    # --- Registration test ---
    # Create a synthetic image with structure
    base = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, 32):
        base[y:y+2, :] = 0.8
    for x in range(0, W, 32):
        base[:, x:x+2] = 0.6
    base += rng.normal(0, 0.05, (H, W)).astype(np.float32)
    base = np.clip(base, 0, 1)

    # Known shifts
    true_shifts = [(0, 0), (3.0, -2.0), (-1.5, 4.5), (0.5, 0.5)]
    N = len(true_shifts)
    stack = np.empty((N, H, W), dtype=np.float32)
    for i, (dy, dx) in enumerate(true_shifts):
        M = np.eye(2, 3, dtype=np.float32)
        M[0, 2] = dx
        M[1, 2] = dy
        stack[i] = cv2.warpAffine(base, M, (W, H), borderValue=0.0)
        stack[i] += rng.normal(0, 0.02, (H, W)).astype(np.float32)
        stack[i] = np.clip(stack[i], 0, 1)

    prep = prepare_for_registration(stack, use_clahe=False)
    cfg = RegistrationConfig(ref_index=0, dft_upsample_factor=20)
    registered, results = register_stack(stack, prep, cfg)

    # Check recovered shifts are close to true shifts
    for i, (dy_true, dx_true) in enumerate(true_shifts):
        r = results[i]
        if i == 0:
            continue
        err_dx = abs(r.dx - (-dx_true))
        err_dy = abs(r.dy - (-dy_true))
        if err_dx > 2.0 or err_dy > 2.0:
            if verbose:
                print(
                    f"  WARN: frame {i} shift error "
                    f"dx={err_dx:.2f} dy={err_dy:.2f} (tolerance=2.0px)"
                )
            ok = False
        elif verbose:
            print(
                f"  OK: frame {i} shift error "
                f"dx={err_dx:.2f} dy={err_dy:.2f}"
            )

    status = "PASSED" if ok else "FAILED (see warnings above)"
    if verbose:
        print(f"\nSelf-test: {status}")
    return ok


# Need cv2 for the self-test synthetic warp
import cv2
