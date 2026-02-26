"""High-level API for OCT post-processing (registration).

Called from ``engine/infer.py`` after prediction completes.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np

from .preproc import prepare_for_registration
from .registration import (
    RegistrationConfig,
    apply_transforms_to_stack,
    register_stack,
    FrameResult,
)
from .reporting import print_summary, save_csv_report, save_json_report

logger = logging.getLogger(__name__)


def postprocess_stacks(
    preds: np.ndarray,
    gts: Optional[np.ndarray],
    outdir: str,
    param_suffix: str,
    tiff_dtype: str = "uint16",
    also_save_float32: bool = False,
    *,
    do_register: bool = True,
    reg_cfg: Optional[RegistrationConfig] = None,
    use_clahe: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Register prediction (and optionally GT) stacks.

    Parameters
    ----------
    preds : [F, H, W] float32 prediction stack.
    gts : [F, H, W] float32 ground-truth stack, or None.
    outdir : directory for output TIFFs and reports.
    param_suffix : filename suffix (e.g. ``"disc_s005_g060"``).
    tiff_dtype : output TIFF dtype (``"uint16"``, ``"uint8"``, ``"float32"``).
    also_save_float32 : save an additional float32 copy of predictions.
    do_register : run inter-frame registration.
    reg_cfg : registration settings (defaults used if None).
    use_clahe : apply CLAHE during registration preprocessing.

    Returns
    -------
    (preds_out, gts_out) — post-processed stacks (same shape as input).
    """
    from utils.io_tiff import save_tiff_stack

    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    if reg_cfg is None:
        reg_cfg = RegistrationConfig()

    # ── Registration ──────────────────────────────────────────────────
    if do_register:
        print("[POSTPROCESS] Preprocessing for registration ...")
        prep = prepare_for_registration(preds, use_clahe=use_clahe)

        print("[POSTPROCESS] Registering prediction stack ...")
        preds, reg_results = register_stack(preds, prep, reg_cfg)

        if gts is not None:
            print("[POSTPROCESS] Applying transforms to ground truth ...")
            gts = apply_transforms_to_stack(gts, reg_results, reg_cfg)

        print_summary(reg_results)
        save_csv_report(
            reg_results,
            os.path.join(outdir, f"registration_report_{param_suffix}.csv"),
        )
        save_json_report(
            reg_results,
            os.path.join(outdir, f"registration_report_{param_suffix}.json"),
        )

        save_tiff_stack(
            os.path.join(outdir, f"pred_registered_{param_suffix}.tiff"),
            preds, dtype=tiff_dtype, scale_per_slice=True,
        )
        if gts is not None:
            save_tiff_stack(
                os.path.join(outdir, f"gt_registered_{param_suffix}.tiff"),
                gts, dtype=tiff_dtype, scale_per_slice=True,
            )

        if also_save_float32 and tiff_dtype != "float32":
            save_tiff_stack(
                os.path.join(outdir, f"pred_registered_{param_suffix}_float32.tiff"),
                preds, dtype="float32", scale_per_slice=True,
            )

    elapsed = time.time() - t0
    print(f"[POSTPROCESS] Done ({elapsed:.1f}s)")
    return preds, gts
