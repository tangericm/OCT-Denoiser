"""CLI entrypoint for the OCT post-processing pipeline.

Usage examples::

    # Register + deconvolve predictions, apply same transforms to ground truth
    python -m postprocess.cli \\
        --pred-dir  runs/exp/pred_tiffs \\
        --gt-dir    runs/exp/gt_tiffs \\
        --out-dir   runs/exp/postprocessed \\
        --deconv-method wiener --psf-sigma 1.5

    # Registration only, with ECC refinement
    python -m postprocess.cli \\
        --pred-dir runs/exp/pred_tiffs \\
        --out-dir  runs/exp/registered \\
        --no-deconv --use-ecc

    # Run built-in self-test
    python -m postprocess.cli --self-test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCT post-processing: registration + axial deconvolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O ───────────────────────────────────────────────────────────
    p.add_argument(
        "--pred-dir", type=str, default=None,
        help="Directory of predicted OCT images (TIFF/PNG/NPY).",
    )
    p.add_argument(
        "--gt-dir", type=str, default=None,
        help="Optional directory of ground-truth images to co-register.",
    )
    p.add_argument(
        "--out-dir", type=str, default="postprocessed",
        help="Output directory (default: postprocessed/).",
    )
    p.add_argument(
        "--out-dtype", type=str, default="float32",
        choices=["float32", "uint16", "uint8"],
        help="Output TIFF dtype (default: float32).",
    )

    # ── Registration ──────────────────────────────────────────────────
    g = p.add_argument_group("Registration")
    g.add_argument(
        "--no-register", action="store_true",
        help="Skip registration (deconvolve only).",
    )
    g.add_argument(
        "--ref-strategy", type=str, default="middle",
        choices=["middle", "sharpness"],
        help="Reference frame selection strategy.",
    )
    g.add_argument(
        "--ref-index", type=int, default=None,
        help="Explicit reference frame index (overrides --ref-strategy).",
    )
    g.add_argument(
        "--transform-model", type=str, default="euclidean",
        choices=["euclidean", "affine"],
        help="RANSAC transform model.",
    )
    g.add_argument(
        "--use-ecc", action="store_true",
        help="Enable ECC refinement after feature/DFT registration.",
    )
    g.add_argument(
        "--no-clahe", action="store_true",
        help="Disable CLAHE preprocessing for registration.",
    )
    g.add_argument(
        "--max-translation", type=float, default=100.0,
        help="Max allowed translation in pixels.",
    )
    g.add_argument(
        "--max-rotation", type=float, default=5.0,
        help="Max allowed rotation in degrees.",
    )

    # ── Deconvolution ─────────────────────────────────────────────────
    d = p.add_argument_group("Deconvolution")
    d.add_argument(
        "--no-deconv", action="store_true",
        help="Skip deconvolution (register only).",
    )
    d.add_argument(
        "--deconv-method", type=str, default="wiener",
        choices=["wiener", "richardson_lucy"],
        help="Deconvolution method.",
    )
    d.add_argument(
        "--psf-sigma", type=float, default=1.5,
        help="Gaussian PSF sigma in axial pixels.",
    )
    d.add_argument(
        "--wiener-nsr", type=float, default=0.01,
        help="Wiener noise-to-signal ratio.",
    )
    d.add_argument(
        "--rl-iterations", type=int, default=15,
        help="Richardson-Lucy iteration count.",
    )
    d.add_argument(
        "--rl-tv-lambda", type=float, default=0.002,
        help="Richardson-Lucy TV regularization weight.",
    )
    d.add_argument(
        "--pre-smooth", type=float, default=0.0,
        help="Pre-smoothing sigma (0 = disabled).",
    )
    d.add_argument(
        "--post-smooth", type=float, default=0.0,
        help="Post-smoothing sigma (0 = disabled).",
    )

    # ── Misc ──────────────────────────────────────────────────────────
    p.add_argument(
        "--self-test", action="store_true",
        help="Run synthetic self-test and exit.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Self-test mode ────────────────────────────────────────────────
    if args.self_test:
        from .reporting import run_self_test

        print("Running self-test...")
        ok = run_self_test(verbose=True)
        sys.exit(0 if ok else 1)

    # ── Validate inputs ───────────────────────────────────────────────
    if args.pred_dir is None:
        print("ERROR: --pred-dir is required (unless --self-test).")
        sys.exit(1)

    from .io_utils import load_stack, save_stack
    from .preproc import prepare_for_registration
    from .registration import (
        RegistrationConfig,
        apply_transforms_to_stack,
        register_stack,
    )
    from .deconvolution import DeconvConfig, deconvolve_stack
    from .reporting import (
        print_summary,
        save_csv_report,
        save_json_report,
    )

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Load stacks ───────────────────────────────────────────────────
    print(f"Loading predictions from {args.pred_dir} ...")
    pred_stack, pred_names = load_stack(args.pred_dir)

    gt_stack = None
    if args.gt_dir is not None:
        print(f"Loading ground truth from {args.gt_dir} ...")
        gt_stack, gt_names = load_stack(args.gt_dir)
        if gt_stack.shape != pred_stack.shape:
            print(
                f"WARNING: pred shape {pred_stack.shape} != "
                f"gt shape {gt_stack.shape}; truncating to min."
            )
            n = min(pred_stack.shape[0], gt_stack.shape[0])
            pred_stack = pred_stack[:n]
            gt_stack = gt_stack[:n]

    # ── Registration ──────────────────────────────────────────────────
    reg_results = None
    if not args.no_register:
        print("Preprocessing for registration ...")
        prep = prepare_for_registration(
            pred_stack, use_clahe=not args.no_clahe
        )

        reg_cfg = RegistrationConfig(
            ref_index=args.ref_index,
            ref_strategy=args.ref_strategy,
            transform_model=args.transform_model,
            use_ecc=args.use_ecc,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation,
        )

        print("Registering prediction stack ...")
        pred_stack, reg_results = register_stack(pred_stack, prep, reg_cfg)

        # Apply same transforms to ground truth
        if gt_stack is not None:
            print("Applying transforms to ground truth ...")
            gt_stack = apply_transforms_to_stack(
                gt_stack, reg_results, reg_cfg
            )

        # Reports
        print_summary(reg_results)
        save_csv_report(reg_results, outdir / "registration_report.csv")
        save_json_report(reg_results, outdir / "registration_report.json")

        # Save registered stacks
        save_stack(
            pred_stack, outdir, prefix="pred_registered",
            dtype=args.out_dtype,
        )
        if gt_stack is not None:
            save_stack(
                gt_stack, outdir, prefix="gt_registered",
                dtype=args.out_dtype,
            )

    # ── Deconvolution ─────────────────────────────────────────────────
    if not args.no_deconv:
        deconv_cfg = DeconvConfig(
            method=args.deconv_method,
            psf_sigma=args.psf_sigma,
            wiener_nsr=args.wiener_nsr,
            rl_iterations=args.rl_iterations,
            rl_tv_lambda=args.rl_tv_lambda,
            pre_smooth_sigma=args.pre_smooth,
            post_smooth_sigma=args.post_smooth,
        )

        print(
            f"Deconvolving predictions ({deconv_cfg.method}, "
            f"sigma={deconv_cfg.psf_sigma}) ..."
        )
        pred_deconv = deconvolve_stack(pred_stack, deconv_cfg)
        save_stack(
            pred_deconv, outdir, prefix="pred_deconvolved",
            dtype=args.out_dtype,
        )

        if gt_stack is not None:
            print("Deconvolving ground truth ...")
            gt_deconv = deconvolve_stack(gt_stack, deconv_cfg)
            save_stack(
                gt_deconv, outdir, prefix="gt_deconvolved",
                dtype=args.out_dtype,
            )

    elapsed = time.time() - t0
    print(f"\nDone. Output in {outdir}/  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
