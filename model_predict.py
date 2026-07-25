"""
model_predict.py — OCT Denoiser standalone inference script.

Point it at a checkpoint produced by model_train.py:

    python model_predict.py --checkpoint runs/OCT-Denoiser/20260725_120000/checkpoints/best.pt

If --outdir is omitted it defaults to <run_dir>/predictions, where <run_dir> is
the parent of the checkpoints/ directory.

The model architecture and FolderSpec in the USER CONFIGURATION section below
must match the ones used to train the checkpoint.

Outputs are written to <outdir>/<data_folder>/:
  pred_*.tiff      — denoised B-scan stack
  gt_*.tiff        — full-bandwidth target stack
  w1_*.tiff        — window-1 input stack
  w2_*.tiff        — window-2 input stack
  snr_per_frame_*.csv  — per-frame SNR / CNR metrics
  snr_rois_frame0_*.png — ROI overlay on first frame
"""

import argparse
import os

from configs.default import FolderSpec, TrainConfig
from engine.infer import predict_from_config


# ===========================================================================
# USER CONFIGURATION — edit this section
# ===========================================================================

# Model architecture — must match the checkpoint
cfg = TrainConfig(
    model_name="resunet_pseudo3d_multilevel",
    base=32,
    device="cuda",
    tiff_dtype="uint16",
    also_save_float32=False,
    snr_sig_y0=111,
    snr_sig_y1=600,
    snr_sig_stat="p99.99",
)

# Dataset to run inference on — must match the FolderSpec used during training
folder_spec = FolderSpec(
    root_folder=r"images\Maestro3",
    data_folder="6mm_1024Aline",
    pixels=2048,
    alines=1024,
    crop_depth=(0, 1024),
    window_sigma=0.05,
    gap=0.60,
    gap_offset=0.015,
    n_sub_windows=2,
    sub_window_spread=0.5,
)

# ===========================================================================
# END USER CONFIGURATION
# ===========================================================================


def default_outdir(ckpt_path: str) -> str:
    """Derive <run_dir>/predictions from <run_dir>/checkpoints/best.pt."""
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "checkpoints" else ckpt_dir
    return os.path.join(run_dir, "predictions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="path to best.pt produced by model_train.py")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: <run_dir>/predictions)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    outdir = args.outdir or default_outdir(args.checkpoint)
    predict_from_config(
        cfg=cfg,
        folder_spec=folder_spec,
        ckpt_path=args.checkpoint,
        outdir=os.path.join(outdir, folder_spec.data_folder),
    )


if __name__ == "__main__":
    main()
