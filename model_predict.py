"""
model_predict.py — OCT Denoiser standalone inference script.

Edit the USER CONFIGURATION section below to point at your checkpoint and
dataset, then run:
    python model_predict.py

Outputs are written to <outdir>/:
  pred_*.tiff      — denoised B-scan stack
  gt_*.tiff        — full-bandwidth target stack
  w1_*.tiff        — window-1 input stack
  w2_*.tiff        — window-2 input stack
  snr_per_frame_*.csv  — per-frame SNR / CNR metrics
  snr_rois_frame0_*.png — ROI overlay on first frame
"""

from configs.default import FolderSpec, TrainConfig
from engine.infer import predict_from_config


# ===========================================================================
# USER CONFIGURATION — edit this section
# ===========================================================================

# Path to the checkpoint produced by model_train.py
CHECKPOINT = r"runs\OCT-Denoiser\<timestamp>\checkpoints\best.pt"

# Where to write output files
OUTDIR = r"runs\OCT-Denoiser\<timestamp>\predictions"

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


def main():
    predict_from_config(
        cfg=cfg,
        folder_spec=folder_spec,
        ckpt_path=CHECKPOINT,
        outdir=OUTDIR + "\\" + folder_spec.data_folder,
    )


if __name__ == "__main__":
    main()
