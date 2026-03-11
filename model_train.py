"""
model_train.py — OCT Denoiser training entry point.

Edit the USER CONFIGURATION section below to match your dataset and hardware,
then run:
    python model_train.py

Outputs are written to:  runs/<experiment_name>/<timestamp>/
  checkpoints/best.pt          — best validation checkpoint
  predictions_tiff/            — TIFF stacks after training completes
  val_outputs/                 — per-epoch validation images + progression stack
  config.json / history.json   — run metadata
"""

import os
from configs.default import TrainConfig, FolderSpec
from utils.helpers import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.train import run_training
from engine.infer import predict_from_config


# ===========================================================================
# USER CONFIGURATION — edit this section to match your dataset and hardware
# ===========================================================================

cfg = TrainConfig(
    # --- Run identity ---
    runs_root="runs",
    experiment_name="OCT-Denoiser",

    # --- Dataset(s) ---
    # Add one FolderSpec per dataset. Multiple datasets are concatenated.
    # root_folder / data_folder:  path to the folder containing bscan*.raw files
    # pixels / alines:            spectral samples and A-lines per B-scan
    # crop_depth:                 [z0, z1) pixel window applied after IFFT
    # window_sigma / gap:         Gaussian spectral window shape (tune with tune.py)
    # n_sub_windows:              sub-windows per parent window; 0 = disabled
    folder_specs=[
        FolderSpec(
            root_folder=r"images\Maestro3",
            data_folder="6mm_1024Aline",
            pixels=2048,
            alines=1024,
            crop_depth=(0, 1024),
            window_sigma=0.05,
            gap=0.60,
            gap_offset=0.015,
            n_sub_windows=2,        # produces 2*2=4 sub-channels → use resunet_pseudo3d_multilevel
            sub_window_spread=0.5,
        ),
        # Add more datasets here, e.g.:
        # FolderSpec(
        #     root_folder=r"images\Maestro2",
        #     data_folder="6mm_2048Aline",
        #     pixels=2048,
        #     alines=2048,
        #     crop_depth=(0, 1024),
        #     window_sigma=0.05,
        #     gap=0.60,
        # ),
    ],
    cache_frames_per_worker=1000,

    # --- Device ---
    device="cuda",                  # "cuda" or "cpu"
    amp=True,                       # AMP requires CUDA

    # --- Model ---
    # "resunet_pseudo3d"            — 2-channel input, no sub-windows
    # "resunet_pseudo3d_multilevel" — 2+n_sub_channels input (requires n_sub_windows > 0)
    model_name="resunet_pseudo3d_multilevel",
    base=32,                        # base channel width (larger = more capacity, more VRAM)

    # --- Training ---
    epochs=300,
    lr=3e-4,
    weight_decay=8e-5,
    batch_size=12,
    num_workers=4,
    augment=True,
    deterministic=True,

    # --- Patch sampling ("strip" = full-depth A-line; "patch" = random crop) ---
    patch_h=288,
    patch_w=32,
    patches_per_frame=32,
    patch_mode="strip",

    # --- Loss weights ---
    w_charb=0.0103,
    w_grad=0.0102,

    # --- Evaluation ROI (y pixel rows for SNR/CNR signal region) ---
    snr_sig_y0=111,
    snr_sig_y1=600,
    snr_sig_stat="p99.99",

    # --- Cadence ---
    val_every=5,
    save_every=5,
    early_stop_patience=20,

    # --- Output ---
    tiff_dtype="uint16",
    also_save_float32=False,
)

# ===========================================================================
# END USER CONFIGURATION
# ===========================================================================


def main():
    seed_all(cfg.seed, deterministic=cfg.deterministic)

    run_dir = make_run_dir(cfg.runs_root, cfg.experiment_name)
    paths = setup_run_dirs(run_dir)

    result = run_training(cfg, paths)

    # Run inference on every dataset using the best checkpoint
    for folder_spec in cfg.folder_specs:
        out_dir = os.path.join(paths["pred_tiff"], folder_spec.data_folder)
        predict_from_config(cfg, folder_spec, result["best_ckpt_path"], out_dir)


if __name__ == "__main__":
    main()
