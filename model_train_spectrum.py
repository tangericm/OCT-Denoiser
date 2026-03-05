"""Entry point for spectrum-domain denoising training.

Trains a 1D UNet on complex OCT spectra (pre-FFT) with a hybrid loss
that optimizes both spectral fidelity and reconstructed image quality.

Usage:
    python model_train_spectrum.py
"""
import os
from configs.default import TrainConfig, FolderSpec
from utils.helpers import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.spectrum_train import run_spectrum_training
from engine.spectrum_infer import predict_spectrum_from_config


def main():
    cfg = TrainConfig(
        runs_root=r"runs",
        experiment_name="Spectrum",
        model_name="spectrum_unet_2d",
        spectrum_mode=True,

        folder_specs=[
            FolderSpec(
                root_folder=r"images\Maestro3",
                data_folder="6mm_1024Aline",
                pixels=2048,
                alines=1024,
                crop_depth=(0, 1024),
                dispersion=[1.315892282e-06, 5.459678905e-10],
                window_sigma=0.05,
                gap=0.60,
                gap_offset=0.015,
            ),
        ],
        cache_frames_per_worker=1000,

        device="cuda",
        amp=True,
        deterministic=True,
        epochs=30,
        base=32,
        batch_size=8,
        lr=3e-4,
        weight_decay=8e-5,
        num_workers=4,
        augment=True,

        # For spectrum training: 2D patches (patch_w A-lines wide, full spectral depth)
        patch_h=2048,
        patch_w=16,
        patches_per_frame=16,
        patch_mode="strip",

        # Hybrid loss weights
        w_spectrum=0.01,
        w_image=0.8,
        w_charb=0.8,
        w_grad=0.5,

        snr_sig_y0=111,
        snr_sig_y1=600,
        snr_sig_stat="p99.99",
        val_every=5,
        save_every=5,
        early_stop_patience=20,
    )

    seed_all(cfg.seed, deterministic=cfg.deterministic)

    run_dir = make_run_dir(cfg.runs_root, cfg.experiment_name)
    paths = setup_run_dirs(run_dir)

    result = run_spectrum_training(cfg, paths)

    for folder_spec in cfg.folder_specs:
        predict_spectrum_from_config(
            cfg, folder_spec, result["best_ckpt_path"],
            os.path.join(paths["pred_tiff"], folder_spec.data_folder),
        )


if __name__ == "__main__":
    main()
