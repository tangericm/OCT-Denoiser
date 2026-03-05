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


def main():
    cfg = TrainConfig(
        runs_root=r"runs",
        experiment_name="Spectrum",
        model_name="spectrum_unet_1d",
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
        cache_frames_per_worker=100,

        device="cuda",
        amp=True,
        deterministic=True,
        epochs=300,
        base=32,
        batch_size=256,
        lr=3e-4,
        weight_decay=8e-5,
        num_workers=4,
        augment=True,

        # Not used for spectrum training (1D per A-line)
        patch_h=128,
        patch_w=1,
        patches_per_frame=256,
        patch_mode="strip",

        # Hybrid loss weights
        w_spectrum=1.0,
        w_image=0.5,
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

    run_spectrum_training(cfg, paths)


if __name__ == "__main__":
    main()
