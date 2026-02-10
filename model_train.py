from configs.default import TrainConfig, FolderSpec
from utils.helpers import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.train import run_training
from engine.infer import predict_from_config


def main():
    cfg = TrainConfig(
        runs_root=r"runs",
        experiment_name="6mm_1024Aline_strip",

        folder_specs=[
            FolderSpec(
                root_folder=r"images\Maestro3",
                data_folder="6mm_1024Aline",
                pixels=2048,
                alines=1024,
                crop_depth=(0, 1024),
                dispersion=[1.315892282e-06, 5.459678905e-10],
                window_sigma=0.08,
                gap=0.25,
            ),
        ],
        cache_frames_per_worker=1000,

        device="cuda",
        amp=True,
        deterministic=True,
        epochs=600,
        base=32,
        batch_size=12,
        lr=3e-4,
        weight_decay=8e-5,
        num_workers=4,
        augment=True,

        patch_h=288,
        patch_w=16,
        patches_per_frame=16,
        patch_mode="strip",

        w_charb=0.010307111599432855,
        w_grad=0.010163544565911599,
        w_snr_loss=0.01,

        snr_sig_y0=111,
        snr_sig_y1=600,
        val_every=5,
        save_every=5,
        early_stop_patience=20,
        score_w_val_loss=1.0,
        score_w_snr=0.3,
        score_w_cnr=0.2,
    )

    seed_all(cfg.seed, deterministic=cfg.deterministic)

    run_dir = make_run_dir(cfg.runs_root, cfg.experiment_name)
    paths = setup_run_dirs(run_dir)

    result = run_training(cfg, paths)

    for folder_spec in cfg.folder_specs:
        predict_from_config(cfg, folder_spec, result["best_ckpt_path"], paths["pred_tiff"])


if __name__ == "__main__":
    main()
