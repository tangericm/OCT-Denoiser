from configs.default import TrainConfig, FolderSpec 
from utils.seed import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.train import run_training
from engine.infer import predict_raw_to_tiffs

def main():
    runs_root = r"runs"

    cfg = TrainConfig(
        npz_path = None,
        runs_root=runs_root,
        experiment_name="multi_folder_raw",

        folder_specs=[
            FolderSpec(
                root_folder=r"images\Maestro3",
                data_folder="6mm_1024Aline",
                pixels=2048,
                alines=1024,
                crop_depth=(0, 1024),
                dispersion=[1.315892282e-06, 5.459678905e-10],
                window_sigma=0.107831154,
                gap=0.005907899,
            ),
            # FolderSpec(
            #     root_folder=r"images\Maestro2",
            #     data_folder="Line_6mm_2048Aline_135degCW_50frame_gain165",
            #     pixels=2048,
            #     alines=2048,
            #     crop_depth=(0, 1024),
            #     dispersion=[4.778474717e-06, 6.475358372e-09],
            #     window_sigma=0.08,
            #     gap=0.25,
            # ),
        ],
        cache_frames_per_worker=1000,

        device="cuda",
        amp=True,
        deterministic=True,
        epochs=300,
        base=32,
        batch_size=12,
        lr=0.0000649267434045653,
        num_workers=4,

        # patch_h=288, # Unused when patch_mode="strip"
        # patch_w=288,
        # patches_per_frame=16,
        # patch_mode="patch",

        patch_h=288, # Unused when patch_mode="strip"
        patch_w=160,
        patches_per_frame=48,
        patch_mode="strip",

        w_charb=0.0105448640343948,
        w_grad=0.0406831396087861,
        weight_decay=3.35E-06,
    )

    seed_all(cfg.seed, deterministic=cfg.deterministic)

    run_dir = make_run_dir(cfg.runs_root, cfg.experiment_name)
    paths = setup_run_dirs(run_dir)

    result = run_training(cfg, paths)

    # Run inference directly on raw frames
    for folder_spec in cfg.folder_specs:
        predict_raw_to_tiffs(
            folder_spec=folder_spec, 
            ckpt_path=result["best_ckpt_path"],
            outdir=paths["pred_tiff"],
            model_name=cfg.model_name,
            base=cfg.base,
            device=cfg.device,
            tiff_dtype=cfg.tiff_dtype,
            also_save_float32=cfg.also_save_float32,
            max_frames=None,
        )

if __name__ == "__main__":
    main()