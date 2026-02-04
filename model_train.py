from configs.default import TrainConfig
from utils.seed import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.train import run_training
from engine.infer import predict_npz_to_tiffs

def main():
    npz_path = r"images\Maestro3\processed\6mm_1024Aline_gapped_dataset_s008_g025.npz"
    runs_root = r"runs"

    cfg = TrainConfig(
        npz_path=npz_path,
        runs_root=runs_root,
        experiment_name="6mm_1024Aline",
        device="cuda",
        amp=True,
        deterministic=True,
        epochs=300,
        base=32, 
        batch_size=12,
        lr=3e-04,
        num_workers=4,
        patch_h=288,
        patch_w=288,
        patches_per_frame=16,
        w_charb=0.010307111599432855,
        w_grad=0.010163544565911599,
        weight_decay=8e-05,
    )

    seed_all(cfg.seed, deterministic=cfg.deterministic)

    run_dir = make_run_dir(cfg.runs_root, cfg.experiment_name)
    paths = setup_run_dirs(run_dir)

    result = run_training(cfg, paths)

    # Predict into the same run folder
    predict_npz_to_tiffs(
        npz_path=cfg.npz_path,
        ckpt_path=result["best_ckpt_path"],
        outdir=paths["pred_tiff"],
        model_name=cfg.model_name,
        base=cfg.base,
        device=cfg.device,
        tiff_dtype=cfg.tiff_dtype,
        also_save_float32=cfg.also_save_float32,
    )

    print(f"[DONE] Run folder: {paths['run']}")

if __name__ == "__main__":
    main()
