from configs.default import TrainConfig
from utils.seed import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.train import run_training
from engine.infer import predict_npz_to_tiffs

def main():
    # Edit only these:
    npz_path = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\images\processed\6mm_1024Aline_gapped_dataset.npz"
    runs_root = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\runs"

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
        lr=0.00030606554343739014,
        num_workers=4,
        patch_h=192,
        patch_w=320,
        patches_per_frame=24,
        w_charb=0.30176532529712513,
        w_grad=0.011968933422333048,
        weight_decay=8.131534094431376e-05,
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
