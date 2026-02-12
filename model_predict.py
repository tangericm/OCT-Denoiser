from configs.default import FolderSpec, TrainConfig
from engine.infer import predict_from_config


def main():
    cfg = TrainConfig(
        model_name="resunet_pseudo3d_multilevel",
        base=32,
        device="cuda",
        also_save_float32=True,
        snr_sig_stat="p99.99",  # change to e.g. "p95" to use percentile statistic
    )

    run = r"sweep_subwindows=8\s004_g050"
    path =r"runs\6mm_1024Aline_strip\\" + run + r"\\checkpoints\best.pt"
    outdir1=r"runs\6mm_1024Aline_strip\\" + run + r"\\predictions\train"
    outdir2=r"runs\6mm_1024Aline_strip\\" + run + r"\\predictions\test"

    folder_spec = FolderSpec(
        root_folder=r"images\Maestro3",
        data_folder="6mm_1024Aline",
        pixels=2048,
        alines=1024,
        crop_depth=(0, 1024),
        dispersion=[1.315892282e-06, 5.459678905e-10],
        window_sigma=0.04,
        gap=0.50,
        n_sub_windows=8,            # 0=disabled; e.g. 8 sub-windows per parent (16 total)
        sub_window_spread=2.0,    # sub-window center spread in sigma units
    )

    predict_from_config(
        cfg=cfg,
        folder_spec=folder_spec,
        ckpt_path=path,
        outdir=outdir1,
    )

    folder_spec = FolderSpec(
        root_folder=r"images\Maestro2",
        data_folder="Line_6mm_2048Aline_135degCW_50frame_gain165",
        pixels=2048,
        alines=2048,
        crop_depth=(0, 1024),
        dispersion=[4.778474717e-06, 6.475358372e-09],
        window_sigma=0.04,
        gap=0.50,
        n_sub_windows=8,            # 0=disabled; e.g. 8 sub-windows per parent (16 total)
        sub_window_spread=2.0,    # sub-window center spread in sigma units
    )

    predict_from_config(
        cfg=cfg,
        folder_spec=folder_spec,
        ckpt_path=path,
        outdir=outdir2,
    )


if __name__ == "__main__":
    main()
