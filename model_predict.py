from configs.default import FolderSpec, PostprocessConfig, TrainConfig
from engine.infer import predict_from_config


def main():
    cfg = TrainConfig(
        model_name="resunet_multilevel_1d",
        base=32,
        device="cuda",
        also_save_float32=False,
        snr_sig_stat="p99.99",  # change to e.g. "p95" to use percentile statistic
        postprocess=PostprocessConfig(
            enable=True,
            # Registration
            do_register=True,
            ref_strategy="middle",
            transform_model="affine",
            use_clahe=True,
            use_ecc=True,
            # Deconvolution
            do_deconv=False,
            deconv_method="wiener",
            psf_sigma=1.5,
            wiener_nsr=0.01,
        ),
    )

    run = r"1D_npatch=256"
    path =r"runs\A-Line\\" + run + r"\\checkpoints\best.pt"
    outdir=r"runs\A-Line\\" + run + r"\\predictions"

    window_sigma = 0.05
    gap = 0.60
    offset = 0.015

    folder_spec = FolderSpec(
        root_folder=r"images\Maestro3",
        data_folder="6mm_1024Aline",
        pixels=2048,
        alines=1024,
        crop_depth=(0, 1024),
        dispersion=[1.315892282e-06, 5.459678905e-10],
        window_sigma=window_sigma,
        gap=gap,
        gap_offset=offset,
        n_sub_windows=2,            # 0=disabled; e.g. 8 sub-windows per parent (16 total)
        sub_window_spread=0.5,    # sub-window center spread in sigma units
    )

    predict_from_config(
        cfg=cfg,
        folder_spec=folder_spec,
        ckpt_path=path,
        outdir=outdir + r"\\" + folder_spec.data_folder,
    )

    folder_spec = FolderSpec(
        root_folder=r"images\Maestro3",
        data_folder="6mm_1024Aline_disc",
        pixels=2048,
        alines=1024,
        crop_depth=(0, 1024),
        dispersion=[1.315892282e-06, 5.459678905e-10],
        window_sigma=window_sigma,
        gap=gap,
        gap_offset=offset,
        n_sub_windows=2,            # 0=disabled; e.g. 8 sub-windows per parent (16 total)
        sub_window_spread=0.5,    # sub-window center spread in sigma units
    )


    # folder_spec = FolderSpec(
    #     root_folder=r"images\Maestro2",
    #     data_folder="Line_6mm_2048Aline_135degCW_50frame_gain165",
    #     pixels=2048,
    #     alines=2048,
    #     crop_depth=(0, 1024),
    #     dispersion=[4.778474717e-06, 6.475358372e-09],
    #     window_sigma=window_sigma,
    #     gap=gap,
    #     gap_offset=offset,
    #     n_sub_windows=2,            # 0=disabled; e.g. 8 sub-windows per parent (16 total)
    #     sub_window_spread=0.5,    # sub-window center spread in sigma units
    # )

    predict_from_config(
        cfg=cfg,
        folder_spec=folder_spec,
        ckpt_path=path,
        outdir=outdir + r"\\" + folder_spec.data_folder,
    )


if __name__ == "__main__":
    main()
