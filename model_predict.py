from configs.default import FolderSpec
from engine.infer import predict_raw_to_tiffs


def main():
    folder_spec = FolderSpec(
        root_folder=r"images\Maestro3",
        data_folder="6mm_1024Aline",
        pixels=2048,
        alines=1024,
        crop_depth=(0, 1024),
        dispersion=[1.315892282e-06, 5.459678905e-10],
        window_sigma=0.08,
        gap=0.25,
    )

    predict_raw_to_tiffs(
        folder_spec=folder_spec,
        ckpt_path=r"runs\6mm_1024Aline_strip\\20260210_132737\checkpoints\best_by_score.pt",
        outdir=r"runs\6mm_1024Aline_strip\\20260210_132737\predictions_tiff\best_by_score",
        model_name="resunet_pseudo3d",
        base=32,
        device="cuda",
        also_save_float32=True,
    )


if __name__ == "__main__":
    main()
