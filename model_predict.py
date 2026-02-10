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
        ckpt_path=r"runs\6mm_1024Aline_strip\\20260210_121656\checkpoints\best.pt",
        outdir=r"runs\6mm_1024Aline_strip\\20260210_121656\predictions_tiff\best",
        model_name="resunet_pseudo3d",
        base=32,
        device="cuda",
    )


if __name__ == "__main__":
    main()
