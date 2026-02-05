from engine.infer import predict_npz_to_tiffs

def main():
    # npz_path = r"images\Maestro3\processed\6mm_1024Aline_gapped_dataset.npz"
    # ckpt_path = r"runs\6mm_1024aline\20260204_123731\checkpoints\best.pt"
    # outdir = r"images\Maestro3\6mm_1024Aline\predictions_tiff"

    npz_path = r"images\Maestro2\processed\Line_6mm_2048Aline_135degCW_50frame_gain165_gapped_dataset_s008_g050.npz"
    ckpt_path = r"runs\6mm_1024aline\6mm_1024Aline_gapped_dataset_s008_g055_nonorm\checkpoints\best.pt"
    outdir = r"images\Maestro2\Line_6mm_2048Aline_135degCW_50frame_gain165\predictions_tiff"

    predict_npz_to_tiffs(
        npz_path=npz_path,
        ckpt_path=ckpt_path,
        outdir=outdir,
        model_name="resunet_pseudo3d",
        base=32,
        device="cuda",
        tiff_dtype="uint16",
        also_save_float32=False,
    )

if __name__ == "__main__":
    main()
