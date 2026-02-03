from engine.infer import predict_npz_to_tiffs

def main():
    npz_path = r"images\processed\6mm_1024Aline_2_gapped_dataset.npz"
    ckpt_path = r"runs\6mm_1024aline\20260203_054736\checkpoints\best.pt"
    outdir = r"images\predictions_tiff\6mm_1024Aline_2"

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
