from engine.infer import predict_npz_to_tiffs

def main():
    npz_path = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\images\processed\6mm_1024Aline_2_gapped_dataset.npz"
    # npz_path = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\images\Maestro2\processed\Line_6mm_2048Aline_135degCW_50frame_gain165_gapped_dataset.npz"
    ckpt_path = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\runs\6mm_1024aline\20260126_174333\checkpoints\best.pt"
    outdir = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\images\predictions_tiff"

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
