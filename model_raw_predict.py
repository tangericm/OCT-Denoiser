from configs.default import FolderSpec
from engine.raw_infer import predict_raw_to_tiffs  # this is what your raw training script uses

def main():
    # Pick the dataset folder (contains bscan*.bin/.raw AND the corresponding .CLB)
    folder_spec = FolderSpec(
        root_folder=r"images\Maestro2",
        data_folder="Line_6mm_2048Aline_135degCW_50frame_gain165",
        pixels=2048,
        alines=2048,
        crop_depth=(1024, 2048),
        clb_path=None,

        # Preprocessing knobs MUST match what you trained with (or what you want to evaluate)
        do_dc_subtract=True,
        window_type="hann",
        use_log=True,
        log_eps=1e-6,
        apply_fftshift_depth=True,
        dispersion=[4.778474717e-06, 6.475358372e-09],  # example (use correct one)
        window_sigma=0.08,
        gap=0.25,
    )

    ckpt_path = r"runs\multi_folder_raw\20260205_085307\checkpoints\best.pt"
    outdir = r"images\Maestro2\Line_6mm_2048Aline_135degCW_50frame_gain165\predictions_tiff"

    predict_raw_to_tiffs(
        folder_spec=folder_spec,
        ckpt_path=ckpt_path,
        outdir=outdir,
        model_name="resunet_pseudo3d",
        base=32,
        device="cuda",
        tiff_dtype="uint16",
        also_save_float32=False,
        max_frames=None, 
    )

if __name__ == "__main__":
    main()