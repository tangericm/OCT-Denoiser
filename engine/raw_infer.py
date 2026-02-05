from __future__ import annotations

import os
import csv
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from networks import create_model
from utils.io_tiff import save_tiff_stack
from utils.run_manager import ensure_dir
from utils.keypoint_registration import register_stack_keypoints_to_pred0


def _roi_snr(img2d: np.ndarray, sig_roi, bg_roi, eps: float = 1e-8) -> float:
    """
    Per-A-line SNR, then average across A-lines within the ROI.

    For each A-line (x column) in the ROI:
      SNR_x(dB) = 20*log10( (max(signal_y_range, x) + eps) / (std(background_y_range, x) + eps) )

    Then return mean(SNR_x) over all x in the ROI where bg std is finite.

    img2d: [H,W] float32 (linear or log-compressed is fine as long as you're consistent)
    ROI format: (y0, y1, x0, x1), y1/x1 exclusive
    """
    y0s, y1s, x0s, x1s = sig_roi
    y0b, y1b, x0b, x1b = bg_roi

    # Ensure the A-line (x) ranges match; otherwise you aren't averaging comparable columns.
    x0 = max(x0s, x0b)
    x1 = min(x1s, x1b)
    if x1 <= x0:
        return float("nan")

    # Extract ROIs, aligned in x
    sig = img2d[y0s:y1s, x0:x1]  # shape [Hs, Wx]
    bg  = img2d[y0b:y1b, x0:x1]  # shape [Hb, Wx]

    sig = (10 ** sig) - 1e-6
    bg  = (10 ** bg) - 1e-6

    # Per-column metrics
    max_sig_per_x = np.max(sig, axis=0)          # [Wx]
    std_bg_per_x  = np.std(bg, axis=0)           # [Wx]

    # Compute per-A-line SNR in dB
    snr_per_x = 20.0 * np.log10((max_sig_per_x + eps) / (std_bg_per_x + eps))

    # Robustness: ignore any non-finite values (can happen if img2d contains NaNs/Infs)
    snr_per_x = snr_per_x[np.isfinite(snr_per_x)]
    if snr_per_x.size == 0:
        return float("nan")

    return float(np.mean(snr_per_x))


def _save_roi_plot_first(
    img2d: np.ndarray,
    sig_roi,
    bg_roi,
    snr: float,
    out_path: str,
    title: str = "Frame 0 SNR ROIs",
) -> None:
    """Save a single PNG showing signal/background ROIs on the first predicted frame."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    p1, p99 = np.percentile(img2d, [1, 99])
    vmin, vmax = float(p1), float(p99)
    if vmax <= vmin:
        vmin, vmax = float(img2d.min()), float(img2d.max())

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()

    # Signal ROI (green)
    y0s, y1s, x0s, x1s = sig_roi
    ax.add_patch(patches.Rectangle((x0s, y0s), x1s - x0s, y1s - y0s,
                                   linewidth=2, edgecolor="lime", facecolor="none"))
    note_y = max(0, y0s - 6)
    ax.text(x0s, note_y, "Signal", color="lime", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.4, pad=2))

    # Background ROI (yellow)
    y0b, y1b, x0b, x1b = bg_roi
    ax.add_patch(patches.Rectangle((x0b, y0b), x1b - x0b, y1b - y0b,
                                   linewidth=2, edgecolor="yellow", facecolor="none"))
    note_yb = max(0, y0b - 6)
    ax.text(x0b, note_yb, "Background", color="yellow", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.4, pad=2))

    ax.set_title(f"{title}\nSNR = {snr:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def predict_raw_to_tiffs(
    *,
    folder_spec,
    ckpt_path: str,
    outdir: str,
    model_name: str,
    base: int,
    device: str,
    tiff_dtype: str = "uint16",
    also_save_float32: bool = False,
    max_frames: int | None = None,

    # if full-frame doesn't fit VRAM, set tile params
    tile_hw: tuple[int, int] | None = None,   # e.g. (512, 512)
    overlap: int = 32,

    # ROI boxes (y0, y1, x0, x1)
    sig_roi: tuple[int, int, int, int] = (111, 600, 20, 1020),
    bg_roi: tuple[int, int, int, int] = (1000, 1020, 20, 1020),
    snr_csv_name: str = "snr_per_frame.csv",
) -> None:
    """
    Raw-folder inference:
      - preprocess each frame on the fly using folder_spec + CLB
      - run model
      - save TIFF stacks + optional SNR CSV + ROI plots (frame 0)
    """
    from preprocess import Config as PreprocessConfig, BscanProcessor  # your preprocess.py module
    import csv
    import numpy as np
    import os
    import time

    torch.backends.cudnn.benchmark = True
    ensure_dir(outdir)

    # Build processor from FolderSpec
    pcfg = PreprocessConfig(
        pixels=folder_spec.pixels,
        alines=folder_spec.alines,
        data_folder=folder_spec.data_folder,
        do_dc_subtract=getattr(folder_spec, "do_dc_subtract", True),
        window_type=getattr(folder_spec, "window_type", "hann"),
        use_log=getattr(folder_spec, "use_log", True),
        log_eps=getattr(folder_spec, "log_eps", 1e-6),
        crop_depth=folder_spec.crop_depth,
        apply_fftshift_depth=getattr(folder_spec, "apply_fftshift_depth", True),
        window_sigma=folder_spec.window_sigma,
        gap=folder_spec.gap,
        dispersion=getattr(folder_spec, "dispersion", None),
        debug_mode=False,
    )
    proc = BscanProcessor(folder_spec.root_folder, pcfg)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = create_model(model_name, base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Determine frame count + shape
    paths = proc.bscan_paths
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    F = len(paths)

    # Pre-run first frame to get H,W
    out0 = proc.process_one(paths[0], frame_idx=0)
    H, W = out0["target_full"].shape

    preds = np.zeros((F, 1, H, W), dtype=np.float32)
    gts   = np.zeros((F, 1, H, W), dtype=np.float32)

    snr_pred_list: list[float] = []
    snr_gt_list: list[float] = []
    times: list[float] = []

    # helper: full-frame or tiled inference
    def _infer_1(x2hw: np.ndarray) -> np.ndarray:
        # x2hw: [2,H,W]
        x = torch.from_numpy(x2hw[None, ...]).to(device, non_blocking=True)

        if tile_hw is None:
            yhat = model(x)  # [1,1,H,W]
            return yhat.detach().cpu().numpy().astype(np.float32)[0, 0]

        # tiled inference
        th, tw = tile_hw
        y_out = np.zeros((H, W), dtype=np.float32)
        w_out = np.zeros((H, W), dtype=np.float32)

        step_h = max(1, th - overlap)
        step_w = max(1, tw - overlap)

        for y0 in range(0, H, step_h):
            for x0 in range(0, W, step_w):
                y1 = min(H, y0 + th)
                x1 = min(W, x0 + tw)
                yy0 = max(0, y1 - th)
                xx0 = max(0, x1 - tw)

                tile = x[:, :, yy0:y1, xx0:x1]  # [1,2,th,tw] (or smaller at edges)
                yhat = model(tile)              # [1,1,*,*]
                yhat_np = yhat.detach().cpu().numpy().astype(np.float32)[0, 0]

                yh, yw = yhat_np.shape
                y_out[yy0:yy0+yh, xx0:xx0+yw] += yhat_np
                w_out[yy0:yy0+yh, xx0:xx0+yw] += 1.0

        return y_out / np.maximum(w_out, 1e-6)

    # ROI clipping (same pattern as your NPZ infer)
    def _clip_roi(r):
        y0, y1, x0, x1 = r
        y0 = int(np.clip(y0, 0, H - 1))
        y1 = int(np.clip(y1, y0 + 1, H))
        x0 = int(np.clip(x0, 0, W - 1))
        x1 = int(np.clip(x1, x0 + 1, W))
        return (y0, y1, x0, x1)

    sig_roi_c = _clip_roi(sig_roi)
    bg_roi_c  = _clip_roi(bg_roi)

    # Warmup
    _ = _infer_1(np.stack([out0["input_w1"], out0["input_w2"]], axis=0).astype(np.float32))
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    for i, p in enumerate(paths):
        out = proc.process_one(p, frame_idx=i)
        x2hw = np.stack([out["input_w1"], out["input_w2"]], axis=0).astype(np.float32)
        gt = out["target_full"].astype(np.float32)

        t0 = time.time()
        pred = _infer_1(x2hw)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        preds[i, 0] = pred
        gts[i, 0] = gt

        # SNR on pred + gt
        snr_pred = _roi_snr(pred, sig_roi_c, bg_roi_c)
        snr_gt   = _roi_snr(gt,   sig_roi_c, bg_roi_c)
        snr_pred_list.append(snr_pred)
        snr_gt_list.append(snr_gt)

        if i == 0:
            _save_roi_plot_first(pred, sig_roi_c, bg_roi_c, snr_pred, os.path.join(outdir, "snr_rois_frame0_pred.png"))
            _save_roi_plot_first(gt,   sig_roi_c, bg_roi_c, snr_gt,   os.path.join(outdir, "snr_rois_frame0_gt.png"))

    # save SNR CSV
    snr_csv_path = os.path.join(outdir, snr_csv_name)
    with open(snr_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "snr_pred", "snr_gt", "snr_pred_minus_gt"])
        for i in range(F):
            sp = float(snr_pred_list[i])
            sg = float(snr_gt_list[i])
            w.writerow([i, sp, sg, sp - sg])

    print(f"[OK] Saved SNR CSV: {snr_csv_path}")
    print(f"[INFO] Mean pred time/frame: {float(np.mean(times)):.6f}s")

    # save TIFFs
    save_tiff_stack(os.path.join(outdir, "pred.tif"), preds[:, 0], dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(os.path.join(outdir, "gt.tif"),   gts[:, 0],   dtype=tiff_dtype, scale_per_slice=True)

    if also_save_float32:
        save_tiff_stack(os.path.join(outdir, "pred_float32.tif"), preds[:, 0], dtype="float32", scale_per_slice=True)
        save_tiff_stack(os.path.join(outdir, "gt_float32.tif"),   gts[:, 0],   dtype="float32", scale_per_slice=True)
