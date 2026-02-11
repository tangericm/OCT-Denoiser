from __future__ import annotations

import os
import csv
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from networks import create_model
from engine.metrics import roi_snr_cnr, roi_bounds, bg_bounds, to_physical_intensity
from utils.io_tiff import save_tiff_stack
from utils.run_manager import ensure_dir
from configs.default import TrainConfig

DEFAULT_SNR_SIG_Y0 = TrainConfig.__dataclass_fields__["snr_sig_y0"].default
DEFAULT_SNR_SIG_Y1 = TrainConfig.__dataclass_fields__["snr_sig_y1"].default
DEFAULT_SNR_SIG_STAT = TrainConfig.__dataclass_fields__["snr_sig_stat"].default



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

    # ROI y-range for SNR/CNR (x-range and background rows match training config)
    snr_sig_y0: int | None = None,
    snr_sig_y1: int | None = None,
    snr_sig_stat: str | None = None,
) -> None:
    """
    Raw-folder inference:
      - preprocess each frame on the fly using folder_spec + CLB
      - run model
      - save TIFF stacks + optional SNR CSV + ROI plots (frame 0)
    """
    from preprocess import BscanProcessor

    ensure_dir(outdir)
    print(f"[START] predict_raw_to_tiffs: outdir={outdir}")

    proc = BscanProcessor(folder_spec.root_folder, folder_spec.to_preprocess_config())
    if not proc.cfg.use_log:
        raise ValueError("predict_raw_to_tiffs currently expects cfg.use_log=True for linear-domain metric recovery")

    # Load model
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"[INFO] Creating model: {model_name}")
    model = create_model(model_name, base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[INFO] Model loaded and set to eval mode")

    # Determine frame count + shape
    paths = proc.bscan_paths
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    F = len(paths)

    # Pre-run first frame to get H,W
    print(f"[INFO] Total frames to process: {F}")
    out0 = proc.process_one(paths[0], frame_idx=0)
    H, W = out0["target_full"].shape
    print(f"[INFO] Frame 0 processed; image shape: {H}x{W}")

    preds = np.zeros((F, H, W), dtype=np.float32)
    gts   = np.zeros((F, H, W), dtype=np.float32)

    snr_pred_list: list[float] = []
    snr_gt_list: list[float] = []
    cnr_pred_list: list[float] = []
    cnr_gt_list: list[float] = []
    times: list[float] = []

    # Format filename with sigma and gap parameters
    sigma = int(round(folder_spec.window_sigma * 100))  # e.g., 0.08 -> 080
    gap = int(round(folder_spec.gap * 100))              # e.g., 0.25 -> 250
    param_suffix = f"s{sigma:03d}_g{gap:03d}"


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

    if snr_sig_y0 is None:
        snr_sig_y0 = DEFAULT_SNR_SIG_Y0
    if snr_sig_y1 is None:
        snr_sig_y1 = DEFAULT_SNR_SIG_Y1
    if snr_sig_stat is None:
        snr_sig_stat = DEFAULT_SNR_SIG_STAT

    sig_roi_c = roi_bounds(H, W, snr_sig_y0, snr_sig_y1)
    sy0, sy1, sx0, sx1 = sig_roi_c
    bg_roi_c = bg_bounds(H, W, x0=sx0, x1=sx1)

    # Warmup
    print(f"[INFO] Running warmup inference...")
    _ = _infer_1(np.stack([out0["input_w1"], out0["input_w2"]], axis=0).astype(np.float32))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"[INFO] Warmup complete, starting predictions...")

    for i, p in enumerate(paths):
        out = proc.process_one(p, frame_idx=i)
        x2hw = np.stack([out["input_w1"], out["input_w2"]], axis=0).astype(np.float32)
        gt = out["target_full"].astype(np.float32)
        target_mu = float(out["target_mu"])
        target_sd = float(out["target_sd"])

        t0 = time.time()
        pred = _infer_1(x2hw)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_time = time.time() - t0
        times.append(elapsed_time)
        # print(f"[INFO] Prediction time: {elapsed_time:.10f} seconds")

        preds[i] = pred
        gts[i] = gt

        # Recover per-frame physical-domain intensities and compute SNR/CNR there.
        sample_meta = {"target_mu": target_mu, "target_sd": target_sd, "log_eps": proc.cfg.log_eps}
        pred_lin = to_physical_intensity(pred, sample_meta)
        gt_lin = to_physical_intensity(gt, sample_meta)

        snr_pred, cnr_pred = roi_snr_cnr(pred_lin, sig_roi_c, bg_roi_c, sig_stat=snr_sig_stat)
        snr_gt, cnr_gt = roi_snr_cnr(gt_lin, sig_roi_c, bg_roi_c, sig_stat="max")
        snr_pred_list.append(snr_pred)
        snr_gt_list.append(snr_gt)
        cnr_pred_list.append(cnr_pred)
        cnr_gt_list.append(cnr_gt)

        print(
            f"[PROGRESS] Frame {i+1}/{F}: inference={elapsed_time:.4f}s, "
            f"SNR_pred={snr_pred:.2f}dB, SNR_gt={snr_gt:.2f}dB, "
            f"CNR_pred={cnr_pred:.2f}dB, CNR_gt={cnr_gt:.2f}dB"
        )

        if i == 0:
            _save_roi_plot_first(pred_lin, sig_roi_c, bg_roi_c, snr_pred, os.path.join(outdir, f"snr_rois_frame0_pred_{param_suffix}.png"))
            _save_roi_plot_first(gt_lin,   sig_roi_c, bg_roi_c, snr_gt,   os.path.join(outdir, f"snr_rois_frame0_gt_{param_suffix}.png"))

    snr_pred_arr = np.asarray(snr_pred_list, dtype=np.float64)
    snr_gt_arr = np.asarray(snr_gt_list, dtype=np.float64)
    cnr_pred_arr = np.asarray(cnr_pred_list, dtype=np.float64)
    cnr_gt_arr = np.asarray(cnr_gt_list, dtype=np.float64)
    mean_pred = float(np.nanmean(np.where(np.isfinite(snr_pred_arr), snr_pred_arr, np.nan))) if snr_pred_list else float("nan")
    mean_gt = float(np.nanmean(np.where(np.isfinite(snr_gt_arr), snr_gt_arr, np.nan))) if snr_gt_list else float("nan")
    mean_cnr_pred = float(np.nanmean(np.where(np.isfinite(cnr_pred_arr), cnr_pred_arr, np.nan))) if cnr_pred_list else float("nan")
    mean_cnr_gt = float(np.nanmean(np.where(np.isfinite(cnr_gt_arr), cnr_gt_arr, np.nan))) if cnr_gt_list else float("nan")
    mean_time = float(np.mean(times)) if times else float("nan")

    # save SNR CSV
    snr_csv_path = os.path.join(outdir, f"snr_per_frame_{param_suffix}.csv")

    with open(snr_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "snr_pred", "snr_gt", "snr_pred_minus_gt", "cnr_pred", "cnr_gt", "cnr_pred_minus_gt"])
        for i in range(F):
            sp = float(snr_pred_list[i])
            sg = float(snr_gt_list[i])
            cp = float(cnr_pred_list[i])
            cg = float(cnr_gt_list[i])
            w.writerow([i, sp, sg, sp - sg, cp, cg, cp - cg])
        w.writerow([])
        w.writerow(["mean_snr_pred", mean_pred])
        w.writerow(["mean_snr_gt", mean_gt])
        w.writerow(["mean_pred_minus_gt", (mean_pred - mean_gt) if np.isfinite(mean_pred) and np.isfinite(mean_gt) else float("nan")])
        w.writerow(["mean_cnr_pred", mean_cnr_pred])
        w.writerow(["mean_cnr_gt", mean_cnr_gt])
        w.writerow(["mean_cnr_pred_minus_gt", (mean_cnr_pred - mean_cnr_gt) if np.isfinite(mean_cnr_pred) and np.isfinite(mean_cnr_gt) else float("nan")])


    print(f"[INFO] Average prediction time per frame: {mean_time:.10f} seconds")
    print(f"[OK] Saved SNR CSV: {snr_csv_path}")
    print(f"[OK] Mean SNR pred: {mean_pred:.4f}")
    print(f"[OK] Mean SNR gt  : {mean_gt:.4f}")
    print(f"[OK] Mean dSNR: {(mean_pred - mean_gt) if np.isfinite(mean_pred) and np.isfinite(mean_gt) else float('nan'):.4f}")
    print(f"[OK] Mean CNR pred: {mean_cnr_pred:.4f}")
    print(f"[OK] Mean CNR gt  : {mean_cnr_gt:.4f}")
    print(f"[OK] Mean dCNR: {(mean_cnr_pred - mean_cnr_gt) if np.isfinite(mean_cnr_pred) and np.isfinite(mean_cnr_gt) else float('nan'):.4f}")

    # save TIFFs
    print(f"[INFO] Saving TIFF stacks to {outdir}...")
    
    
    pred_path = os.path.join(outdir, f"pred_{param_suffix}.tiff")
    gt_path   = os.path.join(outdir, f"gt_{param_suffix}.tiff")

    save_tiff_stack(pred_path, preds, dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(gt_path,   gts,   dtype=tiff_dtype, scale_per_slice=True)

    if also_save_float32:
        save_tiff_stack(os.path.join(outdir, f"pred_{param_suffix}_float32.tiff"), preds, dtype="float32", scale_per_slice=True)
        save_tiff_stack(os.path.join(outdir, f"gt_{param_suffix}_float32.tiff"), gts, dtype="float32", scale_per_slice=True)


def predict_from_config(cfg, folder_spec, ckpt_path: str, outdir: str, **overrides) -> None:
    """Convenience wrapper: extracts inference params from TrainConfig."""
    predict_raw_to_tiffs(
        folder_spec=folder_spec,
        ckpt_path=ckpt_path,
        outdir=outdir,
        model_name=cfg.model_name,
        base=cfg.base,
        device=cfg.device,
        tiff_dtype=cfg.tiff_dtype,
        also_save_float32=cfg.also_save_float32,
        snr_sig_y0=cfg.snr_sig_y0,
        snr_sig_y1=cfg.snr_sig_y1,
        snr_sig_stat=cfg.snr_sig_stat,
        **overrides,
    )
