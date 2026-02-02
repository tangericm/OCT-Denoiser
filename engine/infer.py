from __future__ import annotations

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from networks import create_model
from utils.io_tiff import save_tiff_stack
from utils.run_manager import ensure_dir


def _roi_snr(img2d: np.ndarray, sig_roi, bg_roi, eps: float = 1e-8) -> float:
    """
    SNR = (mean(signal) - mean(background)) / (std(background) + eps)

    img2d: [H,W] float32 (log-compressed is fine)
    ROI format: (y0, y1, x0, x1), y1/x1 exclusive
    """
    y0s, y1s, x0s, x1s = sig_roi
    y0b, y1b, x0b, x1b = bg_roi

    sig = img2d[y0s:y1s, x0s:x1s]
    bg  = img2d[y0b:y1b, x0b:x1b]

    # Convert to linear amplitude
    sig_A = 10.0 ** (sig / 20.0)
    bg_A  = 10.0 ** (bg  / 20.0)


    # mu_sig = float(np.mean(sig))
    # mu_bg  = float(np.mean(bg))
    max_sig = float(np.max(sig_A))
    std_bg = float(np.std(bg_A))

    # return (mu_sig - mu_bg) / (std_bg + eps)
    return 20.0 * np.log10((max_sig + eps) / (std_bg + eps))


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
def predict_npz_to_tiffs(
    *,
    npz_path: str,
    ckpt_path: str,
    outdir: str,
    model_name: str,
    base: int,
    device: str,
    tiff_dtype: str = "uint16",
    also_save_float32: bool = False,
    # ROI boxes (y0, y1, x0, x1)
    sig_roi: tuple[int, int, int, int] = (111, 600, 20, 1020),
    bg_roi: tuple[int, int, int, int] = (1000, 1020, 20, 1020),

    snr_csv_name: str = "snr_per_frame.csv",
) -> None:
    """
    Runs inference on NPZ, saves TIFF stacks, and (optionally) computes SNR per frame based
    on user-defined ROIs and saves ROI visualization plots + CSV.

    sig_roi/bg_roi are pixel-coordinate boxes: (y0, y1, x0, x1), y1/x1 exclusive.
    """
    ensure_dir(outdir)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)  # [F,2,H,W]
    Y = data["Y"].astype(np.float32)  # [F,1,H,W]

    F, C, H, W = X.shape
    assert C == 2

    # Basic ROI bounds safety
    def _clip_roi(r):
        y0, y1, x0, x1 = r
        y0 = int(np.clip(y0, 0, H - 1))
        y1 = int(np.clip(y1, y0 + 1, H))
        x0 = int(np.clip(x0, 0, W - 1))
        x1 = int(np.clip(x1, x0 + 1, W))
        return (y0, y1, x0, x1)

    sig_roi = _clip_roi(sig_roi)
    bg_roi = _clip_roi(bg_roi)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = create_model(model_name, base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    preds = np.zeros((F, 1, H, W), dtype=np.float32)

    # --- NEW: SNR outputs ---
    snr_list = []

    for i in range(F):
        x = torch.from_numpy(X[i:i + 1]).to(device)  # [1,2,H,W]

        yhat = model(x).detach().cpu().numpy().astype(np.float32)  # [1,1,H,W]
        preds[i] = yhat[0]

        # Compute SNR for this predicted frame
        pred_img = preds[i, 0]  # [H,W]
        snr_val = _roi_snr(pred_img, sig_roi, bg_roi)
        snr_list.append(snr_val)

        if i == 0:
            plot_path = os.path.join(outdir, "snr_rois_frame0.png")
            _save_roi_plot_first(pred_img, sig_roi, bg_roi, snr_val, plot_path)

    snr_arr = np.asarray(snr_list, dtype=np.float32)
    mean_snr = float(np.mean(snr_arr)) if snr_arr.size else float("nan")

    snr_csv_path = os.path.join(outdir, snr_csv_name)
    with open(snr_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "snr"])
        for i, s in enumerate(snr_arr):
            w.writerow([i, float(s)])
        w.writerow([])
        w.writerow(["mean_snr", mean_snr])

    print(f"[OK] Saved SNR CSV: {snr_csv_path}")
    print(f"[OK] Mean SNR: {mean_snr:.4f}")
    print(f"[OK] Saved ROI plot (frame 0): {os.path.join(outdir, 'snr_rois_frame0.png')}")

    # Save TIFFs (existing behavior)
    pred_stack = preds[:, 0, :, :]
    gt_stack = Y[:, 0, :, :]
    w1_stack = X[:, 0, :, :]
    w2_stack = X[:, 1, :, :]

    save_tiff_stack(os.path.join(outdir, "input_w1.tif"), w1_stack, dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(os.path.join(outdir, "input_w2.tif"), w2_stack, dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(os.path.join(outdir, "pred.tif"), pred_stack, dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(os.path.join(outdir, "gt.tif"), gt_stack, dtype=tiff_dtype, scale_per_slice=True)

    if also_save_float32:
        save_tiff_stack(os.path.join(outdir, "input_w1_float32.tif"), w1_stack, dtype="float32", scale_per_slice=False)
        save_tiff_stack(os.path.join(outdir, "input_w2_float32.tif"), w2_stack, dtype="float32", scale_per_slice=False)
        save_tiff_stack(os.path.join(outdir, "pred_float32.tif"), pred_stack, dtype="float32", scale_per_slice=False)
        save_tiff_stack(os.path.join(outdir, "gt_float32.tif"), gt_stack, dtype="float32", scale_per_slice=False)
