"""Inference pipeline for spectrum-domain denoising models.

Mirrors predict_raw_to_tiffs from engine/infer.py but operates pre-FFT:
  raw → process_one_spectrum → per-A-line model inference → IFFT → B-scan → TIFF
"""
from __future__ import annotations

import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg
import numpy as np
import scipy.fft as sfft
import torch

from configs.default import TrainConfig
from engine.metrics import bg_bounds, roi_bounds, roi_snr_cnr, to_physical_intensity
from networks import create_model
from utils.helpers import nanmean
from utils.io_tiff import save_tiff_stack
from utils.run_manager import ensure_dir, make_param_suffix

DEFAULT_SNR_SIG_Y0 = TrainConfig.__dataclass_fields__["snr_sig_y0"].default
DEFAULT_SNR_SIG_Y1 = TrainConfig.__dataclass_fields__["snr_sig_y1"].default
DEFAULT_SNR_SIG_STAT = TrainConfig.__dataclass_fields__["snr_sig_stat"].default

_CHUNK = 512  # A-lines per forward pass


def _spectrum_to_bscan(
    spec_complex: np.ndarray,
    crop_depth: tuple[int, int],
    use_log: bool,
    log_eps: float,
    apply_fftshift: bool,
) -> tuple[np.ndarray, float, float]:
    """Complex spectrum [pixels, alines] → normalised B-scan [H, W] + stats."""
    depth = sfft.ifft(spec_complex, axis=0, workers=-1)
    mag = np.abs(depth).astype(np.float32)
    if apply_fftshift:
        mag = sfft.fftshift(mag, axes=0).astype(np.float32)
    z0, z1 = crop_depth
    bscan = mag[z0:z1]
    if use_log:
        bscan = np.log10(bscan + log_eps).astype(np.float32)
    mu = float(bscan.mean())
    sd = float(bscan.std()) + 1e-6
    return ((bscan - mu) / sd).astype(np.float32), mu, sd


def _save_roi_plot(
    img2d: np.ndarray,
    sig_roi,
    bg_roi,
    snr: float,
    out_path: str,
    title: str,
) -> None:
    import matplotlib.patches as mpatches

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p1, p99 = np.percentile(img2d, [1, 99])
    vmin, vmax = float(p1), float(p99)
    if vmax <= vmin:
        vmin, vmax = float(img2d.min()), float(img2d.max())

    fig = matplotlib.figure.Figure(figsize=(6, 5))
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()

    y0s, y1s, x0s, x1s = sig_roi
    ax.add_patch(mpatches.Rectangle(
        (x0s, y0s), x1s - x0s, y1s - y0s,
        linewidth=2, edgecolor="lime", facecolor="none",
    ))
    ax.text(x0s, max(0, y0s - 6), "Signal", color="lime", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.4, pad=2))

    y0b, y1b, x0b, x1b = bg_roi
    ax.add_patch(mpatches.Rectangle(
        (x0b, y0b), x1b - x0b, y1b - y0b,
        linewidth=2, edgecolor="yellow", facecolor="none",
    ))
    ax.text(x0b, max(0, y0b - 6), "Background", color="yellow", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.4, pad=2))

    ax.set_title(f"{title}\nSNR = {snr:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")


@torch.no_grad()
def _infer_full_frame_2d(model, x_np: np.ndarray, patch_w: int, device: str) -> np.ndarray:
    """Sliding-window 2D inference on [2, pixels, alines] → [1, pixels, alines]."""
    alines = x_np.shape[2]
    stride = max(1, patch_w // 2)

    accum = np.zeros((1, x_np.shape[1], alines), dtype=np.float32)
    weights = np.zeros(alines, dtype=np.float32)

    starts = list(range(0, alines - patch_w + 1, stride))
    if not starts or starts[-1] + patch_w < alines:
        starts.append(max(0, alines - patch_w))

    for j in starts:
        chunk = np.ascontiguousarray(x_np[:, :, j:j + patch_w])
        t = torch.from_numpy(chunk).unsqueeze(0).to(device, non_blocking=True)
        pred = model(t)[0].cpu().numpy()  # [1, pixels, patch_w]
        accum[:, :, j:j + patch_w] += pred
        weights[j:j + patch_w] += 1.0

    return (accum / np.maximum(weights[np.newaxis, np.newaxis, :], 1.0)).astype(np.float32)


@torch.no_grad()
def predict_spectrum_raw_to_tiffs(
    *,
    folder_spec,
    ckpt_path: str,
    outdir: str,
    model_name: str,
    base: int,
    device: str,
    patch_w: int = 1,
    tiff_dtype: str = "uint16",
    also_save_float32: bool = False,
    max_frames: int | None = None,
    snr_sig_y0: int | None = None,
    snr_sig_y1: int | None = None,
    snr_sig_stat: str | None = None,
    save_raw_spectra: bool = False,
) -> None:
    """Spectrum-domain inference: raw → TIFF stacks + SNR CSV."""
    from preprocess import BscanProcessor

    ensure_dir(outdir)
    print(f"[START] predict_spectrum_raw_to_tiffs: outdir={outdir}")

    proc = BscanProcessor(folder_spec)

    folder_name = os.path.basename(folder_spec.data_folder.rstrip("/\\")) or "folder"
    window_path = os.path.join(outdir, f"window_figure_{folder_name}.png")
    if not os.path.exists(window_path):
        proc.save_window_figure(window_path)
        print(f"[OK] Saved window figure: {window_path}")

    # Load model
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = create_model(model_name, base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[INFO] Model loaded")

    paths = proc.bscan_paths
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    F = len(paths)
    print(f"[INFO] Total frames: {F}")

    if snr_sig_y0 is None:
        snr_sig_y0 = DEFAULT_SNR_SIG_Y0
    if snr_sig_y1 is None:
        snr_sig_y1 = DEFAULT_SNR_SIG_Y1
    if snr_sig_stat is None:
        snr_sig_stat = DEFAULT_SNR_SIG_STAT

    # Peek at first frame to get output shape
    out0 = proc.process_one_spectrum(paths[0], frame_idx=0)
    crop_depth = tuple(folder_spec.crop_depth)
    use_log = folder_spec.use_log
    log_eps = float(folder_spec.log_eps)
    apply_fftshift = folder_spec.apply_fftshift_depth
    norm_factor0 = float(out0["norm_factor"])
    pred_spec0 = (out0["spec_w1"].astype(np.float32) + out0["spec_w2"].astype(np.float32)) / 2  # rough placeholder
    bscan0, _, _ = _spectrum_to_bscan(pred_spec0 * norm_factor0, crop_depth, use_log, log_eps, apply_fftshift)
    H, W = bscan0.shape
    print(f"[INFO] Output B-scan shape: {H}x{W}")

    sig_roi = roi_bounds(H, W, snr_sig_y0, snr_sig_y1)
    sy0, sy1, sx0, sx1 = sig_roi
    bg_roi = bg_bounds(H, W, x0=sx0, x1=sx1)

    preds = np.zeros((F, H, W), dtype=np.float32)
    gts   = np.zeros((F, H, W), dtype=np.float32)
    w1    = np.zeros((F, H, W), dtype=np.float32)
    w2    = np.zeros((F, H, W), dtype=np.float32)

    pixels_full = folder_spec.pixels
    alines_full = out0["spec_full"].shape[1]
    raw_spectra: list[np.ndarray] | None = [] if save_raw_spectra else None

    snr_pred_list: list[float] = []
    snr_gt_list:   list[float] = []
    cnr_pred_list: list[float] = []
    cnr_gt_list:   list[float] = []
    times: list[float] = []

    param_suffix = make_param_suffix(folder_spec)

    for i, p in enumerate(paths):
        out = proc.process_one_spectrum(p, frame_idx=i)
        norm_factor = float(out["norm_factor"])

        # Build [2, pixels, alines] input
        x_np = np.stack([
            out["spec_w1"], out["spec_w2"],
        ], axis=0).astype(np.float32)  # [2, pixels, alines]

        alines = x_np.shape[2]

        t0 = time.time()
        if patch_w > 1:
            pred_np2d = _infer_full_frame_2d(model, x_np, patch_w, device)  # [1, pixels, alines]
            pred_spec = pred_np2d[0].astype(np.float32) * norm_factor
        else:
            # Reshape to [alines, 2, pixels] for batched A-line inference
            x_alines = torch.from_numpy(
                np.ascontiguousarray(x_np.transpose(2, 0, 1))
            ).to(device, non_blocking=True)  # [alines, 2, pixels]
            pred_chunks = []
            for j in range(0, alines, _CHUNK):
                pred_chunks.append(model(x_alines[j:j + _CHUNK]))
            pred_alines = torch.cat(pred_chunks, dim=0).cpu().numpy()  # [alines, 1, pixels]
            pred_spec = pred_alines[:, 0, :].T.astype(np.float32) * norm_factor
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)

        if raw_spectra is not None:
            raw_spectra.append(pred_spec.astype(np.float32))

        pred_bscan, pred_mu, pred_sd = _spectrum_to_bscan(pred_spec, crop_depth, use_log, log_eps, apply_fftshift)

        # Reconstruct ground-truth B-scan
        tgt_spec = out["spec_full"].astype(np.float32) * norm_factor
        tgt_bscan, tgt_mu, tgt_sd = _spectrum_to_bscan(tgt_spec, crop_depth, use_log, log_eps, apply_fftshift)

        # Window B-scans (for TIFF export)
        w1_spec = out["spec_w1"].astype(np.float32) * norm_factor
        w2_spec = out["spec_w2"].astype(np.float32) * norm_factor
        w1_bscan, _, _ = _spectrum_to_bscan(w1_spec, crop_depth, use_log, log_eps, apply_fftshift)
        w2_bscan, _, _ = _spectrum_to_bscan(w2_spec, crop_depth, use_log, log_eps, apply_fftshift)

        preds[i] = pred_bscan
        gts[i]   = tgt_bscan
        w1[i]    = w1_bscan
        w2[i]    = w2_bscan

        pred_meta = {"target_mu": pred_mu, "target_sd": pred_sd, "log_eps": log_eps}
        tgt_meta  = {"target_mu": tgt_mu,  "target_sd": tgt_sd,  "log_eps": log_eps}
        pred_phys = to_physical_intensity(pred_bscan, pred_meta)
        tgt_phys  = to_physical_intensity(tgt_bscan,  tgt_meta)

        snr_pred, cnr_pred = roi_snr_cnr(pred_phys, sig_roi, bg_roi, sig_stat=snr_sig_stat)
        snr_gt,   cnr_gt   = roi_snr_cnr(tgt_phys,  sig_roi, bg_roi, sig_stat="max")
        snr_pred_list.append(snr_pred)
        snr_gt_list.append(snr_gt)
        cnr_pred_list.append(cnr_pred)
        cnr_gt_list.append(cnr_gt)

        print(
            f"[PROGRESS] Frame {i+1}/{F}: inference={elapsed:.4f}s  "
            f"SNR_pred={snr_pred:.2f}dB  SNR_gt={snr_gt:.2f}dB  "
            f"CNR_pred={cnr_pred:.2f}dB  CNR_gt={cnr_gt:.2f}dB"
        )

        if i == 0:
            _save_roi_plot(pred_phys, sig_roi, bg_roi, snr_pred,
                           os.path.join(outdir, f"snr_rois_frame0_pred_{param_suffix}.png"),
                           "Frame 0 SNR ROIs (pred)")
            _save_roi_plot(tgt_phys, sig_roi, bg_roi, snr_gt,
                           os.path.join(outdir, f"snr_rois_frame0_gt_{param_suffix}.png"),
                           "Frame 0 SNR ROIs (gt)")

    mean_snr_pred = nanmean(snr_pred_list)
    mean_snr_gt   = nanmean(snr_gt_list)
    mean_cnr_pred = nanmean(cnr_pred_list)
    mean_cnr_gt   = nanmean(cnr_gt_list)
    mean_time     = float(np.mean(times)) if times else float("nan")

    snr_csv = os.path.join(outdir, f"snr_per_frame_{param_suffix}.csv")
    with open(snr_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "snr_pred", "snr_gt", "snr_pred_minus_gt",
                     "cnr_pred", "cnr_gt", "cnr_pred_minus_gt"])
        for i in range(F):
            sp, sg = float(snr_pred_list[i]), float(snr_gt_list[i])
            cp, cg = float(cnr_pred_list[i]), float(cnr_gt_list[i])
            wr.writerow([i, sp, sg, sp - sg, cp, cg, cp - cg])
        wr.writerow([])
        wr.writerow(["mean_snr_pred",        mean_snr_pred])
        wr.writerow(["mean_snr_gt",          mean_snr_gt])
        wr.writerow(["mean_pred_minus_gt",   mean_snr_pred - mean_snr_gt
                     if np.isfinite(mean_snr_pred) and np.isfinite(mean_snr_gt) else float("nan")])
        wr.writerow(["mean_cnr_pred",        mean_cnr_pred])
        wr.writerow(["mean_cnr_gt",          mean_cnr_gt])
        wr.writerow(["mean_cnr_pred_minus_gt", mean_cnr_pred - mean_cnr_gt
                     if np.isfinite(mean_cnr_pred) and np.isfinite(mean_cnr_gt) else float("nan")])

    print(f"[INFO] Mean inference time/frame: {mean_time:.4f}s")
    print(f"[OK] Saved SNR CSV: {snr_csv}")
    print(f"[OK] Mean SNR pred={mean_snr_pred:.4f}  gt={mean_snr_gt:.4f}  "
          f"dSNR={mean_snr_pred - mean_snr_gt:.4f}")
    print(f"[OK] Mean CNR pred={mean_cnr_pred:.4f}  gt={mean_cnr_gt:.4f}  "
          f"dCNR={mean_cnr_pred - mean_cnr_gt:.4f}")

    pred_path = os.path.join(outdir, f"pred_{param_suffix}.tiff")
    gt_path   = os.path.join(outdir, f"gt_{param_suffix}.tiff")
    w1_path   = os.path.join(outdir, f"w1_{param_suffix}.tiff")
    w2_path   = os.path.join(outdir, f"w2_{param_suffix}.tiff")

    save_tiff_stack(pred_path, preds, dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(gt_path,   gts,   dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(w1_path,   w1,    dtype=tiff_dtype, scale_per_slice=True)
    save_tiff_stack(w2_path,   w2,    dtype=tiff_dtype, scale_per_slice=True)

    if also_save_float32:
        save_tiff_stack(
            os.path.join(outdir, f"pred_{param_suffix}_float32.tiff"),
            preds, dtype="float32", scale_per_slice=True,
        )
        save_tiff_stack(
            os.path.join(outdir, f"gt_{param_suffix}_float32.tiff"),
            gts, dtype="float32", scale_per_slice=True,
        )

    if raw_spectra is not None:
        raw_array = np.stack(raw_spectra, axis=0)  # [F, pixels, alines], float32
        raw_path  = os.path.join(outdir, f"pred_spectra_{param_suffix}.raw")
        raw_array.tofile(raw_path)
        meta = {
            "shape":   list(raw_array.shape),
            "dtype":   "float32",
            "axes":    ["frames", "pixels", "alines"],
            "units":   "physical (norm_factor applied per frame)",
            "pixels":  pixels_full,
            "alines":  alines_full,
            "frames":  F,
        }
        meta_path = raw_path.replace(".raw", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[OK] Saved raw spectra: {raw_path}  ({raw_array.nbytes / 1e6:.1f} MB)")
        print(f"[OK] Saved spectrum metadata: {meta_path}")

    print(f"[OK] Saved TIFFs to {outdir}")


def predict_spectrum_from_config(cfg, folder_spec, ckpt_path: str, outdir: str, **overrides) -> None:
    """Convenience wrapper: extracts inference params from TrainConfig."""
    predict_spectrum_raw_to_tiffs(
        folder_spec=folder_spec,
        ckpt_path=ckpt_path,
        outdir=outdir,
        model_name=cfg.model_name,
        base=cfg.base,
        device=cfg.device,
        patch_w=cfg.patch_w,
        tiff_dtype=cfg.tiff_dtype,
        also_save_float32=cfg.also_save_float32,
        save_raw_spectra=cfg.save_raw_spectra,
        snr_sig_y0=cfg.snr_sig_y0,
        snr_sig_y1=cfg.snr_sig_y1,
        snr_sig_stat=cfg.snr_sig_stat,
        **overrides,
    )
