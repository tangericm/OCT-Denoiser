"""Training loop for spectrum-domain denoising.

Mirrors the structure of engine/train.py but uses:
- SpectrumDataModule (real windowed spectra as input channels)
- Hybrid spectrum + image loss (via differentiable IFFT)
- Full-frame evaluation that reconstructs B-scans for SNR/CNR metrics
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg
from dataclasses import asdict
from typing import Dict, Any

from engine.early_stopping import EarlyStopping
from engine.spectrum_losses import compute_spectrum_loss
from engine.spectrum_infer import _spectrum_to_bscan, _infer_full_frame_2d
from engine.metrics import roi_bounds, bg_bounds, roi_snr_cnr, to_physical_intensity
from data.spectrum_dataset import SpectrumDataModule
from networks import create_model
from utils.helpers import save_json, nanmean
from utils.io_tiff import save_tiff_stack
from utils.live_plot import LiveLossPlot


def _unpack_batch(batch, device: str):
    x, y, meta = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta


def _loss_params_from_meta(meta):
    """Extract FFT parameters from first sample's metadata."""
    m = meta[0] if isinstance(meta, (list, tuple)) else meta
    return {
        "crop_depth": tuple(m["crop_depth"]),
        "log_eps": float(m["log_eps"]),
        "apply_fftshift": bool(m["apply_fftshift_depth"]),
    }


@torch.no_grad()
def _evaluate_patches(model, loader, device, cfg):
    """Patch-based validation loss."""
    model.eval()
    loss_acc, n = 0.0, 0
    for batch in loader:
        x, y, meta = _unpack_batch(batch, device)
        pred = model(x)
        lp = _loss_params_from_meta(meta)
        loss = compute_spectrum_loss(
            pred, y,
            **lp,
            w_charb=cfg.w_charb,
            w_grad=cfg.w_grad,
            w_spec_mag=cfg.w_spec_mag,
        )
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_acc / max(n, 1)


@torch.no_grad()
def _evaluate_full_frames(model, loader, device, cfg):
    """Full-frame evaluation: reconstruct B-scans from predicted spectra, compute SNR/CNR."""
    model.eval()
    patch_w = getattr(cfg, "patch_w", 1)
    loss_acc, n = 0.0, 0
    snr_pred_list, snr_gt_list = [], []
    cnr_pred_list, cnr_gt_list = [], []
    sample_pred = None

    for batch in loader:
        x, y, meta = _unpack_batch(batch, device)
        # x: [1, 2, pixels, alines], y: [1, 1, pixels, alines]
        m = meta[0]
        norm_factor = float(m["norm_factor"])
        crop_depth = tuple(m["crop_depth"])
        use_log = bool(m["use_log"])
        log_eps = float(m["log_eps"])
        apply_fftshift = bool(m["apply_fftshift_depth"])

        x_np = x[0].cpu().numpy()  # [2, pixels, alines]
        y_2d = y[0]  # [1, pixels, alines] tensor

        if patch_w > 1:
            # 2D model: sliding-window inference
            pred_np = _infer_full_frame_2d(model, x_np, patch_w, device)  # [1, pixels, alines]
            pred_spec = pred_np[0].astype(np.float32) * norm_factor
        else:
            # 1D model: process each A-line as batch element
            alines = x_np.shape[2]
            x_alines = torch.from_numpy(np.ascontiguousarray(x_np.transpose(2, 0, 1))).to(device)
            # [alines, 2, pixels]
            pred_chunks = []
            for i in range(0, alines, 512):
                pred_chunks.append(model(x_alines[i:i + 512]))
            pred_alines = torch.cat(pred_chunks, dim=0)  # [alines, 1, pixels]

            # Frame loss on all A-lines
            lp = _loss_params_from_meta(meta)
            y_alines = y_2d.permute(2, 0, 1)
            frame_loss = compute_spectrum_loss(
                pred_alines, y_alines,
                **lp,
                w_charb=cfg.w_charb,
                w_grad=cfg.w_grad,
                w_spec_mag=cfg.w_spec_mag,
            )
            loss_acc += float(frame_loss.item())

            arr = pred_alines.cpu().numpy()  # [alines, 1, pixels]
            pred_spec = arr[:, 0, :].T.astype(np.float32) * norm_factor  # [pixels, alines]

        n += 1

        y_np = y_2d.cpu().numpy()  # [1, pixels, alines]
        tgt_spec = y_np[0].astype(np.float32) * norm_factor  # [pixels, alines]

        pred_bscan, pred_mu, pred_sd = _spectrum_to_bscan(
            pred_spec, crop_depth, use_log, log_eps, apply_fftshift
        )
        tgt_bscan, tgt_mu, tgt_sd = _spectrum_to_bscan(
            tgt_spec, crop_depth, use_log, log_eps, apply_fftshift
        )

        # SNR/CNR on physical intensity
        h, w = pred_bscan.shape
        sig_roi = roi_bounds(h, w, cfg.snr_sig_y0, cfg.snr_sig_y1)
        bg_roi = bg_bounds(h, w, x0=sig_roi[2], x1=sig_roi[3])

        pred_meta = {"target_mu": pred_mu, "target_sd": pred_sd, "log_eps": log_eps}
        tgt_meta = {"target_mu": tgt_mu, "target_sd": tgt_sd, "log_eps": log_eps}

        pred_phys = to_physical_intensity(pred_bscan, pred_meta)
        tgt_phys = to_physical_intensity(tgt_bscan, tgt_meta)

        snr_p, cnr_p = roi_snr_cnr(pred_phys, sig_roi, bg_roi, sig_stat=cfg.snr_sig_stat)
        snr_g, cnr_g = roi_snr_cnr(tgt_phys, sig_roi, bg_roi, sig_stat="max")

        snr_pred_list.append(snr_p)
        snr_gt_list.append(snr_g)
        cnr_pred_list.append(cnr_p)
        cnr_gt_list.append(cnr_g)

        if sample_pred is None:
            sample_pred = pred_bscan.copy()

    val_loss = loss_acc / max(n, 1)
    return {
        "val_loss": val_loss,
        "snr_pred": nanmean(snr_pred_list),
        "snr_gt": nanmean(snr_gt_list),
        "cnr_pred": nanmean(cnr_pred_list),
        "cnr_gt": nanmean(cnr_gt_list),
        "sample_pred": sample_pred,
    }


def _save_val_png(pred_img, *, snr_pred, snr_gt, cnr_pred, cnr_gt, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p1, p99 = np.percentile(pred_img, [1, 99])
    vmin, vmax = float(p1), float(p99)
    if vmax <= vmin:
        vmin, vmax = float(pred_img.min()), float(pred_img.max())
    fig = matplotlib.figure.Figure(figsize=(6, 5))
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(pred_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(
        "Spectrum Training - Val Prediction\n"
        f"SNR_pred={snr_pred:.2f}dB  SNR_gt={snr_gt:.2f}dB  "
        f"CNR_pred={cnr_pred:.2f}dB  CNR_gt={cnr_gt:.2f}dB",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")


def run_spectrum_training(cfg, paths: Dict[str, str]) -> Dict[str, Any]:
    """Spectrum-domain training loop."""
    device = cfg.device if torch.cuda.is_available() else "cpu"

    save_json(os.path.join(paths["run"], "config.json"), asdict(cfg))

    # Data
    dm = SpectrumDataModule(cfg)
    dm.setup()
    if cfg.folder_specs:
        from preprocess import BscanProcessor
        for folder_spec in cfg.folder_specs:
            folder_name = os.path.basename(folder_spec.data_folder.rstrip("/\\")) or "folder"
            window_path = os.path.join(paths["run"], f"window_figure_{folder_name}.png")
            if not os.path.exists(window_path):
                proc = BscanProcessor(folder_spec)
                proc.save_window_figure(window_path)
    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    val_full_loader = dm.val_full_loader()

    # Model
    model = create_model(cfg.model_name, base=cfg.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    use_cuda_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    best_val = float("inf")
    best_ckpt_path = os.path.join(paths["checkpoints"], "best.pt")

    history: dict = {"train_loss": [], "val_loss": [], "val_full": []}
    val_pred_stack: list[np.ndarray] = []
    val_pred_stack_epochs: list[int] = []

    early_stop = EarlyStopping(
        patience=cfg.early_stop_patience,
        min_delta=cfg.early_stop_min_delta,
        mode="min",
        warmup=cfg.early_stop_warmup_checks,
    )
    plotter = LiveLossPlot(
        out_dir=paths["run"],
        title=f"Loss - {cfg.experiment_name} (Spectrum), Network: {cfg.model_name}",
        show_window=False,
    )

    print(
        f"[INFO] Spectrum training | Device={device}  "
        f"train_batches={len(train_loader)}  val_batches={len(val_loader)}"
    )

    epoch_times = []
    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0

        for batch in train_loader:
            x, y, meta = _unpack_batch(batch, device)
            lp = _loss_params_from_meta(meta)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                pred = model(x)
                loss = compute_spectrum_loss(
                    pred, y,
                    **lp,
                    w_charb=cfg.w_charb,
                    w_grad=cfg.w_grad,
                    w_spec_mag=cfg.w_spec_mag,
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        train_loss = running / max(n, 1)
        history["train_loss"].append({"epoch": epoch, "loss": train_loss})

        if epoch == 1 or (epoch % cfg.val_every) == 0:
            val_loss = _evaluate_patches(model, val_loader, device, cfg)
            val_full = _evaluate_full_frames(model, val_full_loader, device, cfg)

            history["val_loss"].append({
                "epoch": epoch,
                "loss": val_loss,
                "snr_pred": val_full["snr_pred"],
                "snr_gt": val_full["snr_gt"],
                "cnr_pred": val_full["cnr_pred"],
                "cnr_gt": val_full["cnr_gt"],
            })
            history["val_full"].append({
                "epoch": epoch,
                "loss": val_full["val_loss"],
                "snr_pred": val_full["snr_pred"],
                "snr_gt": val_full["snr_gt"],
                "cnr_pred": val_full["cnr_pred"],
                "cnr_gt": val_full["cnr_gt"],
            })
            plotter.update(epoch=epoch, train_loss=train_loss, val_loss=val_full["val_loss"], val_snr=val_full["snr_pred"])
            dt = time.time() - t0
            print(
                f"[E{epoch:04d}] train={train_loss:.10f}  "
                f"val_loss={val_loss:.10f} "
                f"SNR_pred/gt={val_full['snr_pred']:.2f}/{val_full['snr_gt']:.2f}  "
                f"CNR_pred/gt={val_full['cnr_pred']:.2f}/{val_full['cnr_gt']:.2f}  "
                f"time={dt:.5f}s"
            )

            if val_full["sample_pred"] is not None:
                out_path = os.path.join(paths["val_outputs"], f"val_pred_epoch_{epoch:04d}.png")
                _save_val_png(
                    val_full["sample_pred"],
                    snr_pred=float(val_full["snr_pred"]),
                    snr_gt=float(val_full["snr_gt"]),
                    cnr_pred=float(val_full["cnr_pred"]),
                    cnr_gt=float(val_full["cnr_gt"]),
                    out_path=out_path,
                )
                val_pred_stack.append(val_full["sample_pred"])
                val_pred_stack_epochs.append(epoch)

            full_val_loss = val_full["val_loss"]
            if full_val_loss < best_val:
                best_val = full_val_loss
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val": best_val,
                }, best_ckpt_path)
                print(f"[OK] Saved best checkpoint: {best_ckpt_path}")

            stop_now = early_stop.update(float(full_val_loss), epoch)
            if stop_now:
                print(
                    f"[EARLY STOP] No val improvement for {early_stop.patience} validation checks. "
                    f"Best={early_stop.best:.6f} at epoch={early_stop.best_epoch}. "
                    f"Stopping at epoch={epoch}."
                )
                break

        epoch_dt = time.time() - t0
        epoch_times.append(epoch_dt)
        if (epoch % cfg.save_every) == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": asdict(cfg)},
                os.path.join(paths["checkpoints"], f"epoch_{epoch:04d}.pt"),
            )

    print(f"[DONE] Best val loss = {best_val:.10f}")
    history["early_stop"] = {
        "patience": early_stop.patience,
        "min_delta": early_stop.min_delta,
        "best": early_stop.best,
        "best_epoch": early_stop.best_epoch,
        "stop_epoch": early_stop.stop_epoch,
        "num_checks": early_stop.num_checks,
    }
    total_time = time.time() - train_start
    mean_epoch = sum(epoch_times) / max(len(epoch_times), 1)
    print(
        f"[TIMING] Epochs run = {len(epoch_times)} | "
        f"Mean epoch time = {mean_epoch:.10f} s | "
        f"Total training time = {total_time:.10f} s"
    )
    history["timing"] = {
        "epoch_times_sec": epoch_times,
        "mean_epoch_time_sec": mean_epoch,
        "total_train_time_sec": total_time,
    }

    if val_pred_stack:
        stack_path = os.path.join(paths["val_outputs"], "val_pred_progression_stack.tiff")
        save_tiff_stack(
            stack_path,
            np.stack(val_pred_stack, axis=0),
            dtype="uint16",
            scale_per_slice=False,
        )
        history["val_pred_stack"] = {
            "path": stack_path,
            "epochs": val_pred_stack_epochs,
            "num_slices": len(val_pred_stack_epochs),
        }

    save_json(os.path.join(paths["run"], "history.json"), history)
    return {
        "model": model,
        "best_ckpt_path": best_ckpt_path,
        "history": history,
    }
