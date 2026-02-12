from __future__ import annotations

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import Dict, Any

from engine.early_stopping import EarlyStopping
from engine.losses import unpack_batch, compute_total_loss
from engine.eval import evaluate, evaluate_full_frames
from data.datamodule import RawBscanDataModule
from networks import create_model
from utils.helpers import save_json
from utils.io_tiff import save_tiff_stack
from utils.live_plot import LiveLossPlot


def _save_full_frame_val_png(
    pred_img: np.ndarray,
    *,
    snr_pred: float,
    snr_gt: float,
    cnr_pred: float,
    cnr_gt: float,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p1, p99 = np.percentile(pred_img, [1, 99])
    vmin, vmax = float(p1), float(p99)
    if vmax <= vmin:
        vmin, vmax = float(pred_img.min()), float(pred_img.max())

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(pred_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(
        "Validation Prediction\n"
        f"SNR_pred={snr_pred:.2f}dB  SNR_gt={snr_gt:.2f}dB  "
        f"CNR_pred={cnr_pred:.2f}dB  CNR_gt={cnr_gt:.2f}dB",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _make_labeled_val_frame(pred_img: np.ndarray, *, epoch: int) -> np.ndarray:
    p1, p99 = np.percentile(pred_img, [1, 99])
    vmin, vmax = float(p1), float(p99)
    if vmax <= vmin:
        vmin, vmax = float(pred_img.min()), float(pred_img.max())

    h, w = pred_img.shape
    dpi = 100
    fig = plt.figure(figsize=(max(w, 16) / dpi, max(h, 16) / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pred_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        f"Epoch {epoch}",
        transform=ax.transAxes,
        fontsize=12,
        color="white",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    labeled_gray = (0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]).astype(np.float32)
    plt.close(fig)
    return labeled_gray



def run_training(cfg, paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Returns dict with:
      - model (trained model)
      - best_ckpt_path
      - history
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Save config at run root
    save_json(os.path.join(paths["run"], "config.json"), asdict(cfg))

    # Data
    dm = RawBscanDataModule(cfg)
    dm.setup()
    if cfg.folder_specs:
        window_path = os.path.join(paths["run"], "window_figure.png")
        if not os.path.exists(window_path):
            from preprocess import BscanProcessor
            fs = cfg.folder_specs[0]
            proc = BscanProcessor(fs)
            proc.save_window_figure(window_path)
    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    val_full_loader = dm.val_full_loader()

    # Model
    model_kwargs = {"base": cfg.base}
    if cfg.folder_specs:
        n_sub = getattr(cfg.folder_specs[0], "n_sub_windows", 0)
        if n_sub > 0:
            model_kwargs["n_sub_channels"] = 2 * n_sub
    model = create_model(cfg.model_name, **model_kwargs).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cosine schedule stepped PER BATCH
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    use_cuda_amp = (cfg.amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    best_val = float("inf")
    best_ckpt_path = os.path.join(paths["checkpoints"], "best.pt")

    history = {"train_loss": [], "val_loss": [], "val_full": []}
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
        title=f"Loss - {cfg.experiment_name}, Network: {cfg.model_name}",
    )
    
    print(f"[INFO] Device={cfg.device}  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    # ---- Timing bookkeeping ----
    epoch_times = []
    train_start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0

        for batch in train_loader:
            x, y, _meta = unpack_batch(batch, device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                pred = model(x)
                loss = compute_total_loss(
                    pred, y,
                    w_charb=cfg.w_charb,
                    w_grad=cfg.w_grad,
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
            val_loss = evaluate(
                model,
                val_loader,
                device=device,
                w_charb=cfg.w_charb,
                w_grad=cfg.w_grad,
            )
            val_full = evaluate_full_frames(
                model,
                val_full_loader,
                device=device,
                w_charb=cfg.w_charb,
                w_grad=cfg.w_grad,
                snr_sig_y0=cfg.snr_sig_y0,
                snr_sig_y1=cfg.snr_sig_y1,
                snr_sig_stat=cfg.snr_sig_stat,
            )

            history["val_loss"].append(
                {
                    "epoch": epoch,
                    "loss": val_loss,
                    "snr_pred": val_full["snr_pred"],
                    "snr_gt": val_full["snr_gt"],
                    "cnr_pred": val_full["cnr_pred"],
                    "cnr_gt": val_full["cnr_gt"],
                }
            )
            history["val_full"].append(
                {
                    "epoch": epoch,
                    "loss": val_full["val_loss"],
                    "snr_pred": val_full["snr_pred"],
                    "snr_gt": val_full["snr_gt"],
                    "cnr_pred": val_full["cnr_pred"],
                    "cnr_gt": val_full["cnr_gt"],
                }
            )
            plotter.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_snr=val_full["snr_pred"],
            )
            dt = time.time() - t0
            print(
                f"[E{epoch:04d}] train={train_loss:.10f}  "
                f"val_loss={val_loss:.10f} "
                f"SNR_pred/gt={val_full['snr_pred']:.2f}/{val_full['snr_gt']:.2f}  "
                f"CNR_pred/gt={val_full['cnr_pred']:.2f}/{val_full['cnr_gt']:.2f}  "
                f"time={dt:.5f}s"
            )

            if val_full["sample_pred"] is not None:
                val_pred_stack.append(_make_labeled_val_frame(val_full["sample_pred"], epoch=epoch))
                val_pred_stack_epochs.append(epoch)
                out_path = os.path.join(paths["val_outputs"], f"val_pred_epoch_{epoch:04d}.png")
                _save_full_frame_val_png(
                    val_full["sample_pred"],
                    snr_pred=float(val_full["snr_pred"]),
                    snr_gt=float(val_full["snr_gt"]),
                    cnr_pred=float(val_full["cnr_pred"]),
                    cnr_gt=float(val_full["cnr_gt"]),
                    out_path=out_path,
                )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "cfg": asdict(cfg),
                        "best_val": best_val,
                    },
                    best_ckpt_path,
                )
                print(f"[OK] Saved best checkpoint: {best_ckpt_path}")

            stop_now = early_stop.update(float(val_loss), epoch)
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
    total_train_time = time.time() - train_start_time
    mean_epoch_time = sum(epoch_times) / max(len(epoch_times), 1)
    print(
        f"[TIMING] Epochs run = {len(epoch_times)} | "
        f"Mean epoch time = {mean_epoch_time:.10f} s | "
        f"Total training time = {total_train_time:.10f} s"
    )
    history["timing"] = {
        "epoch_times_sec": epoch_times,
        "mean_epoch_time_sec": mean_epoch_time,
        "total_train_time_sec": total_train_time,
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
        print(f"[OK] Saved validation prediction progression stack: {stack_path}")

    save_json(os.path.join(paths["run"], "history.json"), history)
    return {
        "model": model,
        "best_ckpt_path": best_ckpt_path,
        "history": history,
    }
