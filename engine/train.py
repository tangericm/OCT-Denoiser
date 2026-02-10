from __future__ import annotations

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import Dict, Any

from engine.common import unpack_batch
from engine.early_stopping import EarlyStopping
from engine.losses import charbonnier_loss, gradient_l1, smooth_snr_loss
from engine.eval import evaluate, evaluate_full_frames
from data.datamodule import RawBscanDataModule
from networks import create_model
from utils.helpers import save_json
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



def _scheduled_w_snr_loss(cfg, epoch: int) -> float:
    """Return epoch-specific SNR loss weight.

    Uses a linear ramp from ``w_snr_loss_start`` to ``w_snr_loss_end`` between
    ``w_snr_ramp_start_epoch`` and ``w_snr_ramp_end_epoch`` (inclusive).
    If start/end are not configured, falls back to constant ``w_snr_loss``.
    """
    start = cfg.w_snr_loss_start
    end = cfg.w_snr_loss_end
    if start is None or end is None:
        return float(cfg.w_snr_loss)

    ramp_start = int(cfg.w_snr_ramp_start_epoch)
    ramp_end = int(cfg.w_snr_ramp_end_epoch or cfg.epochs)

    if ramp_end <= ramp_start:
        return float(end if epoch >= ramp_start else start)

    if epoch <= ramp_start:
        return float(start)
    if epoch >= ramp_end:
        return float(end)

    alpha = (epoch - ramp_start) / (ramp_end - ramp_start)
    return float(start + alpha * (end - start))

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
            proc = BscanProcessor(fs.root_folder, fs.to_preprocess_config())
            proc.save_window_figure(window_path)
    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    val_full_loader = dm.val_full_loader()

    # Model
    model = create_model(cfg.model_name, base=cfg.base).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cosine schedule stepped PER BATCH
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    use_cuda_amp = (cfg.amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    best_val = float("inf")
    best_ckpt_path = os.path.join(paths["checkpoints"], "best.pt")
    best_score = float("inf")
    best_score_ckpt_path = os.path.join(paths["checkpoints"], "best_by_score.pt")
    best_score_epoch = None

    history = {"train_loss": [], "val_loss": [], "val_full": []}

    score_baseline: dict[str, float] | None = None


    early_stop = EarlyStopping(
        patience=cfg.early_stop_patience,
        min_delta=cfg.early_stop_min_delta,
        mode="min",
        warmup=cfg.early_stop_warmup_checks,
    )

    plotter = LiveLossPlot(
        out_dir=paths["run"],
        title=f"Loss - {cfg.experiment_name}, Network: {cfg.model_name}",
        filename="loss_curve.png",
        save_every_epoch=False,   # set True if you want per-epoch pngs always
        show_window=True,         # set False if you never want an interactive window
    )
    
    print(f"[INFO] Device={cfg.device}  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    # ---- Timing bookkeeping ----
    epoch_times = []
    train_start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        w_snr_loss_epoch = _scheduled_w_snr_loss(cfg, epoch)
        t0 = time.time()
        running = 0.0
        n = 0

        for batch in train_loader:
            x, y, g = unpack_batch(batch, device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):

                pred = model(x)
                loss = (
                    cfg.w_charb * charbonnier_loss(pred, y)
                    + cfg.w_grad * gradient_l1(pred, y)
                )
                if w_snr_loss_epoch > 0:
                    snr_l, _snr_info = smooth_snr_loss(
                        pred,
                        t_peak=cfg.snr_loss_t_peak,
                        t_bg=cfg.snr_loss_t_bg,
                    )
                    loss = loss + w_snr_loss_epoch * snr_l

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
                w_snr_loss=w_snr_loss_epoch,
                snr_loss_t_peak=cfg.snr_loss_t_peak,
                snr_loss_t_bg=cfg.snr_loss_t_bg,
            )
            val_full = evaluate_full_frames(
                model,
                val_full_loader,
                device=device,
                w_charb=cfg.w_charb,
                w_grad=cfg.w_grad,
                snr_sig_y0=cfg.snr_sig_y0,
                snr_sig_y1=cfg.snr_sig_y1,
                w_snr_loss=w_snr_loss_epoch,
                snr_loss_t_peak=cfg.snr_loss_t_peak,
                snr_loss_t_bg=cfg.snr_loss_t_bg,
            )

            val_snr_raw = float(val_full["snr_pred"])
            val_cnr_raw = float(val_full["cnr_pred"])
            val_snr = val_snr_raw if np.isfinite(val_snr_raw) else 0.0
            val_cnr = val_cnr_raw if np.isfinite(val_cnr_raw) else 0.0

            if score_baseline is None:
                score_baseline = {"val_loss": float(val_loss), "snr": val_snr, "cnr": val_cnr}

            norm_val_loss = (float(val_loss) - score_baseline["val_loss"]) / (abs(score_baseline["val_loss"]) + 1e-8)
            norm_val_snr = (val_snr - score_baseline["snr"]) / (abs(score_baseline["snr"]) + 1e-8)
            norm_val_cnr = (val_cnr - score_baseline["cnr"]) / (abs(score_baseline["cnr"]) + 1e-8)


            composite_score = (
                cfg.score_w_val_loss * norm_val_loss
                - cfg.score_w_snr * norm_val_snr
                - cfg.score_w_cnr * norm_val_cnr
            )

            history["val_loss"].append(
                {
                    "epoch": epoch,
                    "loss": val_loss,
                    "w_snr_loss": w_snr_loss_epoch,
                    "score": composite_score,
                    "norm_val_loss": norm_val_loss,
                    "norm_val_snr": norm_val_snr,
                    "norm_val_cnr": norm_val_cnr,
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
                    "w_snr_loss": w_snr_loss_epoch,
                    "score": composite_score,
                    "norm_val_loss": norm_val_loss,
                    "norm_val_snr": norm_val_snr,
                    "norm_val_cnr": norm_val_cnr,
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
                f"w_snr={w_snr_loss_epoch:.6g} "
                f"score={composite_score:.6f} "
                f"SNR_pred/gt={val_full['snr_pred']:.2f}/{val_full['snr_gt']:.2f}  "
                f"CNR_pred/gt={val_full['cnr_pred']:.2f}/{val_full['cnr_gt']:.2f}  "
                f"time={dt:.5f}s"
            )

            if val_full["sample_pred"] is not None:
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

            if composite_score < best_score:
                best_score = composite_score
                best_score_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "cfg": asdict(cfg),
                        "best_score": best_score,
                    },
                    best_score_ckpt_path,
                )
                print(f"[OK] Saved best-by-score checkpoint: {best_score_ckpt_path}")

            stop_now = early_stop.update(float(val_loss), epoch)
            if stop_now:
                print(
                    f"[EARLY STOP] No val improvement for {early_stop.patience} validation checks. "
                    f"Best={early_stop.best:.6f} at epoch={early_stop.best_epoch}. "
                    f"Best Score={best_score:.6f} at epoch={best_score_epoch}. Stopping at epoch={epoch}."
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
    print(f"[DONE] Best composite score = {best_score:.10f} at epoch {best_score_epoch}")
    history["best_by_score"] = {
        "score": best_score,
        "epoch": best_score_epoch,
        "ckpt_path": best_score_ckpt_path,
        "score_baseline": score_baseline,
        "weights": {
            "val_loss": cfg.score_w_val_loss,
            "snr": cfg.score_w_snr,
            "cnr": cfg.score_w_cnr,
        },
        "snr_loss_schedule": {
            "constant_w_snr_loss": cfg.w_snr_loss,
            "start": cfg.w_snr_loss_start,
            "end": cfg.w_snr_loss_end,
            "ramp_start_epoch": cfg.w_snr_ramp_start_epoch,
            "ramp_end_epoch": cfg.w_snr_ramp_end_epoch,
        },
    }
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
    save_json(os.path.join(paths["run"], "history.json"), history)
    return {
        "model": model,
        "best_ckpt_path": best_ckpt_path,
        "best_score_ckpt_path": best_score_ckpt_path,
        "history": history,
    }
