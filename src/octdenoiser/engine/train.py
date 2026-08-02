from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from octdenoiser.configs.default import TrainConfig
from octdenoiser.data.datamodule import BscanDataModule
from octdenoiser.networks import create_model
from octdenoiser.utils.helpers import save_json, seed_all
from octdenoiser.utils.run_manager import make_run_dir

from .early_stopping import EarlyStopping
from .eval import evaluate
from .losses import compute_total_loss


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return device


def _checkpoint(model: torch.nn.Module, cfg: TrainConfig, epoch: int, val_loss: float) -> dict[str, object]:
    return {
        "format_version": 1,
        "arch": "nafnet",
        "base": cfg.base,
        "in_ch": 1,
        "epoch": epoch,
        "val_loss": val_loss,
        "model": model.state_dict(),
    }


def run_training(config: TrainConfig) -> Path:
    """Train NAFNet from adjacent processed B-scans and return the run directory."""

    config.validate()
    device = resolve_device(config.device)
    seed_all(config.seed, deterministic=config.deterministic)
    run_dir = make_run_dir(config.runs_root, config.experiment_name)
    save_json(run_dir / "config.json", asdict(config))

    data = BscanDataModule(config)
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    model = create_model(config.model_name, base=config.base, in_ch=1).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(config.patience, config.min_delta)
    history: list[dict[str, float | int]] = []
    last_val = float("nan")

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                predictions = model(inputs)
                loss = compute_total_loss(
                    predictions,
                    targets,
                    w_charb=config.w_charb,
                    w_grad=config.w_grad,
                )
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = inputs.shape[0]
            running_loss += float(loss.detach()) * batch_size
            samples += batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": running_loss / max(samples, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        should_stop = False
        if epoch % config.val_every == 0:
            last_val = evaluate(
                model,
                val_loader,
                device=device,
                amp=config.amp,
                w_charb=config.w_charb,
                w_grad=config.w_grad,
            )
            record["val_loss"] = last_val
            improved, should_stop = stopper.update(last_val)
            if improved:
                torch.save(
                    _checkpoint(model, config, epoch, last_val),
                    run_dir / "checkpoints" / "best.pt",
                )
        history.append(record)
        save_json(run_dir / "history.json", history)
        if should_stop:
            break

    torch.save(
        _checkpoint(model, config, int(history[-1]["epoch"]), last_val),
        run_dir / "checkpoints" / "final.pt",
    )
    return run_dir
