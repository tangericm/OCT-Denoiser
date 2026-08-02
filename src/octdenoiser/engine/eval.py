from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .losses import compute_total_loss


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    amp: bool,
    w_charb: float,
    w_grad: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    use_amp = amp and device.type == "cuda"
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            predictions = model(inputs)
            loss = compute_total_loss(predictions, targets, w_charb=w_charb, w_grad=w_grad)
        total += float(loss) * inputs.shape[0]
        count += inputs.shape[0]
    if count == 0:
        raise ValueError("Validation loader is empty.")
    return total / count
