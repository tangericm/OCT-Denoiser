from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from .losses import charbonnier_loss, gradient_l1

@torch.no_grad()
def evaluate(model, loader: DataLoader, *, device: str, w_charb: float, w_grad: float) -> float:
    model.eval()
    loss_acc = 0.0
    n = 0
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = w_charb * charbonnier_loss(pred, y) + w_grad * gradient_l1(pred, y)
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)

    return loss_acc / max(n, 1)
