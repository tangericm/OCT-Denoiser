from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from .losses import charbonnier_loss, gradient_l1, snr_cnr_loss

def _unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        x, y, meta = batch
    else:
        x, y = batch
        meta = None
    return x, y, meta

@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    *,
    device: str,
    w_charb: float,
    w_grad: float,
    w_snr_cnr: float,
    snr_sig_y0: int,
    snr_sig_y1: int,
) -> float:
    model.eval()
    loss_acc = 0.0
    n = 0
    for batch in loader:
        x, y, _meta = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = (
            w_charb * charbonnier_loss(pred, y)
            + w_grad * gradient_l1(pred, y)
            + w_snr_cnr * snr_cnr_loss(
                pred,
                y,
                sig_y0=snr_sig_y0,
                sig_y1=snr_sig_y1,
            )
        )
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)

    return loss_acc / max(n, 1)
