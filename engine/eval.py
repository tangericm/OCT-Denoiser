from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from .losses import charbonnier_loss, gradient_l1
from .metrics import roi_bounds, bg_bounds, roi_snr_cnr

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
        )
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)

    return loss_acc / max(n, 1)


@torch.no_grad()
def evaluate_full_frames(
    model,
    loader: DataLoader,
    *,
    device: str,
    w_charb: float,
    w_grad: float,
    snr_sig_y0: int,
    snr_sig_y1: int,
) -> dict[str, float | np.ndarray | None]:
    model.eval()
    loss_acc = 0.0
    snr_cnr_acc = 0.0
    n = 0
    snr_pred_list: list[float] = []
    snr_gt_list: list[float] = []
    cnr_pred_list: list[float] = []
    cnr_gt_list: list[float] = []
    sample_pred: np.ndarray | None = None

    for batch in loader:
        x, y, _meta = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = (
            w_charb * charbonnier_loss(pred, y)
            + w_grad * gradient_l1(pred, y)
        )
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)

        pred_np = pred.detach().cpu().numpy()
        gt_np = y.detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]
            gt_img = gt_np[i, 0]
            h, w = pred_img.shape
            sig_roi = roi_bounds(h, w, snr_sig_y0, snr_sig_y1)
            sy0, sy1, sx0, sx1 = sig_roi
            bg_roi = bg_bounds(h, w, x0=sx0, x1=sx1)

            snr_pred, cnr_pred = roi_snr_cnr(pred_img, sig_roi, bg_roi)
            snr_gt, cnr_gt = roi_snr_cnr(gt_img, sig_roi, bg_roi)
            snr_pred_list.append(snr_pred)
            snr_gt_list.append(snr_gt)
            cnr_pred_list.append(cnr_pred)
            cnr_gt_list.append(cnr_gt)

            if sample_pred is None:
                sample_pred = pred_img.copy()

    val_loss = loss_acc / max(n, 1)
    snr_cnr_loss_val = snr_cnr_acc / max(n, 1)
    snr_pred = float(np.mean(snr_pred_list)) if snr_pred_list else float("nan")
    snr_gt = float(np.mean(snr_gt_list)) if snr_gt_list else float("nan")
    cnr_pred = float(np.mean(cnr_pred_list)) if cnr_pred_list else float("nan")
    cnr_gt = float(np.mean(cnr_gt_list)) if cnr_gt_list else float("nan")

    return {
        "val_loss": val_loss,
        "snr_cnr_loss": snr_cnr_loss_val,
        "snr_pred": snr_pred,
        "snr_gt": snr_gt,
        "cnr_pred": cnr_pred,
        "cnr_gt": cnr_gt,
        "sample_pred": sample_pred,
    }
