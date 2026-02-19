from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import unpack_batch, compute_total_loss
from .metrics import roi_bounds, bg_bounds, roi_snr_cnr, to_physical_intensity


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
        x, y, _meta = unpack_batch(batch, device)
        pred = model(x)
        loss = compute_total_loss(
            pred, y,
            w_charb=w_charb, w_grad=w_grad,
        )
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_acc / max(n, 1)


def _extract_sample_meta(meta, i: int) -> dict | None:
    """Extract per-sample metadata dict from batch metadata (tuple of dicts)."""
    if isinstance(meta, (list, tuple)) and i < len(meta) and isinstance(meta[i], dict):
        m = meta[i]
        if "target_mu" in m and "target_sd" in m and "log_eps" in m:
            return {
                "target_mu": float(m["target_mu"]),
                "target_sd": float(m["target_sd"]),
                "log_eps": float(m["log_eps"]),
            }
    return None


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
    snr_sig_stat: str = "max",
) -> dict[str, float | np.ndarray | None]:
    model.eval()
    loss_acc = 0.0
    n = 0
    snr_pred_list: list[float] = []
    snr_gt_list: list[float] = []
    cnr_pred_list: list[float] = []
    cnr_gt_list: list[float] = []
    sample_pred: np.ndarray | None = None

    for batch in loader:
        x, y, meta = unpack_batch(batch, device)
        pred = model(x)
        loss = compute_total_loss(
            pred, y,
            w_charb=w_charb, w_grad=w_grad,
        )
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)

        pred_np = pred.detach().cpu().numpy()
        gt_np = y.detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]
            gt_img = gt_np[i, 0]

            sample_meta = _extract_sample_meta(meta, i)
            pred_eval = to_physical_intensity(pred_img, sample_meta)
            gt_eval = to_physical_intensity(gt_img, sample_meta)

            h, w = pred_eval.shape
            sig_roi = roi_bounds(h, w, snr_sig_y0, snr_sig_y1)
            sy0, sy1, sx0, sx1 = sig_roi
            bg_roi = bg_bounds(h, w, x0=sx0, x1=sx1)

            snr_pred, cnr_pred = roi_snr_cnr(pred_eval, sig_roi, bg_roi, sig_stat=snr_sig_stat)
            snr_gt, cnr_gt = roi_snr_cnr(gt_eval, sig_roi, bg_roi, sig_stat="max")
            snr_pred_list.append(snr_pred)
            snr_gt_list.append(snr_gt)
            cnr_pred_list.append(cnr_pred)
            cnr_gt_list.append(cnr_gt)

            if sample_pred is None:
                sample_pred = pred_img.copy()

    val_loss = loss_acc / max(n, 1)

    def _safe_mean(arr):
        if len(arr) == 0:
            return float("nan")
        a = np.asarray(arr, dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return float("nan")
        return float(finite.mean())

    return {
        "val_loss": val_loss,
        "snr_pred": _safe_mean(snr_pred_list),
        "snr_gt": _safe_mean(snr_gt_list),
        "cnr_pred": _safe_mean(cnr_pred_list),
        "cnr_gt": _safe_mean(cnr_gt_list),
        "sample_pred": sample_pred,
    }
