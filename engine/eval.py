from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from engine.common import unpack_batch
from .losses import charbonnier_loss, gradient_l1, smooth_snr_loss
from .metrics import roi_bounds, bg_bounds, roi_snr_cnr, to_physical_intensity


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    *,
    device: str,
    w_charb: float,
    w_grad: float,
    w_snr_loss: float = 0.0,
    snr_loss_t_peak: float = 0.1,
    snr_loss_t_bg: float = 0.1,
) -> float:
    model.eval()
    loss_acc = 0.0
    n = 0
    for batch in loader:
        x, y, _meta = unpack_batch(batch, device)
        pred = model(x)
        loss = w_charb * charbonnier_loss(pred, y) + w_grad * gradient_l1(pred, y)
        if w_snr_loss > 0:
            snr_l, _ = smooth_snr_loss(pred, t_peak=snr_loss_t_peak, t_bg=snr_loss_t_bg)
            loss = loss + w_snr_loss * snr_l
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
    snr_sig_stat: str = "max",
    w_snr_loss: float = 0.0,
    snr_loss_t_peak: float = 0.1,
    snr_loss_t_bg: float = 0.1,
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
        loss = w_charb * charbonnier_loss(pred, y) + w_grad * gradient_l1(pred, y)
        if w_snr_loss > 0:
            snr_l, _ = smooth_snr_loss(pred, t_peak=snr_loss_t_peak, t_bg=snr_loss_t_bg)
            loss = loss + w_snr_loss * snr_l
        loss_acc += float(loss.item()) * x.size(0)
        n += x.size(0)

        pred_np = pred.detach().cpu().numpy()
        gt_np = y.detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]
            gt_img = gt_np[i, 0]

            sample_meta = None
            if isinstance(meta, dict):
                target_mu = meta.get("target_mu")
                target_sd = meta.get("target_sd")
                log_eps = meta.get("log_eps")
                if target_mu is not None and target_sd is not None and log_eps is not None:
                    sample_meta = {
                        "target_mu": float(target_mu[i]) if np.ndim(target_mu) > 0 else float(target_mu),
                        "target_sd": float(target_sd[i]) if np.ndim(target_sd) > 0 else float(target_sd),
                        "log_eps": float(log_eps[i]) if np.ndim(log_eps) > 0 else float(log_eps),
                    }
            elif isinstance(meta, (list, tuple)) and i < len(meta) and isinstance(meta[i], dict):
                m = meta[i]
                if "target_mu" in m and "target_sd" in m and "log_eps" in m:
                    sample_meta = {
                        "target_mu": float(m["target_mu"]),
                        "target_sd": float(m["target_sd"]),
                        "log_eps": float(m["log_eps"]),
                    }

            if sample_meta is not None:
                sample_meta = {
                    "target_mu": float(sample_meta["target_mu"]),
                    "target_sd": float(sample_meta["target_sd"]),
                    "log_eps": float(sample_meta["log_eps"]),
                }

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
        return float(np.nanmean(np.where(np.isfinite(a), a, np.nan)))

    return {
        "val_loss": val_loss,
        "snr_pred": _safe_mean(snr_pred_list),
        "snr_gt": _safe_mean(snr_gt_list),
        "cnr_pred": _safe_mean(cnr_pred_list),
        "cnr_gt": _safe_mean(cnr_gt_list),
        "sample_pred": sample_pred,
    }
