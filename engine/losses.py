from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG16_Weights, VGG19_Weights, vgg16, vgg19

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (pred - target).abs().mean()
    # return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean()


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        *,
        use_vgg19: bool = False,
        layer_ids: list[int] | None = None,
        layer_weights: list[float] | None = None,
        use_charbonnier: bool = False,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if use_vgg19:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT)
            default_layers = [3, 8, 17, 26]
        else:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            default_layers = [3, 8, 15, 22]

        self.features = vgg.features.eval()
        self.features.requires_grad_(False)

        self.layer_ids = layer_ids or default_layers
        if layer_weights is None:
            self.layer_weights = torch.ones(len(self.layer_ids))
        else:
            if len(layer_weights) != len(self.layer_ids):
                raise ValueError("layer_weights must match layer_ids length.")
            self.layer_weights = torch.tensor(layer_weights, dtype=torch.float32)

        self.use_charbonnier = use_charbonnier
        self.charbonnier_eps = charbonnier_eps

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)

        weights = self.layer_weights.to(pred.device)
        loss = pred.new_tensor(0.0)
        for feat_pred, feat_target, weight in zip(pred_features, target_features, weights):
            diff = feat_pred - feat_target
            if self.use_charbonnier:
                layer_loss = torch.sqrt(diff * diff + self.charbonnier_eps**2).mean()
            else:
                layer_loss = diff.abs().mean()
            loss = loss + weight * layer_loss
        return loss / weights.sum().clamp_min(1e-6)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("PerceptualLoss expects tensors shaped [B, C, H, W].")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != 3:
            raise ValueError("PerceptualLoss expects 1 or 3 channels.")
        return (x - self.mean) / self.std

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        layer_set = set(self.layer_ids)
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in layer_set:
                outputs.append(x)
        return outputs
