import torch

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))

def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean()