import torch

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    pred_linear = (10 ** pred) - 1e-6
    target_linear = (10 ** target) - 1e-6
    linear_diff = pred_linear - target_linear
    loss_linear = torch.mean(torch.sqrt(linear_diff ** 2 + eps**2))
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2)) #+ 1e-4*loss_linear


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_linear = (10 ** pred) - 1e-6
    target_linear = (10 ** target) - 1e-6
    dy_p_linear = pred_linear[..., 1:, :] - pred_linear[..., :-1, :]
    dx_p_linear = pred_linear[..., :, 1:] - pred_linear[..., :, :-1]
    dy_t_linear = target_linear[..., 1:, :] - target_linear[..., :-1, :]
    dx_t_linear = target_linear[..., :, 1:] - target_linear[..., :, :-1]
    linear_grad_loss = (dy_p_linear - dy_t_linear).abs().mean() + (dx_p_linear - dx_t_linear).abs().mean()

    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean() #+ 1e-4*linear_grad_loss
