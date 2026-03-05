from __future__ import annotations

import torch
import torch.nn as nn


class AxialDeconvolution(nn.Module):
    """Regularized inverse filtering for complex spectrum inputs.

    Accepts real/imag channels as either [B, 2, L] or [B, 2, L, W].
    """

    def __init__(
        self,
        h_kernel: torch.Tensor | None = None,
        lam: float = 1.0e-3,
        learnable_correction: bool = False,
        correction_scale: float = 1.0e-2,
    ) -> None:
        super().__init__()
        if lam <= 0:
            raise ValueError("lam must be > 0 for stable inverse filtering")

        if h_kernel is None:
            h_kernel = torch.ones(1, dtype=torch.complex64)
        h_kernel = h_kernel.to(torch.complex64).reshape(-1)
        self.register_buffer("h_kernel", h_kernel)
        self.register_buffer("lambda_reg", torch.tensor(lam, dtype=torch.float32))

        self.learnable_correction = learnable_correction
        self.correction_scale = correction_scale
        if learnable_correction:
            self.h_correction = nn.Parameter(torch.zeros(2, h_kernel.numel(), dtype=torch.float32))
        else:
            self.register_parameter("h_correction", None)

    def _effective_h(self, length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h = self.h_kernel.to(device=device, dtype=dtype)
        if h.numel() == 1:
            h = h.expand(length)
        elif h.numel() != length:
            raise ValueError(f"h_kernel length ({h.numel()}) must match input length ({length})")

        if self.h_correction is not None:
            corr = self.h_correction.to(device=device)
            if corr.shape[-1] == 1:
                corr = corr.expand(2, length)
            elif corr.shape[-1] != length:
                raise ValueError(
                    f"learnable correction length ({corr.shape[-1]}) must match input length ({length})"
                )
            h = h + self.correction_scale * torch.complex(corr[0], corr[1]).to(dtype)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (3, 4) or x.shape[1] != 2:
            raise ValueError("x must have shape [B,2,L] or [B,2,L,W]")

        had_width = x.ndim == 4
        if had_width:
            b, c, l, w = x.shape
            x = x.permute(0, 3, 1, 2).reshape(b * w, c, l)

        x_complex = torch.complex(x[:, 0], x[:, 1])
        h = self._effective_h(x_complex.shape[-1], device=x.device, dtype=x_complex.dtype)
        denom = h.abs().square() + self.lambda_reg.to(device=x.device, dtype=x_complex.real.dtype)
        inv_filter = torch.conj(h) / denom
        deconv = x_complex * inv_filter.unsqueeze(0)

        out = torch.stack([deconv.real, deconv.imag], dim=1)
        if had_width:
            out = out.reshape(b, w, 2, l).permute(0, 2, 3, 1).contiguous()
        return out
