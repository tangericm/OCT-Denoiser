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
        use_bscan_average_baseline: bool = True,
    ) -> None:
        super().__init__()
        if lam <= 0:
            raise ValueError("lam must be > 0 for stable inverse filtering")

        if h_kernel is None:
            h_kernel = torch.empty(0, dtype=torch.complex64)
        h_kernel = h_kernel.to(torch.complex64).reshape(-1)
        self.register_buffer("h_kernel", h_kernel)
        self.register_buffer("lambda_reg", torch.tensor(lam, dtype=torch.float32))

        self.learnable_correction = learnable_correction
        self.correction_scale = correction_scale
        self.use_bscan_average_baseline = use_bscan_average_baseline
        if learnable_correction:
            correction_len = max(1, h_kernel.numel())
            self.h_correction = nn.Parameter(torch.zeros(2, correction_len, dtype=torch.float32))
        else:
            self.register_parameter("h_correction", None)

    @staticmethod
    def _normalized_bscan_average(x_complex: torch.Tensor) -> torch.Tensor:
        # x_complex: [N, L] where N is flattened A-line count (batch * lateral)
        baseline = x_complex.mean(dim=0)
        scale = baseline.abs().amax().clamp_min(torch.finfo(baseline.real.dtype).eps)
        return baseline / scale

    def _effective_h(self, x_complex: torch.Tensor) -> torch.Tensor:
        length = x_complex.shape[-1]
        device = x_complex.device
        dtype = x_complex.dtype

        if self.h_kernel.numel() == 0:
            if self.use_bscan_average_baseline:
                h = self._normalized_bscan_average(x_complex).detach().to(dtype=dtype)
            else:
                h = torch.ones(length, device=device, dtype=dtype)
        else:
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
        h = self._effective_h(x_complex)
        denom = h.abs().square() + self.lambda_reg.to(device=x.device, dtype=x_complex.real.dtype)
        inv_filter = torch.conj(h) / denom
        deconv = x_complex * inv_filter.unsqueeze(0)

        out = torch.stack([deconv.real, deconv.imag], dim=1)
        if had_width:
            out = out.reshape(b, w, 2, l).permute(0, 2, 3, 1).contiguous()
        return out
