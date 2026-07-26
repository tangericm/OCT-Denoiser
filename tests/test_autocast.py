"""Every registered model must survive bf16 autocast.

Training runs under `torch.amp.autocast(dtype=torch.bfloat16)`, so a model that
only works in fp32 is not merely slower -- it dies on the first forward pass.
`ffc_resunet` did exactly that and was lost from an entire architecture sweep:

    RuntimeError: Expected both inputs to be Half, Float or Double tensors
    but got BFloat16 and BFloat16

Its Fourier branch cast the rfft2 INPUT to float32, but the 1x1 conv between
the forward and inverse transform still ran under autocast and handed bf16 back
to torch.complex. Nothing caught it, because every other test builds models in
fp32. These tests run each registered backbone through autocast on CPU, which
reproduces the failure without needing a GPU.
"""
from __future__ import annotations

import pytest
import torch

from octdenoiser.networks import create_model, list_models

# deform_fusion stacks K neighbouring frames; the rest consume one.
MULTI_FRAME_IN_CH = {"deform_fusion": 5}
# The multi-level stem sizes itself from n_sub_channels rather than in_ch.
SUB_CHANNEL_MODELS = {"resunet_pseudo3d_multilevel": 4}

# torchvision's deform_conv2d has a CUDA bf16 kernel but no CPU one:
#   NotImplementedError: "deformable_im2col" not implemented for 'BFloat16'
# That is an upstream CPU-kernel gap, not a defect here -- deform_fusion trains
# fine under CUDA bf16. Excluded from the CPU sweep and covered by the
# CUDA-gated test below, so the gap stays visible rather than silently passing.
NO_CPU_BF16_KERNEL = {"deform_fusion"}

CPU_MODELS = sorted(set(list_models()) - NO_CPU_BF16_KERNEL)


def _build(name: str) -> tuple[torch.nn.Module, int]:
    if name in SUB_CHANNEL_MODELS:
        n_sub = SUB_CHANNEL_MODELS[name]
        return create_model(name, base=8, n_sub_channels=n_sub), 2 + n_sub
    in_ch = MULTI_FRAME_IN_CH.get(name, 1)
    return create_model(name, base=8, in_ch=in_ch), in_ch


def test_every_registered_model_is_accounted_for():
    """A new backbone must land in one bucket or the other, never neither."""
    covered = set(CPU_MODELS) | NO_CPU_BF16_KERNEL
    assert covered == set(list_models()), f"unaccounted models: {set(list_models()) - covered}"


@pytest.mark.parametrize("name", CPU_MODELS)
def test_forward_survives_bf16_autocast(name):
    model, in_ch = _build(name)
    model.eval()
    x = torch.randn(1, in_ch, 32, 32)

    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = model(x)

    assert out.shape == (1, 1, 32, 32), f"{name} returned {tuple(out.shape)}"
    assert torch.isfinite(out.float()).all(), f"{name} produced non-finite values under bf16"


@pytest.mark.parametrize("name", CPU_MODELS)
def test_backward_survives_bf16_autocast(name):
    """The forward is not enough -- training also differentiates through it."""
    model, in_ch = _build(name)
    x = torch.randn(1, in_ch, 32, 32)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x).float().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name} produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads), f"{name} produced non-finite gradients"


@pytest.mark.needs_cuda
@pytest.mark.parametrize("name", sorted(NO_CPU_BF16_KERNEL))
def test_cuda_only_models_survive_bf16_autocast(name):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: torchvision has no CPU bf16 deform_conv2d kernel")
    model, in_ch = _build(name)
    model = model.cuda()
    x = torch.randn(1, in_ch, 32, 32, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x).float().mean()
    loss.backward()

    assert torch.isfinite(loss), f"{name} produced a non-finite loss under CUDA bf16"


def test_fourier_branch_stays_in_float32():
    """The specific defect: phase must not round-trip through a bf16 mantissa.

    Pinning the mechanism, not just the absence of a crash -- catching the
    exception alone would still pass if someone "fixed" it by casting the
    complex tensor down instead of keeping the branch in fp32.
    """
    from octdenoiser.networks.ffc_resunet import FourierUnit

    unit = FourierUnit(4, 4).eval()
    x = torch.randn(1, 4, 16, 16)

    with torch.no_grad():
        want = unit(x)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            got = unit(x)

    assert got.dtype == x.dtype, "the unit must hand back the dtype it was given"
    # Same arithmetic either way: if autocast reached the transforms, the phase
    # error would push this far past a float32 tolerance.
    assert torch.allclose(want, got, atol=1e-5), (
        "autocast changed the Fourier branch's result; it must run in float32"
    )
