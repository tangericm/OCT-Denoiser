"""Forward/backward shape and finiteness tests for the new architectures.

All must accept [B, C, H, W] and return [B, 1, H, W], including at sizes that
are not multiples of their internal downsampling factor -- OCT crops are not
guaranteed to be round numbers.
"""
from __future__ import annotations

import pytest
import torch

from octdenoiser.networks import create_model, list_models

SINGLE_FRAME = ["nafnet", "restormer", "ffc_resunet", "aniso_resunet"]


def test_all_new_models_are_registered():
    reg = list_models()
    for name in [*SINGLE_FRAME, "deform_fusion"]:
        assert name in reg, f"{name} not registered"


@pytest.mark.parametrize("name", SINGLE_FRAME)
@pytest.mark.parametrize("in_ch", [1, 2])
def test_single_frame_forward_shape(name, in_ch):
    model = create_model(name, base=8, in_ch=in_ch)
    x = torch.randn(2, in_ch, 64, 48)
    y = model(x)
    assert y.shape == (2, 1, 64, 48), f"{name}: {tuple(y.shape)}"
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", SINGLE_FRAME)
def test_handles_non_multiple_sizes(name):
    """37x53 is prime-ish and divides by nothing; padding must round-trip."""
    model = create_model(name, base=8, in_ch=1)
    y = model(torch.randn(1, 1, 37, 53))
    assert y.shape == (1, 1, 37, 53), f"{name}: {tuple(y.shape)}"
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", SINGLE_FRAME)
def test_backward_produces_finite_grads(name):
    model = create_model(name, base=8, in_ch=1)
    out = model(torch.randn(1, 1, 64, 64))
    out.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name}: no gradients"
    assert all(torch.isfinite(g).all() for g in grads), f"{name}: non-finite gradient"


# --------------------------------------------------------------------------
# Deformable multi-frame fusion
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [3, 5])
def test_deform_fusion_forward(k):
    model = create_model("deform_fusion", base=8, in_ch=k)
    y = model(torch.randn(2, k, 64, 48))
    assert y.shape == (2, 1, 64, 48)
    assert torch.isfinite(y).all()


def test_deform_fusion_starts_as_identity_alignment():
    """Offsets are zero-initialised, so the module begins as plain averaging
    and must learn any warping it applies."""
    model = create_model("deform_fusion", base=8, in_ch=3)
    align = model.align
    assert torch.equal(align.to_offset.weight, torch.zeros_like(align.to_offset.weight))
    assert torch.equal(align.to_offset.bias, torch.zeros_like(align.to_offset.bias))


def test_deform_fusion_uses_every_frame():
    """Perturbing any input frame must change the output, or the extra frames
    are decorative."""
    torch.manual_seed(0)
    model = create_model("deform_fusion", base=8, in_ch=3).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        base_out = model(x)
        for i in range(3):
            perturbed = x.clone()
            perturbed[:, i] += 1.0
            assert not torch.allclose(model(perturbed), base_out, atol=1e-6), (
                f"frame {i} had no effect on the output"
            )


def test_deform_fusion_backward():
    model = create_model("deform_fusion", base=8, in_ch=3)
    model(torch.randn(1, 3, 64, 64)).pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------
# Parameter counts, for the efficiency table
# --------------------------------------------------------------------------
def test_parameter_counts_are_reasonable():
    counts = {}
    for name in [*SINGLE_FRAME, "deform_fusion"]:
        kw = {"in_ch": 5} if name == "deform_fusion" else {"in_ch": 1}
        m = create_model(name, base=32, **kw)
        counts[name] = sum(p.numel() for p in m.parameters())
    for name, n in counts.items():
        assert 1e4 < n < 1e8, f"{name} has {n} parameters, outside a sane range"
