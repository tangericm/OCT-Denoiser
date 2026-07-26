"""Model input width must follow the supervision scheme, not `input_mode`.

Both the frame-pair and the complementary scheme emit ONE channel while leaving
`input_mode="bandgap"` on the config. Deriving the width from `input_mode` alone
built a 2-channel stem for a 1-channel tensor: training died on the first
forward pass and inference died on strict state-dict load.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from octdenoiser.configs.default import FolderSpec, TrainConfig
from octdenoiser.networks import create_model
from octdenoiser.networks.build import (
    build_model_kwargs,
    build_model_kwargs_from_cfg,
    effective_in_channels,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # The default spectral scheme: both parent sub-bands in.
        (dict(), 2),
        (dict(n_sub_windows=2), 6),
        # One channel, three different ways.
        (dict(input_mode="fullband"), 1),
        (dict(target_mode="complementary"), 1),
        (dict(supervision="frame_pair"), 1),
        # The two regressions: input_mode still reads "bandgap" in both.
        (dict(input_mode="bandgap", target_mode="complementary"), 1),
        (dict(input_mode="bandgap", supervision="frame_pair"), 1),
    ],
)
def test_effective_in_channels(kwargs, expected):
    assert effective_in_channels(**kwargs) == expected


def test_multilevel_gets_sub_channels_not_in_ch():
    kw = build_model_kwargs(model_name="resunet_pseudo3d_multilevel", base=8, n_sub_windows=2)
    assert kw == {"base": 8, "n_sub_channels": 4}
    assert "in_ch" not in kw, "the multi-level stem sizes itself from n_sub_channels"


@pytest.mark.parametrize(
    ("supervision", "target_mode"),
    [("frame_pair", "fullband"), ("spectral", "complementary")],
)
def test_one_channel_schemes_build_a_model_that_accepts_their_data(supervision, target_mode):
    """The end-to-end check: construct, then push the tensor the dataset emits."""
    kw = build_model_kwargs(
        model_name="resunet_pseudo3d", base=8, input_mode="bandgap",
        target_mode=target_mode, supervision=supervision,
    )
    assert kw["in_ch"] == 1
    model = create_model("resunet_pseudo3d", **kw)
    with torch.no_grad():
        out = model(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, 1, 32, 32)


def test_from_cfg_reads_supervision_off_the_config():
    cfg = TrainConfig(
        supervision="frame_pair",
        model_name="resunet_pseudo3d",
        base=8,
        folder_specs=[FolderSpec(root_folder="r", data_folder="d", pixels=256,
                                 alines=64, crop_depth=(0, 128))],
    )
    assert build_model_kwargs_from_cfg(cfg)["in_ch"] == 1


def test_from_cfg_reads_n_sub_windows_off_the_folder_spec():
    spec = FolderSpec(root_folder="r", data_folder="d", pixels=256, alines=64,
                      crop_depth=(0, 128), n_sub_windows=2, sub_window_spread=0.5)
    cfg = TrainConfig(model_name="resunet_pseudo3d", base=8, folder_specs=[spec])
    assert build_model_kwargs_from_cfg(cfg)["in_ch"] == 6


# --------------------------------------------------------------------------
# The call sites must actually USE the helper
# --------------------------------------------------------------------------
# Testing the helper alone leaves the original bug reachable: train.py and
# infer.py each derived the width inline, and both carried a comment claiming to
# mirror the other. Reverting either call site would keep every test above green,
# so pin the call sites too.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "octdenoiser"
_CALL_SITES = ["engine/train.py", "engine/infer.py"]


def _tree(rel: str) -> ast.Module:
    return ast.parse((_SRC / rel).read_text(encoding="utf-8"))


def _calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


@pytest.mark.parametrize("rel", _CALL_SITES)
def test_call_site_delegates_the_width_to_build(rel):
    tree = _tree(rel)
    helpers = (_calls_named(tree, "build_model_kwargs")
               + _calls_named(tree, "build_model_kwargs_from_cfg"))
    assert helpers, f"{rel} must derive model kwargs via networks/build.py"


@pytest.mark.parametrize("rel", _CALL_SITES)
def test_call_site_never_passes_a_hand_computed_width(rel):
    """`create_model(..., in_ch=<something local>)` is the regression itself."""
    for call in _calls_named(_tree(rel), "create_model"):
        named = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert not named & {"in_ch", "n_sub_channels"}, (
            f"{rel} passes {named & {'in_ch', 'n_sub_channels'}} explicitly to "
            f"create_model; the width belongs to networks/build.py alone"
        )
        assert any(kw.arg is None for kw in call.keywords), (
            f"{rel} must splat the kwargs built by networks/build.py"
        )
