"""Model kwargs derived from the supervision scheme, in one place.

Why the input width is not `input_mode` alone
---------------------------------------------
THREE different settings each collapse the model input to one channel, and only
one of them is `input_mode="fullband"`:

    supervision="frame_pair"      one full-band frame in, a different frame out
    target_mode="complementary"   one sub-band in, the other sub-band out
    input_mode="fullband"         one full-band frame in

The first two keep `input_mode="bandgap"`. They have to: the complementary
target still needs both sub-bands reconstructed, and the frame-pair path
overrides `input_mode` inside the datamodule rather than on the config, so the
config a checkpoint was written with still says "bandgap".

Deriving the width from `input_mode` alone therefore builds a 2-channel stem for
a 1-channel tensor. Training died on the first forward pass and inference died
on strict state-dict load. Both call sites had made the same mistake
independently, each with a comment saying it mirrored the other -- which is the
argument for deriving it once here instead.
"""
from __future__ import annotations

from octdenoiser.configs.default import MULTILEVEL_MODELS


def effective_in_channels(
    *,
    input_mode: str = "bandgap",
    target_mode: str = "fullband",
    supervision: str = "spectral",
    n_sub_windows: int = 0,
) -> int:
    """Channels the dataset actually emits, for non-multi-level models.

    Multi-level models take `n_sub_channels` instead and build their own stem;
    see `build_model_kwargs`.
    """
    if supervision == "frame_pair":
        return 1
    if input_mode == "fullband":
        return 1
    if target_mode == "complementary":
        return 1
    return 2 + 2 * max(int(n_sub_windows), 0)


def build_model_kwargs(
    *,
    model_name: str,
    base: int,
    input_mode: str = "bandgap",
    target_mode: str = "fullband",
    supervision: str = "spectral",
    n_sub_windows: int = 0,
) -> dict:
    """Constructor kwargs for `create_model`, matched to what the data emits."""
    if model_name in MULTILEVEL_MODELS:
        return {"base": base, "n_sub_channels": 2 * max(int(n_sub_windows), 0)}
    return {
        "base": base,
        "in_ch": effective_in_channels(
            input_mode=input_mode,
            target_mode=target_mode,
            supervision=supervision,
            n_sub_windows=n_sub_windows,
        ),
    }


def build_model_kwargs_from_cfg(cfg, n_sub_windows: int | None = None) -> dict:
    """`build_model_kwargs` sourced from a TrainConfig.

    `n_sub_windows` lives on FolderSpec, not TrainConfig, so it is passed in.
    TrainConfig.__post_init__ already rejects heterogeneous folder_specs, so
    folder_specs[0] is representative.
    """
    if n_sub_windows is None:
        specs = getattr(cfg, "folder_specs", None) or []
        n_sub_windows = getattr(specs[0], "n_sub_windows", 0) if specs else 0
    return build_model_kwargs(
        model_name=cfg.model_name,
        base=cfg.base,
        input_mode=getattr(cfg, "input_mode", "bandgap"),
        target_mode=getattr(cfg, "target_mode", "fullband"),
        supervision=getattr(cfg, "supervision", "spectral"),
        n_sub_windows=n_sub_windows,
    )
