from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch
from tqdm.auto import tqdm

from octdenoiser.data.dataset import discover_volumes, normalize_frame, read_frame
from octdenoiser.networks import create_model

from .train import resolve_device

OutputDtype = Literal["preserve", "float32", "uint16"]


def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    """Load both release checkpoints and checkpoints written by this package."""

    blob = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise TypeError("Checkpoint must contain a dictionary.")
    state = blob.get("model", blob.get("model_state_dict", blob))
    if not isinstance(state, dict) or "intro.weight" not in state:
        raise ValueError("Checkpoint does not contain a compatible NAFNet state dictionary.")
    arch = str(blob.get("arch", "nafnet"))
    if arch != "nafnet":
        raise ValueError(f"Checkpoint architecture is {arch!r}; this package supports NAFNet.")
    intro = state["intro.weight"]
    if not isinstance(intro, torch.Tensor):
        raise TypeError("Checkpoint intro.weight is not a tensor.")
    base = int(blob.get("base", intro.shape[0]))
    in_ch = int(blob.get("in_ch", intro.shape[1]))
    if in_ch != 1:
        raise ValueError(f"Checkpoint expects {in_ch} input channels; processed B-scans require one.")

    model = create_model("nafnet", base=base, in_ch=1)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def denoise_frame(
    model: torch.nn.Module,
    frame: np.ndarray,
    *,
    device: torch.device,
    amp: bool = True,
) -> np.ndarray:
    normalized, mean, std = normalize_frame(frame)
    tensor = torch.from_numpy(normalized[None, None]).to(device)
    use_amp = amp and device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
        prediction = model(tensor)
    output = prediction[0, 0].float().cpu().numpy()
    return output * std + mean


def _convert_output(frame: np.ndarray, source_dtype: np.dtype[np.generic], output_dtype: OutputDtype) -> np.ndarray:
    if output_dtype == "preserve":
        if np.issubdtype(source_dtype, np.integer):
            limits = np.iinfo(source_dtype)
            return np.rint(np.clip(frame, limits.min, limits.max)).astype(source_dtype)
        return np.asarray(frame, dtype=np.float32)
    if output_dtype == "float32":
        return np.asarray(frame, dtype=np.float32)
    limits = np.iinfo(np.uint16)
    return np.rint(np.clip(frame, limits.min, limits.max)).astype(np.uint16)


def denoise_path(
    input_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    device: str = "auto",
    amp: bool = True,
    output_dtype: OutputDtype = "preserve",
    overwrite: bool = False,
) -> Path:
    """Denoise an ordered TIFF/NPY stack or a folder of processed B-scans."""

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}. Pass --overwrite to replace it.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    volumes = discover_volumes([str(input_path)])
    references = [frame for volume in volumes for frame in volume.frames]
    shapes = {volume.shape for volume in volumes}
    if len(shapes) != 1:
        raise ValueError("All input B-scans must have the same shape for one TIFF-stack output.")

    target_device = resolve_device(device)
    model = load_model(checkpoint, target_device)
    expected_dtype: np.dtype[np.generic] | None = None
    with tifffile.TiffWriter(destination, bigtiff=True) as writer:
        for reference in tqdm(references, desc="denoising", unit="frame"):
            source = read_frame(reference)
            source_dtype = source.dtype
            if output_dtype == "preserve":
                if expected_dtype is None:
                    expected_dtype = source_dtype
                elif source_dtype != expected_dtype:
                    raise ValueError("All input frames must share one dtype when --dtype preserve is used.")
            prediction = denoise_frame(model, source, device=target_device, amp=amp)
            converted = _convert_output(prediction, source_dtype, output_dtype)
            writer.write(converted, photometric="minisblack", contiguous=True, metadata=None)
    return destination
