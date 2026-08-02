from __future__ import annotations

import argparse
from pathlib import Path

from octdenoiser.engine.infer import denoise_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Denoise an already-processed OCT B-scan, stack, or folder with NAFNet."
    )
    parser.add_argument("--input", required=True, help="TIFF/NPY stack or folder of ordered B-scans.")
    parser.add_argument("--checkpoint", required=True, help="NAFNet .pt checkpoint.")
    parser.add_argument("--output", help="Output TIFF stack. Defaults beside the input.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0.")
    parser.add_argument("--dtype", choices=("preserve", "float32", "uint16"), default="preserve")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed-precision inference.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    return parser


def default_output(input_path: str) -> Path:
    source = Path(input_path).expanduser().resolve()
    name = source.stem if source.is_file() else source.name
    return source.parent / f"{name}_denoised.tiff"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output) if args.output else default_output(args.input)
    result = denoise_path(
        args.input,
        args.checkpoint,
        output,
        device=args.device,
        amp=not args.no_amp,
        output_dtype=args.dtype,
        overwrite=args.overwrite,
    )
    print(f"Saved {result}")


if __name__ == "__main__":
    main()
