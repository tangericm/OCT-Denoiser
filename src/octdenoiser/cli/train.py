from __future__ import annotations

import argparse

from octdenoiser.configs.default import TrainConfig
from octdenoiser.engine.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train NAFNet from ordered pairs of already-processed OCT B-scans."
    )
    parser.add_argument("--input", nargs="+", required=True, help="One or more TIFF/NPY stacks or folders.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--name", default="oct-denoiser")
    parser.add_argument("--base", type=int, choices=(32, 64), default=64)
    parser.add_argument("--pair-offset", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--patch-height", type=int, default=256)
    parser.add_argument("--patch-width", type=int, default=256)
    parser.add_argument("--patches-per-pair", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        inputs=tuple(args.input),
        runs_root=args.runs_root,
        experiment_name=args.name,
        base=args.base,
        pair_offset=args.pair_offset,
        group_size=args.group_size,
        train_fraction=args.train_fraction,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        patches_per_pair=args.patches_per_pair,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        augment=not args.no_augment,
        amp=not args.no_amp,
    )
    run_dir = run_training(config)
    print(f"Training complete: {run_dir}")


if __name__ == "__main__":
    main()
