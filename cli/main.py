"""Command-line interface for the assignment workflows."""

from __future__ import annotations

import argparse
import sys
from pprint import pprint
from typing import Optional

from dataset.minicamels_dataset import summarize_dataset
from util.config import (
    DEFAULT_DYNAMIC_INPUTS,
    DEFAULT_STATIC_ATTRIBUTES,
    DEFAULT_TARGET_VARIABLE,
    load_yaml_config,
    train_defaults_from_config,
)


def _default(defaults: dict, key: str, fallback):
    return defaults.get(key, fallback)


def _extract_train_config_path(argv: list[str]) -> Optional[str]:
    if not argv or argv[0] != "train":
        return None

    for index, value in enumerate(argv):
        if value == "--config" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--config="):
            return value.split("=", 1)[1]
    return None


def _train_defaults_from_argv(argv: list[str]) -> dict:
    config_path = _extract_train_config_path(argv)
    if config_path is None:
        return {}

    defaults = train_defaults_from_config(load_yaml_config(config_path))
    defaults["config"] = config_path
    return defaults


def build_parser(train_defaults: Optional[dict] = None) -> argparse.ArgumentParser:
    train_defaults = train_defaults or {}
    parser = argparse.ArgumentParser(
        description="Streamflow prediction with sequence models on MiniCAMELS."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser(
        "summarize-data", help="Inspect MiniCAMELS metadata and assignment data choices."
    )
    summarize.add_argument("--data-dir", default=None, help="Optional MiniCAMELS data path.")
    summarize.add_argument("--make-plots", action="store_true", help="Create exploratory plots.")
    summarize.add_argument("--output-dir", default="outputs", help="Directory for figures.")

    train = subparsers.add_parser("train", help="Train a streamflow sequence model.")
    train.add_argument("--config", default=_default(train_defaults, "config", None))
    train.add_argument("--model", choices=["lstm", "transformer"], default=_default(train_defaults, "model", "lstm"))
    train.add_argument("--seq-len", type=int, default=_default(train_defaults, "seq_len", 30))
    train.add_argument("--forecast-horizon", type=int, default=_default(train_defaults, "forecast_horizon", 1))
    train.add_argument("--dynamic-inputs", nargs="+", default=_default(train_defaults, "dynamic_inputs", list(DEFAULT_DYNAMIC_INPUTS)))
    train.add_argument("--target-variable", default=_default(train_defaults, "target_variable", DEFAULT_TARGET_VARIABLE))
    train.add_argument("--static-attributes", nargs="+", default=_default(train_defaults, "static_attributes", list(DEFAULT_STATIC_ATTRIBUTES)))
    train.add_argument("--epochs", type=int, default=_default(train_defaults, "epochs", 20))
    train.add_argument("--batch-size", type=int, default=_default(train_defaults, "batch_size", 64))
    train.add_argument("--lr", type=float, default=_default(train_defaults, "lr", 1e-3))
    train.add_argument("--hidden-size", type=int, default=_default(train_defaults, "hidden_size", 64))
    train.add_argument("--num-layers", type=int, default=_default(train_defaults, "num_layers", 1))
    train.add_argument("--nhead", type=int, default=_default(train_defaults, "nhead", 4))
    train.add_argument("--dim-feedforward", type=int, default=_default(train_defaults, "dim_feedforward", 128))
    train.add_argument("--dropout", type=float, default=_default(train_defaults, "dropout", 0.0))
    train.add_argument(
        "--loss",
        choices=["mse", "masked_mse", "mae", "masked_mae", "kge"],
        default=_default(train_defaults, "loss", "mse"),
    )
    train.add_argument("--weight-decay", type=float, default=_default(train_defaults, "weight_decay", 0.0))
    train.add_argument("--grad-clip", type=float, default=_default(train_defaults, "grad_clip", 1.0))
    train.add_argument("--patience", type=int, default=_default(train_defaults, "patience", 10))
    train.add_argument("--min-delta", type=float, default=_default(train_defaults, "min_delta", 0.0))
    train.add_argument("--seed", type=int, default=_default(train_defaults, "seed", 42))
    train.add_argument("--device", default=_default(train_defaults, "device", "auto"))
    train.add_argument("--num-workers", type=int, default=_default(train_defaults, "num_workers", 0))
    train.add_argument("--limit-basins", type=int, default=_default(train_defaults, "limit_basins", None))
    train.add_argument("--train-basin-count", type=int, default=_default(train_defaults, "train_basin_count", None))
    train.add_argument("--val-basin-count", type=int, default=_default(train_defaults, "val_basin_count", None))
    train.add_argument("--test-basin-count", type=int, default=_default(train_defaults, "test_basin_count", None))
    train.add_argument("--data-dir", default=_default(train_defaults, "data_dir", None))
    train.add_argument("--output-dir", default=_default(train_defaults, "output_dir", "outputs"))
    train.add_argument("--checkpoint", default=_default(train_defaults, "checkpoint", "outputs/best_model.pt"))

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-dir", default=None)
    evaluate.add_argument("--output-dir", default="outputs")
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--num-workers", type=int, default=0)

    plot = subparsers.add_parser("plot", help="Generate diagnostic plots.")
    plot.add_argument("--checkpoint", default="outputs/best_model.pt")
    plot.add_argument("--data-dir", default=None)
    plot.add_argument("--output-dir", default="outputs")

    sweep = subparsers.add_parser("sweep", help="Run a hyperparameter sweep.")
    sweep.add_argument("--config", default="configs/default.yaml")
    sweep.add_argument("--seq-lens", nargs="+", type=int, default=[30, 60, 90, 120, 360])
    sweep.add_argument("--hidden-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    sweep.add_argument("--batch-sizes", nargs="+", type=int, default=[26, 32, 64, 128])
    sweep.add_argument("--loss", choices=["mse", "masked_mse", "mae", "masked_mae", "kge"], default="kge")
    sweep.add_argument("--lr", type=float, default=1e-3)
    sweep.add_argument("--output-root", default="outputs/sweeps")
    sweep.add_argument("--eval-batch-size", type=int, default=256)
    sweep.add_argument("--device", default=None)
    sweep.add_argument("--num-workers", type=int, default=None)
    sweep.add_argument("--limit-basins", type=int, default=None)
    sweep.add_argument("--train-basin-count", type=int, default=None)
    sweep.add_argument("--val-basin-count", type=int, default=None)
    sweep.add_argument("--test-basin-count", type=int, default=None)
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--skip-existing", action="store_true")
    sweep.add_argument("--no-evaluate", dest="evaluate", action="store_false")
    sweep.set_defaults(evaluate=True)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser(train_defaults=_train_defaults_from_argv(argv))
    args = parser.parse_args(argv)

    if args.command == "summarize-data":
        pprint(summarize_dataset(data_dir=args.data_dir))
        if args.make_plots:
            from visualization import create_exploratory_plots

            create_exploratory_plots(
                data_dir=args.data_dir,
                output_dir=f"{args.output_dir}/exploratory",
            )
    elif args.command == "train":
        from training.trainer import train_model

        train_model(args)
    elif args.command == "evaluate":
        from evaluation.evaluator import evaluate_checkpoint

        evaluate_checkpoint(args)
    elif args.command == "plot":
        from visualization import create_all_plots

        create_all_plots(args)
    elif args.command == "sweep":
        from training.sweep import run_sweep

        run_sweep(args)
    else:
        parser.error(f"Unknown command: {args.command}")
