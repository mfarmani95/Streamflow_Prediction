"""Command-line interface for the assignment workflows."""

from __future__ import annotations

import argparse
from pprint import pprint

from dataset.minicamels_dataset import summarize_dataset


def build_parser() -> argparse.ArgumentParser:
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
    train.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    train.add_argument("--seq-len", type=int, default=30)
    train.add_argument("--forecast-horizon", type=int, default=1)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=1)
    train.add_argument("--nhead", type=int, default=4)
    train.add_argument("--dim-feedforward", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--loss", choices=["mse", "mae"], default="mse")
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--min-delta", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--limit-basins", type=int, default=None)
    train.add_argument("--data-dir", default=None)
    train.add_argument("--output-dir", default="outputs")
    train.add_argument("--checkpoint", default="outputs/best_model.pt")

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
    else:
        parser.error(f"Unknown command: {args.command}")
