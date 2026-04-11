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

    train = subparsers.add_parser("train", help="Train a streamflow sequence model.")
    train.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    train.add_argument("--seq-len", type=int, default=30)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=1)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--data-dir", default=None)
    train.add_argument("--output-dir", default="outputs")
    train.add_argument("--checkpoint", default="outputs/best_model.pt")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-dir", default=None)
    evaluate.add_argument("--output-dir", default="outputs")

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
