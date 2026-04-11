"""Hyperparameter sweep runner."""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

from util.config import load_yaml_config, train_defaults_from_config


def _base_train_args(config_path: str) -> Dict[str, Any]:
    defaults = train_defaults_from_config(load_yaml_config(config_path))
    return {
        "config": config_path,
        "model": defaults.get("model", "lstm"),
        "seq_len": defaults.get("seq_len", 30),
        "forecast_horizon": defaults.get("forecast_horizon", 1),
        "dynamic_inputs": defaults.get(
            "dynamic_inputs",
            ["prcp", "tmax", "tmin", "srad", "vp"],
        ),
        "target_variable": defaults.get("target_variable", "qobs"),
        "static_attributes": defaults.get("static_attributes"),
        "epochs": defaults.get("epochs", 20),
        "batch_size": defaults.get("batch_size", 64),
        "lr": defaults.get("lr", 1e-3),
        "hidden_size": defaults.get("hidden_size", 64),
        "num_layers": defaults.get("num_layers", 1),
        "nhead": defaults.get("nhead", 4),
        "dim_feedforward": defaults.get("dim_feedforward", 128),
        "dropout": defaults.get("dropout", 0.0),
        "loss": defaults.get("loss", "mse"),
        "weight_decay": defaults.get("weight_decay", 0.0),
        "grad_clip": defaults.get("grad_clip", 1.0),
        "patience": defaults.get("patience", 10),
        "min_delta": defaults.get("min_delta", 0.0),
        "seed": defaults.get("seed", 42),
        "device": defaults.get("device", "auto"),
        "num_workers": defaults.get("num_workers", 0),
        "limit_basins": defaults.get("limit_basins"),
        "train_basin_count": defaults.get("train_basin_count"),
        "val_basin_count": defaults.get("val_basin_count"),
        "test_basin_count": defaults.get("test_basin_count"),
        "data_dir": defaults.get("data_dir"),
        "output_dir": defaults.get("output_dir", "outputs"),
        "checkpoint": defaults.get("checkpoint", "outputs/best_model.pt"),
    }


def _run_name(seq_len: int, hidden_size: int, batch_size: int) -> str:
    return f"seq{seq_len:03d}_hidden{hidden_size:03d}_batch{batch_size:03d}"


def _read_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    report = json.loads(path.read_text())
    overall = report.get("overall", {})
    return {
        "mse": overall.get("mse"),
        "mae": overall.get("mae"),
        "rmse": overall.get("rmse"),
        "nse": overall.get("nse"),
        "kge": overall.get("kge"),
        "best_basin_by_nse": report.get("best_basin_by_nse"),
        "worst_basin_by_nse": report.get("worst_basin_by_nse"),
    }


def _write_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "seq_len",
        "hidden_size",
        "batch_size",
        "loss",
        "lr",
        "output_dir",
        "checkpoint",
        "status",
        "mse",
        "mae",
        "rmse",
        "nse",
        "kge",
        "best_basin_by_nse",
        "worst_basin_by_nse",
    ]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_sweep(args: Namespace) -> None:
    """Run the requested grid of sequence length, hidden size, and batch size."""
    base = _base_train_args(args.config)
    output_root = Path(args.output_root)
    summary_path = output_root / "sweep_results.csv"
    rows: List[Dict[str, Any]] = []

    total = len(args.seq_lens) * len(args.hidden_sizes) * len(args.batch_sizes)
    print(f"Prepared {total} runs.")

    for seq_len in args.seq_lens:
        for hidden_size in args.hidden_sizes:
            for batch_size in args.batch_sizes:
                run_name = _run_name(seq_len, hidden_size, batch_size)
                run_dir = output_root / run_name
                checkpoint = run_dir / "best_model.pt"
                metrics_path = run_dir / "test_metrics.json"
                row = {
                    "run_name": run_name,
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "batch_size": batch_size,
                    "loss": args.loss,
                    "lr": args.lr,
                    "output_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "status": "pending",
                }

                if args.dry_run:
                    print(
                        "DRY RUN | "
                        f"{run_name}: seq_len={seq_len}, hidden_size={hidden_size}, "
                        f"batch_size={batch_size}, output_dir={run_dir}"
                    )
                    row["status"] = "dry_run"
                    rows.append(row)
                    continue

                if args.skip_existing and metrics_path.exists():
                    print(f"Skipping existing run: {run_name}")
                    row.update(_read_metrics(metrics_path))
                    row["status"] = "skipped_existing"
                    rows.append(row)
                    _write_summary(rows, summary_path)
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)
                train_args = Namespace(
                    **{
                        **base,
                        "seq_len": seq_len,
                        "hidden_size": hidden_size,
                        "batch_size": batch_size,
                        "loss": args.loss,
                        "lr": args.lr,
                        "output_dir": str(run_dir),
                        "checkpoint": str(checkpoint),
                        "device": args.device or base.get("device", "auto"),
                        "num_workers": (
                            args.num_workers
                            if args.num_workers is not None
                            else base.get("num_workers", 0)
                        ),
                        "limit_basins": (
                            args.limit_basins
                            if args.limit_basins is not None
                            else base.get("limit_basins")
                        ),
                        "train_basin_count": (
                            args.train_basin_count
                            if args.train_basin_count is not None
                            else base.get("train_basin_count")
                        ),
                        "val_basin_count": (
                            args.val_basin_count
                            if args.val_basin_count is not None
                            else base.get("val_basin_count")
                        ),
                        "test_basin_count": (
                            args.test_basin_count
                            if args.test_basin_count is not None
                            else base.get("test_basin_count")
                        ),
                    }
                )
                (run_dir / "run_config.json").write_text(
                    json.dumps(vars(train_args), indent=2, default=str)
                )

                print(f"Starting run: {run_name}")
                from training.trainer import train_model

                train_model(train_args)

                if args.evaluate:
                    from evaluation.evaluator import evaluate_checkpoint

                    eval_args = Namespace(
                        checkpoint=str(checkpoint),
                        data_dir=train_args.data_dir,
                        output_dir=str(run_dir),
                        batch_size=args.eval_batch_size,
                        device=train_args.device,
                        num_workers=train_args.num_workers,
                    )
                    evaluate_checkpoint(eval_args)
                    row.update(_read_metrics(metrics_path))

                row["status"] = "completed"
                rows.append(row)
                _write_summary(rows, summary_path)

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return

    _write_summary(rows, summary_path)
    print(f"Sweep summary saved to {summary_path}")
