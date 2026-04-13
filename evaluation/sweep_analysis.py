"""Sweep result analysis utilities."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_LOWER_IS_BETTER_TOKENS = ("loss", "mse", "mae", "rmse")
_PARAMETER_COLUMNS = [
    "seq_len",
    "hidden_size",
    "batch_size",
    "lr",
    "learning_rate",
    "num_layers",
    "nhead",
    "dim_feedforward",
    "dropout",
    "weight_decay",
]
_TOP_TABLES = {
    "top_by_validation_nse.csv": ("best_val_nse", False),
    "top_by_validation_kge.csv": ("best_val_kge", False),
    "top_by_validation_loss.csv": ("best_val_loss", True),
    "top_by_test_nse.csv": ("test_nse", False),
    "top_by_test_kge.csv": ("test_kge", False),
}
_PREFERRED_COLUMNS = [
    "run_name",
    "model",
    "loss",
    "seq_len",
    "window_stride",
    "hidden_size",
    "num_layers",
    "nhead",
    "dim_feedforward",
    "dropout",
    "batch_size",
    "lr",
    "weight_decay",
    "best_val_loss",
    "epoch_best_val_loss",
    "best_val_nse",
    "epoch_best_val_nse",
    "best_val_kge",
    "epoch_best_val_kge",
    "test_nse",
    "test_kge",
    "test_rmse",
    "test_mse",
    "test_mae",
    "checkpoint",
    "run_dir",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _best_value(frame: pd.DataFrame, column: str, maximize: bool) -> tuple[float, int | None]:
    if column not in frame.columns:
        return float("nan"), None

    values = _numeric_series(frame, column)
    if values.dropna().empty:
        return float("nan"), None

    index = values.idxmax() if maximize else values.idxmin()
    epoch = None
    if "epoch" in frame.columns:
        epoch_value = frame.loc[index, "epoch"]
        if pd.notna(epoch_value):
            epoch = int(epoch_value)
    return float(values.loc[index]), epoch


def _final_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return float("nan")
    values = _numeric_series(frame, column).dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _read_history_summary(history_path: Path) -> Dict[str, Any]:
    if not history_path.exists():
        return {}

    history = pd.read_csv(history_path)
    if history.empty:
        return {}
    if "epoch" not in history.columns:
        history.insert(0, "epoch", np.arange(1, len(history) + 1))

    best_val_loss, epoch_best_val_loss = _best_value(history, "val_loss", maximize=False)
    best_val_nse, epoch_best_val_nse = _best_value(history, "val_nse", maximize=True)
    best_val_kge, epoch_best_val_kge = _best_value(history, "val_kge", maximize=True)

    return {
        "epochs_completed": int(len(history)),
        "best_val_loss": best_val_loss,
        "epoch_best_val_loss": epoch_best_val_loss,
        "best_val_nse": best_val_nse,
        "epoch_best_val_nse": epoch_best_val_nse,
        "best_val_kge": best_val_kge,
        "epoch_best_val_kge": epoch_best_val_kge,
        "final_train_loss": _final_value(history, "train_loss"),
        "final_val_loss": _final_value(history, "val_loss"),
        "final_val_nse": _final_value(history, "val_nse"),
        "final_val_kge": _final_value(history, "val_kge"),
    }


def _read_test_metrics(metrics_path: Path) -> Dict[str, Any]:
    report = _read_json(metrics_path)
    overall = report.get("overall", {})
    metrics = {
        f"test_{name}": value
        for name, value in overall.items()
        if name in {"mse", "mae", "rmse", "nse", "kge"}
    }
    if "best_basin_by_nse" in report:
        metrics["best_basin_by_nse"] = report["best_basin_by_nse"]
    if "worst_basin_by_nse" in report:
        metrics["worst_basin_by_nse"] = report["worst_basin_by_nse"]
    return metrics


def _read_run(run_dir: Path) -> Dict[str, Any]:
    config = _read_json(run_dir / "run_config.json")
    row = {
        **config,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best_model.pt"),
    }
    row.update(_read_history_summary(run_dir / "training_history.csv"))
    row.update(_read_test_metrics(run_dir / "test_metrics.json"))
    return row


def _read_sweep_runs(sweep_root: Path) -> pd.DataFrame:
    run_dirs = sorted(
        path
        for path in sweep_root.iterdir()
        if path.is_dir() and (path / "training_history.csv").exists()
    )
    rows = [_read_run(run_dir) for run_dir in run_dirs]
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column in {"run_name", "run_dir", "checkpoint", "model", "loss"}:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() == frame[column].notna().sum():
            frame[column] = converted
    return frame


def _lower_is_better(metric: str) -> bool:
    lowered = metric.lower()
    return any(token in lowered for token in _LOWER_IS_BETTER_TOKENS)


def _valid_metric_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"{metric!r} is not available. Available columns: {list(frame.columns)}")
    valid = frame.copy()
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    valid = valid[valid[metric].notna()]
    if valid.empty:
        raise ValueError(f"{metric!r} has no numeric values in the sweep results.")
    return valid


def _ordered_columns(columns: Iterable[str]) -> list[str]:
    existing = list(columns)
    ordered = [column for column in _PREFERRED_COLUMNS if column in existing]
    ordered.extend(column for column in existing if column not in ordered)
    return ordered


def _top_table(frame: pd.DataFrame, metric: str, ascending: bool, top_n: int) -> pd.DataFrame:
    valid = _valid_metric_frame(frame, metric)
    return valid.sort_values(metric, ascending=ascending).head(top_n)


def _write_top_tables(frame: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    for filename, (metric, ascending) in _TOP_TABLES.items():
        if metric not in frame.columns:
            continue
        table = _top_table(frame, metric, ascending=ascending, top_n=top_n)
        table[_ordered_columns(table.columns)].to_csv(output_dir / filename, index=False)


def _write_best_summary(best: pd.Series, selection_metric: str, output_path: Path) -> None:
    keys = [
        "run_name",
        "checkpoint",
        "model",
        "loss",
        "seq_len",
        "window_stride",
        "hidden_size",
        "num_layers",
        "nhead",
        "dim_feedforward",
        "dropout",
        "batch_size",
        "lr",
        "best_val_loss",
        "best_val_nse",
        "best_val_kge",
        "test_nse",
        "test_kge",
        "test_rmse",
        "test_mse",
        "test_mae",
    ]
    lines = [f"Best model selected by {selection_metric}"]
    for key in keys:
        if key in best and pd.notna(best[key]):
            lines.append(f"{key}: {best[key]}")
    output_path.write_text("\n".join(lines) + "\n")


def _plot_top_runs(table: pd.DataFrame, metric: str, output_path: Path) -> None:
    if table.empty:
        return

    plot_df = table.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["run_name"], pd.to_numeric(plot_df[metric], errors="coerce"), color="#0072B2")
    ax.set_xlabel(metric)
    ax.set_ylabel("run")
    ax.set_title(f"Top runs by {metric}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parameter_effect_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in _PARAMETER_COLUMNS
        if column in frame.columns and frame[column].dropna().nunique() > 1
    ]


def _plot_parameter_effect(summary: pd.DataFrame, param_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    score_ax, error_ax = axes

    x = summary[param_name]
    for metric in ("best_val_nse_mean", "best_val_kge_mean", "test_nse_mean", "test_kge_mean"):
        if metric in summary.columns:
            score_ax.plot(x, summary[metric], marker="o", linewidth=1.8, label=metric)
    score_ax.set_xlabel(param_name)
    score_ax.set_ylabel("Score")
    score_ax.grid(alpha=0.25)
    score_ax.legend(fontsize=8)

    for metric in ("best_val_loss_mean", "test_rmse_mean", "test_mse_mean"):
        if metric in summary.columns:
            error_ax.plot(x, summary[metric], marker="o", linewidth=1.8, label=metric)
    error_ax.set_xlabel(param_name)
    error_ax.set_ylabel("Loss / error")
    error_ax.grid(alpha=0.25)
    error_ax.legend(fontsize=8)

    fig.suptitle(f"Mean sweep performance by {param_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_parameter_effects(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        column
        for column in [
            "best_val_loss",
            "best_val_nse",
            "best_val_kge",
            "test_nse",
            "test_kge",
            "test_rmse",
            "test_mse",
            "test_mae",
        ]
        if column in frame.columns
    ]
    if not metrics:
        return

    for param_name in _parameter_effect_columns(frame):
        summary = frame.groupby(param_name, dropna=True)[metrics].agg(["mean", "std", "max", "min"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index().sort_values(param_name)
        summary_path = output_dir / f"parameter_effect_{param_name}.csv"
        summary.to_csv(summary_path, index=False)
        _plot_parameter_effect(summary, param_name, output_dir / f"parameter_effect_{param_name}.png")


def _write_metric_correlations(frame: pd.DataFrame, output_dir: Path) -> None:
    pairs: Sequence[tuple[str, str]] = [
        ("best_val_nse", "test_nse"),
        ("best_val_kge", "test_kge"),
        ("best_val_loss", "test_nse"),
    ]
    rows = []
    for left, right in pairs:
        if left not in frame.columns or right not in frame.columns:
            continue
        subset = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < 2:
            continue
        rows.append({"left": left, "right": right, "correlation": subset[left].corr(subset[right])})
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "validation_test_correlations.csv", index=False)


def analyze_sweep(args: Namespace) -> None:
    """Analyze a completed sweep and write rankings plus parameter-effect summaries."""
    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        raise FileNotFoundError(f"{sweep_root} does not exist. Run the sweep first.")

    output_dir = Path(args.output_dir) if args.output_dir else sweep_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _read_sweep_runs(sweep_root)
    if frame.empty:
        raise FileNotFoundError(
            f"No completed runs with training_history.csv were found under {sweep_root}."
        )

    selection_metric = args.selection_metric
    if selection_metric == "auto":
        selection_metric = "best_val_nse" if "best_val_nse" in frame.columns else "best_val_loss"
    ascending = args.ascending if args.ascending is not None else _lower_is_better(selection_metric)
    best_table = _top_table(frame, selection_metric, ascending=ascending, top_n=args.top_n)
    best = best_table.iloc[0]

    all_runs_path = output_dir / "all_runs_analysis.csv"
    frame[_ordered_columns(frame.columns)].to_csv(all_runs_path, index=False)
    _write_best_summary(best, selection_metric, output_dir / "best_model_summary.txt")
    _write_top_tables(frame, output_dir, top_n=args.top_n)
    _write_parameter_effects(frame, output_dir)
    _write_metric_correlations(frame, output_dir)
    _plot_top_runs(best_table, selection_metric, output_dir / f"top{args.top_n}_{selection_metric}.png")

    print(f"Sweep analysis written to {output_dir}")
    print(f"Best model selected by {selection_metric}: {best['run_name']}")
    for metric in ("best_val_nse", "best_val_kge", "best_val_loss", "test_nse", "test_kge", "test_rmse"):
        if metric in best and pd.notna(best[metric]):
            print(f"  {metric}: {best[metric]}")
    print(f"Checkpoint: {best.get('checkpoint')}")
