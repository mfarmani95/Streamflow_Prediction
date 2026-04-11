"""Plotting helpers for exploratory analysis, training curves, and evaluation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: Mapping[str, Sequence[float]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    loss_ax, metric_ax = axes
    for name in ("train_loss", "val_loss"):
        if name in history:
            loss_ax.plot(history[name], label=name)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    metric_plotted = False
    for name, values in history.items():
        if name not in {"train_loss", "val_loss"}:
            metric_ax.plot(values, label=name)
            metric_plotted = True
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_ylabel("Metric")
    if metric_plotted:
        metric_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_predicted_observed(dates, observed, predicted, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, observed, label="Observed", linewidth=1.5)
    ax.plot(dates, predicted, label="Predicted", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_parity(observed, predicted, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    observed_values = np.asarray(observed)
    predicted_values = np.asarray(predicted)
    lower = float(np.nanmin([observed_values.min(), predicted_values.min()]))
    upper = float(np.nanmax([observed_values.max(), predicted_values.max()]))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(observed_values, predicted_values, s=12, alpha=0.5)
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Observed streamflow")
    ax.set_ylabel("Predicted streamflow")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def create_exploratory_plots(
    data_dir: str | None = None,
    output_dir: str = "outputs/exploratory",
    basin_count: int = 3,
) -> None:
    """Create the exploratory figures requested in Problem 1."""
    from minicamels import MiniCamels

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    client = MiniCamels(local_data_dir=data_dir)
    basins = client.basins()
    basin_ids = basins["basin_id"].astype(str).str.zfill(8).head(basin_count).tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    for basin_id in basin_ids:
        ts = client.get_streamflow(basin_id, start="1999-10-01", end="2000-09-30")
        ax.plot(pd.to_datetime(ts.time.values), ts.values, label=basin_id, linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow qobs (mm/day)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "streamflow_multiple_basins.png", dpi=200)
    plt.close(fig)

    basin_id = basin_ids[0]
    one = client.load_basin(basin_id).sel(time=slice("1999-10-01", "2000-09-30"))
    fig, ax1 = plt.subplots(figsize=(10, 4))
    dates = pd.to_datetime(one.time.values)
    ax1.bar(dates, one["prcp"].values, color="tab:blue", alpha=0.35, label="prcp")
    ax1.set_ylabel("Precipitation (mm/day)")
    ax2 = ax1.twinx()
    ax2.plot(dates, one["qobs"].values, color="tab:green", label="qobs")
    ax2.set_ylabel("Streamflow qobs (mm/day)")
    ax1.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path / "precipitation_streamflow_one_basin.png", dpi=200)
    plt.close(fig)

    qobs = []
    for basin_id in basin_ids:
        qobs.append(client.get_streamflow(basin_id).values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.concatenate(qobs), bins=60, color="tab:gray", edgecolor="white")
    ax.set_xlabel("Streamflow qobs (mm/day)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path / "qobs_histogram.png", dpi=200)
    plt.close(fig)

    attrs = client.attributes().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(attrs["lon"], attrs["lat"], c=attrs["aridity"], cmap="viridis", s=45)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(scatter, ax=ax, label="Aridity")
    fig.tight_layout()
    fig.savefig(path / "basin_aridity_scatter.png", dpi=200)
    plt.close(fig)


def create_all_plots(args: Namespace) -> None:
    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} not found. Run `python main.py evaluate --checkpoint "
            f"{args.checkpoint}` first."
        )

    predictions = pd.read_csv(predictions_path)
    plot_parity(
        predictions["observed"],
        predictions["predicted"],
        str(output_dir / "test_parity.png"),
    )

    metrics_path = output_dir / "test_metrics_by_basin.csv"
    if metrics_path.exists():
        basin_metrics = pd.read_csv(metrics_path).sort_values("nse", ascending=False)
        selected = {
            "best": basin_metrics.iloc[0]["basin_id"],
            "worst": basin_metrics.iloc[-1]["basin_id"],
        }
    else:
        selected = {"example": predictions["basin_id"].iloc[0]}

    for label, basin_id in selected.items():
        subset = predictions[predictions["basin_id"].astype(str).str.zfill(8) == str(basin_id).zfill(8)]
        plot_predicted_observed(
            pd.to_datetime(subset["date"]),
            subset["observed"],
            subset["predicted"],
            str(output_dir / f"test_{label}_basin_timeseries.png"),
        )

    history_path = output_dir / "training_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        history = {
            column: history_df[column].tolist()
            for column in history_df.columns
            if column != "epoch"
        }
        plot_training_history(history, str(output_dir / "training_history.png"))

    print(f"Plots written to {output_dir}")
