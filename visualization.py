"""Plotting helpers for exploratory analysis, training curves, and evaluation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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


def _normalize_basin_id(value: object) -> str:
    return str(value).split(".")[0].zfill(8)


def _split_lookup(splits: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    lookup = {}
    for split_name, basin_ids in splits.items():
        for basin_id in basin_ids:
            lookup[_normalize_basin_id(basin_id)] = split_name
    return lookup


def _load_split_timeseries(client, splits: Mapping[str, Sequence[str]], dynamic_inputs: Sequence[str], target_variable: str) -> pd.DataFrame:
    frames = []
    for split_name, basin_ids in splits.items():
        for basin_id in basin_ids:
            basin_id = _normalize_basin_id(basin_id)
            ds = client.load_basin(basin_id)
            df = ds[[*dynamic_inputs, target_variable]].to_dataframe().reset_index()
            df["date"] = pd.to_datetime(df["time"])
            df["basin_id"] = basin_id
            df["split"] = split_name
            frames.append(df.drop(columns=["time"]))
    return pd.concat(frames, ignore_index=True)


def _plot_hist_by_split(df: pd.DataFrame, column: str, output_path: Path, log_transform: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for split_name, group in df.groupby("split"):
        values = group[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if log_transform:
            values = np.log1p(values[values >= 0])
        ax.hist(values, bins=60, alpha=0.45, density=True, label=split_name)
    ax.set_xlabel(f"log1p({column})" if log_transform else column)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_boxplot(df: pd.DataFrame, columns: Sequence[str], split_column: str, output_path: Path, title: str) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3, 3 * n_rows)))
    axes = np.asarray(axes).reshape(-1)
    for ax, column in zip(axes, columns):
        df.boxplot(column=column, by=split_column, ax=ax, grid=False)
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_ylabel(column)
    for ax in axes[len(columns) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, columns: Sequence[str], output_path: Path, title: str) -> None:
    corr = df[list(columns)].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_sequence_target_distributions(
    ts_df: pd.DataFrame,
    target_variable: str,
    seq_lens: Sequence[int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(seq_lens), 1, figsize=(8, max(3, 2.4 * len(seq_lens))), sharex=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, seq_len in zip(axes, seq_lens):
        for split_name, split_df in ts_df.groupby("split"):
            targets = []
            for _, basin_df in split_df.groupby("basin_id"):
                values = basin_df.sort_values("date")[target_variable].to_numpy(dtype=float)
                target_indices = np.arange(seq_len, len(values), seq_len)
                targets.append(values[target_indices])
            target_values = np.concatenate(targets) if targets else np.array([])
            target_values = target_values[np.isfinite(target_values)]
            ax.hist(target_values, bins=50, alpha=0.45, density=True, label=split_name)
        ax.set_title(f"Non-overlapping sequence targets, seq_len={seq_len}")
        ax.set_xlabel(target_variable)
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_split_data_analysis_plots(
    config_path: str = "configs/default.yaml",
    output_dir: str = "outputs/data_analysis",
    data_dir: str | None = None,
) -> None:
    """Create split-aware data analysis plots from the YAML configuration."""
    from minicamels import MiniCamels

    from dataset.minicamels_dataset import make_basin_splits
    from util.config import load_yaml_config, train_defaults_from_config

    config = load_yaml_config(config_path)
    defaults = train_defaults_from_config(config)
    dynamic_inputs = defaults.get("dynamic_inputs", ["prcp", "tmax", "tmin", "srad", "vp"])
    target_variable = defaults.get("target_variable", "qobs")
    static_attributes = defaults.get("static_attributes")
    seed = defaults.get("seed", 42)

    client = MiniCamels(local_data_dir=data_dir or defaults.get("data_dir"))
    basins = client.basins().copy()
    basins["basin_id"] = basins["basin_id"].map(_normalize_basin_id)
    basin_ids = basins["basin_id"].tolist()

    splits = make_basin_splits(
        basin_ids,
        train_count=defaults.get("train_basin_count"),
        val_count=defaults.get("val_basin_count"),
        test_count=defaults.get("test_basin_count"),
        seed=seed,
    )
    split_names = ["train", "val", "test"]
    split_colors = {"train": "tab:blue", "val": "tab:green", "test": "tab:red"}

    root = Path(output_dir)
    distribution_dir = root / "distributions"
    attribute_dir = root / "attributes"
    timeseries_dir = root / "timeseries"
    diagnostic_dir = root / "diagnostics"
    for path in (distribution_dir, attribute_dir, timeseries_dir, diagnostic_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_df = pd.DataFrame(
        [{"basin_id": basin_id, "split": split_name} for split_name, ids in splits.items() for basin_id in ids]
    )
    split_df.to_csv(root / "split_assignments.csv", index=False)

    ts_df = _load_split_timeseries(client, splits, dynamic_inputs, target_variable)
    attrs = client.attributes().copy()
    attrs.index = attrs.index.map(_normalize_basin_id)
    attrs = attrs.reset_index().rename(columns={"index": "basin_id"})
    attrs["basin_id"] = attrs["basin_id"].map(_normalize_basin_id)
    attrs["split"] = attrs["basin_id"].map(_split_lookup(splits))

    _plot_hist_by_split(ts_df, target_variable, distribution_dir / "qobs_distribution_by_split.png")
    _plot_hist_by_split(ts_df, target_variable, distribution_dir / "log_qobs_distribution_by_split.png", log_transform=True)
    for variable in dynamic_inputs:
        _plot_hist_by_split(ts_df, variable, distribution_dir / f"{variable}_distribution_by_split.png")
    _plot_boxplot(ts_df, [*dynamic_inputs, target_variable], "split", distribution_dir / "dynamic_inputs_qobs_boxplots_by_split.png", "Dynamic Inputs and Streamflow by Split")

    selected_static = static_attributes or [
        column for column in attrs.select_dtypes(include=[np.number]).columns if column != "basin_id"
    ]
    plot_static = [column for column in selected_static if column in attrs.columns]
    extra_static = [column for column in ["q_mean", "runoff_ratio", "baseflow_index", "aridity"] if column in attrs.columns and column not in plot_static]
    static_for_plots = [*plot_static, *extra_static]
    _plot_boxplot(attrs, static_for_plots, "split", attribute_dir / "static_attributes_boxplots_by_split.png", "Static Attributes by Split")

    fig, ax = plt.subplots(figsize=(7, 5))
    for split_name in split_names:
        group = attrs[attrs["split"] == split_name]
        ax.scatter(group["lon"], group["lat"], label=split_name, s=55, alpha=0.85, color=split_colors[split_name])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(attribute_dir / "basin_locations_by_split.png", dpi=200)
    plt.close(fig)

    if {"aridity", "q_mean"}.issubset(attrs.columns):
        fig, ax = plt.subplots(figsize=(6, 5))
        for split_name in split_names:
            group = attrs[attrs["split"] == split_name]
            ax.scatter(group["aridity"], group["q_mean"], label=split_name, s=55, alpha=0.85, color=split_colors[split_name])
        ax.set_xlabel("Aridity index")
        ax.set_ylabel("Mean streamflow q_mean")
        ax.legend()
        fig.tight_layout()
        fig.savefig(attribute_dir / "aridity_vs_qmean_by_split.png", dpi=200)
        plt.close(fig)

    hydro_summary = (
        ts_df.groupby(["split", "basin_id"])[target_variable]
        .agg(q_mean="mean", q_std="std", q_p05=lambda x: np.nanpercentile(x, 5), q_p95=lambda x: np.nanpercentile(x, 95))
        .reset_index()
    )
    hydro_summary.to_csv(root / "basin_streamflow_summary.csv", index=False)
    _plot_boxplot(hydro_summary, ["q_mean", "q_std", "q_p05", "q_p95"], "split", distribution_dir / "basin_streamflow_summary_by_split.png", "Basin Streamflow Summary by Split")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, split_name in zip(axes, split_names):
        basin_id = splits[split_name][0]
        subset = ts_df[(ts_df["split"] == split_name) & (ts_df["basin_id"] == basin_id)]
        subset = subset[(subset["date"] >= "1999-10-01") & (subset["date"] <= "2000-09-30")]
        ax.plot(subset["date"], subset[target_variable], linewidth=1.2, color=split_colors[split_name])
        ax.set_title(f"{split_name}: {basin_id}")
        ax.set_ylabel("qobs")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(timeseries_dir / "qobs_example_hydrographs_by_split.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, split_name in zip(axes, split_names):
        basin_id = splits[split_name][0]
        subset = ts_df[(ts_df["split"] == split_name) & (ts_df["basin_id"] == basin_id)]
        subset = subset[(subset["date"] >= "1999-10-01") & (subset["date"] <= "2000-09-30")]
        ax.bar(subset["date"], subset["prcp"], color="tab:blue", alpha=0.25, label="prcp")
        ax2 = ax.twinx()
        ax2.plot(subset["date"], subset[target_variable], color="tab:green", linewidth=1.0, label="qobs")
        ax.set_title(f"{split_name}: {basin_id}")
        ax.set_ylabel("prcp")
        ax2.set_ylabel("qobs")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(timeseries_dir / "precipitation_qobs_examples_by_split.png", dpi=200)
    plt.close(fig)

    missing = (
        ts_df.groupby("split")[[*dynamic_inputs, target_variable]]
        .apply(lambda frame: frame.isna().mean() * 100)
        .reset_index()
    )
    missing_long = missing.melt(id_vars="split", var_name="variable", value_name="missing_percent")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(missing_long["variable"].unique()))
    width = 0.25
    variables = missing_long["variable"].unique()
    for offset, split_name in enumerate(split_names):
        values = missing_long[missing_long["split"] == split_name].set_index("variable").loc[variables, "missing_percent"]
        ax.bar(x + (offset - 1) * width, values, width=width, label=split_name)
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.set_ylabel("Missing values (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(diagnostic_dir / "missing_values_by_split.png", dpi=200)
    plt.close(fig)

    _plot_correlation_heatmap(ts_df, [*dynamic_inputs, target_variable], diagnostic_dir / "dynamic_qobs_correlation_heatmap.png", "Dynamic Variables and qobs Correlation")
    numeric_static = [column for column in static_for_plots if column in attrs.columns]
    _plot_correlation_heatmap(attrs, numeric_static, diagnostic_dir / "static_attributes_correlation_heatmap.png", "Static Attributes Correlation")
    _plot_sequence_target_distributions(
        ts_df,
        target_variable,
        seq_lens=[30, 60, 90, 120, 360],
        output_path=diagnostic_dir / "sequence_target_distributions_non_overlapping.png",
    )

    print(f"Data analysis plots written to {root}")


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
