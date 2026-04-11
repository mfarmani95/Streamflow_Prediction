"""Plotting helpers for exploratory analysis, training curves, and evaluation."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


_SWEEP_RUN_PATTERN = re.compile(
    r"seq(?P<seq_len>\d+)_hidden(?P<hidden_size>\d+)_batch(?P<batch_size>\d+)"
)
_SWEEP_PARAM_ALIASES = {
    "batch": "batch_size",
    "batches": "batch_size",
    "batch_sizes": "batch_size",
    "hidden": "hidden_size",
    "hidden_sizes": "hidden_size",
    "learning_rate": "lr",
    "learning_rates": "lr",
    "lrs": "lr",
    "sequence_length": "seq_len",
    "sequence_lengths": "seq_len",
    "seq": "seq_len",
    "seq_lens": "seq_len",
}
_SWEEP_PARAM_LABELS = {
    "seq_len": "sequence length",
    "hidden_size": "hidden size",
    "batch_size": "batch size",
    "lr": "learning rate",
    "loss": "loss",
    "dropout": "dropout",
    "num_layers": "number of layers",
    "weight_decay": "weight decay",
    "window_stride": "window stride",
    "forecast_horizon": "forecast horizon",
    "model": "model",
}
_SWEEP_PARAM_TOKENS = {
    "seq_len": "seq",
    "hidden_size": "hidden",
    "batch_size": "batch",
    "lr": "lr",
    "loss": "loss",
    "dropout": "drop",
    "num_layers": "layers",
    "weight_decay": "wd",
    "window_stride": "stride",
    "forecast_horizon": "horizon",
    "model": "model",
}
_SWEEP_COMPARISON_PARAM_ORDER = [
    "model",
    "seq_len",
    "forecast_horizon",
    "window_stride",
    "hidden_size",
    "num_layers",
    "nhead",
    "dim_feedforward",
    "dropout",
    "batch_size",
    "lr",
    "loss",
    "weight_decay",
    "grad_clip",
    "patience",
    "min_delta",
    "seed",
    "train_basin_count",
    "val_basin_count",
    "test_basin_count",
]
_SWEEP_NON_COMPARISON_PARAMS = {
    "checkpoint",
    "config",
    "data_dir",
    "device",
    "dynamic_inputs",
    "num_workers",
    "output_dir",
    "static_attributes",
    "target_variable",
}


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_sweep_param_name(name: str) -> str:
    normalized = name.strip().replace("-", "_").lower()
    return _SWEEP_PARAM_ALIASES.get(normalized, normalized)


def _read_sweep_run_params(run_dir: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
        for param_name, value in config.items():
            params[_canonical_sweep_param_name(param_name)] = value

    match = _SWEEP_RUN_PATTERN.search(run_dir.name)
    if match is not None:
        for param_name, value in match.groupdict().items():
            params.setdefault(param_name, _coerce_int(value))

    return params


def _read_sweep_histories(sweep_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for history_path in sorted(sweep_root.glob("*/training_history.csv")):
        run_dir = history_path.parent
        params = _read_sweep_run_params(run_dir)
        if params is None:
            continue

        history = pd.read_csv(history_path)
        if history.empty:
            continue
        if "epoch" not in history.columns:
            history.insert(0, "epoch", np.arange(1, len(history) + 1))

        for column in ("epoch", "train_loss", "val_loss", "val_nse", "val_kge"):
            if column in history.columns:
                history[column] = pd.to_numeric(history[column], errors="coerce")

        runs.append(
            {
                "params": params,
                "run_name": run_dir.name,
                "run_dir": run_dir,
                "history": history,
            }
        )
    return runs


def _sweep_param_label(param_name: str) -> str:
    return _SWEEP_PARAM_LABELS.get(param_name, param_name.replace("_", " "))


def _sweep_value_key(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_sweep_value(item) for item in value) + "]"
    return str(value)


def _format_sweep_filename_value(value: Any) -> str:
    text = _format_sweep_value(value)
    text = text.replace("-", "m").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "value"


def _sweep_param_token(param_name: str, value: Any) -> str:
    token = _SWEEP_PARAM_TOKENS.get(param_name, param_name)
    if isinstance(value, int) and param_name in {"seq_len", "hidden_size", "batch_size", "window_stride"}:
        return f"{token}{value:03d}"
    return f"{token}{_format_sweep_filename_value(value)}"


def _sweep_fixed_label(fixed_values: Mapping[str, Any]) -> str:
    if not fixed_values:
        return "all other swept settings fixed"
    return ", ".join(
        f"{_sweep_param_label(param_name)}={_format_sweep_value(value)}"
        for param_name, value in fixed_values.items()
    )


def _sweep_group_filename(varied_param: str, fixed_values: Mapping[str, Any]) -> str:
    fixed_part = (
        "_".join(
            _sweep_param_token(param_name, value)
            for param_name, value in fixed_values.items()
        )
        or "all"
    )
    return f"{fixed_part}_{varied_param}_effect.png"


def _sweep_sort_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, _sweep_value_key(value))


def _ordered_sweep_params(param_names: Sequence[str]) -> List[str]:
    order = {name: index for index, name in enumerate(_SWEEP_COMPARISON_PARAM_ORDER)}
    return sorted(param_names, key=lambda name: (order.get(name, len(order)), name))


def _variable_sweep_params(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    all_params = {
        param_name
        for run in runs
        for param_name in run["params"]
        if param_name not in _SWEEP_NON_COMPARISON_PARAMS
    }
    variable_params = []
    for param_name in all_params:
        values = {
            _sweep_value_key(run["params"][param_name])
            for run in runs
            if param_name in run["params"]
        }
        if len(values) > 1:
            variable_params.append(param_name)
    if {"seq_len", "window_stride"}.issubset(variable_params):
        stride_matches_seq = all(
            run["params"].get("window_stride") == run["params"].get("seq_len")
            for run in runs
            if "window_stride" in run["params"] and "seq_len" in run["params"]
        )
        if stride_matches_seq:
            variable_params.remove("window_stride")
    return _ordered_sweep_params(variable_params)


def _plot_sweep_history_group(
    runs: Sequence[Mapping[str, Any]],
    varied_param: str,
    fixed_values: Mapping[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_runs = sorted(
        runs,
        key=lambda run: (
            _sweep_sort_value(run["params"][varied_param]),
            run["run_name"],
        ),
    )
    cmap = plt.get_cmap("tab10" if len(sorted_runs) <= 10 else "tab20")
    colors = [cmap(index % cmap.N) for index in range(len(sorted_runs))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    loss_ax, metric_ax = axes
    color_handles = []
    varied_label = _sweep_param_label(varied_param)

    for color, run in zip(colors, sorted_runs):
        history = run["history"]
        epochs = history["epoch"].to_numpy(dtype=float)
        scenario_label = f"{varied_label}={_format_sweep_value(run['params'][varied_param])}"
        color_handles.append(Line2D([0], [0], color=color, linewidth=2.5, label=scenario_label))

        if "train_loss" in history.columns:
            loss_ax.plot(
                epochs,
                history["train_loss"].to_numpy(dtype=float),
                color=color,
                linestyle="-",
                linewidth=1.8,
            )
        if "val_loss" in history.columns:
            loss_ax.plot(
                epochs,
                history["val_loss"].to_numpy(dtype=float),
                color=color,
                linestyle="--",
                linewidth=1.8,
            )
        if "val_nse" in history.columns:
            metric_ax.plot(
                epochs,
                history["val_nse"].to_numpy(dtype=float),
                color=color,
                linestyle="-",
                linewidth=1.8,
            )
        if "val_kge" in history.columns:
            metric_ax.plot(
                epochs,
                history["val_kge"].to_numpy(dtype=float),
                color=color,
                linestyle="--",
                linewidth=1.8,
            )

    fixed_label = _sweep_fixed_label(fixed_values)
    loss_ax.set_title("Loss (solid=train, dashed=val)")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.25)

    metric_ax.set_title("Validation metrics (solid=NSE, dashed=KGE)")
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_ylabel("Validation metric")
    metric_ax.grid(alpha=0.25)

    fig.suptitle(f"Effect of {varied_label} with {fixed_label}")
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="solid: train loss / val NSE"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="dashed: val loss / val KGE"),
    ]
    legend_handles = [*color_handles, *style_handles]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(7, max(1, len(legend_handles))),
        frameon=False,
        columnspacing=1.2,
        handlelength=2.8,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_sweep_comparison_plots(
    sweep_root: str = "outputs/sweeps",
    output_dir: str | None = None,
    effects: Sequence[str] | None = None,
) -> None:
    """Create grouped comparison plots from completed sweep training histories."""
    root = Path(sweep_root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist. Run the sweep first.")

    runs = _read_sweep_histories(root)
    if not runs:
        raise FileNotFoundError(
            f"No training_history.csv files were found under {root}. Run the sweep first."
        )

    variable_params = _variable_sweep_params(runs)
    selected_effects = [
        _canonical_sweep_param_name(param_name)
        for param_name in (effects or variable_params)
    ]
    invalid_effects = sorted(set(selected_effects) - set(variable_params))
    if invalid_effects:
        raise ValueError(
            f"These effects were not varied in the available runs: {invalid_effects}. "
            f"Available effects: {variable_params}"
        )

    plot_root = Path(output_dir) if output_dir else root / "comparison_plots"
    plot_count = 0
    for varied_param in selected_effects:
        fixed_params = [param_name for param_name in variable_params if param_name != varied_param]
        grouped_runs: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
        for run in runs:
            if varied_param not in run["params"]:
                continue
            fixed_key = tuple(
                _sweep_value_key(run["params"].get(param_name))
                for param_name in fixed_params
            )
            grouped_runs.setdefault(fixed_key, []).append(run)

        for fixed_key, group_runs in sorted(grouped_runs.items()):
            varied_values = {
                _sweep_value_key(run["params"][varied_param])
                for run in group_runs
            }
            if len(varied_values) < 2:
                continue
            first_run_params = group_runs[0]["params"]
            fixed_values = {
                param_name: first_run_params.get(param_name)
                for param_name in fixed_params
            }
            output_path = (
                plot_root
                / f"{varied_param}_effect"
                / _sweep_group_filename(varied_param, fixed_values)
            )
            _plot_sweep_history_group(group_runs, varied_param, fixed_values, output_path)
            plot_count += 1

    if plot_count == 0:
        print(f"No comparable sweep groups found in {root}.")
        return

    print(f"Sweep comparison plots written to {plot_root} ({plot_count} figures).")


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


def _values_for_distribution(df: pd.DataFrame, column: str, split_name: str, log_transform: bool) -> np.ndarray:
    values = df.loc[df["split"] == split_name, column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if log_transform:
        values = np.log1p(values[values >= 0])
    return values


def _plot_distribution_by_split(
    df: pd.DataFrame,
    column: str,
    output_path: Path,
    log_transform: bool = False,
    split_order: Sequence[str] = ("train", "val", "test"),
) -> None:
    values_by_split = {
        split_name: _values_for_distribution(df, column, split_name, log_transform)
        for split_name in split_order
    }
    non_empty_values = [values for values in values_by_split.values() if values.size > 0]
    if not non_empty_values:
        return
    all_values = np.concatenate(non_empty_values)

    x_label = f"log1p({column})" if log_transform else column
    bins = np.histogram_bin_edges(all_values, bins=60)
    colors = {"train": "#0072B2", "val": "#009E73", "test": "#D55E00"}

    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.5])
    hist_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    overlay_ax = fig.add_subplot(grid[1, :2])
    ecdf_ax = fig.add_subplot(grid[1, 2])

    max_density = 0.0
    for values in values_by_split.values():
        if values.size > 0:
            density, _ = np.histogram(values, bins=bins, density=True)
            max_density = max(max_density, float(np.nanmax(density)))

    for ax, split_name in zip(hist_axes, split_order):
        values = values_by_split[split_name]
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=colors[split_name],
            alpha=0.25,
            edgecolor=colors[split_name],
            linewidth=0.8,
        )
        ax.set_title(f"{split_name} (n={values.size:,})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.set_xlim(float(np.nanmin(all_values)), float(np.nanmax(all_values)))
        if max_density > 0:
            ax.set_ylim(0, max_density * 1.08)

        if values.size > 0:
            median = float(np.nanmedian(values))
            ax.axvline(median, color="black", linestyle="--", linewidth=1, label="median")
            ax.legend(loc="upper right", fontsize=8)

    for split_name, values in values_by_split.items():
        if values.size == 0:
            continue
        overlay_ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=colors[split_name],
            label=split_name,
        )
        sorted_values = np.sort(values)
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ecdf_ax.plot(
            sorted_values,
            y,
            label=split_name,
            color=colors[split_name],
            linewidth=1.8,
        )
    overlay_ax.set_title("Density overlay")
    overlay_ax.set_xlabel(x_label)
    overlay_ax.set_ylabel("Density")
    overlay_ax.set_xlim(float(np.nanmin(all_values)), float(np.nanmax(all_values)))
    overlay_ax.legend()

    ecdf_ax.set_title("ECDF")
    ecdf_ax.set_xlabel(x_label)
    ecdf_ax.set_ylabel("ECDF")
    ecdf_ax.set_xlim(float(np.nanmin(all_values)), float(np.nanmax(all_values)))
    ecdf_ax.legend()
    fig.suptitle(f"{column} distribution by split")
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
    colors = {"train": "#0072B2", "val": "#009E73", "test": "#D55E00"}
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
            ax.hist(
                target_values,
                bins=50,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors.get(split_name),
                label=split_name,
            )
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

    _plot_distribution_by_split(ts_df, target_variable, distribution_dir / "qobs_distribution_by_split.png")
    _plot_distribution_by_split(ts_df, target_variable, distribution_dir / "log_qobs_distribution_by_split.png", log_transform=True)
    for variable in dynamic_inputs:
        _plot_distribution_by_split(ts_df, variable, distribution_dir / f"{variable}_distribution_by_split.png")
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
