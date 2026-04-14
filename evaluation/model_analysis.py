"""Paper-style analysis plots for completed model runs."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.metrics import regression_metrics


_SPLIT_COLORS = {
    "LSTM": "#0072B2",
    "Transformer": "#D55E00",
}
_DEFAULT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
_OVERALL_METRICS = ["nse", "kge", "rmse", "mae", "mse"]
_ATTRIBUTE_COLUMNS = [
    "aridity",
    "q_mean",
    "runoff_ratio",
    "baseflow_index",
    "mean_prcp",
    "mean_pet",
    "frac_snow",
    "area_km2",
    "elev_mean",
    "slope_mean",
    "frac_forest",
    "lai_max",
]


def _resolve_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    if run_dir.name in {
        "test_predictions.csv",
        "test_metrics.json",
        "test_metrics_by_basin.csv",
        "training_history.csv",
    }:
        return run_dir.parent
    return run_dir


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _normalize_basin_id(value: object) -> str:
    return str(value).split(".")[0].zfill(8)


def _safe_metrics(observed: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    obs = np.asarray(list(observed), dtype=float).reshape(-1)
    pred = np.asarray(list(predicted), dtype=float).reshape(-1)
    mask = np.isfinite(obs) & np.isfinite(pred)
    if int(mask.sum()) < 2:
        return {metric: float("nan") for metric in _OVERALL_METRICS}
    with np.errstate(all="ignore"):
        return regression_metrics(obs[mask], pred[mask])


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run evaluate first, or use a completed sweep run directory."
        )

    frame = pd.read_csv(path)
    required = {"basin_id", "date", "observed", "predicted"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["basin_id"] = frame["basin_id"].map(_normalize_basin_id)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["observed"] = pd.to_numeric(frame["observed"], errors="coerce")
    frame["predicted"] = pd.to_numeric(frame["predicted"], errors="coerce")
    return frame.sort_values(["basin_id", "date"]).reset_index(drop=True)


def _load_basin_metrics(run_dir: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    path = run_dir / "test_metrics_by_basin.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        rows = []
        for basin_id, group in predictions.groupby("basin_id"):
            rows.append(
                {
                    "basin_id": basin_id,
                    **_safe_metrics(group["observed"], group["predicted"]),
                    "n_samples": int(len(group)),
                }
            )
        frame = pd.DataFrame(rows)

    frame["basin_id"] = frame["basin_id"].map(_normalize_basin_id)
    for column in ["mse", "mae", "rmse", "nse", "kge", "n_samples"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("nse", ascending=False).reset_index(drop=True)


def _load_overall_metrics(run_dir: Path, predictions: pd.DataFrame) -> Dict[str, float]:
    metrics_path = run_dir / "test_metrics.json"
    report = _read_json(metrics_path)
    overall = report.get("overall", {})
    if all(metric in overall for metric in ("mse", "mae", "rmse", "nse", "kge")):
        return {metric: float(overall[metric]) for metric in ("mse", "mae", "rmse", "nse", "kge")}
    return _safe_metrics(predictions["observed"], predictions["predicted"])


def _sample_frame(frame: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(n=max_points, random_state=seed)


def _plot_parity(
    predictions: pd.DataFrame,
    metrics: Mapping[str, float],
    output_path: Path,
    title: str,
    max_points: int,
) -> None:
    plot_df = _sample_frame(predictions.dropna(subset=["observed", "predicted"]), max_points)
    lower = float(np.nanmin([plot_df["observed"].min(), plot_df["predicted"].min()]))
    upper = float(np.nanmax([plot_df["observed"].max(), plot_df["predicted"].max()]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(plot_df["observed"], plot_df["predicted"], s=12, alpha=0.25, color="#0072B2")
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.2)
    ax.set_xlabel("Observed streamflow")
    ax.set_ylabel("Predicted streamflow")
    ax.set_title(title)
    text = "\n".join(
        f"{metric.upper()}={metrics[metric]:.3f}"
        for metric in ("nse", "kge", "rmse")
        if metric in metrics and np.isfinite(metrics[metric])
    )
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_timeseries(
    predictions: pd.DataFrame,
    basin_id: str,
    metrics_by_basin: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    subset = predictions[predictions["basin_id"] == basin_id].sort_values("date")
    if subset.empty:
        return
    basin_metrics = metrics_by_basin[metrics_by_basin["basin_id"] == basin_id]
    metric_text = ""
    if not basin_metrics.empty:
        row = basin_metrics.iloc[0]
        metric_text = f"NSE={row['nse']:.3f}, KGE={row['kge']:.3f}, n={int(row['n_samples'])}"

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(subset["date"], subset["observed"], color="black", linewidth=1.3, label="Observed")
    ax.plot(subset["date"], subset["predicted"], color="#0072B2", linewidth=1.2, label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.set_title(f"{title}: basin {basin_id} {metric_text}".strip())
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_flow_duration(
    predictions: pd.DataFrame,
    basin_id: str,
    output_path: Path,
    title: str,
) -> None:
    subset = predictions[predictions["basin_id"] == basin_id].dropna(subset=["observed", "predicted"])
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for column, label, color in [
        ("observed", "Observed", "black"),
        ("predicted", "Predicted", "#0072B2"),
    ]:
        values = np.sort(subset[column].to_numpy(dtype=float))[::-1]
        exceedance = 100.0 * np.arange(1, len(values) + 1) / (len(values) + 1)
        ax.plot(exceedance, values, color=color, linewidth=1.7, label=label)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Streamflow")
    ax.set_title(f"{title}: basin {basin_id}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_basin_metric_distribution(metrics_by_basin: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, color in zip(axes, ["nse", "kge"], ["#0072B2", "#009E73"]):
        values = pd.to_numeric(metrics_by_basin[metric], errors="coerce").dropna()
        ax.hist(values, bins=min(12, max(4, len(values))), color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Basin count")
        ax.set_title(f"Per-basin {metric.upper()} distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_basin_metric_rank(metrics_by_basin: pd.DataFrame, output_path: Path) -> None:
    ranked = metrics_by_basin.sort_values("nse", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(ranked))))
    ax.barh(ranked["basin_id"], ranked["nse"], color="#0072B2", alpha=0.75, label="NSE")
    if "kge" in ranked.columns:
        ax.scatter(ranked["kge"], ranked["basin_id"], color="#D55E00", s=35, label="KGE")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Basin")
    ax.set_title("Held-out basin performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _metric_cdf_summary(
    frame: pd.DataFrame,
    metric: str,
    group_column: str | None = None,
) -> pd.DataFrame:
    rows = []
    groups = [(None, frame)] if group_column is None else frame.groupby(group_column)
    for group_name, group in groups:
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        row = {
            "metric": metric,
            "n_catchments": int(len(values)),
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
            "fraction_below_0": float((values < 0.0).mean()),
            "fraction_above_0p5": float((values >= 0.5).mean()),
            "fraction_above_0p7": float((values >= 0.7).mean()),
        }
        if group_column is not None:
            row[group_column] = group_name
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_basin_metric_cdf(
    metrics_by_basin: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    values = pd.to_numeric(metrics_by_basin[metric], errors="coerce").dropna().sort_values()
    if values.empty:
        return
    cdf = np.arange(1, len(values) + 1) / len(values)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.step(values, cdf, where="post", color="#0072B2", linewidth=2.0)
    ax.scatter(values, cdf, color="#0072B2", s=25, alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1, label=f"{metric.upper()}=0")
    ax.axvline(float(values.median()), color="#D55E00", linestyle=":", linewidth=1.5, label="median")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Cumulative fraction of catchments")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-3,1)
    ax.set_title(f"Catchment CDF of {metric.upper()}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _monthly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in predictions.groupby(predictions["date"].dt.month):
        metrics = _safe_metrics(group["observed"], group["predicted"])
        rows.append(
            {
                "month": int(month),
                **metrics,
                "observed_mean": float(group["observed"].mean()),
                "predicted_mean": float(group["predicted"].mean()),
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("month")


def _plot_monthly_metrics(monthly: pd.DataFrame, output_path: Path) -> None:
    if monthly.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    score_ax, flow_ax = axes
    score_ax.plot(monthly["month"], monthly["nse"], marker="o", label="NSE", color="#0072B2")
    score_ax.plot(monthly["month"], monthly["kge"], marker="o", label="KGE", color="#009E73")
    score_ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    score_ax.set_xlabel("Month")
    score_ax.set_ylabel("Score")
    score_ax.set_title("Seasonal test skill")
    score_ax.set_xticks(range(1, 13))
    score_ax.grid(alpha=0.25)
    score_ax.legend()

    flow_ax.plot(monthly["month"], monthly["observed_mean"], marker="o", label="Observed", color="black")
    flow_ax.plot(monthly["month"], monthly["predicted_mean"], marker="o", label="Predicted", color="#0072B2")
    flow_ax.set_xlabel("Month")
    flow_ax.set_ylabel("Mean streamflow")
    flow_ax.set_title("Mean seasonal hydrograph")
    flow_ax.set_xticks(range(1, 13))
    flow_ax.grid(alpha=0.25)
    flow_ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _flow_regime_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    obs = predictions["observed"].to_numpy(dtype=float)
    finite_obs = obs[np.isfinite(obs)]
    if finite_obs.size == 0:
        return pd.DataFrame()

    low_threshold = float(np.nanpercentile(finite_obs, 10))
    peak_threshold = float(np.nanpercentile(finite_obs, 95))
    regimes = {
        "all": np.isfinite(obs),
        "low_flow_obs_le_p10": obs <= low_threshold,
        "middle_flow": (obs > low_threshold) & (obs < peak_threshold),
        "peak_flow_obs_ge_p95": obs >= peak_threshold,
    }
    rows = []
    for regime, mask in regimes.items():
        subset = predictions.loc[mask].dropna(subset=["observed", "predicted"])
        metrics = _safe_metrics(subset["observed"], subset["predicted"])
        obs_sum = float(subset["observed"].sum())
        pred_sum = float(subset["predicted"].sum())
        percent_bias = float(100.0 * (pred_sum - obs_sum) / obs_sum) if abs(obs_sum) > 1e-12 else float("nan")
        rows.append(
            {
                "regime": regime,
                "n_samples": int(len(subset)),
                "observed_mean": float(subset["observed"].mean()) if len(subset) else float("nan"),
                "predicted_mean": float(subset["predicted"].mean()) if len(subset) else float("nan"),
                "percent_bias": percent_bias,
                "low_flow_threshold_p10": low_threshold,
                "peak_flow_threshold_p95": peak_threshold,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _plot_flow_regime_bias(regime_metrics: pd.DataFrame, output_path: Path) -> None:
    if regime_metrics.empty:
        return
    plot_df = regime_metrics[regime_metrics["regime"] != "all"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df["regime"], plot_df["percent_bias"], color=["#009E73", "#0072B2", "#D55E00"])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Percent bias (%)")
    ax.set_title("Bias by observed flow regime")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _load_attributes(data_dir: str | None) -> pd.DataFrame | None:
    try:
        from minicamels import MiniCamels
    except ImportError:
        return None

    try:
        client = MiniCamels(local_data_dir=data_dir)
        attrs = client.attributes().copy()
    except Exception:
        return None
    attrs.index = attrs.index.map(_normalize_basin_id)
    return attrs.reset_index().rename(columns={"index": "basin_id"})


def _plot_attributes_vs_nse(metrics_with_attrs: pd.DataFrame, output_path: Path) -> None:
    columns = [
        column
        for column in _ATTRIBUTE_COLUMNS
        if column in metrics_with_attrs.columns and pd.to_numeric(metrics_with_attrs[column], errors="coerce").notna().any()
    ]
    if not columns:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3, 3.2 * n_rows)))
    axes = np.asarray(axes).reshape(-1)
    for ax, column in zip(axes, columns):
        x = pd.to_numeric(metrics_with_attrs[column], errors="coerce")
        y = pd.to_numeric(metrics_with_attrs["nse"], errors="coerce")
        color_values = pd.to_numeric(metrics_with_attrs["kge"], errors="coerce")
        scatter = ax.scatter(x, y, c=color_values, cmap="viridis", s=55, alpha=0.85)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        corr = x.corr(y)
        ax.set_title(f"{column} vs NSE" + (f" (r={corr:.2f})" if np.isfinite(corr) else ""))
        ax.set_xlabel(column)
        ax.set_ylabel("NSE")
    for ax in axes[len(columns) :]:
        ax.axis("off")
    fig.colorbar(scatter, ax=axes[: len(columns)], label="KGE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _selected_attribute_values(row: pd.Series) -> str:
    pieces = []
    for column in _ATTRIBUTE_COLUMNS:
        if column in row and pd.notna(row[column]):
            value = row[column]
            pieces.append(f"{column}={value:.3g}" if isinstance(value, (float, int, np.floating, np.integer)) else f"{column}={value}")
    return ", ".join(pieces)


def _write_discussion_notes(
    output_path: Path,
    run_dir: Path,
    overall: Mapping[str, float],
    metrics_by_basin: pd.DataFrame,
    monthly: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    metrics_with_attrs: pd.DataFrame | None,
) -> None:
    best = metrics_by_basin.sort_values("nse", ascending=False).iloc[0]
    worst = metrics_by_basin.sort_values("nse", ascending=True).iloc[0]
    lines = [
        f"Run directory: {run_dir}",
        "",
        "Overall held-out test metrics:",
    ]
    for metric in ("nse", "kge", "rmse", "mae", "mse"):
        if metric in overall:
            lines.append(f"- {metric.upper()}: {overall[metric]:.4f}")

    lines.extend(
        [
            "",
            "Best and worst test basins by NSE:",
            f"- Best basin {best['basin_id']}: NSE={best['nse']:.4f}, KGE={best['kge']:.4f}, RMSE={best['rmse']:.4f}",
            f"- Worst basin {worst['basin_id']}: NSE={worst['nse']:.4f}, KGE={worst['kge']:.4f}, RMSE={worst['rmse']:.4f}",
        ]
    )
    kge_summary = _metric_cdf_summary(metrics_by_basin, "kge")
    if not kge_summary.empty:
        row = kge_summary.iloc[0]
        lines.extend(
            [
                "",
                "Catchment KGE CDF summary:",
                f"- Median KGE: {row['median']:.4f}",
                f"- Fraction of catchments with KGE < 0: {row['fraction_below_0']:.2%}",
                f"- Fraction of catchments with KGE >= 0.5: {row['fraction_above_0p5']:.2%}",
                f"- Fraction of catchments with KGE >= 0.7: {row['fraction_above_0p7']:.2%}",
            ]
        )

    if metrics_with_attrs is not None and not metrics_with_attrs.empty:
        attr_lookup = metrics_with_attrs.set_index("basin_id")
        lines.extend(["", "Catchment attributes to use in the discussion:"])
        for label, row in [("Best", best), ("Worst", worst)]:
            basin_id = row["basin_id"]
            if basin_id in attr_lookup.index:
                lines.append(f"- {label} basin {basin_id}: {_selected_attribute_values(attr_lookup.loc[basin_id])}")

    if not monthly.empty:
        nse_months = monthly.dropna(subset=["nse"])
        if not nse_months.empty:
            high = nse_months.loc[nse_months["nse"].idxmax()]
            low = nse_months.loc[nse_months["nse"].idxmin()]
            lines.extend(
                [
                    "",
                    "Seasonality:",
                    f"- Highest monthly NSE occurs in month {int(high['month'])}: NSE={high['nse']:.4f}",
                    f"- Lowest monthly NSE occurs in month {int(low['month'])}: NSE={low['nse']:.4f}",
                ]
            )

    if not regime_metrics.empty:
        lines.append("")
        lines.append("Flow-regime bias:")
        for _, row in regime_metrics.iterrows():
            lines.append(
                f"- {row['regime']}: percent_bias={row['percent_bias']:.2f}%, NSE={row['nse']:.4f}, n={int(row['n_samples'])}"
            )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- Use the best/worst basin plots to discuss where timing, amplitude, or recession behavior is captured or missed.",
            "- Use the KGE CDF to describe the distribution of model skill across catchments, not only the average score.",
            "- Use the attribute plots to connect skill differences to climate and catchment properties such as aridity, snow fraction, area, or mean flow.",
            "- Use the seasonal and flow-regime plots to discuss whether the model is better for ordinary flows, low flows, or peaks.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def analyze_model_run(args: Namespace) -> None:
    """Create report-style plots and tables for one evaluated run."""
    run_dir = _resolve_run_dir(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "paper_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = _read_json(run_dir / "run_config.json")
    predictions = _load_predictions(run_dir)
    metrics_by_basin = _load_basin_metrics(run_dir, predictions)
    overall = _load_overall_metrics(run_dir, predictions)

    pd.DataFrame([{**{"run_dir": str(run_dir)}, **overall}]).to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics_by_basin.to_csv(output_dir / "basin_performance_ranked.csv", index=False)

    best_basin = metrics_by_basin.iloc[0]["basin_id"]
    worst_basin = metrics_by_basin.iloc[-1]["basin_id"]

    _plot_parity(
        predictions,
        overall,
        output_dir / "parity_test_all.png",
        "Held-out test predictions",
        max_points=args.max_scatter_points,
    )
    _plot_basin_metric_distribution(metrics_by_basin, output_dir / "basin_metric_distribution.png")
    _plot_basin_metric_rank(metrics_by_basin, output_dir / "basin_metric_rank_by_nse.png")
    _metric_cdf_summary(metrics_by_basin, "kge").to_csv(output_dir / "basin_kge_cdf_summary.csv", index=False)
    _plot_basin_metric_cdf(metrics_by_basin, "kge", output_dir / "basin_kge_cdf.png")
    _plot_timeseries(predictions, best_basin, metrics_by_basin, output_dir / "timeseries_best_basin.png", "Best test basin")
    _plot_timeseries(predictions, worst_basin, metrics_by_basin, output_dir / "timeseries_worst_basin.png", "Worst test basin")
    _plot_flow_duration(predictions, best_basin, output_dir / "flow_duration_best_basin.png", "Flow duration curve, best basin")
    _plot_flow_duration(predictions, worst_basin, output_dir / "flow_duration_worst_basin.png", "Flow duration curve, worst basin")

    monthly = _monthly_metrics(predictions)
    monthly.to_csv(output_dir / "monthly_metrics.csv", index=False)
    _plot_monthly_metrics(monthly, output_dir / "monthly_seasonal_skill.png")

    regime_metrics = _flow_regime_metrics(predictions)
    regime_metrics.to_csv(output_dir / "flow_regime_metrics.csv", index=False)
    _plot_flow_regime_bias(regime_metrics, output_dir / "flow_regime_bias.png")

    attrs = _load_attributes(args.data_dir or run_config.get("data_dir"))
    metrics_with_attrs = None
    if attrs is not None:
        metrics_with_attrs = metrics_by_basin.merge(attrs, on="basin_id", how="left")
        metrics_with_attrs.to_csv(output_dir / "basin_performance_with_attributes.csv", index=False)
        _plot_attributes_vs_nse(metrics_with_attrs, output_dir / "attributes_vs_nse.png")
    else:
        (output_dir / "attribute_analysis_skipped.txt").write_text(
            "MiniCAMELS attributes were not available. Install minicamels and/or pass "
            "--data-dir to analyze-run if you want aridity, q_mean, snow fraction, "
            "catchment area, and other attribute-vs-performance plots.\n"
        )

    _write_discussion_notes(
        output_dir / "discussion_notes.txt",
        run_dir,
        overall,
        metrics_by_basin,
        monthly,
        regime_metrics,
        metrics_with_attrs,
    )

    print(f"Model analysis written to {output_dir}")
    print(
        f"Best basin: {best_basin} | "
        f"NSE={metrics_by_basin.iloc[0]['nse']:.3f}, KGE={metrics_by_basin.iloc[0]['kge']:.3f}"
    )
    print(
        f"Worst basin: {worst_basin} | "
        f"NSE={metrics_by_basin.iloc[-1]['nse']:.3f}, KGE={metrics_by_basin.iloc[-1]['kge']:.3f}"
    )


def _label_for_run(run_dir: Path, labels: Sequence[str] | None, index: int) -> str:
    if labels:
        return labels[index]
    run_config = _read_json(run_dir / "run_config.json")
    model_name = run_config.get("model")
    if model_name:
        return str(model_name).upper() if str(model_name).lower() == "lstm" else str(model_name).title()
    return run_dir.name


def _color_for_label(label: str, index: int) -> str:
    return _SPLIT_COLORS.get(label, _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)])


def _load_comparison_run(run_dir: Path, label: str) -> Dict[str, Any]:
    predictions = _load_predictions(run_dir)
    basin_metrics = _load_basin_metrics(run_dir, predictions)
    overall = _load_overall_metrics(run_dir, predictions)
    return {
        "label": label,
        "run_dir": run_dir,
        "predictions": predictions,
        "basin_metrics": basin_metrics,
        "overall": overall,
    }


def _plot_overall_metric_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [metric for metric in _OVERALL_METRICS if metric in summary.columns]
    if not metrics:
        return
    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, metric in zip(axes, metrics):
        ax.bar(summary["label"], summary[metric], color=[_color_for_label(label, idx) for idx, label in enumerate(summary["label"])])
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)
        if metric in {"nse", "kge"}:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle("Held-out test metric comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_parity_overlay(runs: Sequence[Mapping[str, Any]], output_path: Path, max_points: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lower = float("inf")
    upper = float("-inf")
    for index, run in enumerate(runs):
        plot_df = _sample_frame(run["predictions"].dropna(subset=["observed", "predicted"]), max_points, seed=42 + index)
        lower = min(lower, float(plot_df["observed"].min()), float(plot_df["predicted"].min()))
        upper = max(upper, float(plot_df["observed"].max()), float(plot_df["predicted"].max()))
        ax.scatter(
            plot_df["observed"],
            plot_df["predicted"],
            s=10,
            alpha=0.2,
            color=_color_for_label(run["label"], index),
            label=run["label"],
        )
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.2)
    ax.set_xlabel("Observed streamflow")
    ax.set_ylabel("Predicted streamflow")
    ax.set_title("Parity comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _combined_basin_metrics(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        frame = run["basin_metrics"].copy()
        frame["model_label"] = run["label"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _plot_basin_nse_comparison(combined: pd.DataFrame, output_path: Path) -> None:
    pivot = combined.pivot_table(index="basin_id", columns="model_label", values="nse")
    labels = list(pivot.columns)
    if len(labels) == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pivot[labels[0]], pivot[labels[1]], s=60, alpha=0.8, color="#0072B2")
        lower = float(np.nanmin(pivot.to_numpy()))
        upper = float(np.nanmax(pivot.to_numpy()))
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{labels[0]} NSE")
        ax.set_ylabel(f"{labels[1]} NSE")
        ax.set_title("Per-basin NSE comparison")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [pivot[label].dropna().to_numpy() for label in labels]
        ax.boxplot(data, labels=labels)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("NSE")
        ax.set_title("Per-basin NSE distribution by model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_basin_nse_delta(combined: pd.DataFrame, output_path: Path) -> None:
    pivot = combined.pivot_table(index="basin_id", columns="model_label", values="nse")
    labels = list(pivot.columns)
    if len(labels) != 2:
        return
    delta = (pivot[labels[1]] - pivot[labels[0]]).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(delta))))
    colors = np.where(delta >= 0, "#009E73", "#D55E00")
    ax.barh(delta.index, delta, color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(f"NSE delta ({labels[1]} - {labels[0]})")
    ax.set_ylabel("Basin")
    ax.set_title("Per-basin improvement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_metric_cdf_comparison(
    combined: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if metric not in combined.columns or "model_label" not in combined.columns:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5))
    plotted = False
    for index, (label, group) in enumerate(combined.groupby("model_label")):
        values = pd.to_numeric(group[metric], errors="coerce").dropna().sort_values()
        if values.empty:
            continue
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.step(
            values,
            cdf,
            where="post",
            color=_color_for_label(str(label), index),
            linewidth=2.0,
            label=str(label),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1, label=f"{metric.upper()}=0")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Cumulative fraction of catchments")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-3,1)
    ax.set_title(f"Catchment CDF of {metric.upper()}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_comparison_timeseries(
    runs: Sequence[Mapping[str, Any]],
    basin_id: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    observed_plotted = False
    for index, run in enumerate(runs):
        subset = run["predictions"][run["predictions"]["basin_id"] == basin_id].sort_values("date")
        if subset.empty:
            continue
        if not observed_plotted:
            ax.plot(subset["date"], subset["observed"], color="black", linewidth=1.4, label="Observed")
            observed_plotted = True
        ax.plot(
            subset["date"],
            subset["predicted"],
            linewidth=1.2,
            color=_color_for_label(run["label"], index),
            label=run["label"],
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow")
    ax.set_title(f"{title}: basin {basin_id}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_comparison_flow_duration(
    runs: Sequence[Mapping[str, Any]],
    basin_id: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    observed_plotted = False
    for index, run in enumerate(runs):
        subset = run["predictions"][run["predictions"]["basin_id"] == basin_id].dropna(subset=["observed", "predicted"])
        if subset.empty:
            continue
        if not observed_plotted:
            values = np.sort(subset["observed"].to_numpy(dtype=float))[::-1]
            exceedance = 100.0 * np.arange(1, len(values) + 1) / (len(values) + 1)
            ax.plot(exceedance, values, color="black", linewidth=1.8, label="Observed")
            observed_plotted = True
        values = np.sort(subset["predicted"].to_numpy(dtype=float))[::-1]
        exceedance = 100.0 * np.arange(1, len(values) + 1) / (len(values) + 1)
        ax.plot(
            exceedance,
            values,
            color=_color_for_label(run["label"], index),
            linewidth=1.6,
            label=run["label"],
        )
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Streamflow")
    ax.set_title(f"{title}: basin {basin_id}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_monthly_comparison(runs: Sequence[Mapping[str, Any]], output_path: Path) -> pd.DataFrame:
    frames = []
    for run in runs:
        monthly = _monthly_metrics(run["predictions"])
        monthly["model_label"] = run["label"]
        frames.append(monthly)
    combined = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, label in enumerate(combined["model_label"].unique()):
        subset = combined[combined["model_label"] == label]
        color = _color_for_label(label, index)
        axes[0].plot(subset["month"], subset["nse"], marker="o", color=color, label=label)
        axes[1].plot(subset["month"], subset["kge"], marker="o", color=color, label=label)
    for ax, metric in zip(axes, ["NSE", "KGE"]):
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Month")
        ax.set_ylabel(metric)
        ax.set_title(f"Monthly {metric}")
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return combined


def _write_comparison_summary(
    output_path: Path,
    runs: Sequence[Mapping[str, Any]],
    summary: pd.DataFrame,
    combined_basin_metrics: pd.DataFrame,
) -> None:
    lines = ["Model comparison summary", "", "Overall held-out test metrics:"]
    for _, row in summary.iterrows():
        metrics = ", ".join(
            f"{metric.upper()}={row[metric]:.4f}"
            for metric in ("nse", "kge", "rmse", "mae")
            if metric in row and pd.notna(row[metric])
        )
        lines.append(f"- {row['label']}: {metrics}")

    pivot = combined_basin_metrics.pivot_table(index="basin_id", columns="model_label", values="nse")
    labels = list(pivot.columns)
    if len(labels) == 2:
        delta = (pivot[labels[1]] - pivot[labels[0]]).dropna()
        lines.extend(
            [
                "",
                f"Per-basin NSE delta ({labels[1]} - {labels[0]}):",
                f"- Mean delta: {delta.mean():.4f}",
                f"- Basins improved by {labels[1]}: {int((delta > 0).sum())} of {len(delta)}",
                f"- Largest improvement: basin {delta.idxmax()} ({delta.max():.4f})",
                f"- Largest degradation: basin {delta.idxmin()} ({delta.min():.4f})",
            ]
        )

    kge_cdf = _metric_cdf_summary(combined_basin_metrics, "kge", group_column="model_label")
    if not kge_cdf.empty:
        lines.extend(["", "Catchment KGE CDF summary:"])
        for _, row in kge_cdf.iterrows():
            lines.append(
                f"- {row['model_label']}: median KGE={row['median']:.4f}, "
                f"KGE<0={row['fraction_below_0']:.2%}, "
                f"KGE>=0.5={row['fraction_above_0p5']:.2%}, "
                f"KGE>=0.7={row['fraction_above_0p7']:.2%}"
            )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- Compare overall NSE/KGE first, then check whether one model mainly helps certain basins.",
            "- Use the basin delta plot to avoid hiding failures behind the overall mean.",
            "- Use the KGE CDF overlay to compare the whole distribution of catchment skill.",
            "- Use the reference-basin hydrographs and flow-duration curves to compare timing and high/low-flow bias.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def compare_model_runs(args: Namespace) -> None:
    """Compare evaluated LSTM and Transformer run directories."""
    run_dirs = [_resolve_run_dir(path) for path in args.run_dirs]
    if args.labels and len(args.labels) != len(run_dirs):
        raise ValueError("--labels must have the same length as --run-dirs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        _load_comparison_run(run_dir, _label_for_run(run_dir, args.labels, index))
        for index, run_dir in enumerate(run_dirs)
    ]
    summary_rows = []
    for run in runs:
        summary_rows.append({"label": run["label"], "run_dir": str(run["run_dir"]), **run["overall"]})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "model_metrics_summary.csv", index=False)
    _plot_overall_metric_comparison(summary, output_dir / "overall_metric_comparison.png")
    _plot_parity_overlay(runs, output_dir / "parity_overlay.png", max_points=args.max_scatter_points)

    combined_basin_metrics = _combined_basin_metrics(runs)
    combined_basin_metrics.to_csv(output_dir / "basin_metric_comparison_long.csv", index=False)
    _plot_basin_nse_comparison(combined_basin_metrics, output_dir / "basin_nse_comparison.png")
    _plot_basin_nse_delta(combined_basin_metrics, output_dir / "basin_nse_delta.png")
    _metric_cdf_summary(combined_basin_metrics, "kge", group_column="model_label").to_csv(
        output_dir / "basin_kge_cdf_summary_by_model.csv",
        index=False,
    )
    _plot_metric_cdf_comparison(combined_basin_metrics, "kge", output_dir / "basin_kge_cdf_comparison.png")

    monthly = _plot_monthly_comparison(runs, output_dir / "monthly_metric_comparison.png")
    monthly.to_csv(output_dir / "monthly_metric_comparison.csv", index=False)

    reference_metrics = runs[0]["basin_metrics"].sort_values("nse", ascending=False)
    basin_ids = [args.basin_id] if args.basin_id else [reference_metrics.iloc[0]["basin_id"], reference_metrics.iloc[-1]["basin_id"]]
    names = ["selected"] if args.basin_id else ["best_reference", "worst_reference"]
    for name, basin_id in zip(names, basin_ids):
        basin_id = _normalize_basin_id(basin_id)
        _plot_comparison_timeseries(
            runs,
            basin_id,
            output_dir / f"timeseries_comparison_{name}_basin.png",
            f"Prediction comparison, {name.replace('_', ' ')}",
        )
        _plot_comparison_flow_duration(
            runs,
            basin_id,
            output_dir / f"flow_duration_comparison_{name}_basin.png",
            f"Flow duration comparison, {name.replace('_', ' ')}",
        )

    _write_comparison_summary(output_dir / "comparison_summary.txt", runs, summary, combined_basin_metrics)

    print(f"Model comparison written to {output_dir}")
    for _, row in summary.iterrows():
        print(
            f"{row['label']}: NSE={row.get('nse', float('nan')):.3f}, "
            f"KGE={row.get('kge', float('nan')):.3f}, RMSE={row.get('rmse', float('nan')):.3f}"
        )
