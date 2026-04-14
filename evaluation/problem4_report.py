"""Generate the Problem 4 evaluation report."""

from __future__ import annotations

import subprocess
import textwrap
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from evaluation.model_analysis import (
    _load_basin_metrics,
    _load_overall_metrics,
    _load_predictions,
    _read_json,
    _resolve_run_dir,
    analyze_model_run,
    compare_model_runs,
)


_METRIC_COLUMNS = ["nse", "kge", "rmse", "mae", "mse"]
_REPORT_WIDTH = 90


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _format_percent(value: Any, digits: int = 1) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{100.0 * value:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    table = frame.loc[:, list(columns)].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, int, np.floating, np.integer)):
                values.append(_format_float(value, digits=4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _repo_url_from_git() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    url = result.stdout.strip()
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:").removesuffix(".git")
    return url


def _run_analysis(
    label: str,
    run_dir: Path,
    output_dir: Path,
    data_dir: str | None,
    max_scatter_points: int,
) -> Path:
    analysis_dir = output_dir / f"{label.lower()}_analysis"
    analyze_model_run(
        Namespace(
            run_dir=str(run_dir),
            output_dir=str(analysis_dir),
            data_dir=data_dir,
            max_scatter_points=max_scatter_points,
        )
    )
    return analysis_dir


def _run_comparison(
    lstm_run_dir: Path,
    transformer_run_dir: Path,
    output_dir: Path,
    max_scatter_points: int,
) -> Path:
    comparison_dir = output_dir / "model_comparison"
    compare_model_runs(
        Namespace(
            run_dirs=[str(lstm_run_dir), str(transformer_run_dir)],
            labels=["LSTM", "Transformer"],
            output_dir=str(comparison_dir),
            basin_id=None,
            max_scatter_points=max_scatter_points,
        )
    )
    return comparison_dir


def _collect_run(label: str, run_dir: Path, analysis_dir: Path) -> Dict[str, Any]:
    predictions = _load_predictions(run_dir)
    basin_metrics = _load_basin_metrics(run_dir, predictions)
    overall = _load_overall_metrics(run_dir, predictions)
    run_config = _read_json(run_dir / "run_config.json")

    best = basin_metrics.sort_values("nse", ascending=False).iloc[0]
    worst = basin_metrics.sort_values("nse", ascending=True).iloc[0]
    monthly = pd.read_csv(analysis_dir / "monthly_metrics.csv")
    regimes = pd.read_csv(analysis_dir / "flow_regime_metrics.csv")
    cdf = pd.read_csv(analysis_dir / "basin_kge_cdf_summary.csv")

    return {
        "label": label,
        "run_dir": run_dir,
        "analysis_dir": analysis_dir,
        "predictions": predictions,
        "basin_metrics": basin_metrics,
        "overall": overall,
        "run_config": run_config,
        "best": best,
        "worst": worst,
        "monthly": monthly,
        "regimes": regimes,
        "cdf": cdf.iloc[0] if not cdf.empty else pd.Series(dtype=float),
    }


def _metric_summary_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append({"Model": run["label"], **{metric.upper(): run["overall"][metric] for metric in _METRIC_COLUMNS}})
    return pd.DataFrame(rows)


def _run_config_summary(run: Mapping[str, Any]) -> str:
    config = run["run_config"]
    keys = [
        "model",
        "loss",
        "seq_len",
        "window_stride",
        "hidden_size",
        "num_layers",
        "dropout",
        "batch_size",
        "lr",
        "train_basin_count",
        "val_basin_count",
        "test_basin_count",
        "split_strategy",
    ]
    if config.get("model") == "transformer":
        keys.insert(keys.index("dropout"), "dim_feedforward")
        keys.insert(keys.index("dim_feedforward"), "nhead")
    pieces = []
    for key in keys:
        if key in config and config[key] is not None:
            pieces.append(f"{key}={config[key]}")
    return ", ".join(pieces)


def _training_command(run: Mapping[str, Any], output_subdir: str) -> str:
    config = run["run_config"]
    command = [
        "python main.py train",
        f"--config {config.get('config', 'configs/default.yaml')}",
        f"--model {config.get('model')}",
        f"--seq-len {config.get('seq_len')}",
        f"--window-stride {config.get('window_stride')}",
        f"--hidden-size {config.get('hidden_size')}",
        f"--num-layers {config.get('num_layers')}",
        f"--dropout {config.get('dropout')}",
        f"--batch-size {config.get('batch_size')}",
        f"--lr {config.get('lr')}",
        f"--loss {config.get('loss')}",
        f"--epochs {config.get('epochs')}",
        f"--seed {config.get('seed')}",
        f"--split-strategy {config.get('split_strategy')}",
        f"--split-stratify-attribute {config.get('split_stratify_attribute')}",
        f"--train-basin-count {config.get('train_basin_count')}",
        f"--val-basin-count {config.get('val_basin_count')}",
        f"--test-basin-count {config.get('test_basin_count')}",
        f"--output-dir {output_subdir}",
        f"--checkpoint {output_subdir}/best_model.pt",
    ]
    if config.get("model") == "transformer":
        command.extend(
            [
                f"--nhead {config.get('nhead')}",
                f"--dim-feedforward {config.get('dim_feedforward')}",
            ]
        )
    return " \\\n  ".join(command)


def _best_worst_text(run: Mapping[str, Any]) -> str:
    best = run["best"]
    worst = run["worst"]
    return (
        f"For {run['label']}, the best basin by NSE was {best['basin_id']} "
        f"(NSE={best['nse']:.3f}, KGE={best['kge']:.3f}, RMSE={best['rmse']:.3f}). "
        f"The poorest basin was {worst['basin_id']} "
        f"(NSE={worst['nse']:.3f}, KGE={worst['kge']:.3f}, RMSE={worst['rmse']:.3f})."
    )


def _seasonality_text(run: Mapping[str, Any]) -> str:
    monthly = run["monthly"].dropna(subset=["nse"])
    if monthly.empty:
        return ""
    high = monthly.loc[monthly["nse"].idxmax()]
    low = monthly.loc[monthly["nse"].idxmin()]
    return (
        f"{run['label']} had its strongest monthly NSE in month {int(high['month'])} "
        f"(NSE={high['nse']:.3f}) and weakest monthly NSE in month {int(low['month'])} "
        f"(NSE={low['nse']:.3f})."
    )


def _flow_regime_text(run: Mapping[str, Any]) -> str:
    regimes = run["regimes"].set_index("regime")
    pieces = []
    for regime in ["low_flow_obs_le_p10", "middle_flow", "peak_flow_obs_ge_p95"]:
        if regime in regimes.index:
            row = regimes.loc[regime]
            pieces.append(
                f"{regime.replace('_', ' ')} bias={row['percent_bias']:.1f}% "
                f"and NSE={row['nse']:.3f}"
            )
    return f"{run['label']} flow-regime behavior: " + "; ".join(pieces) + "."


def _cdf_text(run: Mapping[str, Any]) -> str:
    cdf = run["cdf"]
    if cdf.empty:
        return ""
    return (
        f"{run['label']} catchment KGE distribution had median KGE={cdf['median']:.3f}; "
        f"{_format_percent(cdf['fraction_below_0'])} of catchments were below KGE=0, "
        f"{_format_percent(cdf['fraction_above_0p5'])} were at or above KGE=0.5, and "
        f"{_format_percent(cdf['fraction_above_0p7'])} were at or above KGE=0.7."
    )


def _comparison_text(comparison_dir: Path) -> str:
    summary_path = comparison_dir / "model_metrics_summary.csv"
    if not summary_path.exists():
        return ""
    summary = pd.read_csv(summary_path)
    if len(summary) < 2:
        return ""
    rows = {row["label"]: row for _, row in summary.iterrows()}
    if "LSTM" not in rows or "Transformer" not in rows:
        return ""
    delta_nse = rows["Transformer"]["nse"] - rows["LSTM"]["nse"]
    delta_kge = rows["Transformer"]["kge"] - rows["LSTM"]["kge"]
    delta_rmse = rows["Transformer"]["rmse"] - rows["LSTM"]["rmse"]
    direction = "higher" if delta_nse > 0 else "lower"
    return (
        f"Relative to the LSTM, the Transformer had {direction} overall NSE "
        f"(delta={delta_nse:.3f}), KGE delta={delta_kge:.3f}, and RMSE delta={delta_rmse:.3f}. "
        "The basin-level comparison is therefore important: overall means can hide whether one model improves "
        "many catchments or only a few difficult ones."
    )


def _write_markdown_report(
    output_path: Path,
    repo_url: str,
    lstm: Mapping[str, Any],
    transformer: Mapping[str, Any],
    comparison_dir: Path,
) -> None:
    report_root = output_path.parent
    runs = [lstm, transformer]
    metric_table = _metric_summary_frame(runs)
    kge_cdf = pd.read_csv(comparison_dir / "basin_kge_cdf_summary_by_model.csv")
    flow_regime = pd.read_csv(comparison_dir / "flow_regime_comparison.csv")
    kge_display = kge_cdf.rename(
        columns={
            "model_label": "Model",
            "median": "Median KGE",
            "fraction_below_0": "Fraction KGE < 0",
            "fraction_above_0p5": "Fraction KGE >= 0.5",
            "fraction_above_0p7": "Fraction KGE >= 0.7",
        }
    )
    for column in ["Fraction KGE < 0", "Fraction KGE >= 0.5", "Fraction KGE >= 0.7"]:
        kge_display[column] = kge_display[column].map(_format_percent)

    low_peak = flow_regime[flow_regime["regime"].isin(["low_flow_obs_le_p10", "peak_flow_obs_ge_p95"])].copy()
    low_peak = low_peak.rename(
        columns={
            "model_label": "Model",
            "regime": "Regime",
            "nse": "NSE",
            "percent_bias": "Percent bias",
            "n_samples": "n",
        }
    )
    low_peak["Percent bias"] = low_peak["Percent bias"].map(lambda value: f"{float(value):.1f}%")

    figure_paths = {
        "overall": comparison_dir / "overall_metric_comparison.png",
        "parity": comparison_dir / "parity_overlay.png",
        "kge_cdf": comparison_dir / "basin_kge_cdf_comparison.png",
        "basin_delta": comparison_dir / "basin_nse_delta.png",
        "monthly": comparison_dir / "monthly_metric_comparison.png",
        "flow_regime": comparison_dir / "flow_regime_comparison.png",
        "best_ts": comparison_dir / "timeseries_comparison_best_reference_basin.png",
        "worst_ts": comparison_dir / "timeseries_comparison_worst_reference_basin.png",
        "best_fdc": comparison_dir / "flow_duration_comparison_best_reference_basin.png",
        "worst_fdc": comparison_dir / "flow_duration_comparison_worst_reference_basin.png",
    }

    lines = [
        "# Problem 4: Evaluation and Interpretation",
        "",
        f"**Git repository:** {repo_url or 'Add repository link here'}",
        "",
        "## Brief Methods Overview",
        "",
        "I evaluated two sequence models for held-out basin streamflow prediction: an LSTM and a Transformer. "
        "Both models used daily meteorological forcing sequences with static catchment attributes and predicted "
        "streamflow at the target time step after each sequence. The split was by basin, so held-out validation "
        "and test basins were not used during training. Normalization statistics were fit on training basins and "
        "then reused for validation and test data.",
        "",
        f"- LSTM best run: `{lstm['run_dir']}`",
        f"- LSTM configuration: {_run_config_summary(lstm)}",
        f"- Transformer best run: `{transformer['run_dir']}`",
        f"- Transformer configuration: {_run_config_summary(transformer)}",
        "",
        "The current runs use non-overlapping windows because `window_stride` equals `seq_len`. Therefore, the "
        "time-series figures show one prediction per non-overlapping sequence target rather than a fully daily "
        "rolling prediction.",
        "",
        "## Quantitative Test Metrics",
        "",
        _markdown_table(metric_table, ["Model", "NSE", "KGE", "RMSE", "MAE", "MSE"]),
        "",
        "NSE and KGE are hydrology-relevant skill metrics where higher values are better and 1.0 is ideal. "
        "RMSE, MAE, and MSE are error metrics where lower values are better.",
        "",
        "### KGE Over Catchments",
        "",
        _markdown_table(
            kge_display,
            ["Model", "Median KGE", "Fraction KGE < 0", "Fraction KGE >= 0.5", "Fraction KGE >= 0.7"],
        ),
        "",
        f"![Catchment KGE CDF]({_relative(figure_paths['kge_cdf'], report_root)})",
        "",
        "The KGE CDF shows the full distribution of catchment skill. A curve shifted to the right indicates that "
        "more catchments have higher KGE, not only that the mean score improved.",
        "",
        "### Low-Flow and Peak-Flow Metrics",
        "",
        _markdown_table(low_peak[["Model", "Regime", "NSE", "Percent bias", "n"]], ["Model", "Regime", "NSE", "Percent bias", "n"]),
        "",
        f"![Flow regime comparison]({_relative(figure_paths['flow_regime'], report_root)})",
        "",
        "Low flows were defined from the lowest 10% of observed test flows and peak flows from the highest 5% of "
        "observed test flows. This separates ordinary error from hydrologically important low-flow and high-flow behavior.",
        "",
        "## Requested Plots",
        "",
        "### Predicted Versus Observed Scatterplot",
        "",
        f"![Parity overlay]({_relative(figure_paths['parity'], report_root)})",
        "",
        "### Predicted Versus Observed Time Series",
        "",
        f"![Best basin time series]({_relative(figure_paths['best_ts'], report_root)})",
        "",
        f"![Worst basin time series]({_relative(figure_paths['worst_ts'], report_root)})",
        "",
        "### Basin-Level Model Comparison",
        "",
        f"![Per-basin NSE delta]({_relative(figure_paths['basin_delta'], report_root)})",
        "",
        "### Seasonality",
        "",
        f"![Monthly skill comparison]({_relative(figure_paths['monthly'], report_root)})",
        "",
        "### Flow-Duration Curves",
        "",
        f"![Best basin flow duration]({_relative(figure_paths['best_fdc'], report_root)})",
        "",
        f"![Worst basin flow duration]({_relative(figure_paths['worst_fdc'], report_root)})",
        "",
        "## Best and Poorly Performing Basins",
        "",
        _best_worst_text(lstm),
        "",
        _best_worst_text(transformer),
        "",
        "The best basins have hydrographs where the model captures the timing and magnitude of the dominant flow "
        "variability. The poorest basins show the opposite: errors in amplitude, weak low-flow behavior, or missed "
        "high-flow events can create very negative NSE/KGE even when RMSE is not visually large. This is especially "
        "important for basins with low observed variance, because NSE strongly penalizes errors relative to the "
        "observed variance.",
        "",
        "## Interpretation",
        "",
        _comparison_text(comparison_dir),
        "",
        _cdf_text(lstm),
        "",
        _cdf_text(transformer),
        "",
        _seasonality_text(lstm),
        "",
        _seasonality_text(transformer),
        "",
        _flow_regime_text(lstm),
        "",
        _flow_regime_text(transformer),
        "",
        "The LSTM imposes a recurrent inductive bias: information is compressed through a hidden state as the sequence "
        "is read. This can be helpful for smooth memory of antecedent wetness, but it may struggle when different "
        "parts of the input sequence matter unevenly. The Transformer can attend across the full sequence more directly, "
        "which may help with long-range meteorological dependencies, but it also has more flexibility and may be more "
        "sensitive to small training sets or noisy catchment-specific behavior. The basin-level plots are therefore "
        "more informative than a single overall score.",
        "",
        "The flow-regime analysis shows whether the models reproduce hydrologically important extremes. Large positive "
        "low-flow bias means the model predicts water when observed flow is near zero. Negative peak-flow bias means the "
        "model underestimates high-flow events. These errors can come from limited training examples for extremes, "
        "non-overlapping sequence sampling, and the difficulty of using basin-averaged static attributes to represent "
        "catchment storage, snow, and runoff generation processes.",
        "",
        "## Reproduction Commands",
        "",
        "Install dependencies and run from the repository root:",
        "",
        "```bash",
        "uv venv --python 3.11 .venv",
        "source .venv/bin/activate",
        "uv pip install -r requirements.txt",
        "```",
        "",
        "Reproduce the selected LSTM run:",
        "",
        "```bash",
        _training_command(lstm, "outputs/reproduce_best_lstm"),
        "python main.py evaluate --checkpoint outputs/reproduce_best_lstm/best_model.pt --output-dir outputs/reproduce_best_lstm",
        "python main.py analyze-run --run-dir outputs/reproduce_best_lstm",
        "```",
        "",
        "Reproduce the selected Transformer run:",
        "",
        "```bash",
        _training_command(transformer, "outputs/reproduce_best_transformer"),
        "python main.py evaluate --checkpoint outputs/reproduce_best_transformer/best_model.pt --output-dir outputs/reproduce_best_transformer",
        "python main.py analyze-run --run-dir outputs/reproduce_best_transformer",
        "```",
        "",
        "Recreate this comparison report:",
        "",
        "```bash",
        f"python main.py problem4-report --lstm-run-dir {lstm['run_dir']} --transformer-run-dir {transformer['run_dir']} --output-dir {report_root}",
        "```",
        "",
        "## Conclusion",
        "",
        "The strongest part of the workflow is the basin-held-out evaluation: the test metrics, KGE CDF, basin-level "
        "rankings, and flow-regime plots show model behavior beyond a single loss value. The models can capture useful "
        "streamflow variability for some basins, but performance is uneven across catchments and hydrologic regimes. "
        "The main weaknesses are low-flow bias, peak-flow underestimation, and sensitivity to basin-specific behavior. "
        "Future improvements should test denser rolling predictions, more extreme-flow-aware losses or sampling, and "
        "additional catchment attributes or process-informed features.",
        "",
    ]
    output_path.write_text("\n".join(lines))


def _pdf_text_page(pdf: PdfPages, title: str, paragraphs: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold", va="top")
    y = 0.89
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=_REPORT_WIDTH) or [""]
        for line in wrapped:
            fig.text(0.08, y, line, fontsize=10, va="top")
            y -= 0.022
        y -= 0.014
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: _format_float(value, 4))
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_image_page(pdf: PdfPages, title: str, images: Sequence[Path], captions: Sequence[str]) -> None:
    existing = [(image, caption) for image, caption in zip(images, captions) if image.exists()]
    if not existing:
        return
    n = len(existing)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5))
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(title, fontsize=16, weight="bold")
    for ax, (image_path, caption) in zip(axes, existing):
        ax.imshow(plt.imread(image_path))
        ax.axis("off")
        ax.set_title(caption, fontsize=10)
    for ax in axes[len(existing) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_pdf_report(
    output_path: Path,
    repo_url: str,
    lstm: Mapping[str, Any],
    transformer: Mapping[str, Any],
    comparison_dir: Path,
) -> None:
    metric_table = _metric_summary_frame([lstm, transformer])
    kge_cdf = pd.read_csv(comparison_dir / "basin_kge_cdf_summary_by_model.csv").rename(
        columns={
            "model_label": "Model",
            "median": "Median KGE",
            "fraction_below_0": "KGE < 0",
            "fraction_above_0p5": "KGE >= 0.5",
            "fraction_above_0p7": "KGE >= 0.7",
        }
    )
    for column in ["KGE < 0", "KGE >= 0.5", "KGE >= 0.7"]:
        kge_cdf[column] = kge_cdf[column].map(_format_percent)
    flow_regime = pd.read_csv(comparison_dir / "flow_regime_comparison.csv")
    low_peak = flow_regime[flow_regime["regime"].isin(["low_flow_obs_le_p10", "peak_flow_obs_ge_p95"])]
    low_peak = low_peak.rename(
        columns={
            "model_label": "Model",
            "regime": "Regime",
            "nse": "NSE",
            "percent_bias": "Percent bias",
            "n_samples": "n",
        }
    )
    low_peak["Percent bias"] = low_peak["Percent bias"].map(lambda value: f"{float(value):.1f}%")

    with PdfPages(output_path) as pdf:
        _pdf_text_page(
            pdf,
            "Problem 4: Evaluation and Interpretation",
            [
                f"Git repository: {repo_url or 'Add repository link here'}",
                "Methods: I evaluated the best LSTM and Transformer runs on held-out test basins. Both models use meteorological forcing sequences and static catchment attributes. The split is by basin, avoiding basin leakage between training, validation, and test.",
                f"LSTM run: {lstm['run_dir']}",
                f"LSTM configuration: {_run_config_summary(lstm)}",
                f"Transformer run: {transformer['run_dir']}",
                f"Transformer configuration: {_run_config_summary(transformer)}",
                "Because window_stride equals seq_len, the plotted hydrographs show non-overlapping sequence-target predictions rather than rolling daily predictions.",
            ],
        )
        _pdf_table_page(pdf, "Overall Held-Out Test Metrics", metric_table)
        _pdf_table_page(pdf, "Catchment KGE CDF Summary", kge_cdf[["Model", "Median KGE", "KGE < 0", "KGE >= 0.5", "KGE >= 0.7"]])
        _pdf_table_page(pdf, "Low-Flow and Peak-Flow Metrics", low_peak[["Model", "Regime", "NSE", "Percent bias", "n"]])
        _pdf_image_page(
            pdf,
            "Scatter and Catchment Skill",
            [
                comparison_dir / "parity_overlay.png",
                comparison_dir / "basin_kge_cdf_comparison.png",
            ],
            ["Predicted vs. observed", "KGE CDF over catchments"],
        )
        _pdf_image_page(
            pdf,
            "Best and Poorly Performing Basins",
            [
                comparison_dir / "timeseries_comparison_best_reference_basin.png",
                comparison_dir / "timeseries_comparison_worst_reference_basin.png",
            ],
            ["Best reference basin", "Worst reference basin"],
        )
        _pdf_image_page(
            pdf,
            "Basin and Seasonal Diagnostics",
            [
                comparison_dir / "basin_nse_delta.png",
                comparison_dir / "monthly_metric_comparison.png",
            ],
            ["Per-basin NSE delta", "Monthly NSE/KGE"],
        )
        _pdf_image_page(
            pdf,
            "Hydrologic Regime Diagnostics",
            [
                comparison_dir / "flow_regime_comparison.png",
                comparison_dir / "flow_duration_comparison_worst_reference_basin.png",
            ],
            ["Low/middle/peak flow behavior", "Worst-basin flow duration curve"],
        )
        _pdf_text_page(
            pdf,
            "Interpretation and Conclusion",
            [
                _best_worst_text(lstm),
                _best_worst_text(transformer),
                _comparison_text(comparison_dir),
                _cdf_text(lstm),
                _cdf_text(transformer),
                _seasonality_text(lstm),
                _seasonality_text(transformer),
                _flow_regime_text(lstm),
                _flow_regime_text(transformer),
                "The LSTM provides a recurrent memory structure, while the Transformer can attend across the sequence more directly. The comparison should be interpreted at both overall and basin levels because a model can improve the mean score while still failing in specific catchments or flow regimes.",
                "The main successful part of the workflow is useful streamflow skill in some basins and a clear basin-held-out evaluation. The main weaknesses are uneven basin performance, low-flow bias, and peak-flow underestimation. Future work should test rolling predictions, extreme-flow-aware objectives, and additional catchment/process information.",
            ],
        )


def create_problem4_report(args: Namespace) -> None:
    """Create Markdown and PDF reports for Problem 4."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lstm_run_dir = _resolve_run_dir(args.lstm_run_dir)
    transformer_run_dir = _resolve_run_dir(args.transformer_run_dir)
    repo_url = args.repo_url or _repo_url_from_git()

    lstm_analysis = _run_analysis("LSTM", lstm_run_dir, output_dir, args.data_dir, args.max_scatter_points)
    transformer_analysis = _run_analysis("Transformer", transformer_run_dir, output_dir, args.data_dir, args.max_scatter_points)
    comparison_dir = _run_comparison(lstm_run_dir, transformer_run_dir, output_dir, args.max_scatter_points)

    lstm = _collect_run("LSTM", lstm_run_dir, lstm_analysis)
    transformer = _collect_run("Transformer", transformer_run_dir, transformer_analysis)

    markdown_path = output_dir / "problem4_report.md"
    pdf_path = output_dir / "problem4_report.pdf"
    _write_markdown_report(markdown_path, repo_url, lstm, transformer, comparison_dir)
    _write_pdf_report(pdf_path, repo_url, lstm, transformer, comparison_dir)

    print(f"Problem 4 report written to {markdown_path}")
    print(f"Problem 4 PDF written to {pdf_path}")
