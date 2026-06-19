"""Catalog and query evaluated run artifacts for APIs and dashboards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOTS = ("output", "outputs")


def _configured_run_roots(run_roots: Optional[Iterable[Path | str]] = None) -> list[Path]:
    if run_roots is not None:
        return [Path(root) for root in run_roots]

    configured = os.getenv("STREAMFLOW_RUN_ROOTS")
    if configured:
        roots = [Path(value.strip()) for value in configured.split(",") if value.strip()]
    else:
        roots = [REPO_ROOT / root for root in DEFAULT_RUN_ROOTS]

    return [root for root in roots if root.exists()]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_id_from_dir(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        for marker in DEFAULT_RUN_ROOTS:
            if marker in run_dir.parts:
                start = run_dir.parts.index(marker)
                return Path(*run_dir.parts[start:]).as_posix()
        return run_dir.as_posix()


def _discover_run_dirs(run_roots: Optional[Iterable[Path | str]] = None) -> list[Path]:
    run_dirs: list[Path] = []
    for root in _configured_run_roots(run_roots):
        run_dirs.extend(path.parent for path in root.rglob("test_metrics.json"))
    return sorted(set(run_dirs))


def _build_run_summary(run_dir: Path) -> dict:
    metrics_path = run_dir / "test_metrics.json"
    predictions_path = run_dir / "test_predictions.csv"
    basin_metrics_path = run_dir / "test_metrics_by_basin.csv"
    config_path = run_dir / "run_config.json"

    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Run artifacts missing from {run_dir}")

    metrics = _read_json(metrics_path)
    config = _read_json(config_path) if config_path.exists() else {}
    overall = metrics.get("overall", {})

    return {
        "run_id": _run_id_from_dir(run_dir),
        "run_dir": str(run_dir),
        "model": config.get("model"),
        "seq_len": config.get("seq_len"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("lr"),
        "loss": config.get("loss"),
        "overall": overall,
        "best_basin_by_nse": metrics.get("best_basin_by_nse"),
        "worst_basin_by_nse": metrics.get("worst_basin_by_nse"),
        "artifact_files": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "basin_metrics": str(basin_metrics_path) if basin_metrics_path.exists() else None,
            "config": str(config_path) if config_path.exists() else None,
        },
        "config": config,
    }


def list_runs(run_roots: Optional[Iterable[Path | str]] = None) -> list[dict]:
    """Return evaluated runs sorted by descending NSE when available."""
    runs = [_build_run_summary(run_dir) for run_dir in _discover_run_dirs(run_roots)]
    return sorted(
        runs,
        key=lambda run: float(run.get("overall", {}).get("nse", float("-inf"))),
        reverse=True,
    )


def get_run(run_id: str, run_roots: Optional[Iterable[Path | str]] = None) -> dict:
    """Return one run summary by its repo-relative run id."""
    normalized = run_id.strip().strip("/")
    for run in list_runs(run_roots):
        if run["run_id"] == normalized:
            return run
    raise KeyError(f"Run not found: {run_id}")


def _predictions_path_for_run(run_id: str, run_roots: Optional[Iterable[Path | str]] = None) -> Path:
    run = get_run(run_id, run_roots)
    return Path(run["artifact_files"]["predictions"])


def _basin_metrics_path_for_run(run_id: str, run_roots: Optional[Iterable[Path | str]] = None) -> Path:
    run = get_run(run_id, run_roots)
    path = run["artifact_files"]["basin_metrics"]
    if not path:
        raise FileNotFoundError(f"Missing basin metrics for run: {run_id}")
    return Path(path)


def get_run_basin_metrics(
    run_id: str,
    sort_by: str = "nse",
    ascending: bool = False,
    limit: int = 50,
    run_roots: Optional[Iterable[Path | str]] = None,
) -> list[dict]:
    """Return basin-level metrics for one run."""
    basin_metrics = pd.read_csv(_basin_metrics_path_for_run(run_id, run_roots), dtype={"basin_id": str})
    if sort_by not in basin_metrics.columns:
        raise ValueError(f"Unknown basin metric column: {sort_by}")

    ordered = basin_metrics.sort_values(sort_by, ascending=ascending).head(limit)
    return ordered.to_dict(orient="records")


def _load_predictions_with_duckdb(predictions_path: Path) -> "pd.DataFrame":
    import duckdb

    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT basin_id, CAST(date AS VARCHAR) AS date, observed, predicted FROM read_csv_auto(?)",
            [str(predictions_path)],
        ).df()
    finally:
        connection.close()


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    try:
        frame = _load_predictions_with_duckdb(predictions_path)
    except ImportError:
        frame = pd.read_csv(predictions_path, dtype={"basin_id": str})

    frame["basin_id"] = frame["basin_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["residual"] = frame["predicted"] - frame["observed"]
    return frame.sort_values(["basin_id", "date"]).reset_index(drop=True)


def get_run_predictions(
    run_id: str,
    run_roots: Optional[Iterable[Path | str]] = None,
) -> pd.DataFrame:
    """Return all prediction rows for one run."""
    return _load_predictions(_predictions_path_for_run(run_id, run_roots))


def get_run_timeseries(
    run_id: str,
    basin_id: str,
    run_roots: Optional[Iterable[Path | str]] = None,
) -> list[dict]:
    """Return observed/predicted time series for one basin."""
    predictions = _load_predictions(_predictions_path_for_run(run_id, run_roots))
    basin_frame = predictions[predictions["basin_id"].astype(str) == str(basin_id)].copy()
    if basin_frame.empty:
        raise KeyError(f"Basin {basin_id} not found for run {run_id}")

    basin_frame["date"] = basin_frame["date"].dt.strftime("%Y-%m-%d")
    return basin_frame.to_dict(orient="records")


def get_monthly_summary(
    run_id: str,
    run_roots: Optional[Iterable[Path | str]] = None,
) -> list[dict]:
    """Aggregate observed and predicted monthly means for one run."""
    predictions = _load_predictions(_predictions_path_for_run(run_id, run_roots))
    monthly = (
        predictions.assign(month=predictions["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)[["observed", "predicted", "residual"]]
        .mean()
        .sort_values("month")
    )
    return monthly.to_dict(orient="records")
