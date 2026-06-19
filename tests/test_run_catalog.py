"""Tests for evaluated run catalog helpers."""

from __future__ import annotations

import json

from analytics.run_catalog import (
    get_monthly_summary,
    get_run,
    get_run_basin_metrics,
    get_run_timeseries,
    list_runs,
)


def _write_demo_run(tmp_path) -> str:
    run_dir = tmp_path / "output" / "demo_run"
    run_dir.mkdir(parents=True)

    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "overall": {"nse": 0.8, "kge": 0.7, "rmse": 1.2, "mae": 0.9, "mse": 1.44},
                "best_basin_by_nse": "0101",
                "worst_basin_by_nse": "0102",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"model": "lstm", "seq_len": 30, "batch_size": 32, "lr": 0.001, "loss": "mse"}),
        encoding="utf-8",
    )
    (run_dir / "test_metrics_by_basin.csv").write_text(
        "basin_id,mse,mae,rmse,nse,kge,n_samples\n"
        "0101,1.0,0.8,1.0,0.9,0.8,3\n"
        "0102,2.0,1.2,1.4,0.1,0.2,3\n",
        encoding="utf-8",
    )
    (run_dir / "test_predictions.csv").write_text(
        "basin_id,date,observed,predicted\n"
        "0101,2001-01-01,1.0,1.1\n"
        "0101,2001-02-01,2.0,2.2\n"
        "0102,2001-01-01,3.0,2.5\n",
        encoding="utf-8",
    )
    return "output/demo_run"


def test_list_runs_discovers_evaluated_runs(tmp_path) -> None:
    run_id = _write_demo_run(tmp_path)

    runs = list_runs(run_roots=[tmp_path / "output"])

    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["overall"]["nse"] == 0.8


def test_run_detail_and_tables(tmp_path) -> None:
    run_id = _write_demo_run(tmp_path)

    run = get_run(run_id, run_roots=[tmp_path / "output"])
    basin_rows = get_run_basin_metrics(run_id, run_roots=[tmp_path / "output"])
    timeseries_rows = get_run_timeseries(run_id, basin_id="0101", run_roots=[tmp_path / "output"])
    monthly_rows = get_monthly_summary(run_id, run_roots=[tmp_path / "output"])

    assert run["model"] == "lstm"
    assert basin_rows[0]["basin_id"] == "0101"
    assert len(timeseries_rows) == 2
    assert len(monthly_rows) == 2
