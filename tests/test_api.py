"""API smoke tests for the FastAPI application."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import app


def _write_demo_run(tmp_path) -> None:
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
    (run_dir / "run_config.json").write_text(json.dumps({"model": "lstm"}), encoding="utf-8")
    (run_dir / "test_metrics_by_basin.csv").write_text(
        "basin_id,mse,mae,rmse,nse,kge,n_samples\n0101,1.0,0.8,1.0,0.9,0.8,3\n",
        encoding="utf-8",
    )
    (run_dir / "test_predictions.csv").write_text(
        "basin_id,date,observed,predicted\n0101,2001-01-01,1.0,1.1\n",
        encoding="utf-8",
    )


def test_runs_endpoint_lists_discovered_runs(tmp_path, monkeypatch) -> None:
    _write_demo_run(tmp_path)
    monkeypatch.setenv("STREAMFLOW_RUN_ROOTS", str(tmp_path / "output"))
    client = TestClient(app)

    response = client.get("/runs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"][0]["run_id"] == "output/demo_run"
