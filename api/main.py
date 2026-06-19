"""FastAPI endpoints for browsing evaluated streamflow runs."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from analytics.run_catalog import (
    get_monthly_summary,
    get_run,
    get_run_basin_metrics,
    get_run_timeseries,
    list_runs,
)


app = FastAPI(
    title="Streamflow Prediction API",
    description="Serve evaluated streamflow model artifacts for dashboards and lightweight demos.",
    version="0.1.0",
)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/runs")
def runs() -> dict:
    return {"runs": list_runs()}


@app.get("/runs/{run_id:path}/basins")
def run_basin_metrics(
    run_id: str,
    sort_by: str = Query(default="nse"),
    ascending: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    try:
        return {
            "run_id": run_id,
            "rows": get_run_basin_metrics(
                run_id=run_id,
                sort_by=sort_by,
                ascending=ascending,
                limit=limit,
            ),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs/{run_id:path}/basins/{basin_id}/timeseries")
def run_basin_timeseries(run_id: str, basin_id: str) -> dict:
    try:
        return {
            "run_id": run_id,
            "basin_id": basin_id,
            "rows": get_run_timeseries(run_id=run_id, basin_id=basin_id),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id:path}/monthly")
def run_monthly_summary(run_id: str) -> dict:
    try:
        return {
            "run_id": run_id,
            "rows": get_monthly_summary(run_id=run_id),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id:path}")
def run_detail(run_id: str) -> dict:
    try:
        return get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
