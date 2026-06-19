"""Streamlit dashboard for browsing evaluated streamflow runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytics.run_catalog import (  # noqa: E402
    get_monthly_summary,
    get_run,
    get_run_basin_metrics,
    get_run_timeseries,
    list_runs,
)


DEFAULT_API_URL = os.getenv("STREAMFLOW_API_URL", "http://127.0.0.1:8000")


def _get_json(api_url: str, path: str) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _load_data(api_url: str, use_api: bool) -> dict[str, Any]:
    if use_api:
        runs_payload = _get_json(api_url, "/runs")
        runs = runs_payload.get("runs", [])
        return {"runs": runs, "source": f"FastAPI ({api_url})"}

    runs = list_runs()
    return {"runs": runs, "source": "Local run artifacts"}


def _run_detail(run_id: str, api_url: str, use_api: bool) -> dict:
    if use_api:
        return _get_json(api_url, f"/runs/{run_id}")
    return get_run(run_id)


def _basin_rows(run_id: str, api_url: str, use_api: bool) -> list[dict]:
    if use_api:
        return _get_json(api_url, f"/runs/{run_id}/basins?limit=100")["rows"]
    return get_run_basin_metrics(run_id, limit=100)


def _monthly_rows(run_id: str, api_url: str, use_api: bool) -> list[dict]:
    if use_api:
        return _get_json(api_url, f"/runs/{run_id}/monthly")["rows"]
    return get_monthly_summary(run_id)


def _timeseries_rows(run_id: str, basin_id: str, api_url: str, use_api: bool) -> list[dict]:
    if use_api:
        return _get_json(api_url, f"/runs/{run_id}/basins/{basin_id}/timeseries")["rows"]
    return get_run_timeseries(run_id, basin_id=basin_id)


def main() -> None:
    st.set_page_config(page_title="Streamflow Run Explorer", layout="wide")
    st.title("Streamflow Run Explorer")
    st.caption("Compare evaluated model runs and inspect basin-level performance.")

    api_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)
    use_api = st.sidebar.toggle("Use FastAPI backend", value=False)

    try:
        payload = _load_data(api_url, use_api)
    except requests.RequestException as exc:
        st.error(
            "Could not reach the FastAPI service. Start it with "
            "`uvicorn api.main:app --reload` and check the API URL."
        )
        st.exception(exc)
        return
    except Exception as exc:
        st.error("Could not load evaluated run artifacts.")
        st.exception(exc)
        return

    runs = payload["runs"]
    if not runs:
        st.warning("No evaluated runs were found under `output/` or `outputs/`.")
        return

    st.sidebar.caption(f"Data source: {payload['source']}")
    run_options = {run["run_id"]: run for run in runs}
    selected_run_id = st.sidebar.selectbox("Select run", options=list(run_options))
    run = _run_detail(selected_run_id, api_url, use_api)

    overview = pd.DataFrame(runs)[
        ["run_id", "model", "seq_len", "batch_size", "learning_rate", "loss"]
    ].copy()
    overview["nse"] = pd.DataFrame(runs)["overall"].apply(lambda values: values.get("nse"))
    overview["kge"] = pd.DataFrame(runs)["overall"].apply(lambda values: values.get("kge"))

    st.subheader("Run Leaderboard")
    st.dataframe(overview.sort_values("nse", ascending=False), use_container_width=True)

    metrics = run.get("overall", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric("NSE", f"{metrics.get('nse', float('nan')):.3f}")
    metric_columns[1].metric("KGE", f"{metrics.get('kge', float('nan')):.3f}")
    metric_columns[2].metric("RMSE", f"{metrics.get('rmse', float('nan')):.3f}")
    metric_columns[3].metric("MAE", f"{metrics.get('mae', float('nan')):.3f}")

    st.subheader("Run Configuration")
    st.json(run.get("config", {}), expanded=False)

    basin_frame = pd.DataFrame(_basin_rows(selected_run_id, api_url, use_api))
    st.subheader("Top Basin Metrics")
    st.dataframe(basin_frame, use_container_width=True)

    monthly_frame = pd.DataFrame(_monthly_rows(selected_run_id, api_url, use_api))
    if not monthly_frame.empty:
        monthly_chart = monthly_frame.set_index("month")[["observed", "predicted"]]
        st.subheader("Monthly Mean Flow")
        st.line_chart(monthly_chart)

    if basin_frame.empty:
        return

    basin_options = basin_frame["basin_id"].astype(str).tolist()
    default_basin = str(run.get("best_basin_by_nse") or basin_options[0])
    default_index = basin_options.index(default_basin) if default_basin in basin_options else 0
    selected_basin = st.selectbox("Inspect basin", options=basin_options, index=default_index)

    timeseries_frame = pd.DataFrame(
        _timeseries_rows(selected_run_id, selected_basin, api_url, use_api)
    )
    if not timeseries_frame.empty:
        timeseries_chart = timeseries_frame.set_index("date")[["observed", "predicted"]]
        st.subheader(f"Basin {selected_basin} Time Series")
        st.line_chart(timeseries_chart)
        st.dataframe(timeseries_frame.tail(25), use_container_width=True)


if __name__ == "__main__":
    main()
