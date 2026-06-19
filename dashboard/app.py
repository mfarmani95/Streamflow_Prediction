"""Streamlit dashboard for browsing evaluated streamflow runs."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("STREAMFLOW_API_URL", "http://127.0.0.1:8000")


def _get_json(api_url: str, path: str) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Streamflow Run Explorer", layout="wide")
    st.title("Streamflow Run Explorer")
    st.caption("Compare evaluated model runs and inspect basin-level performance.")

    api_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)

    try:
        runs_payload = _get_json(api_url, "/runs")
    except requests.RequestException as exc:
        st.error(
            "Could not reach the FastAPI service. Start it with "
            "`uvicorn api.main:app --reload` and check the API URL."
        )
        st.exception(exc)
        return

    runs = runs_payload.get("runs", [])
    if not runs:
        st.warning("No evaluated runs were found under `output/` or `outputs/`.")
        return

    run_options = {run["run_id"]: run for run in runs}
    selected_run_id = st.sidebar.selectbox("Select run", options=list(run_options))
    run = _get_json(api_url, f"/runs/{selected_run_id}")

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

    basin_payload = _get_json(api_url, f"/runs/{selected_run_id}/basins?limit=100")
    basin_frame = pd.DataFrame(basin_payload["rows"])
    st.subheader("Top Basin Metrics")
    st.dataframe(basin_frame, use_container_width=True)

    monthly_payload = _get_json(api_url, f"/runs/{selected_run_id}/monthly")
    monthly_frame = pd.DataFrame(monthly_payload["rows"])
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

    timeseries_payload = _get_json(
        api_url,
        f"/runs/{selected_run_id}/basins/{selected_basin}/timeseries",
    )
    timeseries_frame = pd.DataFrame(timeseries_payload["rows"])
    if not timeseries_frame.empty:
        timeseries_chart = timeseries_frame.set_index("date")[["observed", "predicted"]]
        st.subheader(f"Basin {selected_basin} Time Series")
        st.line_chart(timeseries_chart)
        st.dataframe(timeseries_frame.tail(25), use_container_width=True)


if __name__ == "__main__":
    main()
