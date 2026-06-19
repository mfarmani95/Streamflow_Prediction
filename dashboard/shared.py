"""Shared helpers for the Streamlit dashboard pages."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytics.run_catalog import (  # noqa: E402
    get_basin_coordinates,
    get_monthly_summary,
    get_run,
    get_run_basin_metrics,
    get_run_predictions,
    get_run_timeseries,
    list_runs,
)


DEFAULT_API_URL = os.getenv("STREAMFLOW_API_URL", "http://127.0.0.1:8000")
ACCENT = "#0F766E"
ACCENT_DARK = "#134E4A"
INK = "#102A43"
MUTED = "#486581"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
                linear-gradient(180deg, #F8FBFD 0%, #F2F7F9 100%);
        }}
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #132238 100%);
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #E6EEF8;
        }}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input {{
            color: #102A43 !important;
            -webkit-text-fill-color: #102A43 !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] > div,
        [data-testid="stSidebar"] [data-testid="stNumberInputRootElement"] > div {{
            background: rgba(255, 255, 255, 0.94) !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
            color: #102A43 !important;
        }}
        [data-testid="stSidebar"] svg {{
            fill: #102A43;
            color: #102A43;
        }}
        .hero-card {{
            padding: 1.6rem 1.7rem;
            border-radius: 24px;
            background: linear-gradient(135deg, {ACCENT_DARK} 0%, {ACCENT} 55%, #38B2AC 100%);
            color: white;
            box-shadow: 0 22px 48px rgba(15, 118, 110, 0.22);
            margin-bottom: 1.1rem;
        }}
        .hero-title {{
            font-size: 2.25rem;
            line-height: 1.05;
            font-weight: 700;
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.03em;
        }}
        .hero-subtitle {{
            font-size: 1rem;
            max-width: 820px;
            opacity: 0.92;
            margin: 0;
        }}
        .section-kicker {{
            color: {ACCENT};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .section-title {{
            color: {INK};
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }}
        .metric-shell {{
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 118, 110, 0.10);
            border-radius: 20px;
            padding: 0.35rem 0.35rem 0.1rem 0.35rem;
            box-shadow: 0 12px 28px rgba(16, 42, 67, 0.06);
        }}
        .info-card {{
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(16, 42, 67, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(16, 42, 67, 0.05);
            min-height: 122px;
        }}
        .info-label {{
            color: {MUTED};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }}
        .info-value {{
            color: {INK};
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }}
        .info-note {{
            color: {MUTED};
            font-size: 0.9rem;
            margin: 0;
        }}
        .context-band {{
            margin: 0.9rem 0 1.2rem 0;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(15, 118, 110, 0.10);
            color: {INK};
        }}
        div[data-testid="stMetric"] {{
            background: transparent;
            border-radius: 16px;
            padding: 0.75rem 0.9rem 0.8rem 0.9rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {MUTED};
        }}
        div[data-testid="stDataFrame"] {{
            background: rgba(255,255,255,0.72);
            border-radius: 18px;
            padding: 0.2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_json(api_url: str, path: str) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def load_data(api_url: str, use_api: bool) -> dict[str, Any]:
    if use_api:
        runs_payload = get_json(api_url, "/runs")
        runs = runs_payload.get("runs", [])
        return {"runs": runs, "source": f"FastAPI ({api_url})"}
    return {"runs": list_runs(), "source": "Local run artifacts"}


def run_detail(run_id: str, api_url: str, use_api: bool) -> dict:
    if use_api:
        return get_json(api_url, f"/runs/{run_id}")
    return get_run(run_id)


def basin_rows(
    run_id: str,
    api_url: str,
    use_api: bool,
    sort_by: str,
    ascending: bool,
    limit: int,
) -> list[dict]:
    if use_api:
        return get_json(
            api_url,
            f"/runs/{run_id}/basins?limit={limit}&sort_by={sort_by}&ascending={str(ascending).lower()}",
        )["rows"]
    return get_run_basin_metrics(run_id, sort_by=sort_by, ascending=ascending, limit=limit)


def monthly_rows(run_id: str, api_url: str, use_api: bool) -> list[dict]:
    if use_api:
        return get_json(api_url, f"/runs/{run_id}/monthly")["rows"]
    return get_monthly_summary(run_id)


def timeseries_rows(run_id: str, basin_id: str, api_url: str, use_api: bool) -> list[dict]:
    if use_api:
        return get_json(api_url, f"/runs/{run_id}/basins/{basin_id}/timeseries")["rows"]
    return get_run_timeseries(run_id, basin_id=basin_id)


def prediction_frame(run_id: str, api_url: str, use_api: bool) -> pd.DataFrame:
    if use_api:
        run = run_detail(run_id, api_url, use_api=True)
        path = Path(run["artifact_files"]["predictions"])
        frame = pd.read_csv(path, dtype={"basin_id": str})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["residual"] = frame["predicted"] - frame["observed"]
        return frame
    return get_run_predictions(run_id)


def basin_coordinate_frame() -> pd.DataFrame:
    return get_basin_coordinates()


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_hero(run: dict, run_count: int, data_source: str) -> None:
    model_name = str(run.get("model") or "Unknown").upper()
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Streamflow Run Explorer</div>
            <p class="hero-subtitle">
                Production-style evaluation console for hydrologic model runs. Compare experiments,
                inspect basin behavior, and surface performance patterns with a cleaner story for
                engineering and stakeholder demos.
            </p>
        </div>
        <div class="context-band">
            <strong>Selected model:</strong> {model_name}
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <strong>Visible runs:</strong> {run_count}
            &nbsp;&nbsp;•&nbsp;&nbsp;
            <strong>Data source:</strong> {data_source}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <p class="info-note">{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_overview_frame(runs: list[dict]) -> pd.DataFrame:
    overview = pd.DataFrame(runs)[
        ["run_id", "model", "seq_len", "batch_size", "learning_rate", "loss"]
    ].copy()
    overall = pd.DataFrame(runs)["overall"]
    overview["nse"] = overall.apply(lambda values: values.get("nse"))
    overview["kge"] = overall.apply(lambda values: values.get("kge"))
    overview["rmse"] = overall.apply(lambda values: values.get("rmse"))
    overview["mae"] = overall.apply(lambda values: values.get("mae"))
    overview["run_label"] = overview["run_id"].apply(lambda value: str(value).split("/")[-1])
    return overview.sort_values("nse", ascending=False).reset_index(drop=True)


def render_section_header(kicker: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def render_timeseries_chart(frame: pd.DataFrame, x_field: str, height: int = 320) -> None:
    chart_frame = frame.copy()
    chart_frame[x_field] = pd.to_datetime(chart_frame[x_field])
    long_frame = chart_frame.melt(
        id_vars=[x_field],
        value_vars=["observed", "predicted"],
        var_name="series",
        value_name="flow",
    )
    long_frame["series_label"] = long_frame["series"].map({"observed": "Observed", "predicted": "Predicted"})
    chart = (
        alt.Chart(long_frame)
        .mark_line(strokeWidth=2.6)
        .encode(
            x=alt.X(f"{x_field}:T", title=None),
            y=alt.Y("flow:Q", title="Flow"),
            color=alt.Color(
                "series_label:N",
                scale=alt.Scale(domain=["Observed", "Predicted"], range=["#0F766E", "#F59E0B"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "series_label:N",
                scale=alt.Scale(domain=["Observed", "Predicted"], range=[[1, 0], [8, 5]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{x_field}:T", title="Date"),
                alt.Tooltip("series_label:N", title="Series"),
                alt.Tooltip("flow:Q", title="Flow", format=".3f"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor="rgba(15, 23, 42, 0.08)")
        .configure_legend(labelColor=INK, titleColor=INK)
    )
    st.altair_chart(chart, use_container_width=True)


def render_bar_chart(
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    color: str = ACCENT,
    height: int = 340,
    x_title: str | None = None,
    y_title: str | None = None,
) -> None:
    chart = (
        alt.Chart(frame)
        .mark_bar(size=18, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color)
        .encode(
            x=alt.X(f"{x_field}:N", sort=None, title=x_title),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[alt.Tooltip(f"{x_field}:N"), alt.Tooltip(f"{y_field}:Q", format=".3f")],
        )
        .properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor="rgba(15, 23, 42, 0.08)")
    )
    st.altair_chart(chart, use_container_width=True)


def render_scatter_chart(frame: pd.DataFrame, x_field: str, y_field: str, color_field: str | None = None) -> None:
    chart = (
        alt.Chart(frame)
        .mark_circle(size=34, opacity=0.34, color=ACCENT)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_field.replace("_", " ").title()),
            y=alt.Y(f"{y_field}:Q", title=y_field.replace("_", " ").title()),
            tooltip=[
                alt.Tooltip(f"{x_field}:Q", format=".3f"),
                alt.Tooltip(f"{y_field}:Q", format=".3f"),
            ],
            color=alt.value(ACCENT) if color_field is None else alt.Color(f"{color_field}:N"),
        )
        .properties(height=360)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor="rgba(15, 23, 42, 0.08)")
    )
    st.altair_chart(chart, use_container_width=True)


def render_histogram(frame: pd.DataFrame, field: str, bins: int = 40, color: str = "#F59E0B") -> None:
    chart = (
        alt.Chart(frame)
        .mark_bar(color=color, opacity=0.86)
        .encode(
            x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=field.replace("_", " ").title()),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=320)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor="rgba(15, 23, 42, 0.08)")
    )
    st.altair_chart(chart, use_container_width=True)


def configure_page(title: str) -> dict[str, Any]:
    st.set_page_config(
        page_title=title,
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    st.sidebar.markdown("## Control Room")
    st.sidebar.caption("Choose the run, basin view, and data backend.")
    api_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)
    use_api = st.sidebar.toggle("Use FastAPI backend", value=False)
    payload = load_data(api_url, use_api)
    runs = payload["runs"]
    if not runs:
        st.warning("No evaluated runs were found under `output/` or `outputs/`.")
        st.stop()

    selected_run_id = st.sidebar.selectbox(
        "Selected run",
        options=[run["run_id"] for run in runs],
        format_func=lambda value: value.split("/")[-1],
    )
    sort_by = st.sidebar.selectbox("Basin ranking metric", options=["nse", "kge", "rmse", "mae", "mse"])
    ascending = st.sidebar.toggle("Ascending sort", value=sort_by in {"rmse", "mae", "mse"})
    basin_limit = st.sidebar.slider("Basin rows to display", min_value=10, max_value=150, value=40, step=10)
    run = run_detail(selected_run_id, api_url, use_api)
    return {
        "api_url": api_url,
        "use_api": use_api,
        "payload": payload,
        "runs": runs,
        "run": run,
        "selected_run_id": selected_run_id,
        "sort_by": sort_by,
        "ascending": ascending,
        "basin_limit": basin_limit,
    }
