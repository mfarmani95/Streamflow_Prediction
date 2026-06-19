"""Overview page for the Streamflow Run Explorer dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.shared import (  # noqa: E402
    basin_rows,
    configure_page,
    format_metric,
    monthly_rows,
    prepare_overview_frame,
    render_hero,
    render_info_card,
    render_section_header,
    render_timeseries_chart,
)


def main() -> None:
    context = configure_page("Streamflow Run Explorer")
    run = context["run"]
    runs = context["runs"]
    render_hero(run, len(runs), context["payload"]["source"])

    metrics = run.get("overall", {})
    metric_columns = st.columns(4)
    for column, label, key in zip(
        metric_columns,
        ["NSE", "KGE", "RMSE", "MAE"],
        ["nse", "kge", "rmse", "mae"],
    ):
        with column:
            st.markdown('<div class="metric-shell">', unsafe_allow_html=True)
            st.metric(label, format_metric(metrics.get(key)))
            st.markdown("</div>", unsafe_allow_html=True)

    info_columns = st.columns(4)
    with info_columns[0]:
        render_info_card("Model", str(run.get("model") or "Unknown").upper(), "Architecture for this evaluated run")
    with info_columns[1]:
        render_info_card("Sequence length", str(run.get("seq_len") or "N/A"), "Input window used for prediction")
    with info_columns[2]:
        render_info_card("Batch size", str(run.get("batch_size") or "N/A"), "Training batch size recorded in run config")
    with info_columns[3]:
        render_info_card("Learning rate", str(run.get("learning_rate") or "N/A"), "Optimization step size for this run")

    overview = prepare_overview_frame(runs)
    monthly_frame = pd.DataFrame(monthly_rows(context["selected_run_id"], context["api_url"], context["use_api"]))
    basin_frame = pd.DataFrame(
        basin_rows(
            context["selected_run_id"],
            context["api_url"],
            context["use_api"],
            context["sort_by"],
            context["ascending"],
            context["basin_limit"],
        )
    )

    left, right = st.columns([1.25, 1])
    with left:
        render_section_header("Run comparison", "Leaderboard across evaluated experiments")
        leaderboard = overview[
            ["run_label", "model", "seq_len", "batch_size", "learning_rate", "loss", "nse", "kge", "rmse"]
        ].rename(columns={"run_label": "run"})
        st.dataframe(
            leaderboard.style.format(
                {"learning_rate": "{:.4g}", "nse": "{:.3f}", "kge": "{:.3f}", "rmse": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        render_section_header("Run signal", "Monthly mean observed vs predicted flow")
        if monthly_frame.empty:
            st.info("Monthly summaries are not available for this run.")
        else:
            render_timeseries_chart(monthly_frame, "month")
            residual_summary = monthly_frame["residual"].abs().mean()
            st.caption(f"Average monthly absolute residual: {format_metric(residual_summary)}")

    render_section_header("Basin snapshot", "Top basins for the selected run")
    if not basin_frame.empty:
        st.dataframe(
            basin_frame.head(12).style.format(
                {"mse": "{:.3f}", "mae": "{:.3f}", "rmse": "{:.3f}", "nse": "{:.3f}", "kge": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
