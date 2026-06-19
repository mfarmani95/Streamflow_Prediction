"""Configuration page for the Streamflow Run Explorer dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.shared import configure_page, render_info_card, render_section_header


def main() -> None:
    context = configure_page("Run Configuration")
    run = context["run"]

    left, right = st.columns([1.2, 1])
    with left:
        render_section_header("Run metadata", "Configuration recorded for this experiment")
        st.json(run.get("config", {}), expanded=False)
    with right:
        render_section_header("Selection context", "Run-level summary")
        render_info_card("Run ID", context["selected_run_id"].split("/")[-1], "Short label used throughout the dashboard")
        render_info_card(
            "Best basin",
            str(run.get("best_basin_by_nse") or "N/A"),
            "Highest NSE basin from saved evaluation outputs",
        )
        render_info_card(
            "Worst basin",
            str(run.get("worst_basin_by_nse") or "N/A"),
            "Lowest NSE basin from saved evaluation outputs",
        )
        render_info_card("Loss", str(run.get("loss") or "N/A").upper(), "Objective recorded in run config")


if __name__ == "__main__":
    main()
