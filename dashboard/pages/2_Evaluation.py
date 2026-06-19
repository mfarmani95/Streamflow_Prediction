"""Evaluation page for the Streamflow Run Explorer dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.shared import (
    configure_page,
    prediction_frame,
    render_histogram,
    render_scatter_chart,
    render_section_header,
)


def main() -> None:
    context = configure_page("Model Evaluation")
    render_section_header("Evaluation diagnostics", "Aggregate quality checks for the selected run")

    predictions = prediction_frame(context["selected_run_id"], context["api_url"], context["use_api"])
    if predictions.empty:
        st.warning("No prediction rows are available for this run.")
        st.stop()

    left, right = st.columns(2)
    with left:
        render_section_header("Parity", "Observed vs predicted calibration")
        sample = predictions.sample(min(len(predictions), 3000), random_state=42) if len(predictions) > 3000 else predictions
        render_scatter_chart(sample, "observed", "predicted")
    with right:
        render_section_header("Residual spread", "Distribution of prediction errors")
        render_histogram(predictions, "residual", bins=45)

    monthly = (
        predictions.assign(month=predictions["date"].dt.month_name().str.slice(0, 3))
        .groupby("month", as_index=False)["residual"]
        .mean()
    )
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["month"] = pd.Categorical(monthly["month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("month")

    bottom_left, bottom_right = st.columns([1.15, 1])
    with bottom_left:
        render_section_header("Seasonal bias", "Mean residual by month")
        st.bar_chart(monthly.set_index("month")["residual"], color="#F59E0B", use_container_width=True)
    with bottom_right:
        render_section_header("Error summary", "Residual and magnitude overview")
        summary = pd.DataFrame(
            {
                "metric": ["Mean residual", "Median residual", "Residual std", "Abs residual mean"],
                "value": [
                    predictions["residual"].mean(),
                    predictions["residual"].median(),
                    predictions["residual"].std(),
                    predictions["residual"].abs().mean(),
                ],
            }
        )
        st.dataframe(summary.style.format({"value": "{:.3f}"}), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
