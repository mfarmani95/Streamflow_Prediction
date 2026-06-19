"""Scenario comparison page for the Streamflow Run Explorer dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.shared import (  # noqa: E402
    configure_page,
    prepare_experiment_frame,
    render_line_chart,
    render_section_header,
)


SCENARIO_FIELDS = [
    "batch_size",
    "hidden_size",
    "seq_len",
    "learning_rate",
    "dropout",
    "num_layers",
    "nhead",
    "dim_feedforward",
]
METRIC_FIELDS = ["kge", "nse", "rmse", "mae", "mse"]


def _scenario_subset(
    frame: pd.DataFrame,
    anchor_row: pd.Series,
    x_field: str,
    series_field: str | None,
) -> pd.DataFrame:
    compare_fields = [
        "model",
        "loss",
        "seq_len",
        "batch_size",
        "learning_rate",
        "hidden_size",
        "num_layers",
        "dropout",
        "nhead",
        "dim_feedforward",
        "forecast_horizon",
        "split_strategy",
        "train_basin_count",
        "val_basin_count",
        "test_basin_count",
    ]
    ignored = {x_field}
    if series_field and series_field != "none":
        ignored.add(series_field)

    subset = frame.copy()
    for field in compare_fields:
        if field in ignored or field not in subset.columns:
            continue
        anchor_value = anchor_row[field]
        if pd.isna(anchor_value):
            subset = subset[subset[field].isna()]
        else:
            subset = subset[subset[field] == anchor_value]
    return subset


def main() -> None:
    context = configure_page("Scenario Analysis")
    render_section_header("Scenario analysis", "Compare how performance changes across experiment settings")

    experiments = prepare_experiment_frame(context["runs"])
    anchor = experiments.loc[experiments["run_id"] == context["selected_run_id"]]
    if anchor.empty:
        st.warning("Selected run could not be found in the experiment table.")
        st.stop()
    anchor_row = anchor.iloc[0]

    control_left, control_mid, control_right = st.columns([1, 1, 1.1])
    with control_left:
        x_field = st.selectbox("X-axis hyperparameter", options=SCENARIO_FIELDS, index=0)
    with control_mid:
        metric_field = st.selectbox("Performance metric", options=METRIC_FIELDS, index=0)
    with control_right:
        series_field = st.selectbox(
            "Split lines by",
            options=["none"] + [field for field in SCENARIO_FIELDS if field != x_field],
            index=0,
        )

    subset = _scenario_subset(experiments, anchor_row, x_field, series_field)
    subset = subset.dropna(subset=[x_field, metric_field]).sort_values(x_field)

    st.markdown(
        f"""
        <div class="context-band">
            Anchor run: <strong>{anchor_row['run_label']}</strong>.
            The scenario view keeps the other experiment settings fixed and varies
            <strong>{x_field}</strong>{'' if series_field == 'none' else f' plus {series_field}'}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if subset.empty or subset[x_field].nunique() < 2:
        st.info(
            "Not enough matching runs were found to compare this scenario. "
            "Try a different hyperparameter or choose another anchor run."
        )
        st.stop()

    plot_left, plot_right = st.columns([1.35, 1])
    with plot_left:
        render_section_header("Scenario trend", f"{metric_field.upper()} response to {x_field}")
        render_line_chart(subset, x_field=x_field, y_field=metric_field, series_field=series_field)
    with plot_right:
        render_section_header("Scenario table", "Runs included in the comparison")
        visible_columns = ["run_label", "model", "loss", x_field, metric_field]
        if series_field != "none":
            visible_columns.insert(4, series_field)
        st.dataframe(
            subset[visible_columns].style.format({metric_field: "{:.3f}", "learning_rate": "{:.4g}"}),
            use_container_width=True,
            hide_index=True,
        )

    render_section_header("Anchor configuration", "Settings held constant for this comparison")
    held_constant = pd.DataFrame(
        {
            "parameter": [
                "model",
                "loss",
                "seq_len",
                "batch_size",
                "learning_rate",
                "hidden_size",
                "num_layers",
                "dropout",
                "nhead",
                "dim_feedforward",
            ],
            "value": [
                anchor_row["model"],
                anchor_row["loss"],
                anchor_row["seq_len"],
                anchor_row["batch_size"],
                anchor_row["learning_rate"],
                anchor_row["hidden_size"],
                anchor_row["num_layers"],
                anchor_row["dropout"],
                anchor_row["nhead"],
                anchor_row["dim_feedforward"],
            ],
        }
    )
    st.dataframe(held_constant, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
