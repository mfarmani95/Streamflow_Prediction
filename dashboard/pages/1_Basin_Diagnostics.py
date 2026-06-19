"""Basin diagnostics page for the Streamflow Run Explorer dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from dashboard.shared import (
    basin_coordinate_frame,
    basin_rows,
    configure_page,
    render_bar_chart,
    render_section_header,
    render_timeseries_chart,
    timeseries_rows,
)


def _flow_duration_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["observed", "predicted"]:
        values = np.sort(frame[label].dropna().to_numpy(dtype=float))[::-1]
        exceedance = np.arange(1, len(values) + 1) / len(values) * 100.0
        rows.append(pd.DataFrame({"exceedance": exceedance, "flow": values, "series": label.title()}))
    return pd.concat(rows, ignore_index=True)


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    value = hex_value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _interpolate_color(start: str, end: str, ratio: float) -> list[int]:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    ratio = min(max(ratio, 0.0), 1.0)
    return [
        int(round(start_channel + (end_channel - start_channel) * ratio))
        for start_channel, end_channel in zip(start_rgb, end_rgb)
    ] + [190]


def _metric_color_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    colored = frame.copy()
    values = pd.to_numeric(colored[metric], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        colored["color"] = [[15, 118, 110, 180]] * len(colored)
        return colored

    metric_lower_is_better = metric in {"mse", "mae", "rmse"}
    minimum = float(valid.min())
    maximum = float(valid.max())
    span = max(maximum - minimum, 1e-12)

    def _color_for_value(value: float) -> list[int]:
        normalized = (float(value) - minimum) / span
        score = 1.0 - normalized if metric_lower_is_better else normalized
        return _interpolate_color("#E76F51", "#0F766E", score)

    colored["color"] = values.map(_color_for_value)
    return colored


def _render_basin_metric_map(frame: pd.DataFrame, metric: str) -> None:
    map_frame = frame.dropna(subset=["lat", "lon", metric]).copy()
    if map_frame.empty:
        st.info("No basin coordinates are available for map rendering.")
        return

    map_frame = _metric_color_frame(map_frame, metric)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_frame,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=32000,
        pickable=True,
        opacity=0.82,
        stroked=True,
        get_line_color=[255, 255, 255, 120],
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(
        latitude=float(map_frame["lat"].mean()),
        longitude=float(map_frame["lon"].mean()),
        zoom=3.15,
        pitch=0,
    )
    tooltip = {
        "html": "<b>Basin:</b> {basin_id}<br/><b>" + metric.upper() + ":</b> {" + metric + "}",
        "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white"},
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip=tooltip,
        ),
        use_container_width=True,
    )


def main() -> None:
    context = configure_page("Basin Diagnostics")
    render_section_header("Basin diagnostics", "Inspect where the selected run succeeds or struggles")

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
    if basin_frame.empty:
        st.warning("No basin metrics were found for this run.")
        st.stop()

    coordinates = basin_coordinate_frame()
    basin_frame = basin_frame.merge(coordinates, on="basin_id", how="left")

    basin_options = basin_frame["basin_id"].astype(str).tolist()
    default_basin = str(context["run"].get("best_basin_by_nse") or basin_options[0])
    default_index = basin_options.index(default_basin) if default_basin in basin_options else 0
    selected_basin = st.selectbox("Inspect basin", options=basin_options, index=default_index)
    map_metric = st.selectbox("Map color metric", options=["nse", "kge", "rmse", "mae", "mse"], index=0)

    top_bottom = pd.concat([basin_frame.head(8), basin_frame.tail(8)], ignore_index=True).copy()
    top_bottom["label"] = top_bottom["basin_id"].astype(str)

    left, right = st.columns([1, 1.2])
    with left:
        render_section_header("Ranking", "Basin leaderboard for the current metric")
        st.dataframe(
            basin_frame.style.format(
                {"mse": "{:.3f}", "mae": "{:.3f}", "rmse": "{:.3f}", "nse": "{:.3f}", "kge": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        render_section_header("Top and bottom basins", f"Visual ranking by {context['sort_by'].upper()}")
        render_bar_chart(top_bottom, "label", context["sort_by"], color="#0F766E", y_title=context["sort_by"].upper())

    render_section_header("Spatial pattern", f"Gauge locations colored by {map_metric.upper()}")
    _render_basin_metric_map(basin_frame, map_metric)

    timeseries_frame = pd.DataFrame(
        timeseries_rows(context["selected_run_id"], selected_basin, context["api_url"], context["use_api"])
    )
    if timeseries_frame.empty:
        st.info("No time series records are available for this basin.")
        st.stop()

    ts_left, ts_right = st.columns([1.35, 1])
    with ts_left:
        render_section_header("Hydrograph", f"Basin {selected_basin} observed vs predicted flow")
        render_timeseries_chart(timeseries_frame, "date", height=360)
    with ts_right:
        render_section_header("Residual signal", "Prediction error over time")
        residual_frame = timeseries_frame.copy()
        residual_frame["observed"] = residual_frame["residual"]
        residual_frame["predicted"] = 0.0
        render_timeseries_chart(residual_frame[["date", "observed", "predicted"]], "date", height=360)

    render_section_header("Flow-duration behavior", "Observed and predicted exceedance curves")
    fdc_frame = _flow_duration_frame(timeseries_frame)
    st.line_chart(
        fdc_frame.pivot(index="exceedance", columns="series", values="flow"),
        color=["#0F766E", "#F59E0B"],
        use_container_width=True,
    )
    st.dataframe(timeseries_frame.tail(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
