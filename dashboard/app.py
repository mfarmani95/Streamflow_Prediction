"""Entrypoint router for the Streamflow Run Explorer dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(
    page_title="Streamflow Run Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview_page = st.Page(
    "overview.py",
    title="Overview",
    icon=":material/dashboard:",
    default=True,
)
basin_page = st.Page(
    "pages/1_Basin_Diagnostics.py",
    title="Basin Diagnostics",
    icon=":material/water:",
)
evaluation_page = st.Page(
    "pages/2_Evaluation.py",
    title="Evaluation",
    icon=":material/analytics:",
)
configuration_page = st.Page(
    "pages/3_Configuration.py",
    title="Configuration",
    icon=":material/tune:",
)

pg = st.navigation(
    {
        "Explore": [overview_page, basin_page, evaluation_page],
        "Run Metadata": [configuration_page],
    },
    position="sidebar",
)
pg.run()
