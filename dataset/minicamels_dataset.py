"""MiniCAMELS access and dataset-level summary helpers.

The detailed loader will be connected after we inspect the installed
``minicamels`` API. This module already owns the assignment's data boundary so
the rest of the project does not need to know how MiniCAMELS is stored.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from util.config import DEFAULT_DYNAMIC_INPUTS, DEFAULT_TARGET_VARIABLE


def _minicamels_available() -> bool:
    try:
        import minicamels  # noqa: F401
    except ImportError:
        return False
    return True


def summarize_dataset(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return the assignment-level MiniCAMELS summary.

    This intentionally includes the known assignment metadata even when the
    local package is not installed yet, so the CLI remains usable while the
    environment is being set up.
    """
    return {
        "dataset": "MiniCAMELS",
        "data_dir": data_dir,
        "minicamels_installed": _minicamels_available(),
        "number_of_basins": 50,
        "time_span": "Water years 1981-2010",
        "dynamic_input_variables": list(DEFAULT_DYNAMIC_INPUTS),
        "target_variable": DEFAULT_TARGET_VARIABLE,
        "static_attributes": "To be selected during data exploration",
        "split_strategy": (
            "Planned: split by basin so validation/test basins are never used "
            "during training, avoiding spatial leakage."
        ),
        "supervised_sample": (
            "Sequence of daily meteorological forcings plus static attributes; "
            "predict qobs for the next day after the sequence."
        ),
    }
