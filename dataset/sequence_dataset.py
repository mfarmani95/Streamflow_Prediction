"""PyTorch dataset for windowed streamflow sequence samples."""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np


class StreamflowSequenceDataset:
    """Windowed streamflow samples backed by per-basin arrays.

    The dataset stores the full normalized time series once per basin and uses a
    sample index of ``(basin_id, start_day)`` pairs. That keeps memory much lower
    than materializing every sequence window.
    """

    def __init__(
        self,
        basin_data: Mapping[str, object],
        sample_index: List[Tuple[str, int]],
        static_attributes: Mapping[str, np.ndarray],
        seq_len: int,
        forecast_horizon: int = 1,
    ) -> None:
        self.basin_data = dict(basin_data)
        self.sample_index = list(sample_index)
        self.static_attributes = dict(static_attributes)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        import torch

        basin_id, start = self.sample_index[index]
        basin = self.basin_data[basin_id]
        target_index = start + self.seq_len + self.forecast_horizon - 1

        dynamic = basin.dynamic[start : start + self.seq_len]
        target = basin.target[target_index]
        static = self.static_attributes[basin_id]

        return {
            "dynamic": torch.as_tensor(dynamic, dtype=torch.float32),
            "static": torch.as_tensor(static, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "basin_id": basin_id,
            "date": str(basin.dates[target_index]),
        }
