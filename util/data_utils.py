"""Data preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (self.std + self.eps)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * (self.std + self.eps) + self.mean


def fit_standardizer(values: np.ndarray) -> Standardizer:
    return Standardizer(mean=np.nanmean(values, axis=0), std=np.nanstd(values, axis=0))


def make_window_indices(length: int, seq_len: int, forecast_horizon: int = 1) -> List[Tuple[int, int]]:
    """Return ``(start, target_index)`` pairs for sequence windows."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    indices = []
    max_start = length - seq_len - forecast_horizon + 1
    for start in range(max(0, max_start)):
        target_index = start + seq_len + forecast_horizon - 1
        indices.append((start, target_index))
    return indices


def split_sequence(items: Iterable[str], train_fraction: float, val_fraction: float) -> Tuple[list, list, list]:
    """Deterministically split an ordered sequence into train/val/test groups."""
    values = list(items)
    n_items = len(values)
    train_end = int(n_items * train_fraction)
    val_end = train_end + int(n_items * val_fraction)
    return values[:train_end], values[train_end:val_end], values[val_end:]
