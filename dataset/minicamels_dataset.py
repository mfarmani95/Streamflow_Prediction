"""MiniCAMELS access, preprocessing, and DataLoader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from util.config import (
    DEFAULT_DYNAMIC_INPUTS,
    DEFAULT_STATIC_ATTRIBUTES,
    DEFAULT_TARGET_VARIABLE,
)


@dataclass
class BasinTimeSeries:
    basin_id: str
    dynamic: np.ndarray
    target: np.ndarray
    dates: np.ndarray


def _minicamels_available() -> bool:
    try:
        import minicamels  # noqa: F401
    except ImportError:
        return False
    return True


def _require_minicamels():
    try:
        from minicamels import MiniCamels
    except ImportError as exc:
        raise ImportError(
            "MiniCAMELS is not installed. In Colab run:\n"
            "!pip install git+https://github.com/BennettHydroLab/minicamels.git"
        ) from exc
    return MiniCamels


def _normalize_basin_id(value: object) -> str:
    return str(value).split(".")[0].zfill(8)


def _as_float_list(values: np.ndarray) -> List[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _fit_mean_std(values: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis)
    std = np.nanstd(values, axis=axis)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    fill_nan: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if fill_nan:
        values = np.where(np.isfinite(values), values, mean)
    return ((values - mean) / (std + 1e-6)).astype(np.float32)


def _choose_static_attributes(
    attributes: pd.DataFrame,
    requested: Optional[Sequence[str]] = None,
) -> List[str]:
    if requested is not None:
        missing = [name for name in requested if name not in attributes.columns]
        if missing:
            raise ValueError(f"Static attribute columns not found: {missing}")
        return list(requested)

    available = [name for name in DEFAULT_STATIC_ATTRIBUTES if name in attributes.columns]
    if available:
        return available

    numeric = attributes.select_dtypes(include=[np.number]).columns.tolist()
    leakage_prone = {"q_mean", "runoff_ratio", "hfd_mean", "baseflow_index"}
    return [name for name in numeric if name not in leakage_prone]


def make_basin_splits(
    basin_ids: Sequence[str],
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split basin IDs into train/validation/test groups without leakage."""
    basin_ids = [_normalize_basin_id(value) for value in basin_ids]
    rng = np.random.default_rng(seed)
    shuffled = np.array(basin_ids, dtype=object)
    rng.shuffle(shuffled)

    n_basins = len(shuffled)
    n_train = max(1, int(n_basins * train_fraction))
    n_val = max(1, int(n_basins * val_fraction)) if n_basins >= 3 else 0
    if n_train + n_val >= n_basins and n_basins > 1:
        n_val = max(0, n_basins - n_train - 1)

    return {
        "train": shuffled[:n_train].tolist(),
        "val": shuffled[n_train : n_train + n_val].tolist(),
        "test": shuffled[n_train + n_val :].tolist(),
    }


def _build_sample_index(
    basin_data: Mapping[str, BasinTimeSeries],
    basin_ids: Iterable[str],
    seq_len: int,
    forecast_horizon: int,
) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    kernel = np.ones(seq_len, dtype=np.int32)

    for basin_id in basin_ids:
        basin = basin_data[basin_id]
        n_time = len(basin.target)
        max_start = n_time - seq_len - forecast_horizon + 1
        if max_start <= 0:
            continue

        dynamic_valid = np.isfinite(basin.dynamic).all(axis=1).astype(np.int32)
        window_valid = np.convolve(dynamic_valid, kernel, mode="valid") == seq_len
        target_valid = np.isfinite(basin.target)

        for start in range(max_start):
            target_index = start + seq_len + forecast_horizon - 1
            if window_valid[start] and target_valid[target_index]:
                samples.append((basin_id, start))

    return samples


def _load_basin_timeseries(
    client: Any,
    basin_id: str,
    dynamic_inputs: Sequence[str],
    target_variable: str,
) -> BasinTimeSeries:
    ds = client.load_basin(basin_id)
    dynamic = np.stack(
        [ds[name].values.astype(np.float32) for name in dynamic_inputs],
        axis=1,
    )
    target = ds[target_variable].values.astype(np.float32)
    dates = pd.to_datetime(ds["time"].values).strftime("%Y-%m-%d").to_numpy()
    return BasinTimeSeries(
        basin_id=basin_id,
        dynamic=dynamic,
        target=target,
        dates=dates,
    )


def summarize_dataset(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return MiniCAMELS metadata and the project split/design choices."""
    summary: Dict[str, Any] = {
        "dataset": "MiniCAMELS",
        "data_dir": data_dir,
        "minicamels_installed": _minicamels_available(),
        "number_of_basins": 50,
        "time_span": "1980-10-01 to 2010-09-30, water years 1981-2010",
        "dynamic_input_variables": list(DEFAULT_DYNAMIC_INPUTS),
        "target_variable": DEFAULT_TARGET_VARIABLE,
        "static_attributes_used_by_default": list(DEFAULT_STATIC_ATTRIBUTES),
        "split_strategy": (
            "Basin split: 70% train, 15% validation, 15% test. The validation "
            "and test basins are never used for fitting model weights or "
            "normalization statistics, which avoids spatial leakage."
        ),
        "supervised_sample": (
            "Input is a sequence of daily forcings plus static catchment "
            "attributes; target is qobs one day after the sequence by default."
        ),
    }

    if not summary["minicamels_installed"]:
        summary["installation"] = (
            "pip install git+https://github.com/BennettHydroLab/minicamels.git"
        )
        return summary

    MiniCamels = _require_minicamels()
    client = MiniCamels(local_data_dir=data_dir)
    basins = client.basins()
    attributes = client.attributes()
    summary.update(
        {
            "number_of_basins": int(len(basins)),
            "basin_columns": basins.columns.tolist(),
            "number_of_static_attributes_available": int(attributes.shape[1]),
            "static_attribute_columns_available": attributes.columns.tolist(),
        }
    )
    return summary


def build_datasets(
    data_dir: Optional[str] = None,
    seq_len: int = 30,
    forecast_horizon: int = 1,
    dynamic_inputs: Sequence[str] = DEFAULT_DYNAMIC_INPUTS,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    static_attributes: Optional[Sequence[str]] = None,
    split_ids: Optional[Mapping[str, Sequence[str]]] = None,
    scalers: Optional[Mapping[str, Sequence[float]]] = None,
    seed: int = 42,
    limit_basins: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create train/validation/test ``Dataset`` objects and metadata."""
    MiniCamels = _require_minicamels()
    client = MiniCamels(local_data_dir=data_dir)

    basins = client.basins().copy()
    basins["basin_id"] = basins["basin_id"].map(_normalize_basin_id)
    basin_ids = basins["basin_id"].tolist()
    if limit_basins is not None:
        basin_ids = basin_ids[:limit_basins]

    attributes_df = client.attributes().copy()
    attributes_df.index = attributes_df.index.map(_normalize_basin_id)
    static_cols = _choose_static_attributes(attributes_df, static_attributes)

    if split_ids is None:
        splits = make_basin_splits(basin_ids, seed=seed)
    else:
        splits = {
            name: [_normalize_basin_id(value) for value in values]
            for name, values in split_ids.items()
        }

    selected_ids = sorted({basin for values in splits.values() for basin in values})
    basin_data_raw = {
        basin_id: _load_basin_timeseries(
            client,
            basin_id,
            dynamic_inputs=dynamic_inputs,
            target_variable=target_variable,
        )
        for basin_id in selected_ids
    }

    train_ids = splits["train"]
    if scalers is None:
        train_dynamic = np.concatenate(
            [basin_data_raw[basin_id].dynamic for basin_id in train_ids],
            axis=0,
        )
        train_target = np.concatenate(
            [basin_data_raw[basin_id].target for basin_id in train_ids],
            axis=0,
        )
        train_static = attributes_df.loc[train_ids, static_cols].astype(float).to_numpy()

        dynamic_mean, dynamic_std = _fit_mean_std(train_dynamic, axis=0)
        target_mean, target_std = _fit_mean_std(train_target, axis=0)
        static_mean, static_std = _fit_mean_std(train_static, axis=0)
        scalers = {
            "dynamic_mean": _as_float_list(dynamic_mean),
            "dynamic_std": _as_float_list(dynamic_std),
            "target_mean": _as_float_list(np.asarray(target_mean)),
            "target_std": _as_float_list(np.asarray(target_std)),
            "static_mean": _as_float_list(static_mean),
            "static_std": _as_float_list(static_std),
        }

    dynamic_mean = np.asarray(scalers["dynamic_mean"], dtype=np.float32)
    dynamic_std = np.asarray(scalers["dynamic_std"], dtype=np.float32)
    target_mean = np.asarray(scalers["target_mean"], dtype=np.float32).reshape(())
    target_std = np.asarray(scalers["target_std"], dtype=np.float32).reshape(())
    static_mean = np.asarray(scalers["static_mean"], dtype=np.float32)
    static_std = np.asarray(scalers["static_std"], dtype=np.float32)

    basin_data = {}
    for basin_id, raw in basin_data_raw.items():
        basin_data[basin_id] = BasinTimeSeries(
            basin_id=basin_id,
            dynamic=_standardize(raw.dynamic, dynamic_mean, dynamic_std, fill_nan=True),
            target=_standardize(raw.target, target_mean, target_std, fill_nan=False),
            dates=raw.dates,
        )

    static_by_basin = {}
    for basin_id in selected_ids:
        static_values = (
            attributes_df.loc[basin_id, static_cols].astype(float).to_numpy(dtype=np.float32)
        )
        static_by_basin[basin_id] = _standardize(
            static_values,
            static_mean,
            static_std,
            fill_nan=True,
        )

    from dataset.sequence_dataset import StreamflowSequenceDataset

    sample_index = {
        name: _build_sample_index(basin_data, ids, seq_len, forecast_horizon)
        for name, ids in splits.items()
    }
    datasets = {
        name: StreamflowSequenceDataset(
            basin_data=basin_data,
            sample_index=samples,
            static_attributes=static_by_basin,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
        )
        for name, samples in sample_index.items()
    }

    metadata = {
        "data_dir": data_dir,
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "dynamic_inputs": list(dynamic_inputs),
        "target_variable": target_variable,
        "static_attributes": static_cols,
        "splits": splits,
        "scalers": dict(scalers),
        "sample_counts": {name: len(samples) for name, samples in sample_index.items()},
        "basin_counts": {name: len(ids) for name, ids in splits.items()},
    }
    return datasets, metadata


def build_dataloaders(
    data_dir: Optional[str] = None,
    seq_len: int = 30,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    dynamic_inputs: Sequence[str] = DEFAULT_DYNAMIC_INPUTS,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    static_attributes: Optional[Sequence[str]] = None,
    split_ids: Optional[Mapping[str, Sequence[str]]] = None,
    scalers: Optional[Mapping[str, Sequence[float]]] = None,
    seed: int = 42,
    limit_basins: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create train/validation/test PyTorch DataLoaders."""
    from torch.utils.data import DataLoader

    datasets, metadata = build_datasets(
        data_dir=data_dir,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        dynamic_inputs=dynamic_inputs,
        target_variable=target_variable,
        static_attributes=static_attributes,
        split_ids=split_ids,
        scalers=scalers,
        seed=seed,
        limit_basins=limit_basins,
    )
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for name, dataset in datasets.items()
    }
    return loaders, metadata
