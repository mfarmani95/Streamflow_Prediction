"""Configuration defaults for the streamflow prediction project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


DEFAULT_DYNAMIC_INPUTS: Tuple[str, ...] = ("prcp", "tmax", "tmin", "srad", "vp")
DEFAULT_TARGET_VARIABLE = "qobs"
DEFAULT_STATIC_ATTRIBUTES: Tuple[str, ...] = (
    "lat",
    "lon",
    "elev_mean",
    "slope_mean",
    "area_km2",
    "mean_prcp",
    "mean_pet",
    "aridity",
    "frac_snow",
    "soil_depth_pelletier",
    "frac_forest",
    "lai_max",
)


@dataclass
class DataConfig:
    seq_len: int = 30
    forecast_horizon: int = 1
    window_stride: Optional[int] = None
    dynamic_inputs: Tuple[str, ...] = DEFAULT_DYNAMIC_INPUTS
    target_variable: str = DEFAULT_TARGET_VARIABLE
    static_attributes: Tuple[str, ...] = DEFAULT_STATIC_ATTRIBUTES
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_strategy: str = "random"
    split_stratify_attribute: str = "aridity"
    train_basin_count: Optional[int] = None
    val_basin_count: Optional[int] = None
    test_basin_count: Optional[int] = None


@dataclass
class ModelConfig:
    model_name: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    checkpoint_path: str = "outputs/best_model.pt"


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML to use --config: pip install PyYAML") from exc

    config_path = Path(path)
    with config_path.open("r") as fp:
        config = yaml.safe_load(fp) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    return config


def _copy_if_present(
    defaults: Dict[str, Any],
    source: Mapping[str, Any],
    source_key: str,
    target_key: Optional[str] = None,
) -> None:
    if source_key in source and source[source_key] is not None:
        defaults[target_key or source_key] = source[source_key]


def train_defaults_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Map nested YAML config keys onto CLI argument names for training."""
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}

    defaults: Dict[str, Any] = {}

    _copy_if_present(defaults, data, "seq_len")
    _copy_if_present(defaults, data, "forecast_horizon")
    _copy_if_present(defaults, data, "window_stride")
    _copy_if_present(defaults, data, "dynamic_inputs")
    _copy_if_present(defaults, data, "target_variable")
    _copy_if_present(defaults, data, "static_attributes")
    _copy_if_present(defaults, data, "split_strategy")
    _copy_if_present(defaults, data, "split_stratify_attribute")
    _copy_if_present(defaults, data, "train_basin_count")
    _copy_if_present(defaults, data, "val_basin_count")
    _copy_if_present(defaults, data, "test_basin_count")

    _copy_if_present(defaults, model, "name", "model")
    _copy_if_present(defaults, model, "hidden_size")
    _copy_if_present(defaults, model, "num_layers")
    _copy_if_present(defaults, model, "dropout")
    _copy_if_present(defaults, model, "nhead")
    _copy_if_present(defaults, model, "dim_feedforward")

    _copy_if_present(defaults, training, "epochs")
    _copy_if_present(defaults, training, "batch_size")
    _copy_if_present(defaults, training, "learning_rate", "lr")
    _copy_if_present(defaults, training, "lr")
    _copy_if_present(defaults, training, "loss")
    _copy_if_present(defaults, training, "weight_decay")
    _copy_if_present(defaults, training, "grad_clip")
    _copy_if_present(defaults, training, "patience")
    _copy_if_present(defaults, training, "min_delta")
    _copy_if_present(defaults, training, "seed")
    _copy_if_present(defaults, training, "device")
    _copy_if_present(defaults, training, "num_workers")
    _copy_if_present(defaults, training, "limit_basins")
    _copy_if_present(defaults, training, "data_dir")
    _copy_if_present(defaults, training, "output_dir")
    _copy_if_present(defaults, training, "checkpoint_path", "checkpoint")
    _copy_if_present(defaults, training, "checkpoint")

    return defaults
